//! Optimizers operating on `ParameterStore` and materialized gradients.
//!
//! Optimizer state is deliberately not part of the graph. This mirrors the
//! design split between graph execution, gradient production, and parameter
//! update, and makes checkpointing optimizer state straightforward.

use std::collections::{BTreeMap, HashMap};

use oxide_ml_core::{DenseTensor, Error, ParameterId, ParameterStore, Result, Scalar};
use serde::{Deserialize, Serialize};

pub type Gradients<T = f32> = HashMap<ParameterId, DenseTensor<T>>;

/// Portable optimizer state.
///
/// Entries are keyed by stable parameter name (not `ParameterId`, which depends
/// on graph construction order) and each slot stores `f64` values so a
/// checkpoint can be reloaded into any scalar instantiation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OptimizerState {
    pub kind: String,
    pub step: u64,
    pub entries: BTreeMap<String, BTreeMap<String, Vec<f64>>>,
}

impl OptimizerState {
    pub fn new(kind: impl Into<String>) -> Self {
        OptimizerState {
            kind: kind.into(),
            step: 0,
            entries: BTreeMap::new(),
        }
    }

    pub fn set_slot(&mut self, parameter: &str, slot: &str, values: Vec<f64>) {
        self.entries
            .entry(parameter.to_string())
            .or_default()
            .insert(slot.to_string(), values);
    }

    pub fn slot(&self, parameter: &str, slot: &str) -> Option<&[f64]> {
        self.entries
            .get(parameter)
            .and_then(|slots| slots.get(slot))
            .map(Vec::as_slice)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

pub trait Optimizer<T: Scalar = f32> {
    /// Update trainable parameters in place.
    fn step(&mut self, parameters: &mut ParameterStore<T>, gradients: &Gradients<T>) -> Result<()>;

    /// Drop accumulated gradients. The graph executor returns fresh gradients
    /// per run, so the reference implementation has nothing to clear.
    fn zero_grad(&mut self) {}

    fn learning_rate(&self) -> f32;

    /// Replace the learning rate (used by schedulers).
    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        let _ = learning_rate;
        Err(Error::unsupported(
            "this optimizer does not support learning-rate updates",
        ))
    }

    /// Export optimizer state keyed by parameter name.
    fn state_dict(&self, parameters: &ParameterStore<T>) -> Result<Option<OptimizerState>> {
        let _ = parameters;
        Ok(None)
    }

    /// Restore state previously produced by [`Optimizer::state_dict`].
    fn load_state_dict(
        &mut self,
        parameters: &ParameterStore<T>,
        state: &OptimizerState,
    ) -> Result<()> {
        let _ = (parameters, state);
        Err(Error::unsupported(
            "this optimizer does not support state restore",
        ))
    }
}

fn parameter_id_by_name<T: Scalar>(
    parameters: &ParameterStore<T>,
    name: &str,
) -> Result<ParameterId> {
    parameters
        .get_by_name(name)
        .map(|parameter| parameter.id)
        .ok_or_else(|| {
            Error::invalid_graph(format!(
                "optimizer state references unknown parameter {name:?}"
            ))
        })
}

fn restore_values<T: Scalar>(
    values: &[f64],
    expected: usize,
    slot: &str,
    parameter: &str,
) -> Result<Vec<T>> {
    if values.len() != expected {
        return Err(Error::invalid_shape(
            format!("optimizer state {slot} for {parameter:?}"),
            format!("expected {expected} values, got {}", values.len()),
        ));
    }
    Ok(values.iter().map(|value| T::from_f64(*value)).collect())
}

fn check_gradient<T: Scalar>(
    parameter: &oxide_ml_core::Parameter<T>,
    gradient: &DenseTensor<T>,
) -> Result<()> {
    if parameter.data.dims() != gradient.dims() {
        return Err(Error::shape_mismatch(
            format!("gradient for {:?}", parameter.name),
            parameter.data.shape().clone(),
            gradient.shape().clone(),
        ));
    }
    Ok(())
}

/// Plain stochastic gradient descent.
#[derive(Clone, Debug)]
pub struct Sgd<T: Scalar = f32> {
    pub learning_rate: f32,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> Sgd<T> {
    pub fn new(learning_rate: f32) -> Result<Self> {
        validate_lr(learning_rate)?;
        Ok(Sgd {
            learning_rate,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: Scalar> Optimizer<T> for Sgd<T> {
    fn step(&mut self, parameters: &mut ParameterStore<T>, gradients: &Gradients<T>) -> Result<()> {
        for parameter in parameters.iter_mut() {
            if !parameter.trainable {
                continue;
            }
            let Some(gradient) = gradients.get(&parameter.id) else {
                continue;
            };
            check_gradient(parameter, gradient)?;
            for (value, &grad) in parameter.data.data_mut().iter_mut().zip(gradient.data()) {
                *value -= T::from_f64(self.learning_rate as f64) * grad;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        validate_lr(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn state_dict(&self, _parameters: &ParameterStore<T>) -> Result<Option<OptimizerState>> {
        Ok(Some(OptimizerState::new("sgd")))
    }

    fn load_state_dict(
        &mut self,
        _parameters: &ParameterStore<T>,
        state: &OptimizerState,
    ) -> Result<()> {
        if state.kind != "sgd" {
            return Err(Error::invalid_graph(format!(
                "optimizer state kind {:?} cannot be loaded into SGD",
                state.kind
            )));
        }
        Ok(())
    }
}

/// SGD with classical momentum.
#[derive(Clone, Debug)]
pub struct SgdMomentum<T: Scalar = f32> {
    pub learning_rate: f32,
    pub momentum: f32,
    pub dampening: f32,
    pub nesterov: bool,
    velocity: HashMap<ParameterId, Vec<T>>,
}

impl<T: Scalar> SgdMomentum<T> {
    pub fn new(learning_rate: f32, momentum: f32) -> Result<Self> {
        validate_lr(learning_rate)?;
        if !(0.0..1.0).contains(&momentum) {
            return Err(Error::invalid_shape(
                "SgdMomentum",
                "momentum must be in [0, 1)",
            ));
        }
        Ok(SgdMomentum {
            learning_rate,
            momentum,
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        })
    }
}

impl<T: Scalar> Optimizer<T> for SgdMomentum<T> {
    fn step(&mut self, parameters: &mut ParameterStore<T>, gradients: &Gradients<T>) -> Result<()> {
        for parameter in parameters.iter_mut() {
            if !parameter.trainable {
                continue;
            }
            let Some(gradient) = gradients.get(&parameter.id) else {
                continue;
            };
            check_gradient(parameter, gradient)?;
            let velocity = self
                .velocity
                .entry(parameter.id)
                .or_insert_with(|| vec![T::zero(); parameter.data.numel()]);
            for ((value, &grad), velocity) in parameter
                .data
                .data_mut()
                .iter_mut()
                .zip(gradient.data())
                .zip(velocity.iter_mut())
            {
                *velocity = T::from_f64(self.momentum as f64) * *velocity
                    + (T::one() - T::from_f64(self.dampening as f64)) * grad;
                let direction = if self.nesterov {
                    grad + T::from_f64(self.momentum as f64) * *velocity
                } else {
                    *velocity
                };
                *value -= T::from_f64(self.learning_rate as f64) * direction;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        validate_lr(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn state_dict(&self, parameters: &ParameterStore<T>) -> Result<Option<OptimizerState>> {
        let mut state = OptimizerState::new("sgd_momentum");
        for parameter in parameters.iter() {
            if let Some(velocity) = self.velocity.get(&parameter.id) {
                state.set_slot(
                    &parameter.name,
                    "velocity",
                    velocity.iter().map(|value| value.to_f64()).collect(),
                );
            }
        }
        Ok(Some(state))
    }

    fn load_state_dict(
        &mut self,
        parameters: &ParameterStore<T>,
        state: &OptimizerState,
    ) -> Result<()> {
        if state.kind != "sgd_momentum" {
            return Err(Error::invalid_graph(format!(
                "optimizer state kind {:?} cannot be loaded into SGD with momentum",
                state.kind
            )));
        }
        self.velocity.clear();
        for (name, slots) in &state.entries {
            let id = parameter_id_by_name(parameters, name)?;
            let expected = parameters.get(id)?.data.numel();
            if let Some(values) = slots.get("velocity") {
                self.velocity
                    .insert(id, restore_values(values, expected, "velocity", name)?);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct AdamState<T: Scalar = f32> {
    m: Vec<T>,
    v: Vec<T>,
}

/// AdamW: decoupled weight decay, bias-corrected first and second moments.
#[derive(Clone, Debug)]
pub struct AdamW<T: Scalar = f32> {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    t: u64,
    state: HashMap<ParameterId, AdamState<T>>,
}

impl<T: Scalar> AdamW<T> {
    pub fn new(learning_rate: f32) -> Result<Self> {
        validate_lr(learning_rate)?;
        Ok(AdamW {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            t: 0,
            state: HashMap::new(),
        })
    }

    pub fn with_config(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Result<Self> {
        validate_lr(learning_rate)?;
        if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
            return Err(Error::invalid_shape(
                "AdamW",
                "beta values must be in [0, 1)",
            ));
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(Error::invalid_shape(
                "AdamW",
                "eps must be positive and finite",
            ));
        }
        Ok(AdamW {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            state: HashMap::new(),
        })
    }
}

impl<T: Scalar> Optimizer<T> for AdamW<T> {
    fn step(&mut self, parameters: &mut ParameterStore<T>, gradients: &Gradients<T>) -> Result<()> {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        for parameter in parameters.iter_mut() {
            if !parameter.trainable {
                continue;
            }
            let Some(gradient) = gradients.get(&parameter.id) else {
                continue;
            };
            check_gradient(parameter, gradient)?;
            let state = self.state.entry(parameter.id).or_insert_with(|| AdamState {
                m: vec![T::zero(); parameter.data.numel()],
                v: vec![T::zero(); parameter.data.numel()],
            });
            for (((value, &grad), m), v) in parameter
                .data
                .data_mut()
                .iter_mut()
                .zip(gradient.data())
                .zip(state.m.iter_mut())
                .zip(state.v.iter_mut())
            {
                *m = T::from_f64(self.beta1 as f64) * *m
                    + (T::one() - T::from_f64(self.beta1 as f64)) * grad;
                *v = T::from_f64(self.beta2 as f64) * *v
                    + (T::one() - T::from_f64(self.beta2 as f64)) * grad * grad;
                let m_hat = *m / T::from_f64(bias_correction1 as f64);
                let v_hat = *v / T::from_f64(bias_correction2 as f64);
                let update = m_hat / (v_hat.sqrt() + T::from_f64(self.eps as f64));
                *value -= T::from_f64(self.learning_rate as f64)
                    * (update + T::from_f64(self.weight_decay as f64) * *value);
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        validate_lr(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn state_dict(&self, parameters: &ParameterStore<T>) -> Result<Option<OptimizerState>> {
        let mut state = OptimizerState::new("adamw");
        state.step = self.t;
        for parameter in parameters.iter() {
            if let Some(adam) = self.state.get(&parameter.id) {
                state.set_slot(
                    &parameter.name,
                    "m",
                    adam.m.iter().map(|value| value.to_f64()).collect(),
                );
                state.set_slot(
                    &parameter.name,
                    "v",
                    adam.v.iter().map(|value| value.to_f64()).collect(),
                );
            }
        }
        Ok(Some(state))
    }

    fn load_state_dict(
        &mut self,
        parameters: &ParameterStore<T>,
        state: &OptimizerState,
    ) -> Result<()> {
        if state.kind != "adamw" {
            return Err(Error::invalid_graph(format!(
                "optimizer state kind {:?} cannot be loaded into AdamW",
                state.kind
            )));
        }
        self.state.clear();
        self.t = state.step;
        for (name, slots) in &state.entries {
            let id = parameter_id_by_name(parameters, name)?;
            let expected = parameters.get(id)?.data.numel();
            let m = match slots.get("m") {
                Some(values) => restore_values(values, expected, "m", name)?,
                None => vec![T::zero(); expected],
            };
            let v = match slots.get("v") {
                Some(values) => restore_values(values, expected, "v", name)?,
                None => vec![T::zero(); expected],
            };
            self.state.insert(id, AdamState { m, v });
        }
        Ok(())
    }
}

/// RMSProp with optional momentum.
#[derive(Clone, Debug)]
pub struct RmsProp<T: Scalar = f32> {
    pub learning_rate: f32,
    pub alpha: f32,
    pub eps: f32,
    pub momentum: f32,
    square_avg: HashMap<ParameterId, Vec<T>>,
    momentum_buffer: HashMap<ParameterId, Vec<T>>,
}

impl<T: Scalar> RmsProp<T> {
    pub fn new(learning_rate: f32) -> Result<Self> {
        validate_lr(learning_rate)?;
        Ok(RmsProp {
            learning_rate,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
        })
    }
}

impl<T: Scalar> Optimizer<T> for RmsProp<T> {
    fn step(&mut self, parameters: &mut ParameterStore<T>, gradients: &Gradients<T>) -> Result<()> {
        for parameter in parameters.iter_mut() {
            if !parameter.trainable {
                continue;
            }
            let Some(gradient) = gradients.get(&parameter.id) else {
                continue;
            };
            check_gradient(parameter, gradient)?;
            let square_avg = self
                .square_avg
                .entry(parameter.id)
                .or_insert_with(|| vec![T::zero(); parameter.data.numel()]);
            let buffer = self
                .momentum_buffer
                .entry(parameter.id)
                .or_insert_with(|| vec![T::zero(); parameter.data.numel()]);
            for (((value, &grad), avg), buffer) in parameter
                .data
                .data_mut()
                .iter_mut()
                .zip(gradient.data())
                .zip(square_avg.iter_mut())
                .zip(buffer.iter_mut())
            {
                *avg = T::from_f64(self.alpha as f64) * *avg
                    + (T::one() - T::from_f64(self.alpha as f64)) * grad * grad;
                let denominator = (*avg + T::from_f64(self.eps as f64)).sqrt();
                if self.momentum > 0.0 {
                    *buffer = T::from_f64(self.momentum as f64) * *buffer + grad / denominator;
                    *value -= T::from_f64(self.learning_rate as f64) * *buffer;
                } else {
                    *value -= T::from_f64(self.learning_rate as f64) * grad / denominator;
                }
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        validate_lr(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn state_dict(&self, parameters: &ParameterStore<T>) -> Result<Option<OptimizerState>> {
        let mut state = OptimizerState::new("rmsprop");
        for parameter in parameters.iter() {
            if let Some(square_avg) = self.square_avg.get(&parameter.id) {
                state.set_slot(
                    &parameter.name,
                    "square_avg",
                    square_avg.iter().map(|value| value.to_f64()).collect(),
                );
            }
            if let Some(buffer) = self.momentum_buffer.get(&parameter.id) {
                state.set_slot(
                    &parameter.name,
                    "momentum_buffer",
                    buffer.iter().map(|value| value.to_f64()).collect(),
                );
            }
        }
        Ok(Some(state))
    }

    fn load_state_dict(
        &mut self,
        parameters: &ParameterStore<T>,
        state: &OptimizerState,
    ) -> Result<()> {
        if state.kind != "rmsprop" {
            return Err(Error::invalid_graph(format!(
                "optimizer state kind {:?} cannot be loaded into RMSProp",
                state.kind
            )));
        }
        self.square_avg.clear();
        self.momentum_buffer.clear();
        for (name, slots) in &state.entries {
            let id = parameter_id_by_name(parameters, name)?;
            let expected = parameters.get(id)?.data.numel();
            if let Some(values) = slots.get("square_avg") {
                self.square_avg
                    .insert(id, restore_values(values, expected, "square_avg", name)?);
            }
            if let Some(values) = slots.get("momentum_buffer") {
                self.momentum_buffer.insert(
                    id,
                    restore_values(values, expected, "momentum_buffer", name)?,
                );
            }
        }
        Ok(())
    }
}

fn validate_lr(learning_rate: f32) -> Result<()> {
    if !(learning_rate.is_finite() && learning_rate > 0.0) {
        return Err(Error::invalid_shape(
            "optimizer",
            format!("learning rate must be positive and finite, got {learning_rate}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use oxide_ml_core::{DenseTensor, Error, ParameterId, ParameterStore};

    use super::*;

    fn quadratic_store() -> ParameterStore {
        let mut store = ParameterStore::new();
        store
            .add(
                ParameterId(0),
                "x",
                DenseTensor::from_vec([1], vec![4.0]).unwrap(),
                true,
            )
            .unwrap();
        store
    }

    #[test]
    fn sgd_moves_toward_zero() {
        let mut store = quadratic_store();
        let mut gradients = Gradients::new();
        gradients.insert(
            ParameterId(0),
            DenseTensor::from_vec([1], vec![2.0]).unwrap(),
        );
        let mut optimizer = Sgd::new(0.25).unwrap();
        optimizer.step(&mut store, &gradients).unwrap();
        assert!((store.get(ParameterId(0)).unwrap().data.data()[0] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn adamw_rejects_bad_lr() {
        assert!(matches!(
            AdamW::<f32>::new(0.0),
            Err(Error::InvalidShape { .. })
        ));
    }
}
