//! Optimizers operating on `ParameterStore` and materialized gradients.
//!
//! Optimizer state is deliberately not part of the graph. This mirrors the
//! design split between graph execution, gradient production, and parameter
//! update, and makes checkpointing optimizer state straightforward.

use std::collections::HashMap;

use oxide_ml_core::{DenseTensor, Error, ParameterId, ParameterStore, Result, Scalar};

pub type Gradients<T = f32> = HashMap<ParameterId, DenseTensor<T>>;

pub trait Optimizer<T: Scalar = f32> {
    /// Update trainable parameters in place.
    fn step(&mut self, parameters: &mut ParameterStore<T>, gradients: &Gradients<T>) -> Result<()>;

    /// Drop accumulated gradients. The graph executor returns fresh gradients
    /// per run, so the reference implementation has nothing to clear.
    fn zero_grad(&mut self) {}

    fn learning_rate(&self) -> f32;
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
