//! Post-training calibration for BatchNorm running statistics.
//!
//! The reference executor is functional (no in-graph state mutation), so
//! BatchNorm normalizes with batch statistics during training but never writes
//! its running buffers. This module runs the model in `ModuleMode::Calibrate`:
//! BatchNorm records its input as a graph probe while still normalizing with
//! batch statistics. The probes are accumulated over calibration batches and
//! folded into the running mean/variance buffers with momentum.

use std::collections::HashMap;

use oxide_ml_core::{DenseTensor, Error, Result};

use crate::module::{Module, ModuleMode, ModulePath};
use crate::runtime::{CompiledModel, InputSpec, ModelCompileOptions};
use crate::tensor::Tensor;

/// Summary of a BatchNorm calibration pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchNormCalibrationReport {
    /// BatchNorm layers whose buffers were updated.
    pub layers: usize,
    /// Samples observed across all calibration batches.
    pub samples: usize,
    /// Module paths that received updated buffers.
    pub updated: Vec<String>,
    /// Probes without a matching BatchNorm buffer pair.
    pub skipped: Vec<String>,
}

/// Calibrate every BatchNorm running buffer in `model`.
///
/// `input_specs` describe one sample; calibration batches must include the
/// batch dimension. `momentum` is the usual BatchNorm momentum (1.0 replaces
/// the old statistics completely).
pub fn calibrate_batch_norm<M: Module + ?Sized>(
    model: &mut M,
    input_specs: &[InputSpec],
    batches: &[Vec<Tensor<f32>>],
    momentum: f32,
) -> Result<BatchNormCalibrationReport> {
    if !(0.0..=1.0).contains(&momentum) {
        return Err(Error::invalid_shape(
            "BatchNorm calibration",
            format!("momentum {momentum} must be in [0, 1]"),
        ));
    }
    if batches.is_empty() {
        return Err(Error::invalid_shape(
            "BatchNorm calibration",
            "at least one calibration batch is required",
        ));
    }
    let batched: Vec<InputSpec> = input_specs
        .iter()
        .map(|spec| InputSpec::batched(spec.name.clone(), spec.dims.clone()))
        .collect();
    let options = ModelCompileOptions {
        mode: ModuleMode::Calibrate,
        ..ModelCompileOptions::default()
    };
    let compiled = CompiledModel::compile(model, &batched, options)?;
    if compiled.probe_names().is_empty() {
        return Ok(BatchNormCalibrationReport::default());
    }
    let store = compiled.parameter_store(model)?;
    let mut sums: HashMap<String, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
    let mut samples = 0usize;
    for batch in batches {
        if batch.len() != batched.len() {
            return Err(Error::invalid_shape(
                "BatchNorm calibration",
                format!(
                    "batch has {} inputs but {} specs were given",
                    batch.len(),
                    batched.len()
                ),
            ));
        }
        let inputs: Vec<(&str, &Tensor<f32>)> = batched
            .iter()
            .zip(batch)
            .map(|(spec, tensor)| (spec.name.as_str(), tensor))
            .collect();
        let probes = compiled.run_probes(&store, &inputs)?;
        samples += batch
            .first()
            .map(|tensor| tensor.dims().first().copied().unwrap_or(0))
            .unwrap_or(0);
        for (name, tensor) in probes {
            let dims = tensor.dims();
            if dims.len() < 2 {
                continue;
            }
            let channels = dims[1];
            let per_channel = dims.iter().skip(2).product::<usize>().max(1);
            let count = dims[0] * per_channel;
            let entry = sums
                .entry(name)
                .or_insert_with(|| (vec![0.0; channels], vec![0.0; channels], 0));
            for row in 0..dims[0] {
                for channel in 0..channels {
                    for index in 0..per_channel {
                        let offset = (row * channels + channel) * per_channel + index;
                        let value = tensor.data()[offset] as f64;
                        entry.0[channel] += value;
                        entry.1[channel] += value * value;
                    }
                }
            }
            entry.2 += count;
        }
    }

    let mut stats: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
    for (name, (sum, sumsq, count)) in &sums {
        if *count == 0 {
            continue;
        }
        let channels = sum.len();
        let mut mean = vec![0.0f32; channels];
        let mut variance = vec![0.0f32; channels];
        for channel in 0..channels {
            let channel_count = (*count).max(1);
            let mean_value = sum[channel] / channel_count as f64;
            let second = sumsq[channel] / channel_count as f64;
            let var = (second - mean_value * mean_value).max(0.0);
            let unbiased = if channel_count > 1 {
                var * channel_count as f64 / (channel_count - 1) as f64
            } else {
                var
            };
            mean[channel] = mean_value as f32;
            variance[channel] = unbiased as f32;
        }
        stats.insert(name.clone(), (mean, variance));
    }

    let mut updated = Vec::new();
    let mut updated_tensors: Vec<(String, DenseTensor<f32>)> = Vec::new();
    let mut skipped = Vec::new();
    model.visit_tensors_mut(&ModulePath::root(), &mut |path, tensor, _role| {
        let name = path.as_string();
        let Some((module, field)) = name.rsplit_once('.') else {
            return Ok(());
        };
        if field != "running_mean" && field != "running_var" {
            return Ok(());
        }
        let Some((mean, variance)) = stats.get(module) else {
            skipped.push(name);
            return Ok(());
        };
        let values = if field == "running_mean" {
            mean
        } else {
            variance
        };
        if tensor.data().len() != values.len() {
            return Err(Error::shape_mismatch(
                format!("BatchNorm buffer {name:?}"),
                tensor.shape().clone(),
                oxide_ml_core::Shape::from_static([values.len()]),
            ));
        }
        let updated_values: Vec<f32> = tensor
            .data()
            .iter()
            .zip(values)
            .map(|(old, new)| (1.0 - momentum) * old + momentum * new)
            .collect();
        *tensor = DenseTensor::from_vec(tensor.dims(), updated_values)?;
        updated_tensors.push((module.to_string(), tensor.clone()));
        Ok(())
    })?;
    for (module, _) in &updated_tensors {
        if !updated.contains(module) {
            updated.push(module.clone());
        }
    }
    Ok(BatchNormCalibrationReport {
        layers: updated.len(),
        samples,
        updated,
        skipped,
    })
}
