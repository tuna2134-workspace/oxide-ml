//! Checkpoint format for model parameters and optimizer state.
//!
//! Checkpoints are versioned independently of the graph IR. Unknown versions
//! and unknown optimizers are rejected instead of being silently ignored.
//! Parameters are keyed by their stable module path name.

use std::collections::BTreeMap;
use std::path::Path;

use oxide_ml_core::{DType, DenseTensor, Error, ParameterStore, Result};
use oxide_ml_optim::{Optimizer, OptimizerState};
use serde::{Deserialize, Serialize};

use crate::module::{Module, ModulePath};

pub const CHECKPOINT_FORMAT: &str = "oxide-ml.checkpoint";
pub const CHECKPOINT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckpointParameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub role: String,
    pub data: Vec<f32>,
}

/// Serialized model + optimizer state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub format: String,
    pub version: u32,
    pub model_type: String,
    pub parameters: Vec<CheckpointParameter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<OptimizerState>,
    #[serde(default)]
    pub step: u64,
    #[serde(default)]
    pub epoch: u64,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub loaded: usize,
    pub missing_in_model: Vec<String>,
    pub missing_in_checkpoint: Vec<String>,
    pub model_type_matches: bool,
}

impl LoadReport {
    pub fn is_exact(&self) -> bool {
        self.missing_in_model.is_empty() && self.missing_in_checkpoint.is_empty()
    }
}

impl Checkpoint {
    /// Capture every parameter tensor owned by `model`.
    pub fn from_model<M: Module + ?Sized>(model: &M) -> Result<Self> {
        let mut parameters = Vec::new();
        model.visit_tensors(&ModulePath::root(), &mut |path, tensor, role| {
            parameters.push(CheckpointParameter {
                name: path.as_string(),
                shape: tensor.dims(),
                dtype: DType::F32.name().to_string(),
                role: role.as_str().to_string(),
                data: tensor.data().to_vec(),
            });
            Ok(())
        })?;
        parameters.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(Checkpoint {
            format: CHECKPOINT_FORMAT.to_string(),
            version: CHECKPOINT_VERSION,
            model_type: model.module_type().to_string(),
            parameters,
            optimizer: None,
            step: 0,
            epoch: 0,
            metadata: BTreeMap::new(),
        })
    }

    /// Attach optimizer state, keyed by parameter name.
    pub fn with_optimizer<O: Optimizer<f32> + ?Sized>(
        mut self,
        optimizer: &O,
        parameters: &ParameterStore<f32>,
    ) -> Result<Self> {
        self.optimizer = optimizer.state_dict(parameters)?;
        Ok(self)
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|error| Error::Serialization(error.to_string()))
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let checkpoint: Checkpoint =
            serde_json::from_str(json).map_err(|error| Error::Serialization(error.to_string()))?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != CHECKPOINT_FORMAT {
            return Err(Error::Serialization(format!(
                "unknown checkpoint format {:?}",
                self.format
            )));
        }
        if self.version != CHECKPOINT_VERSION {
            return Err(Error::Serialization(format!(
                "unsupported checkpoint version {} (this build supports {})",
                self.version, CHECKPOINT_VERSION
            )));
        }
        let mut names = std::collections::HashSet::new();
        for parameter in &self.parameters {
            if !names.insert(parameter.name.as_str()) {
                return Err(Error::Serialization(format!(
                    "checkpoint contains duplicate parameter {:?}",
                    parameter.name
                )));
            }
            let expected: usize = parameter.shape.iter().product();
            if expected != parameter.data.len() {
                return Err(Error::Serialization(format!(
                    "checkpoint parameter {:?} declares shape {:?} but has {} values",
                    parameter.name,
                    parameter.shape,
                    parameter.data.len()
                )));
            }
        }
        Ok(())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        std::fs::write(path, self.to_json()?).map_err(|error| Error::Io(error.to_string()))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(|error| Error::Io(error.to_string()))?;
        Self::from_json(&json)
    }

    /// Copy checkpoint parameters into `model`.
    ///
    /// All names and shapes are validated before any tensor is mutated, so a
    /// partially-applied checkpoint is impossible.
    pub fn apply<M: Module + ?Sized>(&self, model: &mut M) -> Result<LoadReport> {
        self.validate()?;
        let mut model_parameters: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        model.visit_tensors(&ModulePath::root(), &mut |path, tensor, _role| {
            model_parameters.insert(path.as_string(), tensor.dims());
            Ok(())
        })?;

        let checkpoint_names: std::collections::HashSet<&str> =
            self.parameters.iter().map(|p| p.name.as_str()).collect();
        let model_names: std::collections::HashSet<&str> =
            model_parameters.keys().map(String::as_str).collect();

        let mut report = LoadReport {
            model_type_matches: self.model_type == model.module_type(),
            missing_in_model: checkpoint_names
                .difference(&model_names)
                .map(|name| name.to_string())
                .collect(),
            missing_in_checkpoint: model_names
                .difference(&checkpoint_names)
                .map(|name| name.to_string())
                .collect(),
            ..LoadReport::default()
        };
        report.missing_in_model.sort();
        report.missing_in_checkpoint.sort();

        for parameter in &self.parameters {
            if let Some(shape) = model_parameters.get(&parameter.name)
                && *shape != parameter.shape
            {
                return Err(Error::Serialization(format!(
                    "checkpoint parameter {:?} has shape {:?} but the model has {:?}",
                    parameter.name, parameter.shape, shape
                )));
            }
        }

        let by_name: BTreeMap<&str, &CheckpointParameter> = self
            .parameters
            .iter()
            .map(|parameter| (parameter.name.as_str(), parameter))
            .collect();
        model.visit_tensors_mut(&ModulePath::root(), &mut |path, tensor, _role| {
            let name = path.as_string();
            if let Some(parameter) = by_name.get(name.as_str()) {
                if tensor.dims() != parameter.shape {
                    return Err(Error::Serialization(format!(
                        "checkpoint parameter {name:?} shape mismatch"
                    )));
                }
                let restored = DenseTensor::from_vec(&parameter.shape, parameter.data.clone())?;
                tensor.copy_from(&restored);
                report.loaded += 1;
            }
            Ok(())
        })?;
        Ok(report)
    }
}

/// Save `model` to `path`.
pub fn save_model<M: Module + ?Sized>(path: impl AsRef<Path>, model: &M) -> Result<()> {
    Checkpoint::from_model(model)?.save(path)
}

/// Load `path` into `model`.
pub fn load_model<M: Module + ?Sized>(path: impl AsRef<Path>, model: &mut M) -> Result<LoadReport> {
    Checkpoint::load(path)?.apply(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_version() {
        let mut checkpoint = Checkpoint {
            format: CHECKPOINT_FORMAT.to_string(),
            version: CHECKPOINT_VERSION + 1,
            model_type: "Test".into(),
            parameters: Vec::new(),
            optimizer: None,
            step: 0,
            epoch: 0,
            metadata: BTreeMap::new(),
        };
        assert!(checkpoint.validate().is_err());
        checkpoint.version = CHECKPOINT_VERSION;
        assert!(checkpoint.validate().is_ok());
    }

    #[test]
    fn round_trips_json() {
        let checkpoint = Checkpoint {
            format: CHECKPOINT_FORMAT.to_string(),
            version: CHECKPOINT_VERSION,
            model_type: "Test".into(),
            parameters: vec![CheckpointParameter {
                name: "fc.weight".into(),
                shape: vec![2, 2],
                dtype: "f32".into(),
                role: "trainable".into(),
                data: vec![1.0, 2.0, 3.0, 4.0],
            }],
            optimizer: None,
            step: 3,
            epoch: 1,
            metadata: BTreeMap::new(),
        };
        let json = checkpoint.to_json().unwrap();
        assert_eq!(Checkpoint::from_json(&json).unwrap(), checkpoint);
    }
}
