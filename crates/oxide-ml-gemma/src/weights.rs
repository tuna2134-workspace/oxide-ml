//! Mapping HuggingFace Gemma 4 safetensors to oxide-ml tensors.

use oxide_ml_core::{DenseTensor, Error, Result};
use oxide_ml_safetensors::SafeTensorsFile;

use crate::config::Gemma4Config;

pub struct Gemma4Weights<'a> {
    file: &'a SafeTensorsFile,
    prefix: String,
}

impl<'a> Gemma4Weights<'a> {
    pub fn new(file: &'a SafeTensorsFile) -> Result<Self> {
        let candidates = [
            "model.language_model.",
            "language_model.model.",
            "model.",
            "",
        ];
        for prefix in candidates {
            if file.contains(&format!("{prefix}embed_tokens.weight")) {
                return Ok(Gemma4Weights {
                    file,
                    prefix: prefix.to_string(),
                });
            }
        }
        Err(Error::invalid_graph(
            "safetensors does not contain an embed_tokens.weight; \
             is this a Gemma 4 checkpoint?",
        ))
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    pub fn contains(&self, name: &str) -> bool {
        self.file.contains(&format!("{}{name}", self.prefix))
    }

    pub fn tensor(&self, name: &str) -> Result<DenseTensor<f32>> {
        let full = format!("{}{name}", self.prefix);
        let loaded = self.file.tensor(&full).map_err(|error| {
            Error::invalid_graph(format!("missing Gemma 4 weight {full:?}: {error}"))
        })?;
        if !matches!(
            loaded.dtype,
            oxide_ml_safetensors::StoredDType::F32
                | oxide_ml_safetensors::StoredDType::F16
                | oxide_ml_safetensors::StoredDType::BF16
        ) {
            return Err(Error::unsupported(format!(
                "Gemma 4 weight {full:?} has non-floating dtype {:?}",
                loaded.dtype
            )));
        }
        DenseTensor::from_vec(&loaded.shape, loaded.data)
    }

    pub fn optional_tensor(&self, name: &str) -> Result<Option<DenseTensor<f32>>> {
        if self.contains(name) {
            Ok(Some(self.tensor(name)?))
        } else {
            Ok(None)
        }
    }

    pub fn expect_shape(&self, name: &str, dims: &[usize]) -> Result<DenseTensor<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.dims() != dims {
            return Err(Error::shape_mismatch(
                format!("Gemma 4 weight {name:?}"),
                oxide_ml_core::Shape::from_static(dims),
                tensor.shape().clone(),
            ));
        }
        Ok(tensor)
    }

    pub fn validate_config(&self, config: &Gemma4Config) -> Result<()> {
        let _ = self.expect_shape(
            "embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        if config.hidden_size_per_layer_input > 0 {
            let _ = self.expect_shape(
                "embed_tokens_per_layer.weight",
                &[
                    config.vocab_size_per_layer_input,
                    config.num_hidden_layers * config.hidden_size_per_layer_input,
                ],
            )?;
            let _ = self.expect_shape(
                "per_layer_model_projection.weight",
                &[
                    config.num_hidden_layers * config.hidden_size_per_layer_input,
                    config.hidden_size,
                ],
            )?;
            let _ = self.expect_shape(
                "per_layer_projection_norm.weight",
                &[config.hidden_size_per_layer_input],
            )?;
        }
        Ok(())
    }
}
