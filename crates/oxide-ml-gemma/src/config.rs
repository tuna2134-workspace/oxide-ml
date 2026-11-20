//! Gemma 4 text configuration (subset supported by this implementation).

use oxide_ml_core::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Sliding,
    Full,
}

impl LayerKind {
    pub const fn name(self) -> &'static str {
        match self {
            LayerKind::Sliding => "sliding_attention",
            LayerKind::Full => "full_attention",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeKind {
    /// Standard RoPE over the whole head dimension.
    Default { theta: f32 },
    /// Gemma 4 "proportional" RoPE: only `partial` of the head is rotated and
    /// the remaining inverse frequencies are zero (cos=1, sin=0).
    Proportional {
        theta: f32,
        partial_rotary_factor: f32,
    },
}

impl RopeKind {
    /// Inverse frequencies for one head dimension.
    pub fn inverse_frequencies(self, head_dim: usize) -> Vec<f32> {
        match self {
            RopeKind::Default { theta } => (0..head_dim / 2)
                .map(|index| {
                    let exponent = (2 * index) as f32 / head_dim as f32;
                    1.0 / theta.powf(exponent)
                })
                .collect(),
            RopeKind::Proportional {
                theta,
                partial_rotary_factor,
            } => {
                let angles = (partial_rotary_factor * head_dim as f32 / 2.0) as usize;
                let mut frequencies = Vec::with_capacity(head_dim / 2);
                for index in 0..angles {
                    let exponent = (2 * index) as f32 / head_dim as f32;
                    frequencies.push(1.0 / theta.powf(exponent));
                }
                frequencies.resize(head_dim / 2, 0.0);
                frequencies
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub kind: LayerKind,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub rope: RopeKind,
    pub sliding_window: usize,
    pub is_shared: bool,
}

#[derive(Clone, Debug)]
pub struct Gemma4Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: Option<usize>,
    pub num_global_key_value_heads: Option<usize>,
    pub per_layer_head_dims: HashMap<usize, usize>,
    pub per_layer_kv_heads: HashMap<usize, usize>,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub final_logit_softcapping: Option<f32>,
    pub tie_word_embeddings: bool,
    pub hidden_activation: String,
    pub layer_types: Vec<LayerKind>,
    pub rope_sliding: RopeKind,
    pub rope_full: RopeKind,
    pub num_kv_shared_layers: usize,
    pub hidden_size_per_layer_input: usize,
    pub enable_moe_block: bool,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub use_double_wide_mlp: bool,
    pub vocab_size_per_layer_input: usize,
    pub eos_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
}

impl Gemma4Config {
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|error| {
            Error::Io(format!("cannot read {}: {error}", path.as_ref().display()))
        })?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| Error::Serialization(format!("config.json: {error}")))?;
        Self::from_value(&value)
    }

    pub fn from_value(root: &Value) -> Result<Self> {
        let text = root.get("text_config").unwrap_or(root);
        let get_usize = |name: &str| -> Result<usize> {
            text.get(name)
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .ok_or_else(|| {
                    Error::Serialization(format!("config.json: missing integer {name:?}"))
                })
        };
        let get_f32 = |name: &str| -> Option<f32> {
            text.get(name)
                .and_then(Value::as_f64)
                .map(|value| value as f32)
        };
        let hidden_size = get_usize("hidden_size")?;
        let num_hidden_layers = get_usize("num_hidden_layers")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let num_key_value_heads = get_usize("num_key_value_heads")?;
        let head_dim = get_usize("head_dim")?;
        let global_head_dim = text
            .get("global_head_dim")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let intermediate_size = get_usize("intermediate_size")?;
        let vocab_size = get_usize("vocab_size")?;
        let num_global_key_value_heads = text
            .get("num_global_key_value_heads")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let mut per_layer_head_dims = HashMap::new();
        let mut per_layer_kv_heads = HashMap::new();
        if let Some(overrides) = text.get("per_layer_config").and_then(Value::as_object) {
            for (key, value) in overrides {
                let Ok(index) = key.parse::<usize>() else {
                    continue;
                };
                if let Some(head_dim) = value.get("head_dim").and_then(Value::as_u64) {
                    per_layer_head_dims.insert(index, head_dim as usize);
                }
                if let Some(heads) = value.get("num_key_value_heads").and_then(Value::as_u64) {
                    per_layer_kv_heads.insert(index, heads as usize);
                }
            }
        }
        let num_kv_shared_layers = get_usize("num_kv_shared_layers").unwrap_or(0);
        let hidden_size_per_layer_input = get_usize("hidden_size_per_layer_input").unwrap_or(0);
        let enable_moe_block = text
            .get("enable_moe_block")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let use_double_wide_mlp = text
            .get("use_double_wide_mlp")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let num_experts = text
            .get("num_experts")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let top_k_experts = text
            .get("top_k_experts")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let moe_intermediate_size = text
            .get("moe_intermediate_size")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        if enable_moe_block {
            let experts = num_experts
                .ok_or_else(|| Error::Serialization("MoE requires num_experts".into()))?;
            let top_k = top_k_experts
                .ok_or_else(|| Error::Serialization("MoE requires top_k_experts".into()))?;
            if top_k == 0 || top_k > experts {
                return Err(Error::Serialization(format!(
                    "top_k_experts {top_k} is not in 1..={experts}"
                )));
            }
            if moe_intermediate_size.is_none() {
                return Err(Error::Serialization(
                    "MoE requires moe_intermediate_size".into(),
                ));
            }
        }
        if num_kv_shared_layers > num_hidden_layers {
            return Err(Error::Serialization(
                "num_kv_shared_layers exceeds num_hidden_layers".into(),
            ));
        }
        let layer_types = text
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Serialization("config.json: missing layer_types".into()))?
            .iter()
            .map(|value| match value.as_str() {
                Some("sliding_attention") => Ok(LayerKind::Sliding),
                Some("full_attention") => Ok(LayerKind::Full),
                other => Err(Error::Serialization(format!(
                    "unknown layer type {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;
        if layer_types.len() != num_hidden_layers {
            return Err(Error::Serialization(
                "layer_types length does not match num_hidden_layers".into(),
            ));
        }
        let rope = text.get("rope_parameters").cloned().unwrap_or(Value::Null);
        let rope_sliding = parse_rope(
            rope.get("sliding_attention"),
            RopeKind::Default { theta: 10_000.0 },
        )?;
        let rope_full = parse_rope(
            rope.get("full_attention"),
            RopeKind::Default { theta: 1_000_000.0 },
        )?;
        let sliding_window = get_usize("sliding_window").unwrap_or(512);
        let max_position_embeddings = get_usize("max_position_embeddings").unwrap_or(131_072);
        let tie_word_embeddings = text
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .or_else(|| root.get("tie_word_embeddings").and_then(Value::as_bool))
            .unwrap_or(true);
        let mut eos_token_ids = Vec::new();
        if let Some(value) = text
            .get("eos_token_id")
            .or_else(|| root.get("eos_token_id"))
        {
            if let Some(id) = value.as_u64() {
                eos_token_ids.push(id as u32);
            } else if let Some(array) = value.as_array() {
                for id in array {
                    if let Some(id) = id.as_u64() {
                        eos_token_ids.push(id as u32);
                    }
                }
            }
        }
        let bos_token_id = text
            .get("bos_token_id")
            .or_else(|| root.get("bos_token_id"))
            .and_then(Value::as_u64)
            .map(|id| id as u32);
        let vocab_size_per_layer_input =
            get_usize("vocab_size_per_layer_input").unwrap_or(vocab_size);
        Ok(Gemma4Config {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            global_head_dim,
            num_global_key_value_heads,
            per_layer_head_dims,
            per_layer_kv_heads,
            vocab_size,
            rms_norm_eps: get_f32("rms_norm_eps").unwrap_or(1e-6),
            sliding_window,
            max_position_embeddings,
            final_logit_softcapping: get_f32("final_logit_softcapping"),
            tie_word_embeddings,
            hidden_activation: text
                .get("hidden_activation")
                .and_then(Value::as_str)
                .unwrap_or("gelu_pytorch_tanh")
                .to_string(),
            layer_types,
            rope_sliding,
            rope_full,
            num_kv_shared_layers,
            hidden_size_per_layer_input,
            enable_moe_block,
            num_experts,
            top_k_experts,
            moe_intermediate_size,
            use_double_wide_mlp,
            vocab_size_per_layer_input,
            eos_token_ids,
            bos_token_id,
        })
    }

    pub fn layer(&self, index: usize) -> Result<LayerConfig> {
        let kind = *self
            .layer_types
            .get(index)
            .ok_or_else(|| Error::invalid_graph(format!("layer {index} out of range")))?;
        let head_dim = self
            .per_layer_head_dims
            .get(&index)
            .copied()
            .or_else(|| {
                (kind == LayerKind::Full)
                    .then_some(self.global_head_dim)
                    .flatten()
            })
            .unwrap_or(self.head_dim);
        let num_kv_heads = self
            .per_layer_kv_heads
            .get(&index)
            .copied()
            .or_else(|| {
                (kind == LayerKind::Full)
                    .then_some(self.num_global_key_value_heads)
                    .flatten()
            })
            .unwrap_or(self.num_key_value_heads);
        let first_shared = self
            .num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers);
        Ok(LayerConfig {
            kind,
            head_dim,
            num_heads: self.num_attention_heads,
            num_kv_heads,
            rope: match kind {
                LayerKind::Sliding => self.rope_sliding,
                LayerKind::Full => self.rope_full,
            },
            sliding_window: self.sliding_window,
            is_shared: self.num_kv_shared_layers > 0 && index >= first_shared,
        })
    }
}

fn parse_rope(value: Option<&Value>, default: RopeKind) -> Result<RopeKind> {
    let Some(value) = value else {
        return Ok(default);
    };
    let rope_type = value
        .get("rope_type")
        .and_then(Value::as_str)
        .unwrap_or("default");
    match rope_type {
        "default" => Ok(RopeKind::Default {
            theta: value
                .get("rope_theta")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .unwrap_or(10_000.0),
        }),
        "proportional" => Ok(RopeKind::Proportional {
            theta: value
                .get("rope_theta")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .unwrap_or(1_000_000.0),
            partial_rotary_factor: value
                .get("partial_rotary_factor")
                .and_then(Value::as_f64)
                .map(|value| value as f32)
                .unwrap_or(0.25),
        }),
        other => Err(Error::unsupported(format!(
            "RoPE type {other:?} is not implemented"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_config() -> Value {
        serde_json::json!({
            "text_config": {
                "hidden_size": 8,
                "intermediate_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_global_key_value_heads": 4,
                "head_dim": 32,
                "global_head_dim": 64,
                "vocab_size": 1024,
                "rms_norm_eps": 1e-6,
                "sliding_window": 1024,
                "final_logit_softcapping": 30.0,
                "hidden_size_per_layer_input": 0,
                "num_kv_shared_layers": 0,
                "enable_moe_block": false,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
                },
                "eos_token_id": 1
            }
        })
    }

    #[test]
    fn parses_dense_config() {
        let config = Gemma4Config::from_value(&dense_config()).unwrap();
        assert_eq!(config.hidden_size, 8);
        assert_eq!(config.layer(0).unwrap().head_dim, 32);
        assert_eq!(config.layer(1).unwrap().head_dim, 64);
        assert_eq!(config.layer(0).unwrap().kind, LayerKind::Sliding);
        assert_eq!(config.layer(1).unwrap().kind, LayerKind::Full);
        assert_eq!(config.layer(1).unwrap().num_kv_heads, 4);
        assert_eq!(config.eos_token_ids, vec![1]);
        assert_eq!(config.final_logit_softcapping, Some(30.0));
    }

    #[test]
    fn proportional_rope_zeroes_unrotated_frequencies() {
        let inverse = RopeKind::Proportional {
            theta: 1_000_000.0,
            partial_rotary_factor: 0.25,
        }
        .inverse_frequencies(64);
        assert_eq!(inverse.len(), 32);
        assert!(inverse[0] > 0.0);
        assert!(inverse[8..].iter().all(|value| *value == 0.0));
        let default = RopeKind::Default { theta: 10_000.0 }.inverse_frequencies(32);
        assert_eq!(default.len(), 16);
        assert!(default.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn accepts_ple_shared_kv_and_double_wide() {
        let mut root = dense_config();
        root["text_config"]["hidden_size_per_layer_input"] = serde_json::json!(16);
        root["text_config"]["vocab_size_per_layer_input"] = serde_json::json!(1024);
        root["text_config"]["num_kv_shared_layers"] = serde_json::json!(2);
        root["text_config"]["use_double_wide_mlp"] = serde_json::json!(true);
        let config = Gemma4Config::from_value(&root).unwrap();
        assert_eq!(config.hidden_size_per_layer_input, 16);
        assert_eq!(config.vocab_size_per_layer_input, 1024);
        assert!(config.layer(3).unwrap().is_shared);
        assert!(!config.layer(1).unwrap().is_shared);
    }

    #[test]
    fn parses_per_layer_head_dim_overrides() {
        let mut root = dense_config();
        root["text_config"]["global_head_dim"] = serde_json::Value::Null;
        root["text_config"]["per_layer_config"] =
            serde_json::json!({"1": {"head_dim": 128}, "3": {"head_dim": 128}});
        let config = Gemma4Config::from_value(&root).unwrap();
        assert_eq!(config.layer(1).unwrap().head_dim, 128);
        assert_eq!(config.layer(0).unwrap().head_dim, 32);
    }

    #[test]
    fn parses_moe_config_and_checks_top_k() {
        let mut root = dense_config();
        root["text_config"]["enable_moe_block"] = serde_json::json!(true);
        root["text_config"]["num_experts"] = serde_json::json!(4);
        root["text_config"]["top_k_experts"] = serde_json::json!(2);
        root["text_config"]["moe_intermediate_size"] = serde_json::json!(16);
        let config = Gemma4Config::from_value(&root).unwrap();
        assert!(config.enable_moe_block);
        assert_eq!(config.num_experts, Some(4));
        assert_eq!(config.top_k_experts, Some(2));
        assert_eq!(config.moe_intermediate_size, Some(16));

        root["text_config"]["top_k_experts"] = serde_json::json!(5);
        let error = Gemma4Config::from_value(&root).unwrap_err();
        assert!(error.to_string().contains("top_k_experts"), "{error}");
    }
}
