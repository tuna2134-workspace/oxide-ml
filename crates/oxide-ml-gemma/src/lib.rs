//! Gemma 4 text inference on oxide-ml.
//!
//! The dense text architecture (no PLE, no shared KV, no MoE) is implemented
//! with the model API: RMSNorm, GQA attention with per-layer RoPE, GeGLU MLP,
//! tied LM head and final logit soft-capping.

pub mod config;
pub mod modules;
pub mod safetensors {
    pub use oxide_ml_safetensors::*;
}
pub mod weights;

pub use config::{Gemma4Config, LayerConfig, LayerKind, RopeKind};
pub use modules::{Gemma4Text, GemmaAttention, GemmaLayer, GemmaMlp, RmsNorm};
pub use weights::Gemma4Weights;
