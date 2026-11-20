//! Gemma 4 text inference on oxide-ml.
//!
//! PLE, shared KV and double-wide MLP are implemented with the model API:
//! RMSNorm, GQA attention with per-layer RoPE, GeGLU MLP, tied LM head and
//! final logit soft-capping. The MLP and the per-head attention math use the
//! `#[derive(Module)]` and `#[model]` macros; the layer/model glue is manual
//! where the macros cannot express shared K/V, PLE or per-head loops.

pub mod config;
pub mod modules;
pub mod streaming;
pub mod tokenizer;
pub mod safetensors {
    pub use oxide_ml_safetensors::*;
}
pub mod weights;

pub use config::{Gemma4Config, LayerConfig, LayerKind, RopeKind};
pub use modules::{Gemma4Text, GemmaAttention, GemmaDecode, GemmaLayer, GemmaMlp, RmsNorm};
pub use streaming::{StreamingGemma, StreamingSession, kv_cache_config};
pub use tokenizer::GemmaTokenizer;
pub use weights::Gemma4Weights;
