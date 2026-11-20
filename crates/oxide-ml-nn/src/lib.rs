//! Neural-network layers for oxide-ml.
//!
//! - [`modules`] contains the graph-independent model API layers. They own
//!   concrete parameters and are replayed into a graph by `#[derive(Module)]`
//!   or by hand-written [`oxide_ml_model::Module`] implementations.
//! - [`losses`] contains loss functions for the model API.
//! - [`legacy`] keeps the original graph-builder layers (`Linear::new(&mut
//!   builder, ...)`) used by the low-level API and the CUDA workspace.

#![allow(clippy::type_complexity)]

pub mod legacy;
pub mod losses;
pub mod modules;
pub mod tied;
pub mod transformer;

pub use losses::{BCEWithLogitsLoss, CrossEntropyLoss, HuberLoss, MSELoss, NLLLoss};
pub use modules::{
    Attention, AvgPool2d, BatchNorm, Conv1d, Conv2d, Conv3d, Dropout, Embedding, GELU, GroupNorm,
    Gru, InstanceNorm, LayerNorm, Linear, Lstm, MaxPool1d, MaxPool2d, MultiHeadAttention, ReLU,
    Rnn, Sigmoid, Softmax, Tanh,
};
pub use tied::{TiedLinear, TiedWeightSource};
pub use transformer::{
    GatedActivation, GatedMlp, GqaConfig, MixtureOfExperts, MoeRouting, RmsNorm, RmsNormNoScale,
    RopeConfig, apply_rope, apply_rope_at, causal_mask, grouped_query_attention, rms_norm,
    rotate_half, rotate_half_value,
};

/// Backwards-compatible path for the original graph-builder layers.
pub use legacy::{
    LayerNorm as BuilderLayerNorm, Linear as BuilderLinear, Mlp as BuilderMlp,
    Module as GraphModule, init_parameters, parameter_ids,
};
