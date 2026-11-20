//! Reusable transformer building blocks.
//!
//! These modules carry no Gemma-specific knowledge. They are the pieces a
//! decoder architecture is assembled from: RMSNorm (with and without a learned
//! scale), gated feed-forward blocks, rotary position tables and attention
//! masks. Architecture-specific configuration is passed to constructors or to
//! the free functions; parameter paths come from the module tree.

use oxide_ml_core::{DenseTensor, Dim, Error, Result, Shape, ValueId};
use oxide_ml_graph::{ConstantTensor, GraphBuilder};
use oxide_ml_model::{ModelContext, Module};

/// RMSNorm over the last axis with a learned scale.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
pub struct RmsNorm {
    #[param]
    pub weight: DenseTensor<f32>,
    #[module(skip)]
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: DenseTensor<f32>, eps: f32) -> Self {
        RmsNorm { weight, eps }
    }

    pub fn with_dim(dim: usize, eps: f32) -> Self {
        RmsNorm {
            weight: DenseTensor::ones([dim]),
            eps,
        }
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        let input = input(inputs, "RMSNorm")?;
        let weight = cx.parameter("weight")?;
        Ok(vec![rms_norm(cx, input, self.eps, Some(weight))?])
    }
}

/// RMSNorm without a learned scale (value normalization).
#[derive(Clone, Debug)]
pub struct RmsNormNoScale {
    pub eps: f32,
}

impl RmsNormNoScale {
    pub fn new(eps: f32) -> Self {
        RmsNormNoScale { eps }
    }
}

impl Module for RmsNormNoScale {
    fn module_type(&self) -> &'static str {
        "RmsNormNoScale"
    }

    fn visit_parameters(
        &self,
        _: &oxide_ml_model::ModulePath,
        _: &mut oxide_ml_model::ParameterRegistry,
    ) -> Result<()> {
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![rms_norm(cx, input, self.eps, None)?])
    }
}

fn input(inputs: &[ValueId], name: &str) -> Result<ValueId> {
    match inputs {
        [value] => Ok(*value),
        other => Err(Error::invalid_graph(format!(
            "{name} expects exactly one input, got {}",
            other.len()
        ))),
    }
}

/// Graph-level RMS normalization. `weight` is applied when present.
pub fn rms_norm(
    cx: &mut ModelContext,
    x: ValueId,
    eps: f32,
    weight: Option<ValueId>,
) -> Result<ValueId> {
    let rank = cx.value_shape(x)?.rank();
    if rank == 0 {
        return Err(Error::invalid_shape("RMSNorm", "input must have rank >= 1"));
    }
    let square = cx.builder().mul(x, x)?;
    let mean = cx.builder().reduce_mean(square, vec![rank - 1], true)?;
    let eps = cx.builder().constant_scalar(eps);
    let mean_eps = cx.builder().add(mean, eps)?;
    let std = cx.builder().sqrt(mean_eps)?;
    let normalized = cx.builder().div(x, std)?;
    match weight {
        Some(weight) => cx.builder().mul(normalized, weight),
        None => Ok(normalized),
    }
}

/// Activation used by a gated feed-forward block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GatedActivation {
    /// `gelu(gate) * up` (GeGLU, the Gemma/Llama-family default).
    #[default]
    GeGlu,
    /// `silu(gate) * up` (SwiGLU).
    SwiGlu,
    /// `relu(gate) * up`.
    ReGLU,
}

/// Gated feed-forward block: `down(activation(gate(x)) * up(x))`.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
pub struct GatedMlp {
    pub gate_proj: crate::modules::Linear,
    pub up_proj: crate::modules::Linear,
    pub down_proj: crate::modules::Linear,
    #[module(skip)]
    pub activation: GatedActivation,
}

impl GatedMlp {
    pub fn new(
        gate_proj: crate::modules::Linear,
        up_proj: crate::modules::Linear,
        down_proj: crate::modules::Linear,
        activation: GatedActivation,
    ) -> Self {
        GatedMlp {
            gate_proj,
            up_proj,
            down_proj,
            activation,
        }
    }

    pub fn geglu(
        gate_proj: crate::modules::Linear,
        up_proj: crate::modules::Linear,
        down_proj: crate::modules::Linear,
    ) -> Self {
        Self::new(gate_proj, up_proj, down_proj, GatedActivation::GeGlu)
    }

    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let gated = if self.activation == GatedActivation::SwiGlu {
            gate * gate.sigmoid()
        } else if self.activation == GatedActivation::ReGLU {
            gate.relu()
        } else {
            gate.gelu()
        };
        self.down_proj.forward(gated * up)
    }
}

/// Rotary position embedding configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeConfig {
    pub head_dim: usize,
    pub theta: f32,
    /// When set, only this fraction of the head is rotated (Gemma 4
    /// "proportional" RoPE). The unrotated tail gets identity tables.
    pub partial_rotary_factor: Option<f32>,
}

impl RopeConfig {
    pub fn new(head_dim: usize, theta: f32) -> Self {
        RopeConfig {
            head_dim,
            theta,
            partial_rotary_factor: None,
        }
    }

    pub fn proportional(head_dim: usize, theta: f32, partial_rotary_factor: f32) -> Self {
        RopeConfig {
            head_dim,
            theta,
            partial_rotary_factor: Some(partial_rotary_factor),
        }
    }

    /// Inverse frequencies for every rotated pair in the head.
    pub fn inverse_frequencies(&self) -> Vec<f32> {
        let pairs = self.head_dim / 2;
        let rotated = match self.partial_rotary_factor {
            Some(factor) => ((factor * self.head_dim as f32 / 2.0) as usize).min(pairs),
            None => pairs,
        };
        let mut frequencies = Vec::with_capacity(pairs);
        for index in 0..rotated {
            let exponent = (2 * index) as f32 / self.head_dim as f32;
            frequencies.push(1.0 / self.theta.powf(exponent));
        }
        frequencies.resize(pairs, 0.0);
        frequencies
    }

    /// `(cos, sin)` tables shaped `[length, head_dim]` starting at `start`.
    pub fn tables(&self, start: usize, length: usize) -> (Vec<f32>, Vec<f32>) {
        let dim = self.head_dim;
        let half = dim / 2;
        let inverse = self.inverse_frequencies();
        let mut cos = vec![0.0f32; length * dim];
        let mut sin = vec![0.0f32; length * dim];
        for offset in 0..length {
            let position = start + offset;
            for index in 0..half {
                let angle = position as f32 * inverse[index];
                let (s, c) = angle.sin_cos();
                cos[offset * dim + index] = c;
                cos[offset * dim + half + index] = c;
                sin[offset * dim + index] = s;
                sin[offset * dim + half + index] = s;
            }
        }
        (cos, sin)
    }
}

/// Rotate half of the head dimension: `rotate_half([x1, x2]) = [-x2, x1]`.
pub fn rotate_half(cx: &mut ModelContext, x: ValueId, head_dim: usize) -> Result<ValueId> {
    let rank = cx.value_shape(x)?.rank();
    if rank == 0 {
        return Err(Error::invalid_shape("RoPE", "input must have rank >= 1"));
    }
    let axis = rank - 1;
    let half = head_dim / 2;
    let first = cx.builder().slice(x, axis, 0, half)?;
    let second = cx.builder().slice(x, axis, half, half)?;
    let neg_second = cx.builder().neg(second)?;
    cx.builder().concat(&[neg_second, first], axis)
}

/// Builder-level `rotate_half`, used by low-level graph code and tests.
pub fn rotate_half_value(
    builder: &mut GraphBuilder,
    x: ValueId,
    head_dim: usize,
    axis: usize,
) -> Result<ValueId> {
    let half = head_dim / 2;
    let first = builder.slice(x, axis, 0, half)?;
    let second = builder.slice(x, axis, half, half)?;
    let neg_second = builder.neg(second)?;
    builder.concat(&[neg_second, first], axis)
}

/// Apply RoPE to a rank-4 `[batch, length, heads, head_dim]` tensor.
///
/// `start` is the absolute position of the first token, which keeps decode and
/// prefill on the same table definition.
pub fn apply_rope_at(
    cx: &mut ModelContext,
    x: ValueId,
    start: usize,
    length: usize,
    config: &RopeConfig,
) -> Result<ValueId> {
    let (cos, sin) = config.tables(start, length);
    let cos = cx.builder().constant_tensor(ConstantTensor::f32(
        Shape::from_static([length, config.head_dim]),
        cos,
    )?)?;
    let sin = cx.builder().constant_tensor(ConstantTensor::f32(
        Shape::from_static([length, config.head_dim]),
        sin,
    )?)?;
    let table_shape = Shape::new(vec![
        Dim::Static(1),
        Dim::Static(length),
        Dim::Static(1),
        Dim::Static(config.head_dim),
    ]);
    let cos = cx.builder().reshape(cos, table_shape.clone())?;
    let sin = cx.builder().reshape(sin, table_shape)?;
    let scaled = cx.builder().mul(x, cos)?;
    let rotated = rotate_half(cx, x, config.head_dim)?;
    let rotated = cx.builder().mul(rotated, sin)?;
    cx.builder().add(scaled, rotated)
}

/// Apply RoPE starting at position zero.
pub fn apply_rope(
    cx: &mut ModelContext,
    x: ValueId,
    length: usize,
    config: &RopeConfig,
) -> Result<ValueId> {
    apply_rope_at(cx, x, 0, length, config)
}

/// Shape of a grouped-query attention head layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GqaConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GqaConfig {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Result<Self> {
        if num_heads == 0 || num_kv_heads == 0 {
            return Err(Error::invalid_shape(
                "GqaConfig",
                "attention needs at least one query and key/value head",
            ));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::invalid_shape(
                "GqaConfig",
                format!("{num_heads} query heads cannot be grouped over {num_kv_heads} KV heads"),
            ));
        }
        Ok(GqaConfig {
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Grouped-query scaled dot-product attention over static sequence lengths.
///
/// Layout: `query` is `[1, query_len, num_heads, head_dim]`, `key`/`value` are
/// `[1, kv_len, num_kv_heads, head_dim]`, and `mask` (when present) is an
/// additive `[query_len, kv_len]` mask. The result is
/// `[1, query_len, num_heads * head_dim]`.
///
/// The single-batch restriction keeps the implementation on the rank-2 matmul
/// that every backend supports; batching is the caller's outer loop. This is
/// the shared math behind GQA prefill and decode: decode passes a one-token
/// query against the cached keys and values.
pub fn grouped_query_attention(
    cx: &mut ModelContext,
    query: ValueId,
    key: ValueId,
    value: ValueId,
    mask: Option<ValueId>,
    scale: Option<f32>,
    config: GqaConfig,
) -> Result<ValueId> {
    let q_shape = cx.value_shape(query)?;
    let k_shape = cx.value_shape(key)?;
    if q_shape.rank() != 4 || k_shape.rank() != 4 {
        return Err(Error::invalid_shape(
            "grouped_query_attention",
            "query/key must be [1, length, heads, head_dim]",
        ));
    }
    let batch = q_shape.dim(0).and_then(Dim::static_usize).unwrap_or(0);
    if batch != 1 {
        return Err(Error::unsupported(
            "grouped_query_attention supports batch size 1; loop over the batch",
        ));
    }
    let q_len = q_shape.dim(1).and_then(Dim::static_usize).ok_or_else(|| {
        Error::invalid_shape("grouped_query_attention", "query length must be static")
    })?;
    let kv_len = k_shape.dim(1).and_then(Dim::static_usize).ok_or_else(|| {
        Error::invalid_shape("grouped_query_attention", "KV length must be static")
    })?;
    let q_heads = q_shape.dim(2).and_then(Dim::static_usize).ok_or_else(|| {
        Error::invalid_shape("grouped_query_attention", "head count must be static")
    })?;
    let kv_heads = k_shape.dim(2).and_then(Dim::static_usize).ok_or_else(|| {
        Error::invalid_shape("grouped_query_attention", "KV head count must be static")
    })?;
    if q_heads != config.num_heads || kv_heads != config.num_kv_heads {
        return Err(Error::invalid_shape(
            "grouped_query_attention",
            format!(
                "tensor layout has {q_heads}/{kv_heads} heads but the config expects {}/{}",
                config.num_heads, config.num_kv_heads
            ),
        ));
    }
    let scale = scale.unwrap_or_else(|| config.scale());
    let scale = cx.builder().constant_scalar(scale);

    let mut heads = Vec::with_capacity(config.num_heads);
    for head in 0..config.num_heads {
        let kv_head = head / config.groups();
        let qh = cx.builder().slice(query, 2, head, 1)?;
        let qh = cx
            .builder()
            .reshape(qh, Shape::from_static([q_len, config.head_dim]))?;
        let kh = cx.builder().slice(key, 2, kv_head, 1)?;
        let kh = cx
            .builder()
            .reshape(kh, Shape::from_static([kv_len, config.head_dim]))?;
        let vh = cx.builder().slice(value, 2, kv_head, 1)?;
        let vh = cx
            .builder()
            .reshape(vh, Shape::from_static([kv_len, config.head_dim]))?;
        let scores = cx.builder().matmul_with(qh, kh, false, true)?;
        let scores = cx.builder().mul(scores, scale)?;
        let scores = match mask {
            Some(mask) => cx.builder().add(scores, mask)?,
            None => scores,
        };
        let weights = cx.builder().softmax(scores, -1)?;
        let out = cx.builder().matmul(weights, vh)?;
        let out = cx
            .builder()
            .reshape(out, Shape::from_static([1, q_len, config.head_dim]))?;
        heads.push(out);
    }
    let merged = cx.builder().concat(&heads, 2)?;
    let merged = cx.builder().reshape(
        merged,
        Shape::from_static([1, q_len, config.num_heads * config.head_dim]),
    )?;
    Ok(merged)
}

/// Routing strategy for [`MixtureOfExperts`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MoeRouting {
    /// Soft routing over every expert: `sum_e softmax(router(x))[e] * expert_e(x)`.
    /// Fully differentiable and works with any expert module.
    #[default]
    Soft,
}

/// A generic mixture-of-experts block with soft routing.
///
/// Experts are ordinary modules (for example [`GatedMlp`]); the router is a
/// `Linear` projection. With soft routing every expert is evaluated and the
/// outputs are combined by the router distribution, so gradients flow through
/// the router and all experts using the normal graph/autograd machinery.
///
/// Sparse top-k routing with expert gathering is available through the fused
/// backend `Op::MoE` used by Gemma 4; it is an inference optimization, not this
/// module's training path.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
pub struct MixtureOfExperts {
    pub router: crate::modules::Linear,
    pub experts: oxide_ml_model::ModuleList<GatedMlp>,
    #[module(skip)]
    pub routing: MoeRouting,
}

impl MixtureOfExperts {
    pub fn new(
        router: crate::modules::Linear,
        experts: oxide_ml_model::ModuleList<GatedMlp>,
    ) -> Result<Self> {
        if experts.is_empty() {
            return Err(Error::invalid_graph(
                "MixtureOfExperts needs at least one expert",
            ));
        }
        if router.out_features != experts.len() {
            return Err(Error::shape_mismatch(
                "MixtureOfExperts router",
                Shape::from_static([experts.len()]),
                Shape::from_static([router.out_features]),
            ));
        }
        Ok(MixtureOfExperts {
            router,
            experts,
            routing: MoeRouting::Soft,
        })
    }

    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        let input = input(inputs, "MixtureOfExperts")?;
        let logits = oxide_ml_model::module_forward(cx, "router", &self.router, input)?;
        let rank = cx.value_shape(logits)?.rank();
        let weights = cx.builder().softmax(logits, rank as i32 - 1)?;

        let mut output: Option<ValueId> = None;
        for (index, expert) in self.experts.iter().enumerate() {
            let expert_output =
                oxide_ml_model::module_forward(cx, &format!("experts.{index}"), expert, input)?;
            let weight = cx.builder().slice(weights, rank as usize - 1, index, 1)?;
            let scaled = cx.builder().mul(expert_output, weight)?;
            output = Some(match output {
                Some(acc) => cx.builder().add(acc, scaled)?,
                None => scaled,
            });
        }
        output
            .map(|value| vec![value])
            .ok_or_else(|| Error::invalid_graph("MixtureOfExperts has no experts"))
    }
}

/// Additive causal mask, optionally limited to a sliding window.
///
/// Visible positions are `0.0`; masked positions use a large negative value
/// that survives a softmax without producing NaNs.
pub fn causal_mask(length: usize, sliding: Option<usize>) -> Vec<f32> {
    let mut mask = vec![0.0f32; length * length];
    for row in 0..length {
        for column in 0..length {
            let visible =
                column <= row && sliding.map(|window| row - column < window).unwrap_or(true);
            if !visible {
                mask[row * length + column] = -1.0e30;
            }
        }
    }
    mask
}
