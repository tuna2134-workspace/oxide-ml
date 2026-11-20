//! Gemma 4 text decoder implemented with oxide-ml graph modules.
//!
//! Supports the dense text architecture, per-layer input embeddings (PLE),
//! shared KV layers and the double-wide MLP used by Gemma 4 E2B.

use std::sync::Arc;

use oxide_ml::{Module, model};
use oxide_ml_core::{DenseTensor, Dim, Error, Result, Shape, ValueId};
use oxide_ml_graph::ConstantTensor;
use oxide_ml_model::ops;
use oxide_ml_model::{
    ModelContext, ModuleList, ModulePath, ParameterRegistry, ParameterRole, module_forward,
    module_forward_slots, single_output,
};
use oxide_ml_nn::modules::{Embedding, Linear};
use oxide_ml_nn::transformer::RopeConfig;

use crate::config::{Gemma4Config, LayerConfig, LayerKind, RopeKind};
use crate::weights::Gemma4Weights;

fn constant_scalar(cx: &mut ModelContext, value: f32) -> ValueId {
    cx.builder().constant_scalar(value)
}

// Norms, rotary tables and masks are generic transformer components.
pub use oxide_ml_nn::{RmsNorm, RmsNormNoScale};
/// Backwards-compatible name for the no-scale value norm.
pub type NoScaleRmsNorm = RmsNormNoScale;

/// Architecture-specific RoPE parameters for one decoder layer.
fn rope_config(layer: &LayerConfig) -> RopeConfig {
    match layer.rope {
        RopeKind::Default { theta } => RopeConfig::new(layer.head_dim, theta),
        RopeKind::Proportional {
            theta,
            partial_rotary_factor,
        } => RopeConfig::proportional(layer.head_dim, theta, partial_rotary_factor),
    }
}

/// Gemma 4 attention (GQA, per-layer RoPE, optional shared K/V).
///
/// The normal prefill path is a declarative `#[model]` method: projections,
/// QK/V norms, RoPE and grouped-query attention are composed from framework
/// ops. Decode keeps a small explicit wrapper because the RoPE position and the
/// concatenation of cached K/V are runtime state (the framework's KV-cache
/// extension), not model structure.
#[derive(Clone, Debug, Module)]
#[oxide(slots)]
pub struct GemmaAttention {
    pub q_proj: Linear,
    pub k_proj: Option<Linear>,
    pub v_proj: Option<Linear>,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: Option<RmsNorm>,
    pub v_norm: Option<NoScaleRmsNorm>,
    #[module(skip)]
    pub layer: LayerConfig,
    #[module(skip)]
    pub eps: f32,
    #[module(skip)]
    pub store_kv: bool,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub num_kv_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub rope_theta: f32,
    #[module(skip)]
    pub rope_partial: Option<f32>,
    #[module(skip)]
    pub sliding_window: Option<usize>,
    #[module(skip)]
    pub has_v_proj: bool,
    #[module(skip)]
    pub is_shared: bool,
}

impl GemmaAttention {
    /// Declarative attention for prefill (and shared-KV layers, which receive
    /// the source layer's K/V through the optional slots).
    ///
    /// Returns `(context, keys, values)` so the caller can thread K/V through a
    /// KV cache without a second attention implementation.
    #[model]
    fn forward(
        &self,
        x: oxide_ml::Tensor<f32>,
        past_k: Option<oxide_ml::Tensor<f32>>,
        past_v: Option<oxide_ml::Tensor<f32>>,
    ) -> (
        oxide_ml::Tensor<f32>,
        oxide_ml::Tensor<f32>,
        oxide_ml::Tensor<f32>,
    ) {
        let q = self.q_proj.forward(x);
        let q = split_heads(q, self.num_heads, self.head_dim);
        let q = self.q_norm.forward(q);
        let q = apply_rotary(q, self.head_dim, self.rope_theta, self.rope_partial, 0);

        let (k, v) = if self.is_shared {
            let k = require_slot(past_k, "shared attention layer has no cached K")?;
            let v = require_slot(past_v, "shared attention layer has no cached V")?;
            (k, v)
        } else {
            let k_raw = self.k_proj.forward(x);
            let v_raw = if self.has_v_proj {
                self.v_proj.forward(x)
            } else {
                k_raw
            };
            let k = split_heads(k_raw, self.num_kv_heads, self.head_dim);
            let v = split_heads(v_raw, self.num_kv_heads, self.head_dim);
            let k = self.k_norm.forward(k);
            let v = self.v_norm.forward(v);
            let k = apply_rotary(k, self.head_dim, self.rope_theta, self.rope_partial, 0);
            (k, v)
        };

        let context = grouped_attention(
            q,
            k,
            v,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window,
            true,
            Some(1.0),
        );
        let out = self.o_proj.forward(context);
        (out, k, v)
    }

    /// One-token decode against cached K/V (query length 1).
    pub(crate) fn decode(
        &self,
        cx: &mut ModelContext,
        x: ValueId,
        past: Option<(ValueId, ValueId)>,
        position: usize,
    ) -> Result<(ValueId, Option<(ValueId, ValueId)>)> {
        let shape = cx.value_shape(x)?;
        if shape.rank() != 3
            || shape.dim(0) != Some(Dim::Static(1))
            || shape.dim(1) != Some(Dim::Static(1))
        {
            return Err(Error::invalid_shape(
                "Gemma4 decode attention",
                "expected hidden [1, 1, H]",
            ));
        }
        let heads = self.num_heads;
        let kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;

        let q = module_forward(cx, "q_proj", &self.q_proj, x)?;
        let q = ops::split_heads(cx, q, heads, head_dim)?;
        let q = module_forward(cx, "q_norm", &self.q_norm, q)?;
        let q = ops::apply_rotary(
            cx,
            q,
            head_dim,
            self.rope_theta,
            self.rope_partial,
            position,
        )?;

        let (k, v, stored) = if self.is_shared {
            let (k, v) = past.ok_or_else(|| {
                Error::invalid_graph(format!(
                    "shared KV layer {} has no cached state",
                    self.layer.kind.name()
                ))
            })?;
            (k, v, None)
        } else {
            let k_proj = self
                .k_proj
                .as_ref()
                .ok_or_else(|| Error::invalid_graph("missing k_proj"))?;
            let raw_k = module_forward(cx, "k_proj", k_proj, x)?;
            let raw_v = match &self.v_proj {
                Some(projection) => module_forward(cx, "v_proj", projection, x)?,
                None => raw_k,
            };
            let k = ops::split_heads(cx, raw_k, kv_heads, head_dim)?;
            let v = ops::split_heads(cx, raw_v, kv_heads, head_dim)?;
            let k = module_forward(
                cx,
                "k_norm",
                self.k_norm
                    .as_ref()
                    .ok_or_else(|| Error::invalid_graph("missing k_norm"))?,
                k,
            )?;
            let v = module_forward(
                cx,
                "v_norm",
                self.v_norm
                    .as_ref()
                    .ok_or_else(|| Error::invalid_graph("missing v_norm"))?,
                v,
            )?;
            let k = ops::apply_rotary(
                cx,
                k,
                head_dim,
                self.rope_theta,
                self.rope_partial,
                position,
            )?;
            let (full_k, full_v) = match past {
                Some((past_k, past_v)) => (
                    cx.builder().concat(&[past_k, k], 1)?,
                    cx.builder().concat(&[past_v, v], 1)?,
                ),
                None => (k, v),
            };
            (full_k, full_v, Some((full_k, full_v)))
        };

        let kv_len = cx
            .value_shape(k)?
            .dim(1)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("Gemma4 decode", "cache length must be static"))?;
        let mut mask = vec![0.0f32; kv_len];
        if let Some(window) = self.sliding_window {
            for (key, value) in mask.iter_mut().enumerate() {
                if position.saturating_sub(key) >= window {
                    *value = -1.0e30;
                }
            }
        }
        let mask = cx
            .builder()
            .constant_tensor(ConstantTensor::f32(Shape::from_static([1, kv_len]), mask)?)?;
        let merged = ops::grouped_attention_masked(
            cx,
            q,
            k,
            v,
            heads,
            kv_heads,
            head_dim,
            Some(mask),
            Some(1.0),
        )?;
        Ok((module_forward(cx, "o_proj", &self.o_proj, merged)?, stored))
    }
}

fn require_slot(value: Option<ValueId>, message: &str) -> Result<ValueId> {
    value.ok_or_else(|| Error::invalid_graph(message))
}

/// GeGLU feed-forward block (double-wide on shared-KV layers).
///
/// This is the generic [`oxide_ml_nn::GatedMlp`] with the GeGLU activation; the
/// alias keeps the Gemma-specific name used in the weight loader.
pub type GemmaMlp = oxide_ml_nn::GatedMlp;

/// Fused Gemma 4 mixture-of-experts block.
#[derive(Clone, Debug)]
pub struct GemmaMoE {
    pub router_scale: DenseTensor<f32>,
    pub per_expert_scale: DenseTensor<f32>,
    pub router_proj: DenseTensor<f32>,
    pub gate_up: DenseTensor<f32>,
    pub down: DenseTensor<f32>,
    pub expert_norm: DenseTensor<f32>,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
    pub eps: f32,
}

impl GemmaMoE {
    pub(crate) fn from_weights(
        weights: &Gemma4Weights<'_>,
        prefix: &str,
        config: &Gemma4Config,
    ) -> Result<Self> {
        let num_experts = config
            .num_experts
            .ok_or_else(|| Error::invalid_graph("MoE requires num_experts"))?;
        let top_k = config
            .top_k_experts
            .ok_or_else(|| Error::invalid_graph("MoE requires top_k_experts"))?;
        let intermediate = config
            .moe_intermediate_size
            .ok_or_else(|| Error::invalid_graph("MoE requires moe_intermediate_size"))?;
        Ok(GemmaMoE {
            router_scale: weights
                .expect_shape(&format!("{prefix}.router.scale"), &[config.hidden_size])?,
            per_expert_scale: weights
                .expect_shape(&format!("{prefix}.router.per_expert_scale"), &[num_experts])?,
            router_proj: weights.expect_shape(
                &format!("{prefix}.router.proj.weight"),
                &[num_experts, config.hidden_size],
            )?,
            gate_up: weights.expect_shape(
                &format!("{prefix}.experts.gate_up_proj"),
                &[num_experts, 2 * intermediate, config.hidden_size],
            )?,
            down: weights.expect_shape(
                &format!("{prefix}.experts.down_proj"),
                &[num_experts, config.hidden_size, intermediate],
            )?,
            expert_norm: weights.expect_shape(
                &format!("{prefix}.pre_feedforward_layernorm_2.weight"),
                &[config.hidden_size],
            )?,
            num_experts,
            top_k,
            intermediate,
            eps: config.rms_norm_eps,
        })
    }

    pub(crate) fn dummy(config: &Gemma4Config) -> Result<Self> {
        let num_experts = config
            .num_experts
            .ok_or_else(|| Error::invalid_graph("MoE requires num_experts"))?;
        let top_k = config
            .top_k_experts
            .ok_or_else(|| Error::invalid_graph("MoE requires top_k_experts"))?;
        let intermediate = config
            .moe_intermediate_size
            .ok_or_else(|| Error::invalid_graph("MoE requires moe_intermediate_size"))?;
        let zeros = |dims: &[usize]| -> Result<DenseTensor<f32>> {
            let numel: usize = dims.iter().product();
            DenseTensor::from_vec(dims, vec![0.0f32; numel])
        };
        Ok(GemmaMoE {
            router_scale: zeros(&[config.hidden_size])?,
            per_expert_scale: zeros(&[num_experts])?,
            router_proj: zeros(&[num_experts, config.hidden_size])?,
            gate_up: zeros(&[num_experts, 2 * intermediate, config.hidden_size])?,
            down: zeros(&[num_experts, config.hidden_size, intermediate])?,
            expert_norm: zeros(&[config.hidden_size])?,
            num_experts,
            top_k,
            intermediate,
            eps: config.rms_norm_eps,
        })
    }

    fn apply(&self, cx: &mut ModelContext, x: ValueId) -> Result<ValueId> {
        let scale = cx.parameter("router.scale")?;
        let per_expert = cx.parameter("router.per_expert_scale")?;
        let router = cx.parameter("router.proj.weight")?;
        let gate_up = cx.parameter("experts.gate_up_proj")?;
        let down = cx.parameter("experts.down_proj")?;
        let expert_norm = cx.parameter("expert_norm")?;
        cx.builder().moe(
            &[x, scale, per_expert, router, gate_up, down, expert_norm],
            self.num_experts,
            self.top_k,
            self.intermediate,
            self.eps,
        )
    }
}

impl Module for GemmaMoE {
    fn module_type(&self) -> &'static str {
        "Gemma4MoE"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        registry.insert_tensor(
            &path.child("router").child("scale"),
            &self.router_scale,
            ParameterRole::Trainable,
        )?;
        registry.insert_tensor(
            &path.child("router").child("per_expert_scale"),
            &self.per_expert_scale,
            ParameterRole::Trainable,
        )?;
        registry.insert_tensor(
            &path.child("router").child("proj").child("weight"),
            &self.router_proj,
            ParameterRole::Trainable,
        )?;
        registry.insert_tensor(
            &path.child("experts").child("gate_up_proj"),
            &self.gate_up,
            ParameterRole::Trainable,
        )?;
        registry.insert_tensor(
            &path.child("experts").child("down_proj"),
            &self.down,
            ParameterRole::Trainable,
        )?;
        registry.insert_tensor(
            &path.child("expert_norm"),
            &self.expert_norm,
            ParameterRole::Trainable,
        )?;
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![self.apply(cx, input)?])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visitor(
            &path.child("router").child("scale"),
            &self.router_scale,
            ParameterRole::Trainable,
        )?;
        visitor(
            &path.child("router").child("per_expert_scale"),
            &self.per_expert_scale,
            ParameterRole::Trainable,
        )?;
        visitor(
            &path.child("router").child("proj").child("weight"),
            &self.router_proj,
            ParameterRole::Trainable,
        )?;
        visitor(
            &path.child("experts").child("gate_up_proj"),
            &self.gate_up,
            ParameterRole::Trainable,
        )?;
        visitor(
            &path.child("experts").child("down_proj"),
            &self.down,
            ParameterRole::Trainable,
        )?;
        visitor(
            &path.child("expert_norm"),
            &self.expert_norm,
            ParameterRole::Trainable,
        )?;
        Ok(())
    }
}

/// One Gemma 4 decoder layer.
///
/// Parameter traversal comes from `#[derive(Module)]`; the prefill computation
/// is a declarative `#[model]` method. PLE and shared-KV state are optional
/// slots threaded by the framework's `forward_slots` extension; the fused MoE
/// branch is a custom `Module`.
#[derive(Clone, Debug, Module)]
#[oxide(slots)]
pub struct GemmaLayer {
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub pre_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm: RmsNorm,
    pub self_attn: GemmaAttention,
    pub mlp: GemmaMlp,
    #[param]
    pub layer_scalar: DenseTensor<f32>,
    pub per_layer_input_gate: Option<Linear>,
    pub per_layer_projection: Option<Linear>,
    pub post_per_layer_input_norm: Option<RmsNorm>,
    pub moe: Option<GemmaMoE>,
    pub post_feedforward_layernorm_1: Option<RmsNorm>,
    pub post_feedforward_layernorm_2: Option<RmsNorm>,
}

impl GemmaLayer {
    /// Declarative decoder layer for prefill.
    ///
    /// `per_layer` carries the PLE residual when the checkpoint enables it and
    /// `past_k`/`past_v` carry the source layer's K/V for shared-KV layers.
    /// K/V are always returned so the runtime can maintain a KV cache without
    /// duplicating the layer math.
    #[model]
    fn forward(
        &self,
        x: oxide_ml::Tensor<f32>,
        per_layer: Option<oxide_ml::Tensor<f32>>,
        past_k: Option<oxide_ml::Tensor<f32>>,
        past_v: Option<oxide_ml::Tensor<f32>>,
    ) -> (
        oxide_ml::Tensor<f32>,
        oxide_ml::Tensor<f32>,
        oxide_ml::Tensor<f32>,
    ) {
        let residual = x.clone();
        let hidden = self.input_layernorm.forward(x);
        let (attended, k, v) = self.self_attn.forward_slots(Some(hidden), past_k, past_v);
        let attended = self.post_attention_layernorm.forward(attended);
        let mut hidden = residual + attended;

        let residual = hidden.clone();
        let ff = self.pre_feedforward_layernorm.forward(hidden);
        let ff = if self.moe.is_some() {
            let mlp_branch = self.mlp.forward(ff);
            let mlp_branch = self.post_feedforward_layernorm_1.forward(mlp_branch);
            let moe_branch = self.moe.forward(residual);
            let moe_branch = self.post_feedforward_layernorm_2.forward(moe_branch);
            mlp_branch + moe_branch
        } else {
            self.mlp.forward(ff)
        };
        let ff = self.post_feedforward_layernorm.forward(ff);
        hidden = residual + ff;

        if let Some(per_layer) = per_layer {
            let residual = hidden.clone();
            let gated = self.per_layer_input_gate.forward(hidden);
            let gated = gated.gelu() * per_layer;
            let projected = self.per_layer_projection.forward(gated);
            let projected = self.post_per_layer_input_norm.forward(projected);
            hidden = residual + projected;
        }

        let hidden = hidden * parameter("layer_scalar");
        (hidden, k, v)
    }

    pub(crate) fn apply_with(
        &self,
        cx: &mut ModelContext,
        input: ValueId,
        per_layer_input: Option<ValueId>,
        shared: Option<(ValueId, ValueId)>,
    ) -> Result<(ValueId, Option<(ValueId, ValueId)>)> {
        let outputs = self.forward_slots(
            cx,
            &[
                Some(input),
                per_layer_input,
                shared.map(|(keys, _)| keys),
                shared.map(|(_, values)| values),
            ],
        )?;
        let stored = (!self.self_attn.is_shared && self.self_attn.store_kv)
            .then(|| (outputs[1], outputs[2]));
        Ok((outputs[0], stored))
    }
}

impl GemmaLayer {
    /// Zero-weight layer with the shapes implied by `config`; used only to
    /// build and compile the per-layer graph for streaming inference.
    pub(crate) fn dummy(
        config: &Gemma4Config,
        index: usize,
        intermediate: usize,
        store_kv: bool,
    ) -> Result<Self> {
        let layer = config.layer(index)?;
        let hidden = config.hidden_size;
        let heads = layer.num_heads;
        let kv_heads = layer.num_kv_heads;
        let head_dim = layer.head_dim;
        let zeros = |dims: &[usize]| -> Result<DenseTensor<f32>> {
            let numel: usize = dims.iter().product();
            DenseTensor::from_vec(dims, vec![0.0f32; numel])
        };
        let linear = |out_features: usize, in_features: usize| -> Result<Linear> {
            Linear::from_tensors(zeros(&[out_features, in_features])?, None)
        };
        let (k_proj, v_proj, k_norm, v_norm) = if layer.is_shared {
            (None, None, None, None)
        } else {
            (
                Some(linear(kv_heads * head_dim, hidden)?),
                Some(linear(kv_heads * head_dim, hidden)?),
                Some(RmsNorm::new(zeros(&[head_dim])?, config.rms_norm_eps)),
                Some(NoScaleRmsNorm {
                    eps: config.rms_norm_eps,
                }),
            )
        };
        let ple_dim = config.hidden_size_per_layer_input;
        let (gate, projection, norm) = if ple_dim > 0 {
            (
                Some(linear(ple_dim, hidden)?),
                Some(linear(hidden, ple_dim)?),
                Some(RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps)),
            )
        } else {
            (None, None, None)
        };
        let moe = if config.enable_moe_block {
            Some(GemmaMoE::dummy(config)?)
        } else {
            None
        };
        let (post1, post2) = if config.enable_moe_block {
            (
                Some(RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps)),
                Some(RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps)),
            )
        } else {
            (None, None)
        };
        Ok(GemmaLayer {
            input_layernorm: RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps),
            post_attention_layernorm: RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps),
            pre_feedforward_layernorm: RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps),
            post_feedforward_layernorm: RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps),
            self_attn: GemmaAttention {
                q_proj: linear(heads * head_dim, hidden)?,
                has_v_proj: v_proj.is_some(),
                k_proj,
                v_proj,
                o_proj: linear(hidden, heads * head_dim)?,
                q_norm: RmsNorm::new(zeros(&[head_dim])?, config.rms_norm_eps),
                k_norm,
                v_norm,
                num_heads: layer.num_heads,
                num_kv_heads: layer.num_kv_heads,
                head_dim: layer.head_dim,
                rope_theta: rope_config(&layer).theta,
                rope_partial: rope_config(&layer).partial_rotary_factor,
                sliding_window: (layer.kind == LayerKind::Sliding).then_some(layer.sliding_window),
                is_shared: layer.is_shared,
                layer: layer.clone(),
                eps: config.rms_norm_eps,
                store_kv,
            },
            mlp: GemmaMlp::geglu(
                linear(intermediate, hidden)?,
                linear(intermediate, hidden)?,
                linear(hidden, intermediate)?,
            ),
            layer_scalar: zeros(&[1])?,
            per_layer_input_gate: gate,
            per_layer_projection: projection,
            post_per_layer_input_norm: norm,
            moe,
            post_feedforward_layernorm_1: post1,
            post_feedforward_layernorm_2: post2,
        })
    }
}

impl GemmaLayer {
    /// One-token decode using cached K/V.
    pub(crate) fn decode_with(
        &self,
        cx: &mut ModelContext,
        input: ValueId,
        per_layer_input: Option<ValueId>,
        past: Option<(ValueId, ValueId)>,
        position: usize,
    ) -> Result<(ValueId, Option<(ValueId, ValueId)>)> {
        let residual = input;
        let hidden = module_forward(cx, "input_layernorm", &self.input_layernorm, input)?;
        let depth = cx.depth();
        cx.enter("self_attn");
        let result = self.self_attn.decode(cx, hidden, past, position);
        cx.truncate(depth);
        let (attended, stored) = result?;
        let attended = module_forward(
            cx,
            "post_attention_layernorm",
            &self.post_attention_layernorm,
            attended,
        )?;
        let hidden = cx.builder().add(residual, attended)?;

        let residual = hidden;
        let ff = module_forward(
            cx,
            "pre_feedforward_layernorm",
            &self.pre_feedforward_layernorm,
            hidden,
        )?;
        let ff = module_forward(cx, "mlp", &self.mlp, ff)?;
        let ff = if let Some(moe) = &self.moe {
            let mlp_branch = module_forward(
                cx,
                "post_feedforward_layernorm_1",
                self.post_feedforward_layernorm_1
                    .as_ref()
                    .ok_or_else(|| Error::invalid_graph("missing post_feedforward_layernorm_1"))?,
                ff,
            )?;
            let moe_branch = module_forward(cx, "moe", moe, residual)?;
            let moe_branch = module_forward(
                cx,
                "post_feedforward_layernorm_2",
                self.post_feedforward_layernorm_2
                    .as_ref()
                    .ok_or_else(|| Error::invalid_graph("missing post_feedforward_layernorm_2"))?,
                moe_branch,
            )?;
            cx.builder().add(mlp_branch, moe_branch)?
        } else {
            ff
        };
        let ff = module_forward(
            cx,
            "post_feedforward_layernorm",
            &self.post_feedforward_layernorm,
            ff,
        )?;
        let mut hidden = cx.builder().add(residual, ff)?;

        if let (Some(gate), Some(projection), Some(norm), Some(per_layer)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = hidden;
            let gated = module_forward(cx, "per_layer_input_gate", gate, hidden)?;
            let gated = cx.builder().gelu(gated)?;
            let gated = cx.builder().mul(gated, per_layer)?;
            let projected = module_forward(cx, "per_layer_projection", projection, gated)?;
            let projected = module_forward(cx, "post_per_layer_input_norm", norm, projected)?;
            hidden = cx.builder().add(residual, projected)?;
        }

        let scalar = cx.parameter("layer_scalar")?;
        Ok((cx.builder().mul(hidden, scalar)?, stored))
    }
}

/// Language-model head tied to the token embedding.
///
/// This is the general-purpose [`oxide_ml_nn::TiedLinear`] with leading
/// dimensions flattened ([1, sequence, hidden] -> [sequence, hidden]). The
/// registry records `lm_head.weight` as an alias of `embed_tokens.weight`.
pub type TiedHead = oxide_ml_nn::TiedLinear<Embedding>;

/// Complete Gemma 4 text model.
///
/// The model structure is declared by the fields and `#[derive(Module)]`;
/// shared-KV routing across layers remains a runtime concern (see
/// `forward_multi`) and the KV state layout is declared by `state_config`.
#[derive(Clone, Debug, Module)]
#[oxide(name = "Gemma4TextModel", state)]
pub struct Gemma4Text {
    #[module(skip)]
    pub config: Gemma4Config,
    pub embed_tokens: Arc<Embedding>,
    pub embed_tokens_per_layer: Option<Embedding>,
    pub per_layer_model_projection: Option<Linear>,
    pub per_layer_projection_norm: Option<RmsNorm>,
    pub layers: ModuleList<GemmaLayer>,
    pub norm: RmsNorm,
    pub lm_head: TiedHead,
}

impl Gemma4Text {
    /// Decode one token with per-layer cached K/V.
    ///
    /// `past[i]` is the cache for non-shared layer `i`; shared layers reuse the
    /// K/V produced by their source layer inside this call.
    pub fn decode_graph(
        &self,
        cx: &mut ModelContext,
        token: ValueId,
        past: &[Option<(ValueId, ValueId)>],
        position: usize,
    ) -> Result<(ValueId, DecodedKv)> {
        let config = &self.config;
        let hidden_size = config.hidden_size;
        let _ = hidden_size;
        let scaled = single_output(self.embed_scale(cx, token)?)?;
        let per_layer = self.compute_per_layer_inputs(cx, token, scaled, 1)?;
        let mut hidden = cx
            .builder()
            .reshape(scaled, Shape::from_static([1, 1, hidden_size]))?;

        let first_shared = config
            .num_hidden_layers
            .saturating_sub(config.num_kv_shared_layers);
        let mut store_flags = vec![false; config.num_hidden_layers];
        if config.num_kv_shared_layers > 0 {
            for kind in [LayerKind::Sliding, LayerKind::Full] {
                if let Some(index) = (0..first_shared)
                    .filter(|index| config.layer(*index).map(|layer| layer.kind) == Ok(kind))
                    .max()
                {
                    store_flags[index] = true;
                }
            }
        }
        let mut source: [Option<(ValueId, ValueId)>; 2] = [None, None];
        let mut updated = vec![None; config.num_hidden_layers];
        for index in 0..config.num_hidden_layers {
            let layer = config.layer(index)?;
            let slot = if layer.kind == LayerKind::Sliding {
                0
            } else {
                1
            };
            let cached = if layer.is_shared {
                source[slot].ok_or_else(|| {
                    Error::invalid_graph(format!(
                        "shared layer {index} has no source K/V during decode"
                    ))
                })?
            } else {
                past.get(index).and_then(|value| *value).ok_or_else(|| {
                    Error::invalid_graph(format!(
                        "non-shared layer {index} has no cached K/V during decode"
                    ))
                })?
            };
            let per_layer_input = per_layer.as_ref().map(|values| values[index]);
            let depth = cx.depth();
            for segment in format!("layers.{index}").split('.') {
                if !segment.is_empty() {
                    cx.enter(segment);
                }
            }
            let result =
                self.layers[index].decode_with(cx, hidden, per_layer_input, Some(cached), position);
            cx.truncate(depth);
            let (out, stored) = result?;
            if let Some((k, v)) = stored {
                updated[index] = Some((k, v));
                if store_flags[index] {
                    source[slot] = Some((k, v));
                }
            }
            hidden = out;
        }
        let hidden = module_forward(cx, "norm", &self.norm, hidden)?;
        let mut logits = module_forward(cx, "lm_head", &self.lm_head, hidden)?;
        if let Some(cap) = config.final_logit_softcapping {
            let cap_value = constant_scalar(cx, cap);
            let scaled = cx.builder().div(logits, cap_value)?;
            let bounded = cx.builder().tanh(scaled)?;
            logits = cx.builder().mul(bounded, cap_value)?;
        }
        Ok((logits, updated))
    }

    pub fn from_weights(config: Gemma4Config, weights: &Gemma4Weights<'_>) -> Result<Self> {
        weights.validate_config(&config)?;
        let embed_tokens = Arc::new(Embedding::from_weight(
            weights.tensor("embed_tokens.weight")?,
        )?);
        let ple_dim = config.hidden_size_per_layer_input;
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if ple_dim > 0 {
                (
                    Some(Embedding::from_weight(
                        weights.tensor("embed_tokens_per_layer.weight")?,
                    )?),
                    Some(Linear::from_tensors(
                        weights.tensor("per_layer_model_projection.weight")?,
                        None,
                    )?),
                    Some(RmsNorm::new(
                        weights.tensor("per_layer_projection_norm.weight")?,
                        config.rms_norm_eps,
                    )),
                )
            } else {
                (None, None, None)
            };

        let first_shared = config
            .num_hidden_layers
            .saturating_sub(config.num_kv_shared_layers);
        let mut store_kv = vec![false; config.num_hidden_layers];
        if config.num_kv_shared_layers > 0 {
            for kind in [LayerKind::Sliding, LayerKind::Full] {
                if let Some(index) = (0..first_shared)
                    .filter(|index| config.layer(*index).map(|layer| layer.kind) == Ok(kind))
                    .max()
                {
                    store_kv[index] = true;
                }
            }
        }

        let mut layers = ModuleList::with_capacity(config.num_hidden_layers);
        for (index, store) in store_kv.iter().copied().enumerate() {
            let layer = config.layer(index)?;
            let prefix = format!("layers.{index}");
            let attention_prefix = format!("{prefix}.self_attn");
            let q_proj = Linear::from_tensors(
                weights.tensor(&format!("{attention_prefix}.q_proj.weight"))?,
                None,
            )?;
            let o_proj = Linear::from_tensors(
                weights.tensor(&format!("{attention_prefix}.o_proj.weight"))?,
                None,
            )?;
            let (k_proj, v_proj, k_norm, v_norm) = if layer.is_shared {
                (None, None, None, None)
            } else {
                (
                    Some(Linear::from_tensors(
                        weights.tensor(&format!("{attention_prefix}.k_proj.weight"))?,
                        None,
                    )?),
                    Some(Linear::from_tensors(
                        weights.tensor(&format!("{attention_prefix}.v_proj.weight"))?,
                        None,
                    )?),
                    Some(RmsNorm::new(
                        weights.tensor(&format!("{attention_prefix}.k_norm.weight"))?,
                        config.rms_norm_eps,
                    )),
                    Some(NoScaleRmsNorm {
                        eps: config.rms_norm_eps,
                    }),
                )
            };
            let has_v_proj = v_proj.is_some();
            let rope = rope_config(&layer);
            let attention = GemmaAttention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm: RmsNorm::new(
                    weights.tensor(&format!("{attention_prefix}.q_norm.weight"))?,
                    config.rms_norm_eps,
                ),
                k_norm,
                v_norm,
                num_heads: layer.num_heads,
                num_kv_heads: layer.num_kv_heads,
                head_dim: layer.head_dim,
                rope_theta: rope.theta,
                rope_partial: rope.partial_rotary_factor,
                sliding_window: (layer.kind == LayerKind::Sliding).then_some(layer.sliding_window),
                is_shared: layer.is_shared,
                has_v_proj,
                layer,
                eps: config.rms_norm_eps,
                store_kv: store,
            };
            let mlp = GemmaMlp::geglu(
                Linear::from_tensors(
                    weights.tensor(&format!("{prefix}.mlp.gate_proj.weight"))?,
                    None,
                )?,
                Linear::from_tensors(
                    weights.tensor(&format!("{prefix}.mlp.up_proj.weight"))?,
                    None,
                )?,
                Linear::from_tensors(
                    weights.tensor(&format!("{prefix}.mlp.down_proj.weight"))?,
                    None,
                )?,
            );
            let layer_scalar = weights.tensor(&format!("{prefix}.layer_scalar"))?;
            if layer_scalar.numel() != 1 {
                return Err(Error::shape_mismatch(
                    format!("layer {index} scalar"),
                    Shape::from_static([1]),
                    layer_scalar.shape().clone(),
                ));
            }
            let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) = if ple_dim
                > 0
            {
                (
                    Some(Linear::from_tensors(
                        weights.tensor(&format!("{prefix}.per_layer_input_gate.weight"))?,
                        None,
                    )?),
                    Some(Linear::from_tensors(
                        weights.tensor(&format!("{prefix}.per_layer_projection.weight"))?,
                        None,
                    )?),
                    Some(RmsNorm::new(
                        weights.tensor(&format!("{prefix}.post_per_layer_input_norm.weight"))?,
                        config.rms_norm_eps,
                    )),
                )
            } else {
                (None, None, None)
            };
            let (moe, post1, post2) = if config.enable_moe_block {
                (
                    Some(GemmaMoE::from_weights(weights, &prefix, &config)?),
                    Some(RmsNorm::new(
                        weights.tensor(&format!("{prefix}.post_feedforward_layernorm_1.weight"))?,
                        config.rms_norm_eps,
                    )),
                    Some(RmsNorm::new(
                        weights.tensor(&format!("{prefix}.post_feedforward_layernorm_2.weight"))?,
                        config.rms_norm_eps,
                    )),
                )
            } else {
                (None, None, None)
            };
            layers.push(GemmaLayer {
                input_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{prefix}.input_layernorm.weight"))?,
                    config.rms_norm_eps,
                ),
                post_attention_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
                    config.rms_norm_eps,
                ),
                pre_feedforward_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
                    config.rms_norm_eps,
                ),
                post_feedforward_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{prefix}.post_feedforward_layernorm.weight"))?,
                    config.rms_norm_eps,
                ),
                self_attn: attention,
                mlp,
                layer_scalar,
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
                moe,
                post_feedforward_layernorm_1: post1,
                post_feedforward_layernorm_2: post2,
            });
        }
        let norm = RmsNorm::new(weights.tensor("norm.weight")?, config.rms_norm_eps);
        let lm_head = TiedHead::flattening(Arc::clone(&embed_tokens));
        Ok(Gemma4Text {
            config,
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn num_parameters(&self) -> usize {
        let mut registry = ParameterRegistry::new();
        self.visit_parameters(&ModulePath::root(), &mut registry)
            .expect("parameter traversal");
        registry.specs().iter().map(|spec| spec.numel()).sum()
    }

    fn compute_per_layer_inputs(
        &self,
        cx: &mut ModelContext,
        input_ids: ValueId,
        inputs_embeds: ValueId,
        length: usize,
    ) -> Result<Option<Vec<ValueId>>> {
        let (Some(embed), Some(projection), Some(norm)) = (
            &self.embed_tokens_per_layer,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) else {
            return Ok(None);
        };
        let layers = self.config.num_hidden_layers;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let hidden = self.config.hidden_size;

        let token = module_forward(cx, "embed_tokens_per_layer", embed, input_ids)?;
        let scale = constant_scalar(cx, (ple_dim as f32).sqrt());
        let token = cx.builder().mul(token, scale)?;
        let token = cx
            .builder()
            .reshape(token, Shape::from_static([length, layers, ple_dim]))?;

        let context = module_forward(cx, "per_layer_model_projection", projection, inputs_embeds)?;
        let scale = constant_scalar(cx, 1.0 / (hidden as f32).sqrt());
        let context = cx.builder().mul(context, scale)?;
        let context = cx
            .builder()
            .reshape(context, Shape::from_static([length, layers, ple_dim]))?;
        let context = module_forward(cx, "per_layer_projection_norm", norm, context)?;

        let combined = cx.builder().add(context, token)?;
        let scale = constant_scalar(cx, 2.0f32.sqrt().recip());
        let combined = cx.builder().mul(combined, scale)?;
        let combined = cx
            .builder()
            .reshape(combined, Shape::from_static([1, length, layers, ple_dim]))?;
        let mut per_layer = Vec::with_capacity(layers);
        for index in 0..layers {
            let slice = cx.builder().slice(combined, 2, index, 1)?;
            let slice = cx
                .builder()
                .reshape(slice, Shape::from_static([1, length, ple_dim]))?;
            per_layer.push(slice);
        }
        Ok(Some(per_layer))
    }
}

impl Gemma4Text {
    /// Token embedding with the Gemma hidden-size scaling.
    #[model(predict = false)]
    fn embed_scale(&self, ids: oxide_ml::Tensor<f32>) -> oxide_ml::Tensor<f32> {
        let scale = self.config.hidden_size as f32;
        self.embed_tokens.forward(ids) * scale.sqrt()
    }

    /// Final logit soft-capping (`tanh(logits / cap) * cap`).
    #[model(predict = false)]
    fn softcap(&self, logits: oxide_ml::Tensor<f32>) -> oxide_ml::Tensor<f32> {
        if let Some(cap) = self.config.final_logit_softcapping {
            let scaled = logits / cap;
            scaled.tanh() * cap
        } else {
            logits
        }
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        if inputs.len() != 1 {
            return Err(Error::invalid_graph(format!(
                "Gemma4Text expects one token input, got {}",
                inputs.len()
            )));
        }
        let input = inputs[0];
        let shape = cx.value_shape(input)?;
        if shape.rank() != 1 {
            return Err(Error::RankMismatch {
                context: "Gemma4 token input".into(),
                expected: 1,
                got: shape.rank(),
            });
        }
        let length = shape
            .dim(0)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("Gemma4", "sequence length must be static"))?;
        let hidden = self.config.hidden_size;
        let scaled = single_output(self.embed_scale(cx, input)?)?;
        let per_layer = self.compute_per_layer_inputs(cx, input, scaled, length)?;
        let mut hidden_states = cx
            .builder()
            .reshape(scaled, Shape::from_static([1, length, hidden]))?;
        let mut sliding_shared: Option<(ValueId, ValueId)> = None;
        let mut full_shared: Option<(ValueId, ValueId)> = None;
        for (index, layer) in self.layers.iter().enumerate() {
            let shared = match layer.self_attn.layer.kind {
                LayerKind::Sliding => sliding_shared,
                LayerKind::Full => full_shared,
            };
            let per_layer_input = per_layer.as_ref().map(|values| values[index]);
            let outputs = module_forward_slots(
                cx,
                &format!("layers.{index}"),
                layer,
                &[
                    Some(hidden_states),
                    per_layer_input,
                    shared.map(|(keys, _)| keys),
                    shared.map(|(_, values)| values),
                ],
            )?;
            hidden_states = outputs[0];
            if !layer.self_attn.is_shared && layer.self_attn.store_kv {
                let stored = (outputs[1], outputs[2]);
                match layer.self_attn.layer.kind {
                    LayerKind::Sliding => sliding_shared = Some(stored),
                    LayerKind::Full => full_shared = Some(stored),
                }
            }
        }
        let hidden_states = module_forward(cx, "norm", &self.norm, hidden_states)?;
        let logits = module_forward(cx, "lm_head", &self.lm_head, hidden_states)?;
        Ok(vec![single_output(self.softcap(cx, logits)?)?])
    }

    /// KV layout declaration for runtimes that manage execution state.
    pub fn state_config(&self) -> Result<Option<oxide_ml_model::KvCacheConfig>> {
        Ok(Some(crate::streaming::kv_cache_config(&self.config, 0)?))
    }
}

/// Updated K/V per layer from one decode step.
pub type DecodedKv = Vec<Option<(ValueId, ValueId)>>;

/// Module wrapper for the cached decode graph.
pub struct GemmaDecode<'a> {
    pub model: &'a Gemma4Text,
    pub position: usize,
    pub past_len: usize,
}

impl Module for GemmaDecode<'_> {
    fn module_type(&self) -> &'static str {
        "Gemma4Decode"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.model.visit_parameters(path, registry)
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        self.forward_multi(cx, &[input])
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        let config = &self.model.config;
        let mut past = vec![None; config.num_hidden_layers];
        let mut cursor = 1usize;
        for (index, slot) in past.iter_mut().enumerate() {
            if !config.layer(index)?.is_shared {
                let k = *inputs.get(cursor).ok_or_else(|| {
                    Error::invalid_graph("decode graph is missing a cached K input")
                })?;
                let v = *inputs.get(cursor + 1).ok_or_else(|| {
                    Error::invalid_graph("decode graph is missing a cached V input")
                })?;
                *slot = Some((k, v));
                cursor += 2;
            }
        }
        let (logits, updated) = self
            .model
            .decode_graph(cx, inputs[0], &past, self.position)?;
        let mut outputs = vec![logits];
        for (index, slot) in updated.iter().enumerate() {
            if config.layer(index)?.is_shared {
                continue;
            }
            if let Some((k, v)) = slot {
                outputs.push(*k);
                outputs.push(*v);
            }
        }
        Ok(outputs)
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.model.visit_tensors(path, visitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeKind;
    use oxide_ml_nn::causal_mask;

    fn sliding(dim: usize) -> LayerConfig {
        LayerConfig {
            kind: LayerKind::Sliding,
            head_dim: dim,
            num_heads: 2,
            num_kv_heads: 1,
            rope: RopeKind::Default { theta: 10_000.0 },
            sliding_window: 8,
            is_shared: false,
        }
    }

    fn full(dim: usize) -> LayerConfig {
        LayerConfig {
            kind: LayerKind::Full,
            head_dim: dim,
            num_heads: 2,
            num_kv_heads: 1,
            rope: RopeKind::Proportional {
                theta: 1_000_000.0,
                partial_rotary_factor: 0.25,
            },
            sliding_window: 8,
            is_shared: false,
        }
    }

    #[test]
    fn rope_tables_have_expected_shape_and_first_position() {
        let (cos, sin) = rope_config(&sliding(32)).tables(0, 3);
        assert_eq!(cos.len(), 3 * 32);
        assert_eq!(sin.len(), 3 * 32);
        assert!(cos[..32].iter().all(|value| (*value - 1.0).abs() < 1e-6));
        assert!(sin[..32].iter().all(|value| value.abs() < 1e-6));
        assert_eq!(&cos[..16], &cos[16..32]);
    }

    #[test]
    fn proportional_rope_leaves_unrotated_tail_identity() {
        let (cos, sin) = rope_config(&full(64)).tables(0, 2);
        for position in 0..2 {
            for index in 8..32 {
                assert!((cos[position * 64 + index] - 1.0).abs() < 1e-6);
                assert!(sin[position * 64 + index].abs() < 1e-6);
                assert!((cos[position * 64 + 32 + index] - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn causal_mask_blocks_future_tokens_and_sliding_window() {
        let mask = causal_mask(4, None);
        assert_eq!(mask[0], 0.0);
        assert!(mask[1] < -1e20, "future token must be masked");
        assert_eq!(mask[5], 0.0);
        let sliding = causal_mask(4, Some(2));
        assert_eq!(sliding[2 * 4 + 1], 0.0);
        assert!(sliding[8] < -1e20, "outside the window must be masked");
    }

    #[test]
    fn rotate_half_negates_the_second_half() {
        let mut builder = oxide_ml_graph::GraphBuilder::new();
        let x = builder.input(
            "x",
            oxide_ml_core::Shape::from_static([1, 1, 1, 4]),
            oxide_ml_core::DType::F32,
        );
        let rotated = oxide_ml_nn::rotate_half_value(&mut builder, x, 4, 3).unwrap();
        let graph = builder.finish(&[rotated]).unwrap();
        let mut inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
        inputs.insert(
            x,
            oxide_ml_core::DenseTensor::from_vec([1, 1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        let result = oxide_ml_cpu::Executor::<f32>::new(graph)
            .unwrap()
            .run(&oxide_ml_core::ParameterStore::new(), &inputs)
            .unwrap();
        assert_eq!(result.output(0).unwrap().data(), &[-3.0, -4.0, 1.0, 2.0]);
    }
}
