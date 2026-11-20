//! Gemma 4 text decoder implemented with oxide-ml graph modules.
//!
//! Supports the dense text architecture, per-layer input embeddings (PLE),
//! shared KV layers and the double-wide MLP used by Gemma 4 E2B.

use std::sync::Arc;

use oxide_ml::{Module, model};
use oxide_ml_core::{DenseTensor, Dim, Error, Result, Shape, ValueId};
use oxide_ml_graph::ConstantTensor;
use oxide_ml_model::{
    ModelContext, ModuleList, ModulePath, ParameterRegistry, ParameterRole, Tensor, module_forward,
    single_output,
};
use oxide_ml_nn::modules::{Embedding, Linear};

use crate::config::{Gemma4Config, LayerConfig, LayerKind};
use crate::weights::Gemma4Weights;

fn constant_scalar(cx: &mut ModelContext, value: f32) -> ValueId {
    cx.builder().constant_scalar(value)
}

/// RMSNorm over the last axis, using `#[derive(Module)]` for parameter discovery.
#[derive(Clone, Debug, Module)]
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

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        let input = *inputs
            .first()
            .ok_or_else(|| Error::invalid_graph("Gemma4 RMSNorm needs one input"))?;
        let weight = cx.parameter("weight")?;
        Ok(vec![normalize(cx, input, self.eps, Some(weight))?])
    }
}

/// RMSNorm without a learned scale (Gemma 4 value-norm).
#[derive(Clone, Debug)]
pub struct NoScaleRmsNorm {
    pub eps: f32,
}

impl Module for NoScaleRmsNorm {
    fn module_type(&self) -> &'static str {
        "Gemma4RMSNormNoScale"
    }

    fn visit_parameters(
        &self,
        _path: &ModulePath,
        _registry: &mut ParameterRegistry,
    ) -> Result<()> {
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![normalize(cx, input, self.eps, None)?])
    }

    fn visit_tensors(
        &self,
        _path: &ModulePath,
        _visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }
}

fn normalize(
    cx: &mut ModelContext,
    x: ValueId,
    eps: f32,
    weight: Option<ValueId>,
) -> Result<ValueId> {
    let shape = cx.value_shape(x)?;
    let rank = shape.rank();
    if rank == 0 {
        return Err(Error::invalid_shape("RMSNorm", "input must have rank >= 1"));
    }
    let axes = vec![rank - 1];
    let square = cx.builder().mul(x, x)?;
    let mean = cx.builder().reduce_mean(square, axes, true)?;
    let eps = constant_scalar(cx, eps);
    let mean_eps = cx.builder().add(mean, eps)?;
    let std = cx.builder().sqrt(mean_eps)?;
    let normed = cx.builder().div(x, std)?;
    match weight {
        Some(weight) => cx.builder().mul(normed, weight),
        None => Ok(normed),
    }
}

fn rotate_half(cx: &mut ModelContext, x: ValueId, head_dim: usize) -> Result<ValueId> {
    let half = head_dim / 2;
    let rank = cx.value_shape(x)?.rank();
    let axis = rank - 1;
    let first = cx.builder().slice(x, axis, 0, half)?;
    let second = cx.builder().slice(x, axis, half, half)?;
    let neg_second = cx.builder().neg(second)?;
    cx.builder().concat(&[neg_second, first], axis)
}

fn rope_tables_at(layer: &LayerConfig, start: usize, length: usize) -> (Vec<f32>, Vec<f32>) {
    let dim = layer.head_dim;
    let half = dim / 2;
    let inverse = layer.rope.inverse_frequencies(dim);
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

fn rope_apply(
    cx: &mut ModelContext,
    x: ValueId,
    length: usize,
    head_dim: usize,
    layer: &LayerConfig,
) -> Result<ValueId> {
    rope_apply_at(cx, x, 0, length, head_dim, layer)
}

fn rope_apply_at(
    cx: &mut ModelContext,
    x: ValueId,
    start: usize,
    length: usize,
    head_dim: usize,
    layer: &LayerConfig,
) -> Result<ValueId> {
    let (cos, sin) = rope_tables_at(layer, start, length);
    let cos = cx.builder().constant_tensor(ConstantTensor::f32(
        Shape::from_static([length, head_dim]),
        cos,
    )?)?;
    let sin = cx.builder().constant_tensor(ConstantTensor::f32(
        Shape::from_static([length, head_dim]),
        sin,
    )?)?;
    let table_shape = Shape::new(vec![
        Dim::Static(1),
        Dim::Static(length),
        Dim::Static(1),
        Dim::Static(head_dim),
    ]);
    let cos = cx.builder().reshape(cos, table_shape.clone())?;
    let sin = cx.builder().reshape(sin, table_shape)?;
    let scaled = cx.builder().mul(x, cos)?;
    let rotated = rotate_half(cx, x, head_dim)?;
    let rotated = cx.builder().mul(rotated, sin)?;
    cx.builder().add(scaled, rotated)
}

fn causal_mask(length: usize, sliding: Option<usize>) -> Vec<f32> {
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

/// Gemma 4 attention (GQA, per-layer RoPE, optional shared K/V).
#[derive(Clone, Debug)]
pub struct GemmaAttention {
    pub q_proj: Linear,
    pub k_proj: Option<Linear>,
    pub v_proj: Option<Linear>,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: Option<RmsNorm>,
    pub v_norm: Option<NoScaleRmsNorm>,
    pub layer: LayerConfig,
    pub eps: f32,
    pub store_kv: bool,
}

impl GemmaAttention {
    /// Single-head causal attention, expressed with the `#[model]` macro.
    #[model]
    fn head_attention(
        &self,
        q: Tensor<f32>,
        k: Tensor<f32>,
        v: Tensor<f32>,
        mask: Tensor<f32>,
    ) -> Tensor<f32> {
        let scores = q.matmul_t(k, false, true) + mask;
        let weights = scores.softmax(-1);
        weights.matmul(v)
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
        let heads = self.layer.num_heads;
        let kv_heads = self.layer.num_kv_heads;
        let head_dim = self.layer.head_dim;
        let groups = heads / kv_heads;

        let q = module_forward(cx, "q_proj", &self.q_proj, x)?;
        let q = cx
            .builder()
            .reshape(q, Shape::from_static([1, 1, heads, head_dim]))?;
        let q = module_forward(cx, "q_norm", &self.q_norm, q)?;
        let q = rope_apply_at(cx, q, position, 1, head_dim, &self.layer)?;

        let (k, v, stored) = if self.layer.is_shared {
            let (k, v) = past.ok_or_else(|| {
                Error::invalid_graph(format!(
                    "shared KV layer {} has no cached state",
                    self.layer.kind.name()
                ))
            })?;
            (k, v, None)
        } else {
            let k = module_forward(
                cx,
                "k_proj",
                self.k_proj
                    .as_ref()
                    .ok_or_else(|| Error::invalid_graph("missing k_proj"))?,
                x,
            )?;
            let v = module_forward(
                cx,
                "v_proj",
                self.v_proj
                    .as_ref()
                    .ok_or_else(|| Error::invalid_graph("missing v_proj"))?,
                x,
            )?;
            let k = cx
                .builder()
                .reshape(k, Shape::from_static([1, 1, kv_heads, head_dim]))?;
            let v = cx
                .builder()
                .reshape(v, Shape::from_static([1, 1, kv_heads, head_dim]))?;
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
            let k = rope_apply_at(cx, k, position, 1, head_dim, &self.layer)?;
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
        let sliding = (self.layer.kind == LayerKind::Sliding).then_some(self.layer.sliding_window);
        let mut mask = vec![0.0f32; kv_len];
        if let Some(window) = sliding {
            for (key, value) in mask.iter_mut().enumerate() {
                if position.saturating_sub(key) >= window {
                    *value = -1.0e30;
                }
            }
        }
        let mask = cx
            .builder()
            .constant_tensor(ConstantTensor::f32(Shape::from_static([1, kv_len]), mask)?)?;

        let mut heads_out = Vec::with_capacity(heads);
        for head in 0..heads {
            let kv_head = head / groups;
            let qh = cx.builder().slice(q, 2, head, 1)?;
            let qh = cx
                .builder()
                .reshape(qh, Shape::from_static([1, head_dim]))?;
            let kh = cx.builder().slice(k, 2, kv_head, 1)?;
            let kh = cx
                .builder()
                .reshape(kh, Shape::from_static([kv_len, head_dim]))?;
            let vh = cx.builder().slice(v, 2, kv_head, 1)?;
            let vh = cx
                .builder()
                .reshape(vh, Shape::from_static([kv_len, head_dim]))?;
            let scores = cx.builder().matmul_with(qh, kh, false, true)?;
            let scores = cx.builder().add(scores, mask)?;
            let weights = cx.builder().softmax(scores, -1)?;
            let out = cx.builder().matmul(weights, vh)?;
            let out = cx
                .builder()
                .reshape(out, Shape::from_static([1, 1, head_dim]))?;
            heads_out.push(out);
        }
        let merged = cx.builder().concat(&heads_out, 2)?;
        let merged = cx
            .builder()
            .reshape(merged, Shape::from_static([1, 1, heads * head_dim]))?;
        Ok((module_forward(cx, "o_proj", &self.o_proj, merged)?, stored))
    }

    fn apply(
        &self,
        cx: &mut ModelContext,
        x: ValueId,
        shared: Option<(ValueId, ValueId)>,
    ) -> Result<(ValueId, Option<(ValueId, ValueId)>)> {
        let shape = cx.value_shape(x)?;
        if shape.rank() != 3 {
            return Err(Error::RankMismatch {
                context: "Gemma4 attention".into(),
                expected: 3,
                got: shape.rank(),
            });
        }
        let batch = shape.dim(0).unwrap_or(Dim::Static(1));
        let length = shape
            .dim(1)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("Gemma4 attention", "sequence must be static"))?;
        let heads = self.layer.num_heads;
        let kv_heads = self.layer.num_kv_heads;
        let head_dim = self.layer.head_dim;
        let groups = heads / kv_heads;

        let q = module_forward(cx, "q_proj", &self.q_proj, x)?;
        let q = cx.builder().reshape(
            q,
            Shape::new(vec![
                batch,
                Dim::Static(length),
                Dim::Static(heads),
                Dim::Static(head_dim),
            ]),
        )?;
        let q = module_forward(cx, "q_norm", &self.q_norm, q)?;
        let q = rope_apply(cx, q, length, head_dim, &self.layer)?;

        let (k, v, stored) = if self.layer.is_shared {
            let (k, v) = shared.ok_or_else(|| {
                Error::invalid_graph(format!(
                    "shared KV layer {} has no preceding K/V state",
                    self.layer.kind.name()
                ))
            })?;
            (k, v, None)
        } else {
            let k_proj = self
                .k_proj
                .as_ref()
                .ok_or_else(|| Error::invalid_graph("missing k_proj"))?;
            let k = module_forward(cx, "k_proj", k_proj, x)?;
            let v = match &self.v_proj {
                Some(projection) => module_forward(cx, "v_proj", projection, x)?,
                None => k,
            };
            let kv_shape = Shape::new(vec![
                batch,
                Dim::Static(length),
                Dim::Static(kv_heads),
                Dim::Static(head_dim),
            ]);
            let k = cx.builder().reshape(k, kv_shape.clone())?;
            let v = cx.builder().reshape(v, kv_shape)?;
            let k_norm = self
                .k_norm
                .as_ref()
                .ok_or_else(|| Error::invalid_graph("missing k_norm"))?;
            let v_norm = self
                .v_norm
                .as_ref()
                .ok_or_else(|| Error::invalid_graph("missing v_norm"))?;
            let k = module_forward(cx, "k_norm", k_norm, k)?;
            let v = module_forward(cx, "v_norm", v_norm, v)?;
            let k = rope_apply(cx, k, length, head_dim, &self.layer)?;
            let stored = self.store_kv.then_some((k, v));
            (k, v, stored)
        };

        let q = cx.builder().transpose(q, vec![0, 2, 1, 3])?;
        let k = cx.builder().transpose(k, vec![0, 2, 1, 3])?;
        let v = cx.builder().transpose(v, vec![0, 2, 1, 3])?;
        let q = cx
            .builder()
            .reshape(q, Shape::from_static([heads, length, head_dim]))?;
        let k = cx
            .builder()
            .reshape(k, Shape::from_static([kv_heads, length, head_dim]))?;
        let v = cx
            .builder()
            .reshape(v, Shape::from_static([kv_heads, length, head_dim]))?;

        let sliding = (self.layer.kind == LayerKind::Sliding).then_some(self.layer.sliding_window);
        let mask = cx.builder().constant_tensor(ConstantTensor::f32(
            Shape::from_static([length, length]),
            causal_mask(length, sliding),
        )?)?;

        let mut head_outputs = Vec::with_capacity(heads);
        for head in 0..heads {
            let kv_head = head / groups;
            let qh = cx.builder().slice(q, 0, head, 1)?;
            let kh = cx.builder().slice(k, 0, kv_head, 1)?;
            let vh = cx.builder().slice(v, 0, kv_head, 1)?;
            let qh = cx
                .builder()
                .reshape(qh, Shape::from_static([length, head_dim]))?;
            let kh = cx
                .builder()
                .reshape(kh, Shape::from_static([length, head_dim]))?;
            let vh = cx
                .builder()
                .reshape(vh, Shape::from_static([length, head_dim]))?;
            let out = single_output(self.head_attention_multi(cx, &[qh, kh, vh, mask])?)?;
            let out = cx
                .builder()
                .reshape(out, Shape::from_static([1, length, head_dim]))?;
            head_outputs.push(out);
        }
        let merged = cx.builder().concat(&head_outputs, 2)?;
        let merged = cx.builder().reshape(
            merged,
            Shape::new(vec![
                batch,
                Dim::Static(length),
                Dim::Static(heads * head_dim),
            ]),
        )?;
        let out = module_forward(cx, "o_proj", &self.o_proj, merged)?;
        Ok((out, stored))
    }
}

impl Module for GemmaAttention {
    fn module_type(&self) -> &'static str {
        "Gemma4TextAttention"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.q_proj
            .visit_parameters(&path.child("q_proj"), registry)?;
        if let Some(k) = &self.k_proj {
            k.visit_parameters(&path.child("k_proj"), registry)?;
        }
        if let Some(v) = &self.v_proj {
            v.visit_parameters(&path.child("v_proj"), registry)?;
        }
        self.o_proj
            .visit_parameters(&path.child("o_proj"), registry)?;
        self.q_norm
            .visit_parameters(&path.child("q_norm"), registry)?;
        if let Some(k) = &self.k_norm {
            k.visit_parameters(&path.child("k_norm"), registry)?;
        }
        if let Some(v) = &self.v_norm {
            v.visit_parameters(&path.child("v_norm"), registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let (out, _) = self.apply(cx, input, None)?;
        Ok(vec![out])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.q_proj.visit_tensors(&path.child("q_proj"), visitor)?;
        if let Some(k) = &self.k_proj {
            k.visit_tensors(&path.child("k_proj"), visitor)?;
        }
        if let Some(v) = &self.v_proj {
            v.visit_tensors(&path.child("v_proj"), visitor)?;
        }
        self.o_proj.visit_tensors(&path.child("o_proj"), visitor)?;
        self.q_norm.visit_tensors(&path.child("q_norm"), visitor)?;
        if let Some(k) = &self.k_norm {
            k.visit_tensors(&path.child("k_norm"), visitor)?;
        }
        if let Some(v) = &self.v_norm {
            v.visit_tensors(&path.child("v_norm"), visitor)?;
        }
        Ok(())
    }
}

/// GeGLU feed-forward block (double-wide on shared-KV layers).
#[derive(Clone, Debug, Module)]
pub struct GemmaMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl GemmaMlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let gate = self.gate_proj.forward(x).gelu();
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Fused Gemma 4 mixture-of-experts block.
#[derive(Clone, Debug)]
pub struct GemmaMoE {
    pub router_scale: DenseTensor<f32>,
    pub per_expert_scale: DenseTensor<f32>,
    pub router_proj: DenseTensor<f32>,
    pub gate_up: DenseTensor<f32>,
    pub down: DenseTensor<f32>,
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
        let expert_norm = cx.parameter("pre_feedforward_layernorm_2.weight")?;
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
        Ok(())
    }
}

/// One Gemma 4 decoder layer.
#[derive(Clone, Debug)]
pub struct GemmaLayer {
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub pre_feedforward_layernorm: RmsNorm,
    pub post_feedforward_layernorm: RmsNorm,
    pub self_attn: GemmaAttention,
    pub mlp: GemmaMlp,
    pub layer_scalar: DenseTensor<f32>,
    pub per_layer_input_gate: Option<Linear>,
    pub per_layer_projection: Option<Linear>,
    pub post_per_layer_input_norm: Option<RmsNorm>,
    pub moe: Option<GemmaMoE>,
    pub pre_feedforward_layernorm_2: Option<RmsNorm>,
    pub post_feedforward_layernorm_1: Option<RmsNorm>,
    pub post_feedforward_layernorm_2: Option<RmsNorm>,
}

impl GemmaLayer {
    pub(crate) fn apply_with(
        &self,
        cx: &mut ModelContext,
        input: ValueId,
        per_layer_input: Option<ValueId>,
        shared: Option<(ValueId, ValueId)>,
    ) -> Result<(ValueId, Option<(ValueId, ValueId)>)> {
        let residual = input;
        let hidden = module_forward(cx, "input_layernorm", &self.input_layernorm, input)?;
        let depth = cx.depth();
        cx.enter("self_attn");
        let result = self.self_attn.apply(cx, hidden, shared);
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
            let moe_branch = moe.apply(cx, residual)?;
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
        let (pre2, post1, post2) = if config.enable_moe_block {
            (
                Some(RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps)),
                Some(RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps)),
                Some(RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps)),
            )
        } else {
            (None, None, None)
        };
        Ok(GemmaLayer {
            input_layernorm: RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps),
            post_attention_layernorm: RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps),
            pre_feedforward_layernorm: RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps),
            post_feedforward_layernorm: RmsNorm::new(zeros(&[hidden])?, config.rms_norm_eps),
            self_attn: GemmaAttention {
                q_proj: linear(heads * head_dim, hidden)?,
                k_proj,
                v_proj,
                o_proj: linear(hidden, heads * head_dim)?,
                q_norm: RmsNorm::new(zeros(&[head_dim])?, config.rms_norm_eps),
                k_norm,
                v_norm,
                layer: layer.clone(),
                eps: config.rms_norm_eps,
                store_kv,
            },
            mlp: GemmaMlp {
                gate_proj: linear(intermediate, hidden)?,
                up_proj: linear(intermediate, hidden)?,
                down_proj: linear(hidden, intermediate)?,
            },
            layer_scalar: zeros(&[1])?,
            per_layer_input_gate: gate,
            per_layer_projection: projection,
            post_per_layer_input_norm: norm,
            moe,
            pre_feedforward_layernorm_2: pre2,
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
            let moe_branch = moe.apply(cx, residual)?;
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

impl Module for GemmaLayer {
    fn module_type(&self) -> &'static str {
        "Gemma4TextDecoderLayer"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.input_layernorm
            .visit_parameters(&path.child("input_layernorm"), registry)?;
        self.post_attention_layernorm
            .visit_parameters(&path.child("post_attention_layernorm"), registry)?;
        self.pre_feedforward_layernorm
            .visit_parameters(&path.child("pre_feedforward_layernorm"), registry)?;
        self.post_feedforward_layernorm
            .visit_parameters(&path.child("post_feedforward_layernorm"), registry)?;
        self.self_attn
            .visit_parameters(&path.child("self_attn"), registry)?;
        self.mlp.visit_parameters(&path.child("mlp"), registry)?;
        if let Some(gate) = &self.per_layer_input_gate {
            gate.visit_parameters(&path.child("per_layer_input_gate"), registry)?;
        }
        if let Some(projection) = &self.per_layer_projection {
            projection.visit_parameters(&path.child("per_layer_projection"), registry)?;
        }
        if let Some(norm) = &self.post_per_layer_input_norm {
            norm.visit_parameters(&path.child("post_per_layer_input_norm"), registry)?;
        }
        if let Some(moe) = &self.moe {
            moe.visit_parameters(path, registry)?;
        }
        if let Some(norm) = &self.pre_feedforward_layernorm_2 {
            norm.visit_parameters(&path.child("pre_feedforward_layernorm_2"), registry)?;
        }
        if let Some(norm) = &self.post_feedforward_layernorm_1 {
            norm.visit_parameters(&path.child("post_feedforward_layernorm_1"), registry)?;
        }
        if let Some(norm) = &self.post_feedforward_layernorm_2 {
            norm.visit_parameters(&path.child("post_feedforward_layernorm_2"), registry)?;
        }
        registry.insert_tensor(
            &path.child("layer_scalar"),
            &self.layer_scalar,
            ParameterRole::Trainable,
        )?;
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let (out, _) = self.apply_with(cx, input, None, None)?;
        Ok(vec![out])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.input_layernorm
            .visit_tensors(&path.child("input_layernorm"), visitor)?;
        self.post_attention_layernorm
            .visit_tensors(&path.child("post_attention_layernorm"), visitor)?;
        self.pre_feedforward_layernorm
            .visit_tensors(&path.child("pre_feedforward_layernorm"), visitor)?;
        self.post_feedforward_layernorm
            .visit_tensors(&path.child("post_feedforward_layernorm"), visitor)?;
        self.self_attn
            .visit_tensors(&path.child("self_attn"), visitor)?;
        self.mlp.visit_tensors(&path.child("mlp"), visitor)?;
        if let Some(gate) = &self.per_layer_input_gate {
            gate.visit_tensors(&path.child("per_layer_input_gate"), visitor)?;
        }
        if let Some(projection) = &self.per_layer_projection {
            projection.visit_tensors(&path.child("per_layer_projection"), visitor)?;
        }
        if let Some(norm) = &self.post_per_layer_input_norm {
            norm.visit_tensors(&path.child("post_per_layer_input_norm"), visitor)?;
        }
        if let Some(moe) = &self.moe {
            moe.visit_tensors(path, visitor)?;
        }
        if let Some(norm) = &self.pre_feedforward_layernorm_2 {
            norm.visit_tensors(&path.child("pre_feedforward_layernorm_2"), visitor)?;
        }
        if let Some(norm) = &self.post_feedforward_layernorm_1 {
            norm.visit_tensors(&path.child("post_feedforward_layernorm_1"), visitor)?;
        }
        if let Some(norm) = &self.post_feedforward_layernorm_2 {
            norm.visit_tensors(&path.child("post_feedforward_layernorm_2"), visitor)?;
        }
        visitor(
            &path.child("layer_scalar"),
            &self.layer_scalar,
            ParameterRole::Trainable,
        )?;
        Ok(())
    }
}

/// Language-model head tied to the token embedding.
///
/// This is the general-purpose [`oxide_ml_nn::TiedLinear`] with leading
/// dimensions flattened ([1, sequence, hidden] -> [sequence, hidden]). The
/// registry records `lm_head.weight` as an alias of `embed_tokens.weight`.
pub type TiedHead = oxide_ml_nn::TiedLinear<Embedding>;

/// Complete Gemma 4 text model.
#[derive(Clone, Debug)]
pub struct Gemma4Text {
    pub config: Gemma4Config,
    pub embed_tokens: Arc<Embedding>,
    pub embed_tokens_per_layer: Option<Embedding>,
    pub per_layer_model_projection: Option<Linear>,
    pub per_layer_projection_norm: Option<RmsNorm>,
    pub layers: ModuleList<GemmaLayer>,
    pub norm: RmsNorm,
    pub lm_head: TiedHead,
    training: bool,
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
    ) -> Result<(ValueId, Vec<Option<(ValueId, ValueId)>>)> {
        let config = &self.config;
        let hidden_size = config.hidden_size;
        let embed = module_forward(cx, "embed_tokens", self.embed_tokens.as_ref(), token)?;
        let scale = constant_scalar(cx, (hidden_size as f32).sqrt());
        let scaled = cx.builder().mul(embed, scale)?;
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
                layer,
                eps: config.rms_norm_eps,
                store_kv: store,
            };
            let mlp = GemmaMlp {
                gate_proj: Linear::from_tensors(
                    weights.tensor(&format!("{prefix}.mlp.gate_proj.weight"))?,
                    None,
                )?,
                up_proj: Linear::from_tensors(
                    weights.tensor(&format!("{prefix}.mlp.up_proj.weight"))?,
                    None,
                )?,
                down_proj: Linear::from_tensors(
                    weights.tensor(&format!("{prefix}.mlp.down_proj.weight"))?,
                    None,
                )?,
            };
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
            let (moe, pre2, post1, post2) = if config.enable_moe_block {
                (
                    Some(GemmaMoE::from_weights(weights, &prefix, &config)?),
                    Some(RmsNorm::new(
                        weights.tensor(&format!("{prefix}.pre_feedforward_layernorm_2.weight"))?,
                        config.rms_norm_eps,
                    )),
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
                (None, None, None, None)
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
                pre_feedforward_layernorm_2: pre2,
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
            training: false,
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

impl Module for Gemma4Text {
    fn module_type(&self) -> &'static str {
        "Gemma4TextModel"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.embed_tokens
            .visit_parameters(&path.child("embed_tokens"), registry)?;
        if let Some(embed) = &self.embed_tokens_per_layer {
            embed.visit_parameters(&path.child("embed_tokens_per_layer"), registry)?;
        }
        if let Some(projection) = &self.per_layer_model_projection {
            projection.visit_parameters(&path.child("per_layer_model_projection"), registry)?;
        }
        if let Some(norm) = &self.per_layer_projection_norm {
            norm.visit_parameters(&path.child("per_layer_projection_norm"), registry)?;
        }
        let layers = path.child("layers");
        for (index, layer) in self.layers.iter().enumerate() {
            layer.visit_parameters(&layers.child(index.to_string()), registry)?;
        }
        self.norm.visit_parameters(&path.child("norm"), registry)?;
        self.lm_head
            .visit_parameters(&path.child("lm_head"), registry)?;
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
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
        let embed = module_forward(cx, "embed_tokens", self.embed_tokens.as_ref(), input)?;
        let scale = constant_scalar(cx, (hidden as f32).sqrt());
        let scaled = cx.builder().mul(embed, scale)?;
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
            let depth = cx.depth();
            for segment in format!("layers.{index}").split('.') {
                if !segment.is_empty() {
                    cx.enter(segment);
                }
            }
            let result = layer.apply_with(cx, hidden_states, per_layer_input, shared);
            cx.truncate(depth);
            let (out, stored) = result?;
            if let Some(stored) = stored {
                match layer.self_attn.layer.kind {
                    LayerKind::Sliding => sliding_shared = Some(stored),
                    LayerKind::Full => full_shared = Some(stored),
                }
            }
            hidden_states = out;
        }
        let hidden_states = module_forward(cx, "norm", &self.norm, hidden_states)?;
        let mut logits = module_forward(cx, "lm_head", &self.lm_head, hidden_states)?;
        if let Some(cap) = self.config.final_logit_softcapping {
            let cap_value = constant_scalar(cx, cap);
            let scaled = cx.builder().div(logits, cap_value)?;
            let bounded = cx.builder().tanh(scaled)?;
            logits = cx.builder().mul(bounded, cap_value)?;
        }
        Ok(vec![logits])
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_config(&self) -> Result<Option<oxide_ml_model::KvCacheConfig>> {
        // One slot per non-shared layer; shared-KV layers resolve to the last
        // non-shared layer of the same attention kind.
        Ok(Some(crate::streaming::kv_cache_config(&self.config, 0)?))
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.embed_tokens
            .visit_tensors(&path.child("embed_tokens"), visitor)?;
        if let Some(embed) = &self.embed_tokens_per_layer {
            embed.visit_tensors(&path.child("embed_tokens_per_layer"), visitor)?;
        }
        if let Some(projection) = &self.per_layer_model_projection {
            projection.visit_tensors(&path.child("per_layer_model_projection"), visitor)?;
        }
        if let Some(norm) = &self.per_layer_projection_norm {
            norm.visit_tensors(&path.child("per_layer_projection_norm"), visitor)?;
        }
        let layers = path.child("layers");
        for (index, layer) in self.layers.iter().enumerate() {
            layer.visit_tensors(&layers.child(index.to_string()), visitor)?;
        }
        self.norm.visit_tensors(&path.child("norm"), visitor)?;
        self.lm_head
            .visit_tensors(&path.child("lm_head"), visitor)?;
        Ok(())
    }
}

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
        let (cos, sin) = rope_tables_at(&sliding(32), 0, 3);
        assert_eq!(cos.len(), 3 * 32);
        assert_eq!(sin.len(), 3 * 32);
        assert!(cos[..32].iter().all(|value| (*value - 1.0).abs() < 1e-6));
        assert!(sin[..32].iter().all(|value| value.abs() < 1e-6));
        assert_eq!(&cos[..16], &cos[16..32]);
    }

    #[test]
    fn proportional_rope_leaves_unrotated_tail_identity() {
        let (cos, sin) = rope_tables_at(&full(64), 0, 2);
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
        let rotated = rotate_half_value(&mut builder, x, 4).unwrap();
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

    fn rotate_half_value(
        builder: &mut oxide_ml_graph::GraphBuilder,
        x: ValueId,
        head_dim: usize,
    ) -> Result<ValueId> {
        let half = head_dim / 2;
        let first = builder.slice(x, 3, 0, half)?;
        let second = builder.slice(x, 3, half, half)?;
        let neg_second = builder.neg(second)?;
        builder.concat(&[neg_second, first], 3)
    }
}
