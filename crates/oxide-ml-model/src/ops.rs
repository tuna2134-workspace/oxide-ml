//! Tensor operations usable inside `#[model]` forward functions.
//!
//! The macros rewrite `relu(x)`, `x.relu()`, `x + y` and similar syntax into
//! calls to this module. Every function appends a node to the graph owned by
//! [`ModelContext`], validates shapes through the graph builder, and returns a
//! graph value. Scalars are accepted wherever a tensor operand is expected and
//! become graph constants.

use oxide_ml_core::{DType, Dim, Error, Result, Shape, ValueId};
use oxide_ml_graph::ConstantTensor;

use crate::context::ModelContext;

/// Anything that can be used as a tensor operand in a model body.
pub trait Operand {
    fn into_operand(self, cx: &mut ModelContext) -> Result<ValueId>;
}

impl Operand for ValueId {
    fn into_operand(self, _cx: &mut ModelContext) -> Result<ValueId> {
        Ok(self)
    }
}

impl Operand for &ValueId {
    fn into_operand(self, cx: &mut ModelContext) -> Result<ValueId> {
        (*self).into_operand(cx)
    }
}

macro_rules! operand_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Operand for $ty {
                fn into_operand(self, cx: &mut ModelContext) -> Result<ValueId> {
                    cx.constant_scalar_as(self as f32, cx.dtype())
                }
            }
        )*
    };
}

operand_for_scalar!(f32, f64, i32, i64, u32, usize);

impl Operand for bool {
    fn into_operand(self, cx: &mut ModelContext) -> Result<ValueId> {
        cx.constant_scalar_as(if self { 1.0 } else { 0.0 }, cx.dtype())
    }
}

fn binary(
    cx: &mut ModelContext,
    op: oxide_ml_graph::Op,
    lhs: impl Operand,
    rhs: impl Operand,
) -> Result<ValueId> {
    let lhs = lhs.into_operand(cx)?;
    let rhs = rhs.into_operand(cx)?;
    cx.count_operation();
    cx.builder().binary(op, lhs, rhs)
}

fn unary(
    cx: &mut ModelContext,
    op: oxide_ml_graph::Op,
    value: impl Operand,
    attrs: oxide_ml_graph::OpAttrs,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    cx.count_operation();
    cx.builder().single(op, attrs, &[value])
}

pub fn add(cx: &mut ModelContext, lhs: impl Operand, rhs: impl Operand) -> Result<ValueId> {
    binary(cx, oxide_ml_graph::Op::Add, lhs, rhs)
}

pub fn sub(cx: &mut ModelContext, lhs: impl Operand, rhs: impl Operand) -> Result<ValueId> {
    binary(cx, oxide_ml_graph::Op::Sub, lhs, rhs)
}

pub fn mul(cx: &mut ModelContext, lhs: impl Operand, rhs: impl Operand) -> Result<ValueId> {
    binary(cx, oxide_ml_graph::Op::Mul, lhs, rhs)
}

pub fn div(cx: &mut ModelContext, lhs: impl Operand, rhs: impl Operand) -> Result<ValueId> {
    binary(cx, oxide_ml_graph::Op::Div, lhs, rhs)
}

pub fn neg(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::Neg,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn relu(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::ReLU,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn gelu(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::GELU,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn sigmoid(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::Sigmoid,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn tanh(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::Tanh,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn exp(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::Exp,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn log(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::Log,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn sqrt(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::Sqrt,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn abs(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::Abs,
        value,
        oxide_ml_graph::OpAttrs::None,
    )
}

pub fn softmax(cx: &mut ModelContext, value: impl Operand, axis: i32) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::Softmax,
        value,
        oxide_ml_graph::OpAttrs::Softmax { axis },
    )
}

pub fn log_softmax(cx: &mut ModelContext, value: impl Operand, axis: i32) -> Result<ValueId> {
    unary(
        cx,
        oxide_ml_graph::Op::LogSoftmax,
        value,
        oxide_ml_graph::OpAttrs::LogSoftmax { axis },
    )
}

pub fn matmul(cx: &mut ModelContext, lhs: impl Operand, rhs: impl Operand) -> Result<ValueId> {
    let lhs = lhs.into_operand(cx)?;
    let rhs = rhs.into_operand(cx)?;
    cx.count_operation();
    cx.builder().matmul(lhs, rhs)
}

/// Matrix multiplication with explicit transpose flags.
pub fn matmul_t(
    cx: &mut ModelContext,
    lhs: impl Operand,
    rhs: impl Operand,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<ValueId> {
    let lhs = lhs.into_operand(cx)?;
    let rhs = rhs.into_operand(cx)?;
    cx.count_operation();
    cx.builder().matmul_with(lhs, rhs, transpose_a, transpose_b)
}

pub fn reshape(
    cx: &mut ModelContext,
    value: impl Operand,
    shape: impl AsRef<[usize]>,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    cx.count_operation();
    cx.builder().reshape_dims(value, shape)
}

pub fn transpose(
    cx: &mut ModelContext,
    value: impl Operand,
    axes: impl AsRef<[usize]>,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    cx.count_operation();
    cx.builder().transpose(value, axes.as_ref().to_vec())
}

pub fn transpose_last(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    cx.count_operation();
    cx.builder().transpose_last(value)
}

pub fn broadcast_to(
    cx: &mut ModelContext,
    value: impl Operand,
    shape: impl AsRef<[usize]>,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    cx.count_operation();
    cx.builder().broadcast_to(value, shape.as_ref())
}

pub fn mean(
    cx: &mut ModelContext,
    value: impl Operand,
    axes: impl AsRef<[usize]>,
    keep_dims: bool,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    cx.count_operation();
    cx.builder()
        .reduce_mean(value, axes.as_ref().to_vec(), keep_dims)
}

pub fn sum(
    cx: &mut ModelContext,
    value: impl Operand,
    axes: impl AsRef<[usize]>,
    keep_dims: bool,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    cx.count_operation();
    cx.builder()
        .reduce_sum(value, axes.as_ref().to_vec(), keep_dims)
}

pub fn layer_norm(
    cx: &mut ModelContext,
    value: impl Operand,
    gamma: impl Operand,
    beta: impl Operand,
    normalized_shape: impl AsRef<[usize]>,
    eps: f32,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    let gamma = gamma.into_operand(cx)?;
    let beta = beta.into_operand(cx)?;
    cx.count_operation();
    cx.builder()
        .layer_norm(value, gamma, beta, normalized_shape.as_ref().to_vec(), eps)
}

pub fn mse_loss(
    cx: &mut ModelContext,
    prediction: impl Operand,
    target: impl Operand,
) -> Result<ValueId> {
    let prediction = prediction.into_operand(cx)?;
    let target = target.into_operand(cx)?;
    cx.count_operation();
    cx.builder()
        .mse_loss(prediction, target, oxide_ml_graph::Reduction::Mean)
}

pub fn cross_entropy_loss(
    cx: &mut ModelContext,
    logits: impl Operand,
    targets: impl Operand,
) -> Result<ValueId> {
    let logits = logits.into_operand(cx)?;
    let targets = targets.into_operand(cx)?;
    cx.count_operation();
    cx.builder()
        .cross_entropy_loss(logits, targets, oxide_ml_graph::Reduction::Mean)
}

/// Apply a registered custom kernel whose output shape equals its input shape.
///
/// This is the `#[model]`-friendly form of `GraphBuilder::custom`. Kernels that
/// change the shape (or take multiple inputs) should be wired through a
/// hand-written `Module::forward`.
pub fn apply_kernel(cx: &mut ModelContext, value: impl Operand, name: &str) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    let shape = cx.value_shape(value)?;
    cx.count_operation();
    let outputs = cx.builder().custom(name, &[value], vec![shape])?;
    crate::context::single_output(outputs)
}

/// Apply a registered custom kernel with an explicit static output shape.
pub fn apply_kernel_shaped(
    cx: &mut ModelContext,
    value: impl Operand,
    name: &str,
    output_shape: impl AsRef<[usize]>,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    cx.count_operation();
    let outputs = cx.builder().custom(
        name,
        &[value],
        vec![oxide_ml_core::Shape::from_static(output_shape.as_ref())],
    )?;
    crate::context::single_output(outputs)
}

/// Multiply by a compile-time constant.
pub fn scale(cx: &mut ModelContext, value: impl Operand, factor: f32) -> Result<ValueId> {
    mul(cx, value, factor)
}

/// Fail the graph build with a contextual error.
pub fn unsupported<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::unsupported(message))
}

/// Casting is currently a graph-level no-op; the model API is F32-only.
pub fn cast(cx: &mut ModelContext, value: impl Operand, dtype: DType) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    let actual = cx.value_dtype(value)?;
    if actual != dtype {
        return Err(Error::DTypeMismatch {
            context: "cast".into(),
            expected: dtype,
            got: actual,
        });
    }
    Ok(value)
}

/// Resolve a parameter by its path relative to the current module.
///
/// Useful in declarative code for parameters that are not tensors passed as
/// operands, for example a per-layer scalar:
///
/// ```ignore
/// let output = hidden * parameter("layer_scalar");
/// ```
pub fn parameter(cx: &mut ModelContext, name: &str) -> Result<ValueId> {
    cx.parameter(name)
}

/// Split the last dimension into `[heads, head_dim]`.
pub fn split_heads(
    cx: &mut ModelContext,
    value: impl Operand,
    heads: usize,
    head_dim: usize,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    let shape = cx.value_shape(value)?;
    let rank = shape.rank();
    if rank < 1 {
        return Err(Error::invalid_shape(
            "split_heads",
            "input must have rank >= 1",
        ));
    }
    let mut dims = shape.dims().to_vec();
    let last = dims.pop().expect("rank checked");
    if let Some(size) = last.static_usize()
        && size != heads * head_dim
    {
        return Err(Error::invalid_shape(
            "split_heads",
            format!("last dimension {size} is not {heads} * {head_dim}"),
        ));
    }
    dims.push(Dim::Static(heads));
    dims.push(Dim::Static(head_dim));
    cx.count_operation();
    cx.builder().reshape(value, Shape::new(dims))
}

/// Merge the last two dimensions (`[heads, head_dim] -> heads * head_dim`).
pub fn merge_heads(cx: &mut ModelContext, value: impl Operand) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    let shape = cx.value_shape(value)?;
    let rank = shape.rank();
    if rank < 2 {
        return Err(Error::invalid_shape(
            "merge_heads",
            "input must have rank >= 2",
        ));
    }
    let dims = shape.dims();
    let heads = dims[rank - 2]
        .static_usize()
        .ok_or_else(|| Error::invalid_shape("merge_heads", "head count must be static"))?;
    let head_dim = dims[rank - 1]
        .static_usize()
        .ok_or_else(|| Error::invalid_shape("merge_heads", "head dimension must be static"))?;
    let mut out = dims[..rank - 2].to_vec();
    out.push(Dim::Static(heads * head_dim));
    cx.count_operation();
    cx.builder().reshape(value, Shape::new(out))
}

/// Apply rotary position embeddings to `[1, length, heads, head_dim]`.
pub fn apply_rotary(
    cx: &mut ModelContext,
    value: impl Operand,
    head_dim: usize,
    theta: f32,
    partial_rotary_factor: Option<f32>,
    start: usize,
) -> Result<ValueId> {
    let value = value.into_operand(cx)?;
    let shape = cx.value_shape(value)?;
    if shape.rank() != 4 {
        return Err(Error::RankMismatch {
            context: "apply_rotary".into(),
            expected: 4,
            got: shape.rank(),
        });
    }
    let length = shape
        .dim(1)
        .and_then(Dim::static_usize)
        .ok_or_else(|| Error::invalid_shape("apply_rotary", "sequence length must be static"))?;
    let actual_dim = shape
        .dim(3)
        .and_then(Dim::static_usize)
        .ok_or_else(|| Error::invalid_shape("apply_rotary", "head dimension must be static"))?;
    if actual_dim != head_dim {
        return Err(Error::invalid_shape(
            "apply_rotary",
            format!("tensor head dimension {actual_dim} is not {head_dim}"),
        ));
    }
    let (cos, sin) = rotary_tables(head_dim, theta, partial_rotary_factor, start, length);
    let table_shape = Shape::new(vec![
        Dim::Static(1),
        Dim::Static(length),
        Dim::Static(1),
        Dim::Static(head_dim),
    ]);
    let cos = cx.builder().constant_tensor(ConstantTensor::f32(
        Shape::from_static([length, head_dim]),
        cos,
    )?)?;
    let sin = cx.builder().constant_tensor(ConstantTensor::f32(
        Shape::from_static([length, head_dim]),
        sin,
    )?)?;
    let cos = cx.builder().reshape(cos, table_shape.clone())?;
    let sin = cx.builder().reshape(sin, table_shape)?;
    let scaled = cx.builder().mul(value, cos)?;
    let rotated = rotate_half_value_op(cx, value, head_dim)?;
    let rotated = cx.builder().mul(rotated, sin)?;
    cx.count_operation();
    cx.builder().add(scaled, rotated)
}

fn rotary_tables(
    head_dim: usize,
    theta: f32,
    partial_rotary_factor: Option<f32>,
    start: usize,
    length: usize,
) -> (Vec<f32>, Vec<f32>) {
    let pairs = head_dim / 2;
    let rotated = match partial_rotary_factor {
        Some(factor) => ((factor * head_dim as f32 / 2.0) as usize).min(pairs),
        None => pairs,
    };
    let mut inverse = Vec::with_capacity(pairs);
    for index in 0..rotated {
        let exponent = (2 * index) as f32 / head_dim as f32;
        inverse.push(1.0 / theta.powf(exponent));
    }
    inverse.resize(pairs, 0.0);
    let mut cos = vec![0.0f32; length * head_dim];
    let mut sin = vec![0.0f32; length * head_dim];
    for offset in 0..length {
        let position = start + offset;
        for index in 0..pairs {
            let angle = position as f32 * inverse[index];
            let (s, c) = angle.sin_cos();
            cos[offset * head_dim + index] = c;
            cos[offset * head_dim + pairs + index] = c;
            sin[offset * head_dim + index] = s;
            sin[offset * head_dim + pairs + index] = s;
        }
    }
    (cos, sin)
}

fn rotate_half_value_op(cx: &mut ModelContext, value: ValueId, head_dim: usize) -> Result<ValueId> {
    let half = head_dim / 2;
    let first = cx.builder().slice(value, 3, 0, half)?;
    let second = cx.builder().slice(value, 3, half, half)?;
    let neg_second = cx.builder().neg(second)?;
    cx.builder().concat(&[neg_second, first], 3)
}

/// Grouped-query attention with a causal (optionally sliding) mask.
///
/// Layout: `[1, query_len, heads, head_dim]` and
/// `[1, kv_len, kv_heads, head_dim]`; output `[1, query_len, heads*head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn grouped_attention(
    cx: &mut ModelContext,
    query: impl Operand,
    key: impl Operand,
    value: impl Operand,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    causal: bool,
    scale: Option<f32>,
) -> Result<ValueId> {
    let query = query.into_operand(cx)?;
    let key = key.into_operand(cx)?;
    let value = value.into_operand(cx)?;
    let mask = if causal {
        let q_shape = cx.value_shape(query)?;
        let k_shape = cx.value_shape(key)?;
        let q_len = q_shape.dim(1).and_then(Dim::static_usize).ok_or_else(|| {
            Error::invalid_shape("grouped_attention", "query length must be static")
        })?;
        let kv_len = k_shape
            .dim(1)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("grouped_attention", "KV length must be static"))?;
        let mut values = vec![0.0f32; q_len * kv_len];
        for row in 0..q_len {
            for column in 0..kv_len {
                let visible = column <= row
                    && sliding_window
                        .map(|window| row - column < window)
                        .unwrap_or(true);
                if !visible {
                    values[row * kv_len + column] = -1.0e30;
                }
            }
        }
        Some(cx.builder().constant_tensor(ConstantTensor::f32(
            Shape::from_static([q_len, kv_len]),
            values,
        )?)?)
    } else {
        None
    };
    grouped_attention_masked(
        cx,
        query,
        key,
        value,
        num_heads,
        num_kv_heads,
        head_dim,
        mask,
        scale,
    )
}

/// Grouped-query attention with an explicit additive mask `[q_len, kv_len]`.
#[allow(clippy::too_many_arguments)]
pub fn grouped_attention_masked(
    cx: &mut ModelContext,
    query: impl Operand,
    key: impl Operand,
    value: impl Operand,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    mask: Option<ValueId>,
    scale: Option<f32>,
) -> Result<ValueId> {
    let query = query.into_operand(cx)?;
    let key = key.into_operand(cx)?;
    let value = value.into_operand(cx)?;
    if num_heads == 0 || num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(Error::invalid_shape(
            "grouped_attention",
            "heads must be positive and divisible by KV heads",
        ));
    }
    let q_shape = cx.value_shape(query)?;
    let k_shape = cx.value_shape(key)?;
    if q_shape.rank() != 4 || k_shape.rank() != 4 {
        return Err(Error::invalid_shape(
            "grouped_attention",
            "query/key must be [1, length, heads, head_dim]",
        ));
    }
    let batch = q_shape.dim(0).and_then(Dim::static_usize).unwrap_or(0);
    if batch != 1 {
        return Err(Error::unsupported(
            "grouped_attention supports batch size 1; loop over the batch",
        ));
    }
    let q_len = q_shape
        .dim(1)
        .and_then(Dim::static_usize)
        .ok_or_else(|| Error::invalid_shape("grouped_attention", "query length must be static"))?;
    let kv_len = k_shape
        .dim(1)
        .and_then(Dim::static_usize)
        .ok_or_else(|| Error::invalid_shape("grouped_attention", "KV length must be static"))?;
    let q_heads = q_shape
        .dim(2)
        .and_then(Dim::static_usize)
        .ok_or_else(|| Error::invalid_shape("grouped_attention", "head count must be static"))?;
    let kv_heads = k_shape
        .dim(2)
        .and_then(Dim::static_usize)
        .ok_or_else(|| Error::invalid_shape("grouped_attention", "KV head count must be static"))?;
    if q_heads != num_heads || kv_heads != num_kv_heads {
        return Err(Error::invalid_shape(
            "grouped_attention",
            format!(
                "tensor layout has {q_heads}/{kv_heads} heads but the call expects {num_heads}/{num_kv_heads}"
            ),
        ));
    }
    let groups = num_heads / num_kv_heads;
    let scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    let scale = cx.builder().constant_scalar(scale);
    let mut heads = Vec::with_capacity(num_heads);
    for head in 0..num_heads {
        let kv_head = head / groups;
        let qh = cx.builder().slice(query, 2, head, 1)?;
        let qh = cx
            .builder()
            .reshape(qh, Shape::from_static([q_len, head_dim]))?;
        let kh = cx.builder().slice(key, 2, kv_head, 1)?;
        let kh = cx
            .builder()
            .reshape(kh, Shape::from_static([kv_len, head_dim]))?;
        let vh = cx.builder().slice(value, 2, kv_head, 1)?;
        let vh = cx
            .builder()
            .reshape(vh, Shape::from_static([kv_len, head_dim]))?;
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
            .reshape(out, Shape::from_static([1, q_len, head_dim]))?;
        heads.push(out);
    }
    let merged = cx.builder().concat(&heads, 2)?;
    cx.count_operation();
    cx.builder()
        .reshape(merged, Shape::from_static([1, q_len, num_heads * head_dim]))
}
