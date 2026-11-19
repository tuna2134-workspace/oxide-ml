//! Tensor operations usable inside `#[model]` forward functions.
//!
//! The macros rewrite `relu(x)`, `x.relu()`, `x + y` and similar syntax into
//! calls to this module. Every function appends a node to the graph owned by
//! [`ModelContext`], validates shapes through the graph builder, and returns a
//! graph value. Scalars are accepted wherever a tensor operand is expected and
//! become graph constants.

use oxide_ml_core::{DType, Error, Result, ValueId};

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
