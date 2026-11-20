use oxide_ml_core::{DType, DenseTensor, Error, Result, Scalar};
use oxide_ml_graph::{
    ActivationKind, ConstantData, Op, OpAttrs, Reduction, normalize_axes, normalize_axis,
};
use oxide_ml_kernel::KernelCall;

pub(crate) fn dispatch<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    match call.op {
        Op::Add => binary(call, |a, b| a + b),
        Op::AddRelu => binary(call, |a, b| relu(a + b)),
        Op::Sub => binary(call, |a, b| a - b),
        Op::Mul => binary(call, |a, b| a * b),
        Op::Div => binary(call, |a, b| a / b),
        Op::Neg => unary(call, |value| -value),
        Op::ReLU => unary(call, relu),
        Op::GELU => unary(call, gelu),
        Op::Sigmoid => unary(call, sigmoid),
        Op::Tanh => unary(call, |value| value.tanh()),
        Op::Sqrt => unary(call, |value| value.sqrt()),
        Op::Exp => unary(call, |value| value.exp()),
        Op::Log => unary(call, |value| value.ln()),
        Op::Abs => unary(call, |value| value.abs()),
        Op::MatMul => matmul(call),
        Op::Softmax => softmax(call),
        Op::LogSoftmax => log_softmax(call),
        Op::Slice => slice(call),
        Op::Embedding => embedding(call),
        Op::LayerNorm => layer_norm(call),
        Op::Reshape => reshape(call),
        Op::Transpose => transpose(call),
        Op::BroadcastTo => broadcast(call),
        Op::ReduceSum => reduce(call, false),
        Op::ReduceMean => reduce(call, true),
        Op::Conv2d => conv2d(call),
        Op::MaxPool2d => pool2d(call, true),
        Op::AvgPool2d => pool2d(call, false),
        Op::MSELoss => mse_loss(call),
        Op::CrossEntropyLoss => cross_entropy_loss(call),
        Op::Constant => constant(call),
        Op::ConstantScalar => constant_scalar(call),
        Op::ActivationBackward(kind) => activation_backward(call, *kind),
        Op::AddReluBackward => add_relu_backward(call),
        Op::SoftmaxBackward => softmax_backward(call),
        Op::LayerNormBackward => layer_norm_backward(call),
        Op::MSELossBackward => mse_loss_backward(call),
        Op::CrossEntropyLossBackward => cross_entropy_loss_backward(call),
        Op::Conv2dBackward => conv2d_backward(call),
        Op::MaxPool2dBackward => pool2d_backward(call, true),
        Op::AvgPool2dBackward => pool2d_backward(call, false),
        Op::AbsBackward => abs_backward(call),
        Op::SliceBackward => slice_backward(call),
        Op::EmbeddingBackward => embedding_backward(call),
        Op::Unbroadcast => unbroadcast(call),
        Op::Cast => cast(call),
        Op::Concat => concat(call),
        Op::Split => split(call),
        Op::MulAdd => mul_add(call),
        Op::MatMulBias => matmul_bias(call),
        Op::MoE => moe(call),
        Op::Custom => Err(Error::execution(
            "custom kernel op reached the CPU backend without a custom kernel registry",
        )),
    }
}

fn out_dims<T: Scalar>(call: &KernelCall<'_, T>, index: usize) -> Result<Vec<usize>> {
    call.output_specs
        .get(index)
        .map(|spec| spec.shape.clone())
        .ok_or_else(|| {
            Error::execution(format!(
                "missing output shape {index} for {}",
                call.op.name()
            ))
        })
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product()
}

fn strides_for(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut acc = 1;
    for index in (0..dims.len()).rev() {
        strides[index] = acc;
        acc *= dims[index];
    }
    strides
}

fn unravel(mut linear: usize, dims: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut index = vec![0; dims.len()];
    for axis in 0..dims.len() {
        if dims[axis] == 0 {
            continue;
        }
        index[axis] = linear / strides[axis];
        linear %= strides[axis];
    }
    index
}

fn dot(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides)
        .map(|(index, stride)| index * stride)
        .sum()
}

fn can_broadcast_to(input: &[usize], output: &[usize]) -> bool {
    if input.len() > output.len() {
        return false;
    }
    let offset = output.len() - input.len();
    input
        .iter()
        .enumerate()
        .all(|(axis, &dim)| dim == 1 || dim == output[offset + axis])
}

fn broadcast_offset(
    index: &[usize],
    input_dims: &[usize],
    input_strides: &[usize],
    output_rank: usize,
) -> usize {
    let offset = output_rank - input_dims.len();
    let mut flat = 0;
    for (axis, &dim) in input_dims.iter().enumerate() {
        let coordinate = if dim == 1 { 0 } else { index[offset + axis] };
        flat += coordinate * input_strides[axis];
    }
    flat
}

fn expect_inputs<T: Scalar>(call: &KernelCall<'_, T>, count: usize) -> Result<()> {
    if call.inputs.len() != count {
        return Err(Error::execution(format!(
            "{} expects {count} inputs, got {}",
            call.op.name(),
            call.inputs.len()
        )));
    }
    Ok(())
}

/// Reject a call whose lowered input specs are not the dtype this backend was
/// instantiated for. The typed tensors already guarantee this, but a backend
/// must never trust a mismatched spec silently.
fn ensure_input_dtypes<T: Scalar>(call: &KernelCall<'_, T>) -> Result<()> {
    if call.input_specs.len() != call.inputs.len() {
        return Err(Error::execution(format!(
            "{} received {} input specs for {} inputs",
            call.op.name(),
            call.input_specs.len(),
            call.inputs.len()
        )));
    }
    for (index, spec) in call.input_specs.iter().enumerate() {
        if spec.dtype != T::DTYPE {
            return Err(Error::DTypeMismatch {
                context: format!("{} input {index}", call.op.name()),
                expected: T::DTYPE,
                got: spec.dtype,
            });
        }
    }
    Ok(())
}

fn row_for_index<T: Scalar>(value: T, vocab: usize, context: &str) -> Result<usize> {
    let raw = value.to_f64();
    if !raw.is_finite() || raw.fract() != 0.0 || raw < 0.0 || raw >= vocab as f64 {
        return Err(Error::execution(format!(
            "{context} index {raw} is invalid for vocabulary size {vocab}"
        )));
    }
    Ok(raw as usize)
}

fn binary<T: Scalar>(
    call: KernelCall<'_, T>,
    f: impl Fn(T, T) -> T,
) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    let dims = out_dims(&call, 0)?;
    let lhs = &call.inputs[0];
    let rhs = &call.inputs[1];
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    if !can_broadcast_to(&lhs_dims, &dims) || !can_broadcast_to(&rhs_dims, &dims) {
        return Err(Error::InvalidBroadcast {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        });
    }
    let lhs_strides = strides_for(&lhs_dims);
    let rhs_strides = strides_for(&rhs_dims);
    let out_strides = strides_for(&dims);
    let mut data = Vec::with_capacity(product(&dims));
    for linear in 0..product(&dims) {
        let index = unravel(linear, &dims, &out_strides);
        let a = lhs.data()[broadcast_offset(&index, &lhs_dims, &lhs_strides, dims.len())];
        let b = rhs.data()[broadcast_offset(&index, &rhs_dims, &rhs_strides, dims.len())];
        data.push(f(a, b));
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn unary<T: Scalar>(call: KernelCall<'_, T>, f: impl Fn(T) -> T) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let dims = out_dims(&call, 0)?;
    let input = &call.inputs[0];
    if input.dims() != dims {
        return Err(Error::shape_mismatch(
            call.op.name(),
            input.shape().clone(),
            oxide_ml_core::Shape::from_static(&dims),
        ));
    }
    let data = input.data().iter().copied().map(f).collect();
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn relu<T: Scalar>(value: T) -> T {
    if value > T::zero() { value } else { T::zero() }
}

fn sigmoid<T: Scalar>(value: T) -> T {
    let one = T::one();
    if value >= T::zero() {
        one / (one + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (one + exp)
    }
}

fn gelu<T: Scalar>(value: T) -> T {
    // Tanh approximation used consistently in forward and backward.
    let c = T::from_f64((2.0 / std::f64::consts::PI).sqrt());
    let half = T::from_f64(0.5);
    let cubic = T::from_f64(0.044_715);
    half * value * (T::one() + (c * (value + cubic * value.powi(3))).tanh())
}

fn matmul_attributes<T: Scalar>(call: &KernelCall<'_, T>) -> Result<(bool, bool)> {
    match call.attrs {
        OpAttrs::MatMul {
            transpose_a,
            transpose_b,
        } => Ok((*transpose_a, *transpose_b)),
        _ => Err(Error::execution("MatMul missing attributes")),
    }
}

fn matmul<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    let (transpose_a, transpose_b) = matmul_attributes(&call)?;
    let dims = out_dims(&call, 0)?;
    Ok(vec![matmul_compute(
        &call.inputs[0],
        &call.inputs[1],
        transpose_a,
        transpose_b,
        &dims,
    )?])
}

fn matmul_bias<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 3)?;
    let (transpose_a, transpose_b) = matmul_attributes(&call)?;
    let dims = out_dims(&call, 0)?;
    let output = matmul_compute(
        &call.inputs[0],
        &call.inputs[1],
        transpose_a,
        transpose_b,
        &dims,
    )?;
    let bias = call.inputs[2].broadcast_to(&dims)?;
    Ok(vec![output.zip_map(&bias, |value, bias| value + bias)?])
}

/// Dot product with four independent accumulators. The association order is
/// fixed, so serial and threaded paths produce bit-identical results.
fn dot_product<T: Scalar>(a_row: &[T], b_row: &[T]) -> T {
    let mut sums = [T::zero(); 4];
    let mut left = a_row.chunks_exact(4);
    let mut right = b_row.chunks_exact(4);
    for (a_chunk, b_chunk) in left.by_ref().zip(right.by_ref()) {
        sums[0] += a_chunk[0] * b_chunk[0];
        sums[1] += a_chunk[1] * b_chunk[1];
        sums[2] += a_chunk[2] * b_chunk[2];
        sums[3] += a_chunk[3] * b_chunk[3];
    }
    let mut acc = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for (a_value, b_value) in left.remainder().iter().zip(right.remainder()) {
        acc += *a_value * *b_value;
    }
    acc
}

fn matmul_compute<T: Scalar>(
    a: &DenseTensor<T>,
    b: &DenseTensor<T>,
    transpose_a: bool,
    transpose_b: bool,
    dims: &[usize],
) -> Result<DenseTensor<T>> {
    let a_dims = a.dims();
    let b_dims = b.dims();
    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Err(Error::execution("MatMul requires rank >= 2"));
    }
    let a_rank = a_dims.len();
    let b_rank = b_dims.len();
    let (m, k_a) = if transpose_a {
        (a_dims[a_rank - 1], a_dims[a_rank - 2])
    } else {
        (a_dims[a_rank - 2], a_dims[a_rank - 1])
    };
    let (k_b, n) = if transpose_b {
        (b_dims[b_rank - 1], b_dims[b_rank - 2])
    } else {
        (b_dims[b_rank - 2], b_dims[b_rank - 1])
    };
    if k_a != k_b {
        return Err(Error::InvalidBroadcast {
            lhs: a.shape().clone(),
            rhs: b.shape().clone(),
        });
    }
    if dims.len() < 2 || dims[dims.len() - 2] != m || dims[dims.len() - 1] != n {
        return Err(Error::execution(format!(
            "MatMul output shape {dims:?} is incompatible with M={m}, N={n}"
        )));
    }
    let batch_dims = &dims[..dims.len() - 2];
    let batch_count = product(batch_dims);
    let a_batch_dims = &a_dims[..a_rank - 2];
    let b_batch_dims = &b_dims[..b_rank - 2];
    let a_batch_strides = strides_for(a_batch_dims);
    let b_batch_strides = strides_for(b_batch_dims);
    let batch_rank = batch_dims.len();
    let a_offset_rank = batch_rank - a_batch_dims.len();
    let b_offset_rank = batch_rank - b_batch_dims.len();
    let a_matrix_strides = [a_dims[a_rank - 1], 1];
    let b_matrix_strides = [b_dims[b_rank - 1], 1];

    let mut data = vec![T::zero(); batch_count * m * n];

    // Fast path for the shape every `Linear` layer produces: A [M, K] times
    // B^T with B [N, K]. Row slices plus iterator zips let LLVM vectorize the
    // inner loop; the generic path below keeps every other layout correct.
    if !transpose_a
        && transpose_b
        && batch_count == 1
        && a_batch_dims.is_empty()
        && b_batch_dims.is_empty()
    {
        let a_data = a.data();
        let b_data = b.data();
        let total = m * n;
        let workers = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .min(8);
        if workers > 1 && total >= 16 * 16 && k_a >= 16 {
            // Split the contiguous output buffer, not the rows: Gemma's
            // [tokens, hidden] x [hidden, out] matmuls often have a small token
            // count and a large output width, so row-only parallelism would
            // leave most cores idle.
            let chunk_size = total.div_ceil(workers);
            std::thread::scope(|scope| {
                for (chunk_index, chunk) in data.chunks_mut(chunk_size).enumerate() {
                    scope.spawn(move || {
                        let start = chunk_index * chunk_size;
                        for (offset, slot) in chunk.iter_mut().enumerate() {
                            let index = start + offset;
                            let row = index / n;
                            let col = index % n;
                            let a_row = &a_data[row * k_a..(row + 1) * k_a];
                            let b_row = &b_data[col * k_b..(col + 1) * k_b];
                            *slot = dot_product(a_row, b_row);
                        }
                    });
                }
            });
        } else {
            for row in 0..m {
                let a_row = &a_data[row * k_a..(row + 1) * k_a];
                for col in 0..n {
                    let b_row = &b_data[col * k_b..(col + 1) * k_b];
                    data[row * n + col] = dot_product(a_row, b_row);
                }
            }
        }
        return DenseTensor::from_vec(dims, data);
    }

    for batch in 0..batch_count {
        let batch_index = unravel(batch, batch_dims, &strides_for(batch_dims));
        let mut a_base = 0;
        for (axis, &dim) in a_batch_dims.iter().enumerate() {
            let coordinate = if dim == 1 {
                0
            } else {
                batch_index[a_offset_rank + axis]
            };
            a_base += coordinate * a_batch_strides[axis];
        }
        let mut b_base = 0;
        for (axis, &dim) in b_batch_dims.iter().enumerate() {
            let coordinate = if dim == 1 {
                0
            } else {
                batch_index[b_offset_rank + axis]
            };
            b_base += coordinate * b_batch_strides[axis];
        }
        let out_base = batch * m * n;
        for row in 0..m {
            for col in 0..n {
                let mut acc = T::zero();
                for inner in 0..k_a {
                    let a_index = if transpose_a {
                        a_base + inner * a_matrix_strides[0] + row * a_matrix_strides[1]
                    } else {
                        a_base + row * a_matrix_strides[0] + inner * a_matrix_strides[1]
                    };
                    let b_index = if transpose_b {
                        b_base + col * b_matrix_strides[0] + inner * b_matrix_strides[1]
                    } else {
                        b_base + inner * b_matrix_strides[0] + col * b_matrix_strides[1]
                    };
                    acc += a.data()[a_index] * b.data()[b_index];
                }
                data[out_base + row * n + col] = acc;
            }
        }
    }
    DenseTensor::from_vec(dims, data)
}

fn cast<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let target = match call.attrs {
        OpAttrs::Cast { dtype } => *dtype,
        _ => return Err(Error::execution("Cast missing attributes")),
    };
    if target != T::DTYPE {
        return Err(Error::DTypeMismatch {
            context:
                "Cast (the reference executor is single-dtype; mixed precision is not implemented)"
                    .into(),
            expected: T::DTYPE,
            got: target,
        });
    }
    Ok(vec![call.inputs[0].clone()])
}

fn concat<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    if call.inputs.is_empty() {
        return Err(Error::execution("Concat requires at least one input"));
    }
    let axis = match call.attrs {
        OpAttrs::Concat { axis } => *axis,
        _ => return Err(Error::execution("Concat missing attributes")),
    };
    let dims = out_dims(&call, 0)?;
    if axis >= dims.len() {
        return Err(Error::InvalidAxis {
            axis: axis as i32,
            rank: dims.len(),
        });
    }
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    let mut data = Vec::with_capacity(product(&dims));
    for outer_index in 0..outer {
        for input in call.inputs {
            let input_dims = input.dims();
            if input_dims.len() != dims.len() {
                return Err(Error::RankMismatch {
                    context: "Concat input".into(),
                    expected: dims.len(),
                    got: input_dims.len(),
                });
            }
            let axis_len = input_dims[axis];
            let block = axis_len * inner;
            let start = outer_index * block;
            let end = start + block;
            if end > input.numel() {
                return Err(Error::execution("Concat input is smaller than declared"));
            }
            data.extend_from_slice(&input.data()[start..end]);
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn split<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let (axis, sizes) = match call.attrs {
        OpAttrs::Split { axis, sizes } => (*axis, sizes.clone()),
        _ => return Err(Error::execution("Split missing attributes")),
    };
    let input = &call.inputs[0];
    let input_dims = input.dims();
    if axis >= input_dims.len() {
        return Err(Error::InvalidAxis {
            axis: axis as i32,
            rank: input_dims.len(),
        });
    }
    let outer: usize = input_dims[..axis].iter().product();
    let inner: usize = input_dims[axis + 1..].iter().product();
    let input_axis = input_dims[axis];
    let mut offset = 0usize;
    let mut outputs = Vec::with_capacity(sizes.len());
    for (index, size) in sizes.iter().enumerate() {
        if offset + size > input_axis {
            return Err(Error::execution("Split sizes exceed the input axis"));
        }
        let dims = out_dims(&call, index)?;
        let len = size * inner;
        let mut data = Vec::with_capacity(outer * len);
        for outer_index in 0..outer {
            let start = (outer_index * input_axis + offset) * inner;
            data.extend_from_slice(&input.data()[start..start + len]);
        }
        outputs.push(DenseTensor::from_vec(&dims, data)?);
        offset += size;
    }
    Ok(outputs)
}

fn mul_add<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 3)?;
    let dims = out_dims(&call, 0)?;
    let a = call.inputs[0].broadcast_to(&dims)?;
    let b = call.inputs[1].broadcast_to(&dims)?;
    let c = call.inputs[2].broadcast_to(&dims)?;
    let product = a.zip_map(&b, |a, b| a * b)?;
    Ok(vec![product.zip_map(&c, |product, c| product + c)?])
}

fn softmax<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let dims = out_dims(&call, 0)?;
    let axis = match call.attrs {
        OpAttrs::Softmax { axis } => normalize_axis(*axis, dims.len())?,
        _ => return Err(Error::execution("Softmax missing attributes")),
    };
    let input = &call.inputs[0];
    if input.dims() != dims {
        return Err(Error::shape_mismatch(
            "Softmax",
            input.shape().clone(),
            oxide_ml_core::Shape::from_static(&dims),
        ));
    }
    let outer: usize = dims[..axis].iter().product();
    let axis_len = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product();
    let mut data = input.data().to_vec();
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_len * inner + i;
            let mut max = T::from_f64(f64::NEG_INFINITY);
            for a in 0..axis_len {
                max = max.max(data[base + a * inner]);
            }
            let mut sum = T::zero();
            for a in 0..axis_len {
                let value = (data[base + a * inner] - max).exp();
                data[base + a * inner] = value;
                sum += value;
            }
            for a in 0..axis_len {
                data[base + a * inner] /= sum;
            }
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

/// Numerically stable log-softmax: `x - max - ln(sum(exp(x - max)))`.
fn log_softmax<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    ensure_input_dtypes(&call)?;
    let dims = out_dims(&call, 0)?;
    let axis = match call.attrs {
        OpAttrs::LogSoftmax { axis } => normalize_axis(*axis, dims.len())?,
        _ => return Err(Error::execution("LogSoftmax missing attributes")),
    };
    let input = &call.inputs[0];
    if input.dims() != dims {
        return Err(Error::shape_mismatch(
            "LogSoftmax",
            input.shape().clone(),
            oxide_ml_core::Shape::from_static(&dims),
        ));
    }
    let outer: usize = dims[..axis].iter().product();
    let axis_len = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product();
    let mut data = vec![T::zero(); input.numel()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_len * inner + i;
            let mut max = T::from_f64(f64::NEG_INFINITY);
            for a in 0..axis_len {
                max = max.max(input.data()[base + a * inner]);
            }
            let mut sum_exp = T::zero();
            for a in 0..axis_len {
                sum_exp += (input.data()[base + a * inner] - max).exp();
            }
            let log_sum_exp = max + sum_exp.ln();
            for a in 0..axis_len {
                data[base + a * inner] = input.data()[base + a * inner] - log_sum_exp;
            }
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn slice<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    ensure_input_dtypes(&call)?;
    let (axis, start, len) = match call.attrs {
        OpAttrs::Slice { axis, start, len } => (*axis, *start, *len),
        _ => return Err(Error::execution("Slice missing attributes")),
    };
    let dims = out_dims(&call, 0)?;
    let input = &call.inputs[0];
    let in_dims = input.dims();
    if axis >= in_dims.len() {
        return Err(Error::execution(format!(
            "Slice axis {axis} is out of range for rank {}",
            in_dims.len()
        )));
    }
    let end = start.checked_add(len).ok_or_else(|| {
        Error::execution(format!(
            "Slice range start {start} plus length {len} overflows"
        ))
    })?;
    if end > in_dims[axis] {
        return Err(Error::execution(format!(
            "Slice range [{start}, {end}) exceeds dimension size {}",
            in_dims[axis]
        )));
    }
    let mut expected = in_dims.clone();
    expected[axis] = len;
    if dims != expected {
        return Err(Error::execution(format!(
            "Slice output shape {dims:?} does not match expected {expected:?}"
        )));
    }
    let in_strides = strides_for(&in_dims);
    let out_strides = strides_for(&dims);
    let mut data = Vec::with_capacity(product(&dims));
    for linear in 0..product(&dims) {
        let mut index = unravel(linear, &dims, &out_strides);
        index[axis] += start;
        data.push(input.data()[dot(&index, &in_strides)]);
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn embedding<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    ensure_input_dtypes(&call)?;
    let weight = &call.inputs[0];
    let indices = &call.inputs[1];
    let weight_dims = weight.dims();
    if weight_dims.len() != 2 {
        return Err(Error::execution(
            "Embedding weight must be a rank-2 [vocab, dim] tensor",
        ));
    }
    let (vocab, dim) = (weight_dims[0], weight_dims[1]);
    let dims = out_dims(&call, 0)?;
    let mut expected = indices.dims();
    expected.push(dim);
    if dims != expected {
        return Err(Error::execution(format!(
            "Embedding output shape {dims:?} does not match expected {expected:?}"
        )));
    }
    let mut data = Vec::with_capacity(indices.numel() * dim);
    for &raw in indices.data() {
        let row = row_for_index(raw, vocab, "Embedding")?;
        let start = row * dim;
        data.extend_from_slice(&weight.data()[start..start + dim]);
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn layer_norm<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 3)?;
    let (normalized_shape, eps) = match call.attrs {
        OpAttrs::LayerNorm {
            normalized_shape,
            eps,
        } => (normalized_shape, *eps),
        _ => return Err(Error::execution("LayerNorm missing attributes")),
    };
    let dims = out_dims(&call, 0)?;
    let input = &call.inputs[0];
    let gamma = &call.inputs[1];
    let beta = &call.inputs[2];
    if input.dims() != dims {
        return Err(Error::execution("LayerNorm output shape mismatch"));
    }
    let norm: usize = normalized_shape.iter().product();
    if norm == 0 || input.numel() % norm != 0 {
        return Err(Error::execution(
            "LayerNorm normalized size does not divide input",
        ));
    }
    if gamma.numel() != norm || beta.numel() != norm {
        return Err(Error::execution(
            "LayerNorm affine parameters have wrong size",
        ));
    }
    let groups = input.numel() / norm;
    let mut data = vec![T::zero(); input.numel()];
    for group in 0..groups {
        let slice = &input.data()[group * norm..(group + 1) * norm];
        let mean = slice
            .iter()
            .copied()
            .fold(T::zero(), |acc, value| acc + value)
            / T::from_usize(norm);
        let variance = slice
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .fold(T::zero(), |acc, value| acc + value)
            / T::from_usize(norm);
        let inv_std = T::one() / (variance + T::from_f64(eps as f64)).sqrt();
        for index in 0..norm {
            let normalized = (slice[index] - mean) * inv_std;
            data[group * norm + index] = normalized * gamma.data()[index] + beta.data()[index];
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn reshape<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let dims = out_dims(&call, 0)?;
    let input = &call.inputs[0];
    if product(&dims) != input.numel() {
        return Err(Error::execution("reshape element count mismatch"));
    }
    Ok(vec![DenseTensor::from_vec(&dims, input.data().to_vec())?])
}

fn transpose<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let axes = match call.attrs {
        OpAttrs::Transpose { axes } => axes,
        _ => return Err(Error::execution("Transpose missing attributes")),
    };
    let dims = out_dims(&call, 0)?;
    let input = &call.inputs[0];
    let in_dims = input.dims();
    if axes.len() != in_dims.len() || dims.len() != in_dims.len() {
        return Err(Error::execution("Transpose rank mismatch"));
    }
    let in_strides = strides_for(&in_dims);
    let out_strides = strides_for(&dims);
    let mut data = vec![T::zero(); input.numel()];
    for (linear, slot) in data.iter_mut().enumerate() {
        let out_index = unravel(linear, &dims, &out_strides);
        let mut in_index = vec![0; in_dims.len()];
        for (out_axis, &in_axis) in axes.iter().enumerate() {
            in_index[in_axis] = out_index[out_axis];
        }
        *slot = input.data()[dot(&in_index, &in_strides)];
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn broadcast<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let dims = out_dims(&call, 0)?;
    Ok(vec![call.inputs[0].broadcast_to(&dims)?])
}

fn reduce<T: Scalar>(call: KernelCall<'_, T>, mean: bool) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let (axes, keep_dims) = match call.attrs {
        OpAttrs::Reduce { axes, keep_dims } => (axes.clone(), *keep_dims),
        _ => return Err(Error::execution("Reduce missing attributes")),
    };
    let dims = out_dims(&call, 0)?;
    let input = &call.inputs[0];
    let in_dims = input.dims();
    let axes = normalize_axes(&axes, in_dims.len())?;
    let out_rank = if keep_dims {
        in_dims.len()
    } else {
        in_dims.len() - axes.len()
    };
    if dims.len() != out_rank {
        return Err(Error::execution("reduce output rank mismatch"));
    }
    let in_strides = strides_for(&in_dims);
    let out_strides = strides_for(&dims);
    let mut data = vec![T::zero(); product(&dims)];
    let mut counts = vec![0usize; product(&dims)];
    // Iterate input, accumulate into the corresponding reduced coordinate.
    for linear in 0..input.numel() {
        let in_index = unravel(linear, &in_dims, &in_strides);
        let mut out_index = Vec::with_capacity(out_rank);
        for (axis, &coordinate) in in_index.iter().enumerate() {
            if axes.contains(&axis) {
                if keep_dims {
                    out_index.push(0);
                }
            } else {
                out_index.push(coordinate);
            }
        }
        let out_offset = dot(&out_index, &out_strides);
        data[out_offset] += input.data()[linear];
        counts[out_offset] += 1;
    }
    if mean {
        for (value, &count) in data.iter_mut().zip(&counts) {
            if count > 0 {
                *value /= T::from_usize(count);
            }
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn mse_loss<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    let reduction = loss_reduction(call.attrs)?;
    let prediction = &call.inputs[0];
    let target = &call.inputs[1];
    if prediction.dims() != target.dims() {
        return Err(Error::execution("MSELoss operand shape mismatch"));
    }
    let mut total = T::zero();
    for (&p, &t) in prediction.data().iter().zip(target.data()) {
        let delta = p - t;
        total += delta * delta;
    }
    let value = match reduction {
        Reduction::Mean => {
            if prediction.numel() == 0 {
                T::zero()
            } else {
                total / T::from_usize(prediction.numel())
            }
        }
        Reduction::Sum => total,
    };
    Ok(vec![DenseTensor::scalar(value)])
}

fn cross_entropy_loss<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    let reduction = loss_reduction(call.attrs)?;
    let logits = &call.inputs[0];
    let targets = &call.inputs[1];
    let dims = logits.dims();
    if dims.len() != 2 {
        return Err(Error::execution("CrossEntropyLoss logits must be rank 2"));
    }
    let (batch, classes) = (dims[0], dims[1]);
    if targets.numel() != batch {
        return Err(Error::execution("CrossEntropyLoss target count mismatch"));
    }
    let mut total = T::zero();
    for row in 0..batch {
        let logits_row = &logits.data()[row * classes..(row + 1) * classes];
        let target_class = targets.data()[row].to_f64();
        if !target_class.is_finite()
            || target_class < 0.0
            || target_class >= classes as f64
            || target_class.fract() != 0.0
        {
            return Err(Error::execution(format!(
                "CrossEntropyLoss target class {target_class} is invalid for {classes} classes"
            )));
        }
        let max = logits_row
            .iter()
            .copied()
            .fold(T::from_f64(f64::NEG_INFINITY), |acc, value| acc.max(value));
        let sum_exp = logits_row
            .iter()
            .map(|value| (*value - max).exp())
            .fold(T::zero(), |acc, value| acc + value);
        let log_sum_exp = max + sum_exp.ln();
        total += log_sum_exp - logits_row[target_class as usize];
    }
    let value = match reduction {
        Reduction::Mean => {
            if batch == 0 {
                T::zero()
            } else {
                total / T::from_usize(batch)
            }
        }
        Reduction::Sum => total,
    };
    Ok(vec![DenseTensor::scalar(value)])
}

fn constant<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    let value = match call.attrs {
        OpAttrs::Constant { value } => value,
        _ => return Err(Error::execution("Constant missing attributes")),
    };
    value.validate()?;
    let dims = value.shape.static_dims().ok_or_else(|| {
        Error::execution("Constant shape must be fully static for the CPU backend")
    })?;
    let data = match (&value.data, T::DTYPE) {
        (ConstantData::F32(values), DType::F32) => values
            .iter()
            .map(|value| T::from_f64(*value as f64))
            .collect(),
        (ConstantData::F64(values), DType::F64) => {
            values.iter().map(|value| T::from_f64(*value)).collect()
        }
        (data, dtype) => {
            return Err(Error::DTypeMismatch {
                context: "Constant".into(),
                expected: dtype,
                got: match data {
                    ConstantData::F32(_) => DType::F32,
                    ConstantData::F64(_) => DType::F64,
                },
            });
        }
    };
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn constant_scalar<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    let value = match call.attrs {
        OpAttrs::ConstantScalar { value } => T::from_f64(*value as f64),
        _ => return Err(Error::execution("ConstantScalar missing attributes")),
    };
    Ok(vec![DenseTensor::scalar(value)])
}

fn loss_reduction(attrs: &OpAttrs) -> Result<Reduction> {
    match attrs {
        OpAttrs::Loss { reduction } => Ok(*reduction),
        _ => Err(Error::execution(
            "loss operation missing reduction attributes",
        )),
    }
}

fn add_relu_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 3)?;
    let a = &call.inputs[0];
    let b = &call.inputs[1];
    let dy = &call.inputs[2];
    let dims = out_dims(&call, 0)?;
    let a_dims = a.dims();
    let b_dims = b.dims();
    if !can_broadcast_to(&a_dims, &dims) || !can_broadcast_to(&b_dims, &dims) || dy.dims() != dims {
        return Err(Error::execution("AddReluBackward shape mismatch"));
    }
    let a_strides = strides_for(&a_dims);
    let b_strides = strides_for(&b_dims);
    let out_strides = strides_for(&dims);
    let mut data = Vec::with_capacity(product(&dims));
    for linear in 0..product(&dims) {
        let index = unravel(linear, &dims, &out_strides);
        let av = a.data()[broadcast_offset(&index, &a_dims, &a_strides, dims.len())];
        let bv = b.data()[broadcast_offset(&index, &b_dims, &b_strides, dims.len())];
        data.push(if av + bv > T::zero() {
            dy.data()[linear]
        } else {
            T::zero()
        });
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn activation_backward<T: Scalar>(
    call: KernelCall<'_, T>,
    kind: ActivationKind,
) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    let x = &call.inputs[0];
    let dy = &call.inputs[1];
    let dims = out_dims(&call, 0)?;
    if x.dims() != dims || dy.dims() != dims {
        return Err(Error::execution("activation backward shape mismatch"));
    }
    let mut data = Vec::with_capacity(x.numel());
    for (&x, &dy) in x.data().iter().zip(dy.data()) {
        let derivative = match kind {
            ActivationKind::ReLU => {
                if x > T::zero() {
                    T::one()
                } else {
                    T::zero()
                }
            }
            ActivationKind::GELU => {
                let c = T::from_f64((2.0 / std::f64::consts::PI).sqrt());
                let cubic = T::from_f64(0.044_715);
                let half = T::from_f64(0.5);
                let inner = c * (x + cubic * x.powi(3));
                let t = inner.tanh();
                half * (T::one() + t)
                    + half
                        * x
                        * (T::one() - t * t)
                        * c
                        * (T::one() + T::from_f64(3.0) * cubic * x * x)
            }
            ActivationKind::Sigmoid => {
                let s = sigmoid(x);
                s * (T::one() - s)
            }
            ActivationKind::Tanh => {
                let t = x.tanh();
                T::one() - t * t
            }
        };
        data.push(dy * derivative);
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn softmax_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    let y = &call.inputs[0];
    let dy = &call.inputs[1];
    let dims = out_dims(&call, 0)?;
    let axis = match call.attrs {
        OpAttrs::Softmax { axis } => normalize_axis(*axis, dims.len())?,
        _ => return Err(Error::execution("SoftmaxBackward missing attributes")),
    };
    if y.dims() != dims || dy.dims() != dims {
        return Err(Error::execution("SoftmaxBackward shape mismatch"));
    }
    let outer: usize = dims[..axis].iter().product();
    let axis_len = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product();
    let mut data = vec![T::zero(); y.numel()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_len * inner + i;
            let mut dot = T::zero();
            for a in 0..axis_len {
                dot += y.data()[base + a * inner] * dy.data()[base + a * inner];
            }
            for a in 0..axis_len {
                data[base + a * inner] =
                    y.data()[base + a * inner] * (dy.data()[base + a * inner] - dot);
            }
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn abs_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    ensure_input_dtypes(&call)?;
    let x = &call.inputs[0];
    let dy = &call.inputs[1];
    let dims = out_dims(&call, 0)?;
    if x.dims() != dims || dy.dims() != dims {
        return Err(Error::execution("AbsBackward shape mismatch"));
    }
    let mut data = Vec::with_capacity(x.numel());
    for (&x, &dy) in x.data().iter().zip(dy.data()) {
        let sign = if x > T::zero() {
            T::one()
        } else if x < T::zero() {
            -T::one()
        } else {
            T::zero()
        };
        data.push(dy * sign);
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn slice_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    ensure_input_dtypes(&call)?;
    let (axis, start, len) = match call.attrs {
        OpAttrs::SliceBackward {
            axis,
            start,
            len,
            input_shape: _,
        } => (*axis, *start, *len),
        _ => return Err(Error::execution("SliceBackward missing attributes")),
    };
    let dy = &call.inputs[0];
    let dims = out_dims(&call, 0)?;
    if axis >= dims.len() {
        return Err(Error::execution(format!(
            "SliceBackward axis {axis} is out of range for rank {}",
            dims.len()
        )));
    }
    let end = start.checked_add(len).ok_or_else(|| {
        Error::execution(format!(
            "SliceBackward range start {start} plus length {len} overflows"
        ))
    })?;
    if end > dims[axis] {
        return Err(Error::execution(format!(
            "SliceBackward range [{start}, {end}) exceeds dimension size {}",
            dims[axis]
        )));
    }
    let mut expected = dims.clone();
    expected[axis] = len;
    if dy.dims() != expected {
        return Err(Error::execution(format!(
            "SliceBackward upstream shape {:?} does not match expected {expected:?}",
            dy.dims()
        )));
    }
    let out_strides = strides_for(&dims);
    let dy_strides = strides_for(&expected);
    let mut data = vec![T::zero(); product(&dims)];
    for linear in 0..dy.numel() {
        let mut index = unravel(linear, &expected, &dy_strides);
        index[axis] += start;
        data[dot(&index, &out_strides)] = dy.data()[linear];
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn embedding_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    ensure_input_dtypes(&call)?;
    let vocab = match call.attrs {
        OpAttrs::EmbeddingBackward { vocab } => *vocab,
        _ => return Err(Error::execution("EmbeddingBackward missing attributes")),
    };
    let indices = &call.inputs[0];
    let dy = &call.inputs[1];
    let index_dims = indices.dims();
    let dy_dims = dy.dims();
    if dy_dims.len() != index_dims.len() + 1 {
        return Err(Error::execution(
            "EmbeddingBackward upstream gradient rank mismatch",
        ));
    }
    if dy_dims[..index_dims.len()] != index_dims[..] {
        return Err(Error::execution(
            "EmbeddingBackward upstream gradient prefix mismatch",
        ));
    }
    let dim = dy_dims[index_dims.len()];
    let dims = out_dims(&call, 0)?;
    if dims != [vocab, dim] {
        return Err(Error::execution(format!(
            "EmbeddingBackward output shape {dims:?} does not match [{vocab}, {dim}]"
        )));
    }
    let mut data = vec![T::zero(); vocab * dim];
    // Ordered, deterministic scatter-add: indices are visited in row-major
    // order, matching the forward pass exactly.
    for (slot, &raw) in indices.data().iter().enumerate() {
        let row = row_for_index(raw, vocab, "EmbeddingBackward")?;
        let source = slot * dim;
        let target = row * dim;
        for column in 0..dim {
            data[target + column] += dy.data()[source + column];
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn layer_norm_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 3)?;
    let (normalized_shape, eps) = match call.attrs {
        OpAttrs::LayerNorm {
            normalized_shape,
            eps,
        } => (normalized_shape, *eps),
        _ => return Err(Error::execution("LayerNormBackward missing attributes")),
    };
    let x = &call.inputs[0];
    let gamma = &call.inputs[1];
    let dy = &call.inputs[2];
    let dims = out_dims(&call, 0)?;
    if x.dims() != dims || dy.dims() != dims {
        return Err(Error::execution("LayerNormBackward shape mismatch"));
    }
    let norm: usize = normalized_shape.iter().product();
    if norm == 0 || x.numel() % norm != 0 || gamma.numel() != norm {
        return Err(Error::execution(
            "LayerNormBackward normalized size mismatch",
        ));
    }
    let groups = x.numel() / norm;
    let mut dx = vec![T::zero(); x.numel()];
    let mut dgamma = vec![T::zero(); norm];
    let mut dbeta = vec![T::zero(); norm];
    for group in 0..groups {
        let start = group * norm;
        let slice = &x.data()[start..start + norm];
        let mean = slice
            .iter()
            .copied()
            .fold(T::zero(), |acc, value| acc + value)
            / T::from_usize(norm);
        let variance = slice
            .iter()
            .map(|value| (*value - mean).powi(2))
            .fold(T::zero(), |acc, value| acc + value)
            / T::from_usize(norm);
        let inv_std = T::one() / (variance + T::from_f64(eps as f64)).sqrt();
        let mut sum_dxhat = T::zero();
        let mut sum_dxhat_xhat = T::zero();
        for index in 0..norm {
            let xhat = (slice[index] - mean) * inv_std;
            let dxhat = dy.data()[start + index] * gamma.data()[index];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * xhat;
            dgamma[index] += dy.data()[start + index] * xhat;
            dbeta[index] += dy.data()[start + index];
        }
        for index in 0..norm {
            let xhat = (slice[index] - mean) * inv_std;
            let dxhat = dy.data()[start + index] * gamma.data()[index];
            dx[start + index] = inv_std
                * (dxhat
                    - sum_dxhat / T::from_usize(norm)
                    - xhat * sum_dxhat_xhat / T::from_usize(norm));
        }
    }
    Ok(vec![
        DenseTensor::from_vec(&dims, dx)?,
        DenseTensor::from_vec(gamma.dims(), dgamma)?,
        DenseTensor::from_vec(gamma.dims(), dbeta)?,
    ])
}

fn mse_loss_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 3)?;
    let reduction = loss_reduction(call.attrs)?;
    let prediction = &call.inputs[0];
    let target = &call.inputs[1];
    let dy = &call.inputs[2];
    if prediction.dims() != target.dims() || !dy.shape().is_scalar() {
        return Err(Error::execution("MSELossBackward shape mismatch"));
    }
    let count = prediction.numel();
    let scale = match reduction {
        Reduction::Mean => {
            if count == 0 {
                T::zero()
            } else {
                T::from_f64(2.0) / T::from_usize(count)
            }
        }
        Reduction::Sum => T::from_f64(2.0),
    };
    let upstream = dy.data()[0] * scale;
    let mut dpred = Vec::with_capacity(count);
    let mut dtarget = Vec::with_capacity(count);
    for (&p, &t) in prediction.data().iter().zip(target.data()) {
        dpred.push((p - t) * upstream);
        dtarget.push(-(p - t) * upstream);
    }
    Ok(vec![
        DenseTensor::from_vec(prediction.dims(), dpred)?,
        DenseTensor::from_vec(target.dims(), dtarget)?,
    ])
}

fn cross_entropy_loss_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 3)?;
    let reduction = loss_reduction(call.attrs)?;
    let logits = &call.inputs[0];
    let targets = &call.inputs[1];
    let dy = &call.inputs[2];
    if !dy.shape().is_scalar() {
        return Err(Error::execution(
            "CrossEntropyLossBackward dy must be scalar",
        ));
    }
    let dims = logits.dims();
    let (batch, classes) = (dims[0], dims[1]);
    let scale = match reduction {
        Reduction::Mean => {
            if batch == 0 {
                T::zero()
            } else {
                dy.data()[0] / T::from_usize(batch)
            }
        }
        Reduction::Sum => dy.data()[0],
    };
    let mut data = vec![T::zero(); logits.numel()];
    for row in 0..batch {
        let start = row * classes;
        let row_logits = &logits.data()[start..start + classes];
        let max = row_logits
            .iter()
            .copied()
            .fold(T::from_f64(f64::NEG_INFINITY), |acc, value| acc.max(value));
        let sum_exp = row_logits
            .iter()
            .map(|value| (*value - max).exp())
            .fold(T::zero(), |acc, value| acc + value);
        let target_class = targets.data()[row].to_f64();
        if !target_class.is_finite()
            || target_class < 0.0
            || target_class >= classes as f64
            || target_class.fract() != 0.0
        {
            return Err(Error::execution("CrossEntropyLossBackward invalid target"));
        }
        for class in 0..classes {
            let probability = (row_logits[class] - max).exp() / sum_exp;
            let indicator = if class == target_class as usize {
                T::one()
            } else {
                T::zero()
            };
            data[start + class] = (probability - indicator) * scale;
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn conv2d<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    let (stride, padding, dilation, groups) = conv_attrs(call.attrs)?;
    if !(2..=3).contains(&call.inputs.len()) {
        return Err(Error::execution("Conv2d expects 2 or 3 inputs"));
    }
    let x = &call.inputs[0];
    let weight = &call.inputs[1];
    let bias = call.inputs.get(2);
    let x_dims = x.dims();
    let w_dims = weight.dims();
    if x_dims.len() != 4 || w_dims.len() != 4 {
        return Err(Error::execution(
            "Conv2d expects NCHW input and OIHW weight",
        ));
    }
    let dims = out_dims(&call, 0)?;
    if dims.len() != 4 {
        return Err(Error::execution("Conv2d output must be rank 4"));
    }
    let (n, out_c, out_h, out_w) = (dims[0], dims[1], dims[2], dims[3]);
    let (in_c, in_h, in_w) = (x_dims[1], x_dims[2], x_dims[3]);
    let (weight_out, in_per_group, kernel_h, kernel_w) =
        (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
    if out_c != weight_out || in_per_group * groups != in_c {
        return Err(Error::execution("Conv2d channel configuration mismatch"));
    }
    if let Some(bias) = bias
        && bias.numel() != out_c
    {
        return Err(Error::execution("Conv2d bias length mismatch"));
    }
    let mut data = vec![T::zero(); n * out_c * out_h * out_w];
    for ni in 0..n {
        for oc in 0..out_c {
            let group = oc / (out_c / groups);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias.map(|b| b.data()[oc]).unwrap_or_else(T::zero);
                    for ic_local in 0..in_per_group {
                        let ic = group * in_per_group + ic_local;
                        for kh in 0..kernel_h {
                            let ih = oh as isize * stride[0] as isize - padding[0] as isize
                                + kh as isize * dilation[0] as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            for kw in 0..kernel_w {
                                let iw = ow as isize * stride[1] as isize - padding[1] as isize
                                    + kw as isize * dilation[1] as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                let x_index =
                                    ((ni * in_c + ic) * in_h + ih as usize) * in_w + iw as usize;
                                let w_index = ((oc * in_per_group + ic_local) * kernel_h + kh)
                                    * kernel_w
                                    + kw;
                                acc += x.data()[x_index] * weight.data()[w_index];
                            }
                        }
                    }
                    let out_index = ((ni * out_c + oc) * out_h + oh) * out_w + ow;
                    data[out_index] = acc;
                }
            }
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn conv2d_backward<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    let (stride, padding, dilation, groups) = conv_attrs(call.attrs)?;
    expect_inputs(&call, 3)?;
    let x = &call.inputs[0];
    let weight = &call.inputs[1];
    let dy = &call.inputs[2];
    let x_dims = x.dims();
    let w_dims = weight.dims();
    let dy_dims = dy.dims();
    if x_dims.len() != 4 || w_dims.len() != 4 || dy_dims.len() != 4 {
        return Err(Error::execution("Conv2dBackward expects rank 4 tensors"));
    }
    let (n, in_c, in_h, in_w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let (out_c, in_per_group, kernel_h, kernel_w) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
    let (_, dy_c, out_h, out_w) = (dy_dims[0], dy_dims[1], dy_dims[2], dy_dims[3]);
    if dy_c != out_c || in_per_group * groups != in_c {
        return Err(Error::execution("Conv2dBackward channel mismatch"));
    }
    let mut dx = vec![T::zero(); x.numel()];
    let mut dw = vec![T::zero(); weight.numel()];
    let mut db = vec![T::zero(); out_c];
    for ni in 0..n {
        for (oc, db_slot) in db.iter_mut().enumerate() {
            let group = oc / (out_c / groups);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let dy_index = ((ni * out_c + oc) * out_h + oh) * out_w + ow;
                    let g = dy.data()[dy_index];
                    *db_slot += g;
                    for ic_local in 0..in_per_group {
                        let ic = group * in_per_group + ic_local;
                        for kh in 0..kernel_h {
                            let ih = oh as isize * stride[0] as isize - padding[0] as isize
                                + kh as isize * dilation[0] as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            for kw in 0..kernel_w {
                                let iw = ow as isize * stride[1] as isize - padding[1] as isize
                                    + kw as isize * dilation[1] as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                let x_index =
                                    ((ni * in_c + ic) * in_h + ih as usize) * in_w + iw as usize;
                                let w_index = ((oc * in_per_group + ic_local) * kernel_h + kh)
                                    * kernel_w
                                    + kw;
                                dx[x_index] += g * weight.data()[w_index];
                                dw[w_index] += g * x.data()[x_index];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(vec![
        DenseTensor::from_vec(&x_dims, dx)?,
        DenseTensor::from_vec(&w_dims, dw)?,
        DenseTensor::from_vec([out_c], db)?,
    ])
}

type ConvShape = ([usize; 2], [usize; 2], [usize; 2], usize);

fn conv_attrs(attrs: &OpAttrs) -> Result<ConvShape> {
    match attrs {
        OpAttrs::Conv2d {
            stride,
            padding,
            dilation,
            groups,
        } => Ok((*stride, *padding, *dilation, *groups)),
        _ => Err(Error::execution("convolution missing attributes")),
    }
}

fn pool2d<T: Scalar>(call: KernelCall<'_, T>, max: bool) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let (kernel, stride, padding, dilation) = pool_attrs(call.attrs)?;
    let x = &call.inputs[0];
    let x_dims = x.dims();
    if x_dims.len() != 4 {
        return Err(Error::execution("pooling expects NCHW"));
    }
    let dims = out_dims(&call, 0)?;
    let (n, c, out_h, out_w) = (dims[0], dims[1], dims[2], dims[3]);
    let (in_h, in_w) = (x_dims[2], x_dims[3]);
    let mut data = vec![T::zero(); n * c * out_h * out_w];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut best = T::from_f64(f64::NEG_INFINITY);
                    let mut sum = T::zero();
                    let mut count = 0usize;
                    for kh in 0..kernel[0] {
                        let ih = oh as isize * stride[0] as isize - padding[0] as isize
                            + kh as isize * dilation[0] as isize;
                        if ih < 0 || ih >= in_h as isize {
                            continue;
                        }
                        for kw in 0..kernel[1] {
                            let iw = ow as isize * stride[1] as isize - padding[1] as isize
                                + kw as isize * dilation[1] as isize;
                            if iw < 0 || iw >= in_w as isize {
                                continue;
                            }
                            let index = ((ni * c + ci) * in_h + ih as usize) * in_w + iw as usize;
                            let value = x.data()[index];
                            best = best.max(value);
                            sum += value;
                            count += 1;
                        }
                    }
                    let value = if max {
                        if count == 0 { T::zero() } else { best }
                    } else if count == 0 {
                        T::zero()
                    } else {
                        sum / T::from_usize(count)
                    };
                    data[((ni * c + ci) * out_h + oh) * out_w + ow] = value;
                }
            }
        }
    }
    Ok(vec![DenseTensor::from_vec(&dims, data)?])
}

fn pool2d_backward<T: Scalar>(call: KernelCall<'_, T>, max: bool) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 2)?;
    let (kernel, stride, padding, dilation) = pool_attrs(call.attrs)?;
    let x = &call.inputs[0];
    let dy = &call.inputs[1];
    let x_dims = x.dims();
    let dy_dims = dy.dims();
    if x_dims.len() != 4 || dy_dims.len() != 4 {
        return Err(Error::execution("pool backward expects rank 4"));
    }
    let (n, c, out_h, out_w) = (dy_dims[0], dy_dims[1], dy_dims[2], dy_dims[3]);
    let (in_h, in_w) = (x_dims[2], x_dims[3]);
    let mut dx = vec![T::zero(); x.numel()];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let dy_index = ((ni * c + ci) * out_h + oh) * out_w + ow;
                    let g = dy.data()[dy_index];
                    let mut best = T::from_f64(f64::NEG_INFINITY);
                    let mut best_index = None;
                    let mut count = 0usize;
                    for kh in 0..kernel[0] {
                        let ih = oh as isize * stride[0] as isize - padding[0] as isize
                            + kh as isize * dilation[0] as isize;
                        if ih < 0 || ih >= in_h as isize {
                            continue;
                        }
                        for kw in 0..kernel[1] {
                            let iw = ow as isize * stride[1] as isize - padding[1] as isize
                                + kw as isize * dilation[1] as isize;
                            if iw < 0 || iw >= in_w as isize {
                                continue;
                            }
                            let index = ((ni * c + ci) * in_h + ih as usize) * in_w + iw as usize;
                            let value = x.data()[index];
                            count += 1;
                            if value > best {
                                best = value;
                                best_index = Some(index);
                            }
                        }
                    }
                    if max {
                        if let Some(index) = best_index {
                            dx[index] += g;
                        }
                    } else if count > 0 {
                        // The loop above intentionally does not know the count
                        // until it finishes, so distribute in a second pass.
                        for kh in 0..kernel[0] {
                            let ih = oh as isize * stride[0] as isize - padding[0] as isize
                                + kh as isize * dilation[0] as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            for kw in 0..kernel[1] {
                                let iw = ow as isize * stride[1] as isize - padding[1] as isize
                                    + kw as isize * dilation[1] as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                let index =
                                    ((ni * c + ci) * in_h + ih as usize) * in_w + iw as usize;
                                dx[index] += g / T::from_usize(count);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(vec![DenseTensor::from_vec(&x_dims, dx)?])
}

type PoolShape = ([usize; 2], [usize; 2], [usize; 2], [usize; 2]);

fn pool_attrs(attrs: &OpAttrs) -> Result<PoolShape> {
    match attrs {
        OpAttrs::Pool2d {
            kernel,
            stride,
            padding,
            dilation,
        } => Ok((*kernel, *stride, *padding, *dilation)),
        _ => Err(Error::execution("pooling missing attributes")),
    }
}

fn unbroadcast<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 1)?;
    let target = out_dims(&call, 0)?;
    let input = &call.inputs[0];
    let in_dims = input.dims();
    if target.len() > in_dims.len() {
        return Err(Error::execution("unbroadcast target has higher rank"));
    }
    if !can_broadcast_to(&target, &in_dims) {
        return Err(Error::InvalidBroadcast {
            lhs: oxide_ml_core::Shape::from_static(&target),
            rhs: oxide_ml_core::Shape::from_static(&in_dims),
        });
    }
    let in_strides = strides_for(&in_dims);
    let target_strides = strides_for(&target);
    let mut data = vec![T::zero(); product(&target)];
    for linear in 0..input.numel() {
        let in_index = unravel(linear, &in_dims, &in_strides);
        let offset = in_dims.len() - target.len();
        let mut target_index = Vec::with_capacity(target.len());
        for axis in 0..target.len() {
            target_index.push(if target[axis] == 1 {
                0
            } else {
                in_index[offset + axis]
            });
        }
        data[dot(&target_index, &target_strides)] += input.data()[linear];
    }
    Ok(vec![DenseTensor::from_vec(&target, data)?])
}

#[cfg(test)]
mod tests {
    use oxide_ml_core::{Device, Shape};
    use oxide_ml_graph::OpAttrs;
    use oxide_ml_kernel::TensorSpec;

    use super::*;

    fn tensor(dims: &[usize], data: &[f32]) -> DenseTensor {
        DenseTensor::from_vec(dims, data.to_vec()).unwrap()
    }

    fn run(
        op: &Op,
        attrs: &OpAttrs,
        inputs: &[DenseTensor],
        shapes: &[Vec<usize>],
    ) -> Vec<DenseTensor> {
        let input_specs = inputs
            .iter()
            .map(|tensor| TensorSpec::contiguous(tensor.dims(), DType::F32, Device::Cpu))
            .collect::<Vec<_>>();
        let output_specs = shapes
            .iter()
            .map(|shape| TensorSpec::contiguous(shape.clone(), DType::F32, Device::Cpu))
            .collect::<Vec<_>>();
        dispatch(KernelCall {
            op,
            attrs,
            inputs,
            input_specs: &input_specs,
            output_specs: &output_specs,
            aliasing: &[],
            workspace: oxide_ml_kernel::WorkspaceSpec::none(),
            context: oxide_ml_kernel::ExecutionContext::default(),
        })
        .unwrap()
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let out = run(
            &Op::Softmax,
            &OpAttrs::Softmax { axis: -1 },
            &[tensor(&[2, 3], &[1.0, 2.0, 3.0, -1.0, 0.0, 1.0])],
            &[vec![2, 3]],
        );
        assert!((out[0].data()[0] + out[0].data()[1] + out[0].data()[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn matmul_2d() {
        let out = run(
            &Op::MatMul,
            &OpAttrs::matmul(false, false),
            &[
                tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]),
            ],
            &[vec![2, 2]],
        );
        assert_eq!(out[0].data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_broadcast_batch() {
        let a = tensor(&[2, 1, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = tensor(&[1, 2, 2], &[1., 0., 0., 1.]);
        let out = run(
            &Op::MatMul,
            &OpAttrs::matmul(false, false),
            &[a, b],
            &[vec![2, 1, 2, 2]],
        );
        assert_eq!(out[0].dims(), vec![2, 1, 2, 2]);
    }

    #[test]
    fn conv2d_identity_kernel() {
        let out = run(
            &Op::Conv2d,
            &OpAttrs::Conv2d {
                stride: [1, 1],
                padding: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
            &[
                tensor(&[1, 1, 3, 3], &[0., 0., 0., 0., 1., 0., 0., 0., 0.]),
                tensor(&[1, 1, 3, 3], &[0., 0., 0., 0., 1., 0., 0., 0., 0.]),
            ],
            &[vec![1, 1, 3, 3]],
        );
        assert_eq!(out[0].get(&[0, 0, 1, 1]).unwrap(), 1.0);
    }

    #[test]
    fn unbroadcast_sums_batch() {
        let out = run(
            &Op::Unbroadcast,
            &OpAttrs::Unbroadcast {
                shape: Shape::from_static([3]),
            },
            &[tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.])],
            &[vec![3]],
        );
        assert_eq!(out[0].data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn backend_capabilities_advertise_scalar_dtype() {
        use oxide_ml_kernel::KernelBackend;
        let caps = CpuBackend::<f32>::new().capabilities();
        assert!(caps.supports(DType::F32));
        assert!(!caps.supports(DType::F64));
        assert_eq!(caps.abi_version, oxide_ml_kernel::KERNEL_ABI_VERSION);
        let caps64 = CpuBackend::<f64>::new().capabilities();
        assert!(caps64.supports(DType::F64));
        assert!(!caps64.supports(DType::F32));
    }

    #[test]
    fn f64_kernel_executes() {
        let inputs = [DenseTensor::from_vec([2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap()];
        let input_specs = [TensorSpec::contiguous(vec![2, 2], DType::F64, Device::Cpu)];
        let output_specs = [TensorSpec::contiguous(vec![2, 2], DType::F64, Device::Cpu)];
        let out = dispatch(KernelCall {
            op: &Op::ReLU,
            attrs: &OpAttrs::None,
            inputs: &inputs,
            input_specs: &input_specs,
            output_specs: &output_specs,
            aliasing: &[],
            workspace: oxide_ml_kernel::WorkspaceSpec::none(),
            context: oxide_ml_kernel::ExecutionContext::default(),
        })
        .unwrap();
        assert_eq!(out[0].data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    fn run_result(
        op: &Op,
        attrs: &OpAttrs,
        inputs: &[DenseTensor],
        shapes: &[Vec<usize>],
    ) -> Result<Vec<DenseTensor>> {
        let input_specs = inputs
            .iter()
            .map(|tensor| TensorSpec::contiguous(tensor.dims(), DType::F32, Device::Cpu))
            .collect::<Vec<_>>();
        let output_specs = shapes
            .iter()
            .map(|shape| TensorSpec::contiguous(shape.clone(), DType::F32, Device::Cpu))
            .collect::<Vec<_>>();
        dispatch(KernelCall {
            op,
            attrs,
            inputs,
            input_specs: &input_specs,
            output_specs: &output_specs,
            aliasing: &[],
            workspace: oxide_ml_kernel::WorkspaceSpec::none(),
            context: oxide_ml_kernel::ExecutionContext::default(),
        })
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn unary_math_kernels_match_hand_computed_values() {
        let out = run(
            &Op::Sqrt,
            &OpAttrs::None,
            &[tensor(&[2], &[4.0, 9.0])],
            &[vec![2]],
        );
        assert_eq!(out[0].data(), &[2.0, 3.0]);

        let out = run(
            &Op::Exp,
            &OpAttrs::None,
            &[tensor(&[2], &[0.0, 1.0])],
            &[vec![2]],
        );
        assert_close(out[0].data(), &[1.0, std::f32::consts::E]);

        let out = run(
            &Op::Log,
            &OpAttrs::None,
            &[tensor(&[2], &[1.0, std::f32::consts::E])],
            &[vec![2]],
        );
        assert_close(out[0].data(), &[0.0, 1.0]);

        let out = run(
            &Op::Abs,
            &OpAttrs::None,
            &[tensor(&[3], &[-2.0, 0.0, 3.5])],
            &[vec![3]],
        );
        assert_eq!(out[0].data(), &[2.0, 0.0, 3.5]);
    }

    #[test]
    fn log_softmax_matches_hand_computed_row() {
        let out = run(
            &Op::LogSoftmax,
            &OpAttrs::LogSoftmax { axis: -1 },
            &[tensor(&[2, 3], &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0])],
            &[vec![2, 3]],
        );
        let row = &out[0].data()[..3];
        let sum_exp: f32 = row.iter().map(|value| value.exp()).sum();
        assert!((sum_exp - 1.0).abs() < 1e-6, "sum of exp = {sum_exp}");
        let log_sum = (1f32.exp() + 2f32.exp() + 3f32.exp()).ln();
        assert_close(row, &[1.0 - log_sum, 2.0 - log_sum, 3.0 - log_sum]);
        // Uniform rows are a constant shift of the log-softmax normalizer.
        let constant = 3f32.ln();
        assert_close(&out[0].data()[3..], &[-constant, -constant, -constant]);
    }

    #[test]
    fn log_softmax_over_first_axis() {
        let out = run(
            &Op::LogSoftmax,
            &OpAttrs::LogSoftmax { axis: 0 },
            &[tensor(&[2, 2], &[0.0, 0.0, 0.0, 0.0])],
            &[vec![2, 2]],
        );
        let expected = [-std::f32::consts::LN_2; 4];
        assert_close(out[0].data(), &expected);
    }

    #[test]
    fn slice_and_slice_backward_round_trip() {
        let input = tensor(&[2, 4], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let attrs = OpAttrs::Slice {
            axis: 1,
            start: 1,
            len: 2,
        };
        let sliced = run(&Op::Slice, &attrs, &[input], &[vec![2, 2]]);
        assert_eq!(sliced[0].data(), &[1.0, 2.0, 5.0, 6.0]);

        let backward_attrs = OpAttrs::SliceBackward {
            axis: 1,
            start: 1,
            len: 2,
            input_shape: Shape::from_static([2, 4]),
        };
        let grad = run(
            &Op::SliceBackward,
            &backward_attrs,
            &[sliced[0].clone()],
            &[vec![2, 4]],
        );
        assert_eq!(grad[0].data(), &[0.0, 1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn embedding_lookup_and_repeated_index_accumulation() {
        let weight = tensor(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let indices = tensor(&[3], &[0., 0., 2.]);
        let embedded = run(
            &Op::Embedding,
            &OpAttrs::None,
            &[weight, indices],
            &[vec![3, 2]],
        );
        assert_eq!(embedded[0].data(), &[1., 2., 1., 2., 5., 6.]);

        let dy = tensor(&[3, 2], &[1., 1., 2., 2., 3., 3.]);
        let indices = tensor(&[3], &[0., 0., 2.]);
        let grad = run(
            &Op::EmbeddingBackward,
            &OpAttrs::EmbeddingBackward { vocab: 3 },
            &[indices, dy],
            &[vec![3, 2]],
        );
        assert_eq!(grad[0].data(), &[3., 3., 0., 0., 3., 3.]);
    }

    #[test]
    fn embedding_rejects_invalid_indices() {
        let weight = tensor(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        for bad in [-1.0f32, 3.0, 1.5, f32::NAN, f32::INFINITY] {
            let indices = tensor(&[1], &[bad]);
            let err = run_result(
                &Op::Embedding,
                &OpAttrs::None,
                &[weight.clone(), indices],
                &[vec![1, 2]],
            )
            .unwrap_err();
            assert!(
                matches!(err, Error::Execution(_)),
                "index {bad} produced {err:?}"
            );
        }
        let indices = tensor(&[1], &[0.0]);
        let grad = run(
            &Op::EmbeddingBackward,
            &OpAttrs::EmbeddingBackward { vocab: 3 },
            &[indices, tensor(&[1, 2], &[1.0, 1.0])],
            &[vec![3, 2]],
        );
        // A valid index is accepted by the backward kernel.
        assert_eq!(grad[0].data(), &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let err = run_result(
            &Op::EmbeddingBackward,
            &OpAttrs::EmbeddingBackward { vocab: 3 },
            &[tensor(&[1], &[3.0]), tensor(&[1, 2], &[1.0, 1.0])],
            &[vec![3, 2]],
        )
        .unwrap_err();
        assert!(matches!(err, Error::Execution(_)));
    }

    #[test]
    fn abs_backward_uses_sign_with_zero_at_origin() {
        let out = run(
            &Op::AbsBackward,
            &OpAttrs::None,
            &[
                tensor(&[3], &[-2.0, 0.0, 3.0]),
                tensor(&[3], &[1.0, 1.0, 1.0]),
            ],
            &[vec![3]],
        );
        assert_eq!(out[0].data(), &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn f64_math_kernels_execute() {
        let inputs = [DenseTensor::from_vec([2], vec![4.0f64, 9.0]).unwrap()];
        let input_specs = [TensorSpec::contiguous(vec![2], DType::F64, Device::Cpu)];
        let output_specs = [TensorSpec::contiguous(vec![2], DType::F64, Device::Cpu)];
        let out = dispatch(KernelCall {
            op: &Op::Sqrt,
            attrs: &OpAttrs::None,
            inputs: &inputs,
            input_specs: &input_specs,
            output_specs: &output_specs,
            aliasing: &[],
            workspace: oxide_ml_kernel::WorkspaceSpec::none(),
            context: oxide_ml_kernel::ExecutionContext::default(),
        })
        .unwrap();
        assert_eq!(out[0].data(), &[2.0, 3.0]);
    }
}

/// Stateless reference backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend<T: Scalar = f32> {
    marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> CpuBackend<T> {
    pub fn new() -> Self {
        CpuBackend {
            marker: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> oxide_ml_kernel::KernelBackend<T> for CpuBackend<T> {
    fn name(&self) -> &'static str {
        "cpu-reference"
    }

    fn capabilities(&self) -> oxide_ml_kernel::BackendCapabilities {
        oxide_ml_kernel::BackendCapabilities {
            abi_version: oxide_ml_kernel::KERNEL_ABI_VERSION,
            capability_version: oxide_ml_kernel::BACKEND_CAPABILITY_VERSION,
            dtypes: match T::DTYPE {
                DType::F32 => &[DType::F32],
                DType::F64 => &[DType::F64],
                _ => &[],
            },
            supports_workspace: false,
            supports_streams: false,
        }
    }

    fn execute(&self, call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
        dispatch(call)
    }
}

/// Fused Gemma-style mixture-of-experts block.
///
/// Inputs: hidden, router scale, per-expert scale, router projection,
/// gate/up experts, down experts, expert pre-norm weight.
fn moe<T: Scalar>(call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>> {
    expect_inputs(&call, 7)?;
    let (num_experts, top_k, intermediate, eps) = match call.attrs {
        OpAttrs::MoE {
            num_experts,
            top_k,
            intermediate,
            eps,
        } => (*num_experts, *top_k, *intermediate, *eps),
        _ => return Err(Error::execution("MoE requires OpAttrs::MoE")),
    };
    let dims = out_dims(&call, 0)?;
    let rank = dims.len();
    if rank < 2 {
        return Err(Error::execution("MoE output must be rank >= 2"));
    }
    let hidden_size = dims[rank - 1];
    let rows: usize = dims[..rank - 1].iter().product();
    let hidden = call.inputs[0].data();
    let router_scale = call.inputs[1].data();
    let per_expert_scale = call.inputs[2].data();
    let router = call.inputs[3].data();
    let gate_up = call.inputs[4].data();
    let down = call.inputs[5].data();
    let expert_norm = call.inputs[6].data();
    if router_scale.len() != hidden_size
        || per_expert_scale.len() != num_experts
        || router.len() != num_experts * hidden_size
        || gate_up.len() != num_experts * 2 * intermediate * hidden_size
        || down.len() != num_experts * hidden_size * intermediate
        || expert_norm.len() != hidden_size
    {
        return Err(Error::execution(
            "MoE weight shapes do not match its attributes",
        ));
    }
    let scale_factor = 1.0 / (hidden_size as f64).sqrt();
    let gelu = |value: f64| {
        0.5 * value
            * (1.0 + (0.797_884_560_802_865_4 * (value + 0.044_715 * value * value * value)).tanh())
    };
    let mut output = Vec::with_capacity(rows * hidden_size);
    let mut scores = vec![0.0f64; num_experts];
    for row in 0..rows {
        let x = &hidden[row * hidden_size..(row + 1) * hidden_size];
        let mut mean_square = 0.0f64;
        for value in x {
            let value = value.to_f64();
            mean_square += value * value;
        }
        mean_square /= hidden_size as f64;
        let inv = 1.0 / (mean_square + f64::from(eps)).sqrt();

        // Router: no-scale RMSNorm, then scale and H^-0.5.
        for expert in 0..num_experts {
            let weights = &router[expert * hidden_size..(expert + 1) * hidden_size];
            let mut score = 0.0f64;
            for index in 0..hidden_size {
                score += weights[index].to_f64()
                    * x[index].to_f64()
                    * inv
                    * router_scale[index].to_f64();
            }
            scores[expert] = score * scale_factor;
        }
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for score in &mut scores {
            *score = (*score - max).exp();
            sum += *score;
        }
        for score in &mut scores {
            *score /= sum;
        }
        let mut order: Vec<usize> = (0..num_experts).collect();
        order.sort_by(|left, right| {
            scores[*right]
                .partial_cmp(&scores[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        let selected = &order[..top_k];
        let weight_sum: f64 = selected.iter().map(|expert| scores[*expert]).sum();

        // Experts use their own pre-normalization.
        let mut expert_input = vec![0.0f64; hidden_size];
        for index in 0..hidden_size {
            expert_input[index] = x[index].to_f64() * inv * expert_norm[index].to_f64();
        }
        let mut accumulator = vec![0.0f64; hidden_size];
        for expert in selected {
            let weight = scores[*expert] / weight_sum * per_expert_scale[*expert].to_f64();
            let gate_base = expert * 2 * intermediate * hidden_size;
            let mut gate = vec![0.0f64; intermediate];
            let mut up = vec![0.0f64; intermediate];
            for index in 0..intermediate {
                let gate_offset = gate_base + index * hidden_size;
                let up_offset = gate_base + (intermediate + index) * hidden_size;
                let mut gate_value = 0.0f64;
                let mut up_value = 0.0f64;
                for h in 0..hidden_size {
                    gate_value += gate_up[gate_offset + h].to_f64() * expert_input[h];
                    up_value += gate_up[up_offset + h].to_f64() * expert_input[h];
                }
                gate[index] = gate_value;
                up[index] = up_value;
            }
            let mut product = vec![0.0f64; intermediate];
            for index in 0..intermediate {
                product[index] = gelu(gate[index]) * up[index];
            }
            let down_base = expert * hidden_size * intermediate;
            for (h, acc) in accumulator.iter_mut().enumerate() {
                let offset = down_base + h * intermediate;
                let mut value = 0.0f64;
                for index in 0..intermediate {
                    value += down[offset + index].to_f64() * product[index];
                }
                *acc += weight * value;
            }
        }
        output.extend(accumulator.into_iter().map(T::from_f64));
    }
    Ok(vec![DenseTensor::from_vec(&dims, output)?])
}
