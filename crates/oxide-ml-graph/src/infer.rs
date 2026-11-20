use oxide_ml_core::{DType, Dim, Error, Result, Shape};

use crate::ir::{ActivationKind, Op, OpAttrs, Reduction, ValueInfo};

#[derive(Clone, Debug)]
pub struct Inferred {
    pub shape: Shape,
    pub dtype: DType,
}

impl Inferred {
    fn new(shape: Shape, dtype: DType) -> Self {
        Inferred { shape, dtype }
    }
}

/// Validate an operation and infer every output shape.
pub fn infer_outputs(op: &Op, attrs: &OpAttrs, inputs: &[ValueInfo]) -> Result<Vec<Inferred>> {
    match op {
        Op::Add | Op::AddRelu | Op::Sub | Op::Mul | Op::Div => {
            expect_inputs(op, inputs, 2)?;
            let dtype = broadcast_dtype(op, inputs)?;
            let shape = inputs[0].shape.broadcast(&inputs[1].shape)?;
            Ok(vec![Inferred::new(shape, dtype)])
        }
        Op::Neg
        | Op::ReLU
        | Op::GELU
        | Op::Sigmoid
        | Op::Tanh
        | Op::Sqrt
        | Op::Exp
        | Op::Log
        | Op::Abs => {
            expect_inputs(op, inputs, 1)?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::MatMul => {
            expect_inputs(op, inputs, 2)?;
            let (transpose_a, transpose_b) = match attrs {
                OpAttrs::MatMul {
                    transpose_a,
                    transpose_b,
                } => (*transpose_a, *transpose_b),
                _ => {
                    return Err(Error::invalid_graph(
                        "MatMul requires OpAttrs::MatMul".to_string(),
                    ));
                }
            };
            let dtype = same_dtype(op, inputs)?;
            Ok(vec![Inferred::new(
                matmul_shape(&inputs[0].shape, &inputs[1].shape, transpose_a, transpose_b)?,
                dtype,
            )])
        }
        Op::Softmax => {
            expect_inputs(op, inputs, 1)?;
            let axis = match attrs {
                OpAttrs::Softmax { axis } => *axis,
                _ => {
                    return Err(Error::invalid_graph(
                        "Softmax requires OpAttrs::Softmax".to_string(),
                    ));
                }
            };
            normalize_axis(axis, inputs[0].shape.rank())?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::LogSoftmax => {
            expect_inputs(op, inputs, 1)?;
            let axis = match attrs {
                OpAttrs::LogSoftmax { axis } => *axis,
                _ => {
                    return Err(Error::invalid_graph(
                        "LogSoftmax requires OpAttrs::LogSoftmax".to_string(),
                    ));
                }
            };
            normalize_axis(axis, inputs[0].shape.rank())?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::Slice => {
            expect_inputs(op, inputs, 1)?;
            let (axis, start, len) = match attrs {
                OpAttrs::Slice { axis, start, len } => (*axis, *start, *len),
                _ => {
                    return Err(Error::invalid_graph(
                        "Slice requires OpAttrs::Slice".to_string(),
                    ));
                }
            };
            let input_shape = &inputs[0].shape;
            slice_input_dim(input_shape, axis, start, len)?;
            let mut dims = input_shape.dims().to_vec();
            dims[axis] = Dim::Static(len);
            Ok(vec![Inferred::new(Shape::new(dims), inputs[0].dtype)])
        }
        Op::Embedding => {
            expect_inputs(op, inputs, 2)?;
            let dtype = same_dtype(op, inputs)?;
            let weight = &inputs[0].shape;
            if weight.rank() != 2 {
                return Err(Error::RankMismatch {
                    context: "Embedding weight".into(),
                    expected: 2,
                    got: weight.rank(),
                });
            }
            let dim = weight.dim(1).and_then(Dim::static_usize).ok_or_else(|| {
                Error::invalid_shape(
                    "embedding",
                    format!("weight shape {weight} must have a static embedding dimension"),
                )
            })?;
            let mut dims = inputs[1].shape.dims().to_vec();
            dims.push(Dim::Static(dim));
            Ok(vec![Inferred::new(Shape::new(dims), dtype)])
        }
        Op::LayerNorm => {
            expect_inputs(op, inputs, 3)?;
            let (normalized_shape, eps) = match attrs {
                OpAttrs::LayerNorm {
                    normalized_shape,
                    eps,
                } => (normalized_shape, *eps),
                _ => {
                    return Err(Error::invalid_graph(
                        "LayerNorm requires OpAttrs::LayerNorm".to_string(),
                    ));
                }
            };
            validate_layer_norm(inputs, normalized_shape, eps)?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::Reshape => {
            expect_inputs(op, inputs, 1)?;
            let target = match attrs {
                OpAttrs::Reshape { shape } => shape.clone(),
                _ => {
                    return Err(Error::invalid_graph(
                        "Reshape requires OpAttrs::Reshape".to_string(),
                    ));
                }
            };
            if let (Some(from), Some(to)) = (inputs[0].shape.numel_static(), target.numel_static())
                && from != to
            {
                return Err(Error::invalid_shape(
                    "reshape",
                    format!("cannot reshape {} elements into {target}", from),
                ));
            }
            Ok(vec![Inferred::new(target, inputs[0].dtype)])
        }
        Op::Transpose => {
            expect_inputs(op, inputs, 1)?;
            let axes = match attrs {
                OpAttrs::Transpose { axes } => axes,
                _ => {
                    return Err(Error::invalid_graph(
                        "Transpose requires OpAttrs::Transpose".to_string(),
                    ));
                }
            };
            Ok(vec![Inferred::new(
                transpose_shape(&inputs[0].shape, axes)?,
                inputs[0].dtype,
            )])
        }
        Op::BroadcastTo => {
            expect_inputs(op, inputs, 1)?;
            let target = match attrs {
                OpAttrs::BroadcastTo { shape } => shape.clone(),
                _ => {
                    return Err(Error::invalid_graph(
                        "BroadcastTo requires OpAttrs::BroadcastTo".to_string(),
                    ));
                }
            };
            let broadcast = inputs[0].shape.broadcast(&target)?;
            if !broadcast.is_compatible(&target) {
                return Err(Error::invalid_graph(format!(
                    "BroadcastTo target {target} is not equal to inferred shape {broadcast}"
                )));
            }
            Ok(vec![Inferred::new(target, inputs[0].dtype)])
        }
        Op::ReduceSum | Op::ReduceMean => {
            expect_inputs(op, inputs, 1)?;
            let (axes, keep_dims) = match attrs {
                OpAttrs::Reduce { axes, keep_dims } => (axes, *keep_dims),
                _ => {
                    return Err(Error::invalid_graph(
                        "reduction requires OpAttrs::Reduce".to_string(),
                    ));
                }
            };
            let axes = normalize_axes(axes, inputs[0].shape.rank())?;
            Ok(vec![Inferred::new(
                reduce_shape(&inputs[0].shape, &axes, keep_dims),
                inputs[0].dtype,
            )])
        }
        Op::Conv2d => {
            let (stride, padding, dilation, groups) = match attrs {
                OpAttrs::Conv2d {
                    stride,
                    padding,
                    dilation,
                    groups,
                } => (*stride, *padding, *dilation, *groups),
                _ => {
                    return Err(Error::invalid_graph(
                        "Conv2d requires OpAttrs::Conv2d".to_string(),
                    ));
                }
            };
            if !(2..=3).contains(&inputs.len()) {
                return Err(Error::invalid_graph(format!(
                    "Conv2d expects input, weight and optional bias; got {} operands",
                    inputs.len()
                )));
            }
            let dtype = same_dtype(op, &inputs[..2])?;
            let shape = conv2d_shape(
                &inputs[0].shape,
                &inputs[1].shape,
                inputs.get(2).map(|value| &value.shape),
                stride,
                padding,
                dilation,
                groups,
            )?;
            Ok(vec![Inferred::new(shape, dtype)])
        }
        Op::MaxPool2d | Op::AvgPool2d => {
            expect_inputs(op, inputs, 1)?;
            let (kernel, stride, padding, dilation) = match attrs {
                OpAttrs::Pool2d {
                    kernel,
                    stride,
                    padding,
                    dilation,
                } => (*kernel, *stride, *padding, *dilation),
                _ => {
                    return Err(Error::invalid_graph(
                        "pooling requires OpAttrs::Pool2d".to_string(),
                    ));
                }
            };
            Ok(vec![Inferred::new(
                pool2d_shape(&inputs[0].shape, kernel, stride, padding, dilation)?,
                inputs[0].dtype,
            )])
        }
        Op::MSELoss => {
            expect_inputs(op, inputs, 2)?;
            if inputs[0].shape != inputs[1].shape {
                return Err(Error::shape_mismatch(
                    "MSELoss operands must have identical shapes",
                    inputs[0].shape.clone(),
                    inputs[1].shape.clone(),
                ));
            }
            let _ = same_dtype(op, inputs)?;
            validate_loss_attrs(attrs)?;
            Ok(vec![Inferred::new(Shape::scalar(), inputs[0].dtype)])
        }
        Op::CrossEntropyLoss => {
            expect_inputs(op, inputs, 2)?;
            let logits = &inputs[0].shape;
            let targets = &inputs[1].shape;
            if logits.rank() != 2 {
                return Err(Error::RankMismatch {
                    context: "CrossEntropyLoss logits".into(),
                    expected: 2,
                    got: logits.rank(),
                });
            }
            if targets.rank() != 1 {
                return Err(Error::RankMismatch {
                    context: "CrossEntropyLoss targets (class indices)".into(),
                    expected: 1,
                    got: targets.rank(),
                });
            }
            let _ = same_dtype(op, inputs)?;
            if let (Some(n), Some(m)) = (
                logits.dim(0).and_then(Dim::static_usize),
                targets.dim(0).and_then(Dim::static_usize),
            ) && n != m
            {
                return Err(Error::shape_mismatch(
                    "CrossEntropyLoss batch dimension",
                    Shape::from_static([n]),
                    Shape::from_static([m]),
                ));
            }
            validate_loss_attrs(attrs)?;
            Ok(vec![Inferred::new(Shape::scalar(), inputs[0].dtype)])
        }
        Op::Cast => {
            expect_inputs(op, inputs, 1)?;
            let target = match attrs {
                OpAttrs::Cast { dtype } => *dtype,
                _ => {
                    return Err(Error::invalid_graph(
                        "Cast requires OpAttrs::Cast".to_string(),
                    ));
                }
            };
            Ok(vec![Inferred::new(inputs[0].shape.clone(), target)])
        }
        Op::Concat => {
            if inputs.is_empty() {
                return Err(Error::invalid_graph(
                    "Concat requires at least one input".to_string(),
                ));
            }
            let axis = match attrs {
                OpAttrs::Concat { axis } => normalize_axis(*axis as i32, inputs[0].shape.rank())?,
                _ => {
                    return Err(Error::invalid_graph(
                        "Concat requires OpAttrs::Concat".to_string(),
                    ));
                }
            };
            let dtype = same_dtype(op, inputs)?;
            let rank = inputs[0].shape.rank();
            let mut total = 0usize;
            for (index, input) in inputs.iter().enumerate() {
                if input.shape.rank() != rank {
                    return Err(Error::RankMismatch {
                        context: format!("Concat input {index}"),
                        expected: rank,
                        got: input.shape.rank(),
                    });
                }
                for axis_index in 0..rank {
                    if axis_index == axis {
                        continue;
                    }
                    let expected = inputs[0].shape.dim(axis_index);
                    let got = input.shape.dim(axis_index);
                    if let (Some(a), Some(b)) = (expected, got)
                        && !a.is_compatible(b)
                    {
                        return Err(Error::shape_mismatch(
                            format!("Concat input {index} axis {axis_index}"),
                            inputs[0].shape.clone(),
                            input.shape.clone(),
                        ));
                    }
                }
                total += input
                    .shape
                    .dim(axis)
                    .and_then(Dim::static_usize)
                    .ok_or_else(|| {
                        Error::invalid_shape(
                            "Concat",
                            "the concatenated axis must be static".to_string(),
                        )
                    })?;
            }
            let mut dims = inputs[0].shape.dims().to_vec();
            dims[axis] = Dim::Static(total);
            Ok(vec![Inferred::new(Shape::new(dims), dtype)])
        }
        Op::Split => {
            expect_inputs(op, inputs, 1)?;
            let (axis, sizes) = match attrs {
                OpAttrs::Split { axis, sizes } => (*axis, sizes),
                _ => {
                    return Err(Error::invalid_graph(
                        "Split requires OpAttrs::Split".to_string(),
                    ));
                }
            };
            if sizes.is_empty() {
                return Err(Error::invalid_graph(
                    "Split requires at least one output size".to_string(),
                ));
            }
            let rank = inputs[0].shape.rank();
            let axis = normalize_axis(axis as i32, rank)?;
            let total: usize = sizes.iter().sum();
            let axis_dim = inputs[0]
                .shape
                .dim(axis)
                .and_then(Dim::static_usize)
                .ok_or_else(|| {
                    Error::invalid_shape("Split", "the split axis must be static".to_string())
                })?;
            if total != axis_dim {
                return Err(Error::invalid_shape(
                    "Split",
                    format!("sizes {sizes:?} sum to {total}, but axis {axis} has {axis_dim}"),
                ));
            }
            let mut outputs = Vec::with_capacity(sizes.len());
            for size in sizes {
                let mut dims = inputs[0].shape.dims().to_vec();
                dims[axis] = Dim::Static(*size);
                outputs.push(Inferred::new(Shape::new(dims), inputs[0].dtype));
            }
            Ok(outputs)
        }
        Op::MulAdd => {
            expect_inputs(op, inputs, 3)?;
            let dtype = same_dtype(op, inputs)?;
            let shape = inputs[0]
                .shape
                .broadcast(&inputs[1].shape)?
                .broadcast(&inputs[2].shape)?;
            Ok(vec![Inferred::new(shape, dtype)])
        }
        Op::MatMulBias => {
            expect_inputs(op, inputs, 3)?;
            let (transpose_a, transpose_b) = match attrs {
                OpAttrs::MatMul {
                    transpose_a,
                    transpose_b,
                } => (*transpose_a, *transpose_b),
                _ => {
                    return Err(Error::invalid_graph(
                        "MatMulBias requires OpAttrs::MatMul".to_string(),
                    ));
                }
            };
            let dtype = same_dtype(op, inputs)?;
            let output =
                matmul_shape(&inputs[0].shape, &inputs[1].shape, transpose_a, transpose_b)?;
            let broadcast = inputs[2].shape.broadcast(&output)?;
            if broadcast != output {
                return Err(Error::shape_mismatch(
                    "MatMulBias bias",
                    output.clone(),
                    inputs[2].shape.clone(),
                ));
            }
            Ok(vec![Inferred::new(output, dtype)])
        }
        Op::MoE => {
            let (num_experts, top_k, intermediate, _eps) = match attrs {
                OpAttrs::MoE {
                    num_experts,
                    top_k,
                    intermediate,
                    eps,
                } => (*num_experts, *top_k, *intermediate, *eps),
                _ => {
                    return Err(Error::invalid_graph(
                        "MoE requires OpAttrs::MoE".to_string(),
                    ));
                }
            };
            if inputs.len() != 7 {
                return Err(Error::invalid_graph(format!(
                    "MoE expects 7 inputs, got {}",
                    inputs.len()
                )));
            }
            let hidden = &inputs[0];
            let rank = hidden.shape.rank();
            if rank < 2 {
                return Err(Error::invalid_graph("MoE input must be rank >= 2"));
            }
            let hidden_size = hidden
                .shape
                .dim(rank - 1)
                .and_then(|dim| dim.static_usize())
                .ok_or_else(|| Error::invalid_graph("MoE hidden size must be static"))?;
            let expect = |index: usize, dims: &[usize]| -> Result<()> {
                let actual = inputs[index].shape.static_dims().ok_or_else(|| {
                    Error::invalid_graph(format!("MoE input {index} must be static"))
                })?;
                if actual != dims {
                    return Err(Error::shape_mismatch(
                        format!("MoE input {index}"),
                        Shape::from_static(dims),
                        Shape::from_static(&actual),
                    ));
                }
                Ok(())
            };
            expect(1, &[hidden_size])?;
            expect(2, &[num_experts])?;
            expect(3, &[num_experts, hidden_size])?;
            expect(4, &[num_experts, 2 * intermediate, hidden_size])?;
            expect(5, &[num_experts, hidden_size, intermediate])?;
            expect(6, &[hidden_size])?;
            if top_k == 0 || top_k > num_experts {
                return Err(Error::invalid_graph(format!(
                    "MoE top_k {top_k} must be in 1..={num_experts}"
                )));
            }
            Ok(vec![Inferred::new(hidden.shape.clone(), hidden.dtype)])
        }
        Op::Custom => {
            let (input_count, outputs, dtype) = match attrs {
                OpAttrs::Custom {
                    input_count,
                    outputs,
                    dtype,
                    ..
                } => (*input_count, outputs, *dtype),
                _ => {
                    return Err(Error::invalid_graph(
                        "Custom requires OpAttrs::Custom".to_string(),
                    ));
                }
            };
            if inputs.len() != input_count {
                return Err(Error::invalid_graph(format!(
                    "custom kernel expects {input_count} inputs, got {}",
                    inputs.len()
                )));
            }
            if outputs.is_empty() {
                return Err(Error::invalid_graph(
                    "custom kernel must declare at least one output",
                ));
            }
            Ok(outputs
                .iter()
                .map(|shape| Inferred::new(shape.clone(), dtype))
                .collect())
        }
        Op::Constant => {
            if !inputs.is_empty() {
                return Err(Error::invalid_graph(
                    "Constant takes no operands".to_string(),
                ));
            }
            let value = match attrs {
                OpAttrs::Constant { value } => value,
                _ => {
                    return Err(Error::invalid_graph(
                        "Constant requires OpAttrs::Constant".to_string(),
                    ));
                }
            };
            value.validate()?;
            Ok(vec![Inferred::new(value.shape.clone(), value.dtype())])
        }
        Op::ConstantScalar => {
            if !inputs.is_empty() {
                return Err(Error::invalid_graph(
                    "ConstantScalar takes no operands".to_string(),
                ));
            }
            match attrs {
                OpAttrs::ConstantScalar { .. } => {
                    Ok(vec![Inferred::new(Shape::scalar(), DType::F32)])
                }
                _ => Err(Error::invalid_graph(
                    "ConstantScalar requires OpAttrs::ConstantScalar".to_string(),
                )),
            }
        }
        Op::AddReluBackward => {
            expect_inputs(op, inputs, 3)?;
            let dtype = same_dtype(op, &inputs[..2])?;
            if inputs[2].dtype != dtype {
                return Err(Error::DTypeMismatch {
                    context: "AddReluBackward upstream".into(),
                    expected: dtype,
                    got: inputs[2].dtype,
                });
            }
            let shape = inputs[0].shape.broadcast(&inputs[1].shape)?;
            if !shape.is_compatible(&inputs[2].shape) {
                return Err(Error::shape_mismatch(
                    "AddReluBackward upstream",
                    shape,
                    inputs[2].shape.clone(),
                ));
            }
            Ok(vec![Inferred::new(shape, dtype)])
        }
        Op::ActivationBackward(kind) => {
            expect_inputs(op, inputs, 2)?;
            match attrs {
                OpAttrs::ActivationBackward { kind: attr_kind } if attr_kind == kind => {}
                _ => {
                    return Err(Error::invalid_graph(
                        "ActivationBackward op/attrs mismatch".to_string(),
                    ));
                }
            }
            ensure_same_shape(op, &inputs[0].shape, &inputs[1].shape)?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::SoftmaxBackward => {
            expect_inputs(op, inputs, 2)?;
            ensure_same_shape(op, &inputs[0].shape, &inputs[1].shape)?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::LayerNormBackward => {
            expect_inputs(op, inputs, 3)?;
            let (normalized_shape, eps) = match attrs {
                OpAttrs::LayerNorm {
                    normalized_shape,
                    eps,
                } => (normalized_shape, *eps),
                _ => {
                    return Err(Error::invalid_graph(
                        "LayerNormBackward requires OpAttrs::LayerNorm".to_string(),
                    ));
                }
            };
            let gamma = &inputs[1];
            let dy = &inputs[2];
            validate_layer_norm(&inputs[..2], normalized_shape, eps)?;
            ensure_same_shape(op, &inputs[0].shape, &dy.shape)?;
            Ok(vec![
                Inferred::new(inputs[0].shape.clone(), inputs[0].dtype),
                Inferred::new(gamma.shape.clone(), gamma.dtype),
                Inferred::new(gamma.shape.clone(), gamma.dtype),
            ])
        }
        Op::MSELossBackward => {
            expect_inputs(op, inputs, 3)?;
            if inputs[0].shape != inputs[1].shape {
                return Err(Error::shape_mismatch(
                    "MSELossBackward operands",
                    inputs[0].shape.clone(),
                    inputs[1].shape.clone(),
                ));
            }
            let _ = same_dtype(op, &inputs[..2])?;
            validate_loss_attrs(attrs)?;
            validate_loss_attrs_scalar(inputs)?;
            Ok(vec![
                Inferred::new(inputs[0].shape.clone(), inputs[0].dtype),
                Inferred::new(inputs[1].shape.clone(), inputs[1].dtype),
            ])
        }
        Op::CrossEntropyLossBackward => {
            expect_inputs(op, inputs, 3)?;
            if inputs[0].shape.rank() != 2 || inputs[1].shape.rank() != 1 {
                return Err(Error::invalid_graph(
                    "CrossEntropyLossBackward expects logits [N,C], targets [N], dy scalar"
                        .to_string(),
                ));
            }
            if let (Some(logits_n), Some(targets_n)) = (
                inputs[0].shape.dim(0).and_then(Dim::static_usize),
                inputs[1].shape.dim(0).and_then(Dim::static_usize),
            ) && logits_n != targets_n
            {
                return Err(Error::shape_mismatch(
                    "CrossEntropyLossBackward batch dimension",
                    Shape::from_static([logits_n]),
                    Shape::from_static([targets_n]),
                ));
            }
            let _ = same_dtype(op, &inputs[..2])?;
            validate_loss_attrs(attrs)?;
            validate_loss_attrs_scalar(inputs)?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::Conv2dBackward => {
            expect_inputs(op, inputs, 3)?;
            let (stride, padding, dilation, groups) = match attrs {
                OpAttrs::Conv2d {
                    stride,
                    padding,
                    dilation,
                    groups,
                } => (*stride, *padding, *dilation, *groups),
                _ => {
                    return Err(Error::invalid_graph(
                        "Conv2dBackward requires OpAttrs::Conv2d".to_string(),
                    ));
                }
            };
            let weight_shape = &inputs[1].shape;
            if weight_shape.rank() != 4 {
                return Err(Error::RankMismatch {
                    context: "Conv2dBackward weight".into(),
                    expected: 4,
                    got: weight_shape.rank(),
                });
            }
            let out_channels = weight_shape
                .dim(0)
                .and_then(Dim::static_usize)
                .ok_or_else(|| Error::invalid_graph("Conv2d weight must have static channels"))?;
            // Validate that dy is compatible with the forward output shape
            // implied by x and w; this catches transposed argument mistakes.
            conv2d_shape(
                &inputs[0].shape,
                weight_shape,
                None,
                stride,
                padding,
                dilation,
                groups,
            )?;
            Ok(vec![
                Inferred::new(inputs[0].shape.clone(), inputs[0].dtype),
                Inferred::new(weight_shape.clone(), inputs[1].dtype),
                Inferred::new(Shape::from_static([out_channels]), inputs[1].dtype),
            ])
        }
        Op::MaxPool2dBackward | Op::AvgPool2dBackward => {
            expect_inputs(op, inputs, 2)?;
            let (kernel, stride, padding, dilation) = match attrs {
                OpAttrs::Pool2d {
                    kernel,
                    stride,
                    padding,
                    dilation,
                } => (*kernel, *stride, *padding, *dilation),
                _ => {
                    return Err(Error::invalid_graph(
                        "pooling backward requires OpAttrs::Pool2d".to_string(),
                    ));
                }
            };
            pool2d_shape(&inputs[0].shape, kernel, stride, padding, dilation)?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::AbsBackward => {
            expect_inputs(op, inputs, 2)?;
            ensure_same_shape(op, &inputs[0].shape, &inputs[1].shape)?;
            let _ = same_dtype(op, inputs)?;
            Ok(vec![Inferred::new(
                inputs[0].shape.clone(),
                inputs[0].dtype,
            )])
        }
        Op::SliceBackward => {
            expect_inputs(op, inputs, 1)?;
            let (axis, start, len, input_shape) = match attrs {
                OpAttrs::SliceBackward {
                    axis,
                    start,
                    len,
                    input_shape,
                } => (*axis, *start, *len, input_shape),
                _ => {
                    return Err(Error::invalid_graph(
                        "SliceBackward requires OpAttrs::SliceBackward".to_string(),
                    ));
                }
            };
            slice_input_dim(input_shape, axis, start, len)?;
            let mut sliced = input_shape.dims().to_vec();
            sliced[axis] = Dim::Static(len);
            let sliced = Shape::new(sliced);
            if !inputs[0].shape.is_compatible(&sliced) {
                return Err(Error::shape_mismatch(
                    "SliceBackward upstream gradient",
                    sliced,
                    inputs[0].shape.clone(),
                ));
            }
            Ok(vec![Inferred::new(input_shape.clone(), inputs[0].dtype)])
        }
        Op::EmbeddingBackward => {
            expect_inputs(op, inputs, 2)?;
            let vocab = match attrs {
                OpAttrs::EmbeddingBackward { vocab } => *vocab,
                _ => {
                    return Err(Error::invalid_graph(
                        "EmbeddingBackward requires OpAttrs::EmbeddingBackward".to_string(),
                    ));
                }
            };
            let _ = same_dtype(op, inputs)?;
            let indices = &inputs[0].shape;
            let dy = &inputs[1].shape;
            if dy.rank() != indices.rank() + 1 {
                return Err(Error::RankMismatch {
                    context: "EmbeddingBackward upstream gradient".into(),
                    expected: indices.rank() + 1,
                    got: dy.rank(),
                });
            }
            let dy_index_dims = Shape::new(dy.dims()[..indices.rank()].to_vec());
            if !dy_index_dims.is_compatible(indices) {
                return Err(Error::shape_mismatch(
                    "EmbeddingBackward indices prefix",
                    indices.clone(),
                    dy_index_dims,
                ));
            }
            let dim = dy
                .dim(indices.rank())
                .and_then(Dim::static_usize)
                .ok_or_else(|| {
                    Error::invalid_shape(
                        "embedding_backward",
                        format!("upstream shape {dy} must have a static embedding dimension"),
                    )
                })?;
            Ok(vec![Inferred::new(
                Shape::from_static([vocab, dim]),
                inputs[0].dtype,
            )])
        }
        Op::Unbroadcast => {
            expect_inputs(op, inputs, 1)?;
            match attrs {
                OpAttrs::Unbroadcast { shape } => {
                    Ok(vec![Inferred::new(shape.clone(), inputs[0].dtype)])
                }
                _ => Err(Error::invalid_graph(
                    "Unbroadcast requires OpAttrs::Unbroadcast".to_string(),
                )),
            }
        }
    }
}

fn expect_inputs(op: &Op, inputs: &[ValueInfo], count: usize) -> Result<()> {
    if inputs.len() != count {
        return Err(Error::invalid_graph(format!(
            "{} expects {count} operands, got {}",
            op.name(),
            inputs.len()
        )));
    }
    Ok(())
}

fn same_dtype(op: &Op, inputs: &[ValueInfo]) -> Result<DType> {
    let dtype = inputs
        .first()
        .map(|value| value.dtype)
        .ok_or_else(|| Error::invalid_graph(format!("{} has no operands", op.name())))?;
    for input in inputs {
        if input.dtype != dtype {
            return Err(Error::DTypeMismatch {
                context: op.name().to_string(),
                expected: dtype,
                got: input.dtype,
            });
        }
    }
    Ok(dtype)
}

fn broadcast_dtype(op: &Op, inputs: &[ValueInfo]) -> Result<DType> {
    same_dtype(op, inputs)
}

fn ensure_same_shape(op: &Op, lhs: &Shape, rhs: &Shape) -> Result<()> {
    if lhs != rhs {
        return Err(Error::shape_mismatch(
            format!("{} operand shapes", op.name()),
            lhs.clone(),
            rhs.clone(),
        ));
    }
    Ok(())
}

/// Validate a contiguous `[start, start + len)` range along `axis` and return
/// the static size of that dimension.
fn slice_input_dim(shape: &Shape, axis: usize, start: usize, len: usize) -> Result<usize> {
    let rank = shape.rank();
    if axis >= rank {
        return Err(Error::invalid_shape(
            "slice",
            format!("axis {axis} is out of range for rank {rank}"),
        ));
    }
    let dim = shape.dim(axis).and_then(Dim::static_usize).ok_or_else(|| {
        Error::invalid_shape(
            "slice",
            format!("axis {axis} of shape {shape} must be static"),
        )
    })?;
    let end = start.checked_add(len).ok_or_else(|| {
        Error::invalid_shape(
            "slice",
            format!("start {start} plus length {len} overflows usize"),
        )
    })?;
    if end > dim {
        return Err(Error::invalid_shape(
            "slice",
            format!("range [{start}, {end}) exceeds dimension size {dim}"),
        ));
    }
    Ok(dim)
}

fn validate_loss_attrs(attrs: &OpAttrs) -> Result<()> {
    match attrs {
        OpAttrs::Loss { .. } => Ok(()),
        _ => Err(Error::invalid_graph(
            "loss requires OpAttrs::Loss".to_string(),
        )),
    }
}

fn validate_loss_attrs_scalar(inputs: &[ValueInfo]) -> Result<()> {
    if let Some(dy) = inputs.get(2)
        && !dy.shape.is_scalar()
    {
        return Err(Error::shape_mismatch(
            "loss upstream gradient must be scalar",
            Shape::scalar(),
            dy.shape.clone(),
        ));
    }
    Ok(())
}

fn validate_layer_norm(inputs: &[ValueInfo], normalized_shape: &[usize], eps: f32) -> Result<()> {
    if inputs.len() < 2 {
        return Err(Error::invalid_graph(
            "LayerNorm requires x, gamma and beta".to_string(),
        ));
    }
    if normalized_shape.is_empty() {
        return Err(Error::invalid_graph(
            "LayerNorm normalized_shape must not be empty".to_string(),
        ));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::invalid_graph(format!(
            "LayerNorm eps must be positive and finite, got {eps}"
        )));
    }
    let x = &inputs[0].shape;
    if x.rank() < normalized_shape.len() {
        return Err(Error::shape_mismatch(
            "LayerNorm input rank",
            Shape::from_static(normalized_shape),
            x.clone(),
        ));
    }
    let offset = x.rank() - normalized_shape.len();
    for (index, expected) in normalized_shape.iter().enumerate() {
        let actual = x.dim(offset + index).and_then(Dim::static_usize);
        if actual != Some(*expected) {
            return Err(Error::shape_mismatch(
                "LayerNorm trailing dimensions",
                Shape::from_static(normalized_shape),
                x.clone(),
            ));
        }
    }
    for input in &inputs[1..] {
        if input.shape != Shape::from_static(normalized_shape) {
            return Err(Error::shape_mismatch(
                "LayerNorm affine parameter",
                Shape::from_static(normalized_shape),
                input.shape.clone(),
            ));
        }
    }
    Ok(())
}

pub fn normalize_axis(axis: i32, rank: usize) -> Result<usize> {
    if rank == 0 {
        return Err(Error::InvalidAxis { axis, rank });
    }
    let normalized = if axis < 0 { axis + rank as i32 } else { axis };
    if normalized < 0 || normalized as usize >= rank {
        return Err(Error::InvalidAxis { axis, rank });
    }
    Ok(normalized as usize)
}

pub fn normalize_axes(axes: &[usize], rank: usize) -> Result<Vec<usize>> {
    if axes.is_empty() {
        return Err(Error::invalid_graph(
            "reduction axes must not be empty".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(axes.len());
    for &axis in axes {
        if axis >= rank {
            return Err(Error::InvalidAxis {
                axis: axis as i32,
                rank,
            });
        }
        if !out.contains(&axis) {
            out.push(axis);
        }
    }
    out.sort_unstable();
    Ok(out)
}

fn matmul_shape(lhs: &Shape, rhs: &Shape, transpose_a: bool, transpose_b: bool) -> Result<Shape> {
    if lhs.rank() < 2 || rhs.rank() < 2 {
        return Err(Error::invalid_graph(
            "MatMul requires operands with rank >= 2".to_string(),
        ));
    }
    let a = lhs.dims();
    let b = rhs.dims();
    let a_rank = a.len();
    let b_rank = b.len();
    let (m, k_a) = if transpose_a {
        (a[a_rank - 1], a[a_rank - 2])
    } else {
        (a[a_rank - 2], a[a_rank - 1])
    };
    let (k_b, n) = if transpose_b {
        (b[b_rank - 1], b[b_rank - 2])
    } else {
        (b[b_rank - 2], b[b_rank - 1])
    };
    if let (Some(k_a), Some(k_b)) = (k_a.static_usize(), k_b.static_usize())
        && k_a != k_b
    {
        return Err(Error::shape_mismatch(
            "MatMul inner dimensions",
            Shape::from_static([k_a]),
            Shape::from_static([k_b]),
        ));
    }
    let batch_a = Shape::new(a[..a_rank - 2].to_vec());
    let batch_b = Shape::new(b[..b_rank - 2].to_vec());
    let batch = batch_a.broadcast(&batch_b)?;
    let mut dims = batch.into_dims();
    dims.push(m);
    dims.push(n);
    Ok(Shape::new(dims))
}

fn transpose_shape(shape: &Shape, axes: &[usize]) -> Result<Shape> {
    if axes.len() != shape.rank() {
        return Err(Error::RankMismatch {
            context: "transpose axes".into(),
            expected: shape.rank(),
            got: axes.len(),
        });
    }
    let mut seen = vec![false; shape.rank()];
    for &axis in axes {
        if axis >= shape.rank() || seen[axis] {
            return Err(Error::InvalidAxis {
                axis: axis as i32,
                rank: shape.rank(),
            });
        }
        seen[axis] = true;
    }
    Ok(Shape::new(
        axes.iter()
            .map(|&axis| shape.dim(axis).expect("validated axis"))
            .collect::<Vec<_>>(),
    ))
}

fn reduce_shape(shape: &Shape, axes: &[usize], keep_dims: bool) -> Shape {
    let mut dims = Vec::with_capacity(shape.rank());
    for (index, dim) in shape.dims().iter().enumerate() {
        if axes.contains(&index) {
            if keep_dims {
                dims.push(Dim::Static(1));
            }
        } else {
            dims.push(*dim);
        }
    }
    Shape::new(dims)
}

fn conv2d_shape(
    input: &Shape,
    weight: &Shape,
    bias: Option<&Shape>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<Shape> {
    if input.rank() != 4 || weight.rank() != 4 {
        return Err(Error::invalid_graph(
            "Conv2d expects NCHW input and OIHW weight".to_string(),
        ));
    }
    if groups == 0 {
        return Err(Error::invalid_graph(
            "Conv2d groups must be > 0".to_string(),
        ));
    }
    if stride.contains(&0) || dilation.contains(&0) {
        return Err(Error::invalid_graph(
            "Conv2d stride and dilation must be > 0".to_string(),
        ));
    }
    let input_channels = static_dim(input, 1, "Conv2d input channels")?;
    let out_channels = static_dim(weight, 0, "Conv2d output channels")?;
    let weight_channels = static_dim(weight, 1, "Conv2d weight input channels")?;
    let kernel_h = static_dim(weight, 2, "Conv2d kernel height")?;
    let kernel_w = static_dim(weight, 3, "Conv2d kernel width")?;
    if input_channels % groups != 0 || out_channels % groups != 0 {
        return Err(Error::invalid_graph(format!(
            "Conv2d channels must divide groups (in={input_channels}, out={out_channels}, groups={groups})"
        )));
    }
    if weight_channels != input_channels / groups {
        return Err(Error::shape_mismatch(
            "Conv2d weight input channels",
            Shape::from_static([input_channels / groups]),
            Shape::from_static([weight_channels]),
        ));
    }
    if let Some(bias) = bias
        && bias != &Shape::from_static([out_channels])
    {
        return Err(Error::shape_mismatch(
            "Conv2d bias shape",
            Shape::from_static([out_channels]),
            bias.clone(),
        ));
    }
    let batch = static_dim(input, 0, "Conv2d batch")?;
    let height = static_dim(input, 2, "Conv2d input height")?;
    let width = static_dim(input, 3, "Conv2d input width")?;
    let out_h = conv_output_dim(height, kernel_h, stride[0], padding[0], dilation[0])?;
    let out_w = conv_output_dim(width, kernel_w, stride[1], padding[1], dilation[1])?;
    Ok(Shape::from_static([batch, out_channels, out_h, out_w]))
}

fn static_dim(shape: &Shape, axis: usize, context: &str) -> Result<usize> {
    shape
        .dim(axis)
        .and_then(Dim::static_usize)
        .ok_or_else(|| Error::invalid_shape(context, format!("shape {shape} is not fully static")))
}

fn conv_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    let effective = dilation * (kernel - 1) + 1;
    let padded = input + 2 * padding;
    if padded < effective {
        return Err(Error::invalid_shape(
            "convolution",
            format!(
                "input {input} with padding {padding} is smaller than kernel extent {effective}"
            ),
        ));
    }
    Ok((padded - effective) / stride + 1)
}

fn pool2d_shape(
    input: &Shape,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<Shape> {
    if input.rank() != 4 {
        return Err(Error::invalid_graph(
            "pooling expects NCHW input".to_string(),
        ));
    }
    if kernel.contains(&0) || stride.contains(&0) || dilation.contains(&0) {
        return Err(Error::invalid_graph(
            "pool kernel, stride and dilation must be > 0".to_string(),
        ));
    }
    let batch = static_dim(input, 0, "pool batch")?;
    let channels = static_dim(input, 1, "pool channels")?;
    let height = static_dim(input, 2, "pool height")?;
    let width = static_dim(input, 3, "pool width")?;
    let out_h = conv_output_dim(height, kernel[0], stride[0], padding[0], dilation[0])?;
    let out_w = conv_output_dim(width, kernel[1], stride[1], padding[1], dilation[1])?;
    Ok(Shape::from_static([batch, channels, out_h, out_w]))
}

// Keep these imports referenced when building with older feature sets.
#[allow(dead_code)]
fn _activation_marker(kind: ActivationKind) -> ActivationKind {
    kind
}

#[allow(dead_code)]
fn _reduction_marker(reduction: Reduction) -> Reduction {
    reduction
}

#[cfg(test)]
mod tests {
    use oxide_ml_core::{Device, ValueId};

    use super::*;

    fn value(shape: Shape) -> ValueInfo {
        ValueInfo {
            id: ValueId(0),
            name: None,
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
            kind: crate::ir::ValueKind::Intermediate,
        }
    }

    #[test]
    fn matmul_inference() {
        let shape = matmul_shape(
            &Shape::from_static([8, 4, 16]),
            &Shape::from_static([8, 16, 32]),
            false,
            false,
        )
        .unwrap();
        assert_eq!(shape, Shape::from_static([8, 4, 32]));
    }

    #[test]
    fn layer_norm_inference() {
        let inputs = vec![
            value(Shape::from_static([2, 8])),
            value(Shape::from_static([8])),
            value(Shape::from_static([8])),
        ];
        let attrs = OpAttrs::LayerNorm {
            normalized_shape: vec![8],
            eps: 1e-5,
        };
        let out = infer_outputs(&Op::LayerNorm, &attrs, &inputs).unwrap();
        assert_eq!(out[0].shape, Shape::from_static([2, 8]));
    }

    #[test]
    fn conv_output_shape() {
        let shape = conv2d_shape(
            &Shape::from_static([1, 3, 8, 8]),
            &Shape::from_static([4, 3, 3, 3]),
            Some(&Shape::from_static([4])),
            [1, 1],
            [1, 1],
            [1, 1],
            1,
        )
        .unwrap();
        assert_eq!(shape, Shape::from_static([1, 4, 8, 8]));
    }
}
