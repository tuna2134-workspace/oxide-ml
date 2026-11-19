use oxide_ml_core::{Dim, Error, Result, Shape, ValueId};
use oxide_ml_graph::{
    ActivationKind, Conv2dConfig, GraphBuilder, Node, Op, OpAttrs, Pool2dConfig, Reduction,
    normalize_axes, normalize_axis,
};

pub(crate) fn backward_node(
    builder: &mut GraphBuilder,
    node: &Node,
    upstream: &[Option<ValueId>],
) -> Result<Vec<Option<ValueId>>> {
    let dy = upstream
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| Error::Autograd(format!("{} has no upstream gradient", node.id)))?;
    match &node.op {
        Op::AddRelu => {
            let lhs_shape = builder.value_shape(node.inputs[0])?;
            let rhs_shape = builder.value_shape(node.inputs[1])?;
            let dsum = builder.single(
                Op::AddReluBackward,
                OpAttrs::None,
                &[node.inputs[0], node.inputs[1], dy],
            )?;
            Ok(vec![
                Some(unbroadcast(builder, dsum, lhs_shape)?),
                Some(unbroadcast(builder, dsum, rhs_shape)?),
            ])
        }
        Op::Add | Op::Sub => {
            let lhs_shape = builder.value_shape(node.inputs[0])?;
            let rhs_shape = builder.value_shape(node.inputs[1])?;
            let lhs = unbroadcast(builder, dy, lhs_shape)?;
            let rhs_dy = if matches!(node.op, Op::Sub) {
                builder.neg(dy)?
            } else {
                dy
            };
            let rhs = unbroadcast(builder, rhs_dy, rhs_shape)?;
            Ok(vec![Some(lhs), Some(rhs)])
        }
        Op::Mul => {
            let lhs_shape = builder.value_shape(node.inputs[0])?;
            let rhs_shape = builder.value_shape(node.inputs[1])?;
            let lhs = builder.mul(dy, node.inputs[1])?;
            let rhs = builder.mul(dy, node.inputs[0])?;
            Ok(vec![
                Some(unbroadcast(builder, lhs, lhs_shape)?),
                Some(unbroadcast(builder, rhs, rhs_shape)?),
            ])
        }
        Op::Div => {
            let lhs_shape = builder.value_shape(node.inputs[0])?;
            let rhs_shape = builder.value_shape(node.inputs[1])?;
            let lhs = builder.div(dy, node.inputs[1])?;
            let denominator = builder.mul(node.inputs[1], node.inputs[1])?;
            let quotient = builder.div(node.inputs[0], denominator)?;
            let rhs = builder.mul(dy, quotient)?;
            let rhs = builder.neg(rhs)?;
            Ok(vec![
                Some(unbroadcast(builder, lhs, lhs_shape)?),
                Some(unbroadcast(builder, rhs, rhs_shape)?),
            ])
        }
        Op::Neg => Ok(vec![Some(builder.neg(dy)?)]),
        Op::MatMul => matmul_grad(builder, node, dy),
        Op::ReLU => Ok(vec![Some(builder.activation_backward(
            ActivationKind::ReLU,
            node.inputs[0],
            dy,
        )?)]),
        Op::GELU => Ok(vec![Some(builder.activation_backward(
            ActivationKind::GELU,
            node.inputs[0],
            dy,
        )?)]),
        Op::Sigmoid => Ok(vec![Some(builder.activation_backward(
            ActivationKind::Sigmoid,
            node.inputs[0],
            dy,
        )?)]),
        Op::Tanh => Ok(vec![Some(builder.activation_backward(
            ActivationKind::Tanh,
            node.inputs[0],
            dy,
        )?)]),
        Op::Sqrt => {
            let x = node.inputs[0];
            let root = builder.sqrt(x)?;
            let dtype = builder.value_dtype(x)?;
            let two = builder.constant_scalar_as(2.0, dtype)?;
            let denominator = builder.mul(root, two)?;
            Ok(vec![Some(builder.div(dy, denominator)?)])
        }
        Op::Exp => Ok(vec![Some(builder.mul(dy, node.outputs[0])?)]),
        Op::Log => Ok(vec![Some(builder.div(dy, node.inputs[0])?)]),
        Op::Abs => Ok(vec![Some(builder.abs_backward(node.inputs[0], dy)?)]),
        Op::LogSoftmax => {
            let axis = match &node.attrs {
                OpAttrs::LogSoftmax { axis } => *axis,
                _ => return Err(Error::Autograd("LogSoftmax missing axis".into())),
            };
            let rank = builder.value_shape(node.inputs[0])?.rank();
            let axis = normalize_axis(axis, rank)?;
            // dx = dy - exp(y) * sum(dy, axis, keep_dims = true)
            let summed = builder.reduce_sum(dy, vec![axis], true)?;
            let probabilities = builder.exp(node.outputs[0])?;
            let scaled = builder.mul(probabilities, summed)?;
            Ok(vec![Some(builder.sub(dy, scaled)?)])
        }
        Op::Slice => {
            let (axis, start, len) = match &node.attrs {
                OpAttrs::Slice { axis, start, len } => (*axis, *start, *len),
                _ => return Err(Error::Autograd("Slice missing attrs".into())),
            };
            let input_shape = builder.value_shape(node.inputs[0])?;
            Ok(vec![Some(builder.slice_backward(
                dy,
                axis,
                start,
                len,
                input_shape,
            )?)])
        }
        Op::Embedding => {
            let weight_shape = builder.value_shape(node.inputs[0])?;
            let vocab = weight_shape
                .dim(0)
                .and_then(Dim::static_usize)
                .ok_or_else(|| {
                    Error::Autograd("Embedding backward requires a static vocabulary size".into())
                })?;
            let dweight = builder.embedding_backward(node.inputs[1], dy, vocab)?;
            Ok(vec![Some(dweight), None])
        }
        Op::Softmax => {
            let axis = match &node.attrs {
                OpAttrs::Softmax { axis } => *axis,
                _ => return Err(Error::Autograd("Softmax missing axis".into())),
            };
            Ok(vec![Some(builder.softmax_backward(
                node.outputs[0],
                dy,
                axis,
            )?)])
        }
        Op::LayerNorm => {
            let (normalized_shape, eps) = match &node.attrs {
                OpAttrs::LayerNorm {
                    normalized_shape,
                    eps,
                } => (normalized_shape.clone(), *eps),
                _ => return Err(Error::Autograd("LayerNorm missing attrs".into())),
            };
            let [dx, dgamma, dbeta] = builder.layer_norm_backward(
                node.inputs[0],
                node.inputs[1],
                dy,
                normalized_shape,
                eps,
            )?;
            Ok(vec![Some(dx), Some(dgamma), Some(dbeta)])
        }
        Op::Reshape => {
            let target = builder.value_shape(node.inputs[0])?;
            Ok(vec![Some(builder.reshape(dy, target)?)])
        }
        Op::Transpose => {
            let axes = match &node.attrs {
                OpAttrs::Transpose { axes } => axes.clone(),
                _ => return Err(Error::Autograd("Transpose missing axes".into())),
            };
            let inverse = inverse_axes(&axes);
            Ok(vec![Some(builder.transpose(dy, inverse)?)])
        }
        Op::BroadcastTo => {
            let target = builder.value_shape(node.inputs[0])?;
            Ok(vec![Some(builder.unbroadcast(dy, target)?)])
        }
        Op::ReduceSum => reduce_grad(builder, node, dy, false),
        Op::ReduceMean => reduce_grad(builder, node, dy, true),
        Op::Conv2d => {
            let config = conv_config(&node.attrs)?;
            let [dx, dw, db] =
                builder.conv2d_backward(node.inputs[0], node.inputs[1], dy, config)?;
            if node.inputs.len() == 3 {
                Ok(vec![Some(dx), Some(dw), Some(db)])
            } else {
                Ok(vec![Some(dx), Some(dw)])
            }
        }
        Op::MaxPool2d => {
            let config = pool_config(&node.attrs)?;
            Ok(vec![Some(builder.max_pool2d_backward(
                node.inputs[0],
                dy,
                config,
            )?)])
        }
        Op::AvgPool2d => {
            let config = pool_config(&node.attrs)?;
            Ok(vec![Some(builder.avg_pool2d_backward(
                node.inputs[0],
                dy,
                config,
            )?)])
        }
        Op::MSELoss => {
            let reduction = loss_reduction(&node.attrs)?;
            let [dpred, dtarget] =
                builder.mse_loss_backward(node.inputs[0], node.inputs[1], dy, reduction)?;
            Ok(vec![Some(dpred), Some(dtarget)])
        }
        Op::CrossEntropyLoss => {
            let reduction = loss_reduction(&node.attrs)?;
            let dlogits = builder.cross_entropy_loss_backward(
                node.inputs[0],
                node.inputs[1],
                dy,
                reduction,
            )?;
            Ok(vec![Some(dlogits), None])
        }
        Op::Constant | Op::ConstantScalar => Ok(Vec::new()),
        op if op.is_backward() => Err(Error::Autograd(
            "cannot differentiate through a backward graph".into(),
        )),
        other => Err(Error::Autograd(format!(
            "no gradient rule registered for {}",
            other.name()
        ))),
    }
}

fn matmul_grad(
    builder: &mut GraphBuilder,
    node: &Node,
    dy: ValueId,
) -> Result<Vec<Option<ValueId>>> {
    let (transpose_a, transpose_b) = match &node.attrs {
        OpAttrs::MatMul {
            transpose_a,
            transpose_b,
        } => (*transpose_a, *transpose_b),
        _ => return Err(Error::Autograd("MatMul missing attributes".into())),
    };
    let a = node.inputs[0];
    let b = node.inputs[1];
    let a_shape = builder.value_shape(a)?;
    let b_shape = builder.value_shape(b)?;
    let a_logical = if transpose_a {
        builder.transpose_last(a)?
    } else {
        a
    };
    let b_logical = if transpose_b {
        builder.transpose_last(b)?
    } else {
        b
    };
    let b_logical_t = builder.transpose_last(b_logical)?;
    let a_logical_t = builder.transpose_last(a_logical)?;
    let da_logical = builder.matmul(dy, b_logical_t)?;
    let db_logical = builder.matmul(a_logical_t, dy)?;
    let da = if transpose_a {
        builder.transpose_last(da_logical)?
    } else {
        da_logical
    };
    let db = if transpose_b {
        builder.transpose_last(db_logical)?
    } else {
        db_logical
    };
    Ok(vec![
        Some(unbroadcast(builder, da, a_shape)?),
        Some(unbroadcast(builder, db, b_shape)?),
    ])
}

fn reduce_grad(
    builder: &mut GraphBuilder,
    node: &Node,
    dy: ValueId,
    mean: bool,
) -> Result<Vec<Option<ValueId>>> {
    let (axes, keep_dims) = match &node.attrs {
        OpAttrs::Reduce { axes, keep_dims } => (axes.clone(), *keep_dims),
        _ => return Err(Error::Autograd("Reduce missing attrs".into())),
    };
    let input_shape = builder.value_shape(node.inputs[0])?;
    let axes = normalize_axes(&axes, input_shape.rank())?;
    let reduced_shape = builder.value_shape(dy)?;
    let expanded = if keep_dims {
        dy
    } else {
        let mut dims = reduced_shape.dims().to_vec();
        let mut sorted = axes.clone();
        sorted.sort_unstable();
        for &axis in &sorted {
            dims.insert(axis, Dim::Static(1));
        }
        builder.reshape(dy, Shape::new(dims))?
    };
    let mut gradient = builder.broadcast_to(expanded, input_shape.clone())?;
    if mean {
        let static_dims = input_shape.static_dims().ok_or_else(|| {
            Error::Autograd("ReduceMean gradient requires a static input shape".into())
        })?;
        let count: usize = axes.iter().map(|&axis| static_dims[axis]).product();
        if count == 0 {
            return Err(Error::Autograd("ReduceMean over zero elements".into()));
        }
        let dtype = builder.value_dtype(gradient)?;
        let scale = builder.constant_scalar_as(1.0 / count as f64, dtype)?;
        gradient = builder.mul(gradient, scale)?;
    }
    Ok(vec![Some(gradient)])
}

fn unbroadcast(builder: &mut GraphBuilder, gradient: ValueId, target: Shape) -> Result<ValueId> {
    if builder.value_shape(gradient)? == target {
        Ok(gradient)
    } else {
        builder.unbroadcast(gradient, target)
    }
}

fn inverse_axes(axes: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; axes.len()];
    for (output_axis, &input_axis) in axes.iter().enumerate() {
        inverse[input_axis] = output_axis;
    }
    inverse
}

fn conv_config(attrs: &OpAttrs) -> Result<Conv2dConfig> {
    match attrs {
        OpAttrs::Conv2d {
            stride,
            padding,
            dilation,
            groups,
        } => Ok(Conv2dConfig {
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
            groups: *groups,
        }),
        _ => Err(Error::Autograd("Conv2d missing attrs".into())),
    }
}

fn pool_config(attrs: &OpAttrs) -> Result<Pool2dConfig> {
    match attrs {
        OpAttrs::Pool2d {
            kernel,
            stride,
            padding,
            dilation,
        } => Ok(Pool2dConfig {
            kernel: *kernel,
            stride: *stride,
            padding: *padding,
            dilation: *dilation,
        }),
        _ => Err(Error::Autograd("pooling missing attrs".into())),
    }
}

fn loss_reduction(attrs: &OpAttrs) -> Result<Reduction> {
    match attrs {
        OpAttrs::Loss { reduction } => Ok(*reduction),
        _ => Err(Error::Autograd("loss missing reduction attrs".into())),
    }
}
