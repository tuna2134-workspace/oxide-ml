use std::collections::HashMap;

use oxide_ml_core::{
    DType, Device, Dim, Error, NodeId, ParameterId, Result, Shape, SymbolId, ValueId,
};

use crate::infer::infer_outputs;
use crate::ir::{
    ActivationKind, Graph, Node, Op, OpAttrs, ParameterInfo, Reduction, SymbolInfo, ValueInfo,
    ValueKind,
};

/// Configuration for 2-D convolution in NCHW / OIHW layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dConfig {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Conv2dConfig {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

/// Configuration for 2-D pooling in NCHW layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool2dConfig {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
}

impl Pool2dConfig {
    pub fn new(kernel: [usize; 2], stride: [usize; 2]) -> Self {
        Pool2dConfig {
            kernel,
            stride,
            padding: [0, 0],
            dilation: [1, 1],
        }
    }
}

/// Incremental SSA graph builder.
///
/// Every method validates shapes and dtypes before appending a node. Parameters
/// are metadata only; their tensors live in `ParameterStore`.
#[derive(Clone, Debug)]
pub struct GraphBuilder {
    graph: Graph,
    symbols: HashMap<String, SymbolId>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        GraphBuilder {
            graph: Graph::new(),
            symbols: HashMap::new(),
        }
    }

    /// Continue an existing graph, e.g. to append a backward pass.
    pub fn from_graph(graph: Graph) -> Self {
        let symbols = graph
            .symbols
            .iter()
            .map(|symbol| (symbol.name.clone(), symbol.id))
            .collect();
        GraphBuilder { graph, symbols }
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn symbol(&mut self, name: impl Into<String>) -> SymbolId {
        let name = name.into();
        if let Some(&id) = self.symbols.get(&name) {
            return id;
        }
        let id = SymbolId(self.graph.symbols.len() as u32);
        self.graph.symbols.push(SymbolInfo {
            id,
            name: name.clone(),
        });
        self.symbols.insert(name, id);
        id
    }

    pub fn symbol_shape(&mut self, names: &[&str]) -> Shape {
        Shape::new(
            names
                .iter()
                .map(|name| Dim::Symbol(self.symbol(*name)))
                .collect::<Vec<_>>(),
        )
    }

    pub fn input(&mut self, name: impl Into<String>, shape: Shape, dtype: DType) -> ValueId {
        self.input_on(name, shape, dtype, Device::Cpu)
    }

    pub fn input_on(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        device: Device,
    ) -> ValueId {
        let id = self.next_value_id();
        self.graph.values.push(ValueInfo {
            id,
            name: Some(name.into()),
            shape,
            dtype,
            device,
            kind: ValueKind::Input,
        });
        self.graph.inputs.push(id);
        id
    }

    /// Add a parameter specification. The tensor itself is created later by
    /// `ParameterStore` initialization.
    pub fn parameter(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        trainable: bool,
    ) -> Result<ValueId> {
        self.try_parameter(name, shape, dtype, trainable)
    }

    pub fn try_parameter(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        trainable: bool,
    ) -> Result<ValueId> {
        let name = name.into();
        if self.graph.parameters.iter().any(|param| param.name == name) {
            return Err(Error::invalid_graph(format!(
                "duplicate parameter name {name:?}"
            )));
        }
        if shape.numel_static().is_none() {
            return Err(Error::invalid_shape(
                "parameter",
                format!("parameter {name:?} must have a fully static shape, got {shape}"),
            ));
        }
        let value = self.next_value_id();
        let id = ParameterId(self.graph.parameters.len() as u32);
        self.graph.values.push(ValueInfo {
            id: value,
            name: Some(name.clone()),
            shape: shape.clone(),
            dtype,
            device: Device::Cpu,
            kind: ValueKind::Parameter(id),
        });
        self.graph.parameters.push(ParameterInfo {
            id,
            value,
            name,
            shape,
            dtype,
            trainable,
        });
        Ok(value)
    }

    pub fn constant_scalar(&mut self, value: f32) -> ValueId {
        let attrs = OpAttrs::ConstantScalar { value };
        self.single(Op::ConstantScalar, attrs, &[])
            .expect("constant scalar inference cannot fail")
    }

    pub fn constant_tensor(&mut self, value: crate::ir::ConstantTensor) -> Result<ValueId> {
        self.single(Op::Constant, OpAttrs::Constant { value }, &[])
    }

    /// Register a user-provided kernel as a graph operator.
    ///
    /// The descriptor (dtypes, arity, launch metadata) lives in
    /// `oxide-ml-kernel`; the graph stores the name and declared output shapes
    /// so it stays backend-independent.
    pub fn custom(
        &mut self,
        name: impl Into<String>,
        inputs: &[ValueId],
        output_shapes: Vec<Shape>,
    ) -> Result<Vec<ValueId>> {
        if output_shapes.is_empty() {
            return Err(Error::invalid_graph(
                "custom kernel must declare at least one output".to_string(),
            ));
        }
        let dtype = match inputs.first() {
            Some(value) => self.value_dtype(*value)?,
            None => DType::F32,
        };
        self.multi(
            Op::Custom,
            OpAttrs::Custom {
                name: name.into(),
                input_count: inputs.len(),
                outputs: output_shapes,
                dtype,
            },
            inputs,
        )
    }

    /// Create a scalar constant with the dtype required by the surrounding
    /// graph. This is what autograd uses for the F64 seed and reduction scales.
    pub fn constant_scalar_as(&mut self, value: f64, dtype: DType) -> Result<ValueId> {
        match dtype {
            DType::F32 => Ok(self.constant_scalar(value as f32)),
            DType::F64 => self.constant_tensor(crate::ir::ConstantTensor::f64(
                Shape::scalar(),
                vec![value],
            )?),
            other => Err(Error::unsupported(format!(
                "constant scalar dtype {other} is not implemented"
            ))),
        }
    }

    pub fn value_shape(&self, value: ValueId) -> Result<Shape> {
        Ok(self.graph.value(value)?.shape.clone())
    }

    pub fn value_dtype(&self, value: ValueId) -> Result<DType> {
        Ok(self.graph.value(value)?.dtype)
    }

    pub fn set_value_name(&mut self, value: ValueId, name: impl Into<String>) -> Result<()> {
        self.graph.value_mut(value)?.name = Some(name.into());
        Ok(())
    }

    pub fn binary(&mut self, op: Op, lhs: ValueId, rhs: ValueId) -> Result<ValueId> {
        self.single(op, OpAttrs::None, &[lhs, rhs])
    }

    pub fn add(&mut self, lhs: ValueId, rhs: ValueId) -> Result<ValueId> {
        self.binary(Op::Add, lhs, rhs)
    }

    pub fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> Result<ValueId> {
        self.binary(Op::Sub, lhs, rhs)
    }

    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> Result<ValueId> {
        self.binary(Op::Mul, lhs, rhs)
    }

    pub fn div(&mut self, lhs: ValueId, rhs: ValueId) -> Result<ValueId> {
        self.binary(Op::Div, lhs, rhs)
    }

    pub fn neg(&mut self, value: ValueId) -> Result<ValueId> {
        self.single(Op::Neg, OpAttrs::None, &[value])
    }

    pub fn unary(&mut self, op: Op, value: ValueId) -> Result<ValueId> {
        self.single(op, OpAttrs::None, &[value])
    }

    pub fn matmul(&mut self, lhs: ValueId, rhs: ValueId) -> Result<ValueId> {
        self.matmul_with(lhs, rhs, false, false)
    }

    pub fn matmul_with(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<ValueId> {
        self.single(
            Op::MatMul,
            OpAttrs::MatMul {
                transpose_a,
                transpose_b,
            },
            &[lhs, rhs],
        )
    }

    pub fn relu(&mut self, value: ValueId) -> Result<ValueId> {
        self.unary(Op::ReLU, value)
    }

    pub fn gelu(&mut self, value: ValueId) -> Result<ValueId> {
        self.unary(Op::GELU, value)
    }

    pub fn sigmoid(&mut self, value: ValueId) -> Result<ValueId> {
        self.unary(Op::Sigmoid, value)
    }

    pub fn tanh(&mut self, value: ValueId) -> Result<ValueId> {
        self.unary(Op::Tanh, value)
    }

    pub fn activation(&mut self, kind: ActivationKind, value: ValueId) -> Result<ValueId> {
        match kind {
            ActivationKind::ReLU => self.relu(value),
            ActivationKind::GELU => self.gelu(value),
            ActivationKind::Sigmoid => self.sigmoid(value),
            ActivationKind::Tanh => self.tanh(value),
        }
    }

    pub fn softmax(&mut self, value: ValueId, axis: i32) -> Result<ValueId> {
        self.single(Op::Softmax, OpAttrs::Softmax { axis }, &[value])
    }

    pub fn sqrt(&mut self, value: ValueId) -> Result<ValueId> {
        self.unary(Op::Sqrt, value)
    }

    pub fn exp(&mut self, value: ValueId) -> Result<ValueId> {
        self.unary(Op::Exp, value)
    }

    pub fn log(&mut self, value: ValueId) -> Result<ValueId> {
        self.unary(Op::Log, value)
    }

    pub fn abs(&mut self, value: ValueId) -> Result<ValueId> {
        self.unary(Op::Abs, value)
    }

    pub fn log_softmax(&mut self, value: ValueId, axis: i32) -> Result<ValueId> {
        self.single(Op::LogSoftmax, OpAttrs::LogSoftmax { axis }, &[value])
    }

    /// Take the contiguous range `[start, start + len)` along `axis`.
    pub fn slice(
        &mut self,
        value: ValueId,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<ValueId> {
        self.single(Op::Slice, OpAttrs::Slice { axis, start, len }, &[value])
    }

    pub fn embedding(&mut self, weight: ValueId, indices: ValueId) -> Result<ValueId> {
        self.single(Op::Embedding, OpAttrs::None, &[weight, indices])
    }

    pub fn layer_norm(
        &mut self,
        x: ValueId,
        gamma: ValueId,
        beta: ValueId,
        normalized_shape: Vec<usize>,
        eps: f32,
    ) -> Result<ValueId> {
        self.single(
            Op::LayerNorm,
            OpAttrs::LayerNorm {
                normalized_shape,
                eps,
            },
            &[x, gamma, beta],
        )
    }

    pub fn reshape(&mut self, value: ValueId, shape: impl Into<Shape>) -> Result<ValueId> {
        self.single(
            Op::Reshape,
            OpAttrs::Reshape {
                shape: shape.into(),
            },
            &[value],
        )
    }

    pub fn reshape_dims(&mut self, value: ValueId, dims: impl AsRef<[usize]>) -> Result<ValueId> {
        self.reshape(value, Shape::from_static(dims))
    }

    pub fn transpose(&mut self, value: ValueId, axes: Vec<usize>) -> Result<ValueId> {
        self.single(Op::Transpose, OpAttrs::Transpose { axes }, &[value])
    }

    pub fn transpose_last(&mut self, value: ValueId) -> Result<ValueId> {
        let rank = self.value_shape(value)?.rank();
        if rank < 2 {
            return Err(Error::RankMismatch {
                context: "transpose_last".into(),
                expected: 2,
                got: rank,
            });
        }
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(rank - 1, rank - 2);
        self.transpose(value, axes)
    }

    pub fn broadcast_to(&mut self, value: ValueId, shape: impl Into<Shape>) -> Result<ValueId> {
        self.single(
            Op::BroadcastTo,
            OpAttrs::BroadcastTo {
                shape: shape.into(),
            },
            &[value],
        )
    }

    pub fn reduce_sum(
        &mut self,
        value: ValueId,
        axes: Vec<usize>,
        keep_dims: bool,
    ) -> Result<ValueId> {
        self.single(Op::ReduceSum, OpAttrs::Reduce { axes, keep_dims }, &[value])
    }

    pub fn reduce_mean(
        &mut self,
        value: ValueId,
        axes: Vec<usize>,
        keep_dims: bool,
    ) -> Result<ValueId> {
        self.single(
            Op::ReduceMean,
            OpAttrs::Reduce { axes, keep_dims },
            &[value],
        )
    }

    pub fn conv2d(
        &mut self,
        x: ValueId,
        weight: ValueId,
        bias: Option<ValueId>,
        config: Conv2dConfig,
    ) -> Result<ValueId> {
        let mut inputs = vec![x, weight];
        if let Some(bias) = bias {
            inputs.push(bias);
        }
        self.single(
            Op::Conv2d,
            OpAttrs::Conv2d {
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
                groups: config.groups,
            },
            &inputs,
        )
    }

    pub fn max_pool2d(&mut self, x: ValueId, config: Pool2dConfig) -> Result<ValueId> {
        self.single(
            Op::MaxPool2d,
            OpAttrs::Pool2d {
                kernel: config.kernel,
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
            },
            &[x],
        )
    }

    pub fn avg_pool2d(&mut self, x: ValueId, config: Pool2dConfig) -> Result<ValueId> {
        self.single(
            Op::AvgPool2d,
            OpAttrs::Pool2d {
                kernel: config.kernel,
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
            },
            &[x],
        )
    }

    pub fn mse_loss(
        &mut self,
        prediction: ValueId,
        target: ValueId,
        reduction: Reduction,
    ) -> Result<ValueId> {
        self.single(
            Op::MSELoss,
            OpAttrs::Loss { reduction },
            &[prediction, target],
        )
    }

    pub fn cross_entropy_loss(
        &mut self,
        logits: ValueId,
        target_classes: ValueId,
        reduction: Reduction,
    ) -> Result<ValueId> {
        self.single(
            Op::CrossEntropyLoss,
            OpAttrs::Loss { reduction },
            &[logits, target_classes],
        )
    }

    // ---- Internal backward graph primitives --------------------------------

    pub fn activation_backward(
        &mut self,
        kind: ActivationKind,
        x: ValueId,
        dy: ValueId,
    ) -> Result<ValueId> {
        self.single(
            Op::ActivationBackward(kind),
            OpAttrs::ActivationBackward { kind },
            &[x, dy],
        )
    }

    pub fn softmax_backward(&mut self, y: ValueId, dy: ValueId, axis: i32) -> Result<ValueId> {
        self.single(Op::SoftmaxBackward, OpAttrs::Softmax { axis }, &[y, dy])
    }

    pub fn layer_norm_backward(
        &mut self,
        x: ValueId,
        gamma: ValueId,
        dy: ValueId,
        normalized_shape: Vec<usize>,
        eps: f32,
    ) -> Result<[ValueId; 3]> {
        let outputs = self.multi(
            Op::LayerNormBackward,
            OpAttrs::LayerNorm {
                normalized_shape,
                eps,
            },
            &[x, gamma, dy],
        )?;
        Ok([outputs[0], outputs[1], outputs[2]])
    }

    pub fn mse_loss_backward(
        &mut self,
        prediction: ValueId,
        target: ValueId,
        dy: ValueId,
        reduction: Reduction,
    ) -> Result<[ValueId; 2]> {
        let outputs = self.multi(
            Op::MSELossBackward,
            OpAttrs::Loss { reduction },
            &[prediction, target, dy],
        )?;
        Ok([outputs[0], outputs[1]])
    }

    pub fn cross_entropy_loss_backward(
        &mut self,
        logits: ValueId,
        targets: ValueId,
        dy: ValueId,
        reduction: Reduction,
    ) -> Result<ValueId> {
        self.single(
            Op::CrossEntropyLossBackward,
            OpAttrs::Loss { reduction },
            &[logits, targets, dy],
        )
    }

    pub fn conv2d_backward(
        &mut self,
        x: ValueId,
        weight: ValueId,
        dy: ValueId,
        config: Conv2dConfig,
    ) -> Result<[ValueId; 3]> {
        let outputs = self.multi(
            Op::Conv2dBackward,
            OpAttrs::Conv2d {
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
                groups: config.groups,
            },
            &[x, weight, dy],
        )?;
        Ok([outputs[0], outputs[1], outputs[2]])
    }

    pub fn max_pool2d_backward(
        &mut self,
        x: ValueId,
        dy: ValueId,
        config: Pool2dConfig,
    ) -> Result<ValueId> {
        self.single(
            Op::MaxPool2dBackward,
            OpAttrs::Pool2d {
                kernel: config.kernel,
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
            },
            &[x, dy],
        )
    }

    pub fn avg_pool2d_backward(
        &mut self,
        x: ValueId,
        dy: ValueId,
        config: Pool2dConfig,
    ) -> Result<ValueId> {
        self.single(
            Op::AvgPool2dBackward,
            OpAttrs::Pool2d {
                kernel: config.kernel,
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
            },
            &[x, dy],
        )
    }

    pub fn unbroadcast(&mut self, value: ValueId, target: Shape) -> Result<ValueId> {
        self.single(
            Op::Unbroadcast,
            OpAttrs::Unbroadcast { shape: target },
            &[value],
        )
    }

    pub fn abs_backward(&mut self, x: ValueId, dy: ValueId) -> Result<ValueId> {
        self.single(Op::AbsBackward, OpAttrs::None, &[x, dy])
    }

    /// Scatter `dy` back into a zero tensor of `input_shape` at the slice
    /// `[start, start + len)` along `axis`.
    pub fn slice_backward(
        &mut self,
        dy: ValueId,
        axis: usize,
        start: usize,
        len: usize,
        input_shape: Shape,
    ) -> Result<ValueId> {
        self.single(
            Op::SliceBackward,
            OpAttrs::SliceBackward {
                axis,
                start,
                len,
                input_shape,
            },
            &[dy],
        )
    }

    /// Scatter-add `dy` rows into a `[vocab, dim]` gradient for `indices`.
    pub fn embedding_backward(
        &mut self,
        indices: ValueId,
        dy: ValueId,
        vocab: usize,
    ) -> Result<ValueId> {
        self.single(
            Op::EmbeddingBackward,
            OpAttrs::EmbeddingBackward { vocab },
            &[indices, dy],
        )
    }

    // ---- Core node construction -------------------------------------------

    pub fn single(&mut self, op: Op, attrs: OpAttrs, inputs: &[ValueId]) -> Result<ValueId> {
        let outputs = self.multi(op, attrs, inputs)?;
        debug_assert_eq!(outputs.len(), 1);
        Ok(outputs[0])
    }

    pub fn multi(&mut self, op: Op, attrs: OpAttrs, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        let input_infos = inputs
            .iter()
            .map(|&id| self.graph.value(id).cloned())
            .collect::<Result<Vec<_>>>()?;
        let inferred = infer_outputs(&op, &attrs, &input_infos)?;
        let mut outputs = Vec::with_capacity(inferred.len());
        for output in inferred {
            let id = self.next_value_id();
            let kind = if matches!(op, Op::ConstantScalar) {
                ValueKind::Constant
            } else {
                ValueKind::Intermediate
            };
            self.graph.values.push(ValueInfo {
                id,
                name: None,
                shape: output.shape,
                dtype: output.dtype,
                device: input_infos
                    .first()
                    .map(|value| value.device)
                    .unwrap_or(Device::Cpu),
                kind,
            });
            outputs.push(id);
        }
        let id = NodeId(self.graph.nodes.len() as u32);
        self.graph.nodes.push(Node {
            id,
            op,
            inputs: inputs.to_vec(),
            outputs: outputs.clone(),
            attrs,
        });
        Ok(outputs)
    }

    pub fn finish(mut self, outputs: &[ValueId]) -> Result<Graph> {
        if outputs.is_empty() {
            return Err(Error::invalid_graph(
                "graph must have at least one output".to_string(),
            ));
        }
        for &output in outputs {
            self.graph.value(output)?;
        }
        self.graph.outputs = outputs.to_vec();
        self.graph.validate()?;
        Ok(self.graph)
    }

    fn next_value_id(&self) -> ValueId {
        ValueId(self.graph.values.len() as u32)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_mlp_graph() {
        let mut builder = GraphBuilder::new();
        let batch = builder.symbol("batch");
        let x = builder.input(
            "x",
            Shape::new(vec![Dim::Symbol(batch), Dim::Static(4)]),
            DType::F32,
        );
        let w = builder
            .try_parameter("w", Shape::from_static([4, 3]), DType::F32, true)
            .unwrap();
        let b = builder
            .try_parameter("b", Shape::from_static([3]), DType::F32, true)
            .unwrap();
        let z = builder.matmul(x, w).unwrap();
        let z = builder.add(z, b).unwrap();
        let y = builder.relu(z).unwrap();
        let graph = builder.finish(&[y]).unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.parameters.len(), 2);
    }
}
