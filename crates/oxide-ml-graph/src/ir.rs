use std::path::Path;

use oxide_ml_core::{DType, Device, Error, NodeId, ParameterId, Result, Shape, SymbolId, ValueId};
use serde::{Deserialize, Serialize};

/// Version of the JSON envelope (field layout and encoding).
///
/// This is deliberately independent from [`IR_VERSION`]. The envelope can gain
/// fields while the semantic IR is unchanged, and the IR can evolve while an
/// older envelope remains readable.
pub const SERIALIZATION_FORMAT_VERSION: u32 = 2;

/// Backwards-compatible name for the serialization format version.
pub const GRAPH_FORMAT_VERSION: u32 = SERIALIZATION_FORMAT_VERSION;

/// Semantic Graph IR version. This version changes only when the meaning of
/// nodes, values, or attributes changes in a way an older compiler must reject.
///
/// Adding new `Op`/`OpAttrs` variants is additive: existing graphs keep their
/// meaning and remain valid, so the version stays at 1. Only a change to the
/// semantics of an existing node or attribute requires a bump.
pub const IR_VERSION: u32 = 1;

pub(crate) const fn current_ir_version() -> u32 {
    IR_VERSION
}

/// Dense constant payload used by constant folding and by user-created constants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConstantData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantTensor {
    pub shape: Shape,
    pub data: ConstantData,
}

impl ConstantTensor {
    pub fn f32(shape: Shape, values: Vec<f32>) -> Result<Self> {
        let tensor = ConstantTensor {
            shape,
            data: ConstantData::F32(values),
        };
        tensor.validate()?;
        Ok(tensor)
    }

    pub fn f64(shape: Shape, values: Vec<f64>) -> Result<Self> {
        let tensor = ConstantTensor {
            shape,
            data: ConstantData::F64(values),
        };
        tensor.validate()?;
        Ok(tensor)
    }

    pub fn dtype(&self) -> DType {
        match self.data {
            ConstantData::F32(_) => DType::F32,
            ConstantData::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            ConstantData::F32(values) => values.len(),
            ConstantData::F64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn validate(&self) -> Result<()> {
        let expected = self.shape.numel_static().ok_or_else(|| {
            Error::invalid_shape("constant", "constant shape must be fully static")
        })?;
        if expected != self.len() {
            return Err(Error::invalid_shape(
                "constant",
                format!(
                    "shape {} requires {expected} elements, got {}",
                    self.shape,
                    self.len()
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationKind {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
}

impl ActivationKind {
    pub const fn name(self) -> &'static str {
        match self {
            ActivationKind::ReLU => "relu",
            ActivationKind::GELU => "gelu",
            ActivationKind::Sigmoid => "sigmoid",
            ActivationKind::Tanh => "tanh",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Reduction {
    Mean,
    Sum,
}

/// ML-semantic operation. This is intentionally backend-independent: the CUDA
/// backend maps these variants to kernel launches; it never leaks CUDA types
/// back into the IR.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Add,
    AddRelu,
    Sub,
    Mul,
    Div,
    Neg,
    MatMul,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    Sqrt,
    Exp,
    Log,
    Abs,
    LogSoftmax,
    Slice,
    Embedding,
    /// A user-registered kernel (see `oxide_ml_kernel::custom`). Arity and
    /// output shapes are carried by `OpAttrs::Custom`.
    Custom,
    LayerNorm,
    Reshape,
    Transpose,
    BroadcastTo,
    ReduceSum,
    ReduceMean,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    MSELoss,
    CrossEntropyLoss,
    Constant,
    ConstantScalar,
    // Internal fused backward primitives. They are part of the IR because the
    // backward graph is a first-class executable graph, not an interpreter
    // special case.
    ActivationBackward(ActivationKind),
    AddReluBackward,
    SoftmaxBackward,
    LayerNormBackward,
    MSELossBackward,
    CrossEntropyLossBackward,
    Conv2dBackward,
    MaxPool2dBackward,
    AvgPool2dBackward,
    AbsBackward,
    SliceBackward,
    EmbeddingBackward,
    Unbroadcast,
}

impl Op {
    pub const fn name(&self) -> &'static str {
        match self {
            Op::Add => "add",
            Op::AddRelu => "add_relu",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Div => "div",
            Op::Neg => "neg",
            Op::MatMul => "matmul",
            Op::ReLU => "relu",
            Op::GELU => "gelu",
            Op::Sigmoid => "sigmoid",
            Op::Tanh => "tanh",
            Op::Softmax => "softmax",
            Op::Sqrt => "sqrt",
            Op::Exp => "exp",
            Op::Log => "log",
            Op::Abs => "abs",
            Op::LogSoftmax => "log_softmax",
            Op::Slice => "slice",
            Op::Embedding => "embedding",
            Op::Custom => "custom",
            Op::LayerNorm => "layer_norm",
            Op::Reshape => "reshape",
            Op::Transpose => "transpose",
            Op::BroadcastTo => "broadcast_to",
            Op::ReduceSum => "reduce_sum",
            Op::ReduceMean => "reduce_mean",
            Op::Conv2d => "conv2d",
            Op::MaxPool2d => "max_pool2d",
            Op::AvgPool2d => "avg_pool2d",
            Op::MSELoss => "mse_loss",
            Op::CrossEntropyLoss => "cross_entropy_loss",
            Op::Constant => "constant",
            Op::ConstantScalar => "constant_scalar",
            Op::ActivationBackward(_) => "activation_backward",
            Op::AddReluBackward => "add_relu_backward",
            Op::SoftmaxBackward => "softmax_backward",
            Op::LayerNormBackward => "layer_norm_backward",
            Op::MSELossBackward => "mse_loss_backward",
            Op::CrossEntropyLossBackward => "cross_entropy_loss_backward",
            Op::Conv2dBackward => "conv2d_backward",
            Op::MaxPool2dBackward => "max_pool2d_backward",
            Op::AvgPool2dBackward => "avg_pool2d_backward",
            Op::AbsBackward => "abs_backward",
            Op::SliceBackward => "slice_backward",
            Op::EmbeddingBackward => "embedding_backward",
            Op::Unbroadcast => "unbroadcast",
        }
    }

    pub const fn is_backward(&self) -> bool {
        matches!(
            self,
            Op::ActivationBackward(_)
                | Op::SoftmaxBackward
                | Op::LayerNormBackward
                | Op::MSELossBackward
                | Op::CrossEntropyLossBackward
                | Op::Conv2dBackward
                | Op::MaxPool2dBackward
                | Op::AvgPool2dBackward
                | Op::AbsBackward
                | Op::SliceBackward
                | Op::EmbeddingBackward
                | Op::Unbroadcast
                | Op::AddReluBackward
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OpAttrs {
    None,
    MatMul {
        transpose_a: bool,
        transpose_b: bool,
    },
    Conv2d {
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    },
    Pool2d {
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    },
    Reduce {
        axes: Vec<usize>,
        keep_dims: bool,
    },
    Softmax {
        axis: i32,
    },
    LogSoftmax {
        axis: i32,
    },
    Slice {
        axis: usize,
        start: usize,
        len: usize,
    },
    SliceBackward {
        axis: usize,
        start: usize,
        len: usize,
        input_shape: Shape,
    },
    EmbeddingBackward {
        vocab: usize,
    },
    Custom {
        name: String,
        input_count: usize,
        outputs: Vec<Shape>,
        dtype: DType,
    },
    LayerNorm {
        normalized_shape: Vec<usize>,
        eps: f32,
    },
    Reshape {
        shape: Shape,
    },
    Transpose {
        axes: Vec<usize>,
    },
    BroadcastTo {
        shape: Shape,
    },
    Loss {
        reduction: Reduction,
    },
    ActivationBackward {
        kind: ActivationKind,
    },
    Unbroadcast {
        shape: Shape,
    },
    Constant {
        value: ConstantTensor,
    },
    ConstantScalar {
        value: f32,
    },
}

impl OpAttrs {
    pub fn none() -> Self {
        OpAttrs::None
    }

    pub fn matmul(transpose_a: bool, transpose_b: bool) -> Self {
        OpAttrs::MatMul {
            transpose_a,
            transpose_b,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ValueKind {
    Input,
    Parameter(ParameterId),
    Constant,
    Intermediate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueInfo {
    pub id: ValueId,
    pub name: Option<String>,
    pub shape: Shape,
    pub dtype: DType,
    pub device: Device,
    pub kind: ValueKind,
}

impl ValueInfo {
    pub fn is_parameter(&self) -> bool {
        matches!(self.kind, ValueKind::Parameter(_))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub id: ParameterId,
    pub value: ValueId,
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    pub trainable: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub id: SymbolId,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub attrs: OpAttrs,
}

/// SSA computation graph.
///
/// Nodes are stored in topological order. Every value is defined exactly once,
/// either as an input, a parameter, a constant node, or an intermediate node
/// output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    /// JSON envelope version ([`SERIALIZATION_FORMAT_VERSION`]).
    pub format_version: u32,
    /// Semantic IR version ([`IR_VERSION`]).
    #[serde(default = "current_ir_version")]
    pub ir_version: u32,
    pub symbols: Vec<SymbolInfo>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub nodes: Vec<Node>,
    pub values: Vec<ValueInfo>,
    pub parameters: Vec<ParameterInfo>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            format_version: SERIALIZATION_FORMAT_VERSION,
            ir_version: IR_VERSION,
            symbols: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            values: Vec::new(),
            parameters: Vec::new(),
        }
    }

    pub fn value(&self, id: ValueId) -> Result<&ValueInfo> {
        self.values
            .get(id.0 as usize)
            .ok_or(Error::UnknownValue(id))
    }

    pub fn value_mut(&mut self, id: ValueId) -> Result<&mut ValueInfo> {
        self.values
            .get_mut(id.0 as usize)
            .ok_or(Error::UnknownValue(id))
    }

    pub fn value_by_name(&self, name: &str) -> Option<&ValueInfo> {
        self.values
            .iter()
            .find(|value| value.name.as_deref() == Some(name))
    }

    pub fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes
            .get(id.0 as usize)
            .ok_or_else(|| Error::invalid_graph(format!("unknown node {id}")))
    }

    pub fn parameter(&self, id: ParameterId) -> Result<&ParameterInfo> {
        self.parameters
            .get(id.0 as usize)
            .ok_or(Error::UnknownParameter(id))
    }

    pub fn parameters(&self) -> &[ParameterInfo] {
        &self.parameters
    }

    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn values(&self) -> &[ValueInfo] {
        &self.values
    }

    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    pub fn symbol(&self, id: SymbolId) -> Option<&SymbolInfo> {
        self.symbols.iter().find(|symbol| symbol.id == id)
    }

    pub fn symbol_by_name(&self, name: &str) -> Option<&SymbolInfo> {
        self.symbols.iter().find(|symbol| symbol.name == name)
    }

    /// Output values as `(name, tensor shape, dtype)` records.
    pub fn output_info(&self) -> Result<Vec<(&str, &Shape, DType)>> {
        self.outputs
            .iter()
            .map(|&id| {
                let value = self.value(id)?;
                Ok((
                    value.name.as_deref().unwrap_or("<unnamed>"),
                    &value.shape,
                    value.dtype,
                ))
            })
            .collect()
    }

    /// Compact structural hash derived from the canonical BLAKE3 graph hash.
    ///
    /// This is a compatibility convenience for callers that only need a
    /// `u64` cache key. New identity-sensitive code should use
    /// [`crate::graph_structure_hash`] / [`crate::GraphHash`].
    pub fn structural_hash(&self) -> Result<u64> {
        let hash = crate::graph_structure_hash(self)?;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&hash.0[..8]);
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let mut graph: Graph = serde_json::from_str(json)?;
        match graph.format_version {
            // Version 1 had no separate IR version field. Its node semantics
            // are the same as IR v1, so the reader can migrate it explicitly.
            1 => {
                graph.format_version = SERIALIZATION_FORMAT_VERSION;
                graph.ir_version = IR_VERSION;
            }
            SERIALIZATION_FORMAT_VERSION => {}
            other => {
                return Err(Error::Serialization(format!(
                    "unsupported graph format version {other} (this build supports 1 and {SERIALIZATION_FORMAT_VERSION})"
                )));
            }
        }
        crate::verifier::GraphVerifier::verify(&graph)?;
        Ok(graph)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    /// Compact, deterministic textual dump for pass debugging.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "graph format=v{} ir=v{} values={} nodes={} parameters={}\n",
            self.format_version,
            self.ir_version,
            self.values.len(),
            self.nodes.len(),
            self.parameters.len()
        ));
        for value in &self.values {
            out.push_str(&format!(
                "  {} name={:?} shape={} dtype={} device={} kind={:?}\n",
                value.id, value.name, value.shape, value.dtype, value.device, value.kind
            ));
        }
        for node in &self.nodes {
            out.push_str(&format!(
                "  {} = {} {:?} attrs={:?}\n",
                node.outputs
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                node.op.name(),
                node.inputs,
                node.attrs
            ));
        }
        out.push_str("  outputs:");
        for output in &self.outputs {
            out.push_str(&format!(" {output}"));
        }
        out.push('\n');
        out
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_attrs_are_serializable() {
        let attrs = OpAttrs::Conv2d {
            stride: [2, 2],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        };
        let json = serde_json::to_string(&attrs).unwrap();
        assert_eq!(serde_json::from_str::<OpAttrs>(&json).unwrap(), attrs);
    }
}
