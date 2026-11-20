use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use oxide_ml_core::{DType, Device, Dim, Error, NodeId, ParameterId, Shape, SymbolId, ValueId};

use crate::ir::{
    ActivationKind, ConstantData, ConstantTensor, Graph, Op, OpAttrs, Reduction, ValueInfo,
    ValueKind,
};

/// Version of the canonical graph representation.
///
/// Bump this whenever the canonicalization rules or encoded fields change.
/// It participates in both the canonical representation and the graph hash
/// domain, so old and new hashes cannot be confused.
pub const CANONICAL_GRAPH_VERSION: u32 = 1;

const EXTERNAL_DOMAIN: &[u8] = b"oxide-ml.graph.external.v1";
const NODE_DOMAIN: &[u8] = b"oxide-ml.graph.node.v1";
pub(crate) const GRAPH_HASH_DOMAIN: &[u8] = b"oxide-ml.graph.v1";
const RAW_GRAPH_IDENTITY_DOMAIN: &[u8] = b"oxide-ml.graph.raw-execution.v1";

/// A stable reference to a graph value in canonical form.
///
/// Inputs and parameters are referenced by their canonical (name-based)
/// position. Node outputs are referenced by canonical node position, which is
/// derived from the structural fingerprint rather than the original NodeId.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalValueRef {
    Input(u32),
    Parameter(u32),
    Node { node: u32, output: u32 },
}

/// A dimension with symbolic ids replaced by their stable symbol names.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalDim {
    Static(u64),
    Symbol(String),
    Dynamic,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalShape {
    pub dims: Vec<CanonicalDim>,
}

impl CanonicalShape {
    pub fn from_shape(
        shape: &Shape,
        symbols: &HashMap<SymbolId, String>,
    ) -> Result<Self, GraphHashError> {
        let mut dims = Vec::with_capacity(shape.rank());
        for dim in shape.dims() {
            dims.push(match *dim {
                Dim::Static(value) => CanonicalDim::Static(value as u64),
                Dim::Dynamic => CanonicalDim::Dynamic,
                Dim::Symbol(id) => CanonicalDim::Symbol(
                    symbols
                        .get(&id)
                        .cloned()
                        .ok_or(GraphHashError::UnknownSymbol(id))?,
                ),
            });
        }
        Ok(CanonicalShape { dims })
    }

    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        sink.usize(self.dims.len());
        for dim in &self.dims {
            match dim {
                CanonicalDim::Static(value) => {
                    sink.u8(0);
                    sink.u64(*value);
                }
                CanonicalDim::Symbol(name) => {
                    sink.u8(1);
                    sink.str(name);
                }
                CanonicalDim::Dynamic => sink.u8(2),
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalDevice {
    Cpu,
    Cuda(u32),
}

impl CanonicalDevice {
    fn from_device(device: Device) -> Self {
        match device {
            Device::Cpu => CanonicalDevice::Cpu,
            Device::Cuda(index) => CanonicalDevice::Cuda(index),
        }
    }

    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        match self {
            CanonicalDevice::Cpu => sink.u8(0),
            CanonicalDevice::Cuda(index) => {
                sink.u8(1);
                sink.u32(*index);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalValueInfo {
    pub shape: CanonicalShape,
    pub dtype: DType,
    pub device: CanonicalDevice,
}

impl CanonicalValueInfo {
    fn from_value(
        value: &ValueInfo,
        symbols: &HashMap<SymbolId, String>,
    ) -> Result<Self, GraphHashError> {
        Ok(CanonicalValueInfo {
            shape: CanonicalShape::from_shape(&value.shape, symbols)?,
            dtype: value.dtype,
            device: CanonicalDevice::from_device(value.device),
        })
    }

    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        self.shape.encode(sink);
        sink.u8(dtype_tag(self.dtype));
        self.device.encode(sink);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalExternalKind {
    Input,
    Parameter { trainable: bool },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalExternal {
    pub kind: CanonicalExternalKind,
    pub name: String,
    pub value: CanonicalValueInfo,
}

impl CanonicalExternal {
    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        match self.kind {
            CanonicalExternalKind::Input => sink.u8(0),
            CanonicalExternalKind::Parameter { trainable } => {
                sink.u8(1);
                sink.u8(u8::from(trainable));
            }
        }
        sink.str(&self.name);
        self.value.encode(sink);
    }

    fn fingerprint(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(EXTERNAL_DOMAIN);
        self.encode(&mut hasher);
        *hasher.finalize().as_bytes()
    }
}

/// Constant payload in canonical bit form.
///
/// Equality is bitwise: `0.0` and `-0.0` differ, and two NaNs are equal only
/// when their payload bits match. This makes the graph identity independent of
/// host float comparison semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalConstantData {
    F32(Vec<u32>),
    F64(Vec<u64>),
}

impl CanonicalConstantData {
    fn from_constant(data: &ConstantData) -> Self {
        match data {
            ConstantData::F32(values) => {
                CanonicalConstantData::F32(values.iter().map(|value| value.to_bits()).collect())
            }
            ConstantData::F64(values) => {
                CanonicalConstantData::F64(values.iter().map(|value| value.to_bits()).collect())
            }
        }
    }

    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        match self {
            CanonicalConstantData::F32(values) => {
                sink.u8(0);
                sink.usize(values.len());
                for value in values {
                    sink.u32(*value);
                }
            }
            CanonicalConstantData::F64(values) => {
                sink.u8(1);
                sink.usize(values.len());
                for value in values {
                    sink.u64(*value);
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalConstant {
    pub shape: CanonicalShape,
    pub data: CanonicalConstantData,
}

impl CanonicalConstant {
    fn from_constant(
        value: &ConstantTensor,
        symbols: &HashMap<SymbolId, String>,
    ) -> Result<Self, GraphHashError> {
        Ok(CanonicalConstant {
            shape: CanonicalShape::from_shape(&value.shape, symbols)?,
            data: CanonicalConstantData::from_constant(&value.data),
        })
    }

    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        self.shape.encode(sink);
        self.data.encode(sink);
    }
}

/// Operator attributes with symbolic shapes canonicalized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalAttrs {
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
        input_shape: CanonicalShape,
    },
    EmbeddingBackward {
        vocab: usize,
    },
    Cast {
        dtype: u8,
    },
    Concat {
        axis: usize,
    },
    Split {
        axis: usize,
        sizes: Vec<usize>,
    },
    MoE {
        num_experts: usize,
        top_k: usize,
        intermediate: usize,
        eps_bits: u32,
    },
    Custom {
        name: String,
        input_count: usize,
        outputs: Vec<CanonicalShape>,
        dtype: u8,
    },
    LayerNorm {
        normalized_shape: Vec<usize>,
        /// Bit pattern of the epsilon value.
        eps_bits: u32,
    },
    Reshape {
        shape: CanonicalShape,
    },
    Transpose {
        axes: Vec<usize>,
    },
    BroadcastTo {
        shape: CanonicalShape,
    },
    Loss {
        reduction: Reduction,
    },
    ActivationBackward {
        kind: ActivationKind,
    },
    Unbroadcast {
        shape: CanonicalShape,
    },
    Constant {
        value: CanonicalConstant,
    },
    ConstantScalar {
        value_bits: u32,
    },
}

impl CanonicalAttrs {
    pub fn from_attrs(
        attrs: &OpAttrs,
        symbols: &HashMap<SymbolId, String>,
    ) -> Result<Self, GraphHashError> {
        Ok(match attrs {
            OpAttrs::None => CanonicalAttrs::None,
            OpAttrs::MatMul {
                transpose_a,
                transpose_b,
            } => CanonicalAttrs::MatMul {
                transpose_a: *transpose_a,
                transpose_b: *transpose_b,
            },
            OpAttrs::Conv2d {
                stride,
                padding,
                dilation,
                groups,
            } => CanonicalAttrs::Conv2d {
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
                groups: *groups,
            },
            OpAttrs::Pool2d {
                kernel,
                stride,
                padding,
                dilation,
            } => CanonicalAttrs::Pool2d {
                kernel: *kernel,
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
            },
            OpAttrs::Reduce { axes, keep_dims } => CanonicalAttrs::Reduce {
                axes: axes.clone(),
                keep_dims: *keep_dims,
            },
            OpAttrs::Softmax { axis } => CanonicalAttrs::Softmax { axis: *axis },
            OpAttrs::LogSoftmax { axis } => CanonicalAttrs::LogSoftmax { axis: *axis },
            OpAttrs::Slice { axis, start, len } => CanonicalAttrs::Slice {
                axis: *axis,
                start: *start,
                len: *len,
            },
            OpAttrs::SliceBackward {
                axis,
                start,
                len,
                input_shape,
            } => CanonicalAttrs::SliceBackward {
                axis: *axis,
                start: *start,
                len: *len,
                input_shape: CanonicalShape::from_shape(input_shape, symbols)?,
            },
            OpAttrs::EmbeddingBackward { vocab } => {
                CanonicalAttrs::EmbeddingBackward { vocab: *vocab }
            }
            OpAttrs::Cast { dtype } => CanonicalAttrs::Cast {
                dtype: dtype_tag(*dtype),
            },
            OpAttrs::Concat { axis } => CanonicalAttrs::Concat { axis: *axis },
            OpAttrs::Split { axis, sizes } => CanonicalAttrs::Split {
                axis: *axis,
                sizes: sizes.clone(),
            },
            OpAttrs::MoE {
                num_experts,
                top_k,
                intermediate,
                eps,
            } => CanonicalAttrs::MoE {
                num_experts: *num_experts,
                top_k: *top_k,
                intermediate: *intermediate,
                eps_bits: eps.to_bits(),
            },
            OpAttrs::Custom {
                name,
                input_count,
                outputs,
                dtype,
            } => CanonicalAttrs::Custom {
                name: name.clone(),
                input_count: *input_count,
                outputs: outputs
                    .iter()
                    .map(|shape| CanonicalShape::from_shape(shape, symbols))
                    .collect::<Result<Vec<_>, _>>()?,
                dtype: dtype_tag(*dtype),
            },
            OpAttrs::LayerNorm {
                normalized_shape,
                eps,
            } => CanonicalAttrs::LayerNorm {
                normalized_shape: normalized_shape.clone(),
                eps_bits: eps.to_bits(),
            },
            OpAttrs::Reshape { shape } => CanonicalAttrs::Reshape {
                shape: CanonicalShape::from_shape(shape, symbols)?,
            },
            OpAttrs::Transpose { axes } => CanonicalAttrs::Transpose { axes: axes.clone() },
            OpAttrs::BroadcastTo { shape } => CanonicalAttrs::BroadcastTo {
                shape: CanonicalShape::from_shape(shape, symbols)?,
            },
            OpAttrs::Loss { reduction } => CanonicalAttrs::Loss {
                reduction: *reduction,
            },
            OpAttrs::ActivationBackward { kind } => {
                CanonicalAttrs::ActivationBackward { kind: *kind }
            }
            OpAttrs::Unbroadcast { shape } => CanonicalAttrs::Unbroadcast {
                shape: CanonicalShape::from_shape(shape, symbols)?,
            },
            OpAttrs::Constant { value } => CanonicalAttrs::Constant {
                value: CanonicalConstant::from_constant(value, symbols)?,
            },
            OpAttrs::ConstantScalar { value } => CanonicalAttrs::ConstantScalar {
                value_bits: value.to_bits(),
            },
        })
    }

    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        match self {
            CanonicalAttrs::None => sink.u8(0),
            CanonicalAttrs::MatMul {
                transpose_a,
                transpose_b,
            } => {
                sink.u8(1);
                sink.u8(u8::from(*transpose_a));
                sink.u8(u8::from(*transpose_b));
            }
            CanonicalAttrs::Conv2d {
                stride,
                padding,
                dilation,
                groups,
            } => {
                sink.u8(2);
                encode_usize_pair(sink, *stride);
                encode_usize_pair(sink, *padding);
                encode_usize_pair(sink, *dilation);
                sink.usize(*groups);
            }
            CanonicalAttrs::Pool2d {
                kernel,
                stride,
                padding,
                dilation,
            } => {
                sink.u8(3);
                encode_usize_pair(sink, *kernel);
                encode_usize_pair(sink, *stride);
                encode_usize_pair(sink, *padding);
                encode_usize_pair(sink, *dilation);
            }
            CanonicalAttrs::Reduce { axes, keep_dims } => {
                sink.u8(4);
                encode_usize_vec(sink, axes);
                sink.u8(u8::from(*keep_dims));
            }
            CanonicalAttrs::Softmax { axis } => {
                sink.u8(5);
                sink.u32(*axis as u32);
            }
            CanonicalAttrs::LayerNorm {
                normalized_shape,
                eps_bits,
            } => {
                sink.u8(6);
                encode_usize_vec(sink, normalized_shape);
                sink.u32(*eps_bits);
            }
            CanonicalAttrs::Reshape { shape } => {
                sink.u8(7);
                shape.encode(sink);
            }
            CanonicalAttrs::Transpose { axes } => {
                sink.u8(8);
                encode_usize_vec(sink, axes);
            }
            CanonicalAttrs::BroadcastTo { shape } => {
                sink.u8(9);
                shape.encode(sink);
            }
            CanonicalAttrs::Loss { reduction } => {
                sink.u8(10);
                sink.u8(reduction_tag(*reduction));
            }
            CanonicalAttrs::ActivationBackward { kind } => {
                sink.u8(11);
                sink.u8(activation_tag(*kind));
            }
            CanonicalAttrs::Unbroadcast { shape } => {
                sink.u8(12);
                shape.encode(sink);
            }
            CanonicalAttrs::Constant { value } => {
                sink.u8(13);
                value.encode(sink);
            }
            CanonicalAttrs::ConstantScalar { value_bits } => {
                sink.u8(14);
                sink.u32(*value_bits);
            }
            CanonicalAttrs::LogSoftmax { axis } => {
                sink.u8(15);
                sink.u32(*axis as u32);
            }
            CanonicalAttrs::Slice { axis, start, len } => {
                sink.u8(16);
                sink.usize(*axis);
                sink.usize(*start);
                sink.usize(*len);
            }
            CanonicalAttrs::SliceBackward {
                axis,
                start,
                len,
                input_shape,
            } => {
                sink.u8(17);
                sink.usize(*axis);
                sink.usize(*start);
                sink.usize(*len);
                input_shape.encode(sink);
            }
            CanonicalAttrs::EmbeddingBackward { vocab } => {
                sink.u8(18);
                sink.usize(*vocab);
            }
            CanonicalAttrs::Cast { dtype } => {
                sink.u8(20);
                sink.u8(*dtype);
            }
            CanonicalAttrs::Concat { axis } => {
                sink.u8(21);
                sink.usize(*axis);
            }
            CanonicalAttrs::Split { axis, sizes } => {
                sink.u8(22);
                sink.usize(*axis);
                encode_usize_vec(sink, sizes);
            }
            CanonicalAttrs::MoE {
                num_experts,
                top_k,
                intermediate,
                eps_bits,
            } => {
                sink.u8(23);
                sink.usize(*num_experts);
                sink.usize(*top_k);
                sink.usize(*intermediate);
                sink.u32(*eps_bits);
            }
            CanonicalAttrs::Custom {
                name,
                input_count,
                outputs,
                dtype,
            } => {
                sink.u8(19);
                sink.str(name);
                sink.usize(*input_count);
                sink.usize(outputs.len());
                for shape in outputs {
                    shape.encode(sink);
                }
                sink.u8(*dtype);
            }
        }
    }
}

/// One canonical node. Structurally identical nodes are hash-consed and keep
/// their occurrence count, so two identical independent computations are
/// distinguishable from one shared computation while remaining order-free.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalNode {
    pub op: Op,
    pub attrs: CanonicalAttrs,
    pub inputs: Vec<CanonicalValueRef>,
    pub outputs: Vec<CanonicalValueInfo>,
    pub occurrences: u32,
}

impl CanonicalNode {
    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        sink.u16(op_tag(&self.op));
        self.attrs.encode(sink);
        sink.usize(self.inputs.len());
        for input in &self.inputs {
            encode_value_ref(sink, *input);
        }
        sink.usize(self.outputs.len());
        for output in &self.outputs {
            output.encode(sink);
        }
        sink.u32(self.occurrences);
    }
}

/// Complete canonical graph descriptor.
///
/// The `value_map` and `node_map` fields are lookup aids for memory-plan
/// canonicalization and do not participate in equality or hashing.
#[derive(Clone)]
pub struct CanonicalGraph {
    pub version: u32,
    pub ir_version: u32,
    pub symbols: Vec<String>,
    pub inputs: Vec<CanonicalExternal>,
    pub parameters: Vec<CanonicalExternal>,
    pub nodes: Vec<CanonicalNode>,
    pub outputs: Vec<CanonicalValueRef>,
    value_map: HashMap<ValueId, CanonicalValueRef>,
    node_map: HashMap<NodeId, u32>,
}

impl PartialEq for CanonicalGraph {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.ir_version == other.ir_version
            && self.symbols == other.symbols
            && self.inputs == other.inputs
            && self.parameters == other.parameters
            && self.nodes == other.nodes
            && self.outputs == other.outputs
    }
}

impl Eq for CanonicalGraph {}

impl fmt::Debug for CanonicalGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CanonicalGraph")
            .field("version", &self.version)
            .field("ir_version", &self.ir_version)
            .field("symbols", &self.symbols)
            .field("inputs", &self.inputs)
            .field("parameters", &self.parameters)
            .field("nodes", &self.nodes)
            .field("outputs", &self.outputs)
            .finish_non_exhaustive()
    }
}

impl CanonicalGraph {
    pub fn value_ref(&self, value: ValueId) -> Option<CanonicalValueRef> {
        self.value_map.get(&value).copied()
    }

    pub fn node_ref(&self, node: NodeId) -> Option<u32> {
        self.node_map.get(&node).copied()
    }

    /// Computes the BLAKE3 structure hash with domain separation and version.
    pub fn hash(&self) -> GraphHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(GRAPH_HASH_DOMAIN);
        self.encode(&mut hasher);
        GraphHash(*hasher.finalize().as_bytes())
    }

    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        sink.u32(self.version);
        sink.u32(self.ir_version);
        sink.usize(self.symbols.len());
        for symbol in &self.symbols {
            sink.str(symbol);
        }
        sink.usize(self.inputs.len());
        for input in &self.inputs {
            input.encode(sink);
        }
        sink.usize(self.parameters.len());
        for parameter in &self.parameters {
            parameter.encode(sink);
        }
        sink.usize(self.nodes.len());
        for node in &self.nodes {
            node.encode(sink);
        }
        sink.usize(self.outputs.len());
        for output in &self.outputs {
            encode_value_ref(sink, *output);
        }
    }
}

/// Canonical graph hash.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GraphHash(pub [u8; 32]);

impl GraphHash {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use fmt::Write;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

impl fmt::Debug for GraphHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GraphHash({})", self.to_hex())
    }
}

impl fmt::Display for GraphHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphHashError {
    InvalidGraph(String),
    UnnamedInput(ValueId),
    UnnamedParameter(ParameterId),
    UnknownSymbol(SymbolId),
    UnknownValue(ValueId),
    UnknownNode(NodeId),
}

impl fmt::Display for GraphHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphHashError::InvalidGraph(message) => {
                write!(f, "cannot canonicalize graph: {message}")
            }
            GraphHashError::UnnamedInput(value) => {
                write!(f, "graph input {value} has no stable name")
            }
            GraphHashError::UnnamedParameter(id) => {
                write!(f, "graph parameter {id} has no stable name")
            }
            GraphHashError::UnknownSymbol(id) => {
                write!(f, "shape references unknown symbol {id:?}")
            }
            GraphHashError::UnknownValue(id) => write!(f, "graph references unknown value {id}"),
            GraphHashError::UnknownNode(id) => write!(f, "graph references unknown node {id}"),
        }
    }
}

impl std::error::Error for GraphHashError {}

impl From<Error> for GraphHashError {
    fn from(value: Error) -> Self {
        GraphHashError::InvalidGraph(value.to_string())
    }
}

impl From<GraphHashError> for Error {
    fn from(value: GraphHashError) -> Self {
        Error::invalid_graph(value.to_string())
    }
}

/// Builds and hashes the canonical graph in one call.
pub fn graph_structure_hash(graph: &Graph) -> Result<GraphHash, GraphHashError> {
    Ok(graph.canonicalize()?.hash())
}

impl Graph {
    /// Hash of the graph's raw execution numbering.
    ///
    /// Unlike [`Graph::canonicalize`], this intentionally includes original
    /// `NodeId`/`ValueId` numbering. Persistent arenas store ValueId-keyed
    /// assignments, so this hash guards against attaching a plan produced from
    /// a semantically equal but differently numbered graph instance.
    pub fn raw_identity_hash(&self) -> Result<[u8; 32], GraphHashError> {
        self.validate()?;
        let symbol_names: HashMap<SymbolId, String> = self
            .symbols
            .iter()
            .map(|symbol| (symbol.id, symbol.name.clone()))
            .collect();
        let mut hasher = blake3::Hasher::new();
        hasher.update(RAW_GRAPH_IDENTITY_DOMAIN);
        hasher.u32(self.format_version);
        hasher.u32(self.ir_version);
        hasher.usize(self.symbols.len());
        for symbol in &self.symbols {
            hasher.u32(symbol.id.0);
            hasher.str(&symbol.name);
        }
        hasher.usize(self.inputs.len());
        for &input in &self.inputs {
            encode_raw_value(&mut hasher, self.value(input)?, &symbol_names)?;
        }
        hasher.usize(self.parameters.len());
        for parameter in &self.parameters {
            hasher.u32(parameter.id.0);
            hasher.u32(parameter.value.0);
            hasher.str(&parameter.name);
            encode_raw_shape(&mut hasher, &parameter.shape);
            hasher.u8(dtype_tag(parameter.dtype));
            hasher.u8(u8::from(parameter.trainable));
        }
        hasher.usize(self.values.len());
        for value in &self.values {
            encode_raw_value(&mut hasher, value, &symbol_names)?;
        }
        hasher.usize(self.nodes.len());
        for node in &self.nodes {
            hasher.u32(node.id.0);
            hasher.u16(op_tag(&node.op));
            CanonicalAttrs::from_attrs(&node.attrs, &symbol_names)?.encode(&mut hasher);
            hasher.usize(node.inputs.len());
            for input in &node.inputs {
                hasher.u32(input.0);
            }
            hasher.usize(node.outputs.len());
            for output in &node.outputs {
                hasher.u32(output.0);
            }
        }
        hasher.usize(self.outputs.len());
        for output in &self.outputs {
            hasher.u32(output.0);
        }
        Ok(*hasher.finalize().as_bytes())
    }

    /// Builds a construction-order-independent canonical representation.
    ///
    /// Nodes are identified by a BLAKE3 fingerprint of their operator,
    /// canonical attributes, input producer fingerprints and output metadata.
    /// Identical fingerprints are hash-consed. The resulting node list is
    /// sorted by fingerprint, so no HashMap iteration order or original
    /// NodeId/ValueId reaches the hash.
    pub fn canonicalize(&self) -> Result<CanonicalGraph, GraphHashError> {
        self.validate()?;

        let mut symbol_names: HashMap<SymbolId, String> = HashMap::new();
        let mut sorted_symbols = Vec::with_capacity(self.symbols.len());
        for symbol in &self.symbols {
            if symbol.name.is_empty() {
                return Err(GraphHashError::InvalidGraph(
                    "symbol name must not be empty".into(),
                ));
            }
            symbol_names.insert(symbol.id, symbol.name.clone());
            sorted_symbols.push(symbol.name.clone());
        }
        sorted_symbols.sort();
        sorted_symbols.dedup();

        let mut external_fingerprints: HashMap<ValueId, [u8; 32]> = HashMap::new();
        let mut external_refs: HashMap<[u8; 32], CanonicalValueRef> = HashMap::new();
        let mut inputs = Vec::with_capacity(self.inputs.len());
        let mut parameters = Vec::with_capacity(self.parameters.len());

        for &input_id in &self.inputs {
            let value = self.value(input_id)?;
            let name = value
                .name
                .clone()
                .ok_or(GraphHashError::UnnamedInput(input_id))?;
            let external = CanonicalExternal {
                kind: CanonicalExternalKind::Input,
                name,
                value: CanonicalValueInfo::from_value(value, &symbol_names)?,
            };
            let fingerprint = external.fingerprint();
            external_fingerprints.insert(input_id, fingerprint);
            inputs.push((fingerprint, external, input_id));
        }
        for parameter in &self.parameters {
            let value = self.value(parameter.value)?;
            let external = CanonicalExternal {
                kind: CanonicalExternalKind::Parameter {
                    trainable: parameter.trainable,
                },
                name: parameter.name.clone(),
                value: CanonicalValueInfo::from_value(value, &symbol_names)?,
            };
            let fingerprint = external.fingerprint();
            external_fingerprints.insert(value.id, fingerprint);
            parameters.push((fingerprint, external, value.id));
        }

        // Sort externals by their full canonical encoding, not by ids.
        inputs.sort_by(|a, b| {
            let mut left = Vec::new();
            a.1.encode(&mut left);
            let mut right = Vec::new();
            b.1.encode(&mut right);
            left.cmp(&right)
        });
        parameters.sort_by(|a, b| {
            let mut left = Vec::new();
            a.1.encode(&mut left);
            let mut right = Vec::new();
            b.1.encode(&mut right);
            left.cmp(&right)
        });
        for (index, (fingerprint, _, _)) in inputs.iter().enumerate() {
            external_refs
                .entry(*fingerprint)
                .or_insert(CanonicalValueRef::Input(index as u32));
        }
        for (index, (fingerprint, _, _)) in parameters.iter().enumerate() {
            external_refs
                .entry(*fingerprint)
                .or_insert(CanonicalValueRef::Parameter(index as u32));
        }

        let mut value_producer: HashMap<ValueId, (usize, u32)> = HashMap::new();
        for (node_index, node) in self.nodes().iter().enumerate() {
            for (output_index, &output) in node.outputs.iter().enumerate() {
                value_producer.insert(output, (node_index, output_index as u32));
            }
        }

        let mut computed: Vec<ComputedNode> = Vec::with_capacity(self.nodes.len());
        for node in self.nodes() {
            let attrs = CanonicalAttrs::from_attrs(&node.attrs, &symbol_names)?;
            let mut input_keys = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                if let Some(&(producer, output_index)) = value_producer.get(&input) {
                    let fingerprint = computed
                        .get(producer)
                        .map(|node| node.fingerprint)
                        .ok_or(GraphHashError::UnknownValue(input))?;
                    input_keys.push(InputKey::Node {
                        fingerprint,
                        output: output_index,
                    });
                } else if let Some(fingerprint) = external_fingerprints.get(&input) {
                    input_keys.push(InputKey::Fingerprint(*fingerprint));
                } else {
                    return Err(GraphHashError::UnknownValue(input));
                }
            }
            let outputs = node
                .outputs
                .iter()
                .map(|&output| {
                    let value = self.value(output)?;
                    CanonicalValueInfo::from_value(value, &symbol_names)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut hasher = blake3::Hasher::new();
            hasher.update(NODE_DOMAIN);
            hasher.u16(op_tag(&node.op));
            attrs.encode(&mut hasher);
            for key in &input_keys {
                key.encode(&mut hasher);
            }
            hasher.usize(outputs.len());
            for output in &outputs {
                output.encode(&mut hasher);
            }
            let fingerprint = *hasher.finalize().as_bytes();
            computed.push(ComputedNode {
                node_id: node.id,
                op: node.op.clone(),
                attrs,
                input_keys,
                outputs,
                output_values: node.outputs.clone(),
                fingerprint,
            });
        }

        // Hash-cons identical nodes. The representative index keeps the
        // original computed metadata. A deterministic topological sort with a
        // fingerprint tie-break supplies the construction-order-independent
        // node order below.
        let mut groups: BTreeMap<[u8; 32], NodeGroup> = BTreeMap::new();
        for (index, node) in computed.iter().enumerate() {
            groups
                .entry(node.fingerprint)
                .and_modify(|group| group.occurrences = group.occurrences.saturating_add(1))
                .or_insert(NodeGroup {
                    representative: index,
                    occurrences: 1,
                });
        }

        // Deterministic topological order: among all currently-ready nodes,
        // pick the smallest fingerprint. Producers therefore precede consumers
        // while independent nodes get a stable tie-break.
        let mut dependencies: HashMap<[u8; 32], Vec<[u8; 32]>> = HashMap::new();
        let mut indegree: HashMap<[u8; 32], usize> = HashMap::new();
        for (fingerprint, group) in &groups {
            indegree.entry(*fingerprint).or_insert(0);
            let representative = &computed[group.representative];
            for key in &representative.input_keys {
                if let InputKey::Node {
                    fingerprint: producer,
                    ..
                } = key
                {
                    dependencies
                        .entry(*producer)
                        .or_default()
                        .push(*fingerprint);
                    *indegree.entry(*fingerprint).or_insert(0) += 1;
                }
            }
        }
        let mut ready: BTreeSet<[u8; 32]> = indegree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(fingerprint, _)| *fingerprint)
            .collect();
        let mut order = Vec::with_capacity(groups.len());
        while let Some(fingerprint) = ready.pop_first() {
            order.push(fingerprint);
            if let Some(consumers) = dependencies.get(&fingerprint) {
                for consumer in consumers {
                    let degree = indegree.get_mut(consumer).ok_or_else(|| {
                        GraphHashError::InvalidGraph(
                            "canonical dependency graph is inconsistent".into(),
                        )
                    })?;
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.insert(*consumer);
                    }
                }
            }
        }
        if order.len() != groups.len() {
            return Err(GraphHashError::InvalidGraph(
                "canonical graph contains a dependency cycle".into(),
            ));
        }

        let mut canonical_index: HashMap<[u8; 32], u32> = HashMap::new();
        for (index, fingerprint) in order.iter().enumerate() {
            canonical_index.insert(*fingerprint, index as u32);
        }

        let mut node_map: HashMap<NodeId, u32> = HashMap::new();
        let mut value_map: HashMap<ValueId, CanonicalValueRef> = HashMap::new();
        for node in computed.iter() {
            let index = canonical_index[&node.fingerprint];
            node_map.insert(node.node_id, index);
            for (output_index, &value) in node.output_values.iter().enumerate() {
                value_map.insert(
                    value,
                    CanonicalValueRef::Node {
                        node: index,
                        output: output_index as u32,
                    },
                );
            }
        }

        for (input_index, (_, _, value)) in inputs.iter().enumerate() {
            value_map.insert(*value, CanonicalValueRef::Input(input_index as u32));
        }
        for (parameter_index, (_, _, value)) in parameters.iter().enumerate() {
            value_map.insert(*value, CanonicalValueRef::Parameter(parameter_index as u32));
        }

        let mut nodes = Vec::with_capacity(groups.len());
        for fingerprint in &order {
            let group = groups
                .get(fingerprint)
                .ok_or_else(|| GraphHashError::InvalidGraph("canonical node disappeared".into()))?;
            let representative = &computed[group.representative];
            let mut node_inputs = Vec::with_capacity(representative.input_keys.len());
            for key in &representative.input_keys {
                node_inputs.push(match key {
                    InputKey::Fingerprint(fingerprint) => {
                        *external_refs
                            .get(fingerprint)
                            .ok_or(GraphHashError::InvalidGraph(
                                "node input references an unknown external".into(),
                            ))?
                    }
                    InputKey::Node {
                        fingerprint,
                        output,
                    } => CanonicalValueRef::Node {
                        node: canonical_index.get(fingerprint).copied().ok_or(
                            GraphHashError::InvalidGraph(
                                "node input references an unknown producer".into(),
                            ),
                        )?,
                        output: *output,
                    },
                });
            }
            nodes.push(CanonicalNode {
                op: representative.op.clone(),
                attrs: representative.attrs.clone(),
                inputs: node_inputs,
                outputs: representative.outputs.clone(),
                occurrences: group.occurrences,
            });
        }

        let outputs = self
            .outputs
            .iter()
            .map(|&output| {
                value_map
                    .get(&output)
                    .copied()
                    .ok_or(GraphHashError::UnknownValue(output))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(CanonicalGraph {
            version: CANONICAL_GRAPH_VERSION,
            ir_version: self.ir_version,
            symbols: sorted_symbols,
            inputs: inputs
                .into_iter()
                .map(|(_, external, _)| external)
                .collect(),
            parameters: parameters
                .into_iter()
                .map(|(_, external, _)| external)
                .collect(),
            nodes,
            outputs,
            value_map,
            node_map,
        })
    }
}

struct ComputedNode {
    node_id: NodeId,
    op: Op,
    attrs: CanonicalAttrs,
    input_keys: Vec<InputKey>,
    outputs: Vec<CanonicalValueInfo>,
    output_values: Vec<ValueId>,
    fingerprint: [u8; 32],
}

struct NodeGroup {
    representative: usize,
    occurrences: u32,
}

#[derive(Clone, Copy)]
enum InputKey {
    Fingerprint([u8; 32]),
    Node { fingerprint: [u8; 32], output: u32 },
}

impl InputKey {
    fn encode<S: EncodeSink>(&self, sink: &mut S) {
        match self {
            InputKey::Fingerprint(fingerprint) => {
                sink.u8(0);
                sink.bytes(fingerprint);
            }
            InputKey::Node {
                fingerprint,
                output,
            } => {
                sink.u8(1);
                sink.bytes(fingerprint);
                sink.u32(*output);
            }
        }
    }
}

fn encode_raw_value<S: EncodeSink>(
    sink: &mut S,
    value: &ValueInfo,
    symbols: &HashMap<SymbolId, String>,
) -> Result<(), GraphHashError> {
    sink.u32(value.id.0);
    match &value.name {
        Some(name) => {
            sink.u8(1);
            sink.str(name);
        }
        None => sink.u8(0),
    }
    let shape = CanonicalShape::from_shape(&value.shape, symbols)?;
    shape.encode(sink);
    sink.u8(dtype_tag(value.dtype));
    CanonicalDevice::from_device(value.device).encode(sink);
    match value.kind {
        ValueKind::Input => sink.u8(0),
        ValueKind::Parameter(id) => {
            sink.u8(1);
            sink.u32(id.0);
        }
        ValueKind::Constant => sink.u8(2),
        ValueKind::Intermediate => sink.u8(3),
    }
    Ok(())
}

fn encode_raw_shape<S: EncodeSink>(sink: &mut S, shape: &Shape) {
    sink.usize(shape.rank());
    for dim in shape.dims() {
        match *dim {
            Dim::Static(value) => {
                sink.u8(0);
                sink.u64(value as u64);
            }
            Dim::Symbol(id) => {
                sink.u8(1);
                sink.u32(id.0);
            }
            Dim::Dynamic => sink.u8(2),
        }
    }
}

fn encode_value_ref<S: EncodeSink>(sink: &mut S, value: CanonicalValueRef) {
    match value {
        CanonicalValueRef::Input(index) => {
            sink.u8(0);
            sink.u32(index);
        }
        CanonicalValueRef::Parameter(index) => {
            sink.u8(1);
            sink.u32(index);
        }
        CanonicalValueRef::Node { node, output } => {
            sink.u8(2);
            sink.u32(node);
            sink.u32(output);
        }
    }
}

fn encode_usize_pair<S: EncodeSink>(sink: &mut S, value: [usize; 2]) {
    sink.usize(value[0]);
    sink.usize(value[1]);
}

fn encode_usize_vec<S: EncodeSink>(sink: &mut S, values: &[usize]) {
    sink.usize(values.len());
    for value in values {
        sink.usize(*value);
    }
}

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::F64 => 3,
        DType::I8 => 4,
        DType::I16 => 5,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::U8 => 8,
        DType::Bool => 9,
    }
}

fn reduction_tag(reduction: Reduction) -> u8 {
    match reduction {
        Reduction::Mean => 0,
        Reduction::Sum => 1,
    }
}

fn activation_tag(kind: ActivationKind) -> u8 {
    match kind {
        ActivationKind::ReLU => 0,
        ActivationKind::GELU => 1,
        ActivationKind::Sigmoid => 2,
        ActivationKind::Tanh => 3,
    }
}

fn op_tag(op: &Op) -> u16 {
    match op {
        Op::Add => 0,
        Op::AddRelu => 1,
        Op::Sub => 2,
        Op::Mul => 3,
        Op::Div => 4,
        Op::Neg => 5,
        Op::MatMul => 6,
        Op::ReLU => 7,
        Op::GELU => 8,
        Op::Sigmoid => 9,
        Op::Tanh => 10,
        Op::Softmax => 11,
        Op::LayerNorm => 12,
        Op::Reshape => 13,
        Op::Transpose => 14,
        Op::BroadcastTo => 15,
        Op::ReduceSum => 16,
        Op::ReduceMean => 17,
        Op::Conv2d => 18,
        Op::MaxPool2d => 19,
        Op::AvgPool2d => 20,
        Op::MSELoss => 21,
        Op::CrossEntropyLoss => 22,
        Op::Constant => 23,
        Op::ConstantScalar => 24,
        Op::ActivationBackward(kind) => 25 + u16::from(activation_tag(*kind)),
        Op::AddReluBackward => 29,
        Op::SoftmaxBackward => 30,
        Op::LayerNormBackward => 31,
        Op::MSELossBackward => 32,
        Op::CrossEntropyLossBackward => 33,
        Op::Conv2dBackward => 34,
        Op::MaxPool2dBackward => 35,
        Op::AvgPool2dBackward => 36,
        Op::Unbroadcast => 37,
        Op::Sqrt => 38,
        Op::Exp => 39,
        Op::Log => 40,
        Op::Abs => 41,
        Op::LogSoftmax => 42,
        Op::Slice => 43,
        Op::Embedding => 44,
        Op::AbsBackward => 45,
        Op::SliceBackward => 46,
        Op::EmbeddingBackward => 47,
        Op::Custom => 48,
        Op::Cast => 49,
        Op::Concat => 50,
        Op::Split => 51,
        Op::MulAdd => 52,
        Op::MatMulBias => 53,
        Op::MoE => 54,
    }
}

pub(crate) trait EncodeSink {
    fn u8(&mut self, value: u8);
    fn u16(&mut self, value: u16);
    fn u32(&mut self, value: u32);
    fn u64(&mut self, value: u64);

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn bytes(&mut self, value: &[u8]);

    fn str(&mut self, value: &str) {
        self.usize(value.len());
        self.bytes(value.as_bytes());
    }
}

impl EncodeSink for Vec<u8> {
    fn u8(&mut self, value: u8) {
        self.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.extend_from_slice(&value.to_le_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.extend_from_slice(value);
    }
}

impl EncodeSink for blake3::Hasher {
    fn u8(&mut self, value: u8) {
        self.update(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.update(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.update(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.update(&value.to_le_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.update(value);
    }
}
