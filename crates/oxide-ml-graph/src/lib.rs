//! SSA graph IR and shape-inference front end.
//!
//! The IR is backend-independent. Execution backends receive `Op`/`OpAttrs`
//! through the kernel ABI in `oxide-ml-kernel`; no CUDA type appears here.

mod builder;
pub mod canonical;
mod infer;
mod ir;
mod validate;
mod verifier;

pub use builder::{Conv2dConfig, GraphBuilder, Pool2dConfig};
pub use canonical::{
    CANONICAL_GRAPH_VERSION, CanonicalAttrs, CanonicalConstant, CanonicalConstantData,
    CanonicalDevice, CanonicalDim, CanonicalExternal, CanonicalExternalKind, CanonicalGraph,
    CanonicalNode, CanonicalShape, CanonicalValueInfo, CanonicalValueRef, GraphHash,
    GraphHashError, graph_structure_hash,
};
pub use infer::{Inferred, infer_outputs, normalize_axes, normalize_axis};
pub use ir::{
    ActivationKind, ConstantData, ConstantTensor, GRAPH_FORMAT_VERSION, Graph, IR_VERSION, Node,
    Op, OpAttrs, ParameterInfo, Reduction, SERIALIZATION_FORMAT_VERSION, SymbolInfo, ValueInfo,
    ValueKind,
};
pub use verifier::{GraphVerifier, VerificationReport};
