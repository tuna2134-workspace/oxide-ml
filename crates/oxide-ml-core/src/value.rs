use serde::{Deserialize, Serialize};

use crate::{DType, Device, Shape};

/// Stable identifier of an SSA value inside one graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Stable identifier of a graph node inside one graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Stable identifier of a trainable parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ParameterId(pub u32);

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node{}", self.0)
    }
}

impl std::fmt::Display for ParameterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "param{}", self.0)
    }
}

/// Metadata for a value in the graph. Tensor bytes never live here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorValue {
    pub id: ValueId,
    pub shape: Shape,
    pub dtype: DType,
    pub device: Device,
}
