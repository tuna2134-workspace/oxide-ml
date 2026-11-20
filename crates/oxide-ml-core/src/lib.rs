//! Core value types shared by every oxide-ml crate.
//!
//! This crate deliberately has no dependency on the graph IR or on a backend.
//! It defines the type-level vocabulary used at the API boundary: dtypes,
//! devices, symbolic shapes, dense reference tensors, value ids, errors and the
//! parameter store.

mod device;
mod dtype;
mod error;
mod memory;
mod parameter;
mod rng;
mod scalar;
mod shape;
mod tensor;
mod value;

pub use device::Device;
pub use dtype::DType;
pub use error::{Error, Result};
pub use memory::{
    BufferAssignment, BufferId, BufferSlot, LifetimePoint, MemoryPlan, ValueLifetime,
};
pub use parameter::{Parameter, ParameterStore};
pub use rng::Rng;
pub use scalar::Scalar;
pub use shape::{Dim, Shape, SymbolId};
pub use tensor::{DenseTensor, Tensor};
pub use value::{NodeId, ParameterId, TensorValue, ValueId};

/// Re-exported so downstream crates can build the same shape bindings map
/// without each depending on `std::collections` conventions directly.
pub type SymbolBindings = std::collections::HashMap<SymbolId, usize>;

/// Re-exported for F16/BF16 tensor construction and inspection.
pub use half;
