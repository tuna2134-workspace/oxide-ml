// Kernel launch wrappers mirror the CUDA ABI and can legitimately exceed
// clippy's argument-count and type-complexity heuristics.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

//! CUDA backend for oxide-ml.
//!
//! Device kernels are written with NVIDIA cuda-oxide (`#[kernel]` +
//! `#[cuda_module]`) and compiled to PTX by `cargo oxide`. Host-side memory,
//! streams, and launches use the shared `cuda-core` runtime.
//!
//! This crate is deliberately a standalone workspace: building it requires
//! the pinned Rust 1.99.1 toolchain, CUDA 13, and the cuda-oxide codegen
//! backend. The CPU workspace excludes it so `cargo test --workspace` remains
//! usable on machines without CUDA.

pub mod access;
pub mod arena;
pub mod arena_key;
pub mod arena_registry;
pub mod backend;
pub mod buffer;
pub mod device;
pub mod error;
pub mod event;
pub mod executor;
pub mod kernels;
pub mod memplan;
pub mod optimizer;
pub mod param;
pub mod trainer;

pub use access::{BufferAccess, BufferAccessState};
pub use arena::{ArenaStats, CudaPlannedArena, SlotSetLease, SlotSetState};
pub use arena_key::{ContextIdentity, DeviceIdentity};
pub use arena_registry::CudaArenaRegistry;
pub use backend::CudaBackend;
pub use buffer::{
    BufferPoolStats, BufferQuarantineRecord, CudaBufferPool, CudaTensor, PoolStats, PooledBuffer,
};
pub use device::{CudaDevice, CudaDeviceInfo, device_count, list_devices};
pub use error::{CudaError, Result};
pub use event::{CudaEvent, EventLease, EventPool, EventPoolStats, EventState};
pub use executor::{CudaExecution, CudaExecutor, CudaRun};
pub use kernels::{AdamWParams, CudaKernels};
pub use memplan::{CudaMemoryPlan, CudaMemoryStats, CudaSlotAssignment, CudaSlotPlan};
pub use optimizer::{CudaAdamW, CudaOptimizer, CudaSgd};
pub use oxide_ml_compiler::{
    ArenaContextId, ArenaDescriptor, ArenaKey, ArenaRegistryError, ArenaRegistryStats, DeviceId,
};
pub use oxide_ml_event_safety::{EventError, EventOp, ReclaimDecision, ReclaimVerdict};
pub use param::CudaParameterStore;
pub use trainer::CudaTrainer;
