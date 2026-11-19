//! Convenience imports for CUDA users.
//!
//! ```ignore
//! use oxide_ml::prelude::*;
//! use oxide_ml_cuda::prelude::*;
//!
//! #[cuda_module]
//! mod kernels {
//!     use cuda_device::{kernel, thread, DisjointSlice};
//!     #[kernel]
//!     pub fn relu(input: &[f32], mut output: DisjointSlice<f32>, n: u32) { /* ... */ }
//! }
//!
//! let device = CudaDevice::new(0)?;
//! let kernel = KernelDescriptor::new("custom_relu", DType::F32, ...);
//! # Ok::<(), oxide_ml_cuda::CudaError>(())
//! ```

pub use crate::{
    AccessGuard, AccessMode, ArenaStats, CudaAdamW, CudaBackend, CudaBufferPool, CudaDevice,
    CudaDeviceInfo, CudaError, CudaEvent, CudaExecution, CudaExecutor, CudaKernels, CudaModel,
    CudaParameterStore, CudaRun, CudaSgd, CudaTensor, CudaTrainer, CudaTrainingModel, EventLease,
    EventPool, EventState, PoolStats, Result,
};
pub use oxide_ml_kernel::{
    AutogradSupport, Dimension, KernelDescriptor, KernelOperator, LaunchConfig, SideEffect,
    SpecializationKey, TensorConstraint,
};

/// Re-exported so a user crate can write `cuda_device::...` without adding the
/// dependency directly.
pub use cuda_core;
pub use cuda_device;
pub use cuda_host;
