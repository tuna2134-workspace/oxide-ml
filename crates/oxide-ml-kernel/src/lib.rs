//! Backend ABI shared by CPU and future CUDA implementations.
//!
//! The IR stays in `oxide-ml-graph`; this crate only defines how a lowered
//! kernel call looks. A CUDA backend can implement the same trait without
//! changing the graph, autograd, or training loop.
//!
//! Hand-written kernels are described and registered through [`custom`].

use oxide_ml_core::{DType, DenseTensor, Device, Result, Scalar};
use oxide_ml_graph::{Op, OpAttrs};

pub mod custom;
pub mod provider;

pub use custom::{
    AutogradSupport, ConstraintKind, CustomBackward, CustomCall, CustomForward, CustomKernel,
    CustomKernelRegistry, Dimension, KERNEL_DESCRIPTOR_VERSION, KernelDescriptor, KernelOperator,
    LaunchConfig, SideEffect, SpecializationKey, TensorConstraint,
};
pub use provider::{
    KERNEL_PROVIDER_VERSION, KernelCapabilities, KernelCatalog, KernelProvider, KernelRequest,
    KernelSelection, ProviderKind,
};

/// Version of the Rust kernel-call ABI. Bump when `KernelCall`,
/// `KernelBackend`, or buffer-spec semantics change incompatibly.
pub const KERNEL_ABI_VERSION: u32 = 2;

/// Version of the optional capability descriptor. Capabilities can grow
/// without changing the ABI; this version lets a compiler decide whether a
/// backend advertises a feature it understands.
pub const BACKEND_CAPABILITY_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLayout {
    Contiguous,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub strides: Vec<usize>,
    pub alignment: usize,
    pub layout: MemoryLayout,
    pub device: Device,
}

impl TensorSpec {
    pub fn contiguous(shape: Vec<usize>, dtype: DType, device: Device) -> Self {
        let mut strides = vec![0; shape.len()];
        let mut acc = 1usize;
        for index in (0..shape.len()).rev() {
            strides[index] = acc;
            acc *= shape[index];
        }
        TensorSpec {
            shape,
            dtype,
            strides,
            alignment: 64,
            layout: MemoryLayout::Contiguous,
            device,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AliasingConstraint {
    pub input: usize,
    pub output: usize,
    pub allowed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceSpec {
    pub bytes: usize,
    pub alignment: usize,
}

impl WorkspaceSpec {
    pub const fn none() -> Self {
        WorkspaceSpec {
            bytes: 0,
            alignment: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionContext {
    pub device: Device,
    pub stream_id: u64,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        ExecutionContext {
            device: Device::Cpu,
            stream_id: 0,
        }
    }
}

/// One lowered operation with fully materialized input tensors.
///
/// `output_specs` are already resolved against symbolic input bindings by the
/// executor. This keeps shape specialization out of individual kernels.
#[derive(Debug)]
pub struct KernelCall<'a, T: Scalar = f32> {
    pub op: &'a Op,
    pub attrs: &'a OpAttrs,
    pub inputs: &'a [DenseTensor<T>],
    pub input_specs: &'a [TensorSpec],
    pub output_specs: &'a [TensorSpec],
    pub aliasing: &'a [AliasingConstraint],
    pub workspace: WorkspaceSpec,
    pub context: ExecutionContext,
}

impl<T: Scalar> KernelCall<'_, T> {
    pub fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.output_specs
            .iter()
            .map(|spec| spec.shape.clone())
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub abi_version: u32,
    pub capability_version: u32,
    /// Dtypes the backend can execute, in preference order.
    pub dtypes: &'static [DType],
    pub supports_workspace: bool,
    pub supports_streams: bool,
}

impl BackendCapabilities {
    pub fn supports(&self, dtype: DType) -> bool {
        self.dtypes.contains(&dtype)
    }
}

/// Synchronous kernel dispatch contract.
///
/// The reference backend executes on the calling thread. A CUDA backend will
/// record stream/event work inside `execute` and return device-resident tensor
/// handles; the executor above this trait is intentionally unaware of that.
pub trait KernelBackend<T: Scalar = f32>: Send + Sync {
    fn name(&self) -> &'static str;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            abi_version: KERNEL_ABI_VERSION,
            capability_version: BACKEND_CAPABILITY_VERSION,
            dtypes: &[],
            supports_workspace: false,
            supports_streams: false,
        }
    }

    fn execute(&self, call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>>;
}
