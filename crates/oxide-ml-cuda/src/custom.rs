//! CUDA kernels registered as graph operators.
//!
//! The device launch is supplied by the caller, usually a closure that calls a
//! `#[cuda_module]` / `#[kernel]` wrapper. The registry only owns metadata and
//! launch functions; it never generates PTX or bypasses `CudaExecutor`'s
//! `AccessGuard`/event machinery, because launches still go through the
//! executor's stream and pool.

use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::CudaStream;
use oxide_ml_kernel::KernelDescriptor;

use crate::buffer::CudaTensor;
use crate::error::{CudaError, Result};
use crate::executor::CudaExecutor;

/// Device launch function for a registered kernel.
///
/// The closure receives the executor (for `pool()` and `kernels()`), the stream
/// selected by the caller, the device-resident inputs and the resolved output
/// shapes. It must allocate outputs from `executor.pool()` so access tracking
/// and deferred reclamation keep working.
pub type CudaLaunch = Arc<
    dyn Fn(&CudaExecutor, &CudaStream, &[CudaTensor], &[Vec<usize>]) -> Result<Vec<CudaTensor>>
        + Send
        + Sync,
>;

/// One registered device kernel.
pub struct CudaKernelEntry {
    pub descriptor: KernelDescriptor,
    pub launch: CudaLaunch,
}

/// Registry of device kernels, keyed by descriptor name.
pub struct CudaKernelRegistry {
    entries: HashMap<String, CudaKernelEntry>,
}

impl Default for CudaKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaKernelRegistry {
    pub fn new() -> Self {
        CudaKernelRegistry {
            entries: HashMap::new(),
        }
    }

    /// Register a kernel. Descriptor and dtype are validated immediately.
    pub fn register<F>(&mut self, descriptor: KernelDescriptor, launch: F) -> Result<()>
    where
        F: Fn(&CudaExecutor, &CudaStream, &[CudaTensor], &[Vec<usize>]) -> Result<Vec<CudaTensor>>
            + Send
            + Sync
            + 'static,
    {
        descriptor.validate().map_err(|error| {
            CudaError::Unsupported(format!("invalid kernel descriptor: {error}"))
        })?;
        if descriptor.dtype != oxide_ml_core::DType::F32 {
            return Err(CudaError::Unsupported(format!(
                "CUDA custom kernels currently support F32 only, got {}",
                descriptor.dtype
            )));
        }
        if self.entries.contains_key(&descriptor.name) {
            return Err(CudaError::Unsupported(format!(
                "custom CUDA kernel {:?} is already registered",
                descriptor.name
            )));
        }
        self.entries.insert(
            descriptor.name.clone(),
            CudaKernelEntry {
                descriptor,
                launch: Arc::new(launch),
            },
        );
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&CudaKernelEntry> {
        self.entries.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Execute a registered kernel on `stream`.
    pub fn execute(
        &self,
        name: &str,
        executor: &CudaExecutor,
        stream: &CudaStream,
        inputs: &[CudaTensor],
        output_shapes: &[Vec<usize>],
    ) -> Result<Vec<CudaTensor>> {
        let entry = self.get(name).ok_or_else(|| {
            CudaError::Unsupported(format!("custom CUDA kernel {name:?} is not registered"))
        })?;
        if inputs.len() != entry.descriptor.inputs.len() {
            return Err(CudaError::Shape(format!(
                "custom kernel {:?} expects {} inputs, got {}",
                name,
                entry.descriptor.inputs.len(),
                inputs.len()
            )));
        }
        if output_shapes.len() != entry.descriptor.outputs.len() {
            return Err(CudaError::Shape(format!(
                "custom kernel {:?} expects {} outputs, got {}",
                name,
                entry.descriptor.outputs.len(),
                output_shapes.len()
            )));
        }
        let outputs = (entry.launch)(executor, stream, inputs, output_shapes)?;
        if outputs.len() != output_shapes.len() {
            return Err(CudaError::Shape(format!(
                "custom kernel {:?} produced {} outputs, expected {}",
                name,
                outputs.len(),
                output_shapes.len()
            )));
        }
        for (index, (tensor, expected)) in outputs.iter().zip(output_shapes).enumerate() {
            if tensor.shape() != expected.as_slice() {
                return Err(CudaError::Shape(format!(
                    "custom kernel {name:?} output {index} has shape {:?}, expected {:?}",
                    tensor.shape(),
                    expected
                )));
            }
        }
        Ok(outputs)
    }
}

impl std::fmt::Debug for CudaKernelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKernelRegistry")
            .field("kernels", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}
