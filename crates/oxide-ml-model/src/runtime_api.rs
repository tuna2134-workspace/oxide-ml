//! Execution runtime selection.
//!
//! A [`Runtime`] groups a device, a compile cache and optional custom kernels.
//! The CPU workspace executes compiled models directly; a CUDA runtime exists
//! so the same model code can select a device, but execution is performed by
//! `oxide-ml-cuda`'s `CudaModel`/`CudaTrainingModel`. There is no CPU fallback
//! for a CUDA device: `run_with` returns an explicit error.

use std::sync::Arc;

use oxide_ml_core::{Device, Error, Result};
use oxide_ml_kernel::CustomKernelRegistry;

use crate::Module;
use crate::cache::{CompileCache, compile_cached_with};
use crate::runtime::{CompiledModel, InputSpec, ModelCompileOptions};
use crate::tensor::Tensor;

/// Execution runtime configuration.
#[derive(Clone)]
pub struct Runtime {
    device: Device,
    cache: Arc<CompileCache>,
    custom_kernels: Option<Arc<CustomKernelRegistry<f32>>>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("device", &self.device)
            .field("cache", &self.cache.stats())
            .finish()
    }
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn cache(&self) -> &Arc<CompileCache> {
        &self.cache
    }

    pub fn custom_kernels(&self) -> Option<&Arc<CustomKernelRegistry<f32>>> {
        self.custom_kernels.as_ref()
    }

    /// Compile (or reuse) a model for this runtime.
    pub fn compile<M: Module + ?Sized>(
        &self,
        model: &M,
        inputs: &[InputSpec],
        options: ModelCompileOptions,
    ) -> Result<Arc<CompiledModel>> {
        match self.device {
            Device::Cpu => {
                let mut options = options;
                if options.custom_kernels.is_none() {
                    options.custom_kernels = self.custom_kernels.clone();
                }
                compile_cached_with(&self.cache, model, inputs, &options)
            }
            Device::Cuda(index) => Err(Error::unsupported(format!(
                "Runtime for cuda:{index} is provided by oxide-ml-cuda; \
                 use CudaModel::compile / CudaTrainingModel with the same model"
            ))),
        }
    }
}

/// Builder for [`Runtime`].
#[derive(Clone, Debug)]
pub struct RuntimeBuilder {
    device: Device,
    cache_capacity: usize,
    custom_kernels: Option<Arc<CustomKernelRegistry<f32>>>,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        RuntimeBuilder {
            device: Device::Cpu,
            cache_capacity: crate::cache::DEFAULT_CAPACITY,
            custom_kernels: None,
        }
    }
}

impl RuntimeBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.cache_capacity = capacity.max(1);
        self
    }

    pub fn custom_kernels(mut self, kernels: Arc<CustomKernelRegistry<f32>>) -> Self {
        self.custom_kernels = Some(kernels);
        self
    }

    pub fn build(self) -> Result<Runtime> {
        Ok(Runtime {
            device: self.device,
            cache: Arc::new(CompileCache::new(self.cache_capacity)),
            custom_kernels: self.custom_kernels,
        })
    }
}

impl CompiledModel {
    /// Run through a [`Runtime`].
    ///
    /// The compiled model already carries its backend executor; the runtime
    /// mainly selects the device and owns the compile cache. A CUDA runtime is
    /// rejected here because this model executes on the CPU workspace.
    pub fn run_with(
        &self,
        runtime: &Runtime,
        parameters: &oxide_ml_core::ParameterStore<f32>,
        inputs: &[(&str, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>> {
        match runtime.device() {
            Device::Cpu => self.run(parameters, inputs),
            Device::Cuda(index) => Err(Error::unsupported(format!(
                "this CompiledModel was compiled for the CPU workspace; \
                 cuda:{index} requires oxide-ml-cuda's CudaModel"
            ))),
        }
    }
}
