use cuda_core::CudaStream;
use oxide_ml_core::{DenseTensor, Error, Result as MlResult};
use oxide_ml_kernel::{
    BACKEND_CAPABILITY_VERSION, BackendCapabilities, KERNEL_ABI_VERSION, KernelBackend, KernelCall,
};

use crate::device::CudaDevice;
use crate::error::{CudaError, Result};
use crate::executor::CudaExecutor;

/// ABI-level CUDA kernel backend.
///
/// `KernelBackend::execute` receives host tensors by contract, so this adapter
/// performs H2D/D2H for a single operation. The graph executor above uses
/// device-resident `CudaTensor`s directly and is the path used for training;
/// this adapter exists for backend conformance tests and one-op comparisons.
pub struct CudaBackend {
    executor: CudaExecutor,
}

impl CudaBackend {
    pub fn new(device: CudaDevice) -> Result<Self> {
        Ok(CudaBackend {
            executor: CudaExecutor::new(device)?,
        })
    }

    pub fn device(&self) -> &CudaDevice {
        self.executor.device()
    }
}

impl KernelBackend<f32> for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda-cuda-oxide"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            abi_version: KERNEL_ABI_VERSION,
            capability_version: BACKEND_CAPABILITY_VERSION,
            dtypes: &[oxide_ml_core::DType::F32],
            supports_workspace: false,
            supports_streams: true,
        }
    }

    fn execute(&self, call: KernelCall<'_, f32>) -> MlResult<Vec<DenseTensor<f32>>> {
        let stream: &CudaStream = &self.executor.device().default_stream();
        let mut inputs = Vec::with_capacity(call.inputs.len());
        for (tensor, spec) in call.inputs.iter().zip(call.input_specs) {
            let shape = spec.shape.clone();
            let device = self
                .executor
                .pool()
                .acquire(tensor.numel())
                .map_err(to_ml_error)?;
            let mut device_tensor =
                crate::buffer::CudaTensor::from_storage(shape, std::sync::Arc::new(device))
                    .map_err(to_ml_error)?;
            device_tensor
                .copy_from_host(stream, tensor.data())
                .map_err(to_ml_error)?;
            inputs.push(device_tensor);
        }
        let shapes = call
            .output_specs
            .iter()
            .map(|spec| spec.shape.clone())
            .collect::<Vec<_>>();
        let outputs = self
            .executor
            .execute_operation(call.op, call.attrs, &inputs, &shapes)
            .map_err(to_ml_error)?;
        let mut host = Vec::with_capacity(outputs.len());
        for output in outputs {
            host.push(DenseTensor::from_vec(
                output.shape(),
                output.to_host(stream).map_err(to_ml_error)?,
            )?);
        }
        Ok(host)
    }
}

fn to_ml_error(error: CudaError) -> Error {
    Error::execution(error.to_string())
}
