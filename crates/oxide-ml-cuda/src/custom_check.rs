//! Verify a cuda-oxide kernel registered as a graph operator.
//!
//! The same `Op::Custom` graph runs through the CPU reference registry and the
//! CUDA registry; outputs are compared and the CUDA launch goes through
//! `launch_with_access` so buffer hazards and completion events are tracked.

use std::sync::Arc;

use cuda_core::LaunchConfig;
use cuda_host::cuda_module;
use oxide_ml_core::{DenseTensor, ParameterStore};
use oxide_ml_graph::GraphBuilder;
use oxide_ml_kernel::{
    CustomCall, CustomKernel, CustomKernelRegistry, KernelDescriptor, TensorConstraint,
};

use crate::access::{AccessRequest, launch_with_access};
use crate::buffer::CudaTensor;
use crate::device::CudaDevice;
use crate::error::{CudaError, Result};
use crate::executor::CudaExecutor;
use crate::param::CudaParameterStore;

#[cuda_module]
mod custom_kernels {
    use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};

    #[kernel]
    #[launch_bounds(256)]
    pub fn custom_activation(input: &[f32], mut output: DisjointSlice<f32>, n: u32) {
        let index = thread::index_1d();
        let i = index.get();
        if i as u32 >= n {
            return;
        }
        if let Some(slot) = output.get_mut(index) {
            let x = input[i];
            *slot = x / (1.0 + x.abs());
        }
    }
}

fn descriptor() -> KernelDescriptor {
    KernelDescriptor::new(
        "custom_activation",
        oxide_ml_core::DType::F32,
        vec![TensorConstraint::any().rank(1)],
        vec![TensorConstraint::any().rank(1)],
    )
    .with_launch(oxide_ml_kernel::LaunchConfig::one_dimensional(256))
    .with_architecture("sm_75")
    .with_autograd()
}

fn cpu_registry() -> Arc<CustomKernelRegistry<f32>> {
    let mut registry = CustomKernelRegistry::new();
    let kernel = CustomKernel::new(
        descriptor(),
        Arc::new(|call: &CustomCall<'_, f32>| {
            let input = &call.inputs[0];
            Ok(vec![input.map(|value| value / (1.0 + value.abs()))])
        }),
    )
    .expect("CPU kernel descriptor");
    registry.register(kernel).expect("register CPU kernel");
    Arc::new(registry)
}

fn cuda_registry() -> Result<Arc<crate::custom::CudaKernelRegistry>> {
    let mut registry = crate::custom::CudaKernelRegistry::new();
    registry.register(descriptor(), |executor, stream, inputs, output_shapes| {
        let module = custom_kernels::load(executor.device().context())?;
        let n = inputs[0].shape().iter().product::<usize>() as u32;
        let mut output = CudaTensor::zeros(executor.pool(), output_shapes[0].clone())?;
        let requests = [
            AccessRequest::read(&inputs[0]),
            AccessRequest::write(&output),
        ];
        let completion = launch_with_access(executor.event_pool(), stream, &requests, || unsafe {
            module.custom_activation(
                stream,
                LaunchConfig::for_num_elems(n),
                inputs[0].buffer()?,
                output.buffer_mut()?,
                n,
            )?;
            Ok(())
        })?;
        inputs[0].mark_used(completion.event.clone())?;
        output.mark_used(completion.event.clone())?;
        Ok(vec![output])
    })?;
    Ok(Arc::new(registry))
}

/// Run the comparison. Returns the maximum absolute difference.
pub fn run(device: CudaDevice) -> Result<f32> {
    let data: Vec<f32> = vec![-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 10.0];
    let shape = vec![data.len()];

    let mut builder = GraphBuilder::new();
    let x = builder.input(
        "x",
        oxide_ml_core::Shape::from_static([data.len()]),
        oxide_ml_core::DType::F32,
    );
    let y = builder
        .custom(
            "custom_activation",
            &[x],
            vec![oxide_ml_core::Shape::from_static([data.len()])],
        )
        .map_err(|error| CudaError::Execution(error.to_string()))?;
    let graph = builder
        .finish(&y)
        .map_err(|error| CudaError::Execution(error.to_string()))?;

    let mut cpu_inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
    cpu_inputs.insert(
        x,
        DenseTensor::from_vec(shape.clone(), data.clone()).expect("CPU input"),
    );
    let cpu_executor = oxide_ml_cpu::Executor::with_custom(graph.clone(), cpu_registry())?;
    let cpu_result = cpu_executor
        .run(&ParameterStore::new(), &cpu_inputs)
        .map_err(|error| CudaError::Execution(error.to_string()))?;
    let cpu_output = cpu_result
        .output(0)
        .map_err(|error| CudaError::Execution(error.to_string()))?
        .data()
        .to_vec();

    let executor = CudaExecutor::new(device)?.with_custom_kernels(cuda_registry()?);
    let mut host = std::collections::HashMap::new();
    host.insert(x, DenseTensor::from_vec(shape, data).expect("host input"));
    let parameters =
        CudaParameterStore::from_host(&graph, &ParameterStore::new(), executor.pool())?;
    let run = executor.run(&graph, &parameters, &host)?;
    let stream = executor.device().default_stream();
    let gpu_output = run.download(&stream, y[0])?;

    let mut max_diff = 0.0f32;
    for (cpu, gpu) in cpu_output.iter().zip(gpu_output.data()) {
        max_diff = max_diff.max((cpu - gpu).abs());
    }
    Ok(max_diff)
}
