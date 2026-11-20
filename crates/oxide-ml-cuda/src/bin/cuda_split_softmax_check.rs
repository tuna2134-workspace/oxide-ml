//! CUDA Split / LogSoftmax parity against the CPU executor.

use std::collections::HashMap;
use std::error::Error;

use oxide_ml_core::{DType, DenseTensor, ParameterStore, Shape};
use oxide_ml_cpu::{ExecutionInputs, Executor};
use oxide_ml_cuda::{CudaDevice, CudaExecutor, CudaParameterStore};
use oxide_ml_graph::GraphBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let log_softmax = builder.log_softmax(x, -1)?;
    let parts = builder.split(x, 1, vec![2, 2])?;
    let graph = builder.finish(&[log_softmax, parts[0], parts[1]])?;
    let values = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, 0.0];
    let tensor = DenseTensor::from_vec([2, 4], values.clone())?;

    let parameters = ParameterStore::<f32>::new();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(x, tensor.clone());
    let cpu = Executor::<f32>::new(graph.clone())?.run(&parameters, &inputs)?;

    let executor = CudaExecutor::new(CudaDevice::new(0)?)?;
    let gpu_parameters = CudaParameterStore::from_host(&graph, &parameters, executor.pool())?;
    let mut host = HashMap::new();
    host.insert(x, tensor);
    let run = executor.run(&graph, &gpu_parameters, &host)?;
    let stream = executor.device().default_stream();

    let mut max_diff = 0.0f32;
    for (index, output) in graph.outputs().iter().enumerate() {
        let cpu_values = cpu.output(index)?.data();
        let gpu = run.download(&stream, *output)?;
        for (a, b) in cpu_values.iter().zip(gpu.data()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    println!("Split/LogSoftmax CPU/CUDA max |diff| = {max_diff:.3e}");
    if max_diff > 1e-5 {
        return Err("parity failed".into());
    }
    println!("CUDA_SPLIT_LOGSOFTMAX_CHECK: PASS");
    Ok(())
}
