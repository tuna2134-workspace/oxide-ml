use std::collections::HashMap;

use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, DenseTensor, Dim, ParameterId, ParameterStore, Shape};
use oxide_ml_cuda::{
    CudaBufferPool, CudaDevice, CudaExecutor, CudaMemoryPlan, CudaParameterStore, CudaPlannedArena,
};
use oxide_ml_graph::GraphBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let pool = CudaBufferPool::new(device.context().clone(), device.default_stream());
    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(2)]),
        DType::F32,
    );
    let w = builder.try_parameter("w", Shape::from_static([2, 1]), DType::F32, true)?;
    let y = builder.matmul(x, w)?;
    let graph = builder.finish(&[y])?;
    let mut params = ParameterStore::<f32>::new();
    params.add(
        ParameterId(0),
        "w",
        DenseTensor::from_vec([2, 1], vec![0.5, -0.25])?,
        true,
    )?;
    let cuda_params = CudaParameterStore::from_host(&graph, &params, &pool)?;
    let mut max_bindings = HashMap::new();
    max_bindings.insert(batch, 8usize);
    let (bounded_plan, _) = executor.plan_bounded(&graph, &max_bindings)?;
    let arena = CudaPlannedArena::new(bounded_plan.clone(), pool.clone(), 4)?;

    for batch_size in [1usize, 4, 8] {
        let mut inputs = HashMap::new();
        inputs.insert(
            graph.inputs()[0],
            DenseTensor::from_vec(
                [batch_size, 2],
                (0..batch_size * 2).map(|index| index as f32).collect(),
            )?,
        );
        let normal = executor.run(&graph, &cuda_params, &inputs)?;
        let normal_host = normal.output(0)?.to_host(&device.default_stream())?;
        let planned = executor.run_planned(&graph, &cuda_params, &inputs, &bounded_plan)?;
        let planned_host = planned.output(0)?.to_host(&device.default_stream())?;
        assert_eq!(&normal_host[..batch_size], &planned_host[..batch_size]);
        let arena_run = executor.run_arena(
            &graph,
            &cuda_params,
            &inputs,
            &arena,
            device.default_stream(),
        )?;
        let arena_host = arena_run.output(0)?.to_host(&device.default_stream())?;
        assert_eq!(&normal_host[..batch_size], &arena_host[..batch_size]);
    }

    let mut builder = GraphBuilder::new();
    let xu = builder.input(
        "xu",
        Shape::new(vec![Dim::Dynamic, Dim::Static(2)]),
        DType::F32,
    );
    let wu = builder.try_parameter("wu", Shape::from_static([2, 1]), DType::F32, true)?;
    let yu = builder.matmul(xu, wu)?;
    let ugraph = builder.finish(&[yu])?;
    let mut uparams = ParameterStore::<f32>::new();
    uparams.add(
        ParameterId(0),
        "wu",
        DenseTensor::from_vec([2, 1], vec![0.5, -0.25])?,
        true,
    )?;
    let ucuda = CudaParameterStore::from_host(&ugraph, &uparams, &pool)?;
    let uplan = MemoryPlanner::default().plan(&ugraph)?;
    let uplan = CudaMemoryPlan::from_memory_plan(&ugraph, &uplan)?;
    for batch_size in [3usize, 5] {
        let mut inputs = HashMap::new();
        inputs.insert(
            ugraph.inputs()[0],
            DenseTensor::from_vec(
                [batch_size, 2],
                (0..batch_size * 2).map(|index| index as f32).collect(),
            )?,
        );
        let run = executor.run_planned(&ugraph, &ucuda, &inputs, &uplan)?;
        let host = run.output(0)?.to_host(&device.default_stream())?;
        assert_eq!(host.len(), batch_size);
    }

    println!("dynamic_check: PASS");
    Ok(())
}
