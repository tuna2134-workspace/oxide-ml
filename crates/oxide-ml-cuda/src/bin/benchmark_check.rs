use std::collections::HashMap;
use std::time::Instant;

use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, DenseTensor, Dim, ParameterId, ParameterStore, Shape, ValueId};
use oxide_ml_cuda::{
    CudaBufferPool, CudaDevice, CudaEvent, CudaExecution, CudaExecutor, CudaMemoryPlan,
    CudaParameterStore, CudaPlannedArena,
};
use oxide_ml_graph::{Graph, GraphBuilder};

const ITERS: usize = 10;
const ROUNDS: usize = 3;

type BenchData = (
    Graph,
    CudaParameterStore,
    HashMap<ValueId, DenseTensor<f32>>,
);

fn build(pool: &CudaBufferPool) -> Result<BenchData, Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let w1 = builder.try_parameter("w1", Shape::from_static([3, 4]), DType::F32, true)?;
    let b1 = builder.try_parameter("b1", Shape::from_static([4]), DType::F32, true)?;
    let w2 = builder.try_parameter("w2", Shape::from_static([4, 1]), DType::F32, true)?;
    let h = builder.matmul(x, w1)?;
    let h = builder.add(h, b1)?;
    let h = builder.sigmoid(h)?;
    let y = builder.matmul(h, w2)?;
    let graph = builder.finish(&[y])?;
    let mut params = ParameterStore::<f32>::new();
    params.add(
        ParameterId(0),
        "w1",
        DenseTensor::from_vec(
            [3, 4],
            vec![
                0.1, 0.2, -0.3, 0.4, 0.2, -0.5, 0.3, 0.1, -0.2, 0.4, 0.1, 0.2,
            ],
        )?,
        true,
    )?;
    params.add(
        ParameterId(1),
        "b1",
        DenseTensor::from_vec([4], vec![0.1, -0.2, 0.0, 0.3])?,
        true,
    )?;
    params.add(
        ParameterId(2),
        "w2",
        DenseTensor::from_vec([4, 1], vec![0.5, -0.4, 0.2, 0.1])?,
        true,
    )?;
    let cuda_params = CudaParameterStore::from_host(&graph, &params, pool)?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 3], vec![1.0, 2.0, -1.0, 0.5, -1.0, 2.0])?,
    );
    Ok((graph, cuda_params, inputs))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn measure<F>(
    executor: &CudaExecutor,
    label: &str,
    mut f: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..2 {
        f()?;
    }
    let stream = executor.device().default_stream();
    let mut gpu = Vec::new();
    let mut host = Vec::new();
    for _ in 0..ROUNDS {
        let start = CudaEvent::new(executor.device().context())?;
        let end = CudaEvent::new(executor.device().context())?;
        start.record(&stream)?;
        let host_start = Instant::now();
        for _ in 0..ITERS {
            f()?;
        }
        let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
        end.record(&stream)?;
        let gpu_ms = f64::from(start.elapsed_ms(&end)?);
        gpu.push(gpu_ms / ITERS as f64);
        host.push(host_ms / ITERS as f64);
    }
    println!(
        "{label}: gpu_ms={:.4} host_submit_ms={:.4}",
        median(&mut gpu),
        median(&mut host)
    );
    Ok(())
}

fn measure_async<F>(
    executor: &CudaExecutor,
    label: &str,
    mut submit: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<CudaExecution, Box<dyn std::error::Error>>,
{
    for _ in 0..2 {
        let mut execution = submit()?;
        let _ = execution.wait()?;
    }
    let stream = executor.device().default_stream();
    let mut gpu = Vec::new();
    let mut host = Vec::new();
    for _ in 0..ROUNDS {
        let start = CudaEvent::new(executor.device().context())?;
        let end = CudaEvent::new(executor.device().context())?;
        start.record(&stream)?;
        let host_start = Instant::now();
        let mut executions = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            executions.push(submit()?);
        }
        let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
        for execution in &mut executions {
            let _ = execution.wait()?;
        }
        end.record(&stream)?;
        let gpu_ms = f64::from(start.elapsed_ms(&end)?);
        gpu.push(gpu_ms / ITERS as f64);
        host.push(host_ms / ITERS as f64);
    }
    println!(
        "{label}: gpu_ms={:.4} host_submit_ms={:.4}",
        median(&mut gpu),
        median(&mut host)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let pool = CudaBufferPool::new(device.context().clone(), device.default_stream());
    let (graph, params, inputs) = build(&pool)?;
    let plan = MemoryPlanner::default().plan(&graph)?;
    let cuda_plan = CudaMemoryPlan::from_memory_plan(&graph, &plan)?;
    let arena = CudaPlannedArena::new(cuda_plan.clone(), pool.clone(), 16)?;
    let stream = device.default_stream();

    measure(&executor, "normal", || {
        executor.run(&graph, &params, &inputs)?;
        Ok(())
    })?;
    measure(&executor, "planned", || {
        executor.run_planned(&graph, &params, &inputs, &cuda_plan)?;
        Ok(())
    })?;
    measure(&executor, "arena", || {
        executor.run_arena(&graph, &params, &inputs, &arena, stream.clone())?;
        Ok(())
    })?;
    measure_async(&executor, "async_planned", || {
        Ok(executor.execute_async_planned(&graph, &params, &inputs, &cuda_plan, stream.clone())?)
    })?;
    measure_async(&executor, "async_arena", || {
        Ok(executor.execute_async_arena(&graph, &params, &inputs, &arena, stream.clone())?)
    })?;

    // Dynamic batch normal/planned.
    let mut b = GraphBuilder::new();
    let batch = b.symbol("batch");
    let x = b.input(
        "xd",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(3)]),
        DType::F32,
    );
    let w = b.try_parameter("wd", Shape::from_static([3, 1]), DType::F32, true)?;
    let y = b.matmul(x, w)?;
    let dgraph = b.finish(&[y])?;
    let mut dparams = ParameterStore::<f32>::new();
    dparams.add(
        ParameterId(0),
        "wd",
        DenseTensor::from_vec([3, 1], vec![0.5, -0.25, 0.1])?,
        true,
    )?;
    let dparams = CudaParameterStore::from_host(&dgraph, &dparams, &pool)?;
    let mut dinputs = HashMap::new();
    dinputs.insert(
        dgraph.inputs()[0],
        DenseTensor::from_vec([4, 3], vec![1.0; 12])?,
    );
    let dplan = MemoryPlanner::default().plan(&dgraph)?;
    let dplan = CudaMemoryPlan::from_memory_plan(&dgraph, &dplan)?;
    measure(&executor, "dynamic_normal", || {
        executor.run(&dgraph, &dparams, &dinputs)?;
        Ok(())
    })?;
    measure(&executor, "dynamic_planned", || {
        executor.run_planned(&dgraph, &dparams, &dinputs, &dplan)?;
        Ok(())
    })?;
    println!("pool_stats: {:?}", executor.pool_stats());
    println!("arena_stats: {:?}", arena.stats());
    println!("event_stats: {:?}", executor.event_stats());
    Ok(())
}
