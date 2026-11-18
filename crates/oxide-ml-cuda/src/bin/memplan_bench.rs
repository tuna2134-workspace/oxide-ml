use std::collections::HashMap;
use std::time::Instant;

use oxide_ml_core::{DType, DenseTensor, Shape};
use oxide_ml_cuda::{
    CudaDevice, CudaExecutor, CudaMemoryPlan, CudaParameterStore, CudaPlannedArena,
};
use oxide_ml_graph::{GraphBuilder, Reduction};
use oxide_ml_nn::{Mlp, init_parameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([16, 8]), DType::F32);
    let y = builder.input("y", Shape::from_static([16, 8]), DType::F32);
    let mlp = Mlp::new(
        &mut builder,
        "mlp",
        &[8, 32, 32, 8],
        oxide_ml_graph::ActivationKind::ReLU,
        false,
    )?;
    let logits = mlp.forward(&mut builder, x)?;
    let loss = builder.mse_loss(logits, y, Reduction::Mean)?;
    let graph = builder.finish(&[loss])?;
    let params = init_parameters::<f32>(&graph, 7)?;

    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device)?;
    let stream = executor.device().default_stream();
    let cuda_params = CudaParameterStore::from_host(&graph, &params, executor.pool())?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([16, 8], vec![0.1; 16 * 8])?,
    );
    inputs.insert(
        graph.inputs()[1],
        DenseTensor::from_vec([16, 8], vec![0.0; 16 * 8])?,
    );

    let memory_plan = oxide_ml_compiler::MemoryPlanner::default().plan(&graph)?;
    let cuda_plan = CudaMemoryPlan::from_memory_plan(&graph, &memory_plan)?;
    let arena = CudaPlannedArena::new(cuda_plan.clone(), executor.pool().clone(), 1)?;

    // Warm-up.
    for _ in 0..5 {
        let _ = executor.run(&graph, &cuda_params, &inputs)?;
    }
    let normal = measure(100, || {
        executor.run(&graph, &cuda_params, &inputs).unwrap();
    });
    let planned = measure(100, || {
        executor
            .run_planned(&graph, &cuda_params, &inputs, &cuda_plan)
            .unwrap();
    });
    let arena_time = measure(100, || {
        executor
            .run_arena(&graph, &cuda_params, &inputs, &arena, stream.clone())
            .unwrap();
    });

    let event_overhead = measure(1000, || {
        let lease = executor.event_pool().acquire().unwrap();
        lease.record(&stream).unwrap();
        drop(lease);
    });

    for (name, samples) in [
        ("normal", normal),
        ("planned", planned),
        ("arena", arena_time),
    ] {
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
        println!(
            "{name}: min={:.1}us median={:.1}us p95={:.1}us max={:.1}us",
            sorted[0] * 1e6,
            median * 1e6,
            p95 * 1e6,
            sorted[sorted.len() - 1] * 1e6
        );
    }
    let mut sorted_events = event_overhead.clone();
    sorted_events.sort_by(f64::total_cmp);
    let event_median = sorted_events[sorted_events.len() / 2];
    let event_p95 = sorted_events[(sorted_events.len() * 95 / 100).min(sorted_events.len() - 1)];
    println!(
        "event acquire+record+release: median={:.2}us p95={:.2}us (1000 iterations)",
        event_median * 1e6,
        event_p95 * 1e6
    );
    println!("events: {:?}", executor.event_stats());
    println!("pool: {:?}", executor.pool_stats());
    println!("arena: {:?}", arena.stats());
    println!(
        "plan logical_bytes={} slots={}",
        cuda_plan.logical_bytes,
        cuda_plan.slot_count()
    );
    Ok(())
}

fn measure(mut n: usize, mut f: impl FnMut()) -> Vec<f64> {
    let mut samples = Vec::with_capacity(n);
    while n > 0 {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64());
        n -= 1;
    }
    samples
}
