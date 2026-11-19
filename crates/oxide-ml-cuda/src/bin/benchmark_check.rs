//! Phase 7F-19 forward benchmark with separated timing and full statistics.
//!
//! Measurement definitions are documented in `bench_common` and in
//! `docs/phase7f19-benchmark-methodology.md`.

#[path = "bench_common/mod.rs"]
mod bench_common;

use std::collections::HashMap;
use std::time::Instant;

use bench_common::{
    Case, MemorySnapshot, WARMUP, collect_meta, graph_hash, measure_async, measure_sync, out_dir,
    plan_hash, within_tolerance, write_results,
};
use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, DenseTensor, ParameterId, ParameterStore, Shape, ValueId};
use oxide_ml_cuda::{
    CudaBufferPool, CudaDevice, CudaExecution, CudaExecutor, CudaMemoryPlan, CudaParameterStore,
    CudaPlannedArena,
};
use oxide_ml_graph::{Graph, GraphBuilder};

type Inputs = HashMap<ValueId, DenseTensor<f32>>;

struct GraphCase {
    name: &'static str,
    graph: Graph,
    params: ParameterStore<f32>,
    inputs: Inputs,
    shape: String,
    compile_ms: f64,
}

fn build_small_mlp() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
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
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 3], vec![1.0, 2.0, -1.0, 0.5, -1.0, 2.0])?,
    );
    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "small_mlp",
        graph,
        params,
        inputs,
        shape: "x[2,3] w1[3,4] w2[4,1]".to_string(),
        compile_ms,
    })
}

fn build_large_mlp() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([16, 128]), DType::F32);
    let w1 = builder.try_parameter("w1", Shape::from_static([128, 256]), DType::F32, true)?;
    let b1 = builder.try_parameter("b1", Shape::from_static([256]), DType::F32, true)?;
    let w2 = builder.try_parameter("w2", Shape::from_static([256, 64]), DType::F32, true)?;
    let b2 = builder.try_parameter("b2", Shape::from_static([64]), DType::F32, true)?;
    let w3 = builder.try_parameter("w3", Shape::from_static([64, 1]), DType::F32, true)?;
    let h = builder.matmul(x, w1)?;
    let h = builder.add(h, b1)?;
    let h = builder.relu(h)?;
    let h = builder.matmul(h, w2)?;
    let h = builder.add(h, b2)?;
    let h = builder.relu(h)?;
    let y = builder.matmul(h, w3)?;
    let graph = builder.finish(&[y])?;
    let mut params = ParameterStore::<f32>::new();
    params.add(
        ParameterId(0),
        "w1",
        DenseTensor::from_vec([128, 256], vec![0.01f32; 128 * 256])?,
        true,
    )?;
    params.add(
        ParameterId(1),
        "b1",
        DenseTensor::from_vec([256], vec![0.0f32; 256])?,
        true,
    )?;
    params.add(
        ParameterId(2),
        "w2",
        DenseTensor::from_vec([256, 64], vec![0.01f32; 256 * 64])?,
        true,
    )?;
    params.add(
        ParameterId(3),
        "b2",
        DenseTensor::from_vec([64], vec![0.0f32; 64])?,
        true,
    )?;
    params.add(
        ParameterId(4),
        "w3",
        DenseTensor::from_vec([64, 1], vec![0.01f32; 64])?,
        true,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([16, 128], vec![0.5f32; 16 * 128])?,
    );
    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "large_mlp",
        graph,
        params,
        inputs,
        shape: "x[16,128] 128->256->64->1".to_string(),
        compile_ms,
    })
}

fn build_matmul() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([64, 64]), DType::F32);
    let w = builder.try_parameter("w", Shape::from_static([64, 64]), DType::F32, true)?;
    let y = builder.matmul(x, w)?;
    let graph = builder.finish(&[y])?;
    let mut params = ParameterStore::<f32>::new();
    params.add(
        ParameterId(0),
        "w",
        DenseTensor::from_vec([64, 64], vec![0.01f32; 64 * 64])?,
        true,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([64, 64], vec![1.0f32; 64 * 64])?,
    );
    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "matmul",
        graph,
        params,
        inputs,
        shape: "x[64,64] w[64,64]".to_string(),
        compile_ms,
    })
}

fn build_activation() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4096]), DType::F32);
    let y = builder.relu(x)?;
    let graph = builder.finish(&[y])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([4096], vec![-1.0f32; 4096])?,
    );
    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "activation",
        graph,
        params: ParameterStore::new(),
        inputs,
        shape: "x[4096] relu".to_string(),
        compile_ms,
    })
}

fn build_reduction() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([64, 64]), DType::F32);
    let y = builder.reduce_sum(x, vec![1], true)?;
    let graph = builder.finish(&[y])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([64, 64], vec![1.0f32; 64 * 64])?,
    );
    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "reduction",
        graph,
        params: ParameterStore::new(),
        inputs,
        shape: "x[64,64] reduce_sum(axis=1)".to_string(),
        compile_ms,
    })
}

fn build_dynamic_batch() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![oxide_ml_core::Dim::Symbol(batch), oxide_ml_core::Dim::Static(3)]),
        DType::F32,
    );
    let w = builder.try_parameter("w", Shape::from_static([3, 1]), DType::F32, true)?;
    let y = builder.matmul(x, w)?;
    let graph = builder.finish(&[y])?;
    let mut params = ParameterStore::<f32>::new();
    params.add(
        ParameterId(0),
        "w",
        DenseTensor::from_vec([3, 1], vec![0.5, -0.25, 0.1])?,
        true,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([4, 3], vec![1.0f32; 12])?,
    );
    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "dynamic_batch",
        graph,
        params,
        inputs,
        shape: "batch=4 x[4,3] w[3,1]".to_string(),
        compile_ms,
    })
}

fn memory_snapshot(
    executor: &CudaExecutor,
    arena: &CudaPlannedArena,
    plan: &CudaMemoryPlan,
    outputs: &[oxide_ml_cuda::CudaTensor],
) -> MemorySnapshot {
    let pool = executor.pool_stats();
    let arena_stats = arena.stats();
    let events = executor.event_stats();
    let mut access = oxide_ml_execution_safety::AccessStats::default();
    for tensor in outputs {
        let stats = tensor.access_stats();
        access.event_waits += stats.event_waits;
        access.same_stream_elided_waits += stats.same_stream_elided_waits;
        access.cross_stream_waits += stats.cross_stream_waits;
        access.read_read_concurrent += stats.read_read_concurrent;
    }
    MemorySnapshot {
        logical_bytes: Some(plan.logical_bytes),
        planned_logical_bytes: Some(plan.logical_bytes),
        runtime_required_bytes: Some(plan.logical_bytes),
        pool_reserved_bytes: Some(pool.reserved_bytes),
        pool_active_bytes: Some(pool.current_bytes),
        pool_pending_release_bytes: Some(pool.pending_release_bytes),
        pool_quarantined_bytes: Some(pool.quarantined_bytes),
        pool_allocation_count: Some(pool.allocations),
        pool_reuse_count: Some(pool.reuses),
        arena_capacity_bytes: Some(arena_stats.physical_reserved_bytes as usize),
        arena_used_bytes: Some((arena_stats.active_slot_set_count * arena_stats.logical_bytes) as usize),
        arena_physical_reserved_bytes: Some(arena_stats.physical_reserved_bytes as usize),
        arena_slot_allocation_count: Some(arena_stats.allocation_count),
        arena_slot_reuse_count: Some(arena_stats.slot_set_reuse_count),
        arena_active_lease_count: Some(arena_stats.active_slot_set_count),
        event_allocation_count: Some(events.allocation_count as u64),
        event_reuse_count: Some(events.reuse_count as u64),
        event_pending_count: Some(events.pending_count as u64),
        event_quarantine_count: Some(events.quarantine_count as u64),
        access_reads: Some(access.event_waits),
        access_writes: Some(access.cross_stream_waits),
        access_read_write: Some(access.read_read_concurrent),
        access_same_stream_elided: Some(access.same_stream_elided_waits),
        access_cross_stream_waits: Some(access.cross_stream_waits),
        access_read_read_elision: Some(access.read_read_concurrent),
        driver_used_bytes: None,
    }
}

fn bench_case(
    executor: &CudaExecutor,
    pool: &CudaBufferPool,
    case: GraphCase,
    cases: &mut Vec<Case>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan_begin = Instant::now();
    let memory_plan = MemoryPlanner::default().plan(&case.graph)?;
    let plan = CudaMemoryPlan::from_memory_plan(&case.graph, &memory_plan)?;
    let compile_ms = case.compile_ms + plan_begin.elapsed().as_secs_f64() * 1000.0;
    let params = CudaParameterStore::from_host(&case.graph, &case.params, pool)?;
    let arena = CudaPlannedArena::new(plan.clone(), pool.clone(), 2)?;
    let arena_stream = executor.device().new_stream()?;
    let default_stream = executor.device().default_stream();
    let hash = graph_hash(&case.graph);
    let phash = plan_hash(&plan);

    // Correctness: CPU reference vs CUDA normal / planned / arena.
    let mut cpu_inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
    for (&id, tensor) in &case.inputs {
        cpu_inputs.insert(id, tensor.clone());
    }
    let cpu = oxide_ml_cpu::Executor::<f32>::new(case.graph.clone())?;
    let cpu_outputs = cpu.run(&case.params, &cpu_inputs)?;
    let normal = executor.run(&case.graph, &params, &case.inputs)?;
    let mut correctness = String::from("PASS");
    for (index, &value) in case.graph.outputs().iter().enumerate() {
        let expected = cpu_outputs.output(index)?.data();
        let actual = normal.download(&default_stream, value)?;
        if let Err(error) = within_tolerance(actual.data(), expected, 1e-4, 3e-3) {
            correctness = format!("FAIL forward[{index}]: {error}");
            break;
        }
    }
    let planned = executor.run_planned(&case.graph, &params, &case.inputs, &plan)?;
    let arena_run = executor.run_arena(
        &case.graph,
        &params,
        &case.inputs,
        &arena,
        arena_stream.clone(),
    )?;
    if correctness == "PASS" {
        for &value in case.graph.outputs() {
            let base = normal.download(&default_stream, value)?;
            let planned_value = planned.download(&default_stream, value)?;
            let arena_value = arena_run.download(&default_stream, value)?;
            for (label, candidate) in [
                ("planned", planned_value.data()),
                ("arena", arena_value.data()),
            ] {
                if let Err(error) = within_tolerance(candidate, base.data(), 1e-6, 1e-5) {
                    correctness = format!("FAIL {label}: {error}");
                }
            }
        }
    }
    if !bench_common::all_finite(&normal.output(0)?.to_host(&default_stream)?) {
        correctness = "FAIL non-finite output".to_string();
    }
    let access_outputs = vec![normal.output(0)?.clone()];
    let memory = memory_snapshot(executor, &arena, &plan, &access_outputs);

    // Normal.
    let mut record = Case::new("forward", case.name, "normal", &case.shape);
    record.graph_hash = hash.clone();
    record.plan_hash = phash.clone();
    record.correctness = correctness.clone();
    record.compile_ms = compile_ms;
    record.memory = memory.clone();
    record.note = "sync API; host_call includes implicit wait".to_string();
    record.samples = measure_sync(&default_stream, bench_common::ROUNDS, bench_common::ITERATIONS, WARMUP, || {
        executor.run(&case.graph, &params, &case.inputs)?;
        Ok(())
    })?;
    cases.push(record);

    // Planned.
    let mut record = Case::new("forward", case.name, "planned", &case.shape);
    record.graph_hash = hash.clone();
    record.plan_hash = phash.clone();
    record.correctness = correctness.clone();
    record.compile_ms = compile_ms;
    record.memory = memory.clone();
    record.note = "sync API; host_call includes implicit wait".to_string();
    record.samples = measure_sync(&default_stream, bench_common::ROUNDS, bench_common::ITERATIONS, WARMUP, || {
        executor.run_planned(&case.graph, &params, &case.inputs, &plan)?;
        Ok(())
    })?;
    cases.push(record);

    // Arena.
    let mut record = Case::new("forward", case.name, "arena", &case.shape);
    record.graph_hash = hash.clone();
    record.plan_hash = phash.clone();
    record.correctness = correctness.clone();
    record.compile_ms = compile_ms;
    record.memory = memory.clone();
    record.note = "sync API; host_call includes implicit wait".to_string();
    record.samples = measure_sync(&default_stream, bench_common::ROUNDS, bench_common::ITERATIONS, WARMUP, || {
        executor.run_arena(
            &case.graph,
            &params,
            &case.inputs,
            &arena,
            arena_stream.clone(),
        )?;
        Ok(())
    })?;
    cases.push(record);

    // Async planned.
    let mut record = Case::new("forward", case.name, "async_planned", &case.shape);
    record.graph_hash = hash.clone();
    record.plan_hash = phash.clone();
    record.correctness = correctness.clone();
    record.compile_ms = compile_ms;
    record.memory = memory.clone();
    record.note = "submit does not wait; GPU event between submit and completion".to_string();
    record.samples = measure_async(&default_stream, bench_common::ROUNDS, bench_common::ITERATIONS, WARMUP, || {
        Ok(executor.execute_async_planned(
            &case.graph,
            &params,
            &case.inputs,
            &plan,
            default_stream.clone(),
        )? as CudaExecution)
    })?;
    cases.push(record);

    // Async arena.
    let mut record = Case::new("forward", case.name, "async_arena", &case.shape);
    record.graph_hash = hash;
    record.plan_hash = phash;
    record.correctness = correctness;
    record.compile_ms = compile_ms;
    record.memory = memory;
    record.note = "submit does not wait; arena lease released after completion".to_string();
    record.samples = measure_async(&default_stream, bench_common::ROUNDS, bench_common::ITERATIONS, WARMUP, || {
        Ok(executor.execute_async_arena(
            &case.graph,
            &params,
            &case.inputs,
            &arena,
            default_stream.clone(),
        )? as CudaExecution)
    })?;
    cases.push(record);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let pool = CudaBufferPool::new(device.context().clone(), device.default_stream());
    let meta = collect_meta(&device);
    println!(
        "benchmark_check: gpu={} cc={} uuid={}",
        meta.gpu_name, meta.compute_capability, meta.gpu_uuid
    );

    let mut cases = Vec::new();
    for builder in [
        build_small_mlp,
        build_large_mlp,
        build_matmul,
        build_activation,
        build_reduction,
        build_dynamic_batch,
    ] {
        let case = builder()?;
        let name = case.name;
        bench_case(&executor, &pool, case, &mut cases)?;
        println!("benchmark_check: {name} measured");
    }

    let dir = out_dir("../../docs");
    let json_path = dir.join("phase7f19-benchmark-results.json");
    write_results(&json_path, "benchmark_check", &meta, &cases)?;
    println!("benchmark_check: wrote {}", json_path.display());
    println!(
        "benchmark_check: cases={} samples/case={}",
        cases.len(),
        bench_common::ROUNDS * bench_common::ITERATIONS
    );
    Ok(())
}
