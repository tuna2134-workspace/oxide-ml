//! Phase 7F-19 backward benchmark (normal execution path).
//!
//! The full-mode variant is kept in `backward_check_full.rs`; the planned
//! training path currently hits a pre-existing executor constant-copy panic
//! (16 vs 1) and is therefore reported as NOT RUN/BLOCKED, not faked.

#[path = "bench_common/mod.rs"]
mod bench_common;

use std::collections::HashMap;
use std::time::Instant;

use bench_common::{
    Case, MemorySnapshot, WARMUP, collect_meta, graph_hash, measure_sync, out_dir, plan_hash,
    within_tolerance, write_results,
};
use oxide_ml_autograd::TrainingGraph;
use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, DenseTensor, ParameterId, ParameterStore, Shape, ValueId};
use oxide_ml_cuda::{CudaBufferPool, CudaDevice, CudaExecutor, CudaMemoryPlan, CudaParameterStore};
use oxide_ml_graph::{Graph, GraphBuilder};

type Inputs = HashMap<ValueId, DenseTensor<f32>>;
struct C {
    name: &'static str,
    fwd: Graph,
    train: TrainingGraph,
    params: ParameterStore<f32>,
    inputs: Inputs,
    shape: &'static str,
    atol: f64,
    rtol: f64,
}

#[allow(clippy::too_many_arguments)]
fn make(
    name: &'static str,
    begin: Instant,
    fwd: Graph,
    loss: ValueId,
    params: ParameterStore<f32>,
    inputs: Inputs,
    shape: &'static str,
    atol: f64,
    rtol: f64,
) -> Result<C, Box<dyn std::error::Error>> {
    let train = oxide_ml_autograd::build_training_graph(&fwd, loss)?;
    let _ = begin;
    Ok(C {
        name,
        fwd,
        train,
        params,
        inputs,
        shape,
        atol,
        rtol,
    })
}

fn linear() -> Result<C, Box<dyn std::error::Error>> {
    let t = Instant::now();
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([8, 4]), DType::F32);
    let y = b.input("y", Shape::from_static([8, 2]), DType::F32);
    let w = b.try_parameter("w", Shape::from_static([4, 2]), DType::F32, true)?;
    let bias = b.try_parameter("b", Shape::from_static([2]), DType::F32, true)?;
    let m = b.matmul(x, w)?;
    let p = b.add(m, bias)?;
    let loss = b.mse_loss(p, y, oxide_ml_graph::Reduction::Mean)?;
    let fwd = b.finish(&[p, loss])?;
    let mut params = ParameterStore::new();
    params.add(
        ParameterId(0),
        "w",
        DenseTensor::from_vec([4, 2], vec![0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.4, -0.3])?,
        true,
    )?;
    params.add(
        ParameterId(1),
        "b",
        DenseTensor::from_vec([2], vec![0.05, -0.1])?,
        true,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        fwd.inputs()[0],
        DenseTensor::from_vec([8, 4], (0..32).map(|i| i as f32 * 0.1 - 1.2).collect())?,
    );
    inputs.insert(
        fwd.inputs()[1],
        DenseTensor::from_vec([8, 2], (0..16).map(|i| (i as f32 * 0.3).sin()).collect())?,
    );
    make(
        "linear_mse",
        t,
        fwd,
        loss,
        params,
        inputs,
        "x[8,4] w[4,2] mse",
        5e-3,
        5e-3,
    )
}
fn mlp() -> Result<C, Box<dyn std::error::Error>> {
    let t = Instant::now();
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([8, 3]), DType::F32);
    let y = b.input("y", Shape::from_static([8, 1]), DType::F32);
    let w1 = b.try_parameter("w1", Shape::from_static([3, 6]), DType::F32, true)?;
    let b1 = b.try_parameter("b1", Shape::from_static([6]), DType::F32, true)?;
    let w2 = b.try_parameter("w2", Shape::from_static([6, 1]), DType::F32, true)?;
    let b2 = b.try_parameter("b2", Shape::from_static([1]), DType::F32, true)?;
    let m1 = b.matmul(x, w1)?;
    let a1 = b.add(m1, b1)?;
    let h = b.relu(a1)?;
    let m2 = b.matmul(h, w2)?;
    let a2 = b.add(m2, b2)?;
    let p = b.sigmoid(a2)?;
    let loss = b.mse_loss(p, y, oxide_ml_graph::Reduction::Mean)?;
    let fwd = b.finish(&[p, loss])?;
    let mut params = ParameterStore::new();
    params.add(
        ParameterId(0),
        "w1",
        DenseTensor::from_vec([3, 6], vec![0.2; 18])?,
        true,
    )?;
    params.add(
        ParameterId(1),
        "b1",
        DenseTensor::from_vec([6], vec![0.0; 6])?,
        true,
    )?;
    params.add(
        ParameterId(2),
        "w2",
        DenseTensor::from_vec([6, 1], vec![-0.1; 6])?,
        true,
    )?;
    params.add(
        ParameterId(3),
        "b2",
        DenseTensor::from_vec([1], vec![0.0])?,
        true,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        fwd.inputs()[0],
        DenseTensor::from_vec([8, 3], (0..24).map(|i| (i as f32 * 0.17).cos()).collect())?,
    );
    inputs.insert(
        fwd.inputs()[1],
        DenseTensor::from_vec([8, 1], (0..8).map(|i| (i as f32 * 0.4).sin()).collect())?,
    );
    make(
        "mlp_mse",
        t,
        fwd,
        loss,
        params,
        inputs,
        "x[8,3] 3->6->1 mse",
        1e-2,
        1e-2,
    )
}
fn add_relu() -> Result<C, Box<dyn std::error::Error>> {
    let t = Instant::now();
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([8, 4]), DType::F32);
    let bias = b.try_parameter("bias", Shape::from_static([4]), DType::F32, true)?;
    let a = b.add(x, bias)?;
    let r = b.relu(a)?;
    let loss = b.reduce_sum(r, vec![0, 1], false)?;
    let fwd = b.finish(&[r, loss])?;
    let mut params = ParameterStore::new();
    params.add(
        ParameterId(0),
        "bias",
        DenseTensor::from_vec([4], vec![0.1, -0.2, 0.3, -0.4])?,
        true,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        fwd.inputs()[0],
        DenseTensor::from_vec([8, 4], (0..32).map(|i| (i as f32 * 0.13).sin()).collect())?,
    );
    make(
        "add_relu",
        t,
        fwd,
        loss,
        params,
        inputs,
        "x[8,4]+bias[4] relu sum",
        5e-3,
        5e-3,
    )
}
fn unbroadcast() -> Result<C, Box<dyn std::error::Error>> {
    let t = Instant::now();
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([4, 3, 2]), DType::F32);
    let bias = b.try_parameter("bias", Shape::from_static([2]), DType::F32, true)?;
    let a = b.add(x, bias)?;
    let loss = b.reduce_sum(a, vec![0, 1, 2], false)?;
    let fwd = b.finish(&[a, loss])?;
    let mut params = ParameterStore::new();
    params.add(
        ParameterId(0),
        "bias",
        DenseTensor::from_vec([2], vec![0.25, -0.5])?,
        true,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        fwd.inputs()[0],
        DenseTensor::from_vec([4, 3, 2], (0..24).map(|i| (i as f32 * 0.2).cos()).collect())?,
    );
    make(
        "unbroadcast",
        t,
        fwd,
        loss,
        params,
        inputs,
        "x[4,3,2]+bias[2] sum",
        5e-3,
        5e-3,
    )
}
fn activation() -> Result<C, Box<dyn std::error::Error>> {
    let t = Instant::now();
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([32]), DType::F32);
    let y = b.try_parameter("y", Shape::from_static([32]), DType::F32, true)?;
    let a = b.relu(x)?;
    let p = b.mul(a, y)?;
    let loss = b.reduce_mean(p, vec![0], false)?;
    let fwd = b.finish(&[p, loss])?;
    let mut params = ParameterStore::new();
    params.add(
        ParameterId(0),
        "y",
        DenseTensor::from_vec([32], (0..32).map(|i| 0.5 + i as f32 * 0.01).collect())?,
        true,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        fwd.inputs()[0],
        DenseTensor::from_vec([32], (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect())?,
    );
    make(
        "activation_relu",
        t,
        fwd,
        loss,
        params,
        inputs,
        "x[32] relu*param mean",
        5e-3,
        5e-3,
    )
}

fn snapshot(executor: &CudaExecutor, plan: &CudaMemoryPlan) -> MemorySnapshot {
    let p = executor.pool_stats();
    let e = executor.event_stats();
    MemorySnapshot {
        logical_bytes: Some(plan.logical_bytes),
        planned_logical_bytes: Some(plan.logical_bytes),
        runtime_required_bytes: Some(plan.logical_bytes),
        pool_reserved_bytes: Some(p.reserved_bytes),
        pool_active_bytes: Some(p.current_bytes),
        pool_pending_release_bytes: Some(p.pending_release_bytes),
        pool_quarantined_bytes: Some(p.quarantined_bytes),
        pool_allocation_count: Some(p.allocations),
        pool_reuse_count: Some(p.reuses),
        arena_capacity_bytes: None,
        arena_used_bytes: None,
        arena_physical_reserved_bytes: None,
        arena_slot_allocation_count: None,
        arena_slot_reuse_count: None,
        arena_active_lease_count: None,
        event_allocation_count: Some(e.allocation_count as u64),
        event_reuse_count: Some(e.reuse_count as u64),
        event_pending_count: Some(e.pending_count as u64),
        event_quarantine_count: Some(e.quarantine_count as u64),
        access_reads: None,
        access_writes: None,
        access_read_write: None,
        access_same_stream_elided: None,
        access_cross_stream_waits: None,
        access_read_read_elision: None,
        driver_used_bytes: None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let pool = CudaBufferPool::new(device.context().clone(), device.default_stream());
    let meta = collect_meta(&device);
    println!(
        "backward_check: gpu={} cc={} uuid={}",
        meta.gpu_name, meta.compute_capability, meta.gpu_uuid
    );
    let mut cases = Vec::new();
    for builder in [linear, mlp, add_relu, unbroadcast, activation] {
        let case = builder()?;
        let mut cpu_inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
        for (&id, tensor) in &case.inputs {
            cpu_inputs.insert(id, tensor.clone());
        }
        let cpu = oxide_ml_cpu::Executor::<f32>::new(case.train.graph.clone())?;
        let cpu_out = cpu.run(&case.params, &cpu_inputs)?;
        let params = CudaParameterStore::from_host(&case.train.graph, &case.params, &pool)?;
        let plan = CudaMemoryPlan::from_memory_plan(
            &case.train.graph,
            &MemoryPlanner::default().plan(&case.train.graph)?,
        )?;
        let run = executor.run(&case.train.graph, &params, &case.inputs)?;
        let mut ok = "PASS".to_string();
        for (i, &v) in case.train.graph.outputs().iter().enumerate() {
            let actual = run.download(&device.default_stream(), v)?;
            if !bench_common::all_finite(actual.data()) {
                ok = format!("FAIL non-finite output[{i}]");
                break;
            }
            if let Err(e) = within_tolerance(
                actual.data(),
                cpu_out.output(i)?.data(),
                case.atol,
                case.rtol,
            ) {
                ok = format!("FAIL output[{i}]: {e}");
                break;
            }
        }
        println!("backward_check: {} training={}", case.name, ok);
        let mut rec = Case::new("backward_train", case.name, "normal", case.shape);
        rec.graph_hash = graph_hash(&case.train.graph);
        rec.plan_hash = plan_hash(&plan);
        rec.compile_ms = 0.0;
        rec.correctness = ok.clone();
        rec.memory = snapshot(&executor, &plan);
        rec.note = "training graph fwd+bwd; sync normal API; planned/arena/async NOT RUN (executor constant-copy panic)".to_string();
        rec.samples = measure_sync(
            &device.default_stream(),
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                executor.run(&case.train.graph, &params, &case.inputs)?;
                Ok(())
            },
        )?;
        cases.push(rec);
        let mut rec = Case::new("backward_forward", case.name, "normal", case.shape);
        rec.graph_hash = graph_hash(&case.fwd);
        rec.plan_hash = plan_hash(&plan);
        rec.compile_ms = 0.0;
        rec.correctness = ok.clone();
        rec.memory = snapshot(&executor, &plan);
        rec.note = "forward-only graph; sync normal API".to_string();
        rec.samples = measure_sync(
            &device.default_stream(),
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                executor.run(&case.fwd, &params, &case.inputs)?;
                Ok(())
            },
        )?;
        cases.push(rec);
    }
    let path = out_dir("../../docs").join("phase7f19-backward-results.json");
    write_results(&path, "backward_check", &meta, &cases)?;
    println!(
        "backward_check: wrote {} cases={}",
        path.display(),
        cases.len()
    );
    Ok(())
}
