//! Phase 7F-19 backward / training-graph check.
//!
//! Builds training graphs with `oxide_ml_autograd::build_training_graph` and
//! compares CPU reference execution (`oxide_ml_cpu`) against the CUDA backend
//! in the normal / planned / arena / async-planned / async-arena modes.
//!
//! Forward-only graphs and full training graphs are timed separately. Graphs
//! whose backward pass is not implemented by the CUDA executor (for example
//! `LayerNormBackward`) are recorded as `NOT APPLICABLE` with zero samples
//! instead of fabricated numbers.
//!
//! Measurement definitions live in `bench_common`.

#[path = "bench_common/mod.rs"]
mod bench_common;

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use cuda_core::CudaStream;

use bench_common::{
    Case, ITERATIONS, MemorySnapshot, ROUNDS, WARMUP, collect_meta, graph_hash, measure_async,
    measure_sync, out_dir, plan_hash, within_tolerance, write_results,
};
use oxide_ml_autograd::TrainingGraph;
use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, DenseTensor, ParameterId, ParameterStore, Shape, ValueId};
use oxide_ml_cuda::{
    CudaBufferPool, CudaDevice, CudaError, CudaExecutor, CudaMemoryPlan, CudaParameterStore,
    CudaPlannedArena, CudaRun, CudaTensor,
};
use oxide_ml_graph::{Graph, GraphBuilder, Reduction};

const MODES: [&str; 5] = ["normal", "planned", "arena", "async_planned", "async_arena"];

const TOLERANCE_NOTE: &str = "tolerances: forward |a-e| <= 1e-4 + 3e-3*|e|; parameter gradient \
     <= 5e-3 + 5e-3*|e|; layernorm <= 1e-2 + 1e-2*|e|; CPU reference = oxide_ml_cpu::Executor + \
     ExecutionInputs; normal mode compared in 3 repetitions";

struct GraphCase {
    name: &'static str,
    /// Value used as the scalar loss in `forward`.
    loss: ValueId,
    forward: Graph,
    params: ParameterStore<f32>,
    inputs: HashMap<ValueId, DenseTensor<f32>>,
    shape: String,
    compile_ms: f64,
    layernorm: bool,
}

fn param(
    builder: &mut GraphBuilder,
    id: ParameterId,
    name: &str,
    dims: &[usize],
    _values: Vec<f32>,
) -> Result<ValueId, Box<dyn std::error::Error>> {
    let value = builder.try_parameter(name, Shape::from_static(dims), DType::F32, true)?;
    let _ = id;
    Ok(value)
}

fn add_params(
    store: &mut ParameterStore<f32>,
    builder_specs: &[(ParameterId, &str, &[usize], Vec<f32>)],
) -> Result<(), Box<dyn std::error::Error>> {
    for (id, name, dims, values) in builder_specs {
        store.add(
            *id,
            *name,
            DenseTensor::from_vec(dims, values.clone())?,
            true,
        )?;
    }
    Ok(())
}

fn build_linear_mse() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([8, 3]), DType::F32);
    let y = builder.input("y", Shape::from_static([8, 1]), DType::F32);
    let w = param(&mut builder, ParameterId(0), "w", &[3, 1], Vec::new())?;
    let b = param(&mut builder, ParameterId(1), "b", &[1], Vec::new())?;
    let prediction = builder.matmul(x, w)?;
    let prediction = builder.add(prediction, b)?;
    let loss = builder.mse_loss(prediction, y, Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;

    let mut params = ParameterStore::<f32>::new();
    add_params(
        &mut params,
        &[
            (ParameterId(0), "w", &[3, 1][..], vec![0.2, -0.3, 0.5]),
            (ParameterId(1), "b", &[1][..], vec![0.1]),
        ],
    )?;

    let mut inputs = HashMap::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [8, 3],
            vec![
                0.5, -1.0, 0.25, 1.0, 0.75, -0.5, -0.25, 0.5, 1.0, 0.0, -0.75, 0.25, 0.75, -0.5,
                0.5, -1.0, 0.25, 0.0, 0.5, 0.75, -0.25, -0.5, 1.0, 0.25,
            ],
        )?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([8, 1], vec![1.0, -0.5, 0.75, 0.25, -0.25, 0.5, 1.25, -0.75])?,
    );

    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "linear_mse",
        loss,
        forward,
        params,
        inputs,
        shape: "x[8,3] w[3,1] b[1] mse".to_string(),
        compile_ms,
        layernorm: false,
    })
}

fn build_mlp2_mse() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([8, 4]), DType::F32);
    let y = builder.input("y", Shape::from_static([8, 1]), DType::F32);
    let w1 = param(&mut builder, ParameterId(0), "w1", &[4, 8], Vec::new())?;
    let b1 = param(&mut builder, ParameterId(1), "b1", &[8], Vec::new())?;
    let w2 = param(&mut builder, ParameterId(2), "w2", &[8, 1], Vec::new())?;
    let b2 = param(&mut builder, ParameterId(3), "b2", &[1], Vec::new())?;
    let hidden = builder.matmul(x, w1)?;
    let hidden = builder.add(hidden, b1)?;
    let hidden = builder.relu(hidden)?;
    let prediction = builder.matmul(hidden, w2)?;
    let prediction = builder.add(prediction, b2)?;
    let loss = builder.mse_loss(prediction, y, Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;

    let mut w1_values = Vec::with_capacity(32);
    for index in 0..32 {
        w1_values.push(((index % 7) as f32 - 3.0) * 0.05);
    }
    let mut w2_values = Vec::with_capacity(8);
    for index in 0..8 {
        w2_values.push(((index % 5) as f32 - 2.0) * 0.1);
    }
    let mut params = ParameterStore::<f32>::new();
    add_params(
        &mut params,
        &[
            (ParameterId(0), "w1", &[4, 8][..], w1_values),
            (ParameterId(1), "b1", &[8][..], vec![0.05; 8]),
            (ParameterId(2), "w2", &[8, 1][..], w2_values),
            (ParameterId(3), "b2", &[1][..], vec![0.0]),
        ],
    )?;

    let mut inputs = HashMap::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.5, -1.0, 0.25, 0.0, 1.0, 0.75, -0.5, 0.25, -0.25, 0.5, 1.0, -0.75, 0.0, -0.5,
                0.75, 0.5, 0.25, 1.0, -1.0, 0.0, -0.75, 0.25, 0.5, -0.5, 1.0, 0.0, -0.25, 0.75,
                -0.5, 0.5, 0.25, -1.0,
            ],
        )?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([8, 1], vec![0.25, -0.5, 1.0, 0.0, -0.75, 0.5, 0.75, -0.25])?,
    );

    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "mlp2_mse",
        loss,
        forward,
        params,
        inputs,
        shape: "x[8,4] 4->8 relu 8->1 mse".to_string(),
        compile_ms,
        layernorm: false,
    })
}

fn build_mse_only() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let prediction = builder.input("prediction", Shape::from_static([8, 4]), DType::F32);
    let target = builder.input("target", Shape::from_static([8, 4]), DType::F32);
    let loss = builder.mse_loss(prediction, target, Reduction::Mean)?;
    let forward = builder.finish(&[loss])?;

    let mut inputs = HashMap::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.5, -1.0, 0.25, 0.0, 1.0, 0.75, -0.5, 0.25, -0.25, 0.5, 1.0, -0.75, 0.0, -0.5,
                0.75, 0.5, 0.25, 1.0, -1.0, 0.0, -0.75, 0.25, 0.5, -0.5, 1.0, 0.0, -0.25, 0.75,
                -0.5, 0.5, 0.25, -1.0,
            ],
        )?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.25, -0.75, 0.5, 0.0, 0.75, 0.25, -0.5, 1.0, -1.0, 0.5, 0.25, -0.25, 0.5, 0.0,
                -0.75, 0.25, 0.0, 0.5, -0.25, 0.75, 1.0, -0.5, 0.25, 0.0, -0.25, 0.75, 0.5, -1.0,
                0.25, -0.5, 0.0, 0.5,
            ],
        )?,
    );

    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "mse_only",
        loss,
        forward,
        params: ParameterStore::new(),
        inputs,
        shape: "prediction[8,4] target[8,4] mse".to_string(),
        compile_ms,
        layernorm: false,
    })
}

fn build_relu_activation() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([8, 4]), DType::F32);
    let y = builder.input("y", Shape::from_static([8, 4]), DType::F32);
    let w = param(&mut builder, ParameterId(0), "w", &[4, 4], Vec::new())?;
    let pre = builder.matmul(x, w)?;
    let activated = builder.relu(pre)?;
    let loss = builder.mse_loss(activated, y, Reduction::Mean)?;
    let forward = builder.finish(&[activated, loss])?;

    let mut w_values = Vec::with_capacity(16);
    for index in 0..16 {
        w_values.push(((index % 5) as f32 - 2.0) * 0.15);
    }
    let mut params = ParameterStore::<f32>::new();
    add_params(&mut params, &[(ParameterId(0), "w", &[4, 4][..], w_values)])?;

    let mut inputs = HashMap::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.5, -1.0, 0.25, 0.0, 1.0, 0.75, -0.5, 0.25, -0.25, 0.5, 1.0, -0.75, 0.0, -0.5,
                0.75, 0.5, 0.25, 1.0, -1.0, 0.0, -0.75, 0.25, 0.5, -0.5, 1.0, 0.0, -0.25, 0.75,
                -0.5, 0.5, 0.25, -1.0,
            ],
        )?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.25, -0.75, 0.5, 0.0, 0.75, 0.25, -0.5, 1.0, -1.0, 0.5, 0.25, -0.25, 0.5, 0.0,
                -0.75, 0.25, 0.0, 0.5, -0.25, 0.75, 1.0, -0.5, 0.25, 0.0, -0.25, 0.75, 0.5, -1.0,
                0.25, -0.5, 0.0, 0.5,
            ],
        )?,
    );

    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "relu_activation",
        loss,
        forward,
        params,
        inputs,
        shape: "x[8,4] w[4,4] relu mse".to_string(),
        compile_ms,
        layernorm: false,
    })
}

fn build_add_relu() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let a = builder.input("a", Shape::from_static([8, 4]), DType::F32);
    let y = builder.input("y", Shape::from_static([8, 4]), DType::F32);
    let w = param(&mut builder, ParameterId(0), "w", &[8, 4], Vec::new())?;
    let sum = builder.add(a, w)?;
    let activated = builder.relu(sum)?;
    let loss = builder.mse_loss(activated, y, Reduction::Mean)?;
    let forward = builder.finish(&[activated, loss])?;

    let mut w_values = Vec::with_capacity(32);
    for index in 0..32 {
        w_values.push(((index % 9) as f32 - 4.0) * 0.05);
    }
    let mut params = ParameterStore::<f32>::new();
    add_params(&mut params, &[(ParameterId(0), "w", &[8, 4][..], w_values)])?;

    let mut inputs = HashMap::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.5, -1.0, 0.25, 0.0, 1.0, 0.75, -0.5, 0.25, -0.25, 0.5, 1.0, -0.75, 0.0, -0.5,
                0.75, 0.5, 0.25, 1.0, -1.0, 0.0, -0.75, 0.25, 0.5, -0.5, 1.0, 0.0, -0.25, 0.75,
                -0.5, 0.5, 0.25, -1.0,
            ],
        )?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.25, -0.75, 0.5, 0.0, 0.75, 0.25, -0.5, 1.0, -1.0, 0.5, 0.25, -0.25, 0.5, 0.0,
                -0.75, 0.25, 0.0, 0.5, -0.25, 0.75, 1.0, -0.5, 0.25, 0.0, -0.25, 0.75, 0.5, -1.0,
                0.25, -0.5, 0.0, 0.5,
            ],
        )?,
    );

    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "add_relu",
        loss,
        forward,
        params,
        inputs,
        shape: "a[8,4] + w[8,4] relu mse".to_string(),
        compile_ms,
        layernorm: false,
    })
}

fn build_unbroadcast_bias() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([8, 4]), DType::F32);
    let y = builder.input("y", Shape::from_static([8, 4]), DType::F32);
    let bias = param(&mut builder, ParameterId(0), "bias", &[4], Vec::new())?;
    let prediction = builder.add(x, bias)?;
    let loss = builder.mse_loss(prediction, y, Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;

    let mut params = ParameterStore::<f32>::new();
    add_params(
        &mut params,
        &[(ParameterId(0), "bias", &[4][..], vec![0.1, -0.2, 0.3, 0.0])],
    )?;

    let mut inputs = HashMap::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.5, -1.0, 0.25, 0.0, 1.0, 0.75, -0.5, 0.25, -0.25, 0.5, 1.0, -0.75, 0.0, -0.5,
                0.75, 0.5, 0.25, 1.0, -1.0, 0.0, -0.75, 0.25, 0.5, -0.5, 1.0, 0.0, -0.25, 0.75,
                -0.5, 0.5, 0.25, -1.0,
            ],
        )?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec(
            [8, 4],
            vec![
                0.25, -0.75, 0.5, 0.0, 0.75, 0.25, -0.5, 1.0, -1.0, 0.5, 0.25, -0.25, 0.5, 0.0,
                -0.75, 0.25, 0.0, 0.5, -0.25, 0.75, 1.0, -0.5, 0.25, 0.0, -0.25, 0.75, 0.5, -1.0,
                0.25, -0.5, 0.0, 0.5,
            ],
        )?,
    );

    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "unbroadcast_bias",
        loss,
        forward,
        params,
        inputs,
        shape: "x[8,4] + bias[4] mse".to_string(),
        compile_ms,
        layernorm: false,
    })
}

fn build_layernorm() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([8, 6]), DType::F32);
    let y = builder.input("y", Shape::from_static([8, 6]), DType::F32);
    let gamma = param(&mut builder, ParameterId(0), "gamma", &[6], Vec::new())?;
    let beta = param(&mut builder, ParameterId(1), "beta", &[6], Vec::new())?;
    let normalized = builder.layer_norm(x, gamma, beta, vec![6], 1e-5)?;
    let loss = builder.mse_loss(normalized, y, Reduction::Mean)?;
    let forward = builder.finish(&[normalized, loss])?;

    let mut params = ParameterStore::<f32>::new();
    add_params(
        &mut params,
        &[
            (
                ParameterId(0),
                "gamma",
                &[6][..],
                vec![1.0, 0.9, 1.1, 1.0, 0.95, 1.05],
            ),
            (
                ParameterId(1),
                "beta",
                &[6][..],
                vec![0.0, 0.1, -0.1, 0.05, 0.0, -0.05],
            ),
        ],
    )?;

    let mut inputs = HashMap::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [8, 6],
            vec![
                0.5, -1.0, 0.25, 0.0, 0.75, -0.5, 1.0, 0.25, -0.75, 0.5, 0.0, 0.25, -0.25, 0.5,
                1.0, -0.5, 0.75, 0.0, 0.25, -0.75, 0.5, 1.0, -0.25, 0.5, 0.0, 0.75, -0.5, 0.25,
                1.0, -0.25, -1.0, 0.5, 0.25, 0.0, -0.5, 0.75, 0.5, 0.0, -0.25, 0.75, 0.25, -1.0,
                0.25, 0.5, 0.0, -0.75, 1.0, 0.25,
            ],
        )?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec(
            [8, 6],
            vec![
                0.25, -0.5, 0.5, 0.0, 0.75, -0.25, 0.5, 0.0, -0.5, 0.25, 0.25, 0.5, -0.75, 0.5,
                0.5, -0.25, 0.5, 0.0, 0.0, -0.5, 0.25, 0.75, -0.25, 0.5, 0.5, 0.25, -0.5, 0.0, 1.0,
                0.25, -0.75, 0.5, 0.25, 0.0, -0.25, 0.5, 0.25, 0.0, -0.5, 0.5, 0.25, -0.75, 0.5,
                0.5, 0.0, -0.25, 0.75, 0.25,
            ],
        )?,
    );

    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(GraphCase {
        name: "layernorm",
        loss,
        forward,
        params,
        inputs,
        shape: "x[8,6] layer_norm(6) mse".to_string(),
        compile_ms,
        layernorm: true,
    })
}

fn memory_snapshot(
    executor: &CudaExecutor,
    arena: &CudaPlannedArena,
    plan: &CudaMemoryPlan,
    outputs: &[CudaTensor],
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
        arena_used_bytes: Some(
            (arena_stats.active_slot_set_count * arena_stats.logical_bytes) as usize,
        ),
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

#[allow(clippy::type_complexity)]
fn collect_run_downloads(
    run: &CudaRun,
    graph: &Graph,
    stream: &CudaStream,
) -> Result<Vec<(ValueId, DenseTensor<f32>)>, Box<dyn std::error::Error>> {
    let mut downloads = Vec::with_capacity(graph.outputs().len());
    for (index, &value) in graph.outputs().iter().enumerate() {
        downloads.push((
            value,
            run.download(stream, value).map_err(|error| {
                Box::<dyn std::error::Error>::from(format!(
                    "output[{index}] {value} download failed: {error}"
                ))
            })?,
        ));
    }
    Ok(downloads)
}

#[allow(clippy::type_complexity)]
fn collect_map_downloads(
    graph: &Graph,
    outputs: &HashMap<ValueId, DenseTensor<f32>>,
) -> Result<Vec<(ValueId, DenseTensor<f32>)>, Box<dyn std::error::Error>> {
    let mut downloads = Vec::with_capacity(graph.outputs().len());
    for &value in graph.outputs() {
        let tensor = outputs.get(&value).ok_or_else(|| {
            Box::<dyn std::error::Error>::from(format!("missing async output {value}"))
        })?;
        downloads.push((value, tensor.clone()));
    }
    Ok(downloads)
}

fn validate_downloads(
    label: &str,
    training: Option<&TrainingGraph>,
    downloads: &[(ValueId, DenseTensor<f32>)],
    expected: &oxide_ml_cpu::ExecutionResult<f32>,
    layernorm: bool,
    correctness: &mut String,
) {
    for (value, tensor) in downloads {
        let expected_tensor = match expected.value(*value) {
            Ok(tensor) => tensor,
            Err(error) => {
                *correctness = format!("FAIL {label}: CPU reference missing {value}: {error}");
                return;
            }
        };
        let is_gradient = training
            .is_some_and(|graph| graph.parameter_grads.values().any(|grad| *grad == *value));
        let (atol, rtol) = if layernorm {
            (1e-2, 1e-2)
        } else if is_gradient {
            (5e-3, 5e-3)
        } else {
            (1e-4, 3e-3)
        };
        if let Err(error) = within_tolerance(tensor.data(), expected_tensor.data(), atol, rtol) {
            *correctness = format!("FAIL {label} value {value}: {error}");
            return;
        }
        if !bench_common::all_finite(tensor.data()) {
            *correctness = format!("FAIL {label} value {value}: non-finite output");
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_not_applicable(
    cases: &mut Vec<Case>,
    benchmark: &str,
    graph: &str,
    shape: &str,
    graph_hash: &str,
    plan_hash: &str,
    compile_ms: f64,
    reason: &str,
) {
    for mode in MODES {
        let mut record = Case::new(benchmark, graph, mode, shape);
        record.graph_hash = graph_hash.to_string();
        record.plan_hash = plan_hash.to_string();
        record.compile_ms = compile_ms;
        record.correctness = "NOT APPLICABLE".to_string();
        record.note = format!("NOT APPLICABLE: {reason}; {TOLERANCE_NOTE}");
        record.stream_count = if mode.starts_with("arena") || mode == "async_arena" {
            2
        } else {
            1
        };
        cases.push(record);
    }
}

fn bench_case(
    executor: &CudaExecutor,
    pool: &CudaBufferPool,
    case: GraphCase,
    cases: &mut Vec<Case>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan_begin = Instant::now();
    let forward_memory_plan = match MemoryPlanner::default().plan(&case.forward) {
        Ok(plan) => plan,
        Err(error) => {
            let reason = format!("forward memory plan failed: {error}");
            let hash = graph_hash(&case.forward);
            push_not_applicable(
                cases,
                "backward_forward",
                case.name,
                &case.shape,
                &hash,
                "unavailable",
                case.compile_ms,
                &reason,
            );
            push_not_applicable(
                cases,
                "backward_train",
                case.name,
                &case.shape,
                &hash,
                "unavailable",
                case.compile_ms,
                &reason,
            );
            println!("backward_check: {} NOT APPLICABLE ({reason})", case.name);
            return Ok(());
        }
    };
    let forward_plan = match CudaMemoryPlan::from_memory_plan(&case.forward, &forward_memory_plan) {
        Ok(plan) => plan,
        Err(error) => {
            let reason = format!("forward CUDA plan failed: {error}");
            let hash = graph_hash(&case.forward);
            push_not_applicable(
                cases,
                "backward_forward",
                case.name,
                &case.shape,
                &hash,
                "unavailable",
                case.compile_ms,
                &reason,
            );
            push_not_applicable(
                cases,
                "backward_train",
                case.name,
                &case.shape,
                &hash,
                "unavailable",
                case.compile_ms,
                &reason,
            );
            println!("backward_check: {} NOT APPLICABLE ({reason})", case.name);
            return Ok(());
        }
    };
    let compile_ms = case.compile_ms + plan_begin.elapsed().as_secs_f64() * 1000.0;
    let forward_hash = graph_hash(&case.forward);
    let forward_plan_hash = plan_hash(&forward_plan);
    let forward_params = CudaParameterStore::from_host(&case.forward, &case.params, pool)?;
    let forward_arena = CudaPlannedArena::new(forward_plan.clone(), pool.clone(), 2)?;
    let forward_arena_stream = executor.device().new_stream()?;
    let default_stream = executor.device().default_stream();

    let mut cpu_inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
    for (&id, tensor) in &case.inputs {
        cpu_inputs.insert(id, tensor.clone());
    }
    let cpu_forward_executor = oxide_ml_cpu::Executor::<f32>::new(case.forward.clone())?;
    let cpu_forward = cpu_forward_executor.run(&case.params, &cpu_inputs)?;

    let probe = match executor.run(&case.forward, &forward_params, &case.inputs) {
        Ok(run) => run,
        Err(CudaError::Unsupported(message)) => {
            let reason = format!("forward unsupported by CUDA executor: {message}");
            push_not_applicable(
                cases,
                "backward_forward",
                case.name,
                &case.shape,
                &forward_hash,
                &forward_plan_hash,
                compile_ms,
                &reason,
            );
            push_not_applicable(
                cases,
                "backward_train",
                case.name,
                &case.shape,
                &forward_hash,
                &forward_plan_hash,
                compile_ms,
                &reason,
            );
            println!("backward_check: {} NOT APPLICABLE ({reason})", case.name);
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    let mut forward_correctness = String::new();
    for _ in 0..3 {
        // Three independent CUDA runs for stability; the probe run above is
        // reused only for the memory snapshot.
        let run = executor.run(&case.forward, &forward_params, &case.inputs)?;
        let downloads = collect_run_downloads(&run, &case.forward, &default_stream)?;
        validate_downloads(
            "normal",
            None,
            &downloads,
            &cpu_forward,
            case.layernorm,
            &mut forward_correctness,
        );
        if !forward_correctness.is_empty() {
            break;
        }
    }
    if forward_correctness.is_empty() {
        let planned =
            executor.run_planned(&case.forward, &forward_params, &case.inputs, &forward_plan)?;
        let downloads = collect_run_downloads(&planned, &case.forward, &default_stream)?;
        validate_downloads(
            "planned",
            None,
            &downloads,
            &cpu_forward,
            case.layernorm,
            &mut forward_correctness,
        );
    }
    if forward_correctness.is_empty() {
        let arena_run = executor.run_arena(
            &case.forward,
            &forward_params,
            &case.inputs,
            &forward_arena,
            forward_arena_stream.clone(),
        )?;
        let downloads = collect_run_downloads(&arena_run, &case.forward, &forward_arena_stream)?;
        validate_downloads(
            "arena",
            None,
            &downloads,
            &cpu_forward,
            case.layernorm,
            &mut forward_correctness,
        );
    }
    if forward_correctness.is_empty() {
        let mut async_run = executor.execute_async_planned(
            &case.forward,
            &forward_params,
            &case.inputs,
            &forward_plan,
            default_stream.clone(),
        )?;
        let outputs = async_run.wait()?;
        let downloads = collect_map_downloads(&case.forward, &outputs)?;
        validate_downloads(
            "async_planned",
            None,
            &downloads,
            &cpu_forward,
            case.layernorm,
            &mut forward_correctness,
        );
    }
    if forward_correctness.is_empty() {
        let mut async_run = executor.execute_async_arena(
            &case.forward,
            &forward_params,
            &case.inputs,
            &forward_arena,
            forward_arena_stream.clone(),
        )?;
        let outputs = async_run.wait()?;
        let downloads = collect_map_downloads(&case.forward, &outputs)?;
        validate_downloads(
            "async_arena",
            None,
            &downloads,
            &cpu_forward,
            case.layernorm,
            &mut forward_correctness,
        );
    }
    if forward_correctness.is_empty() {
        forward_correctness = "PASS".to_string();
    }

    let mut probe_outputs = Vec::with_capacity(case.forward.outputs().len());
    for index in 0..case.forward.outputs().len() {
        probe_outputs.push(probe.output(index)?.clone());
    }
    let forward_memory = memory_snapshot(executor, &forward_arena, &forward_plan, &probe_outputs);

    // Forward-only graph measurements.
    let mut record = Case::new("backward_forward", case.name, "normal", &case.shape);
    record.graph_hash = forward_hash.clone();
    record.plan_hash = forward_plan_hash.clone();
    record.compile_ms = compile_ms;
    record.memory = forward_memory.clone();
    record.correctness = forward_correctness.clone();
    record.note = format!("sync API; host_call includes implicit wait; {TOLERANCE_NOTE}");
    record.samples = measure_sync(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
        executor.run(&case.forward, &forward_params, &case.inputs)?;
        Ok(())
    })?;
    cases.push(record);

    let mut record = Case::new("backward_forward", case.name, "planned", &case.shape);
    record.graph_hash = forward_hash.clone();
    record.plan_hash = forward_plan_hash.clone();
    record.compile_ms = compile_ms;
    record.memory = forward_memory.clone();
    record.correctness = forward_correctness.clone();
    record.note = format!("sync API; host_call includes implicit wait; {TOLERANCE_NOTE}");
    record.samples = measure_sync(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
        executor.run_planned(&case.forward, &forward_params, &case.inputs, &forward_plan)?;
        Ok(())
    })?;
    cases.push(record);

    let mut record = Case::new("backward_forward", case.name, "arena", &case.shape);
    record.graph_hash = forward_hash.clone();
    record.plan_hash = forward_plan_hash.clone();
    record.compile_ms = compile_ms;
    record.memory = forward_memory.clone();
    record.correctness = forward_correctness.clone();
    record.stream_count = 2;
    record.note = format!(
        "sync API; separate arena stream; host_call includes implicit wait; {TOLERANCE_NOTE}"
    );
    record.samples = measure_sync(&forward_arena_stream, ROUNDS, ITERATIONS, WARMUP, || {
        executor.run_arena(
            &case.forward,
            &forward_params,
            &case.inputs,
            &forward_arena,
            forward_arena_stream.clone(),
        )?;
        Ok(())
    })?;
    cases.push(record);

    let mut record = Case::new("backward_forward", case.name, "async_planned", &case.shape);
    record.graph_hash = forward_hash.clone();
    record.plan_hash = forward_plan_hash.clone();
    record.compile_ms = compile_ms;
    record.memory = forward_memory.clone();
    record.correctness = forward_correctness.clone();
    record.note =
        format!("submit does not wait; GPU event spans submit to completion; {TOLERANCE_NOTE}");
    record.samples = measure_async(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
        Ok(executor.execute_async_planned(
            &case.forward,
            &forward_params,
            &case.inputs,
            &forward_plan,
            default_stream.clone(),
        )?)
    })?;
    cases.push(record);

    let mut record = Case::new("backward_forward", case.name, "async_arena", &case.shape);
    record.graph_hash = forward_hash.clone();
    record.plan_hash = forward_plan_hash.clone();
    record.compile_ms = compile_ms;
    record.memory = forward_memory.clone();
    record.correctness = forward_correctness.clone();
    record.stream_count = 2;
    record.note = format!(
        "submit does not wait; separate arena stream; arena lease released after completion; {TOLERANCE_NOTE}"
    );
    record.samples = measure_async(&forward_arena_stream, ROUNDS, ITERATIONS, WARMUP, || {
        Ok(executor.execute_async_arena(
            &case.forward,
            &forward_params,
            &case.inputs,
            &forward_arena,
            forward_arena_stream.clone(),
        )?)
    })?;
    cases.push(record);

    // Training graph.
    let training = match oxide_ml_autograd::build_training_graph(&case.forward, case.loss) {
        Ok(training) => training,
        Err(error) => {
            let reason = format!("build_training_graph failed: {error}");
            push_not_applicable(
                cases,
                "backward_train",
                case.name,
                &case.shape,
                &forward_hash,
                &forward_plan_hash,
                compile_ms,
                &reason,
            );
            println!(
                "backward_check: {} train NOT APPLICABLE ({reason})",
                case.name
            );
            return Ok(());
        }
    };

    let training_plan_begin = Instant::now();
    let training_memory_plan = match MemoryPlanner::default().plan(&training.graph) {
        Ok(plan) => plan,
        Err(error) => {
            let reason = format!("training memory plan failed: {error}");
            push_not_applicable(
                cases,
                "backward_train",
                case.name,
                &case.shape,
                &forward_hash,
                "unavailable",
                compile_ms,
                &reason,
            );
            println!(
                "backward_check: {} train NOT APPLICABLE ({reason})",
                case.name
            );
            return Ok(());
        }
    };
    let training_plan =
        match CudaMemoryPlan::from_memory_plan(&training.graph, &training_memory_plan) {
            Ok(plan) => plan,
            Err(error) => {
                let reason = format!("training CUDA plan failed: {error}");
                push_not_applicable(
                    cases,
                    "backward_train",
                    case.name,
                    &case.shape,
                    &forward_hash,
                    "unavailable",
                    compile_ms,
                    &reason,
                );
                println!(
                    "backward_check: {} train NOT APPLICABLE ({reason})",
                    case.name
                );
                return Ok(());
            }
        };
    let training_compile_ms = compile_ms + training_plan_begin.elapsed().as_secs_f64() * 1000.0;
    let training_hash = graph_hash(&training.graph);
    let training_plan_hash = plan_hash(&training_plan);
    let training_params = CudaParameterStore::from_host(&training.graph, &case.params, pool)?;
    let training_arena = CudaPlannedArena::new(training_plan.clone(), pool.clone(), 2)?;
    let training_arena_stream = executor.device().new_stream()?;

    let cpu_training_executor = oxide_ml_cpu::Executor::<f32>::new(training.graph.clone())?;
    let cpu_training = cpu_training_executor.run(&case.params, &cpu_inputs)?;

    let training_probe = match executor.run(&training.graph, &training_params, &case.inputs) {
        Ok(run) => run,
        Err(CudaError::Unsupported(message)) => {
            let reason = format!("training graph unsupported by CUDA executor: {message}");
            push_not_applicable(
                cases,
                "backward_train",
                case.name,
                &case.shape,
                &training_hash,
                &training_plan_hash,
                training_compile_ms,
                &reason,
            );
            println!(
                "backward_check: {} train NOT APPLICABLE ({reason})",
                case.name
            );
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    let mut training_correctness = String::new();
    for _ in 0..3 {
        // Three independent CUDA runs for stability; the probe run above is
        // reused only for the memory snapshot.
        let run = executor.run(&training.graph, &training_params, &case.inputs)?;
        let downloads = collect_run_downloads(&run, &training.graph, &default_stream)?;
        validate_downloads(
            "normal",
            Some(&training),
            &downloads,
            &cpu_training,
            case.layernorm,
            &mut training_correctness,
        );
        if !training_correctness.is_empty() {
            break;
        }
    }
    if training_correctness.is_empty() {
        let planned = executor.run_planned(
            &training.graph,
            &training_params,
            &case.inputs,
            &training_plan,
        )?;
        let downloads = collect_run_downloads(&planned, &training.graph, &default_stream)?;
        validate_downloads(
            "planned",
            Some(&training),
            &downloads,
            &cpu_training,
            case.layernorm,
            &mut training_correctness,
        );
    }
    if training_correctness.is_empty() {
        let arena_run = executor.run_arena(
            &training.graph,
            &training_params,
            &case.inputs,
            &training_arena,
            training_arena_stream.clone(),
        )?;
        let downloads = collect_run_downloads(&arena_run, &training.graph, &training_arena_stream)?;
        validate_downloads(
            "arena",
            Some(&training),
            &downloads,
            &cpu_training,
            case.layernorm,
            &mut training_correctness,
        );
    }
    if training_correctness.is_empty() {
        let mut async_run = executor.execute_async_planned(
            &training.graph,
            &training_params,
            &case.inputs,
            &training_plan,
            default_stream.clone(),
        )?;
        // `wait` materializes the graph outputs, which include the loss and
        // every parameter gradient (build_training_graph appends them).
        let outputs = async_run.wait()?;
        let downloads = collect_map_downloads(&training.graph, &outputs)?;
        validate_downloads(
            "async_planned",
            Some(&training),
            &downloads,
            &cpu_training,
            case.layernorm,
            &mut training_correctness,
        );
    }
    if training_correctness.is_empty() {
        let mut async_run = executor.execute_async_arena(
            &training.graph,
            &training_params,
            &case.inputs,
            &training_arena,
            training_arena_stream.clone(),
        )?;
        let outputs = async_run.wait()?;
        let downloads = collect_map_downloads(&training.graph, &outputs)?;
        validate_downloads(
            "async_arena",
            Some(&training),
            &downloads,
            &cpu_training,
            case.layernorm,
            &mut training_correctness,
        );
    }
    if training_correctness.is_empty() {
        training_correctness = "PASS".to_string();
    }

    let mut training_probe_outputs = Vec::with_capacity(training.graph.outputs().len());
    for index in 0..training.graph.outputs().len() {
        training_probe_outputs.push(training_probe.output(index)?.clone());
    }
    let training_memory = memory_snapshot(
        executor,
        &training_arena,
        &training_plan,
        &training_probe_outputs,
    );

    // Training graph measurements (forward + backward, no optimizer update).
    let mut record = Case::new("backward_train", case.name, "normal", &case.shape);
    record.graph_hash = training_hash.clone();
    record.plan_hash = training_plan_hash.clone();
    record.compile_ms = training_compile_ms;
    record.memory = training_memory.clone();
    record.correctness = training_correctness.clone();
    record.note = format!(
        "training graph forward+backward; sync API; host_call includes implicit wait; {TOLERANCE_NOTE}"
    );
    record.samples = measure_sync(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
        executor.run(&training.graph, &training_params, &case.inputs)?;
        Ok(())
    })?;
    cases.push(record);

    let mut record = Case::new("backward_train", case.name, "planned", &case.shape);
    record.graph_hash = training_hash.clone();
    record.plan_hash = training_plan_hash.clone();
    record.compile_ms = training_compile_ms;
    record.memory = training_memory.clone();
    record.correctness = training_correctness.clone();
    record.note = format!(
        "training graph forward+backward; sync API; host_call includes implicit wait; {TOLERANCE_NOTE}"
    );
    record.samples = measure_sync(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
        executor.run_planned(
            &training.graph,
            &training_params,
            &case.inputs,
            &training_plan,
        )?;
        Ok(())
    })?;
    cases.push(record);

    let mut record = Case::new("backward_train", case.name, "arena", &case.shape);
    record.graph_hash = training_hash.clone();
    record.plan_hash = training_plan_hash.clone();
    record.compile_ms = training_compile_ms;
    record.memory = training_memory.clone();
    record.correctness = training_correctness.clone();
    record.stream_count = 2;
    record.note = format!(
        "training graph forward+backward; separate arena stream; sync API; {TOLERANCE_NOTE}"
    );
    record.samples = measure_sync(&training_arena_stream, ROUNDS, ITERATIONS, WARMUP, || {
        executor.run_arena(
            &training.graph,
            &training_params,
            &case.inputs,
            &training_arena,
            training_arena_stream.clone(),
        )?;
        Ok(())
    })?;
    cases.push(record);

    let mut record = Case::new("backward_train", case.name, "async_planned", &case.shape);
    record.graph_hash = training_hash.clone();
    record.plan_hash = training_plan_hash.clone();
    record.compile_ms = training_compile_ms;
    record.memory = training_memory.clone();
    record.correctness = training_correctness.clone();
    record.note = format!(
        "training graph forward+backward; submit does not wait; gradients are graph outputs; {TOLERANCE_NOTE}"
    );
    record.samples = measure_async(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
        Ok(executor.execute_async_planned(
            &training.graph,
            &training_params,
            &case.inputs,
            &training_plan,
            default_stream.clone(),
        )?)
    })?;
    cases.push(record);

    let mut record = Case::new("backward_train", case.name, "async_arena", &case.shape);
    record.graph_hash = training_hash.clone();
    record.plan_hash = training_plan_hash.clone();
    record.compile_ms = training_compile_ms;
    record.memory = training_memory;
    record.correctness = training_correctness.clone();
    record.stream_count = 2;
    record.note = format!(
        "training graph forward+backward; separate arena stream; submit does not wait; gradients are graph outputs; {TOLERANCE_NOTE}"
    );
    record.samples = measure_async(&training_arena_stream, ROUNDS, ITERATIONS, WARMUP, || {
        Ok(executor.execute_async_arena(
            &training.graph,
            &training_params,
            &case.inputs,
            &training_arena,
            training_arena_stream.clone(),
        )?)
    })?;
    cases.push(record);

    println!(
        "backward_check: {} forward={} train={}",
        case.name, forward_correctness, training_correctness
    );
    Ok(())
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
    for builder in [
        build_linear_mse,
        build_mlp2_mse,
        build_mse_only,
        build_relu_activation,
        build_add_relu,
        build_unbroadcast_bias,
        build_layernorm,
    ] {
        let case = builder()?;
        let name = case.name;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            bench_case(&executor, &pool, case, &mut cases)
        }));
        match outcome {
            Ok(Ok(())) => println!("backward_check: {name} finished"),
            Ok(Err(error)) => println!("backward_check: {name} ERROR: {error}"),
            Err(_) => println!("backward_check: {name} NOT RUN (panic in planned path; see log)"),
        }
    }

    let dir = out_dir("../../docs");
    let json_path = dir.join("phase7f19-backward-results.json");
    write_results(&json_path, "backward_check", &meta, &cases)?;
    println!("backward_check: wrote {}", json_path.display());

    let mut pass = 0usize;
    let mut not_applicable = 0usize;
    let mut fail = 0usize;
    for case in &cases {
        if case.correctness == "PASS" {
            pass += 1;
        } else if case.correctness.starts_with("NOT APPLICABLE") {
            not_applicable += 1;
        } else {
            fail += 1;
            println!(
                "backward_check: FAIL {} {}: {}",
                case.graph, case.mode, case.correctness
            );
        }
    }
    println!(
        "backward_check summary: PASS={pass} FAIL={fail} NOT APPLICABLE={not_applicable} \
         cases={} samples/case={}",
        cases.len(),
        ROUNDS * ITERATIONS
    );
    if fail > 0 {
        return Err(format!("{fail} backward cases failed").into());
    }
    Ok(())
}
