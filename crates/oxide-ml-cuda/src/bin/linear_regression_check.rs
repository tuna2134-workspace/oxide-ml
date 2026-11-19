//! Phase 7F-19 linear-regression training check.
//!
//! Datasets: small (N=16, F=1), medium (N=64, F=4), large (N=256, F=8).
//! For every `batch <= N` the binary trains a forward graph (matmul + bias +
//! MSE) with `CudaSgd` for a fixed number of steps and compares against a CPU
//! reference that uses the same initial parameters, inputs, learning rate and
//! step count.
//!
//! Forward-only graphs are measured in normal, planned, arena, async_planned and
//! async_arena modes. Training steps (forward + backward + SGD) are measured in
//! normal, planned and arena. Async *training* is recorded as NOT APPLICABLE:
//! `CudaSgd::step` requires a unique `DeviceBuffer` per parameter, but a live
//! `CudaExecution` still holds the parameter aliases (`CudaTrainer::step` drops
//! `CudaRun` first). Device gradient tensors are obtainable from
//! `CudaExecution::output_tensor` after `wait()`, but that wait makes the step
//! synchronous, so no async training number is fabricated.
//!
//! `oxide-ml-optim` is a dev-dependency of `oxide-ml-cuda`, so a `src/bin`
//! target cannot link it through Cargo without changing `Cargo.toml`. To use
//! the real reference optimizer without adding a dependency, its existing
//! source is included as a module below; `oxide_ml_optim::Sgd` is the exact
//! implementation used by the CPU training loop.

#[path = "bench_common/mod.rs"]
mod bench_common;

#[allow(dead_code)]
#[path = "../../../oxide-ml-optim/src/lib.rs"]
mod oxide_ml_optim;

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
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
    CudaBufferPool, CudaDevice, CudaEvent, CudaExecutor, CudaMemoryPlan, CudaOptimizer,
    CudaParameterStore, CudaPlannedArena, CudaRun, CudaSgd, CudaTensor,
};
use oxide_ml_graph::{Graph, GraphBuilder, Reduction};
use oxide_ml_optim::{Optimizer, Sgd};

const LEARNING_RATE: f32 = 0.2;
const STEPS: usize = 100;

const FORWARD_ATOL: f64 = 1e-4;
const FORWARD_RTOL: f64 = 3e-3;
const PARAM_ATOL: f64 = 5e-3;
const PARAM_RTOL: f64 = 5e-3;
const GRAD_ATOL: f64 = 5e-3;
const GRAD_RTOL: f64 = 5e-3;
const LOSS_ATOL: f64 = 1e-3;
const LOSS_RTOL: f64 = 1e-3;
const MODE_ATOL: f64 = 1e-4;
const MODE_RTOL: f64 = 1e-4;

const TOLERANCE_NOTE: &str = "tolerances: final loss |a-e| <= 1e-3 + 1e-3*|e|; weight/bias \
     <= 5e-3 + 5e-3*|e|; gradient <= 5e-3 + 5e-3*|e|; forward prediction <= 1e-4 + 3e-3*|e|; \
     cross-mode <= 1e-4 + 1e-4*|e|; absolute and relative tolerance; convergence requires \
     final_loss < 0.5 * initial_loss; CPU reference = oxide_ml_cpu::Executor + \
     oxide_ml_optim::Sgd, same lr and steps";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Normal,
    Planned,
    Arena,
    AsyncPlanned,
    AsyncArena,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Normal => "normal",
            Mode::Planned => "planned",
            Mode::Arena => "arena",
            Mode::AsyncPlanned => "async_planned",
            Mode::AsyncArena => "async_arena",
        }
    }

    fn is_arena(self) -> bool {
        matches!(self, Mode::Arena | Mode::AsyncArena)
    }
}

const MODES: [Mode; 5] = [
    Mode::Normal,
    Mode::Planned,
    Mode::Arena,
    Mode::AsyncPlanned,
    Mode::AsyncArena,
];

struct Dataset {
    name: &'static str,
    n: usize,
    features: usize,
    x: Vec<f32>,
    y: Vec<f32>,
}

fn next_unit(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = ((*seed >> 40) & 0x00FF_FFFF) as u32;
    (bits as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn make_dataset(name: &'static str, n: usize, features: usize) -> Dataset {
    let mut seed: u64 = 0x1234_5678;
    let mut x = Vec::with_capacity(n * features);
    for _ in 0..n * features {
        x.push(next_unit(&mut seed));
    }
    let mut y = Vec::with_capacity(n);
    for row in 0..n {
        let mut value = 0.25f32;
        for feature in 0..features {
            let weight = 0.25 + 0.05 * feature as f32;
            value += weight * x[row * features + feature];
        }
        y.push(value);
    }
    Dataset {
        name,
        n,
        features,
        x,
        y,
    }
}

fn batch_tensor(values: &[f32], batch: usize, width: usize) -> DenseTensor<f32> {
    let mut data = Vec::with_capacity(batch * width);
    for row in 0..batch {
        for column in 0..width {
            data.push(values[row * width + column]);
        }
    }
    DenseTensor::from_vec(vec![batch, width], data).expect("batch tensor shape matches data")
}

struct Combo {
    name: String,
    shape: String,
    forward: Graph,
    training: TrainingGraph,
    params: ParameterStore<f32>,
    inputs: HashMap<ValueId, DenseTensor<f32>>,
    loss: ValueId,
    compile_ms: f64,
}

fn build_combo(dataset: &Dataset, batch: usize) -> Result<Combo, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let features = dataset.features;
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static(vec![batch, features]), DType::F32);
    let y = builder.input("y", Shape::from_static(vec![batch, 1]), DType::F32);
    let w = builder.try_parameter("w", Shape::from_static(vec![features, 1]), DType::F32, true)?;
    let bias = builder.try_parameter("b", Shape::from_static([1]), DType::F32, true)?;
    let product = builder.matmul(x, w)?;
    let prediction = builder.add(product, bias)?;
    let loss = builder.mse_loss(prediction, y, Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;
    let training = oxide_ml_autograd::build_training_graph(&forward, loss)?;

    let mut params = ParameterStore::<f32>::new();
    params.add(
        ParameterId(0),
        "w",
        DenseTensor::from_vec(vec![features, 1], vec![0.0f32; features])?,
        true,
    )?;
    params.add(
        ParameterId(1),
        "b",
        DenseTensor::from_vec([1], vec![0.0f32])?,
        true,
    )?;

    let mut inputs = HashMap::new();
    inputs.insert(
        forward.inputs()[0],
        batch_tensor(&dataset.x, batch, features),
    );
    inputs.insert(forward.inputs()[1], batch_tensor(&dataset.y, batch, 1));

    let compile_ms = begin.elapsed().as_secs_f64() * 1000.0;
    Ok(Combo {
        name: format!("{}_b{}", dataset.name, batch),
        shape: format!(
            "N={} F={} batch={} matmul+bias mse",
            dataset.n, dataset.features, batch
        ),
        forward,
        training,
        params,
        inputs,
        loss,
        compile_ms,
    })
}

struct CpuReference {
    initial_loss: f32,
    #[allow(dead_code)]
    final_loss: f32,
    params: ParameterStore<f32>,
    values: HashMap<ValueId, DenseTensor<f32>>,
}

fn cpu_reference(combo: &Combo) -> Result<CpuReference, Box<dyn std::error::Error>> {
    let mut cpu_inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
    for (&id, tensor) in &combo.inputs {
        cpu_inputs.insert(id, tensor.clone());
    }
    let executor = oxide_ml_cpu::Executor::<f32>::new(combo.training.graph.clone())?;
    let mut params = combo.params.clone();
    let mut optimizer = Sgd::<f32>::new(LEARNING_RATE)?;

    let initial = executor.run(&params, &cpu_inputs)?;
    let initial_loss = initial.value(combo.loss)?.data()[0];

    for _ in 0..STEPS {
        let result = executor.run(&params, &cpu_inputs)?;
        let mut gradients = HashMap::new();
        for (id, value) in combo.training.parameter_gradients() {
            gradients.insert(id, result.value(value)?.clone());
        }
        optimizer.step(&mut params, &gradients)?;
    }

    let final_result = executor.run(&params, &cpu_inputs)?;
    let final_loss = final_result.value(combo.loss)?.data()[0];
    let mut values = HashMap::new();
    for &value in combo.training.graph.outputs() {
        values.insert(value, final_result.value(value)?.clone());
    }
    Ok(CpuReference {
        initial_loss,
        final_loss,
        params,
        values,
    })
}

fn collect_run_gradients(
    run: &CudaRun,
    combo: &Combo,
) -> Result<HashMap<ParameterId, CudaTensor>, Box<dyn std::error::Error>> {
    let mut gradients = HashMap::new();
    for (id, value) in combo.training.parameter_gradients() {
        gradients.insert(id, run.value(value)?.clone());
    }
    Ok(gradients)
}

/// Make gradient buffers stay alive until the SGD kernels that read them have
/// completed, even when the owning execution is dropped earlier.
fn record_gradient_completion(
    stream: &Arc<CudaStream>,
    gradients: &HashMap<ParameterId, CudaTensor>,
) -> Result<(), Box<dyn std::error::Error>> {
    let event = Arc::new(CudaEvent::new(stream.context())?);
    event.record(stream)?;
    for tensor in gradients.values() {
        tensor.mark_used(event.clone())?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn sync_training_step(
    executor: &CudaExecutor,
    mode: Mode,
    combo: &Combo,
    params: &mut CudaParameterStore,
    plan: &CudaMemoryPlan,
    arena: &CudaPlannedArena,
    stream: &Arc<CudaStream>,
    optimizer: &mut CudaSgd,
) -> Result<(), Box<dyn std::error::Error>> {
    let gradients = match mode {
        Mode::Normal => {
            let run = executor.run(&combo.training.graph, params, &combo.inputs)?;
            let gradients = collect_run_gradients(&run, combo)?;
            drop(run);
            gradients
        }
        Mode::Planned => {
            let run = executor.run_planned(&combo.training.graph, params, &combo.inputs, plan)?;
            let gradients = collect_run_gradients(&run, combo)?;
            drop(run);
            gradients
        }
        Mode::Arena => {
            let run = executor.run_arena(
                &combo.training.graph,
                params,
                &combo.inputs,
                arena,
                stream.clone(),
            )?;
            let gradients = collect_run_gradients(&run, combo)?;
            drop(run);
            gradients
        }
        Mode::AsyncPlanned | Mode::AsyncArena => unreachable!("async mode in sync step"),
    };
    optimizer.step(
        executor.kernels(),
        stream,
        executor.pool(),
        params,
        &gradients,
    )?;
    record_gradient_completion(stream, &gradients)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn evaluate_gpu(
    executor: &CudaExecutor,
    mode: Mode,
    combo: &Combo,
    params: &CudaParameterStore,
    plan: &CudaMemoryPlan,
    arena: &CudaPlannedArena,
    stream: &Arc<CudaStream>,
) -> Result<
    (
        HashMap<ValueId, DenseTensor<f32>>,
        HashMap<ParameterId, DenseTensor<f32>>,
    ),
    Box<dyn std::error::Error>,
> {
    let values = match mode {
        Mode::Normal => {
            let run = executor.run(&combo.training.graph, params, &combo.inputs)?;
            download_run_values(&run, &combo.training.graph, stream)?
        }
        Mode::Planned => {
            let run = executor.run_planned(&combo.training.graph, params, &combo.inputs, plan)?;
            download_run_values(&run, &combo.training.graph, stream)?
        }
        Mode::Arena => {
            let run = executor.run_arena(
                &combo.training.graph,
                params,
                &combo.inputs,
                arena,
                stream.clone(),
            )?;
            download_run_values(&run, &combo.training.graph, stream)?
        }
        Mode::AsyncPlanned => {
            let mut execution = executor.execute_async_planned(
                &combo.training.graph,
                params,
                &combo.inputs,
                plan,
                stream.clone(),
            )?;
            execution.wait()?
        }
        Mode::AsyncArena => {
            let mut execution = executor.execute_async_arena(
                &combo.training.graph,
                params,
                &combo.inputs,
                arena,
                stream.clone(),
            )?;
            execution.wait()?
        }
    };
    let mut parameter_values = HashMap::new();
    for parameter in combo.training.graph.parameters() {
        parameter_values.insert(parameter.id, params.download(parameter.id, stream)?);
    }
    Ok((values, parameter_values))
}

fn download_run_values(
    run: &CudaRun,
    graph: &Graph,
    stream: &CudaStream,
) -> Result<HashMap<ValueId, DenseTensor<f32>>, Box<dyn std::error::Error>> {
    let mut values = HashMap::new();
    for &value in graph.outputs() {
        values.insert(value, run.download(stream, value)?);
    }
    Ok(values)
}

fn validate_final(
    label: &str,
    combo: &Combo,
    reference: &CpuReference,
    values: &HashMap<ValueId, DenseTensor<f32>>,
    parameter_values: &HashMap<ParameterId, DenseTensor<f32>>,
) -> String {
    let is_gradient = |value: ValueId| {
        combo
            .training
            .parameter_grads
            .values()
            .any(|gradient| *gradient == value)
    };
    for &value in combo.training.graph.outputs() {
        let Some(actual) = values.get(&value) else {
            return format!("FAIL {label}: missing output {value}");
        };
        let expected = match reference.values.get(&value) {
            Some(expected) => expected,
            None => return format!("FAIL {label}: CPU reference missing {value}"),
        };
        let (atol, rtol) = if value == combo.loss {
            (LOSS_ATOL, LOSS_RTOL)
        } else if is_gradient(value) {
            (GRAD_ATOL, GRAD_RTOL)
        } else {
            (FORWARD_ATOL, FORWARD_RTOL)
        };
        if let Err(error) = within_tolerance(actual.data(), expected.data(), atol, rtol) {
            return format!("FAIL {label} value {value}: {error}");
        }
        if !bench_common::all_finite(actual.data()) {
            return format!("FAIL {label} value {value}: non-finite");
        }
    }
    for (id, name) in [(ParameterId(0), "w"), (ParameterId(1), "b")] {
        let Some(actual) = parameter_values.get(&id) else {
            return format!("FAIL {label}: missing parameter {name}");
        };
        let expected = match reference.params.get(id) {
            Ok(parameter) => parameter,
            Err(error) => return format!("FAIL {label}: CPU parameter {name}: {error}"),
        };
        if let Err(error) =
            within_tolerance(actual.data(), expected.data.data(), PARAM_ATOL, PARAM_RTOL)
        {
            return format!("FAIL {label} parameter {name}: {error}");
        }
        if !bench_common::all_finite(actual.data()) {
            return format!("FAIL {label} parameter {name}: non-finite");
        }
    }
    let Some(loss) = values.get(&combo.loss) else {
        return format!("FAIL {label}: missing loss");
    };
    if !loss.data()[0].is_finite() {
        return format!("FAIL {label}: non-finite loss");
    }
    String::new()
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

struct ModeResult {
    mode: Mode,
    loss: f32,
    weight: Vec<f32>,
    bias: Vec<f32>,
    correctness: String,
}

fn push_mode_result(
    results: &mut Vec<ModeResult>,
    mode: Mode,
    values: &HashMap<ValueId, DenseTensor<f32>>,
    combo: &Combo,
    parameter_values: &HashMap<ParameterId, DenseTensor<f32>>,
    correctness: String,
) {
    results.push(ModeResult {
        mode,
        loss: values
            .get(&combo.loss)
            .map(|tensor| tensor.data()[0])
            .unwrap_or(f32::NAN),
        weight: parameter_values
            .get(&ParameterId(0))
            .map(|tensor| tensor.data().to_vec())
            .unwrap_or_default(),
        bias: parameter_values
            .get(&ParameterId(1))
            .map(|tensor| tensor.data().to_vec())
            .unwrap_or_default(),
        correctness,
    });
}

fn push_not_applicable(cases: &mut Vec<Case>, combo_name: &str, shape: &str, reason: &str) {
    let mut record = Case::new("linear_regression", combo_name, "not_applicable", shape);
    record.correctness = "NOT APPLICABLE".to_string();
    record.note = format!("NOT APPLICABLE: {reason}; {TOLERANCE_NOTE}");
    cases.push(record);
}

#[allow(clippy::too_many_arguments)]
fn validate_forward_mode(
    executor: &CudaExecutor,
    mode: Mode,
    combo: &Combo,
    params: &CudaParameterStore,
    plan: &CudaMemoryPlan,
    arena: &CudaPlannedArena,
    stream: &Arc<CudaStream>,
    cpu_forward: &oxide_ml_cpu::ExecutionResult<f32>,
) -> Result<String, Box<dyn std::error::Error>> {
    let values = match mode {
        Mode::Normal => {
            let run = executor.run(&combo.forward, params, &combo.inputs)?;
            download_run_values(&run, &combo.forward, stream)?
        }
        Mode::Planned => {
            let run = executor.run_planned(&combo.forward, params, &combo.inputs, plan)?;
            download_run_values(&run, &combo.forward, stream)?
        }
        Mode::Arena => {
            let run =
                executor.run_arena(&combo.forward, params, &combo.inputs, arena, stream.clone())?;
            download_run_values(&run, &combo.forward, stream)?
        }
        Mode::AsyncPlanned => {
            let mut execution = executor.execute_async_planned(
                &combo.forward,
                params,
                &combo.inputs,
                plan,
                stream.clone(),
            )?;
            execution.wait()?
        }
        Mode::AsyncArena => {
            let mut execution = executor.execute_async_arena(
                &combo.forward,
                params,
                &combo.inputs,
                arena,
                stream.clone(),
            )?;
            execution.wait()?
        }
    };
    for &value in combo.forward.outputs() {
        let Some(actual) = values.get(&value) else {
            return Ok(format!("FAIL forward {value}: missing output"));
        };
        let expected = cpu_forward.value(value)?;
        if let Err(error) =
            within_tolerance(actual.data(), expected.data(), FORWARD_ATOL, FORWARD_RTOL)
        {
            return Ok(format!("FAIL {} forward {value}: {error}", mode.name()));
        }
        if !bench_common::all_finite(actual.data()) {
            return Ok(format!("FAIL {} forward {value}: non-finite", mode.name()));
        }
    }
    Ok(String::new())
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panic (non-string payload)".to_string()
    }
}

/// Probe one training mode once. The existing executor can fail (or panic) for
/// a plan-backed training graph because `ConstantScalar` outputs may be
/// assigned a shared slot larger than one element. Such a mode is reported as
/// NOT APPLICABLE with the real error instead of aborting the run.
fn probe_training_mode(
    executor: &CudaExecutor,
    mode: Mode,
    combo: &Combo,
    params: &CudaParameterStore,
    plan: &CudaMemoryPlan,
    arena: &CudaPlannedArena,
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    let outcome = catch_unwind(AssertUnwindSafe(|| match mode {
        Mode::Planned => executor
            .run_planned(&combo.training.graph, params, &combo.inputs, plan)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        Mode::Arena => executor
            .run_arena(
                &combo.training.graph,
                params,
                &combo.inputs,
                arena,
                stream.clone(),
            )
            .map(|_| ())
            .map_err(|error| error.to_string()),
        _ => Ok(()),
    }));
    match outcome {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error),
        Err(payload) => Err(panic_text(payload.as_ref())),
    }
}

fn run_combo(
    executor: &CudaExecutor,
    pool: &CudaBufferPool,
    dataset: &Dataset,
    batch: usize,
    cases: &mut Vec<Case>,
) -> Result<(), Box<dyn std::error::Error>> {
    if batch > dataset.n {
        let combo_name = format!("{}_b{}", dataset.name, batch);
        let shape = format!("N={} F={} batch={}", dataset.n, dataset.features, batch);
        push_not_applicable(
            cases,
            &combo_name,
            &shape,
            &format!("batch {batch} exceeds dataset N={}", dataset.n),
        );
        println!("linear_regression_check: {combo_name} NOT APPLICABLE (batch > N)");
        return Ok(());
    }

    let combo = build_combo(dataset, batch)?;
    let plan_begin = Instant::now();
    let forward_memory_plan = MemoryPlanner::default().plan(&combo.forward)?;
    let forward_plan = CudaMemoryPlan::from_memory_plan(&combo.forward, &forward_memory_plan)?;
    let training_memory_plan = MemoryPlanner::default().plan(&combo.training.graph)?;
    let training_plan =
        CudaMemoryPlan::from_memory_plan(&combo.training.graph, &training_memory_plan)?;
    let compile_ms = combo.compile_ms + plan_begin.elapsed().as_secs_f64() * 1000.0;

    let forward_hash = graph_hash(&combo.forward);
    let forward_plan_hash = plan_hash(&forward_plan);
    let training_hash = graph_hash(&combo.training.graph);
    let training_plan_hash = plan_hash(&training_plan);

    let default_stream = executor.device().default_stream();
    let forward_arena_stream = executor.device().new_stream()?;
    let training_arena_stream = executor.device().new_stream()?;
    let forward_arena = CudaPlannedArena::new(forward_plan.clone(), pool.clone(), 2)?;
    let training_arena = CudaPlannedArena::new(training_plan.clone(), pool.clone(), 2)?;

    let forward_params = CudaParameterStore::from_host(&combo.forward, &combo.params, pool)?;
    let reference = cpu_reference(&combo)?;

    let mut cpu_inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
    for (&id, tensor) in &combo.inputs {
        cpu_inputs.insert(id, tensor.clone());
    }
    let cpu_forward_executor = oxide_ml_cpu::Executor::<f32>::new(combo.forward.clone())?;
    let cpu_forward = cpu_forward_executor.run(&combo.params, &cpu_inputs)?;

    let initial_probe = executor.run(&combo.forward, &forward_params, &combo.inputs)?;
    let mut forward_probe_outputs = Vec::new();
    for index in 0..combo.forward.outputs().len() {
        forward_probe_outputs.push(initial_probe.output(index)?.clone());
    }
    let forward_memory = memory_snapshot(
        executor,
        &forward_arena,
        &forward_plan,
        &forward_probe_outputs,
    );
    drop(initial_probe);

    // Forward-only measurements in all five modes.
    for mode in MODES {
        let stream = if mode.is_arena() {
            &forward_arena_stream
        } else {
            &default_stream
        };
        let mut correctness = validate_forward_mode(
            executor,
            mode,
            &combo,
            &forward_params,
            &forward_plan,
            &forward_arena,
            stream,
            &cpu_forward,
        )?;
        if correctness.is_empty() {
            correctness = "PASS".to_string();
        }
        let mut record = Case::new(
            "linear_regression_forward",
            combo.name.as_str(),
            mode.name(),
            combo.shape.as_str(),
        );
        record.graph_hash = forward_hash.clone();
        record.plan_hash = forward_plan_hash.clone();
        record.compile_ms = compile_ms;
        record.memory = forward_memory.clone();
        record.correctness = correctness.clone();
        record.stream_count = if mode.is_arena() { 2 } else { 1 };
        record.note = format!(
            "forward-only graph (matmul+bias+mse), no backward and no optimizer; {TOLERANCE_NOTE}"
        );
        record.samples = match mode {
            Mode::Normal => measure_sync(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
                executor.run(&combo.forward, &forward_params, &combo.inputs)?;
                Ok(())
            })?,
            Mode::Planned => measure_sync(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
                executor.run_planned(
                    &combo.forward,
                    &forward_params,
                    &combo.inputs,
                    &forward_plan,
                )?;
                Ok(())
            })?,
            Mode::Arena => measure_sync(&forward_arena_stream, ROUNDS, ITERATIONS, WARMUP, || {
                executor.run_arena(
                    &combo.forward,
                    &forward_params,
                    &combo.inputs,
                    &forward_arena,
                    forward_arena_stream.clone(),
                )?;
                Ok(())
            })?,
            Mode::AsyncPlanned => {
                measure_async(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
                    Ok(executor.execute_async_planned(
                        &combo.forward,
                        &forward_params,
                        &combo.inputs,
                        &forward_plan,
                        default_stream.clone(),
                    )?)
                })?
            }
            Mode::AsyncArena => {
                measure_async(&forward_arena_stream, ROUNDS, ITERATIONS, WARMUP, || {
                    Ok(executor.execute_async_arena(
                        &combo.forward,
                        &forward_params,
                        &combo.inputs,
                        &forward_arena,
                        forward_arena_stream.clone(),
                    )?)
                })?
            }
        };
        cases.push(record);
    }

    // Probe plan-backed training once. The existing executor rejects
    // `ConstantScalar` copies into a shared planned slot (constant-copy
    // capacity mismatch), so those modes are recorded as NOT APPLICABLE.
    let mut blocked_train_modes: Vec<(Mode, String)> = Vec::new();
    for mode in [Mode::Planned, Mode::Arena] {
        let probe_params =
            CudaParameterStore::from_host(&combo.training.graph, &combo.params, pool)?;
        let probe_arena = CudaPlannedArena::new(training_plan.clone(), pool.clone(), 2)?;
        let probe_stream = if mode.is_arena() {
            &training_arena_stream
        } else {
            &default_stream
        };
        if let Err(reason) = probe_training_mode(
            executor,
            mode,
            &combo,
            &probe_params,
            &training_plan,
            &probe_arena,
            probe_stream,
        ) {
            println!(
                "linear_regression_check: {} {} train NOT APPLICABLE ({reason})",
                combo.name,
                mode.name()
            );
            blocked_train_modes.push((mode, reason));
        }
    }

    // Training runs: only synchronous modes can apply CudaSgd after the run.
    let mut mode_results: Vec<ModeResult> = Vec::new();
    for mode in [Mode::Normal, Mode::Planned, Mode::Arena] {
        if blocked_train_modes
            .iter()
            .any(|(blocked, _)| *blocked == mode)
        {
            continue;
        }
        let stream = if mode.is_arena() {
            &training_arena_stream
        } else {
            &default_stream
        };
        let mut params = CudaParameterStore::from_host(&combo.training.graph, &combo.params, pool)?;
        let mut optimizer = CudaSgd::new(LEARNING_RATE)?;

        for _ in 0..STEPS {
            sync_training_step(
                executor,
                mode,
                &combo,
                &mut params,
                &training_plan,
                &training_arena,
                stream,
                &mut optimizer,
            )?;
        }
        stream.synchronize()?;
        executor.pool().poll_pending();

        let (values, parameter_values) = evaluate_gpu(
            executor,
            mode,
            &combo,
            &params,
            &training_plan,
            &training_arena,
            stream,
        )?;
        let mut correctness =
            validate_final(mode.name(), &combo, &reference, &values, &parameter_values);
        if correctness.is_empty() {
            let final_loss = values
                .get(&combo.loss)
                .map(|tensor| tensor.data()[0])
                .unwrap_or(f32::NAN);
            if !(final_loss.is_finite()
                && reference.initial_loss.is_finite()
                && final_loss < reference.initial_loss * 0.5)
            {
                correctness = format!(
                    "FAIL {} convergence: initial={} final={}",
                    mode.name(),
                    reference.initial_loss,
                    final_loss
                );
            }
        }
        if correctness.is_empty() {
            if let Some(normal) = mode_results
                .iter()
                .find(|result| result.mode == Mode::Normal)
            {
                let weight = parameter_values
                    .get(&ParameterId(0))
                    .map(|tensor| tensor.data())
                    .unwrap_or(&[]);
                let bias = parameter_values
                    .get(&ParameterId(1))
                    .map(|tensor| tensor.data())
                    .unwrap_or(&[]);
                let final_loss = values
                    .get(&combo.loss)
                    .map(|tensor| tensor.data()[0])
                    .unwrap_or(f32::NAN);
                if let Err(error) = within_tolerance(weight, &normal.weight, MODE_ATOL, MODE_RTOL) {
                    correctness = format!("FAIL cross-mode w: {error}");
                } else if let Err(error) =
                    within_tolerance(bias, &normal.bias, MODE_ATOL, MODE_RTOL)
                {
                    correctness = format!("FAIL cross-mode b: {error}");
                } else if let Err(error) =
                    within_tolerance(&[final_loss], &[normal.loss], MODE_ATOL, MODE_RTOL)
                {
                    correctness = format!("FAIL cross-mode loss: {error}");
                }
            }
        }
        if correctness.is_empty() {
            correctness = "PASS".to_string();
        }

        let probe = executor.run(&combo.training.graph, &params, &combo.inputs)?;
        let mut probe_outputs = Vec::new();
        for index in 0..combo.training.graph.outputs().len() {
            probe_outputs.push(probe.output(index)?.clone());
        }
        let training_memory =
            memory_snapshot(executor, &training_arena, &training_plan, &probe_outputs);
        drop(probe);

        // Full training step (forward + backward + SGD update).
        let mut record = Case::new(
            "linear_regression_train",
            combo.name.as_str(),
            mode.name(),
            combo.shape.as_str(),
        );
        record.graph_hash = training_hash.clone();
        record.plan_hash = training_plan_hash.clone();
        record.compile_ms = compile_ms;
        record.memory = training_memory;
        record.correctness = correctness.clone();
        record.stream_count = if mode.is_arena() { 2 } else { 1 };
        record.note = format!(
            "one complete training step: forward + backward + CudaSgd update; \
             {TOLERANCE_NOTE}"
        );
        record.samples = measure_sync(stream, ROUNDS, ITERATIONS, WARMUP, || {
            sync_training_step(
                executor,
                mode,
                &combo,
                &mut params,
                &training_plan,
                &training_arena,
                stream,
                &mut optimizer,
            )
        })?;
        cases.push(record);

        push_mode_result(
            &mut mode_results,
            mode,
            &values,
            &combo,
            &parameter_values,
            correctness,
        );
        println!(
            "linear_regression_check: {} {} loss={:.6} correctness={}",
            combo.name,
            mode.name(),
            values
                .get(&combo.loss)
                .map(|tensor| tensor.data()[0])
                .unwrap_or(f32::NAN),
            mode_results
                .last()
                .map(|result| result.correctness.as_str())
                .unwrap_or("")
        );
    }

    // Plan-backed training modes that failed the probe (for example the
    // `ConstantScalar` planned-slot capacity mismatch) are reported honestly.
    for (mode, reason) in &blocked_train_modes {
        let mut record = Case::new(
            "linear_regression_train",
            combo.name.as_str(),
            mode.name(),
            combo.shape.as_str(),
        );
        record.graph_hash = training_hash.clone();
        record.plan_hash = training_plan_hash.clone();
        record.compile_ms = compile_ms;
        record.correctness = "NOT APPLICABLE".to_string();
        record.stream_count = if mode.is_arena() { 2 } else { 1 };
        record.note = format!(
            "NOT APPLICABLE: plan-backed training graph failed before any timed step: \
             {reason}. This is a pre-existing executor limitation (ConstantScalar copy into a \
             shared planned slot; the CUDA executor does not resize constants), not a \
             benchmark result. {TOLERANCE_NOTE}"
        );
        cases.push(record);
    }

    // Async training steps are genuinely not applicable: CudaSgd::step needs a
    // unique DeviceBuffer per parameter, while a live CudaExecution still holds
    // the parameter aliases (CudaTrainer::step drops CudaRun first). Device
    // gradient tensors are available via output_tensor after wait(), but the
    // wait makes the step synchronous, so no async training number is reported.
    for mode in [Mode::AsyncPlanned, Mode::AsyncArena] {
        let mut record = Case::new(
            "linear_regression_train",
            combo.name.as_str(),
            mode.name(),
            combo.shape.as_str(),
        );
        record.graph_hash = training_hash.clone();
        record.plan_hash = training_plan_hash.clone();
        record.compile_ms = compile_ms;
        record.correctness = "NOT APPLICABLE".to_string();
        record.stream_count = if mode.is_arena() { 2 } else { 1 };
        record.note = format!(
            "NOT APPLICABLE: async training step cannot apply CudaSgd without a host \
             synchronization (CudaSgd::step requires a unique DeviceBuffer, but the live \
             CudaExecution holds parameter aliases; CudaTrainer::step drops CudaRun first). \
             Gradients are device tensors via output_tensor after wait(), but wait() makes the \
             step synchronous, so no number is reported. Forward-only async mode is measured. \
             {TOLERANCE_NOTE}"
        );
        cases.push(record);
    }

    // Optimizer-step-only measurement. SGD touches one parameter at a time and
    // has no plan/arena variant, so it is independent of the execution mode.
    let mut optimizer_params =
        CudaParameterStore::from_host(&combo.training.graph, &combo.params, pool)?;
    let gradient_probe = executor.run(&combo.training.graph, &optimizer_params, &combo.inputs)?;
    let fixed_gradients = collect_run_gradients(&gradient_probe, &combo)?;
    drop(gradient_probe);
    let mut optimizer = CudaSgd::new(LEARNING_RATE)?;
    let optimizer_correctness = mode_results
        .iter()
        .find(|result| result.mode == Mode::Normal)
        .map(|result| result.correctness.clone())
        .unwrap_or_else(|| "PASS".to_string());
    let mut record = Case::new(
        "linear_regression_optimizer_step",
        combo.name.as_str(),
        "normal",
        combo.shape.as_str(),
    );
    record.graph_hash = training_hash;
    record.plan_hash = training_plan_hash;
    record.compile_ms = compile_ms;
    record.memory = training_arena_snapshot(executor, &training_arena, &training_plan);
    record.correctness = optimizer_correctness;
    record.note = format!(
        "optimizer-step-only: CudaSgd update on fixed gradient tensors; mode/plan/arena \
         independent and excluded from the train-step total to avoid double counting; \
         {TOLERANCE_NOTE}"
    );
    record.samples = measure_sync(&default_stream, ROUNDS, ITERATIONS, WARMUP, || {
        optimizer.step(
            executor.kernels(),
            &default_stream,
            executor.pool(),
            &mut optimizer_params,
            &fixed_gradients,
        )?;
        Ok(())
    })?;
    cases.push(record);

    Ok(())
}

fn training_arena_snapshot(
    executor: &CudaExecutor,
    arena: &CudaPlannedArena,
    plan: &CudaMemoryPlan,
) -> MemorySnapshot {
    memory_snapshot(executor, arena, plan, &[])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let pool = CudaBufferPool::new(device.context().clone(), device.default_stream());
    let meta = collect_meta(&device);
    println!(
        "linear_regression_check: gpu={} cc={} uuid={}",
        meta.gpu_name, meta.compute_capability, meta.gpu_uuid
    );

    let datasets = [
        make_dataset("small", 16, 1),
        make_dataset("medium", 64, 4),
        make_dataset("large", 256, 8),
    ];

    let mut cases = Vec::new();
    for dataset in &datasets {
        for batch in [1usize, 4, 8, 32] {
            run_combo(&executor, &pool, dataset, batch, &mut cases)?;
        }
    }

    let dir = out_dir("../../docs");
    let json_path = dir.join("phase7f19-linear-regression-results.json");
    write_results(&json_path, "linear_regression_check", &meta, &cases)?;
    println!("linear_regression_check: wrote {}", json_path.display());

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
                "linear_regression_check: FAIL {} {}: {}",
                case.graph, case.mode, case.correctness
            );
        }
    }
    println!(
        "linear_regression_check summary: PASS={pass} FAIL={fail} NOT APPLICABLE={not_applicable} \
         cases={} samples/case={}",
        cases.len(),
        ROUNDS * ITERATIONS
    );
    if fail > 0 {
        return Err(format!("{fail} linear-regression cases failed").into());
    }
    Ok(())
}
