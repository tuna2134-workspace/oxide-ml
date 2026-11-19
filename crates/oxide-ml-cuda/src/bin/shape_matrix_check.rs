//! Phase 7F-19 shape matrix check for static / bounded / unbounded / mixed
//! graphs.
//!
//! Op choice
//! ---------
//! The static shapes use a single elementwise ReLU. It is memory-bound and
//! `[256,256]` F32 is only 256 KiB, so small/medium/large all fit a T4 with
//! ample headroom; the matrix therefore exercises allocation paths rather than
//! GEMM tiling. Bounded and unbounded graphs use the same ReLU on `[batch,16]`
//! and `[dynamic,16]`. The mixed graph combines a static `[8,16]` input, a
//! bounded `[batch,16]` input and an unbounded `[Dynamic,16]` input, each with
//! its own ReLU output.
//!
//! `CudaExecutor::plan_bounded` rejects any graph containing `Dim::Dynamic`,
//! so the mixed plan is built through public APIs only: specialize the bounded
//! symbol to its maximum while preserving `Dim::Dynamic`, run the compiler
//! `MemoryPlanner` (dynamic values become `unplanned`), then convert with
//! `CudaMemoryPlan::from_memory_plan`. No executor / memplan / safety code is
//! modified.
//!
//! Cases that the public API cannot express are reported as
//! `NOT APPLICABLE` in the case note and carry no samples. No measurement is
//! fabricated.

#[path = "bench_common/mod.rs"]
mod bench_common;

use std::collections::HashMap;
use std::sync::Arc;

use bench_common::{
    Case, MemorySnapshot, WARMUP, collect_meta, graph_hash, is_capacity_error, measure_async,
    measure_sync, out_dir, plan_hash, within_tolerance, write_results,
};
use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, DenseTensor, Dim, Shape, SymbolId, ValueId};
use oxide_ml_cuda::{
    CudaBufferPool, CudaDevice, CudaExecution, CudaExecutor, CudaMemoryPlan, CudaParameterStore,
    CudaPlannedArena, CudaRun,
};
use oxide_ml_graph::{Graph, GraphBuilder, ValueInfo, infer_outputs};

type Inputs = HashMap<ValueId, DenseTensor<f32>>;

const BATCH_MAX: usize = 8;
const ATOL: f64 = 1e-4;
const RTOL: f64 = 3e-3;

#[derive(Clone, Copy, PartialEq, Eq)]
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

    const ALL: [Mode; 5] = [
        Mode::Normal,
        Mode::Planned,
        Mode::Arena,
        Mode::AsyncPlanned,
        Mode::AsyncArena,
    ];
}

struct Fixture {
    name: &'static str,
    graph: Graph,
    params: CudaParameterStore,
    plan: CudaMemoryPlan,
    arena: CudaPlannedArena,
    shape: String,
    plan_note: String,
    arena_applicable: bool,
    inputs: Inputs,
    expected: Vec<Vec<f32>>,
}

struct FixtureBuilder {
    name: &'static str,
    graph: Graph,
    shape: String,
    plan_note: String,
    arena_applicable: bool,
    inputs: Inputs,
    expected: Vec<Vec<f32>>,
}

fn relu(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| value.max(0.0)).collect()
}

fn input_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let raw = ((index * 37 + 11) % 41) as f32;
            raw / 7.0 - 2.5
        })
        .collect()
}

fn runtime_shapes(
    graph: &Graph,
    inputs: &Inputs,
) -> Result<HashMap<ValueId, Vec<usize>>, Box<dyn std::error::Error>> {
    let mut shapes = HashMap::new();
    for (&value, tensor) in inputs {
        shapes.insert(value, tensor.dims().to_vec());
    }
    for parameter in graph.parameters() {
        shapes.insert(
            parameter.value,
            parameter
                .shape
                .static_dims()
                .ok_or("non-static parameter shape")?,
        );
    }
    for node in graph.nodes() {
        let mut operands = Vec::with_capacity(node.inputs.len());
        for &input in &node.inputs {
            let info = graph.value(input)?;
            let dims = shapes.get(&input).ok_or("missing operand shape")?;
            operands.push(ValueInfo {
                id: info.id,
                name: info.name.clone(),
                shape: Shape::from_static(dims),
                dtype: info.dtype,
                device: info.device,
                kind: info.kind.clone(),
            });
        }
        let inferred = infer_outputs(&node.op, &node.attrs, &operands)?;
        for (output, inferred) in node.outputs.iter().zip(inferred) {
            shapes.insert(
                *output,
                inferred
                    .shape
                    .static_dims()
                    .ok_or("unresolved output shape")?,
            );
        }
    }
    Ok(shapes)
}

fn runtime_bytes(
    graph: &Graph,
    inputs: &Inputs,
) -> Result<usize, Box<dyn std::error::Error>> {
    let shapes = runtime_shapes(graph, inputs)?;
    let mut bytes = 0usize;
    for value in &graph.values {
        let dims = shapes
            .get(&value.id)
            .ok_or_else(|| format!("missing shape for value {}", value.id))?;
        bytes += dims.iter().product::<usize>() * value.dtype.size_in_bytes();
    }
    Ok(bytes)
}

fn mixed_plan(
    graph: &Graph,
    bindings: &HashMap<SymbolId, usize>,
) -> Result<CudaMemoryPlan, Box<dyn std::error::Error>> {
    let mut specialized = graph.clone();
    for value in &mut specialized.values {
        let dims = value
            .shape
            .dims()
            .iter()
            .map(|dim| match dim {
                Dim::Static(value) => Ok(Dim::Static(*value)),
                Dim::Symbol(symbol) => bindings
                    .get(symbol)
                    .copied()
                    .map(Dim::Static)
                    .ok_or_else(|| format!("unbound symbol {symbol:?}")),
                Dim::Dynamic => Ok(Dim::Dynamic),
            })
            .collect::<Result<Vec<_>, String>>()?;
        value.shape = Shape::new(dims);
    }
    specialized.validate()?;
    let memory_plan = MemoryPlanner::default().plan(&specialized)?;
    Ok(CudaMemoryPlan::from_memory_plan(&specialized, &memory_plan)?)
}

fn static_fixture(
    executor: &CudaExecutor,
    pool: &CudaBufferPool,
    name: &'static str,
    dims: [usize; 2],
) -> Result<FixtureBuilder, Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static(dims), DType::F32);
    let y = builder.relu(x)?;
    let graph = builder.finish(&[y])?;
    let host = input_values(dims[0] * dims[1]);
    let mut inputs = HashMap::new();
    inputs.insert(x, DenseTensor::from_vec(dims, host.clone())?);
    let _ = executor;
    let _ = pool;
    Ok(FixtureBuilder {
        name,
        graph,
        shape: format!("x{} relu", shape_text(&dims)),
        plan_note: "MemoryPlanner static plan".to_string(),
        arena_applicable: true,
        inputs,
        expected: vec![relu(&host)],
    })
}

fn bounded_fixture(
    pool: &CudaBufferPool,
    executor: &CudaExecutor,
    batch: usize,
) -> Result<FixtureBuilder, Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    let batch_symbol = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch_symbol), Dim::Static(16)]),
        DType::F32,
    );
    let y = builder.relu(x)?;
    let graph = builder.finish(&[y])?;
    let host = input_values(batch * 16);
    let mut inputs = HashMap::new();
    inputs.insert(x, DenseTensor::from_vec([batch, 16], host.clone())?);
    let categories = if batch == 1 {
        "min".to_string()
    } else if batch == BATCH_MAX {
        "max".to_string()
    } else if batch == 4 {
        "typical".to_string()
    } else {
        "other".to_string()
    };
    let _ = (pool, executor);
    Ok(FixtureBuilder {
        name: "dynamic_bounded",
        graph,
        shape: format!("batch={batch} ({categories}) x[{batch},16] relu"),
        plan_note: format!("plan_bounded(batch={BATCH_MAX}) capacity plan"),
        arena_applicable: true,
        inputs,
        expected: vec![relu(&host)],
    })
}

fn unbounded_fixture(batch: usize) -> Result<FixtureBuilder, Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Dynamic, Dim::Static(16)]),
        DType::F32,
    );
    let y = builder.relu(x)?;
    let graph = builder.finish(&[y])?;
    let host = input_values(batch * 16);
    let mut inputs = HashMap::new();
    inputs.insert(x, DenseTensor::from_vec([batch, 16], host.clone())?);
    Ok(FixtureBuilder {
        name: "unbounded",
        graph,
        shape: format!("dim=Dynamic batch={batch} x[{batch},16] relu"),
        plan_note:
            "MemoryPlanner: Dim::Dynamic values are unplanned; execution uses pool fallback"
                .to_string(),
        arena_applicable: false,
        inputs,
        expected: vec![relu(&host)],
    })
}

fn mixed_fixture(batch: usize) -> Result<FixtureBuilder, Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    let batch_symbol = builder.symbol("batch");
    let xs = builder.input("xs", Shape::from_static([8, 16]), DType::F32);
    let xb = builder.input(
        "xb",
        Shape::new(vec![Dim::Symbol(batch_symbol), Dim::Static(16)]),
        DType::F32,
    );
    let xu = builder.input(
        "xu",
        Shape::new(vec![Dim::Dynamic, Dim::Static(16)]),
        DType::F32,
    );
    let ys = builder.relu(xs)?;
    let yb = builder.relu(xb)?;
    let yu = builder.relu(xu)?;
    let graph = builder.finish(&[ys, yb, yu])?;
    let host_s = input_values(8 * 16);
    let host_b = input_values(batch * 16);
    let host_u = input_values(batch * 16);
    let mut inputs = HashMap::new();
    inputs.insert(xs, DenseTensor::from_vec([8, 16], host_s.clone())?);
    inputs.insert(xb, DenseTensor::from_vec([batch, 16], host_b.clone())?);
    inputs.insert(xu, DenseTensor::from_vec([batch, 16], host_u.clone())?);
    Ok(FixtureBuilder {
        name: "mixed",
        graph,
        shape: format!("static xs[8,16] bounded xb[{batch},16] unbounded xu[{batch},16] relu"),
        plan_note:
            "public-API mixed plan: bounded symbol specialized to max, Dim::Dynamic unplanned"
                .to_string(),
        arena_applicable: true,
        inputs,
        expected: vec![relu(&host_s), relu(&host_b), relu(&host_u)],
    })
}

fn shape_text(dims: &[usize]) -> String {
    format!("{:?}", dims)
}

fn prepare(
    executor: &CudaExecutor,
    pool: &CudaBufferPool,
    builder: FixtureBuilder,
    bind_bounded: bool,
) -> Result<Fixture, Box<dyn std::error::Error>> {
    let FixtureBuilder {
        name,
        graph,
        shape,
        plan_note,
        arena_applicable,
        inputs,
        expected,
    } = builder;
    let plan = if name == "mixed" {
        let mut bindings = HashMap::new();
        bindings.insert(SymbolId(0), BATCH_MAX);
        mixed_plan(&graph, &bindings)?
    } else if bind_bounded {
        let mut bindings = HashMap::new();
        bindings.insert(SymbolId(0), BATCH_MAX);
        executor.plan_bounded(&graph, &bindings)?.0
    } else {
        let memory_plan = MemoryPlanner::default().plan(&graph)?;
        CudaMemoryPlan::from_memory_plan(&graph, &memory_plan)?
    };
    let host_params = oxide_ml_core::ParameterStore::<f32>::new();
    let params = CudaParameterStore::from_host(&graph, &host_params, pool)?;
    let arena = CudaPlannedArena::new(plan.clone(), pool.clone(), 1)?;
    Ok(Fixture {
        name,
        graph,
        params,
        plan,
        arena,
        shape,
        plan_note,
        arena_applicable,
        inputs,
        expected,
    })
}

fn download_run(
    run: &CudaRun,
    stream: &Arc<cuda_core::CudaStream>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    for &output in &run.outputs {
        values.push(run.download(stream, output)?.data().to_vec());
    }
    Ok(values)
}

fn download_execution(
    execution: &mut CudaExecution,
    graph: &Graph,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let outputs = execution.wait()?;
    let mut values = Vec::new();
    for &output in graph.outputs() {
        values.push(
            outputs
                .get(&output)
                .ok_or_else(|| format!("execution missing output {output}"))?
                .data()
                .to_vec(),
        );
    }
    Ok(values)
}

fn run_once(
    executor: &CudaExecutor,
    fixture: &Fixture,
    inputs: &Inputs,
    mode: Mode,
    stream: &Arc<cuda_core::CudaStream>,
    arena_stream: &Arc<cuda_core::CudaStream>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    match mode {
        Mode::Normal => {
            let run = executor.run(&fixture.graph, &fixture.params, inputs)?;
            download_run(&run, stream)
        }
        Mode::Planned => {
            let run =
                executor.run_planned(&fixture.graph, &fixture.params, inputs, &fixture.plan)?;
            download_run(&run, stream)
        }
        Mode::Arena => {
            let run = executor.run_arena(
                &fixture.graph,
                &fixture.params,
                inputs,
                &fixture.arena,
                arena_stream.clone(),
            )?;
            download_run(&run, stream)
        }
        Mode::AsyncPlanned => {
            let mut execution = executor.execute_async_planned(
                &fixture.graph,
                &fixture.params,
                inputs,
                &fixture.plan,
                stream.clone(),
            )?;
            download_execution(&mut execution, &fixture.graph)
        }
        Mode::AsyncArena => {
            let mut execution = executor.execute_async_arena(
                &fixture.graph,
                &fixture.params,
                inputs,
                &fixture.arena,
                arena_stream.clone(),
            )?;
            download_execution(&mut execution, &fixture.graph)
        }
    }
}

fn measure_mode(
    executor: &CudaExecutor,
    fixture: &Fixture,
    mode: Mode,
    stream: &Arc<cuda_core::CudaStream>,
    arena_stream: &Arc<cuda_core::CudaStream>,
) -> Result<Vec<bench_common::Sample>, Box<dyn std::error::Error>> {
    let graph = &fixture.graph;
    let params = &fixture.params;
    let inputs = &fixture.inputs;
    let plan = &fixture.plan;
    let arena = &fixture.arena;
    match mode {
        Mode::Normal => measure_sync(
            stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                executor.run(graph, params, inputs)?;
                Ok(())
            },
        ),
        Mode::Planned => measure_sync(
            stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                executor.run_planned(graph, params, inputs, plan)?;
                Ok(())
            },
        ),
        Mode::Arena => measure_sync(
            stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                executor.run_arena(graph, params, inputs, arena, arena_stream.clone())?;
                Ok(())
            },
        ),
        Mode::AsyncPlanned => measure_async(
            stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || Ok(executor.execute_async_planned(graph, params, inputs, plan, stream.clone())?),
        ),
        Mode::AsyncArena => measure_async(
            stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                Ok(executor.execute_async_arena(
                    graph,
                    params,
                    inputs,
                    arena,
                    arena_stream.clone(),
                )?)
            },
        ),
    }
}

fn snapshot(
    executor: &CudaExecutor,
    fixture: &Fixture,
    logical_bytes: usize,
) -> MemorySnapshot {
    let pool = executor.pool_stats();
    let arena = fixture.arena.stats();
    let events = executor.event_stats();
    MemorySnapshot {
        logical_bytes: Some(logical_bytes),
        planned_logical_bytes: Some(fixture.plan.logical_bytes),
        runtime_required_bytes: Some(logical_bytes),
        pool_reserved_bytes: Some(pool.reserved_bytes),
        pool_active_bytes: Some(pool.current_bytes),
        pool_pending_release_bytes: Some(pool.pending_release_bytes),
        pool_quarantined_bytes: Some(pool.quarantined_bytes),
        pool_allocation_count: Some(pool.allocations),
        pool_reuse_count: Some(pool.reuses),
        arena_capacity_bytes: Some(arena.physical_reserved_bytes as usize),
        arena_used_bytes: Some((arena.active_slot_set_count * arena.logical_bytes) as usize),
        arena_physical_reserved_bytes: Some(arena.physical_reserved_bytes as usize),
        arena_slot_allocation_count: Some(arena.allocation_count),
        arena_slot_reuse_count: Some(arena.slot_set_reuse_count),
        arena_active_lease_count: Some(arena.active_slot_set_count),
        event_allocation_count: Some(events.allocation_count as u64),
        event_reuse_count: Some(events.reuse_count as u64),
        event_pending_count: Some(events.pending_count as u64),
        event_quarantine_count: Some(events.quarantine_count as u64),
        access_reads: None,
        access_writes: None,
        access_read_write: None,
        access_same_stream_elided: None,
        access_cross_stream_waits: None,
        access_read_read_elision: None,
        driver_used_bytes: None,
    }
}

fn push_not_applicable(cases: &mut Vec<Case>, fixture: &Fixture, mode: Mode, reason: &str) {
    let mut record = Case::new(
        "shape_matrix",
        fixture.name,
        mode.name(),
        fixture.shape.clone(),
    );
    record.graph_hash = graph_hash(&fixture.graph);
    record.plan_hash = plan_hash(&fixture.plan);
    record.correctness = "NOT APPLICABLE".to_string();
    record.note = reason.to_string();
    cases.push(record);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let pool: CudaBufferPool = executor.pool().clone();
    let default_stream = device.default_stream();
    let arena_stream = device.new_stream()?;
    let meta = collect_meta(&device);
    println!(
        "shape_matrix_check: gpu={} cc={}",
        meta.gpu_name, meta.compute_capability
    );

    let mut fixtures = Vec::new();
    for (name, dims) in [
        ("static_small", [8usize, 16usize]),
        ("static_medium", [64usize, 64usize]),
        ("static_large", [256usize, 256usize]),
    ] {
        let builder = static_fixture(&executor, &pool, name, dims)?;
        fixtures.push(prepare(&executor, &pool, builder, false)?);
    }
    for batch in [1usize, 4, BATCH_MAX] {
        let builder = bounded_fixture(&pool, &executor, batch)?;
        fixtures.push(prepare(&executor, &pool, builder, true)?);
    }
    let builder = unbounded_fixture(7)?;
    fixtures.push(prepare(&executor, &pool, builder, false)?);
    let builder = mixed_fixture(BATCH_MAX)?;
    fixtures.push(prepare(&executor, &pool, builder, true)?);

    let mut cases = Vec::new();
    let mut failures = 0usize;
    for fixture in &fixtures {
        let bytes = runtime_bytes(&fixture.graph, &fixture.inputs)?;
        let baseline = run_once(
            &executor,
            fixture,
            &fixture.inputs,
            Mode::Normal,
            &default_stream,
            &arena_stream,
        )?;
        for (index, expected) in fixture.expected.iter().enumerate() {
            if let Err(error) = within_tolerance(&baseline[index], expected, ATOL, RTOL) {
                failures += 1;
                println!(
                    "shape_matrix_check: FAIL {} normal output {index}: {error}",
                    fixture.name
                );
            }
        }
        for mode in Mode::ALL {
            if !fixture.arena_applicable && matches!(mode, Mode::Arena | Mode::AsyncArena) {
                push_not_applicable(
                    &mut cases,
                    fixture,
                    mode,
                    "NOT APPLICABLE: the plan has zero slots (Dim::Dynamic values are \
                     unplanned), so arena storage is empty and this mode would measure only \
                     the pool fallback",
                );
                continue;
            }
            let outputs = match run_once(&executor, fixture, &fixture.inputs, mode, &default_stream, &arena_stream) {
                Ok(outputs) => outputs,
                Err(error) => {
                    failures += 1;
                    println!(
                        "shape_matrix_check: FAIL {} {}: {error}",
                        fixture.name,
                        mode.name()
                    );
                    let mut record = Case::new(
                        "shape_matrix",
                        fixture.name,
                        mode.name(),
                        fixture.shape.clone(),
                    );
                    record.correctness = format!("FAIL: {error}");
                    record.note = fixture.plan_note.clone();
                    cases.push(record);
                    continue;
                }
            };
            let mut correctness = String::from("PASS");
            for (index, expected) in fixture.expected.iter().enumerate() {
                if let Err(error) = within_tolerance(&outputs[index], expected, ATOL, RTOL) {
                    correctness = format!("FAIL vs reference output {index}: {error}");
                    break;
                }
                if let Err(error) =
                    within_tolerance(&outputs[index], &baseline[index], 1e-6, 1e-5)
                {
                    correctness = format!("FAIL vs normal mode output {index}: {error}");
                    break;
                }
            }
            if correctness != "PASS" {
                failures += 1;
                println!(
                    "shape_matrix_check: FAIL {} {} {correctness}",
                    fixture.name,
                    mode.name()
                );
            }
            let mut record = Case::new(
                "shape_matrix",
                fixture.name,
                mode.name(),
                fixture.shape.clone(),
            );
            record.graph_hash = graph_hash(&fixture.graph);
            record.plan_hash = if mode == Mode::Normal {
                "unavailable".to_string()
            } else {
                plan_hash(&fixture.plan)
            };
            record.correctness = correctness;
            record.memory = snapshot(&executor, fixture, bytes);
            record.note = match mode {
                Mode::Normal => "sync API; ordinary pool path",
                Mode::Planned => fixture.plan_note.as_str(),
                Mode::Arena => "sync API; persistent arena slot set",
                Mode::AsyncPlanned => "submit does not wait; plan slots + pool fallback",
                Mode::AsyncArena => "submit does not wait; arena lease released on completion",
            }
            .to_string();
            record.samples = measure_mode(
                &executor,
                fixture,
                mode,
                &default_stream,
                &arena_stream,
            )?;
            cases.push(record);
        }
    }

    // ---- Capacity boundary (bounded, max = 8) -----------------------------
    let bounded = fixtures
        .iter()
        .find(|fixture| fixture.name == "dynamic_bounded" && fixture.shape.contains("max"))
        .ok_or("missing bounded max fixture")?;
    let events_before = executor.event_stats().allocation_count;
    let mut inputs9 = HashMap::new();
    for (&value, tensor) in &bounded.inputs {
        let dims = tensor.dims().to_vec();
        let mut next = dims.clone();
        next[0] = BATCH_MAX + 1;
        inputs9.insert(
            value,
            DenseTensor::from_vec(next, input_values((BATCH_MAX + 1) * 16))?,
        );
    }
    let overflow = executor.run_planned(&bounded.graph, &bounded.params, &inputs9, &bounded.plan);
    let events_after = executor.event_stats().allocation_count;
    let rejected = match &overflow {
        Err(error) => is_capacity_error(error),
        Ok(_) => false,
    };
    let no_launch = events_before == events_after;
    let mut boundary = Case::new(
        "shape_matrix",
        "capacity_boundary",
        "planned",
        format!("batch={} (max+1)", BATCH_MAX + 1),
    );
    boundary.graph_hash = graph_hash(&bounded.graph);
    boundary.plan_hash = plan_hash(&bounded.plan);
    boundary.correctness = if rejected && no_launch {
        "PASS".to_string()
    } else if rejected && !no_launch {
        "FAIL: Capacity error but event allocations changed (possible launch)".to_string()
    } else {
        "FAIL: API accepted batch 9 above capacity 8".to_string()
    };
    boundary.note = format!(
        "capacity overflow: error={}, event_allocations {} -> {}",
        overflow
            .as_ref()
            .err()
            .map(|error| error.to_string())
            .unwrap_or_else(|| "accepted".to_string()),
        events_before,
        events_after
    );
    if !(rejected && no_launch) {
        failures += 1;
    }
    println!(
        "shape_matrix_check: {} capacity max+1 rejected before launch ({})",
        if rejected && no_launch { "PASS" } else { "FAIL" },
        boundary.note
    );
    cases.push(boundary);

    // ---- Reuse ------------------------------------------------------------
    let repeated = fixtures
        .iter()
        .find(|fixture| fixture.name == "dynamic_bounded" && fixture.shape.contains("typical"))
        .ok_or("missing typical bounded fixture")?;
    let before = repeated.arena.stats();
    let first = run_once(
        &executor,
        repeated,
        &repeated.inputs,
        Mode::Arena,
        &default_stream,
        &arena_stream,
    )?;
    let second = run_once(
        &executor,
        repeated,
        &repeated.inputs,
        Mode::Arena,
        &default_stream,
        &arena_stream,
    )?;
    let after = repeated.arena.stats();
    let outputs_repeat_ok = first.iter().zip(&second).all(|(left, right)| {
        within_tolerance(left, right, 1e-6, 1e-5).is_ok()
    });
    let same_shape_ok = outputs_repeat_ok
        && after.slot_set_reuse_count > before.slot_set_reuse_count
        && after.allocation_count == before.allocation_count;
    let mut reuse_case = Case::new(
        "shape_matrix",
        "reuse_same_shape",
        "reuse",
        "arena batch=4 repeated twice",
    );
    reuse_case.correctness = if same_shape_ok { "PASS".to_string() } else { "FAIL".to_string() };
    reuse_case.note = format!(
        "arena slot_set_reuse {} -> {}, slot allocations {} -> {}",
        before.slot_set_reuse_count,
        after.slot_set_reuse_count,
        before.allocation_count,
        after.allocation_count
    );
    if !same_shape_ok {
        failures += 1;
    }
    cases.push(reuse_case);

    let bounded8 = fixtures
        .iter()
        .find(|fixture| fixture.name == "dynamic_bounded" && fixture.shape.contains("max"))
        .ok_or("missing bounded batch 8 fixture")?;
    let mut sequence_ok = true;
    for batch in [1usize, 4, BATCH_MAX, 1] {
        let host = input_values(batch * 16);
        let mut inputs = HashMap::new();
        inputs.insert(
            bounded8.graph.inputs()[0],
            DenseTensor::from_vec([batch, 16], host.clone())?,
        );
        let outputs = run_once(
            &executor,
            bounded8,
            &inputs,
            Mode::Arena,
            &default_stream,
            &arena_stream,
        )?;
        if within_tolerance(&outputs[0], &relu(&host), ATOL, RTOL).is_err() {
            sequence_ok = false;
        }
    }
    let mut sequence_case = Case::new(
        "shape_matrix",
        "reuse_batch_sequence",
        "reuse",
        "arena 1->4->8->1",
    );
    sequence_case.correctness = if sequence_ok { "PASS".to_string() } else { "FAIL".to_string() };
    sequence_case.note = "bounded arena sequence 1 -> 4 -> 8 -> 1; outputs matched reference"
        .to_string();
    if !sequence_ok {
        failures += 1;
    }
    cases.push(sequence_case);

    let unbounded = fixtures
        .iter()
        .find(|fixture| fixture.name == "unbounded")
        .ok_or("missing unbounded fixture")?;
    let pool_before = executor.pool_stats();
    let bounded_then_unbounded = run_once(
        &executor,
        repeated,
        &repeated.inputs,
        Mode::Planned,
        &default_stream,
        &arena_stream,
    )
    .and_then(|_| {
        run_once(
            &executor,
            unbounded,
            &unbounded.inputs,
            Mode::Planned,
            &default_stream,
            &arena_stream,
        )
    });
    let unbounded_then_bounded = run_once(
        &executor,
        unbounded,
        &unbounded.inputs,
        Mode::Planned,
        &default_stream,
        &arena_stream,
    )
    .and_then(|_| {
        run_once(
            &executor,
            repeated,
            &repeated.inputs,
            Mode::Planned,
            &default_stream,
            &arena_stream,
        )
    });
    let pool_after = executor.pool_stats();
    let transitions_ok = bounded_then_unbounded.is_ok()
        && unbounded_then_bounded.is_ok()
        && pool_after.reuses >= pool_before.reuses;
    let mut transition_case = Case::new(
        "shape_matrix",
        "reuse_bounded_unbounded",
        "reuse",
        "planned bounded->unbounded and unbounded->bounded",
    );
    transition_case.correctness = if transitions_ok { "PASS".to_string() } else { "FAIL".to_string() };
    transition_case.note = format!(
        "pool reuses {} -> {}; bounded->unbounded={} unbounded->bounded={}",
        pool_before.reuses,
        pool_after.reuses,
        bounded_then_unbounded.is_ok(),
        unbounded_then_bounded.is_ok()
    );
    if !transitions_ok {
        failures += 1;
    }
    cases.push(transition_case);

    println!(
        "shape_matrix_check: cases={} failures={failures}",
        cases.len()
    );
    let dir = out_dir("../../docs");
    let json_path = dir.join("phase7f19-shape-matrix-results.json");
    write_results(&json_path, "shape_matrix_check", &meta, &cases)?;
    println!("shape_matrix_check: wrote {}", json_path.display());
    if failures > 0 {
        return Err(format!("{failures} shape matrix check(s) failed").into());
    }
    println!("shape_matrix_check: PASS");
    Ok(())
}
