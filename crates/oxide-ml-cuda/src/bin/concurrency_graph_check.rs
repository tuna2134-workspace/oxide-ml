//! Phase 7F-19 concurrency matrix over actual graph execution.
//!
//! Each case submits `stream_count` independent graph executions. Serial and
//! parallel GPU spans are measured with per-stream CudaEvents; the batch span
//! is the distance from the earliest start to the latest end and is never the
//! sum of per-stream times. Host submit time excludes event waits.
//!
//! `SharedWrite` is NOT APPLICABLE: graph ops are functional (outputs are
//! freshly allocated) and `CudaPlannedArena` leases a slot set exclusively, so
//! two concurrent executions cannot write the same slots. No raw kernels are
//! used, so the pattern is recorded without samples.

#[path = "bench_common/mod.rs"]
mod bench_common;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use bench_common::{
    Case, ITERATIONS, MemorySnapshot, ROUNDS, Sample, Stats, WARMUP, collect_meta, graph_hash,
    out_dir, plan_hash, within_tolerance, write_results,
};
use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{
    DType, DenseTensor, Dim, ParameterId, ParameterStore, Shape, SymbolId, ValueId,
};
use oxide_ml_cuda::{
    ArenaStats, CudaDevice, CudaEvent, CudaExecution, CudaExecutor, CudaMemoryPlan,
    CudaParameterStore, CudaPlannedArena,
};
use oxide_ml_execution_safety::AccessStats;
use oxide_ml_graph::{Graph, GraphBuilder};

type Inputs = HashMap<ValueId, DenseTensor<f32>>;

const BENCHMARK: &str = "concurrency_graph_check";
const STREAM_COUNTS: [usize; 3] = [1, 2, 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    Independent,
    SharedRead,
    SharedParameters,
    SharedArena,
    PoolArenaMixed,
}

impl Pattern {
    fn label(self) -> &'static str {
        match self {
            Pattern::Independent => "independent_buffers",
            Pattern::SharedRead => "shared_read",
            Pattern::SharedParameters => "shared_parameters",
            Pattern::SharedArena => "shared_arena",
            Pattern::PoolArenaMixed => "pool_arena_mixed",
        }
    }

    fn uses_arena(self) -> bool {
        matches!(self, Pattern::SharedArena | Pattern::PoolArenaMixed)
    }
}

enum PlanKind {
    Static,
    Bounded(HashMap<SymbolId, usize>),
    MixedFallback,
}

struct GraphCase {
    name: &'static str,
    graph: Graph,
    params: ParameterStore<f32>,
    inputs: Inputs,
    shape: String,
    plan_kind: PlanKind,
    /// Symbol used to prove `plan_bounded` rejects a graph with `Dim::Dynamic`.
    probe_symbol: Option<SymbolId>,
    compile_ms: f64,
}

enum Params {
    Shared(CudaParameterStore),
    PerStream(Vec<CudaParameterStore>),
}

impl Params {
    fn get(&self, index: usize) -> &CudaParameterStore {
        match self {
            Params::Shared(store) => store,
            Params::PerStream(stores) => &stores[index],
        }
    }

    fn stores(&self) -> Vec<&CudaParameterStore> {
        match self {
            Params::Shared(store) => vec![store],
            Params::PerStream(stores) => stores.iter().collect(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn graph_case(
    name: &'static str,
    graph: Graph,
    params: ParameterStore<f32>,
    inputs: Inputs,
    shape: &str,
    plan_kind: PlanKind,
    probe_symbol: Option<SymbolId>,
    begin: Instant,
) -> GraphCase {
    GraphCase {
        name,
        graph,
        params,
        inputs,
        shape: shape.to_string(),
        plan_kind,
        probe_symbol,
        compile_ms: begin.elapsed().as_secs_f64() * 1000.0,
    }
}

fn add_param(
    params: &mut ParameterStore<f32>,
    id: u32,
    name: &str,
    dims: &[usize],
    data: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    params.add(
        ParameterId(id),
        name,
        DenseTensor::from_vec(dims, data)?,
        true,
    )?;
    Ok(())
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
    add_param(
        &mut params,
        0,
        "w1",
        &[3, 4],
        vec![
            0.1, 0.2, -0.3, 0.4, 0.2, -0.5, 0.3, 0.1, -0.2, 0.4, 0.1, 0.2,
        ],
    )?;
    add_param(&mut params, 1, "b1", &[4], vec![0.1, -0.2, 0.0, 0.3])?;
    add_param(&mut params, 2, "w2", &[4, 1], vec![0.5, -0.4, 0.2, 0.1])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 3], vec![1.0, 2.0, -1.0, 0.5, -1.0, 2.0])?,
    );
    Ok(graph_case(
        "small_mlp",
        graph,
        params,
        inputs,
        "x[2,3] w1[3,4] w2[4,1]",
        PlanKind::Static,
        None,
        begin,
    ))
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
    add_param(&mut params, 0, "w1", &[128, 256], vec![0.01; 128 * 256])?;
    add_param(&mut params, 1, "b1", &[256], vec![0.0; 256])?;
    add_param(&mut params, 2, "w2", &[256, 64], vec![0.01; 256 * 64])?;
    add_param(&mut params, 3, "b2", &[64], vec![0.0; 64])?;
    add_param(&mut params, 4, "w3", &[64, 1], vec![0.01; 64])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([16, 128], vec![0.5; 16 * 128])?,
    );
    Ok(graph_case(
        "large_mlp",
        graph,
        params,
        inputs,
        "x[16,128] 128->256->64->1",
        PlanKind::Static,
        None,
        begin,
    ))
}

fn build_matmul() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([64, 64]), DType::F32);
    let w = builder.try_parameter("w", Shape::from_static([64, 64]), DType::F32, true)?;
    let y = builder.matmul(x, w)?;
    let graph = builder.finish(&[y])?;
    let mut params = ParameterStore::<f32>::new();
    add_param(&mut params, 0, "w", &[64, 64], vec![0.01; 64 * 64])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([64, 64], vec![1.0; 64 * 64])?,
    );
    Ok(graph_case(
        "matmul",
        graph,
        params,
        inputs,
        "x[64,64] w[64,64]",
        PlanKind::Static,
        None,
        begin,
    ))
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
        DenseTensor::from_vec([4096], vec![-1.0; 4096])?,
    );
    Ok(graph_case(
        "activation",
        graph,
        ParameterStore::new(),
        inputs,
        "x[4096] relu",
        PlanKind::Static,
        None,
        begin,
    ))
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
        DenseTensor::from_vec([64, 64], vec![1.0; 64 * 64])?,
    );
    Ok(graph_case(
        "reduction",
        graph,
        ParameterStore::new(),
        inputs,
        "x[64,64] reduce_sum(axis=1)",
        PlanKind::Static,
        None,
        begin,
    ))
}

fn build_dynamic_batch() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(3)]),
        DType::F32,
    );
    let w = builder.try_parameter("w", Shape::from_static([3, 1]), DType::F32, true)?;
    let y = builder.matmul(x, w)?;
    let graph = builder.finish(&[y])?;
    let mut params = ParameterStore::<f32>::new();
    add_param(&mut params, 0, "w", &[3, 1], vec![0.5, -0.25, 0.1])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([4, 3], vec![1.0; 12])?,
    );
    let mut max_bindings = HashMap::new();
    max_bindings.insert(batch, 8usize);
    Ok(graph_case(
        "dynamic_batch",
        graph,
        params,
        inputs,
        "batch=4 (max 8) x[4,3] w[3,1]",
        PlanKind::Bounded(max_bindings),
        None,
        begin,
    ))
}

fn build_mixed() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    // Static branch: planned into the arena.
    let xs = builder.input("xs", Shape::from_static([2, 3]), DType::F32);
    let ws = builder.try_parameter("ws", Shape::from_static([3, 1]), DType::F32, true)?;
    let ys = builder.matmul(xs, ws)?;
    // Bounded symbol branch: bounded plan capacity, pool fallback in the mixed plan.
    let batch = builder.symbol("batch");
    let xb = builder.input(
        "xb",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(2)]),
        DType::F32,
    );
    let wb = builder.try_parameter("wb", Shape::from_static([2, 1]), DType::F32, true)?;
    let yb = builder.matmul(xb, wb)?;
    // Unbounded branch: always pool fallback.
    let xu = builder.input(
        "xu",
        Shape::new(vec![Dim::Dynamic, Dim::Static(2)]),
        DType::F32,
    );
    let wu = builder.try_parameter("wu", Shape::from_static([2, 1]), DType::F32, true)?;
    let yu = builder.matmul(xu, wu)?;
    let graph = builder.finish(&[ys, yb, yu])?;
    let mut params = ParameterStore::<f32>::new();
    add_param(&mut params, 0, "ws", &[3, 1], vec![0.5, -0.25, 0.1])?;
    add_param(&mut params, 1, "wb", &[2, 1], vec![1.0, -1.0])?;
    add_param(&mut params, 2, "wu", &[2, 1], vec![0.25, 0.75])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 3], vec![1.0; 6])?,
    );
    inputs.insert(
        graph.inputs()[1],
        DenseTensor::from_vec([4, 2], vec![0.5; 8])?,
    );
    inputs.insert(
        graph.inputs()[2],
        DenseTensor::from_vec([5, 2], vec![0.5; 10])?,
    );
    Ok(graph_case(
        "mixed",
        graph,
        params,
        inputs,
        "static[2,3]+symbol[4,2]+dynamic[5,2]",
        PlanKind::MixedFallback,
        Some(batch),
        begin,
    ))
}

#[allow(dead_code)]
fn build_backward() -> Result<GraphCase, Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 1]), DType::F32);
    let w1 = builder.try_parameter("w1", Shape::from_static([3, 4]), DType::F32, true)?;
    let b1 = builder.try_parameter("b1", Shape::from_static([4]), DType::F32, true)?;
    let w2 = builder.try_parameter("w2", Shape::from_static([4, 1]), DType::F32, true)?;
    let h = builder.matmul(x, w1)?;
    let h = builder.add(h, b1)?;
    let h = builder.sigmoid(h)?;
    let prediction = builder.matmul(h, w2)?;
    let loss = builder.mse_loss(prediction, y, oxide_ml_graph::Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;
    let training = oxide_ml_autograd::build_training_graph(&forward, loss)?;
    let mut params = ParameterStore::<f32>::new();
    add_param(&mut params, 0, "w1", &[3, 4], vec![0.2; 12])?;
    add_param(&mut params, 1, "b1", &[4], vec![0.0; 4])?;
    add_param(&mut params, 2, "w2", &[4, 1], vec![0.2; 4])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        training.graph.inputs()[0],
        DenseTensor::from_vec([2, 3], vec![1.0, 2.0, -1.0, 0.5, -1.0, 2.0])?,
    );
    inputs.insert(
        training.graph.inputs()[1],
        DenseTensor::from_vec([2, 1], vec![1.0, 0.0])?,
    );
    Ok(graph_case(
        "backward_mlp",
        training.graph,
        params,
        inputs,
        "x[2,3] y[2,1] loss+grads",
        PlanKind::Static,
        None,
        begin,
    ))
}

fn build_plan(
    executor: &CudaExecutor,
    case: &GraphCase,
) -> Result<CudaMemoryPlan, Box<dyn std::error::Error>> {
    match &case.plan_kind {
        PlanKind::Bounded(bindings) => Ok(executor.plan_bounded(&case.graph, bindings)?.0),
        PlanKind::Static | PlanKind::MixedFallback => {
            let memory = MemoryPlanner::default().plan(&case.graph)?;
            Ok(CudaMemoryPlan::from_memory_plan(&case.graph, &memory)?)
        }
    }
}

fn patterns_for(case: &GraphCase) -> Vec<Pattern> {
    let mut patterns = vec![
        Pattern::Independent,
        Pattern::SharedRead,
        Pattern::SharedParameters,
        Pattern::SharedArena,
    ];
    if case.name == "mixed" {
        patterns.push(Pattern::PoolArenaMixed);
    }
    patterns
}

fn build_params(
    executor: &CudaExecutor,
    case: &GraphCase,
    pattern: Pattern,
    count: usize,
) -> Result<Params, Box<dyn std::error::Error>> {
    if pattern == Pattern::SharedParameters {
        return Ok(Params::Shared(CudaParameterStore::from_host(
            &case.graph,
            &case.params,
            executor.pool(),
        )?));
    }
    let mut stores = Vec::with_capacity(count);
    for _ in 0..count {
        stores.push(CudaParameterStore::from_host(
            &case.graph,
            &case.params,
            executor.pool(),
        )?);
    }
    Ok(Params::PerStream(stores))
}

fn add_access(total: &mut AccessStats, stats: &AccessStats) {
    total.event_waits = total.event_waits.saturating_add(stats.event_waits);
    total.same_stream_elided_waits = total
        .same_stream_elided_waits
        .saturating_add(stats.same_stream_elided_waits);
    total.cross_stream_waits = total
        .cross_stream_waits
        .saturating_add(stats.cross_stream_waits);
    total.read_read_concurrent = total
        .read_read_concurrent
        .saturating_add(stats.read_read_concurrent);
    total.pending_readers = total.pending_readers.max(stats.pending_readers);
    total.pending_writer = total.pending_writer.max(stats.pending_writer);
}

fn params_access(params: &Params) -> AccessStats {
    let mut total = AccessStats::default();
    for store in params.stores() {
        for (_, tensor) in store.iter() {
            add_access(&mut total, &tensor.access_stats());
        }
    }
    total
}

fn total_overlap(intervals: &[(f32, f32)]) -> f32 {
    let mut events = Vec::new();
    for &(start, end) in intervals {
        events.push((start, 1i32));
        events.push((end, -1i32));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let mut active = 0;
    let mut previous = 0.0;
    let mut overlap = 0.0;
    for (time, delta) in events {
        if active >= 2 {
            overlap += time - previous;
        }
        active += delta;
        previous = time;
    }
    overlap
}

#[derive(Default)]
struct InFlight {
    pending_release_bytes: usize,
    pending_release_count: usize,
    event_allocations: usize,
    event_reuses: usize,
    event_pending: usize,
    arena_slot_reuse: u64,
    arena_pending_sets: u64,
    arena_active_leases: u64,
}

impl InFlight {
    fn capture(executor: &CudaExecutor, arena: Option<&CudaPlannedArena>) -> Self {
        let pool = executor.pool_stats();
        let events = executor.event_stats();
        let arena_stats = arena.map(|arena| arena.stats());
        InFlight {
            pending_release_bytes: pool.pending_release_bytes,
            pending_release_count: pool.pending_release_count,
            event_allocations: events.allocation_count,
            event_reuses: events.reuse_count,
            event_pending: events.pending_count,
            arena_slot_reuse: arena_stats
                .map(|stats| stats.slot_set_reuse_count)
                .unwrap_or(0),
            arena_pending_sets: arena_stats
                .map(|stats| stats.pending_slot_set_count)
                .unwrap_or(0),
            arena_active_leases: arena_stats
                .map(|stats| stats.active_slot_set_count)
                .unwrap_or(0),
        }
    }
}

struct Measurement {
    samples: Vec<Sample>,
    per_stream: Vec<Stats>,
    overlap: Stats,
    span: Stats,
    access: AccessStats,
    in_flight: InFlight,
    arena: Option<ArenaStats>,
}

fn measure_batch<F>(
    executor: &CudaExecutor,
    streams: &[Arc<cuda_core::CudaStream>],
    arena: Option<&CudaPlannedArena>,
    output_count: usize,
    serial: bool,
    mut submit: F,
) -> Result<Measurement, Box<dyn std::error::Error>>
where
    F: FnMut(usize) -> oxide_ml_cuda::Result<CudaExecution>,
{
    let count = streams.len();
    let context = streams[0].context();
    let starts = (0..count)
        .map(|_| CudaEvent::new(context))
        .collect::<oxide_ml_cuda::Result<Vec<_>>>()?;
    let ends = (0..count)
        .map(|_| CudaEvent::new(context))
        .collect::<oxide_ml_cuda::Result<Vec<_>>>()?;

    for _ in 0..WARMUP {
        let mut executions = Vec::with_capacity(count);
        for index in 0..count {
            if serial && index > 0 {
                streams[index - 1].synchronize()?;
            }
            executions.push(submit(index)?);
        }
        for execution in &mut executions {
            execution.wait()?;
        }
    }

    let mut samples = Vec::with_capacity(ROUNDS * ITERATIONS);
    let mut per_stream_values = vec![Vec::<f64>::new(); count];
    let mut overlap_values = Vec::with_capacity(ROUNDS * ITERATIONS);
    let mut access = AccessStats::default();
    let mut in_flight = InFlight::default();

    for round in 0..ROUNDS {
        for iteration in 0..ITERATIONS {
            let e2e_begin = Instant::now();
            let mut submit_seconds = 0.0f64;
            let mut executions = Vec::with_capacity(count);
            if serial {
                for index in 0..count {
                    if index > 0 {
                        ends[index - 1].wait_on(&streams[index])?;
                    }
                    starts[index].record(&streams[index])?;
                    let begin = Instant::now();
                    let execution = submit(index)?;
                    ends[index].record(&streams[index])?;
                    submit_seconds += begin.elapsed().as_secs_f64();
                    executions.push(execution);
                }
            } else {
                for index in 0..count {
                    starts[index].record(&streams[index])?;
                }
                let begin = Instant::now();
                for index in 0..count {
                    let execution = submit(index)?;
                    ends[index].record(&streams[index])?;
                    executions.push(execution);
                }
                submit_seconds = begin.elapsed().as_secs_f64();
            }
            // In-flight accounting is sampled after submit and before any wait.
            in_flight = InFlight::capture(executor, arena);
            let wait_begin = Instant::now();
            for end in ends.iter().take(count) {
                end.synchronize()?;
            }
            let wait_ms = wait_begin.elapsed().as_secs_f64() * 1000.0;
            let e2e_ms = e2e_begin.elapsed().as_secs_f64() * 1000.0;

            // AccessGuard counters are read from live output tensors once at the
            // end of the batch (buffer stats are cumulative per generation).
            if round + 1 == ROUNDS && iteration + 1 == ITERATIONS {
                for execution in &executions {
                    for output in 0..output_count {
                        if let Ok(tensor) = execution.output_tensor(output) {
                            add_access(&mut access, &tensor.access_stats());
                        }
                    }
                }
            }

            let origin = &starts[0];
            let mut intervals = Vec::with_capacity(count);
            for index in 0..count {
                let start = f64::from(origin.elapsed_ms(&starts[index])?);
                let end = f64::from(origin.elapsed_ms(&ends[index])?);
                per_stream_values[index].push(end - start);
                intervals.push((start as f32, end as f32));
            }
            let latest_end = intervals
                .iter()
                .map(|(_, end)| f64::from(*end))
                .fold(f64::NEG_INFINITY, f64::max);
            let earliest_start = intervals
                .iter()
                .map(|(start, _)| f64::from(*start))
                .fold(f64::INFINITY, f64::min);
            let span = latest_end - earliest_start;
            overlap_values.push(f64::from(total_overlap(&intervals)));
            samples.push(Sample {
                round,
                iteration,
                gpu_ms: span,
                host_submit_ms: submit_seconds * 1000.0,
                wait_ms,
                d2h_ms: 0.0,
                e2e_ms,
                host_call_includes_wait: false,
            });
            drop(executions);
        }
    }

    let span_stats = Stats::new(
        &samples
            .iter()
            .map(|sample| sample.gpu_ms)
            .collect::<Vec<_>>(),
    );
    Ok(Measurement {
        samples,
        per_stream: per_stream_stats(&per_stream_values),
        overlap: Stats::new(&overlap_values),
        span: span_stats,
        access,
        in_flight,
        arena: arena.map(|arena| arena.stats()),
    })
}

fn per_stream_stats(values: &[Vec<f64>]) -> Vec<Stats> {
    values.iter().map(|stream| Stats::new(stream)).collect()
}

fn memory_snapshot(
    executor: &CudaExecutor,
    plan: &CudaMemoryPlan,
    arena: Option<&ArenaStats>,
    access: &AccessStats,
) -> MemorySnapshot {
    let pool = executor.pool_stats();
    let events = executor.event_stats();
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
        arena_capacity_bytes: arena.map(|stats| stats.physical_reserved_bytes as usize),
        arena_used_bytes: arena
            .map(|stats| (stats.active_slot_set_count * stats.logical_bytes) as usize),
        arena_physical_reserved_bytes: arena.map(|stats| stats.physical_reserved_bytes as usize),
        arena_slot_allocation_count: arena.map(|stats| stats.allocation_count),
        arena_slot_reuse_count: arena.map(|stats| stats.slot_set_reuse_count),
        arena_active_lease_count: arena.map(|stats| stats.active_slot_set_count),
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

fn describe(
    measurement: &Measurement,
    serial: bool,
    serial_mean: Option<f64>,
    speedup: Option<f64>,
    case: &GraphCase,
) -> String {
    let per_stream = measurement
        .per_stream
        .iter()
        .map(|stats| stats.mean)
        .collect::<Vec<_>>();
    let mut note = format!(
        "{}; {}; per_stream_gpu_mean_ms={per_stream:?}; total_overlap_mean_ms={:.4}; \
         host_submit_excludes_waits=true; access.event_waits={}; \
         access.cross_stream_waits={}; access.read_read={}; \
         pending_release_bytes_at_submit={}; pending_release_count={}; \
         event_allocations={}; event_reuses={}; event_pending={}; \
         arena_slot_reuse={}; arena_pending_sets={}; arena_active_leases={}",
        if serial {
            "serial baseline (wait previous end before next submit)"
        } else {
            "parallel (all streams submitted back-to-back)"
        },
        case.shape,
        measurement.overlap.mean,
        measurement.access.event_waits,
        measurement.access.cross_stream_waits,
        measurement.access.read_read_concurrent,
        measurement.in_flight.pending_release_bytes,
        measurement.in_flight.pending_release_count,
        measurement.in_flight.event_allocations,
        measurement.in_flight.event_reuses,
        measurement.in_flight.event_pending,
        measurement.in_flight.arena_slot_reuse,
        measurement.in_flight.arena_pending_sets,
        measurement.in_flight.arena_active_leases,
    );
    if !serial {
        if let (Some(serial_mean), Some(speedup)) = (serial_mean, speedup) {
            note.push_str(&format!(
                "; serial_mean_ms={serial_mean:.4}; speedup={speedup:.3}"
            ));
        }
    }
    note
}

#[allow(clippy::too_many_arguments)]
fn make_case(
    case: &GraphCase,
    pattern: Pattern,
    parallel: bool,
    stream_count: usize,
    graph_hash_value: &str,
    plan_hash_value: &str,
    correctness: &str,
    compile_ms: f64,
    samples: Vec<Sample>,
    memory: MemorySnapshot,
    note: String,
) -> Case {
    let mut record = Case::new(
        BENCHMARK,
        case.name,
        format!(
            "{}/{}",
            pattern.label(),
            if parallel { "parallel" } else { "serial" }
        ),
        &case.shape,
    );
    record.stream_count = stream_count;
    record.graph_hash = graph_hash_value.to_string();
    record.plan_hash = plan_hash_value.to_string();
    record.correctness = correctness.to_string();
    record.compile_ms = compile_ms;
    record.samples = samples;
    record.memory = memory;
    record.note = note;
    record
}

fn check_correctness(
    case: &GraphCase,
    plan: &CudaMemoryPlan,
    executor: &CudaExecutor,
) -> Result<String, Box<dyn std::error::Error>> {
    let stream = executor.device().default_stream();
    let cuda_params = CudaParameterStore::from_host(&case.graph, &case.params, executor.pool())?;
    let mut cpu_inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
    for (&id, tensor) in &case.inputs {
        cpu_inputs.insert(id, tensor.clone());
    }
    let cpu = oxide_ml_cpu::Executor::<f32>::new(case.graph.clone())?;
    let expected = match cpu.run(&case.params, &cpu_inputs) {
        Ok(expected) => expected,
        Err(error) => return Ok(format!("CPU REFERENCE UNAVAILABLE: {error}")),
    };
    let actual = executor.run_planned(&case.graph, &cuda_params, &case.inputs, plan)?;
    let mut correctness = "PASS".to_string();
    for (index, &value) in case.graph.outputs().iter().enumerate() {
        let expected = expected.output(index)?.data();
        let actual = actual.download(&stream, value)?;
        if let Err(error) = within_tolerance(actual.data(), expected, 1e-4, 3e-3) {
            correctness = format!("FAIL output[{index}]: {error}");
            break;
        }
    }
    Ok(correctness)
}

fn run_matrix(
    executor: &CudaExecutor,
    case: &GraphCase,
    plan: &CudaMemoryPlan,
    compile_ms: f64,
    cases: &mut Vec<Case>,
) -> Result<(), Box<dyn std::error::Error>> {
    let correctness = check_correctness(case, plan, executor)?;
    let graph_hash_value = graph_hash(&case.graph);
    let plan_hash_value = plan_hash(plan);
    let output_count = case.graph.outputs().len();

    for &stream_count in &STREAM_COUNTS {
        let streams = (0..stream_count)
            .map(|_| executor.device().new_stream())
            .collect::<oxide_ml_cuda::Result<Vec<_>>>()?;
        let inputs = (0..stream_count)
            .map(|_| case.inputs.clone())
            .collect::<Vec<_>>();

        for pattern in patterns_for(case) {
            let params = build_params(executor, case, pattern, stream_count)?;
            let arena_plan = plan;

            let run = |serial: bool| -> Result<Measurement, Box<dyn std::error::Error>> {
                let arena = if pattern.uses_arena() {
                    Some(CudaPlannedArena::new(
                        arena_plan.clone(),
                        executor.pool().clone(),
                        stream_count,
                    )?)
                } else {
                    None
                };
                let mut submit = |index: usize| -> oxide_ml_cuda::Result<CudaExecution> {
                    if let Some(arena_ref) = arena.as_ref() {
                        executor.execute_async_arena(
                            &case.graph,
                            params.get(index),
                            &inputs[index],
                            arena_ref,
                            streams[index].clone(),
                        )
                    } else {
                        executor.execute_async_planned(
                            &case.graph,
                            params.get(index),
                            &inputs[index],
                            plan,
                            streams[index].clone(),
                        )
                    }
                };
                let measurement = measure_batch(
                    executor,
                    &streams,
                    arena.as_ref(),
                    output_count,
                    serial,
                    &mut submit,
                )?;
                Ok(measurement)
            };

            let serial_measurement = run(true)?;
            let serial_mean = serial_measurement.span.mean;
            let serial_access = {
                let mut total = serial_measurement.access.clone();
                add_access(&mut total, &params_access(&params));
                total
            };
            let serial_memory = memory_snapshot(
                executor,
                plan,
                serial_measurement.arena.as_ref(),
                &serial_access,
            );
            let serial_note = describe(&serial_measurement, true, None, None, case);
            cases.push(make_case(
                case,
                pattern,
                false,
                stream_count,
                &graph_hash_value,
                &plan_hash_value,
                &correctness,
                compile_ms,
                serial_measurement.samples,
                serial_memory,
                serial_note,
            ));

            let parallel_measurement = run(false)?;
            let speedup = serial_mean / parallel_measurement.span.mean;
            let parallel_access = {
                let mut total = parallel_measurement.access.clone();
                add_access(&mut total, &params_access(&params));
                total
            };
            let parallel_memory = memory_snapshot(
                executor,
                plan,
                parallel_measurement.arena.as_ref(),
                &parallel_access,
            );
            let parallel_note = describe(
                &parallel_measurement,
                false,
                Some(serial_mean),
                Some(speedup),
                case,
            );
            cases.push(make_case(
                case,
                pattern,
                true,
                stream_count,
                &graph_hash_value,
                &plan_hash_value,
                &correctness,
                compile_ms,
                parallel_measurement.samples,
                parallel_memory,
                parallel_note,
            ));
        }
        println!(
            "concurrency_graph_check: {} streams={stream_count} measured",
            case.name
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let meta = collect_meta(&device);
    println!(
        "concurrency_graph_check: gpu={} cc={} uuid={}",
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
        build_mixed,
    ] {
        let case = builder()?;
        let plan_begin = Instant::now();
        let plan = build_plan(&executor, &case)?;
        let compile_ms = case.compile_ms + plan_begin.elapsed().as_secs_f64() * 1000.0;
        if let Some(symbol) = case.probe_symbol {
            let mut max_bindings = HashMap::new();
            max_bindings.insert(symbol, 8usize);
            match executor.plan_bounded(&case.graph, &max_bindings) {
                Ok(_) => println!(
                    "concurrency_graph_check: {} bounded plan unexpectedly succeeded",
                    case.name
                ),
                Err(error) => println!(
                    "concurrency_graph_check: {} bounded plan rejected (expected for Dim::Dynamic): {error}",
                    case.name
                ),
            }
        }
        println!(
            "concurrency_graph_check: {} prepared plan_slots={} logical_bytes={}",
            case.name,
            plan.slot_count(),
            plan.logical_bytes
        );
        run_matrix(&executor, &case, &plan, compile_ms, &mut cases)?;
    }

    // Shared Write is not expressible through graph execution: ops allocate
    // fresh outputs and arena leases are exclusive. Record the pattern without
    // samples instead of calling raw kernels.
    for &stream_count in &STREAM_COUNTS {
        let mut record = Case::new(BENCHMARK, "all_graphs", "shared_write/not_applicable", "-");
        record.stream_count = stream_count;
        record.correctness = "NOT APPLICABLE".to_string();
        record.note = "NOT APPLICABLE: graph ops are functional (fresh output allocations) and \
                       CudaPlannedArena leases each slot set exclusively; two concurrent \
                       executions cannot write shared slots, and raw kernels are forbidden in \
                       this matrix."
            .to_string();
        cases.push(record);
    }

    let dir = out_dir("../../docs");
    let json_path = dir.join("phase7f19-concurrency-graph-results.json");
    write_results(&json_path, BENCHMARK, &meta, &cases)?;
    println!("concurrency_graph_check: wrote {}", json_path.display());
    println!(
        "concurrency_graph_check: cases={} warmup={WARMUP} rounds={ROUNDS} iterations={ITERATIONS}",
        cases.len()
    );
    Ok(())
}
