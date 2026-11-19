//! Phase 7F-19 mixed static / bounded / unbounded graph check.
//!
//! One graph contains, simultaneously:
//!   * a bounded symbol branch  (x_b[batch,4] -> matmul -> add -> relu)
//!   * an unbounded `Dim::Dynamic` branch (x_u[dynamic,4] -> matmul -> relu)
//!   * a fully static branch (x_s[2,4] -> matmul -> sigmoid)
//!   * one parameter `w` shared by the bounded and unbounded branches
//!   * one static bias parameter `b` used by the bounded branch
//!
//! API constraint observed by this binary
//! --------------------------------------
//! `CudaExecutor::plan_bounded` calls `specialize_graph_with_bindings`, which
//! rejects *any* value whose shape still contains `Dim::Dynamic`. The mixed
//! graph intentionally contains such values, so `plan_bounded` cannot produce
//! its plan. The binary records that API result honestly (FAIL for the
//! `plan_bounded` mixed-graph requirement) and then builds an equivalent
//! maximum-capacity plan through public APIs only:
//!   clone graph -> replace bounded symbols with their maxima, keep
//!   `Dim::Dynamic` -> `MemoryPlanner::plan` (dynamic values become
//!   `unplanned`) -> `CudaMemoryPlan::from_memory_plan`.
//! No executor / memplan / safety code is modified.

#[path = "bench_common/mod.rs"]
mod bench_common;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bench_common::{
    Case, MemorySnapshot, WARMUP, collect_meta, graph_hash, is_capacity_error, measure_async,
    measure_sync, out_dir, plan_hash, within_tolerance, write_results,
};
use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{
    DType, DenseTensor, Dim, ParameterId, ParameterStore, Shape, SymbolId, ValueId,
};
use oxide_ml_cuda::{
    AccessRequest, BufferIdentity, CudaBufferPool, CudaDevice, CudaExecutor, CudaMemoryPlan,
    CudaParameterStore, CudaPlannedArena, CudaRun, CudaTensor, launch_with_access,
};
use oxide_ml_graph::{Graph, GraphBuilder, ValueInfo, infer_outputs};

type Inputs = HashMap<ValueId, DenseTensor<f32>>;

const BATCH_MAX: usize = 8;
const ATOL: f64 = 1e-4;
const RTOL: f64 = 3e-3;

struct MixedGraph {
    graph: Graph,
    host_params: ParameterStore<f32>,
    s: ValueId,
    b: ValueId,
    w: ValueId,
    w_host: Vec<f32>,
    b_host: Vec<f32>,
    xs_host: Vec<f32>,
}

impl MixedGraph {
    fn inputs_for(&self, batch: usize) -> Inputs {
        let xb = (0..batch * 4)
            .map(|index| index as f32 * 0.1 - 0.5)
            .collect::<Vec<_>>();
        let xd = (0..batch * 4)
            .map(|index| ((index * 7 + 3) % 11) as f32 - 3.0)
            .collect::<Vec<_>>();
        let mut inputs = HashMap::new();
        inputs.insert(
            self.s,
            DenseTensor::from_vec([batch, 4], xb).expect("bounded input shape"),
        );
        inputs.insert(
            self.b,
            DenseTensor::from_vec([batch, 4], xd).expect("unbounded input shape"),
        );
        inputs.insert(
            self.w,
            DenseTensor::from_vec([2, 4], self.xs_host.clone()).expect("static input shape"),
        );
        inputs
    }

    // `s` is the bounded input, `b` the dynamic input, `w` the static input.
    fn expected(&self, batch: usize) -> Vec<Vec<f32>> {
        let inputs = self.inputs_for(batch);
        let xb = inputs[&self.s].data();
        let xd = inputs[&self.b].data();
        let xs = inputs[&self.w].data();
        let weight = &self.w_host;
        let bias = &self.b_host;
        let mut yb = vec![0.0f32; batch * 4];
        let mut yd = vec![0.0f32; batch * 4];
        let mut ys = vec![0.0f32; 2 * 4];
        for row in 0..batch {
            for col in 0..4 {
                let mut bounded = bias[col];
                let mut unbounded = 0.0f32;
                for k in 0..4 {
                    bounded += xb[row * 4 + k] * weight[k * 4 + col];
                    unbounded += xd[row * 4 + k] * weight[k * 4 + col];
                }
                yb[row * 4 + col] = bounded.max(0.0);
                yd[row * 4 + col] = unbounded.max(0.0);
            }
        }
        for row in 0..2 {
            for col in 0..4 {
                let mut acc = 0.0f32;
                for k in 0..4 {
                    acc += xs[row * 4 + k] * weight[k * 4 + col];
                }
                ys[row * 4 + col] = 1.0 / (1.0 + (-acc).exp());
            }
        }
        vec![yb, yd, ys]
    }
}

fn build_mixed_graph() -> Result<MixedGraph, Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");

    let x_bounded = builder.input(
        "x_bounded",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(4)]),
        DType::F32,
    );
    let x_unbounded = builder.input(
        "x_unbounded",
        Shape::new(vec![Dim::Dynamic, Dim::Static(4)]),
        DType::F32,
    );
    let x_static = builder.input("x_static", Shape::from_static([2, 4]), DType::F32);

    // One parameter is deliberately shared by the bounded and unbounded
    // branches; a second static parameter is used by the bounded branch only.
    let w = builder.try_parameter("shared_w", Shape::from_static([4, 4]), DType::F32, true)?;
    let bias = builder.try_parameter("bias", Shape::from_static([4]), DType::F32, true)?;

    let hb = builder.matmul(x_bounded, w)?;
    let hb = builder.add(hb, bias)?;
    let yb = builder.relu(hb)?;

    let hd = builder.matmul(x_unbounded, w)?;
    let yd = builder.relu(hd)?;

    let hs = builder.matmul(x_static, w)?;
    let ys = builder.sigmoid(hs)?;

    let graph = builder.finish(&[yb, yd, ys])?;

    let w_host = (0..16)
        .map(|index| {
            let (row, col) = (index / 4, index % 4);
            if row == col {
                1.0
            } else {
                (index % 3) as f32 * 0.1 - 0.1
            }
        })
        .collect::<Vec<_>>();
    let b_host = vec![0.5, -0.5, 0.25, -0.25];
    let xs_host = vec![1.0, -1.0, 0.5, -0.5, 2.0, 0.25, -0.75, 1.5];

    let mut host_params = ParameterStore::<f32>::new();
    host_params.add(
        ParameterId(0),
        "shared_w",
        DenseTensor::from_vec([4, 4], w_host.clone())?,
        true,
    )?;
    host_params.add(
        ParameterId(1),
        "bias",
        DenseTensor::from_vec([4], b_host.clone())?,
        true,
    )?;

    Ok(MixedGraph {
        graph,
        host_params,
        s: x_bounded,
        b: x_unbounded,
        w: x_static,
        w_host,
        b_host,
        xs_host,
    })
}

/// Maximum-capacity plan that keeps `Dim::Dynamic` values unplanned.
///
/// This mirrors `plan_bounded` for the bounded symbols while deliberately not
/// specializing the unbounded dynamic dimension (which `plan_bounded` rejects).
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
    Ok(CudaMemoryPlan::from_memory_plan(
        &specialized,
        &memory_plan,
    )?)
}

/// Concrete shapes for every graph value at the given inputs (public
/// `infer_outputs`, same rules used by the executor).
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

fn total_logical_bytes(
    graph: &Graph,
    shapes: &HashMap<ValueId, Vec<usize>>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut bytes = 0usize;
    for value in &graph.values {
        let dims = shapes
            .get(&value.id)
            .ok_or_else(|| format!("missing shape for value {}", value.id))?;
        bytes += dims.iter().product::<usize>() * value.dtype.size_in_bytes();
    }
    Ok(bytes)
}

fn check_outputs(
    run: &CudaRun,
    expected: &[Vec<f32>],
    stream: &Arc<cuda_core::CudaStream>,
) -> Result<(), String> {
    for (index, expected) in expected.iter().enumerate() {
        let actual = run
            .download(stream, run.outputs[index])
            .map_err(|error| format!("download output {index}: {error}"))?;
        within_tolerance(actual.data(), expected, ATOL, RTOL)?;
    }
    Ok(())
}

#[allow(dead_code)]
fn download_run(
    run: &CudaRun,
    stream: &Arc<cuda_core::CudaStream>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut outputs = Vec::new();
    for &value in &run.outputs {
        outputs.push(run.download(stream, value)?.data().to_vec());
    }
    Ok(outputs)
}

fn snapshot(
    executor: &CudaExecutor,
    arena: &CudaPlannedArena,
    plan: &CudaMemoryPlan,
    logical_bytes: usize,
    runtime_required: Option<usize>,
    access: oxide_ml_execution_safety::AccessStats,
) -> MemorySnapshot {
    let pool = executor.pool_stats();
    let arena_stats = arena.stats();
    let events = executor.event_stats();
    MemorySnapshot {
        logical_bytes: Some(logical_bytes),
        planned_logical_bytes: Some(plan.logical_bytes),
        runtime_required_bytes: runtime_required,
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
        // AccessStats has no separate read/write counters; leave those `None`
        // instead of relabelling an unrelated counter.
        access_reads: None,
        access_writes: None,
        access_read_write: None,
        access_same_stream_elided: Some(access.same_stream_elided_waits),
        access_cross_stream_waits: Some(access.cross_stream_waits),
        access_read_read_elision: Some(access.read_read_concurrent),
        driver_used_bytes: None,
    }
}

fn print_result(name: &str, ok: bool, detail: &str) {
    println!(
        "mixed_graph_check: {} {} ({detail})",
        if ok { "PASS" } else { "FAIL" },
        name
    );
}

#[allow(clippy::too_many_arguments)]
fn measure_mode(
    mode: &str,
    executor: &CudaExecutor,
    graph: &Graph,
    params: &CudaParameterStore,
    inputs: &Inputs,
    plan: &CudaMemoryPlan,
    arena: &CudaPlannedArena,
    arena_stream: &Arc<cuda_core::CudaStream>,
    default_stream: &Arc<cuda_core::CudaStream>,
) -> Result<Vec<bench_common::Sample>, Box<dyn std::error::Error>> {
    match mode {
        "normal" => measure_sync(
            default_stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                executor.run(graph, params, inputs)?;
                Ok(())
            },
        ),
        "planned" => measure_sync(
            default_stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                executor.run_planned(graph, params, inputs, plan)?;
                Ok(())
            },
        ),
        "arena" => measure_sync(
            default_stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                executor.run_arena(graph, params, inputs, arena, arena_stream.clone())?;
                Ok(())
            },
        ),
        "async_planned" => measure_async(
            default_stream,
            bench_common::ROUNDS,
            bench_common::ITERATIONS,
            WARMUP,
            || {
                Ok(executor.execute_async_planned(
                    graph,
                    params,
                    inputs,
                    plan,
                    default_stream.clone(),
                )?)
            },
        ),
        "async_arena" => measure_async(
            default_stream,
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
        other => Err(format!("unknown mode {other}").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let pool: CudaBufferPool = executor.pool().clone();
    let default_stream = device.default_stream();
    let arena_stream = device.new_stream()?;
    let meta = collect_meta(&device);
    println!(
        "mixed_graph_check: gpu={} cc={}",
        meta.gpu_name, meta.compute_capability
    );

    let mixed = build_mixed_graph()?;
    let graph = mixed.graph.clone();
    let params = CudaParameterStore::from_host(&graph, &mixed.host_params, &pool)?;
    let inputs8 = mixed.inputs_for(BATCH_MAX);
    let inputs4 = mixed.inputs_for(4);

    let mut binding_max = HashMap::new();
    binding_max.insert(SymbolId(0), BATCH_MAX);

    // ---- Plan construction -------------------------------------------------
    let plan_bounded_result = executor.plan_bounded(&graph, &binding_max);
    match &plan_bounded_result {
        Ok(_) => print_result(
            "plan_bounded(mixed graph with Dim::Dynamic)",
            true,
            "API accepted the mixed graph",
        ),
        Err(error) => print_result(
            "plan_bounded(mixed graph with Dim::Dynamic)",
            false,
            &format!(
                "API limitation: plan_bounded rejects Dim::Dynamic with {} (expected for this API)",
                error
            ),
        ),
    }
    let plan = mixed_plan(&graph, &binding_max)?;
    println!(
        "mixed_graph_check: fallback mixed plan slots={} planned_bytes={} unplanned={:?} capacity_bindings={:?}",
        plan.slot_count(),
        plan.logical_bytes,
        plan.unplanned,
        plan.capacity_bindings()
    );

    let (hb, hb2, yb) = (
        graph.nodes()[0].outputs[0],
        graph.nodes()[1].outputs[0],
        graph.nodes()[2].outputs[0],
    );
    let (hd, yd, hs, ys) = (
        graph.nodes()[3].outputs[0],
        graph.nodes()[4].outputs[0],
        graph.nodes()[5].outputs[0],
        graph.nodes()[6].outputs[0],
    );
    let static_values_planned = plan.assignment(hs).is_some()
        && plan.assignment(ys).is_some()
        && plan.assignment(yb).is_some()
        && plan.assignment(hb).is_some()
        && plan.assignment(hb2).is_some();
    print_result(
        "static + bounded values use plan slots",
        static_values_planned,
        "hs/ys/hb/hb2/yb all have slot assignments",
    );
    let unbounded_unplanned = plan.assignment(hd).is_none()
        && plan.assignment(yd).is_none()
        && plan.assignment(yb).is_some();
    print_result(
        "unbounded values are unplanned (pool fallback)",
        unbounded_unplanned,
        "hd/yd have no slot assignment",
    );
    let bounded_capacity_ok = plan
        .assignment(hb)
        .map(|assignment| assignment.size_bytes >= BATCH_MAX * 4 * 4)
        .unwrap_or(false);
    print_result(
        "bounded capacity covers batch max",
        bounded_capacity_ok,
        &format!("hb capacity >= {} bytes", BATCH_MAX * 16),
    );
    let shared_parameter = graph
        .nodes()
        .iter()
        .filter(|node| {
            matches!(&node.op, oxide_ml_graph::Op::MatMul)
                && node.inputs.len() == 2
                && (node.inputs[1] == graph.parameters()[0].value)
        })
        .count();
    print_result(
        "shared parameter used by bounded and unbounded branches",
        shared_parameter >= 2,
        &format!("matmul nodes consuming shared_w = {shared_parameter}"),
    );

    let shapes8 = runtime_shapes(&graph, &inputs8)?;
    let logical8 = total_logical_bytes(&graph, &shapes8)?;
    println!("mixed_graph_check: logical bytes at batch 8 = {logical8}");

    // ---- Arena and correctness --------------------------------------------
    let arena = CudaPlannedArena::new(plan.clone(), pool.clone(), 1)?;
    let expected8 = mixed.expected(BATCH_MAX);
    let normal8 = executor.run(&graph, &params, &inputs8)?;
    check_outputs(&normal8, &expected8, &default_stream)
        .map_err(|error| format!("normal: {error}"))?;
    let planned8 = executor.run_planned(&graph, &params, &inputs8, &plan)?;
    check_outputs(&planned8, &expected8, &default_stream)
        .map_err(|error| format!("planned: {error}"))?;
    print_result(
        "normal/planned batch 8 correctness",
        true,
        "relu/matmul reference within tolerance",
    );

    let pool_after_planned = executor.pool_stats();
    let both_paths = pool_after_planned.planned_slot_count > 0
        && pool_after_planned.fallback_allocation_count > 0;
    print_result(
        "both allocation paths coexist in one planned execution",
        both_paths,
        &format!(
            "planned_slots={} fallback_allocations={}",
            pool_after_planned.planned_slot_count, pool_after_planned.fallback_allocation_count
        ),
    );

    let arena8 = executor.run_arena(&graph, &params, &inputs8, &arena, arena_stream.clone())?;
    check_outputs(&arena8, &expected8, &default_stream)
        .map_err(|error| format!("arena: {error}"))?;
    print_result("arena batch 8 correctness", true, "arena + pool fallback");

    // Normal-run buffers vs arena slot buffers must not share identity.
    let normal_ids: HashSet<(u64, u64)> = normal8
        .values
        .values()
        .map(|tensor| {
            let id = tensor.identity();
            (id.id.0, id.generation)
        })
        .collect();
    let mut lease = arena.lease(&arena_stream)?;
    let (slots, events, streams) = lease.take()?;
    let arena_ids: HashSet<(u64, u64)> = slots
        .iter()
        .map(|tensor| {
            let id = tensor.identity();
            (id.id.0, id.generation)
        })
        .collect();
    let collisions = normal_ids.intersection(&arena_ids).count();
    lease.finish(slots, events, streams)?;
    arena_stream.synchronize()?;
    let _ = arena.poll()?;
    print_result(
        "pool and arena buffer identities do not collide",
        collisions == 0,
        &format!(
            "normal_ids={} arena_ids={} collisions={collisions}",
            normal_ids.len(),
            arena_ids.len()
        ),
    );

    // Generation increments across arena slot reuse.
    let mut gen_lease = arena.lease(&arena_stream)?;
    let (slots, events, streams) = gen_lease.take()?;
    let first: Vec<BufferIdentity> = slots.iter().map(|tensor| tensor.identity()).collect();
    gen_lease.finish(slots, events, streams)?;
    arena_stream.synchronize()?;
    let _ = arena.poll()?;
    let mut gen_lease = arena.lease(&arena_stream)?;
    let (slots, events, streams) = gen_lease.take()?;
    let second: Vec<BufferIdentity> = slots.iter().map(|tensor| tensor.identity()).collect();
    gen_lease.finish(slots, events, streams)?;
    arena_stream.synchronize()?;
    let _ = arena.poll()?;
    let generations_increment = !first.is_empty()
        && first
            .iter()
            .zip(&second)
            .all(|(a, b)| a.id == b.id && b.generation == a.generation + 1);
    print_result(
        "buffer generation increments across arena reuse",
        generations_increment,
        &format!(
            "slot0 id={} generation {} -> {}",
            first.first().map(|id| id.id.0).unwrap_or(0),
            first.first().map(|id| id.generation).unwrap_or(0),
            second.first().map(|id| id.generation).unwrap_or(0)
        ),
    );

    // ---- AccessGuard -------------------------------------------------------
    let access_tensor: CudaTensor = planned8.output(0)?.clone();
    let access_before = access_tensor.access_stats();
    let completion = launch_with_access(
        executor.event_pool(),
        &default_stream,
        &[AccessRequest::read(&access_tensor)],
        || Ok(()),
    )?;
    drop(completion.guards);
    let completion = launch_with_access(
        executor.event_pool(),
        &default_stream,
        &[AccessRequest::write(&access_tensor)],
        || Ok(()),
    )?;
    drop(completion.guards);
    default_stream.synchronize()?;
    let access_after = access_tensor.access_stats();
    let access_ok = access_after.same_stream_elided_waits > access_before.same_stream_elided_waits
        && access_after.pending_writer == 1;
    print_result(
        "AccessGuard stats work",
        access_ok,
        &format!(
            "same_stream_elided={} cross_stream={} pending_writer={}",
            access_after.same_stream_elided_waits,
            access_after.cross_stream_waits,
            access_after.pending_writer
        ),
    );

    // ---- Drop / reclaim ----------------------------------------------------
    let reclaim_pending_before = executor.pool_stats().pending_release_count;
    {
        let execution = executor.execute_async_planned(
            &graph,
            &params,
            &inputs4,
            &plan,
            default_stream.clone(),
        )?;
        drop(execution);
    }
    default_stream.synchronize()?;
    executor.pool().poll_pending();
    let reclaim_pending_after = executor.pool_stats().pending_release_count;
    let planned_reclaim =
        reclaim_pending_after == 0 && reclaim_pending_after <= reclaim_pending_before;
    print_result(
        "dropped async planned execution reclaims after stream sync",
        planned_reclaim,
        &format!(
            "pending_release_bytes={}",
            executor.pool_stats().pending_release_bytes
        ),
    );

    let reclaim_arena = CudaPlannedArena::new(plan.clone(), pool.clone(), 1)?;
    {
        let execution = executor.execute_async_arena(
            &graph,
            &params,
            &inputs4,
            &reclaim_arena,
            arena_stream.clone(),
        )?;
        drop(execution);
    }
    arena_stream.synchronize()?;
    let _ = reclaim_arena.poll()?;
    executor.pool().poll_pending();
    let reclaim_arena_stats = reclaim_arena.stats();
    let arena_reclaim = reclaim_arena_stats.active_slot_set_count == 0
        && executor.pool_stats().pending_release_count == 0;
    print_result(
        "dropped async arena execution reclaims after stream sync",
        arena_reclaim,
        &format!(
            "active_slot_sets={} pending_release={}",
            reclaim_arena_stats.active_slot_set_count,
            executor.pool_stats().pending_release_count
        ),
    );

    // ---- Capacity overflow must not disturb a live arena lease -------------
    let mut held =
        executor.execute_async_arena(&graph, &params, &inputs4, &arena, arena_stream.clone())?;
    let inputs9 = mixed.inputs_for(BATCH_MAX + 1);
    let events_before = executor.event_stats().allocation_count;
    let overflow = executor.run_planned(&graph, &params, &inputs9, &plan);
    let events_after = executor.event_stats().allocation_count;
    let capacity_rejected = match &overflow {
        Err(error) => is_capacity_error(error),
        Ok(_) => false,
    };
    let no_launch = events_before == events_after;
    print_result(
        "batch 9 rejected with CudaError::Capacity before launch",
        capacity_rejected && no_launch,
        &format!(
            "error={} event_allocations {} -> {}",
            overflow
                .as_ref()
                .err()
                .map(|error| error.to_string())
                .unwrap_or_else(|| "accepted".to_string()),
            events_before,
            events_after
        ),
    );
    if !(capacity_rejected && no_launch) {
        return Err("capacity boundary check failed".into());
    }
    let held_outputs = held.wait()?;
    for (index, expected) in mixed.expected(4).iter().enumerate() {
        let value = graph.outputs()[index];
        let actual = held_outputs
            .get(&value)
            .ok_or_else(|| format!("held execution missing output {index}"))?;
        within_tolerance(actual.data(), expected, ATOL, RTOL)?;
    }
    let valid_after_overflow =
        executor.run_arena(&graph, &params, &inputs4, &arena, arena_stream.clone())?;
    check_outputs(&valid_after_overflow, &mixed.expected(4), &default_stream)?;
    print_result(
        "failed overflow keeps the existing arena lease and later batch 4 run valid",
        true,
        "held async arena lease completed; subsequent run_arena(batch 4) PASS",
    );

    // ---- Timed modes -------------------------------------------------------
    let hash = graph_hash(&graph);
    let phash = plan_hash(&plan);
    let access_stats = planned8.output(0)?.access_stats();
    let memory = snapshot(
        &executor,
        &arena,
        &plan,
        logical8,
        Some(logical8),
        access_stats,
    );
    let mut cases = Vec::new();
    for mode in ["normal", "planned", "arena", "async_planned", "async_arena"] {
        let mut record = Case::new(
            "mixed_graph",
            "mixed_static_bounded_unbounded",
            mode,
            "batch=8 bounded[8,4] unbounded[8,4] static[2,4]",
        );
        record.graph_hash = hash.clone();
        record.plan_hash = if mode == "normal" {
            "unavailable".to_string()
        } else {
            phash.clone()
        };
        record.correctness = "PASS".to_string();
        record.memory = memory.clone();
        record.note = match mode {
            "normal" => "sync API; mixed graph uses the ordinary pool path only",
            "planned" => {
                "sync API; bounded+static values use plan slots, Dim::Dynamic values fall back to pool"
            }
            "arena" => {
                "sync API; bounded+static values use arena slots, Dim::Dynamic values fall back to pool"
            }
            "async_planned" => "submit does not wait; plan slots + pool fallback",
            _ => "submit does not wait; arena slots released after completion",
        }
        .to_string();
        record.samples = measure_mode(
            mode,
            &executor,
            &graph,
            &params,
            &inputs8,
            &plan,
            &arena,
            &arena_stream,
            &default_stream,
        )?;
        cases.push(record);
    }
    println!(
        "mixed_graph_check: timed modes={} samples/case={}",
        cases.len(),
        bench_common::ROUNDS * bench_common::ITERATIONS
    );

    let dir = out_dir("../../docs");
    let json_path = dir.join("phase7f19-mixed-graph-results.json");
    write_results(&json_path, "mixed_graph_check", &meta, &cases)?;
    println!("mixed_graph_check: wrote {}", json_path.display());
    Ok(())
}
