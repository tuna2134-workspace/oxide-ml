# Phase 7F-19 Benchmark Methodology

## 1. Scope

This document defines the measurement contract used by the Phase 7F-19
benchmark binaries. It applies to:

- `benchmark_check` (forward graphs)
- `backward_check`
- `linear_regression_check`
- `mixed_graph_check`
- `concurrency_graph_check`
- `shape_matrix_check`

The benchmark binaries only use the existing public CUDA, AccessGuard, event,
arena, pool, and execution-lifecycle APIs. No safety mechanism is disabled or
simplified for measurement. No host synchronisation is inserted inside an
asynchronous submit region.

## 2. Measurement intervals

| Metric | Interval | Notes |
|---|---|---|
| Compile Time | graph construction + `MemoryPlanner::plan` + `CudaMemoryPlan` conversion | measured once per graph and stored as `compile_ms` |
| Host Submit Time | host wall clock around the submission call only | for async APIs this excludes `wait()` |
| GPU Execution Time | CUDA event elapsed between start/end events recorded on the measured stream | not host wall clock |
| End-to-End Time | host wall clock from the start of the measured call to completion/wait | includes submit + wait + any D2H |
| H2D Time | host-to-device copy of graph inputs/parameters | measured outside the hot loop; parameters are uploaded once per mode |
| D2H Time | device-to-host copy of outputs | only measured where the binary explicitly downloads; otherwise `n=0` |
| Synchronization Time | explicit `CudaEvent::synchronize` / `CudaExecution::wait` | async cases only |
| Allocation Time | plan/arena/pool allocation and slot acquisition | reported as counts and bytes; not counted inside GPU time |

### 2.1 Synchronous APIs

`CudaExecutor::run`, `run_planned`, and `run_arena` synchronise the stream
before returning. Their host call therefore contains both submission and an
implicit wait. The binaries report:

- `host_submit` = the full synchronous call duration (submit + implicit wait),
  flagged by `host_call_includes_wait=true`;
- `gpu` = CUDA event span from before the call to after it; this includes the
  host tail between kernel completion and the end-event record;
- `wait` = 0 / not separately measurable.

They are never presented as a pure asynchronous submit time.

### 2.2 Asynchronous APIs

`execute_async_planned` and `execute_async_arena` do not synchronise. The
measurement is:

1. record start event on the execution stream;
2. call the submit API;
3. stop the host submit timer;
4. record end event on the same stream;
5. start the wait timer, synchronise the end event;
6. stop the wait timer.

The GPU interval is exactly the event span between (1) and (4). It does not
include the host wait in (5). `e2e = submit + wait`. No `wait()` is placed in
the submit interval.

### 2.3 Multi-stream interval rules

For stream counts 2 and 4 the binaries record one start and one end event per
stream. The serial GPU span is measured by running the executions in sequence
with an explicit dependency before each submission. The parallel GPU span is
the elapsed time from the earliest start event to the latest end event. It is
**not** the sum of per-stream times. Overlap is computed with an interval union
over the per-stream `[start, end]` intervals, counting the time during which at
least two streams are active. Speedup is `serial_span / parallel_span`.

## 3. Statistics

- Warmup: 10 executions.
- Rounds: 10.
- Iterations per round: 10.
- Samples per case and metric: 100 (one sample per iteration).
- Round means are emitted separately so round-to-round variation is visible.

Reported values: minimum, maximum, mean, median, P90, P95, P99, and
population standard deviation. Percentiles use the nearest-rank definition:

```
sorted[ceil(p / 100 * n) - 1]
```

P99 is only marked reliable when `n >= 100`. Cases with fewer samples carry
`p99_reliable=false`; P99 must not be interpreted as a tail estimate there.

## 4. Fixed conditions recorded with each result

Every JSON file contains one `meta` object with:

- GPU name, UUID, compute capability, total memory
- CUDA driver version, CUDA toolkit / nvcc version
- Rust version, cuda-oxide version
- GPU SM/memory clocks, temperature, power draw (queried from `nvidia-smi`;
  `unavailable` when the query fails, never guessed)
- timestamp, warmup/round/iteration counts, percentile method

Every case contains:

- benchmark name, graph name, execution mode, input shape, dtype
- stream count, graph hash, MemoryPlan hash
- compile time, sample count, all five metric statistic blocks
- round GPU means
- memory snapshot (see below)
- correctness verdict and a note describing the exact interval semantics

`Graph Hash` is `Graph::canonicalize().hash()`; `MemoryPlan Hash` is
`CudaMemoryPlan::plan_hash()`. Both are content hashes, not timestamps.

## 5. Correctness and tolerance

Unless a graph-specific report states otherwise:

- forward: `|cuda - cpu| <= 1e-4 + 3e-3 * |cpu|`
- gradients: `|cuda - cpu| <= 5e-3 + 5e-3 * |cpu|`
- LayerNorm backward: `|cuda - cpu| <= 1e-2 + 1e-2 * |cpu|`
- planned/arena vs normal (same CUDA implementation): `|a - b| <= 1e-6 + 1e-5 * |b|`

Every checked tensor is also tested for NaN/Inf. Stability is checked by
running the same input more than once and comparing outputs/gradients.

## 6. Memory statistics

The JSON separates these namespaces deliberately:

### Logical (compiler)

- `logical_bytes`: sum of planned value sizes at plan time.
- `planned_logical_bytes`: same quantity for the plan actually executed.
- `runtime_required_bytes`: size implied by runtime shapes, when computable;
  otherwise `null`.

### Pool (allocator)

- `reserved_bytes` (pool-retained, not driver total)
- `active_bytes` (currently leased)
- `pending_release_bytes`
- `quarantined_bytes`
- `allocation_count`, `reuse_count`

### Arena

- `capacity_bytes` (physical slot storage reserved by the arena)
- `used_bytes` (active leased slot sets * logical bytes; derived, marked as such)
- `physical_reserved_bytes`
- `slot_allocation_count`, `slot_reuse_count`, `active_lease_count`

### Event

- `allocation_count`, `reuse_count`, `pending_count`, `quarantine_count`

### AccessGuard

- `read_count`, `write_count`, `read_write_count`
- `same_stream_elided_count`
- `cross_stream_wait_count`
- `read_read_elision_count`

AccessGuard counters are per-buffer and are summed only over the buffers the
case actually inspects. They are not global process counters.

### Driver memory

`driver_used_bytes` is `null` unless a reliable per-process driver query is
available. The allocator's `reserved_bytes` is never substituted for driver
physical memory. Device-wide `nvidia-smi` memory is not attributed to a
benchmark.

## 7. Units and rounding

- Times are milliseconds (`ms`) as floating point.
- Bytes are bytes.
- Counts are integers.
- JSON statistics carry 9 decimal places; CSV carries 6.
- Values are written directly from the measured sample vector. No result is
  transcribed or hard-coded by hand.

## 8. What the benchmark does not claim

- Event timing is not a profiler trace. It measures stream-ordered GPU spans.
- Synchronous API GPU spans include a host tail; async spans do not.
- T4 clocks and thermal state are recorded but not controlled.
- `nvidia-smi` memory is device-wide and cannot be attributed per benchmark.
- Real driver fault injection is not performed by these binaries; see
  `phase7f19-driver-fault-audit.md`.
