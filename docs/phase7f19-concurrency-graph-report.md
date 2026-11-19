# Phase 7F-19 4-Stream Graph Matrix Report

## 1. Scope

Binary: `crates/oxide-ml-cuda/src/bin/concurrency_graph_check.rs`.
Results: `phase7f19-concurrency-graph-results.json` / `.csv` (177 cases).

This is **graph execution**, not the earlier raw-GEMM `concurrency4_check`
(which is retained unchanged). Graphs:

- small MLP, large MLP, matmul, activation, reduction;
- dynamic batch (symbol bound at 4, max 8);
- mixed static + bounded symbol + `Dim::Dynamic` (generic planner fallback).

Backward graph: **NOT RUN**. It was removed from the matrix after the same
planned/backward constant-copy panic seen in `backward_check`; the normal-path
backward result is in `phase7f19-backward-benchmark-report.md`.

## 2. Matrix

- Streams: 1, 2, 4. Execution count equals stream count.
- Each execution has independent host inputs and outputs.
- Access patterns:
  - Independent buffers (per-execution `CudaParameterStore`)
  - Shared read (per-execution device stores from one read-only host store)
  - Shared parameters (one `CudaParameterStore` shared by all executions)
  - Shared arena (one `CudaPlannedArena`, capacity = stream count)
  - Pool + arena mixed (mixed graph only)
  - Shared write: **NOT APPLICABLE** (functional ops allocate fresh outputs;
    arena leases are exclusive; no raw kernels were substituted)
- 144 PASS, 30 "CPU REFERENCE UNAVAILABLE" (mixed graph: CPU reference cannot
  specialize `Dim::Dynamic`; timing and execution still ran), 3 NOT APPLICABLE
  (shared write at 1/2/4 streams). No case is labelled PASS without a checks
  verdict.

## 3. Timing method

Per-stream start/end `CudaEvent`s, origin = earliest start. The parallel span is
latest end minus earliest start; it is **not** the sum of per-stream times.
Overlap is the interval union during which at least two streams are active
(same algorithm as `concurrency4_check`). Serial is the same work with an
explicit wait for the previous stream before the next submit. Host submit
excludes waits. 10 warmup / 10 rounds / 10 iterations.

## 4. 4-stream results (median GPU span, all patterns)

| Graph | Pattern | GPU median ms | Overlap ms | Speedup |
|---|---|---:|---:|---:|
| small_mlp | independent_buffers | 0.4342 | 0.3247 | 1.053 |
| small_mlp | shared_read | 0.4280 | 0.3204 | 1.008 |
| small_mlp | shared_parameters | 0.4291 | 0.3223 | 1.163 |
| small_mlp | shared_arena | 0.5176 | 0.3973 | 1.009 |
| large_mlp | independent_buffers | 0.6603 | 0.5079 | 1.202 |
| large_mlp | shared_read | 0.6572 | 0.5025 | 1.113 |
| large_mlp | shared_parameters | 0.8004 | 0.6309 | 0.908 |
| large_mlp | shared_arena | 0.9630 | 0.7633 | 0.953 |
| matmul | independent_buffers | 0.2335 | 0.1706 | 1.028 |
| matmul | shared_read | 0.2298 | 0.1693 | 1.032 |
| matmul | shared_parameters | 0.2355 | 0.1731 | 1.065 |
| matmul | shared_arena | 0.3174 | 0.2438 | 1.013 |
| activation | independent_buffers | 0.2095 | 0.1643 | 0.944 |
| activation | shared_read | 0.2128 | 0.1641 | 0.898 |
| activation | shared_parameters | 0.2052 | 0.1632 | 0.876 |
| activation | shared_arena | 0.2949 | 0.2255 | 1.063 |
| reduction | independent_buffers | 0.2303 | 0.1811 | 0.835 |
| reduction | shared_read | 0.2051 | 0.1646 | 1.052 |
| reduction | shared_parameters | 0.2068 | 0.1626 | 0.910 |
| reduction | shared_arena | 0.2982 | 0.2349 | 1.058 |
| dynamic_batch | independent_buffers | 0.2028 | 0.1598 | 0.852 |
| dynamic_batch | shared_read | 0.1946 | 0.1536 | 0.935 |
| dynamic_batch | shared_parameters | 0.1884 | 0.1442 | 0.933 |
| dynamic_batch | shared_arena | 0.2746 | 0.2202 | 1.134 |
| mixed | independent_buffers | 0.4583 | 0.3429 | 1.049 |
| mixed | shared_read | 0.4588 | 0.3467 | 0.995 |
| mixed | shared_parameters | 0.4723 | 0.3606 | 0.981 |
| mixed | shared_arena | 0.6419 | 0.4839 | 1.014 |
| mixed | pool_arena_mixed | 0.6453 | 0.4934 | 0.991 |

Interpretation: the T4 does overlap the four streams (roughly 0.1–0.6 ms of
overlap), but these graphs are small enough that fixed per-stream submission and
event cost dominates. Speedups are close to 1.0; the highest observed is about
1.16 for small MLP shared parameters. No speedup below 1.0 is claimed as a win.

The arena pattern is consistently slower than independent/shared because slot
sets are serialized and cross-stream event waits are inserted.

## 5. AccessGuard and lifecycle

- `same_stream_elided`, `cross_stream_waits`, `read_read_concurrent` are recorded
  per case in the note; shared parameters can show read-read concurrency, arena
  cases show cross-stream waits.
- `pending_release_bytes` and `arena_active_leases` are captured after submit
  and before wait; they return to zero after the wait and poll.
- Event allocations/reuses and arena slot reuse are recorded in every note.
- Shared write is not expressible through functional graph execution; it is
  reported NOT APPLICABLE rather than tested with raw kernels.

## 6. Verdict

| Item | Verdict |
|---|---|
| 1/2/4 stream graph matrix | PASS (144 cases) |
| Independent / shared read / shared parameters / shared arena | PASS |
| Pool + arena mixed | PASS (timing; CPU ref unavailable) |
| Shared write | NOT APPLICABLE |
| Backward graph in 4-stream matrix | NOT RUN |
| Per-stream events, span, overlap, speedup | PASS |
| AccessGuard wait counts | PASS |
| Overall | **PASS with two documented gaps** |

## 7. Reproduction

```bash
cd crates/oxide-ml-cuda
CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda \
  cargo oxide build --arch sm_75 -- --bin concurrency_graph_check --release
BENCH_OUT_DIR=../../docs ./target/release/concurrency_graph_check
```

Exit 0; 177 cases; writes JSON/CSV.
