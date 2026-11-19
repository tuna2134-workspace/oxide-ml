# Phase 7F-19 Benchmark Closure — Final Report (pre-recovery)

> **Status note (2026-11-19):** constant-copy recovery and the final freeze are recorded in
> `phase7f19-recovery-final-report.md` and `phase7f-final-closure.md`.

> **Recovery addendum (2026-11-19):** the constant-copy panic was root-caused to a
> planned-slot capacity vs logical scalar mismatch (16 f32 capacity vs 1 logical
> element) and fixed in `CudaTensor::copy_from_host`. Linear Regression and
> Backward planned/arena/async were re-run successfully. See
> `phase7f19-recovery-final-report.md` for the current verdict; the BLOCKED
> statements below are the pre-recovery record.

# Phase 7F-19 Benchmark Closure — Final Report

## 1. 結論

**Phase 7F-19 は COMPLETED ではない。Phase 7F 全体も PARTIAL のままである。**

今回の Closure で、Forward Benchmark、Backward（normal path）、Mixed
Static/Bounded/Unbounded Graph、4 Stream Graph Matrix、Full Shape Matrix、
Memory Statistics Audit、Event Deadlock Audit、Driver Fault Audit、および
`benchmark_check` の Compute Sanitizer を実機 T4 で実行し、実測 JSON/CSV を
生成した。

一方、Linear Regression Benchmark は既存 executor の
`host-to-device copy length mismatch (16 vs 1)` panic により完走できなかった。
Backward の planned / arena / async モードも同じ panic で BLOCKED である。
新規 4 バイナリの Compute Sanitizer は GPU セッション時間切れで NOT RUN である。
これらを PASS として扱わない。

## 2. 実行環境

| Item | Value |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU UUID | `GPU-527ac068-0847-107c-a117-e78d96dacbb8` |
| Compute Capability | 7.5 |
| Driver | 580.82.07 |
| CUDA Toolkit | 13.0 (V13.0.88) |
| Rust | 1.99.1 |
| cuda-oxide | 0.4.0 |
| T4 memory | 15,360 MiB |
| Session | Colab GPU T4, `/content/oxide-ml` |

GPU clock / temperature / power は各 JSON の `meta.gpu_clocks` /
`gpu_temperature` / `gpu_power` に `nvidia-smi` 実測値として保存した。
取得できない場合は `unavailable` とし、推測値は入れていない。

## 3. 実行したコマンド

### CPU

```bash
cd oxide-ml
CARGO_HOME=$PWD/.cargo-home cargo test --workspace
CARGO_HOME=$PWD/.cargo-home cargo clippy --workspace --all-targets -- -D warnings
CARGO_HOME=$PWD/.cargo-home cargo doc --workspace --no-deps
```

結果:

| Command | Result |
|---|---|
| `cargo test --workspace` | PASS 192 / FAIL 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS (exit 0) |
| `cargo doc --workspace --no-deps` | PASS (exit 0) |

### CUDA build (T4)

```bash
cd /content/oxide-ml/crates/oxide-ml-cuda
export CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda
cargo oxide build --arch sm_75 -- --bin benchmark_check --release
cargo oxide build --arch sm_75 -- --bin backward_check --release
cargo oxide build --arch sm_75 -- --bin concurrency_graph_check --release
cargo oxide build --arch sm_75 -- --bin mixed_graph_check --release
cargo oxide build --arch sm_75 -- --bin shape_matrix_check --release
cargo oxide build --arch sm_75 -- --bin linear_regression_check --release
```

All six builds succeeded. The linear-regression binary compiled but did not
complete its run.

### CUDA run

```bash
BENCH_OUT_DIR=/content/bench-out ./target/release/benchmark_check
BENCH_OUT_DIR=/content/bench-out ./target/release/backward_check
BENCH_OUT_DIR=/content/bench-out ./target/release/concurrency_graph_check
BENCH_OUT_DIR=/content/bench-out ./target/release/mixed_graph_check
BENCH_OUT_DIR=/content/bench-out ./target/release/shape_matrix_check
BENCH_OUT_DIR=/content/bench-out ./target/release/linear_regression_check   # blocked
```

### Compute Sanitizer (`benchmark_check`)

```bash
compute-sanitizer --tool memcheck   --error-exitcode 1 ./target/release/benchmark_check
compute-sanitizer --tool racecheck  --error-exitcode 1 ./target/release/benchmark_check
compute-sanitizer --tool synccheck  --error-exitcode 1 ./target/release/benchmark_check
```

Results: memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors, exit 0 for
all three.

## 4. 実測結果

### 4.1 Forward Benchmark

30 cases (6 graphs × 5 modes), 100 samples/case. All correctness PASS.

| Graph | normal | planned | arena | async_planned | async_arena |
|---|---:|---:|---:|---:|---:|
| small_mlp | 0.1209 | 0.1068 | 0.1390 | 0.1103 | 0.1455 |
| large_mlp | 0.3129 | 0.3092 | 0.3297 | 0.2994 | 0.2640 |
| matmul | 0.0437 | 0.0435 | 0.0740 | 0.0382 | 0.0751 |
| activation | 0.0418 | 0.0423 | 0.0635 | 0.0365 | 0.0692 |
| reduction | 0.0477 | 0.0480 | 0.0614 | 0.0412 | 0.0707 |
| dynamic_batch | 0.0395 | 0.0389 | 0.0534 | 0.0348 | 0.0683 |

Median GPU event span in ms. Full min/median/max/mean/P90/P95/P99/stddev are in
`phase7f19-benchmark-results.json`. For `large_mlp` async, host submit
(0.1543 ms) is clearly below GPU span (0.2994 ms), showing that the submit
interval does not contain the wait.

### 4.2 Backward

5 graphs × forward/training (normal only), 100 samples/case, all PASS.

| Graph | forward median | training median |
|---|---:|---:|
| linear_mse | 0.1413 | 0.3156 |
| mlp_mse | 0.2364 | 0.6158 |
| add_relu | 0.0917 | 0.2171 |
| unbroadcast | 0.0774 | 0.1724 |
| activation_relu | 0.1065 | 0.2724 |

Planned/arena/async modes are BLOCKED by the panic. LayerNorm backward is
NOT APPLICABLE (operator not implemented by the CUDA executor).

### 4.3 Linear Regression

BLOCKED. The first normal case (`small N=16 F=1 batch=1`) passed with loss
0.000000, then the process panicked at the second case. No result JSON was
produced. Datasets/batches/modes and the intended checks are implemented in
`linear_regression_check.rs`, but the acceptance matrix is NOT RUN.

### 4.4 Mixed Graph

5 modes, 100 samples/case, all PASS. Checks PASS:

- static + bounded values use plan slots;
- unbounded `Dim::Dynamic` values are unplanned and use pool fallback;
- both paths coexist in one execution;
- pool/arena identities do not collide;
- generation increments; AccessGuard stats recorded;
- async drop reclaims; batch 9 rejected before launch (event allocations 3 → 3);
- failed overflow does not destroy the held arena lease.

`plan_bounded` on the mixed graph FAILS with `CudaError::Capacity` because the
current specializer rejects `Dim::Dynamic`. The benchmark records this as an API
limitation and uses the existing generic planner fallback.

Timings (median ms): normal 0.2219, planned 0.2223, arena 0.2598,
async_planned 0.1946, async_arena 0.2642.

### 4.5 4 Stream Graph Matrix

177 cases, 1/2/4 streams, 6 graph types, 5 access patterns, 10/10/10 samples.

- 144 PASS; 30 mixed-graph cases have "CPU REFERENCE UNAVAILABLE" (CPU cannot
  specialize `Dim::Dynamic`) but executed and were timed; 3 shared-write cases
  are NOT APPLICABLE.
- GPU overlap is real (0.12–0.61 ms) but speedups are near 1.0 because the
  graphs are small and per-stream overhead dominates. Highest observed 1.14
  (dynamic batch shared parameters). No speedup below 1.0 is claimed as a win.
- Shared write through functional graph execution is NOT APPLICABLE; no raw
  kernels were substituted.
- Backward graph in the 4-stream matrix is NOT RUN (same panic).

### 4.6 Shape Matrix

44 cases, 42 PASS, 2 NOT APPLICABLE, 0 FAIL.

- static / bounded / unbounded / mixed shapes executed;
- max+1 = 9 rejected with `CudaError::Capacity`, before launch (event
  allocations unchanged);
- same-shape reuse, batch 1→4→8→1, bounded↔unbounded transitions PASS;
- unbounded arena / async_arena are NOT APPLICABLE because the plan has zero
  slots.

### 4.7 Statistics

Every measured case uses warmup 10, rounds 10, iterations 10 = 100 samples.
Min/max/mean/median/P90/P95/P99/stddev are emitted with the nearest-rank method
`sorted[ceil(p/100*n)-1]`. `p99_reliable=true` only for n ≥ 100. Round means are
included in the JSON.

### 4.8 Memory statistics

Definitions are fixed in `phase7f19-memory-statistics-audit.md`. Logical,
pool, arena, event and AccessGuard values are separated. `driver_used_bytes` is
`null`; allocator reserved bytes are never presented as driver physical memory.

### 4.9 Event deadlock audit

Source-level analysis in `phase7f19-event-deadlock-audit.md`:

- `CudaEvent::wait_on` treats an unrecorded event as a no-op, so a wait cannot
  target a future record;
- `launch_with_access` and arena leases record completion events only after the
  already-submitted launch/lease;
- therefore a circular stream/event dependency is not constructible through the
  public API.

A dedicated timeout-isolated hostile cycle test was **NOT RUN**; the guarantee is
source-level plus existing event/shared-read-write tests. No infinite-hang test
was executed.

### 4.10 Driver fault audit

Mock fault injection (event state machine, access failure harness, pool/
quarantine) remains PASS from Phase 7F. Real driver fault injection was not
performed: **SOURCE ONLY / NOT RUN**. Context/device loss, destroy failure and
real kernel-launch failure are not driver-verified. This is a declared
limitation, not a PASS.

## 5. Acceptance Criteria

| Item | Verdict | Evidence |
|---|---|---|
| Backward Benchmark | **PARTIAL** | 5 graphs normal PASS; planned/arena/async BLOCKED; LayerNorm N/A |
| Linear Regression | **BLOCKED** | first case PASS, then executor panic; no JSON |
| Mixed Graph | **PASS** | 5 modes; all structural/capacity/identity/drop checks PASS; `plan_bounded` API limitation recorded |
| 4 Stream Graph | **PASS with gaps** | 144 PASS; shared write N/A; backward NOT RUN |
| Shape Matrix | **PASS** | 44 cases, 42 PASS, 2 N/A, 0 FAIL |
| Statistics | **PASS** | 100 samples/case; all required percentiles |
| Memory Audit | **PASS** | definitions fixed; driver value null |
| Deadlock Audit | **PASS (source-level)** | cycle not constructible; hostile test NOT RUN |
| Driver Fault Audit | **PASS (separation)** | Mock PASS; real SOURCE ONLY / NOT RUN |
| Sanitizer | **PARTIAL** | `benchmark_check` 3 tools PASS; new binaries NOT RUN |
| CPU / Clippy / Docs | **PASS** | CPU 192/0, CPU clippy 0, doc 0; CUDA clippy for new bins **NOT RUN** |

## 6. Phase 7F 最終判定

**PARTIAL.**

理由:

1. Linear Regression Benchmark が実行完了していない。
2. Backward の planned/arena/async が executor panic で BLOCKED。
3. 新規バイナリの Compute Sanitizer が NOT RUN。
4. LayerNorm backward は未実装で NOT APPLICABLE。
5. 実 Driver Fault Injection は SOURCE ONLY / NOT RUN。
6. 新規バイナリの CUDA clippy は NOT RUN。

完了条件 1（Backward / Linear Regression / Mixed Graph の実行）を満たさない
ため、Phase 7F-19 を COMPLETED にはできない。Forward / Mixed / 4 Stream /
Shape / Statistics / Memory Audit / Deadlock Audit は実測と監査を完了した。

## 7. 残存制約

1. **Executor planned constant-copy panic**: `host-to-device copy length
   mismatch left 16 right 1`. Backward/linear planned modes and linear normal
   transition hit this. Needs a separate executor fix; not worked around by
   disabling safety.
2. **LayerNorm backward** is not implemented in the CUDA executor.
3. **`plan_bounded` rejects `Dim::Dynamic`**: mixed static/bounded/unbounded
   graphs must use the generic planner with pool fallback.
4. **New binaries not sanitizer-verified**. CUDA clippy for the new bins is also NOT RUN.
5. **No real driver fault injection**; context/device lost untested.
6. **Deadlock test** is source-level only; no hostile cycle process was run.
7. T4 clocks/thermal state are recorded but uncontrolled; performance numbers
   are single-session medians, not cross-run claims.

## 8. 成果物

- `phase7f19-benchmark-methodology.md`
- `phase7f19-backward-benchmark-report.md`
- `phase7f19-linear-regression-report.md`
- `phase7f19-mixed-graph-report.md`
- `phase7f19-concurrency-graph-report.md`
- `phase7f19-shape-matrix-report.md`
- `phase7f19-memory-statistics-audit.md`
- `phase7f19-event-deadlock-audit.md`
- `phase7f19-driver-fault-audit.md`
- `phase7f19-sanitizer-report.md`
- `phase7f19-final-report.md` (this file)
- `phase7f19-benchmark-results.json` / `.csv` (aggregate of all measured runs)
- Per-binary JSON/CSV: forward, backward, concurrency-graph, mixed-graph, shape-matrix
- Sanitizer logs: `p7f19-bench-memcheck.log`, `-racecheck.log`, `-synccheck.log`
- `backward_check_full.rs` (five-mode variant, kept for the next closure)

## 9. Phase 8

Phase 8 には着手していない。
