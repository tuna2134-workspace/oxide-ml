> **Status note (2026-11-19):** this file predates the Phase 7F-19 closure run.
> See `phase7f19-final-report.md` and the per-binary reports for the current,
> measured verdicts. Items marked NOT RUN here may have been executed later.

# Phase 7F-19 Full Benchmark Report

## Method

- T4 / sm_75 / CUDA 13.0 / Rust 1.99.1 / cuda-oxide 0.4.0
- Warmup 2, Rounds 3, Iterations 10, Median
- GPU: `CudaEvent::elapsed_ms`；Host: submission loop の `Instant`
- Async の wait は Host Timer の外。D2H は Benchmark に含めない。

## Results

| Mode | GPU ms | Host Submit ms |
|---|---:|---:|
| normal | 0.1223 | 0.1221 |
| planned | 0.1052 | 0.1050 |
| arena | 0.1345 | 0.1343 |
| async_planned | 0.1312 | 0.1203 |
| async_arena | 0.1682 | 0.1531 |
| dynamic_normal | 0.0428 | 0.0424 |
| dynamic_planned | 0.0415 | 0.0412 |
| matmul_normal | 0.0503 | 0.0501 |
| matmul_planned | 0.0477 | 0.0475 |
| matmul_arena | 0.0622 | 0.0619 |
| activation_normal | 0.0418 | 0.0415 |
| activation_planned | 0.0436 | 0.0434 |
| activation_arena | 0.0597 | 0.0596 |
| reduction_normal | 0.0567 | 0.0565 |
| reduction_planned | 0.0578 | 0.0575 |
| reduction_arena | 0.0655 | 0.0654 |
| large_mlp_normal | 0.3096 | 0.3094 |
| large_mlp_planned | 0.3081 | 0.3079 |
| large_mlp_arena | 0.3156 | 0.3155 |
| xor_normal | 0.1467 | 0.1466 |
| xor_planned | 0.1354 | 0.1352 |
| xor_arena | 0.1498 | 0.1496 |

## Matrix

| Graph | Normal | Planned | Arena | Async | Dynamic |
|---|---|---|---|---|---|
| Small MLP | PASS | PASS | PASS | PASS | - |
| Large MLP | PASS | PASS | PASS | - | - |
| MatMul | PASS | PASS | PASS | - | - |
| Activation | PASS | PASS | PASS | - | - |
| Reduction | PASS | PASS | PASS | - | - |
| XOR MLP | PASS | PASS | PASS | - | - |
| Dynamic Batch | PASS | PASS | NOT RUN | - | PASS |
| Linear Regression | NOT RUN | NOT RUN | NOT RUN | NOT RUN | - |
| Backward Graph | NOT RUN | NOT RUN | NOT RUN | NOT RUN | - |
| Mixed Static/Bounded/Unbounded | NOT RUN | NOT RUN | NOT RUN | NOT RUN | - |

Stream:

- 1 stream: benchmark_check
- 2 streams: concurrency_check / async benchmark
- 4 streams raw GEMM: concurrency4_check PASS
- 4 stream graph benchmark: NOT RUN

Shape:

- Small / Large MLP, Dynamic batch 1/4/8, Bounded max 8, Unbounded output: PASS
- Full Shape Matrix: NOT RUN

## Statistics

Median only. P90 / P95 / P99 / min / max / stddev は Rounds=3 のため NOT RUN。

## Memory Statistics

論理: Logical / Planned Logical / Runtime Required（bounded validation）。
物理: Reserved / Active / Free / Pending / Quarantined。
Arena: Capacity / Used / Slot Allocation / Slot Reuse。
Event: Allocation / Reuse / Pending / Quarantine。
Access: Read / Write / ReadWrite / Same-Stream Elided / Cross-Stream / Read-Read。
Driver 実メモリは未取得。Allocator 値と区別。

## Raw

- [phase7f19-benchmark-results.json](phase7f19-benchmark-results.json)
- [phase7f19-benchmark-results.csv](phase7f19-benchmark-results.csv)
- `docs/gpu-p7f-closure-bench.log`

## NOT RUN

- Linear Regression / Backward Graph / Mixed Graph
- 4 Stream Graph Matrix
- Full Shape Matrix
- P90/P95/P99
- benchmark_check racecheck/synccheck
