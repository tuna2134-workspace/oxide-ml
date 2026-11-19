> **Recovery note:** the constant-prefix copy fix and the re-run sanitizer matrix are in
> `phase7f19-recovery-sanitizer-report.md`. The NOT RUN rows below are the pre-recovery record.

# Phase 7F-19 Compute Sanitizer Report

## 1. Environment

- GPU: NVIDIA Tesla T4, compute capability 7.5, UUID
  `GPU-527ac068-0847-107c-a117-e78d96dacbb8`
- Driver: 580.82.07
- CUDA toolkit: 13.0 (V13.0.88)
- `compute-sanitizer` from the same CUDA 13.0 toolkit
- Binaries built with `cargo oxide build --arch sm_75 -- --bin <name> --release`

## 2. Results

| Target | memcheck | racecheck | synccheck |
|---|---|---|---|
| `benchmark_check` | PASS (0 errors) | PASS (0 hazards) | PASS (0 errors) |
| `backward_check` | NOT RUN | NOT RUN | NOT RUN |
| `linear_regression_check` | NOT RUN (binary blocked) | NOT RUN | NOT RUN |
| `mixed_graph_check` | NOT RUN | NOT RUN | NOT RUN |
| `concurrency_graph_check` | NOT RUN | NOT RUN | NOT RUN |
| `shape_matrix_check` | NOT RUN | NOT RUN | NOT RUN |
| `gpu_check` | PASS (prior closure log) | NOT RUN this closure | NOT RUN this closure |
| `race4_check` | PASS (prior closure log) | PASS (prior closure log) | PASS (prior closure log) |

### 2.1 `benchmark_check`

Commands:

```text
compute-sanitizer --tool memcheck --error-exitcode 1 ./target/release/benchmark_check
compute-sanitizer --tool racecheck --error-exitcode 1 ./target/release/benchmark_check
compute-sanitizer --tool synccheck --error-exitcode 1 ./target/release/benchmark_check
```

Observed:

```text
========= ERROR SUMMARY: 0 errors
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
========= ERROR SUMMARY: 0 errors
EXIT:0
```

Logs:

- `p7f19-bench-memcheck.log`
- `p7f19-bench-racecheck.log`
- `p7f19-bench-synccheck.log`

The full 30-case, 100-sample matrix was run under each tool. No test was
reduced for sanitizer execution.

### 2.2 New closure binaries

`backward_check`, `concurrency_graph_check`, `mixed_graph_check` and
`shape_matrix_check` were built and executed normally on the T4, but the GPU
session expired before Compute Sanitizer could be run on them. They are
**NOT RUN** under sanitizer, not PASS. `linear_regression_check` did not
complete its normal run (executor panic) and is also NOT RUN.

## 3. Existing targets

`gpu_check` memcheck and the `dynamic_check` / `race4_check` memcheck+racecheck+
synccheck sets were run in the preceding Phase 7F closure. Their logs are kept
in `docs/p7f-*.log` and `docs/gpu-p7f-closure-gpu.log`. They are not re-labelled
as Phase 7F-19 results.

## 4. NOT RUN / limitations

- Real driver fault injection is outside Compute Sanitizer's scope. See
  `phase7f19-driver-fault-audit.md`.
- Sanitizer changes timing; no performance number from a sanitizer run is used
  in the benchmark reports.
- If a new binary could not finish under sanitizer because of the session
  timeout, it is recorded as NOT RUN with the exact target/tool in section 2.

## 5. Interpretation

`benchmark_check` exercises all five execution modes on six forward graphs,
including dynamic batch and arena reuse, under all three tools with zero
findings. This covers the path most likely to expose an out-of-bounds slot or a
missing stream dependency. It does **not** prove the absence of faults in the
new backward/linear/mixed/4-stream/shape binaries; those remain NOT RUN under
sanitizer. The newly measured normal-path results are therefore valid as
functional/performance data, not as sanitizer-clean evidence.
