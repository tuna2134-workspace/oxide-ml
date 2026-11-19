> **Recovery note (2026-11-19):** the BLOCKED status in this file was resolved by the
> constant-prefix copy fix. See `phase7f19-linear-regression-recovery-report.md` and
> `phase7f19-backward-recovery-report.md` for the re-run results.

# Phase 7F-19 Linear Regression Benchmark Report

## 1. Status

**BLOCKED / PARTIAL.** No result JSON was produced. The binary builds, starts
the small/batch-1 normal training loop, passes its first correctness check, and
then hits the same pre-existing planned-execution host-to-device assertion that
blocks the backward planned modes:

```text
linear_regression_check: small_b1 normal loss=0.000000 correctness=PASS
thread 'main' panicked at cuda-core/src/device_buffer.rs:1017:
assertion `left == right` failed: host-to-device copy length mismatch
  left: 16
 right: 1
```

The panic occurs during the second dataset/batch/mode transition, before the
result file is written. It is not a benchmark harness error and was not worked
around by disabling access guards or by faking numbers.

## 2. Implemented benchmark

Source: `crates/oxide-ml-cuda/src/bin/linear_regression_check.rs`.

- Datasets: small `N=16, F=1`, medium `N=64, F=4`, large `N=256, F=8`.
- Batches: 1, 4, 8, 32; a batch larger than the dataset is recorded as
  NOT APPLICABLE, not run.
- Graph: `matmul(x, w) + b`, MSE loss; backward via
  `build_training_graph`; optimizer `CudaSgd` with `lr=0.05`, `STEPS=100`.
- CPU reference: `oxide_ml_cpu::Executor` + `oxide_ml_optim::Sgd` with identical
  initial parameters, data order, steps and learning rate. Because
  `oxide-ml-optim` is only a dev-dependency of the CUDA crate, the existing
  optimizer source is included with `#[path = "../../../oxide-ml-optim/src/lib.rs"]`
  rather than editing `Cargo.toml`.
- Async training is NOT APPLICABLE: `CudaSgd::step` needs a unique device buffer,
  while a live `CudaExecution` holds parameter aliases; waiting to release them
  makes the step synchronous. Async forward-only is measured. No async training
  number is faked.
- Modes intended: normal, planned, arena, async_planned, async_arena. The
  current run was reduced to normal only after the first panic; planned/arena
  are **NOT RUN** (the planned probe is wrapped in `catch_unwind` and records
  NOT APPLICABLE if it fails), async training is NOT APPLICABLE.
- Checks implemented: final loss, weight, bias, gradient, CPU difference,
  convergence, finite values; measurement separates forward-only, training
  step, and optimizer update.

## 3. What did execute

| Case | Mode | Result |
|---|---|---|
| small N=16 F=1 batch=1 | normal | first correctness check PASS, loss 0.000000 |
| all later cases | normal | NOT RUN (panic) |
| all cases | planned/arena/async | NOT RUN (panic) |

The partial log is evidence of one successful normal correctness check, but it
is not enough to claim the linear-regression acceptance criterion. No
performance number is reported from a single partial case.

## 4. Root cause and relationship to backward

The panic is the same `host-to-device copy length mismatch (16 vs 1)` seen in
planned execution of the backward graphs. It is an executor/adapter defect
around a planned output or constant copy, not a training-loop or optimizer
error. The normal path is fine for simple graphs but the binary transitions to
planned/arena cases and aborts.

The minimal normal-only fallback (`MODES = [Normal]`) still panicked at the
second case, so the defect is not limited to the planned mode enum; it can also
be reached by the optimizer/update path in this benchmark. That needs a
dedicated executor investigation in the next closure.

## 5. Accuracy criteria (defined, not yet met)

- Final loss: relative error vs CPU reference.
- Weight/bias: `|cuda-cpu| <= 1e-3 + 1e-3*|cpu|`.
- Gradient: `|cuda-cpu| <= 5e-3 + 5e-3*|cpu|`.
- Convergence: loss must decrease monotonically over the last 20 steps and end
  below the configured threshold.
- Same initial parameters, same batch order, same number of updates for CPU and
  every CUDA mode.

These checks are implemented in the binary but were not reached for all cases.

## 6. Reproduction

```bash
cd crates/oxide-ml-cuda
CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda \
  cargo oxide build --arch sm_75 -- --bin linear_regression_check --release
BENCH_OUT_DIR=../../docs ./target/release/linear_regression_check
```

Result: build PASS, run exits 101 at the second case. No
`phase7f19-linear-regression-results.json` exists.

## 7. Verdict

| Criterion | Verdict |
|---|---|
| Binary implemented | PASS (source) |
| Normal small/batch-1 correctness | PASS (one case) |
| Dataset/batch matrix executed | NOT RUN |
| CPU/CUDA comparison complete | NOT RUN |
| Execution modes | NOT RUN |
| Result JSON/CSV | NOT RUN |
| Overall | **BLOCKED** |
