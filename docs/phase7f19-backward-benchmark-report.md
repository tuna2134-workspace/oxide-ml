> **Recovery note (2026-11-19):** the BLOCKED status in this file was resolved by the
> constant-prefix copy fix. See `phase7f19-linear-regression-recovery-report.md` and
> `phase7f19-backward-recovery-report.md` for the re-run results.

# Phase 7F-19 Backward Benchmark Report

## 1. What was executed

Binary: `crates/oxide-ml-cuda/src/bin/backward_check.rs`
(plus the full five-mode variant `backward_check_full.rs`, kept for the next
closure).

GPU: Tesla T4, CC 7.5, UUID `GPU-527ac068-0847-107c-a117-e78d96dacbb8`,
driver 580.82.07, CUDA 13.0.

Graphs: linear+MSE, 2-layer MLP+MSE, add+ReLU, unbroadcast bias, ReLU
activation. Each graph is a materialized backward graph from
`oxide_ml_autograd::build_training_graph`. CPU reference is
`oxide_ml_cpu::Executor::<f32>` with `ExecutionInputs`.

Conditions: warmup 10, rounds 10, iterations 10, 100 samples per case.
Tolerances: forward/gradient `1e-4 + 3e-3*|expected|` and
`5e-3 + 5e-3*|expected|`; the exact values are in each case note.

## 2. Results (normal path)

Measured JSON: `phase7f19-backward-results.json` / `.csv`.

| Graph | Forward-only median ms | Training graph median ms | Correctness |
|---|---:|---:|---|
| linear_mse | 0.1413 | 0.3156 | PASS |
| mlp_mse | 0.2364 | 0.6158 | PASS |
| add_relu | 0.0917 | 0.2171 | PASS |
| unbroadcast | 0.0774 | 0.1724 | PASS |
| activation_relu | 0.1065 | 0.2724 | PASS |

The training graph is the full forward+backward graph. The difference between
the two columns is not reported as a separately measured "backward-only" time;
the two intervals are measured independently, as required.

For every graph the comparison covered the forward output, scalar loss, and all
parameter gradients. No NaN/Inf was observed. The CPU reference and CUDA normal
path agreed within tolerance for all five graphs.

## 3. Execution modes

| Mode | Status |
|---|---|
| normal | PASS, measured |
| planned | **NOT RUN / BLOCKED** |
| arena | **NOT RUN / BLOCKED** |
| async_planned | **NOT RUN / BLOCKED** |
| async_arena | **NOT RUN / BLOCKED** |

Reason: a pre-existing host-to-device copy assertion in planned execution:
`host-to-device copy length mismatch left: 16 right: 1` in
`cuda-core/device_buffer.rs`. It reproduces on the training graph for every
graph tested. The full-mode binary catches the panic per graph at the top level,
prints `NOT RUN`, and continues; it does not fabricate planned/arena numbers.
The normal path is measured by the separate normal-path binary and is the data
above.

This is an executor/adapter defect, not a benchmark harness defect. It is
reported for the next closure and is not worked around by disabling
AccessGuard, arena, or event safety.

## 4. LayerNorm

`Op::LayerNormBackward` is not implemented by the CUDA executor
(`operation layer_norm_backward is not implemented by the CUDA backend`).
The full-mode probe records LayerNorm training as **NOT APPLICABLE** with zero
samples. LayerNorm forward itself is supported and is not claimed as a backward
result.

## 5. MLP / MSE / Add+ReLU / Unbroadcast

- MLP: 3→6→1 with ReLU hidden and sigmoid output; parameter gradients for four
  parameters matched CPU.
- MSE: elementwise product + mean-squared error; weight gradient matched CPU.
- Add+ReLU: unfused `add` then `relu` followed by sum; bias gradient matched CPU.
- Unbroadcast: `[4,3,2] + bias[2]`; the bias gradient required unbroadcast and
  matched CPU.
- Activation: ReLU activation backward through `mul` and `reduce_mean`.

## 6. Why this is not a full PASS

The acceptance criterion asks for the implemented backward graphs to be
measured and checked. Five graphs are measured and correct on the normal path,
but the planned/arena/async modes did not execute. The LayerNorm backward
operator is not implemented. Therefore the item is **PARTIAL**: normal-path
PASS, planned/arena/async BLOCKED, LayerNorm NOT APPLICABLE.

## 7. Reproduction

```bash
cd crates/oxide-ml-cuda
CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda \
  cargo oxide build --arch sm_75 -- --bin backward_check --release
BENCH_OUT_DIR=../../docs ./target/release/backward_check
```

The measured run exited 0 and printed `training=PASS` for all five graphs.
