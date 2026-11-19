# Phase 7F-19 Full Shape Matrix Report

## 1. Scope

Binary: `crates/oxide-ml-cuda/src/bin/shape_matrix_check.rs`.
Results: `phase7f19-shape-matrix-results.json` / `.csv` (44 cases).

Shapes:

- static small `[8,16]`, medium `[64,64]`, large `[256,256]` ReLU;
- bounded symbol batch min 1, typical 4, max 8;
- unbounded `Dim::Dynamic` batch 7;
- mixed static/bounded/unbounded graph.

Modes: normal, planned, arena, async_planned, async_arena. 42 PASS, 2 NOT
APPLICABLE, 0 FAIL.

## 2. Capacity boundary

| Case | Input | Result |
|---|---|---|
| min | batch 1 | PASS |
| typical | batch 4 | PASS |
| max | batch 8 | PASS |
| max+1 | batch 9 | PASS: rejected with `CudaError::Capacity` |

The max+1 rejection message:

```text
capacity overflow: CUDA capacity exceeded: symbol 0 runtime 9 exceeds maximum 8,
event_allocations 2 -> 2
```

Event allocations did not change, supporting rejection before kernel launch.
The error is raised by `validate_runtime_capacity` before the node loop.

## 3. Modes

| Shape | normal | planned | arena | async_planned | async_arena |
|---|---|---|---|---|---|
| static small/medium/large | PASS | PASS | PASS | PASS | PASS |
| bounded 1/4/8 | PASS | PASS | PASS | PASS | PASS |
| unbounded dynamic | PASS | PASS | **N/A** | PASS | **N/A** |
| mixed | PASS | PASS | PASS | PASS | PASS |

Unbounded arena modes are NOT APPLICABLE because the generic plan has zero
slots for `Dim::Dynamic`; an empty arena would not exercise arena storage.
They are not reported as PASS.

## 4. Reuse

| Scenario | Check | Result |
|---|---|---|
| same shape repeated | arena slot_set_reuse 221 → 223, allocations 1 → 1 | PASS |
| batch 1 → 4 → 8 → 1 | outputs match reference; same arena | PASS |
| bounded → unbounded | planned bounded then unbounded run; pool reuses 10893 → 10901 | PASS |
| unbounded → bounded | reverse transition valid | PASS |

The pool/arena reuse counters increase while outputs remain correct; no buffer
identity collision or premature reuse was observed.

## 5. Timings (median GPU ms, selected)

| Shape | normal | planned | arena | async_planned | async_arena |
|---|---:|---:|---:|---:|---:|
| static_small | 0.0405 | 0.0389 | 0.0644 | 0.0336 | 0.0830 |
| static_medium | 0.0429 | 0.0435 | 0.0811 | 0.0375 | 0.0922 |
| static_large | 0.1252 | 0.1270 | 0.1448 | 0.1209 | 0.1415 |

All cases use 10 warmup / 10 rounds / 10 iterations (100 samples). The complete
matrix is in the JSON; `shape` and `mode` identify each row.


## 6. Compile time

`compile_ms` is recorded as `0.0` in this binary: plan creation is outside the
timed matrix and was not separately measured. No compile-time number is claimed
for the shape matrix.

## 7. Verdict

| Item | Verdict |
|---|---|
| Static / bounded / unbounded matrix | PASS |
| Capacity max+1 rejection | PASS |
| Reuse same shape / batch transitions | PASS |
| Arena for unbounded | NOT APPLICABLE (zero-slot plan) |
| Mixed shape | PASS |
| Statistics | PASS (100 samples/case, P90/P95/P99) |
| Overall | **PASS with 2 NOT APPLICABLE combos** |

## 8. Reproduction

```bash
cd crates/oxide-ml-cuda
CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda \
  cargo oxide build --arch sm_75 -- --bin shape_matrix_check --release
BENCH_OUT_DIR=../../docs ./target/release/shape_matrix_check
```

Exit 0; prints `cases=44 failures=0` and `PASS`.
