# Phase 7F-19 Constant Copy Regression Report

## 1. CPU regression

| Test | Result |
|---|---|
| `oxide-ml-graph/tests/constants.rs::scalar_constant_is_one_f32_element` | PASS |
| `...::tensor_constant_preserves_declared_shape` | PASS |
| `...::broadcast_does_not_expand_constant_source` | PASS |
| `oxide-ml-compiler/tests/constant_capacity.rs::planned_constant_capacity_covers_logical_bytes` | PASS |
| `...::planned_scalar_constant_never_exceeds_one_element` | PASS |
| `...::scalar_constant_can_share_a_larger_slot` | PASS (64-byte slot, 4-byte scalar) |

The last test pins the exact `16` vs `1` relationship: a 64-byte slot holds 16
f32 elements, while the scalar constant has 1 logical element.

Full CPU workspace: **198 PASS / 0 FAIL**.

## 2. GPU regression: `constant_copy_check`

Command:

```bash
cd crates/oxide-ml-cuda
CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda \
  cargo oxide build --arch sm_75 -- --bin constant_copy_check --release
./target/release/constant_copy_check
```

Result: **PASS**, exit 0.

| Case | Result |
|---|---|
| Direct capacity 16 / logical scalar 1 copy | PASS |
| Only the logical prefix is written | PASS |
| Capacity 16 / logical tensor 4 copy | PASS |
| Scalar constant normal graph | PASS |
| Scalar constant planned | PASS |
| Scalar constant arena | PASS |
| Scalar constant async planned | PASS |
| Multi-element tensor constant planned | PASS |
| Linear+MSE training graph planned (loss + all parameter grads) | PASS |
| Linear+MSE training graph arena | PASS |
| Linear+MSE training graph async planned | PASS |

The linear training graph is the exact graph that previously triggered the
panic during planned/arena/async execution. It now matches the normal run for
every output (`ValueId` 5, 6, 14, 10).

## 3. GPU regression: recovered benchmarks

| Binary | Result |
|---|---|
| `linear_regression_check` | PASS=99 / FAIL=0 / N/A=23 (122 cases) |
| `backward_check_full` | PASS=65 / FAIL=0 / N/A=5 (70 cases) |
| `mixed_graph_check` | 5/5 PASS |
| `concurrency_graph_check` | 144 PASS / 30 CPU-ref-unavailable / 3 N/A |
| `shape_matrix_check` | 42 PASS / 2 N/A / 0 FAIL |

The previously blocked linear-regression and backward planned/arena/async paths
now execute. This is a functional recovery; performance numbers are in the
per-benchmark recovery reports.

## 4. Compute Sanitizer

`constant_copy_check` was run under all three tools:

| Tool | Result |
|---|---|
| memcheck | PASS, 0 errors |
| racecheck | PASS, 0 hazards |
| synccheck | PASS, 0 errors |

Logs: `san-const-memcheck.log`, `san-const-racecheck.log`,
`san-const-synccheck.log`.

## 5. What is not claimed

- No real driver fault injection.
- The prefix copy still synchronizes the stream; this is unchanged semantics,
  not an async-performance fix.
- Sanitizer results for the larger benchmarks are recorded separately and only
  as PASS when the tool actually completed.
