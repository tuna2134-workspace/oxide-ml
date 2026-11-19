> **Final freeze note (2026-11-19):** the authoritative Phase 7F decision is in
> `phase7f-final-closure.md`: Implementation **COMPLETE**, Verification **PARTIAL**.
> The "PARTIAL" in this recovery report refers to verification completeness.

# Phase 7F-19 Recovery — Final Report

## 1. Root Cause

The failure was a **logical-versus-physical length confusion** in
`CudaTensor::copy_from_host`:

- `left: 16` = device buffer capacity in f32 elements (a 64-byte planned slot)
- `right: 1` = logical scalar constant (1 f32)

A planned slot is allowed to be larger than the value placed in it. The old
copy checked `data.len() == self.numel()` (1 == 1) and then called
`DeviceBuffer::copy_from_host` on the whole slot buffer, whose length was 16.
The external `DeviceBuffer` asserts exact equality, so it panicked.

This is confirmed by a CPU-only plan dump:

```text
scalar constant: assignment size=4, slot size=64 (16 f32)
```

and pinned by `scalar_constant_can_share_a_larger_slot`.

## 2. Fix

`crates/oxide-ml-cuda/src/buffer.rs::CudaTensor::copy_from_host` now:

1. requires `data.len() == self.numel()`;
2. requires `device capacity >= logical`;
3. uses the exact `DeviceBuffer::copy_from_host` when capacity == logical;
4. otherwise copies only the logical prefix with
   `cuda_core::memory::memcpy_htod_async`, then synchronizes;
5. rejects longer host data and too-small capacity instead of truncating or
   overflowing.

No fixed length (1 or 16), no benchmark-specific branch, no safety mechanism
change, no async submit synchronization.

## 3. Changed files

| File | Change |
|---|---|
| `crates/oxide-ml-cuda/src/buffer.rs` | logical-prefix copy |
| `crates/oxide-ml-cuda/src/bin/constant_copy_check.rs` | new GPU regression binary |
| `crates/oxide-ml-graph/tests/constants.rs` | new CPU constant tests |
| `crates/oxide-ml-compiler/tests/constant_capacity.rs` | new plan capacity tests (incl. 16-f32 slot) |
| `crates/oxide-ml-cuda/src/bin/linear_regression_check.rs` | clippy annotations; recovery run |
| `crates/oxide-ml-cuda/src/bin/backward_check_full.rs` | clippy annotations; recovery run |

No Event Safety / Execution Lifecycle / AccessGuard / MemoryPlan / Arena /
Dynamic Shape semantics were changed.

## 4. Constant Copy Regression

- CPU: 198 tests PASS / 0 FAIL (6 new constant tests).
- GPU `constant_copy_check`: PASS, including:
  - direct capacity-16 / scalar-1 copy;
  - scalar/tensor constants normal/planned/arena/async planned;
  - linear+MSE training graph planned/arena/async outputs.
- `constant_copy_check` sanitizer: memcheck/racecheck/synccheck all 0.

## 5. Linear Regression Benchmark

**Recovered.**

```text
PASS=99 FAIL=0 NOT APPLICABLE=23 cases=122
```

- 3 datasets × batches 1/4/8/32 (where valid).
- normal / planned / arena all PASS.
- async_planned / async_arena forward-only PASS; async training
  NOT APPLICABLE (CudaSgd requires unique device buffers and a host sync).
- batch 32 on small dataset NOT APPLICABLE (batch > N).
- CPU reference: loss, weight, bias, gradients, convergence, finite, stability
  all PASS.

## 6. Backward Benchmark

**Recovered.**

```text
PASS=65 FAIL=0 NOT APPLICABLE=5 cases=70
```

- Linear+MSE, MLP+MSE, MSE-only, ReLU activation, Add+ReLU, Unbroadcast:
  forward + training graph in normal/planned/arena/async_planned/async_arena.
- LayerNorm forward measured; LayerNorm backward NOT APPLICABLE
  (`layer_norm_backward` not implemented by CUDA backend).
- CPU reference: forward output, loss, parameter/input/weight/bias gradients,
  finite, all PASS.

## 7. CPU / CUDA Tests

| Item | Result |
|---|---|
| `cargo test --workspace` | PASS 198 / FAIL 0 |
| CPU clippy `-D warnings` | PASS |
| CPU doc | PASS |
| CUDA `cargo clippy --all-targets -- -D warnings` | PASS (`clippy-cuda2.log`, EXIT 0) |
| CUDA builds | PASS |
| CUDA `cargo oxide test --arch sm_75` | FAIL: cuda-oxide test-harness `DisjointSlice` MIR verification (not a benchmark binary) |
| GPU regression binaries | PASS (see below) |

## 8. Sanitizer

See `phase7f19-recovery-sanitizer-report.md` for the final matrix. Summary at
the time of writing:

- constant / backward / shape / benchmark: memcheck+racecheck+synccheck PASS.
- mixed: memcheck/racecheck PASS; synccheck 0 sanitizer errors but the
  application AccessGuard counter assertion failed (observed elision count
  differed from the expected 1; cause not determined).
- linear: memcheck PASS; racecheck NOT RUN; synccheck running/NOT RUN if not
  complete.
- concurrency: memcheck running/NOT RUN; racecheck/synccheck NOT RUN.

A NOT RUN is not PASS.

## 9. Unmet / Constraints

1. Async training for linear regression is NOT APPLICABLE (optimizer requires
   host-visible unique device buffers; applying it asynchronously would change
   semantics).
2. LayerNorm backward is NOT APPLICABLE (operator not implemented).
3. Some sanitizer cells are NOT RUN due to time; mixed synccheck has an
   application-level AccessGuard counter assertion failure (0 sanitizer errors).
4. Real CUDA driver fault injection is SOURCE ONLY / NOT RUN.
5. The prefix copy remains synchronous for lifetime safety; this is existing
   semantics, not a new async optimization.
6. `cargo oxide test --arch sm_75` fails in the cuda-oxide test-harness device
   codegen (`DisjointSlice` MIR verification); the benchmark binaries build and
   run independently.

## 10. Phase 7F-19 Decision

**PARTIAL.** The two named blockers are fixed and re-measured:

- Linear Regression: **PASS** (99 cases; 5 modes where applicable).
- Backward: **PASS** (65 cases; 5 modes; LayerNorm backward N/A).

The benchmark acceptance items 1–6 are met. Items 7–8 are not fully met:

- Sanitizer: constant/backward/shape/benchmark/linear-sync/linear-mem and
  concurrency-mem are clean. Linear racecheck and concurrency racecheck/synccheck
  are NOT RUN (timeout), and mixed synccheck has 0 sanitizer errors but an
  application-level counter assertion failure (exit 11).
- CUDA test suite (`cargo oxide test --arch sm_75`) fails to compile device code
  in the cuda-oxide test harness (`DisjointSlice` MIR verification). The
  benchmark binaries themselves build and run; this is a toolchain/test-harness
  limitation, not a benchmark failure.

Because conditions 7/8 are not fully satisfied, Phase 7F-19 remains **PARTIAL**.
The constant-copy fix and the linear/backward recovery are successful and are
not downgraded by this verdict.

## 11. Phase 7F Overall

**PARTIAL.** The benchmark blockers are resolved, but the sanitizer gaps and the
CUDA test-harness compilation failure remain. No safety mechanism was weakened.

## 12. Next actions

1. Fix or isolate the cuda-oxide test-harness `DisjointSlice` codegen failure so
   `cargo oxide test` can run.
2. Run the remaining sanitizer cells with a smaller but representative matrix,
   or on a dedicated session.
3. Relax the mixed-graph AccessGuard counter assertion to a scheduling-independent
   check and re-run synccheck.
4. Keep async-training and LayerNorm-backward as declared N/A; do not fake them.
