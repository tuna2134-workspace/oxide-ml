# Phase 7F-19 Constant Copy Fix Report

## 1. Change

`CudaTensor::copy_from_host` in
`crates/oxide-ml-cuda/src/buffer.rs` now implements a validated logical-prefix
copy instead of requiring the device storage length to equal the logical
element count.

### Before

```rust
if data.len() != self.numel() { ... }
Arc::get_mut(&mut self.storage)?
    .buffer_mut()?
    .copy_from_host(stream, data)?;   // requires storage.len == data.len
```

### After

```rust
let logical = self.numel();
if data.len() != logical { ... }
let buffer = storage.buffer_mut()?;
let capacity = buffer.len();
if capacity < logical { return Capacity error; }
if capacity == logical {
    buffer.copy_from_host(stream, data)?;   // exact fast path
} else {
    // logical prefix only; never the slot capacity
    memcpy_htod_async(buffer.cu_deviceptr(), data.as_ptr(), logical * 4, stream)
    stream.synchronize()?;
}
```

## 2. Contract

| Quantity | Rule |
|---|---|
| `data.len()` | must equal `self.numel()` (logical element count) |
| device capacity | must be `>= logical`; otherwise `CudaError::Capacity` |
| exact-size case | uses `DeviceBuffer::copy_from_host` |
| larger planned slot | writes exactly `logical` f32 elements |
| longer host data | rejected, never truncated |
| smaller device capacity | rejected, never overflowed |
| dtype | f32, unchanged (CUDA adapter contract) |
| synchronization | stream synchronized before return, preserving the existing constant-upload lifetime guarantee |
| broadcast | not expanded by this method; broadcast remains an operator responsibility |

## 3. Files changed

| File | Change |
|---|---|
| `crates/oxide-ml-cuda/src/buffer.rs` | logical-prefix copy + capacity/error checks |
| `crates/oxide-ml-cuda/src/bin/constant_copy_check.rs` | GPU regression binary (new) |
| `crates/oxide-ml-graph/tests/constants.rs` | constant shape / dtype / byte length CPU tests (new) |
| `crates/oxide-ml-compiler/tests/constant_capacity.rs` | plan capacity tests incl. the 64-byte/16-f32 slot case (new) |
| `crates/oxide-ml-cuda/src/bin/linear_regression_check.rs` | clippy annotations and full-mode recovery run |
| `crates/oxide-ml-cuda/src/bin/backward_check_full.rs` | clippy annotations and full-mode recovery run |

No Event Safety, Execution Lifecycle, AccessGuard, MemoryPlan, Arena, Dynamic
Shape, or async API semantics were changed. No `synchronize` was added to an
async submit path; the prefix copy preserves the pre-existing constant-upload
synchronization.

## 4. Why this is shared

The fix is in the common tensor copy method, so normal allocation, planned
slots, arena slots and pool fallback all use the same validation. There is no
benchmark-specific branch and no fixed copy length.

## 5. Verification summary

- CPU: 198 tests PASS (6 new constant tests).
- GPU `constant_copy_check`: PASS, including capacity-16/scalar-1 direct case,
  scalar/tensor constants in normal/planned/arena/async planned, and linear
  training graph planned/arena/async outputs.
- GPU `linear_regression_check`: 99 PASS, 23 NOT APPLICABLE (async training and
  batch 32 on the small dataset), 0 FAIL.
- GPU `backward_check_full`: 65 PASS, 5 NOT APPLICABLE (LayerNorm backward),
  0 FAIL.
- Compute Sanitizer: see `phase7f19-recovery-sanitizer-report.md`.

## 6. Residual limits

- The prefix copy is synchronous because the host slice may be dropped
  immediately; this matches the previous behavior but means constant upload is
  a host synchronization point inside an execution. A future async constant
  upload would need ownership/lifetime tracking; not changed here.
- `DeviceBuffer`'s internal access tracker is bypassed for the prefix path; the
  stream is synchronized and oxide-ml's own event/access tracking still applies
  to the tensor.
- No real driver fault injection was performed.
