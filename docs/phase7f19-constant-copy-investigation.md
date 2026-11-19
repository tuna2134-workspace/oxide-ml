# Phase 7F-19 Constant Copy Investigation

## 1. Symptom

```text
thread 'main' panicked at cuda-core/src/device_buffer.rs:1017:
assertion `left == right` failed: host-to-device copy length mismatch
  left: 16
 right: 1
```

Triggered by planned execution of graphs containing a scalar constant
(backward seed `1.0`, linear-regression optimizer graphs), and by the
linear-regression benchmark after the first normal case.

## 2. What 16 and 1 mean

The assertion is:

```rust
assert_eq!(self.len, src.len(), "host-to-device copy length mismatch");
```

- `left = self.len` is the **device buffer capacity in f32 elements**.
- `right = src.len()` is the **host logical element count**.

So the failing copy was:

| Value | Meaning |
|---:|---|
| 16 | planned slot capacity: 64 bytes = 16 × f32 |
| 1 | logical scalar constant: 1 × f32 = 4 bytes |

They are elements, not bytes. The slot is larger than the logical value because
the compiler aligns slots to 64 bytes; a scalar constant can be placed in a
64-byte slot.

This was confirmed without CUDA by planning a graph with a scalar constant:

```text
value ValueId(1) shape=[] size=4   slot=1 slot_size=64
value ValueId(2) shape=[2, 4] size=32 slot=0 slot_size=64
constant slot size=64 assignment size=4
```

The same invariant is now pinned by
`crates/oxide-ml-compiler/tests/constant_capacity.rs::scalar_constant_can_share_a_larger_slot`.

## 3. Code path

1. `MemoryPlanner` assigns the scalar constant a 4-byte assignment inside a
   64-byte slot.
2. `run_planned_on` takes the slot tensor, whose storage length is the slot
   capacity (16 f32), and calls `CudaTensor::set_shape(vec![1])`.
   `set_shape` only checks `logical <= capacity`, so this is valid.
3. `execute_node` for `Op::ConstantScalar` calls
   `CudaTensor::copy_from_host(stream, &[value])`.
4. The old `CudaTensor::copy_from_host` checked
   `data.len() == self.numel()` (1 == 1, passes) and then called
   `DeviceBuffer::copy_from_host(stream, data)` on the whole slot buffer.
5. `DeviceBuffer::copy_from_host` requires `storage.len == data.len`
   (16 == 1), so it panicked.

Normal execution was unaffected because `CudaTensor::from_pool` allocates
exactly `numel` elements, so `storage.len == numel` always held.

## 4. Why this is a real design bug, not a benchmark artifact

The planned executor is allowed to place a logical value in a larger physical
slot. That is the whole point of slot reuse. Any host-to-device copy of a
planned value must therefore write the **logical prefix** only. The old
`copy_from_host` conflated logical element count with physical storage length.

This affects:

- scalar constants (backward seed, scalar loss scale),
- any tensor constant smaller than its assigned slot,
- normal/planned output comparison for backward and linear-regression graphs,
- arena execution of the same graphs (same slot tensors).

It does not affect normal execution of graphs whose tensors are exactly sized,
which is why the earlier forward benchmarks passed.

## 5. Rejected fixes

- **Force copy length to 1 or 16** — forbidden and wrong for tensor constants.
- **Pad host data to slot capacity** — would write beyond the logical value and
  risk clobbering slot contents; also expands constants unnecessarily.
- **Skip planned slots for constants** — would leave plan assignments unused
  and change allocation policy.
- **Disable planned execution in benchmarks** — hides the bug, violates the
  task.
- **Change `DeviceBuffer` assertion** — external dependency; would silently
  allow short copies and break its safety contract.

## 6. Chosen fix direction

`CudaTensor::copy_from_host` now distinguishes:

- logical element count (`self.numel()`),
- host source length (`data.len()`),
- device destination capacity (`buffer.len()`).

It requires `data.len() == logical` and `capacity >= logical`, uses the exact
`DeviceBuffer::copy_from_host` when `capacity == logical`, and otherwise copies
only the logical prefix with `cuda_core::memory::memcpy_htod_async`, then
synchronizes the stream (preserving the pre-existing constant-upload lifetime
contract). See `phase7f19-constant-copy-fix-report.md`.

## 7. Evidence commands

```bash
cargo test -p oxide-ml-compiler --test constant_capacity
cargo test -p oxide-ml-graph --test constants
# GPU:
./target/release/constant_copy_check
./target/release/linear_regression_check
./target/release/backward_check_full
```

The first two pass on CPU; the GPU runs are recorded in
`phase7f19-constant-copy-regression-report.md`.
