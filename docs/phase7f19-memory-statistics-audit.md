# Phase 7F-19 Memory Statistics Audit

## 1. Purpose

This audit fixes the meaning of every memory statistic used by the Phase 7F-19
benchmarks. The values come from three different namespaces and must never be
added together or substituted for one another:

1. compiler logical plan (bytes the graph would like to use),
2. oxide-ml allocator bookkeeping (pool/arena), and
3. GPU driver physical memory (device-wide).

The JSON field names in `bench_common::MemorySnapshot` are the canonical names
used by all result files.

## 2. Logical (compiler plan)

| Field | Definition | Update point | Scope | Double counting |
|---|---|---|---|---|
| `logical_bytes` | Sum of all value sizes assigned by `MemoryPlanner` for the plan used | plan construction | one plan | No; sum over assignments |
| `planned_logical_bytes` | Same quantity for the plan actually handed to execution | plan hand-off | one execution | No |
| `runtime_required_bytes` | Bytes implied by the concrete runtime shapes, when every value has a static/symbolic shape | per execution | one execution | No; `null` when unbounded values make it undefined |

`logical_bytes` is not physical allocation. A plan may assign several values to
the same slot with non-overlapping lifetimes; the slot size is the max, not the
sum.

## 3. Pool

Source: `PoolStats` in `crates/oxide-ml-cuda/src/buffer.rs`.

| Field | Definition | Update point | Scope | Notes |
|---|---|---|---|---|
| `reserved_bytes` | Bytes retained by the pool: free + leased + pending release + quarantined | every acquire/release/poll | process pool | Not driver total |
| `active_bytes` (`current_bytes`) | Bytes currently leased to callers | acquire/release | process pool | Includes planned slots while leased |
| `free_bytes` | `reserved_bytes - active_bytes - pending_release_bytes - quarantined_bytes` | derived | process pool | Derived, not stored |
| `pending_release_bytes` | Bytes waiting for completion events before reuse | drop/poll | process pool | Not free yet |
| `quarantined_bytes` | Bytes retained because reuse could not be proven safe | failure handling | process pool | Never reused |
| `allocation_count` (`allocations`) | Driver allocations performed by the pool | acquire miss | process pool | Cumulative since pool creation |
| `reuse_count` (`reuses`) | Successful free-list reuses | acquire hit | process pool | Cumulative |

`reserved_bytes` is updated from the pool's own bookkeeping. It is **not** the
GPU's physical usage and must not be compared to `nvidia-smi` memory as if it
were.

## 4. Arena

Source: `ArenaStats` in `crates/oxide-ml-cuda/src/arena.rs`.

| Field | Definition | Update point | Scope | Notes |
|---|---|---|---|---|
| `capacity_bytes` | Physical slot storage reserved by all allocated slot sets | slot-set allocation | one arena | Equals `physical_reserved_bytes` today; kept separate for API stability |
| `used_bytes` | `active_slot_set_count * logical_bytes` | lease/finish/poll | one arena | Derived; an upper bound, not a driver measurement |
| `physical_reserved_bytes` | Sum of slot capacities actually allocated | slot-set allocation | one arena | Monotonic until arena drop |
| `slot_allocation_count` (`allocation_count`) | Slot buffers allocated | slot-set allocation | one arena | Counts individual slots |
| `slot_reuse_count` (`slot_set_reuse_count`) | Leases satisfied by a free existing slot set | lease | one arena | Not the same as pool reuse |
| `active_lease_count` (`active_slot_set_count`) | Slot sets currently leased | lease/finish/poll | one arena | |

Arena slots are allocated through `CudaBufferPool::acquire`; their bytes are
therefore also reflected in pool `reserved_bytes`. This is intentional nesting,
not double counting: the arena number describes the arena's own reservation,
the pool number describes the pool's retention. Reports must not sum them.

## 5. Event pool

Source: `EventPoolStats` in `crates/oxide-ml-event-safety`.

| Field | Definition | Update point | Scope |
|---|---|---|---|
| `allocation_count` | CUDA events created by the pool | event-pool miss | process event pool |
| `reuse_count` | Events reused from the free/pending lists | acquire | process event pool |
| `pending_count` | Events retained because their generation is not complete | release/query | process event pool |
| `quarantine_count` | Events that could not be proven safe | failure handling | process event pool |

`event.reuse_count` counts event leases, not arena or pool reuse. It is
reported per case but it is a cumulative process counter unless the binary
creates a fresh executor for the case (the binaries note which).

## 6. AccessGuard

Source: `AccessStats` in `crates/oxide-ml-execution-safety/src/access.rs`.

| Field | Definition | Update point |
|---|---|---|
| `read_count` | Recorded `AccessMode::Read` accesses | launch completion |
| `write_count` | Recorded `AccessMode::Write` accesses | launch completion |
| `read_write_count` | Recorded `AccessMode::ReadWrite` accesses | launch completion |
| `same_stream_elided_count` | Hazards skipped because CUDA stream order already enforces them | access planning |
| `cross_stream_wait_count` | Hazards resolved with an explicit event wait | access planning |
| `read_read_elision_count` | Concurrent readers that did not wait for each other | access planning |

Counters are per-buffer. A benchmark sums them only over the buffers it
explicitly inspects; they are not process-global. `event_waits` in the raw
struct is the sum of reader/writer waits and is reported as `read_count` only
when the binary has no separate read/write breakdown.

## 7. Driver memory

`driver_used_bytes` is `null` in the result files. A reliable per-process
driver query is not available through the public cuda-core API used here.
Device-wide `nvidia-smi` values are recorded only in `meta` (total memory) and
are never attributed to a case. The allocator's `reserved_bytes` is not used as
a substitute.

## 8. Double-counting rules

1. `logical_bytes` and `reserved_bytes` are different quantities; never add.
2. `pending_release_bytes` is part of `reserved_bytes`, not part of `free`.
3. `quarantined_bytes` is part of `reserved_bytes`, never reused.
4. Arena `physical_reserved_bytes` is backed by pool acquisitions; report both,
   never sum.
5. Event counts are event leases, not memory bytes.
6. AccessGuard waits are ordering events, not bytes or time.
7. GPU event time and host wall time are separate metrics; see the methodology.
8. In synchronous APIs the implicit wait is inside `host_submit`; it is not also
   reported as `wait`.

## 9. Verified sources

- `crates/oxide-ml-cuda/src/buffer.rs` (`PoolStats`)
- `crates/oxide-ml-cuda/src/arena.rs` (`ArenaStats`)
- `crates/oxide-ml-cuda/src/memplan.rs` (`CudaMemoryStats`, bounded plans)
- `crates/oxide-ml-event-safety/src/pool.rs` (`EventPoolStats`)
- `crates/oxide-ml-execution-safety/src/access.rs` (`AccessStats`)
- `crates/oxide-ml-compiler/src` (`MemoryPlan`, canonical plan hash)

## 10. Result

The definitions are consistent with the implementation. The only derived
fields are `free_bytes`, arena `used_bytes`, and the AccessGuard sum; they are
labelled as derived in the methodology and are not used as driver
measurements.
