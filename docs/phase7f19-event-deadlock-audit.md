# Phase 7F-19 Circular Event Wait / Deadlock Audit

## 1. Question

Can a user construct a circular CUDA event/stream dependency through the public
oxide-ml-cuda API, and if so, does the implementation detect or reject it?

## 2. API surface that creates dependencies

Only two public paths enqueue a cross-stream wait:

1. `CudaEvent::wait_on(&stream)` — enqueues `cuStreamWaitEvent`.
2. `AccessGuard` / `launch_with_access` — plans hazards and calls
   `event.wait_on(stream)` for each cross-stream reader/writer conflict.
   Arena slot-set reuse also calls `event.wait_on(stream)` in `arena.rs` when a
   slot set was last used on a different stream.

`CudaEvent::record(stream)` enqueues `cuEventRecord`. The event state machine is
in `crates/oxide-ml-event-safety/src/state.rs`.

## 3. Can a cycle be constructed?

### 3.1 Direct API

A cycle would require a stream to wait for an event whose record happens
*after* that wait in host call order. `CudaEvent::wait_on` handles this case as
follows:

- `Unrecorded` → returns `Ok(())` without calling the driver. It does **not**
  enqueue a future wait. A later `record` therefore cannot be reached by the
  earlier wait.
- `Completed` → returns `Ok(())`; the dependency is already satisfied.
- `Recorded` → calls `cuStreamWaitEvent` for the record that already exists at
  that host instant.
- `Failed` → returns an error.

Because host submissions are totally ordered and a wait can only target a
record that was submitted earlier, every dependency edge points backwards in
host submission order. A directed cycle would require an edge to a future
record, which the `Unrecorded` branch makes impossible.

Re-recording an event does not change this. `record_success` increments the
generation, and a wait enqueued before the re-record targets the earlier
generation. A later `wait_on` targets the new generation, which itself is ordered
after all work already submitted on its stream. There is still no edge to the
future.

### 3.2 AccessGuard

`launch_with_access` merges requests per buffer, plans hazards against the
buffer's previous readers/writer, enqueues the returned waits, then launches the
kernel and records the completion event. The event it waits on was recorded by
an earlier launch; the new launch's event is recorded only afterwards. This is
the same backwards-only edge structure. There is no API to attach a completion
event to an arbitrary earlier point or to declare a dependency from a future
launch.

### 3.3 Arena

Arena lease waits only on events stored in a slot set that was returned to the
arena by a previous lease. A lease cannot store an event from a future lease
because `SlotSetLease::finish` is what records the events. The same
backwards-only argument applies.

## 4. Conclusion

**Circular dependency is not constructible through the public API.**

The guarantee rests on `CudaEvent::wait_on` treating `Unrecorded` as a no-op
instead of enqueueing a future wait, and on completion events being recorded
only at the end of the already-submitted launch/lease. This is a source-level
guarantee; it is not a runtime cycle detector.

## 5. What is not claimed

- A user who drops to raw `cuda_core` / `cuStreamWaitEvent` outside
  oxide-ml-cuda can construct cycles. That is outside this API boundary.
- A driver bug, context loss, or a kernel that never completes can still hang a
  stream. The audit covers logical API-level cycles, not hardware/driver
  faults.
- No dedicated hostile cycle test was executed in this closure. Constructing a
  real infinite hang is deliberately avoided. The reasoning above is backed by
  the existing event tests (`gpu_check` event state machine, pending retention,
  drop safety; `race4_check` and `concurrency4_check` cross-stream shared
  read/write) but a timeout-isolated cycle attempt is **NOT RUN**.

## 6. Follow-up if a detector is ever required

If the API is extended with a "wait for a future record" primitive, add a
generation-aware dependency graph keyed by `(ContextId, event generation,
stream id)` and reject an edge whose target generation is not yet recorded.
Such a detector must not synchronize the host; it can be pure host bookkeeping.
It is not needed for the current API.

## 7. Source references

- `crates/oxide-ml-cuda/src/event.rs` (`wait_on`, `record`)
- `crates/oxide-ml-event-safety/src/state.rs` (`EventStateMachine`)
- `crates/oxide-ml-cuda/src/access.rs` (`launch_with_access`)
- `crates/oxide-ml-cuda/src/arena.rs` (`lease`)
- `crates/oxide-ml-cuda/src/bin/gpu_check.rs` (event state/pending/drop tests)
- `crates/oxide-ml-cuda/src/bin/concurrency4_check.rs` (shared read/write waits)
