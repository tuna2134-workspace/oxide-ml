# Phase 7 status: Event-aware Memory Management (partial)

## Verified on Colab Tesla T4 / sm_75

- `CudaMemoryPlan` adapter: slot/alignment/F32/static-shape validation
- planned slot execution; normal vs planned outputs equal
- planned graph: 2 slots, 6 assignments, 4 slot reuses, 116 logical bytes
- async planned execution on an external stream
- `CudaExecution::{is_complete, wait, collect}`
- CUDA Event wrapper: create/record/query/synchronize
- per-buffer completion-event tracking
- deferred release queue:
  - a 512x512 GEMM output was dropped while its event was incomplete
  - `pending_release_count=1`, `pending_release_bytes=1,048,576`
  - after stream synchronization + `poll_pending`, pending count returned to 0
- pool statistics: allocation/reuse/release, pending, reserved, peak, planned/fallback
- Compute Sanitizer `memcheck`: **0 errors** on the full GPU acceptance binary

## Implemented design

### Event

`CudaEvent` wraps `cuda_core::Event` in an `Arc` and binds it to its
`CudaContext`. `record`, `query` (`is_complete`), and `synchronize` are used by
the pool and executor.

### Buffer lifecycle

`PooledBuffer` holds completion events in a `Mutex<Vec<Arc<CudaEvent>>>`.
`CudaTensor::mark_used(event)` appends the event. On `PooledBuffer::drop`, the
pool checks every event; if all are complete the buffer returns to the free
list, otherwise it enters the pending-release queue. `acquire` and `stats`
poll the queue. This handles multiple streams by requiring every recorded event
to complete.

### Planned slots

`run_planned_on` keeps a completion-event vector and last stream per slot.
Same-stream reuse relies on stream ordering; a different stream waits on the
slot's recorded events before writing. Planned slots are currently allocated
per execution, so cross-execution cross-stream slot reuse is not yet exercised.

### Async ownership

`CudaExecution` owns the `CudaRun` (and therefore its buffers) plus the external
stream. Drop does **not** synchronize; buffers move to the deferred-release
queue with their completion events. `wait` synchronizes and downloads; repeated
`wait` calls are allowed. `collect` returns `None` while incomplete.

## Not implemented / not verified

- Cross-execution persistent planned slot arena
- Real cross-stream slot-reuse test (the wait code exists but no execution
  currently reuses a slot across two streams)
- Read/Read vs Read/Write/Write/Write access-mode arbitration
- Event pool
- Planned vs unplanned allocation/peak benchmark (`memplan_bench` absent)
- CPU unit tests for `CudaMemoryPlan` (the crate needs CUDA to build)
- Dedicated `memplan_bench` allocation/peak comparison

Phase 7 is therefore **partial**, not complete.
