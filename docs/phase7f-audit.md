# Phase 7F audit (2026-11-18)

## Event
- `CudaEvent` wraps `Arc<cuda_core::Event>` and owns the context through it.
- `record`/`query`/`synchronize` are cuda-core APIs.
- Buffers keep `Arc<CudaEvent>` in a `Mutex<Vec<_>>`; all events must complete.
- An unrecorded cuda-core event queries as complete; the wrapper does not
  distinguish "never recorded" from "recorded and complete".

## Buffer / Pool
- `PooledBuffer` owns one `DeviceBuffer` and the event list.
- Drop checks all events; incomplete buffers go to the pending queue.
- `CudaTensor` clones share `Arc<PooledBuffer>`.
- Pool free/pending/stats are protected by separate mutexes.
- Pool holds `Arc<CudaContext>`, so events outlive executor handles.

## Executor
- `run` and `run_planned_on` mark every operand/output with a node event.
- `run_planned_on` keeps slot events and last stream.
- Planned slots are created inside each execution.
- `CudaExecution` owns `CudaRun` and the external stream; Drop does not sync.
- Because slots are execution-local, execution-to-execution cross-stream reuse
  cannot occur.

## Statistics
- Pool: leased bytes, peak, reserved, pending bytes/count, allocations,
  reuse, release.
- Plan: slot count, slot reuse, fallback, logical bytes.
- No arena/event-pool statistics are included in the verified run.

## Missing for Phase 7F
- Persistent slot-set arena and lease
- Buffer access modes / read-read optimization
- Dedicated `memplan_bench`
- Event pool GPU verification
- Async-Drop-specific test and cross-stream execution test
