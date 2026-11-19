# Phase 7F-19 CUDA Driver Fault Audit

## 1. Scope

This audit separates two different things:

- **Mock fault injection** in the CPU safety crates, which exercises the state
  machines used by the CUDA adapter.
- **Real CUDA driver faults** on the T4, which are not deliberately injected in
  this closure.

Real driver faults are not forced. No artificial context loss, device removal,
or invalid handle is created. The goal is to state exactly which error paths
exist, which are tested by mock, and which remain source-only.

## 2. Error-return paths

### 2.1 Context / device

- `CudaDevice::new` returns `CudaError::Driver` if the ordinal is invalid or
  initialization fails.
- `CudaContext::bind_to_thread` is called before every raw driver operation;
  failure is converted to a driver/event error.
- There is no automatic "context lost => recreate" path. A lost context
  surfaces as a driver error; the caller must stop using the process's CUDA
  objects. This is intentional: silently recreating a context would invalidate
  every arena/event identity.

### 2.2 Event create / record / query / synchronize

Source: `crates/oxide-ml-cuda/src/event.rs`.

- `CudaEvent::new` → `CudaError::Driver` on `cuEventCreate` failure.
- `record` → checks context identity first, then `cuEventRecord`; failure marks
  the event `Failed` and returns `CudaError::EventFailure`.
- `is_complete` → a driver error marks the event failed; `NOT_READY` is the only
  result mapped to `false`.
- `synchronize` → failure marks the event failed. A failed event cannot be
  re-recorded or reused.
- Failed events are quarantined by `EventPool`; `EventPoolStats::quarantine_count`
  and `quarantine_total` expose this.

### 2.3 Stream wait event

`launch_with_access` returns the error from `event.wait_on(stream)` before the
kernel launch. The event is marked failed by `wait_on`; the launch is not
submitted. The buffer's access state is not updated with a completion for a
launch that did not happen.

### 2.4 Buffer allocation / release

Source: `crates/oxide-ml-cuda/src/buffer.rs`.

- Pool acquisition failure propagates as `CudaError::Driver`; no free-list entry
  is fabricated.
- `PooledBuffer::drop` evaluates completion events. If the state cannot prove
  safety, the buffer moves to quarantine instead of the free list
  (`quarantined_count`, `quarantined_bytes`, `quarantine_total`).
- `poll_pending` only frees entries whose events are complete; a query error is
  treated as "not safe", not as complete.

### 2.5 Kernel launch

Source: `crates/oxide-ml-cuda/src/kernels.rs`, `access.rs`.

`launch_with_access` performs the access plan, records the completion event,
and only then marks tensors as used. A launch failure returns without recording
a completion; the event lease is not published. No buffer is marked as used by
a failed launch.

### 2.6 Execution lifecycle

`CudaExecution` tracks `Submitted → Running → Completed → Collected` and
`Failed`. A failed completion wait marks the execution failed and retains
resources for `Drop`; `Drop` never synchronizes and never panics.

## 3. Mock-verified vs real-driver-verified

| Fault path | Mock | Real driver |
|---|---|---|
| Event create failure | PASS (event-safety mock) | NOT RUN |
| Event record failure | PASS (event-safety mock) | NOT RUN |
| Event query failure | PASS (event-safety mock) | NOT RUN |
| Event synchronize failure | PASS (event-safety mock) | NOT RUN |
| Event destroy failure / quarantine | PASS (event-safety mock) | NOT RUN |
| Stream wait failure | PASS (access-failure mock) | NOT RUN |
| Kernel launch failure | PASS (access-failure mock) | NOT RUN |
| Buffer allocation failure | PASS (pool/execution-safety mock) | NOT RUN |
| Buffer release / quarantine | PASS (mock) | Normal path only |
| Context binding failure | PASS (execution-safety mock) | NOT RUN |
| Device lost | PASS (execution-safety mock) | NOT RUN |
| Execution drop safety | PASS (mock + normal GPU path) | Normal path PASS |

The existing mock results are not invalidated by the absence of real faults.
They are the regression boundary for the production state machines. See
`phase7f-fault-injection-report.md` for the original mock matrix and
`phase7f19-final-report.md` for the commands re-run in this closure.

## 4. Real-driver fault injection: why not

Safe, reproducible real fault injection on a shared T4 is not available:

- CUDA does not expose a public "fail this API call" hook.
- Context/device loss requires either a driver-level fault or killing the
  context, which can destabilize the runtime and other sessions.
- `compute-sanitizer` detects memory/race/sync defects; it does not inject
  driver API failures.
- A forced invalid handle would test the driver, not oxide-ml's safety logic,
  and could poison the process.

Therefore real driver fault injection is **SOURCE ONLY / NOT RUN**. This is a
declared limitation, not a PASS.

## 5. Residual risks

1. A driver error returned as an unexpected success code could bypass a
   branch; normal-path sanitizer runs do not cover this.
2. Quarantine is only as good as the completion event's state; a driver that
   reports a false completion would defeat it.
3. The real context-loss path has not been executed end-to-end.
4. Destroy-failure behavior is mock-verified, not driver-verified.

These risks are carried into the Phase 7F final judgment and are not hidden by
the normal-path PASS results.
