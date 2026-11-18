# Phase 7F-18 Concurrency Results

## Environment

- Tesla T4 / sm_75 / CUDA 13.0
- 2 streams, independent 1024x1024 GEMM buffers
- 4 GEMM launches per stream
- CudaEvent timestamps

## Raw

```text
serial_span_ms=103.001
parallel_span_ms=90.894
serial_submit_ms=0.077
parallel_submit_ms=0.072
a_ms=81.834
b_ms=90.891
overlap_ms=81.831
speedup=1.133
PASS actual GPU overlap
```

## Interpretation

- Serial: Stream B waits on Stream A's end event.
- Parallel: no cross-stream wait; Stream B starts while A is running.
- Overlap is measured from GPU event timestamps, not host wall clock.
- Overlap < full duration because GEMM is resource-bound; still 81.8 ms
  concurrent execution was observed.

## Other Checks

- 2 threads / 2 streams / shared persistent arena: PASS.
- Async planned / arena execution: PASS.
- Executor drop / execution drop / external output: PASS.
- Arena capacity 1 busy / capacity 2 parallel lease: PASS.

## NOT RUN

- 4 streams / 4 executions
- concurrency_check racecheck (runtime)
