# Phase 7F Final Report

## Final judgment
**PARTIAL.** Persistent Arena, Arena Key registry, capacity/concurrent leasing,
EventPool/EventState, external-output async Drop, Read/Read/Write access,
deferred reclamation and a real benchmark are verified on T4. Event failure
injection/quarantine, AccessGuard-on-every-kernel, Dynamic Shape fallback,
true async concurrent submit, and the full 5-mode/statistics benchmark remain
incomplete.

## Verified PASS
| Item | Result |
|---|---|
| EventPool | PASS: 2 alloc, 842 reuse, 0 pending |
| EventState Unrecorded/Recorded/Completed | PASS |
| Persistent Arena 100 runs | PASS: 2 slot allocations, 99 reuse |
| Arena Key registry | PASS: same graph/plan returns same Arc |
| Arena capacity 1 Busy | PASS |
| Concurrent threads/capacity 2 | PASS (not proof of GPU overlap) |
| Async Drop external output | PASS |
| Pending release/reclamation | PASS: 1,048,576 B -> 0 |
| Access Read/Read/Write | PASS: 0 read waits, 2 write waits |
| Planned/normal output | PASS |
| CPU tests | PASS 87 |
| Clippy | PASS |
| Sanitizer memcheck | PASS 0 errors |

## Benchmark (T4, host wall clock)
100 iterations, warm-up 5.

| Mode | Allocation Count | Peak In-use | Peak Reserved | Median | P95 |
|---|---:|---:|---:|---:|---:|
| Normal + Pool | 11 cumulative | 15,876 B | 15,876 B | 138.9 us | 162.5 us |
| Planned + Pool | 未測定 (cumulative) | 15,876 B | 15,876 B | 125.0 us | 147.8 us |
| Persistent Arena / Same Stream | 2 slots | 15,876 B pool | 4,096 B arena | 133.1 us | 148.7 us |
| Persistent Arena / Multi Stream | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |
| Dynamic Shape + Fallback | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 |

## Not implemented / NOT RUN
- Event Record Failure injection; Failed Event quarantine
- EventPool Drop dedicated test
- AccessGuard type and integration into every kernel launch
- Executor Drop / Context lifetime dedicated test
- Dynamic Shape graph GPU test
- True async concurrent submit (two in-flight executions)
- Multi-stream benchmark, CUDA Event kernel timing, H2D/D2H split
- Driver memory comparison
- Statistics consistency unit tests
- racecheck / synccheck

## Known constraints
- AccessState is standalone; arena/executor kernels do not yet acquire guards.
- Arena output copy adds allocations; arena-backed slots are internal only.
- ArenaKey uses structural hashes; no full collision fallback beyond plan-slot comparison.
- Concurrent thread test is not proof of GPU overlap.
- Benchmark pool statistics are cumulative across modes.
