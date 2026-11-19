# Benchmark report

Measurements are release builds on the reference CPU workspace. They are
wall-clock host timings; there is no GPU run in this turn (the Colab session
expired after the earlier T4 verification).

## Methodology

- `cargo bench -p oxide-ml --bench compiler` — graph compile/execute scenarios
  (small: 3 layers/width 16/batch 8; large: 16 layers/width 64/batch 32).
- `cargo bench -p oxide-ml --bench model` — MLP 784-256-10.
- Timings are averaged over the iterations in the harness; runs vary by machine.
- "Compile" is the optimizer + memory planner; "compiled run" excludes
  parameter upload. Parameter binding is a host clone in the reference backend.

## Compiler

| Graph | Scenario | Compile µs | Nodes | Exec µs | Peak B | Allocs |
|---|---|---:|---:|---:|---:|---:|
| small | no-opt/no-plan | 68.2 | 16 | 55.3 | 6,724 | 16 |
| small | full-opt/plan | 172.0 | 11 | 48.1 | 1,024 | 2 |
| large | no-opt/no-plan | 227.4 | 55 | 5,775.1 | 426,244 | 55 |
| large | full-opt/plan | 879.3 | 37 | 4,618.4 | 16,384 | 2 |

Optimization trades compile time for node count and peak memory; execution-time
deltas at this scale are run-dependent.

## Model API and compile cache

MLP 784-256-10, batch 64:

| Path | Time |
|---|---:|
| compile (cold) | 250 µs |
| compile (cached structural hit) | 10.1 µs |
| compiled inference | 5.0–5.3 ms/pass |
| `predict` (cached graph, parameter binding + run) | 6.1 ms/call |
| training step (forward + backward + SGD) | 71–73 ms/batch |

Batch scaling (compiled inference):

| Batch | µs/pass |
|---:|---:|
| 1 | 0.9–1.0 ms |
| 16 | 1.7–1.8 ms |
| 64 | 4.9–5.0 ms |
| 256 | 17.4–18.8 ms |

The generated model graph compiles to 3 nodes with 2 fusions
(`MatMulBias`, `AddRelu`) and a 128 KiB peak memory plan.

## Interpretation

- The cache removes compilation from the repeated-`predict` path (250 µs →
  10 µs); inference itself is dominated by the reference matmul and parameter
  cloning. This backend is a correctness oracle, not a tuned BLAS.
- The backward pass is roughly an order of magnitude slower than the forward
  pass; it has not been optimized.
- Small-graph compile overhead is tens to hundreds of microseconds; large-graph
  optimization is the expensive case.

## Not measured

- GPU kernel latency, GPU event time, multi-stream scaling, batch scaling on
  CUDA, fusion performance on CUDA (session expired; earlier T4 results are in
  the historical phase reports and are not comparable).
- CPU thread-pool/SIMD scaling (not implemented).
