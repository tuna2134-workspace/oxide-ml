# Compiler benchmark

Run from the repository root:

```bash
cargo bench -p oxide-ml --bench compiler
```

The harness is plain `std::time::Instant`; there is no benchmark dependency.
It builds a small (3-layer, width 16, batch 8) and a large (16-layer, width 64,
batch 32) MLP with two constant tensors and an Add+ReLU pattern.

Scenarios:

- `no-opt/no-plan`: all passes disabled, full executor.
- `full-opt/plan`: default pass order + memory planner + planned executor.
- `constant-fold`: only constant folding (memory planning enabled so the
  execution comparison is like-for-like).
- `fusion`: only Add+ReLU fusion.

Reported metrics: compiler wall time, node count, non-constant node count,
average execution time over 20 runs, planned peak bytes, and planned allocation
count. For `no-opt/no-plan`, peak bytes and allocations are conservative
estimates from the value table/node count.

Latest sample on this CPU-only workstation (release):

```
== small (3 layers, width 16, batch 8) ==
baseline nodes=16 estimated_value_bytes=6724
scenario             compile_us      nodes    non_const    exec_us   peak_B  allocs
no-opt/no-plan             76.7         16           14      40.97     6724      16
full-opt/plan             146.3         11           10      32.14     1024       2
constant-fold              46.1         16           13      34.22     1024       2
fusion                     52.4         13           11      37.18     1536       3

== large (16 layers, width 64, batch 32) ==
baseline nodes=55 estimated_value_bytes=426244
scenario             compile_us      nodes    non_const    exec_us   peak_B  allocs
no-opt/no-plan            161.4         55           53    4232.23   426244      55
full-opt/plan             750.0         37           36    3952.77    16384       2
constant-fold             257.5         55           52    4051.14    16384       2
fusion                    156.8         39           37    4093.51    24576       3
```

These numbers are machine-dependent and should be regenerated rather than
treated as absolute targets.
