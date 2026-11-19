# Phase 7F-19 Backward Recovery Report

## 1. Result

**RECOVERED.** `backward_check_full` completed all five execution modes after
the constant-prefix copy fix:

```text
backward_check summary: PASS=65 FAIL=0 NOT APPLICABLE=5 cases=70
```

Graphs:

- Linear + MSE
- 2-layer MLP + MSE
- MSE-only
- ReLU activation + MSE
- Add + ReLU
- Unbroadcast bias
- LayerNorm (forward only; backward NOT APPLICABLE)

Each graph has forward-only and full training-graph measurements. Results:
`phase7f19-backward-results.json` / `.csv`.

## 2. Correctness

For every PASS case:

- forward output, scalar loss, input/weight/bias/parameter gradients compared
  against the CPU reference;
- no NaN / Inf;
- normal / planned / arena / async_planned / async_arena all compared;
- tolerances: forward `1e-4 + 3e-3|e|`, gradient `5e-3 + 5e-3|e|`,
  LayerNorm `1e-2 + 1e-2|e|`.

LayerNorm training is NOT APPLICABLE:

```text
operation layer_norm_backward is not implemented by the CUDA backend
```

LayerNorm forward is measured (5 modes). No backward was faked or added.

## 3. Timings (median GPU event ms)

### Forward-only

| Graph | normal | planned | arena | async_planned | async_arena |
|---|---:|---:|---:|---:|---:|
| linear_mse | 0.1348 | 0.1301 | 0.1413 | 0.1044 | 0.1498 |
| mlp2_mse | 0.2247 | 0.2080 | 0.2350 | 0.1683 | 0.2355 |
| mse_only | 0.0734 | 0.0735 | 0.0890 | 0.0689 | 0.0984 |
| relu_activation | 0.1007 | 0.0971 | 0.1290 | 0.0932 | 0.1702 |
| add_relu | 0.1029 | 0.0978 | 0.1308 | 0.0933 | 0.1372 |
| unbroadcast_bias | 0.0863 | 0.0863 | 0.1152 | 0.0800 | 0.1164 |
| layernorm | 0.0924 | 0.0929 | 0.1167 | 0.0836 | 0.1267 |

### Training graph (forward + backward)

| Graph | normal | planned | arena | async_planned | async_arena |
|---|---:|---:|---:|---:|---:|
| linear_mse | 0.3277 | 0.2698 | 0.3210 | 0.2437 | 0.3546 |
| mlp2_mse | 0.6042 | 0.4937 | 0.6061 | 0.4607 | 0.6224 |
| mse_only | 0.1576 | 0.1244 | 0.1564 | 0.1059 | 0.1391 |
| relu_activation | 0.2747 | 0.2113 | 0.2396 | 0.1980 | 0.2793 |
| add_relu | 0.1927 | 0.1516 | 0.1896 | 0.1428 | 0.2068 |
| unbroadcast_bias | 0.1561 | 0.1447 | 0.1745 | 0.1302 | 0.1964 |
| layernorm | N/A | N/A | N/A | N/A | N/A |

Forward-only and training are separate Cases; the difference is not presented
as a separately measured backward-only interval.

## 4. Recovery evidence

Before the fix, planned training graphs panicked with the constant-copy
assertion. After the fix:

```text
linear_mse forward=PASS train=PASS
mlp2_mse forward=PASS train=PASS
mse_only forward=PASS train=PASS
relu_activation forward=PASS train=PASS
add_relu forward=PASS train=PASS
unbroadcast_bias forward=PASS train=PASS
layernorm train NOT APPLICABLE
PASS=65 FAIL=0 NOT APPLICABLE=5
```

Every previously blocked mode (planned, arena, async_planned, async_arena) is
now measured for the six implemented training graphs.

## 5. Limits

- LayerNorm backward remains unimplemented; this is NOT APPLICABLE, not PASS.
- Async training in the linear-regression benchmark is N/A (optimizer needs
  unique device buffers); backward async is a pure graph execution and is
  measured.
- Real driver fault injection is not part of this benchmark.
