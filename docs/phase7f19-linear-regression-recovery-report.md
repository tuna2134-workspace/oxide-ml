# Phase 7F-19 Linear Regression Recovery Report

## 1. Result

**RECOVERED.** After the constant-prefix copy fix, the full linear-regression
matrix completed on the T4:

```text
linear_regression_check summary: PASS=99 FAIL=0 NOT APPLICABLE=23 cases=122
```

- 3 datasets: small N=16/F=1, medium N=64/F=4, large N=256/F=8.
- batches 1, 4, 8, 32 where batch <= N.
- normal / planned / arena executed for every combo.
- async_planned / async_arena measured for forward-only; async **training** is
  NOT APPLICABLE because applying `CudaSgd` requires unique device buffers and
  a host synchronization, which would make the step synchronous. No async
  training number is fabricated.
- batch 32 on the small dataset is NOT APPLICABLE (batch > N).

Results: `phase7f19-linear-regression-results.json` / `.csv`.

## 2. Correctness

Every executed combo used the same deterministic data, initial parameters,
learning rate (0.2), and 100 steps for CPU reference and all CUDA modes.

| Check | Result |
|---|---|
| Final loss vs CPU | PASS for all 99 cases |
| Weight / bias vs CPU | PASS |
| Parameter gradients vs CPU | PASS |
| NaN / Inf | none |
| Convergence | PASS (`final_loss < 0.5 * initial_loss`) |
| Cross-mode consistency | PASS |
| Stability | PASS (deterministic inputs, repeated runs) |

Tolerances: loss `1e-3 + 1e-3|e|`, weight/bias/gradient `5e-3 + 5e-3|e|`,
forward `1e-4 + 3e-3|e|`, cross-mode `1e-4 + 1e-4|e|`.

## 3. Timings (median GPU event ms, training step)

| Combo | normal | planned | arena | async_planned | async_arena |
|---|---:|---:|---:|---:|---:|
| small_b1 | 0.1119 | 0.1133 | 0.1311 | 0.0910 | 0.1344 |
| small_b4 | 0.1196 | 0.1167 | 0.1297 | 0.0948 | 0.1356 |
| small_b8 | 0.1179 | 0.1147 | 0.1326 | 0.0903 | 0.1388 |
| medium_b1 | 0.1183 | 0.1147 | 0.1310 | 0.0927 | 0.1372 |
| medium_b4 | 0.1167 | 0.1158 | 0.1311 | 0.0942 | 0.1424 |
| medium_b8 | 0.1204 | 0.1101 | 0.1309 | 0.0922 | 0.1352 |
| medium_b32 | 0.1198 | 0.1183 | 0.1290 | 0.0901 | 0.1372 |
| large_b1 | 0.1209 | 0.1213 | 0.1326 | 0.0946 | 0.1454 |
| large_b4 | 0.1147 | 0.1145 | 0.1302 | 0.0942 | 0.1352 |
| large_b8 | 0.1204 | 0.1188 | 0.1311 | 0.0922 | 0.1380 |
| large_b32 | 0.1178 | 0.1152 | 0.1324 | 0.0942 | 0.1454 |

Arena is consistently slower because of slot leasing and cross-stream event
waits. Async planned shows a lower host/GPU span; its wait is measured
separately in the JSON.

## 4. Measurement separation

The binary records, per mode:

- forward-only graph timing;
- full training step timing (forward + loss + backward + optimizer update);
- optimizer-step-only timing with fixed gradients, once per combo, excluded
  from the training total;
- end-to-end host time.

The training total includes the optimizer update, so the reported step time is
not double-counted as forward-only.

## 5. Recovery evidence

Before the fix (Phase 7F-19 first closure):

```text
small_b1 normal loss=0.000000 correctness=PASS
thread 'main' panicked ... host-to-device copy length mismatch left: 16 right: 1
```

After the fix:

```text
small_b1 normal ... PASS
small_b1 planned ... PASS
small_b1 arena ... PASS
...
large_b32 arena loss=0.000001 correctness=PASS
PASS=99 FAIL=0 NOT APPLICABLE=23
```

The first failing case (small N=16 F=1 batch=1 normal) and all planned/arena
cases now complete.

## 6. Notes / limits

- 23 NOT APPLICABLE = 22 async-training cases + 1 batch > N case.
- The source includes `oxide-ml-optim/src/lib.rs` via `#[path]` because
  `oxide-ml-optim` is a dev-dependency of the CUDA crate; this uses the real
  optimizer rather than a copy.
- No safety API was weakened. Planned/arena use the same executor paths as
  before.
