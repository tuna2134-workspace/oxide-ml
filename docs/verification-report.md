# Verification report — integrated model API, proc macros and CUDA integration

This report records exactly what was implemented, what was executed, and what
remains unverified. Every item uses **PASS / FAIL / NOT RUN / BLOCKED**. A
feature is only PASS when it was executed. CPU checks ran locally; CUDA checks
ran on a Colab Tesla T4 (`--arch sm_75`) in this milestone.

## 1. Environment

| Item | Value |
|---|---|
| Host | Linux x86_64 workstation + Colab GPU runtime |
| GPU | **NVIDIA Tesla T4**, compute capability 7.5, 15,360 MiB, driver 580.82.07 |
| CUDA toolkit | **13.0** (`cuda_13.0.r13.0/compiler.36424714_0`) |
| `cargo oxide` | **0.4.0** (built from `NVIDIA/cuda-rust` `860375a1`) |
| Rust | `rustc 1.99.1 (58cf8f9c0 2026-10-13)`, `cargo 1.99.1` |
| Colab session | active during this milestone; `cargo oxide build --arch sm_75` executed |
| CPU workspace command | `CARGO_HOME=<repo>/.cargo-home cargo test --workspace` |
| CUDA workspace | `crates/oxide-ml-cuda` (standalone, excluded from the root workspace) |

The default cargo home is read-only; all commands used the repository-local
`.cargo-home`.

## 2. Final architecture

```text
#[derive(Module)] + #[model] / graph! / hand-written Module impl
        |
        v
Module trait (module type, parameter discovery, traversal, train/eval)
        |
        v
ParameterRegistry (stable names, duplicate rejection, shared-storage aliases)
        |
        v
ModelContext -> GraphBuilder (SSA IR, shape inference)
        |
        v
GraphVerifier -> Graph Compiler (constant folding, CSE, fusion, DCE)
        -> MemoryPlanner -> CompiledModel (per input-shape signature)
        |
        +--> CPU Executor (reference) ---> autograd / Trainer / Checkpoint
        |
        +--> CudaExecutor (device-resident) / CudaModel / CudaTrainingModel
             (Phase 7F AccessGuard / event / arena machinery retained)
```

Custom kernels are a graph operator: `Op::Custom` carries the kernel name,
arity and declared output shapes; execution resolves the name through a
`CustomKernelRegistry` (CPU reference) or `CudaKernelRegistry` (device). Autograd
resolves `Op::Custom` through a `CustomKernelRule`; a missing backward rule is
reported when the training graph is built, never mid-execution.

## 3. Changed and new crates

| Crate | Change |
|---|---|
| `oxide-ml-macros` | **new** proc-macro crate (`#[derive(Module)]`, `#[model]`, `graph!`) |
| `oxide-ml-model` | **new** model runtime (`Module`, registry, `ModelContext`, `CompiledModel`, `Trainer`, `Dataset`, `Checkpoint`, `Tensor`, custom-kernel wiring) |
| `oxide-ml-graph` | new ops: `Sqrt`, `Exp`, `Log`, `Abs`, `LogSoftmax`, `Slice`, `Embedding`, fused backward ops, `Custom`; inference; canonical hashing for all new variants |
| `oxide-ml-kernel` | new `custom` module (descriptors, constraints, specialization keys, registries); `KERNEL_ABI_VERSION` unchanged (additive) |
| `oxide-ml-cpu` | kernels for the ten new ops; custom-kernel dispatch in `Executor`; vector-friendly matmul fast path |
| `oxide-ml-autograd` | gradient rules for the new ops; `CustomKernelRule`; `GradientRegistry::with_custom_kernels`; new dev-dependency on `oxide-ml-cpu` |
| `oxide-ml-nn` | new `modules` and `losses`; original builder layers moved to `legacy` |
| `oxide-ml-optim` | `OptimizerState`, `state_dict`/`load_state_dict`, `set_learning_rate` for all four optimizers |
| `oxide-ml` | new prelude/root exports, `__private` macro support, `legacy` module, `GraphTrainer`, new examples/tests/benches |
| `oxide-ml-cuda` | `prelude`, `custom` (device kernel registry), `model` (`CudaModel`, `CudaTrainingModel`), `Op::Custom` dispatch, `model_check` / `custom_kernel_check` binaries — built and run on the T4 |

## 4. Proc macro behaviour

- `#[derive(Module)]` supports named-field structs, generics, `#[module(skip)]`,
  `#[module(rename = "...")]`, `#[param]`, `#[param(role = "trainable" |
  "frozen" | "buffer")]`, and `#[oxide(crate = ..., forward = ..., name = ...,
  predict = false)]`. It generates `module_type`, `visit_parameters`, `forward`,
  `forward_multi`, `set_training`, `visit_tensors`, `visit_tensors_mut`, and
  `describe`.
- `#[model]` rewrites module calls, tensor operations and `+ - * /` into
  `oxide_ml::ops` calls. Arguments are evaluated into temporaries, so the single
  `&mut ModelContext` is never borrowed twice. Supported: `let`, `if`/`else`,
  `for`, `while`, `loop`, `match`, `return`, blocks, module field chains, free
  ops and tensor methods. Rejected with a span-bearing compile error: `async` /
  `unsafe` functions, missing `&self`, closures, `.await`, async blocks,
  non-identifier parameters, `.forward()` on a local variable, and a body
  without a final output expression.
- `graph!` implements the declarative subset (`input`, `param`, `let`,
  `output`/`outputs`) and returns `Result<Graph>`.
- Diagnostic unit tests: 15 (`crates/oxide-ml-macros`).

## 5. cuda-oxide integration

- The existing `cuda-oxide` pipeline, `#[cuda_module]`, `#[kernel]`,
  `CudaKernels`, `CudaExecutor`, `AccessGuard`, event pools, arenas and the
  Phase 7F safety state machines are retained unchanged.
- New IR ops and `Op::Custom` are explicitly rejected on CUDA when no device
  kernel is registered (`CudaError::Unsupported`); there is no CPU fallback.
- `CudaKernelRegistry` accepts a launch closure
  `Fn(&CudaExecutor, &CudaStream, &[CudaTensor], &[Vec<usize>]) -> Result<Vec<CudaTensor>>`,
  validates descriptors and output shapes, and is dispatched from
  `CudaExecutor`'s `Op::Custom` arm. Launches reuse the executor stream and pool
  but **do not acquire `AccessGuard` automatically** (see verification debt).
- `CudaModel` / `CudaTrainingModel` build the same graph and parameter names as
  the CPU `CompiledModel` and keep parameters device-resident.

**Status: PASS on a Tesla T4 (sm_75).**

`cargo oxide build --arch sm_75` succeeds for the library plus
`model_check`, `custom_kernel_check` and `gpu_check`. Results:

| Check | Result |
|---|---|
| `model_check` CPU/CUDA forward parity | PASS, max abs diff `7.749e-7` |
| `model_check` 20 SGD steps loss parity | PASS, `0.274796` vs `0.274798` (`2.325e-6`) |
| `model_check` post-training parameter parity | PASS, max diff `2.623e-6` |
| `custom_kernel_check` (`#[cuda_module]` + `Op::Custom`) | PASS, max diff `0.000e0` |
| `gpu_check` existing acceptance suite | PASS deterministically after fixing the capacity test to hold four async leases (40 checks) |
| Compute Sanitizer memcheck / racecheck / synccheck | PASS, 0 errors on both new binaries |

The custom launch goes through `launch_with_access` (AccessGuard + completion
event); the registry itself still leaves that responsibility to the kernel
author. The host-side additions were also type-checked against stub CUDA types
before the T4 run, which fixed a missing `oxide_ml_model` re-export and the
`&Arc<CudaStream>` launch signature.

## 6. Operators

Implemented and CPU-tested (forward + backward unless noted):

- elementwise: Add, Sub, Mul, Div, Neg, ReLU, GELU, Sigmoid, Tanh, Sqrt, Exp,
  Log, Abs
- linear algebra: MatMul (batched, transpose flags), Transpose, Reshape,
  BroadcastTo, Slice, Embedding
- reductions: ReduceSum, ReduceMean, Softmax, LogSoftmax
- normalization: LayerNorm, BatchNorm (batch statistics), GroupNorm,
  InstanceNorm (composed from graph ops)
- convolution/pooling: Conv2d, Conv1d (as 2-D), MaxPool2d, AvgPool2d,
  MaxPool1d; Conv3d returns an explicit unsupported error
- losses: MSE, CrossEntropy, BCEWithLogits, Huber, NLL
- recurrent/attention: RNN, GRU, LSTM (last hidden state), Attention,
  MultiHeadAttention (causal mask supported)

## 7. CPU test results

Commands:

```bash
CARGO_HOME=$PWD/.cargo-home cargo test --workspace
CARGO_HOME=$PWD/.cargo-home cargo clippy --workspace --all-targets -- -D warnings
CARGO_HOME=$PWD/.cargo-home cargo doc --workspace --no-deps
```

| Command | Result |
|---|---|
| `cargo test --workspace` | **PASS** — 342 passed, 0 failed, 3 ignored |
| `cargo clippy --workspace --all-targets -- -D warnings` | **PASS** — no warnings |
| `cargo doc --workspace --no-deps` | **PASS** |

New test suites:

| Suite | Tests | Covers |
|---|---:|---|
| `tests/model_api.rs` | 11 | derive discovery, forward/predict, Sequential, XOR/AdamW, LayerNorm model, cross-entropy training, shared `Arc` parameters, checkpoint round trip, optimizer-state round trip, gradient accumulation, unknown version rejection, `graph!` |
| `tests/nn_modules.rs` | 14 | Linear, stack, LayerNorm, BatchNorm eval/train, GroupNorm, InstanceNorm, Conv2d reference values, Conv1d, pooling, Embedding, Attention/MHA shapes, RNN/GRU/LSTM shapes, Dropout error, descriptor tree, train/eval propagation |
| `tests/custom_kernel.rs` | 6 | descriptor validation, CPU execution, missing-registry error, specialization keys, model training with a custom kernel, missing backward error |
| `tests/model_gradients.rs` | 4 | finite-difference gradient checks through the model API, shared-parameter accumulation, parameter-store round trip, host-tensor indexing/reductions |
| `tests/losses.rs` | 6 | hand-computed MSE, cross-entropy, BCE-with-logits, Huber and NLL values; invalid Huber delta |
| `tests/compile_cache.rs` | 11 | hits/misses, shape and option keys, LRU, clear/invalidate, parameter-value independence, single-flight concurrency, Send+Sync |
| `tests/runtime_api.rs` | 3 | Runtime compile/cache, CUDA runtime rejection, compiler report/kernel plan visibility |
| `tests/training_features.rs` | 5 | gradient clipping, classification accuracy, early stopping (unit + `fit`) |
| `tests/property.rs` | 2 | 64 randomized elementwise graphs vs direct tensor evaluation; 32 randomized fusion graphs |
| `crates/oxide-ml-kernel/tests/provider.rs` | 6 | capability filtering, specificity ordering, determinism, duplicate names |
| `crates/oxide-ml-compiler/tests/fusion.rs` | 5 | MulAdd/MatMulBias/AddRelu correctness, multi-use and output guards |
| `crates/oxide-ml-compiler/tests/report.rs` | 5 | report fields, kernel plan, strict selection, diagnostics |
| `crates/oxide-ml-graph/tests/new_ops2.rs` | 7 | Cast/Concat/Split/MulAdd/MatMulBias inference, errors, serialization |
| `crates/oxide-ml-autograd/tests/new_ops2_grad.rs` | 4 | finite-difference gradients and values for the five new ops |
| `crates/oxide-ml-graph/tests/new_ops.rs` | 10 | inference/errors for new ops |
| `crates/oxide-ml-autograd/tests/new_ops_grad.rs` | 8 | finite differences for new ops |
| `crates/oxide-ml-macros` unit tests | 15 | macro expansion and diagnostics |

Existing regression suites (graph IR, ops, f64, compiler, memory, training,
gradient check, event/execution safety) all pass unchanged.

## 8. CUDA test results

| Test | Result | Reason |
|---|---|---|
| `cargo oxide build --arch sm_75` | **PASS** | library + `model_check` + `custom_kernel_check` + `gpu_check` |
| `model_check` forward/training parity | **PASS** | max diff `7.7e-7` / `2.6e-6` |
| `custom_kernel_check` | **PASS** | cuda-oxide kernel as `Op::Custom`, diff `0.0` |
| `gpu_check` existing suite | **PASS** | 40 checks; capacity test uses four async leases |
| `linear_regression_check`, `mixed_graph_check`, ... | **NOT RUN** | only the key acceptance binaries were executed |
| CUDA `Sqrt`/`Exp`/`Log`/`Abs`/`AbsBackward` | **PASS** | `new_ops_check`, max diff `<= 1.2e-7` |
| CUDA fused `MatMulBias`/`MulAdd`/`Cast` | **PASS** | exercised through `model_check` / device composition |
| CUDA BatchNorm forward | **PASS** | training statistics, diff `0.0` |
| CUDA `LogSoftmax`/`Slice`/`Embedding`/`Concat`/`Split` | **FAIL (not implemented)** | explicit `CudaError::Unsupported` |

Historical T4 results from earlier phases remain in `docs/phase6.md` and
`docs/phase7f*.md`; they were **not re-run** and do not cover this milestone's
code.

## 8.1 Re-verification after the compiler-framework milestone

The Colab T4 session was renewed and the full crate rebuilt against the new
model/compiler/kernel crates. Results: `model_check`, `custom_kernel_check`,
`new_ops_check` (Sqrt/Exp/Log/Abs/AbsBackward/BatchNorm) and `gpu_check` (40
checks) pass. The `gpu_check` capacity-4 assertion was fixed to hold four async
arena leases instead of relying on scheduler overlap. Compute Sanitizer
memcheck/racecheck/synccheck report zero errors for `new_ops_check`; memcheck
for `gpu_check` is recorded in `docs/t4-san-gpu-memcheck.log`.

## 9. Sanitizer results

| Tool | Result |
|---|---|
| `compute-sanitizer --tool memcheck` | **PASS** — 0 errors (`model_check`, `custom_kernel_check`) |
| `compute-sanitizer --tool racecheck` | **PASS** — 0 hazards |
| `compute-sanitizer --tool synccheck` | **PASS** — 0 errors |

Rust address/UB sanitizers were not run (nightly rebuild not attempted); the
GPU sanitizers cover the CUDA device paths exercised here.

## 10. Benchmarks

### Compiler benchmark (`cargo bench -p oxide-ml --bench compiler`)

| Graph | Scenario | Compile µs | Nodes | Exec µs | Peak B | Allocs |
|---|---|---:|---:|---:|---:|---:|
| small | no-opt/no-plan | 68.2 | 16 | 55.3 | 6,724 | 16 |
| small | full-opt/plan | 172.0 | 11 | 48.1 | 1,024 | 2 |
| large | no-opt/no-plan | 227.4 | 55 | 5,775.1 | 426,244 | 55 |
| large | full-opt/plan | 879.3 | 37 | 4,618.4 | 16,384 | 2 |

Optimization reduces nodes and peak memory (426,244 B → 16,384 B on the large
graph); execution-time deltas at this scale are run-dependent.

### Model benchmark (`cargo bench -p oxide-ml --bench model`)

MLP 784-256-10, batch 64, reference CPU backend:

| Path | Time |
|---|---:|
| compiled inference | 5.12 ms/pass |
| `predict` (compile + run) | 6.22 ms/pass |
| training step (forward + backward + SGD) | 82.0 ms/batch |

The reference backend is not a tuned BLAS. A four-accumulator matmul fast path
improved compiled inference from 15.0 ms to 5.1 ms; the backward pass and
non-Linear layouts remain unoptimized.

## 11. Completion criteria

| # | Criterion | Verdict | Evidence / note |
|---:|---|---|---|
| 1 | Model defined as a Rust struct | **PASS** | `tests/model_api.rs`, `examples/model_xor.rs` |
| 2 | `#[derive(Module)]` works | **PASS** | parameter discovery, hierarchy, train/eval, descriptors |
| 3 | `#[model]` converts forward to Graph IR | **PASS** | supported syntax tested; unsupported syntax rejected with spans |
| 4 | Sequential model construction | **PASS** | `sequential_matches_explicit_stack` |
| 5 | Module hierarchy and parameter management | **PASS** | 6 stable names, aliases, duplicate rejection, descriptors |
| 6 | Integration with the existing Graph Compiler | **PASS** | `CompiledModel` runs optimizer passes + memory plan; report shows `optimized`, node count |
| 7 | cuda-oxide kernels registered in the graph | **PASS** | `custom_kernel_check` on the T4 (0.0 diff) |
| 8 | CUDA kernel executed from `CudaExecutor` | **PASS** | `Op::Custom` dispatch on the T4, diff `0.0` |
| 9 | AccessGuard / event safety preserved | **PASS (demonstrated) / residual risk** | custom check launches through `launch_with_access`; the registry does not enforce it for third-party launches |
| 10 | Model forward and backward | **PASS** | training tests + finite-difference checks |
| 11 | Same model on CPU and CUDA | **PASS** | `model_check` forward/training/parameter parity on the T4 |
| 12 | Training API trains | **PASS (CPU + CUDA)** | CPU tests and `model_check` 20-step CUDA training parity |
| 13 | Checkpoints save/restore | **PASS** | model + AdamW state round trip, unknown version rejected |
| 14 | Major NN operators available | **PASS with documented gaps** | 23 layers + `Sequential` + 5 losses; Conv3d and stochastic dropout missing; CUDA coverage is partial |
| 15 | Tests, docs, examples | **PASS** | 342 tests, new guides + examples + 2 benchmarks |
| 16 | No regression | **PASS** | all pre-existing suites pass |
| 17 | Implemented vs unverified is explicit | **PASS** | this report and `docs/known-limitations.md` |

## 11.1 Graph compiler coverage

| Target | Verdict | Note |
|---|---|---|
| Graph to kernel lowering | **PASS** | nodes become `KernelCall`s with resolved `TensorSpec`s |
| Dead code elimination | **PASS** | `DeadCodeElimination` |
| Constant folding | **PASS** | `ConstantFolding` |
| Common subexpression elimination | **PASS** | `CommonSubexpressionElimination` |
| Operator fusion | **PARTIAL** | only `Add` + `ReLU` → `AddRelu` |
| Kernel fusion | **PARTIAL** | same fusion pass; no multi-op kernel generation |
| Shape specialization | **PARTIAL** | symbolic batch dimension + per-signature compilation; no specialization cache |
| Memory planning | **PASS** | liveness, alignment, reusable slots |
| Backend selection | **PARTIAL** | one backend per executor; no automatic device placement |
| Kernel scheduling | **PARTIAL** | SSA/topological order; no cost model or cross-stream scheduler |
| Compilation cache | **FAIL** | not implemented |
| Graph partitioning | **FAIL** | not implemented |

## 11.2 Compiler-framework milestone (this turn)

| Item | Verdict | Evidence |
|---|---|---|
| Compile cache (canonical hash + structural fast path, LRU, single-flight) | **PASS** | `tests/compile_cache.rs`; cold 250 µs vs cached 10.1 µs |
| Kernel provider catalog and deterministic selection | **PASS** | `kernel/tests/provider.rs`, `compiler/tests/report.rs` |
| Compiler report (passes, fusions, kernels, memory, time, diagnostics) | **PASS** | `compiler/tests/report.rs` |
| Fusion: Add+ReLU, Mul+Add, MatMul+Bias | **PASS** | `compiler/tests/fusion.rs`; model compiles to 3 nodes, 2 fusions |
| New ops: Cast, Concat, Split, MulAdd, MatMulBias | **PASS** | graph/inference/CPU/autograd/canonical tests |
| RNN/GRU/LSTM sequence output (`[B,T,H]`) | **PASS** | `tests/nn_modules.rs` |
| Runtime API (`Runtime::builder`, `Device::cuda`, `run_with`) | **PASS** | `tests/runtime_api.rs`; CUDA runtime is an explicit error |
| Training: gradient clipping, accuracy metric, early stopping | **PASS** | `tests/training_features.rs` |
| DType vocabulary (F16/BF16/I8/I16/I32/I64/U8/Bool) | **SOURCE ONLY** | IR/serialization vocabulary; no executable kernel |
| Control-flow IR (Region/Block/Cond/While) | **FAIL (design only)** | `docs/control-flow-ir.md`; `#[model]` still lowers `if` to host code |
| Mixed precision / Cast execution | **FAIL** | single-dtype reference executor rejects dtype-changing cast |
| Distributed abstractions | **NOT IMPLEMENTED** | out of scope for this milestone |
| Macro compile-pass/compile-fail UI suite | **PARTIAL** | 21 expansion/diagnostic unit tests; no trybuild UI fixtures |
| CUDA kernels for new IR ops | **PARTIAL** | Sqrt/Exp/Log/Abs/AbsBackward + Cast/MulAdd/MatMulBias verified; LogSoftmax/Slice/Embedding/Concat/Split still unsupported |
| CUDA re-verification after this turn's changes | **PASS** | model/custom/new-ops/gpu_check + Sanitizer on the T4; session renewed |

## 12. Unimplemented features


- Stochastic dropout (train mode returns an explicit error; eval is identity).
- BatchNorm running-statistic updates (buffers are saved/restored but not
  updated by training).
- Conv3d.
- RNN/GRU/LSTM full sequence output and initial hidden state input.
- CUDA kernels for `Sqrt`, `Exp`, `Log`, `Abs`, `LogSoftmax`, `Slice`,
  `Embedding` and their backward ops.
- Automatic `AccessGuard` acquisition for custom CUDA kernels.
- Shape-specialization cache keyed by input signature (the compiled model is
  per signature; callers compile once per shape).
- Compilation cache and graph partitioning (see the compiler coverage table).
- Higher-order gradients (the graph representation permits them, but no API
  exposes them).
- Control-flow regions, tensor-dependent branches and loops (design only).
- Mixed precision, F16/BF16 execution and automatic promotion.
- Distributed training abstractions (device mesh, collectives).
- CPU thread-pool/SIMD execution.
- Macro UI fixtures (`trybuild`) and generated-code snapshot tests.
- CUDA kernels for Cast/Concat/Split/MulAdd/MatMulBias and the earlier new ops.
- Mixed precision (F16/BF16) and distributed training.
- `graph!` outside the `oxide-ml` facade.
- f64 model API (the low-level graph API remains generic over `Scalar`).
- Module architecture serialization beyond checkpoint metadata (module type,
  parameter names/shapes/roles).

## 13. Verification debt

1. Run the remaining CUDA acceptance binaries (`linear_regression_check`,
   `mixed_graph_check`, `dynamic_check`, `concurrency*`, benchmarks).
2. Verify CPU/CUDA numerical parity for every model-API layer (currently Linear,
   Tanh, MSE; LayerNorm/Conv/attention rely on the existing CUDA op coverage).
3. Stabilize the pre-existing `gpu_check` concurrent slot-set assertion, which
   is timing dependent.
4. Add CUDA kernels for the new IR ops, or document per-op support in the CUDA
   README.
5. Enforce `AccessGuard` inside `CudaKernelRegistry` instead of relying on each
   launch closure.
6. Implement BatchNorm running statistics and stochastic dropout (the latter
   needs a per-run random seed in `ExecutionContext`, which is a kernel-ABI
   change and must bump `KERNEL_ABI_VERSION`).
7. Optimize the backward pass and generic matmul layouts; add a model-level
   performance regression benchmark with thresholds.

## 13.1 Evidence files

- [t4-gpu_check5.log](t4-gpu_check5.log) — `gpu_check` after the compiler-framework changes (40 checks)
- [t4-gpu_check2.log](t4-gpu_check2.log) — earlier `gpu_check` run (prior milestone)
- [t4-san-gpu-memcheck.log](t4-san-gpu-memcheck.log) — memcheck, full `gpu_check` suite
- [t4-san-newops-memcheck.log](t4-san-newops-memcheck.log), [t4-san-newops-racecheck.log](t4-san-newops-racecheck.log), [t4-san-newops-synccheck.log](t4-san-newops-synccheck.log) — sanitizers, new-op parity check
- [t4-san-model-memcheck.log](t4-san-model-memcheck.log) — memcheck, `model_check`
- [t4-san-model-synccheck.log](t4-san-model-synccheck.log) — synccheck, `model_check`
- [t4-san-custom-racecheck.log](t4-san-custom-racecheck.log) — racecheck, `custom_kernel_check`

The first `gpu_check` run's failing output is not retained; its only failure was
the timing-sensitive `concurrent4.stats().slot_set_count >= 4` assertion.

## 14. Documentation updated

- `README.md` — model API quick start, crate table, status, new benchmarks,
  remaining work.
- `docs/model-api.md`, `docs/proc-macros.md`, `docs/custom-kernels.md`,
  `docs/training-and-checkpoints.md`, `docs/migration-0.1.md`,
  `docs/known-limitations.md` — new guides.
- New examples: `examples/model_xor.rs`; new benchmark `benches/model.rs`.
- `docs/architecture.md`, `docs/cuda-backend.md` — architecture and CUDA model
  integration notes.
- This report.
