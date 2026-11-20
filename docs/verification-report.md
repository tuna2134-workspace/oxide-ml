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
| `oxide-ml-safetensors` | **new** memory-mapped safetensors reader (F32/F16/BF16 → host f32) |
| `oxide-ml-gemma` | **new** Gemma 4 text config/weight mapping/modules and `gemma4_infer` |

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
| `cargo test --workspace` | **PASS** — 394 passed, 0 failed, 3 ignored |
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
| `tests/vision_models.rs` | 2 | MobileNetV4 forward shapes/parameter count, static-batch CNN training step |
| `tests/safetensors_export.rs` | 3 | module round trip, tied alias metadata, parameter-store export |
| `crates/oxide-ml-safetensors/tests/writer.rs` | 4 | writer layout, dtypes, metadata, duplicate/shape validation |
| `tests/batch_norm_calibration.rs` | 2 | probes, running-stat updates, eval consistency |
| `tests/optional_inputs.rs` | 2 | `Option<Tensor<f32>>` slots, absent/present calls, declarative `forward_slots` |
| `tests/module_system.rs` | 12 | `ModuleList` paths/forward, nested module loops, enum variants/body marker, tied head aliasing/parity, tuple structs, multiple inputs/outputs |
| `tests/kv_cache.rs` | 7 | append/read, capacity growth, shape errors, shared KV, invalid layouts, reset/truncate, independent sessions |
| `tests/transformer_components.rs` | 10 | RMSNorm, gated MLP, RoPE, masks, rotate-half, GQA, soft MoE forward and router gradients |
| `crates/oxide-ml-gemma/tests/streaming_sessions.rs` | 3 | prefill+decode vs full forward (`1e-5`), session replay/reset, concurrent sessions on threads |
| `tests/paged_kv_cache.rs` | 6 | block tables, gather order, capacity, sharing, truncate, session independence |
| `tests/half_precision.rs` | 2 | F16 and BF16 graph execution through the generic `Scalar` executor |
| `tests/compile_fail.rs` | 7 fail + 2 pass fixtures | trybuild diagnostics (unit struct, enum variants, skip+param, local forward, closure, missing output) plus compile-pass declarative/multi-IO models |
| `tests/training_features.rs` | 5 | gradient clipping, classification accuracy, early stopping (unit + `fit`) |
| `tests/property.rs` | 2 | 64 randomized elementwise graphs vs direct tensor evaluation; 32 randomized fusion graphs |
| `crates/oxide-ml-kernel/tests/provider.rs` | 6 | capability filtering, specificity ordering, determinism, duplicate names |
| `crates/oxide-ml-compiler/tests/fusion.rs` | 5 | MulAdd/MatMulBias/AddRelu correctness, multi-use and output guards |
| `crates/oxide-ml-compiler/tests/report.rs` | 5 | report fields, kernel plan, strict selection, diagnostics |
| `crates/oxide-ml-graph/tests/new_ops2.rs` | 7 | Cast/Concat/Split/MulAdd/MatMulBias inference, errors, serialization |
| `crates/oxide-ml-autograd/tests/new_ops2_grad.rs` | 4 | finite-difference gradients and values for the five new ops |
| `crates/oxide-ml-graph/tests/new_ops.rs` | 10 | inference/errors for new ops |
| `crates/oxide-ml-autograd/tests/new_ops_grad.rs` | 8 | finite differences for new ops |
| `crates/oxide-ml-macros` unit tests | 39 | macro expansion and diagnostics (structs, tuple structs, enums, loops, multi-IO, errors) |

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
| CUDA `Slice`/`Embedding`/`Concat`/rank-3 MatMul | **PASS** | A100 parity `1.609e-6`; sanitizer 0 errors |
| CUDA `Split` / `LogSoftmax` | **PASS** | synthetic graph parity `1.192e-7`; sanitizer 0 errors |
| CUDA fused `MoE` | **PASS** | tiny MoE parity `1.162e-6`; sanitizer 0 errors |
| CUDA generation (recompute) | **PASS** | E2B `Hello` + 1 greedy token on A100 |
| CUDA KV-cache decode | **PASS** | A100, ≈0.30 s/token after CPU prefill; 4 tokens 1.188 s |
| Full E2B f32 on CUDA | **PASS** | A100 40 GB, 4.63 B params, top-5 matches CPU |
| CUDA `LogSoftmax`/`Split`/`MoE` | **PASS** | implemented and A100-verified |

The CUDA kernels were not modified in the declarative-architecture revision;
the A100 results above were recorded before it, and the CPU-side Gemma checks
were re-run after the component migration (`2.999e-7` dense, `3.427e-7`
streaming).

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
| 15 | Tests, docs, examples | **PASS** | 394 tests, new guides + examples + 2 benchmarks |
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
| Distributed abstractions | **PARTIAL** | `DeviceMesh`/`ProcessGroup`/`DataParallel` + in-process `LocalGroup` collectives; multi-process NCCL/launcher NOT IMPLEMENTED |
| Macro compile-pass/compile-fail UI suite | **PASS** | 33 expansion/diagnostic unit tests + 7 `trybuild` fixtures with committed `.stderr` |
| CUDA kernels for new IR ops | **PASS (except backward)** | Sqrt/Exp/Log/Abs/AbsBackward, Cast/MulAdd/MatMulBias, LogSoftmax/Slice/Embedding/Concat/Split/MoE verified; `SliceBackward`/`EmbeddingBackward` unsupported |
| CUDA re-verification after this turn's changes | **PASS** | model/custom/new-ops/gpu_check + Sanitizer on the T4; session renewed |

## 11.2b Module-system, macro and state redesign

| Item | Verdict | Evidence |
|---|---|---|
| `ModuleList<M>` (paths `0..n`, indexing, iteration, container ops) | **PASS** | `tests/module_system.rs` |
| Enum architectural variants (named/tuple, `#[module(body)]`, generics) | **PASS** | `tests/module_system.rs`, `derive::tests::*` |
| Module-collection loops in `#[model]` (nested, tuple patterns, plain-loop safety) | **PASS** | `tests/module_system.rs`, `model::tests::*` |
| Tied parameters (`TiedLinear`, `Arc` aliasing, one canonical store entry) | **PASS** | `tests/module_system.rs`; alias table + `parameter_ids` |
| Reusable KV state (`KvCache`, shared KV, growth, truncate, reset) | **PASS** | `tests/kv_cache.rs`, Gemma layout tests |
| Gemma decoder stack on `ModuleList`, tied head on `TiedLinear` | **PASS** | tiny dense + E2B-feature parity `2.999e-7` unchanged |
| Streaming uses framework `KvCache` | **PASS** | tiny streaming parity `3.427e-7`, generation unchanged |
| Refactored Gemma on CUDA (A100 rebuild) | **PASS** | dense `1.609e-6`, MoE `1.162e-6`, memcheck 0 errors after `TiedLinear`/`ModuleList`/state changes |
| Full E2B f32 on CUDA after the refactor | **PASS** | A100, `"Hello"` top-5 identical to CPU (`9259 4.861705`, ...), prefill 29.553 s ([log](gemma4-cuda-e2b-refactor.log)) |
| Compile-fail diagnostics | **PASS** | `tests/ui/*.stderr` |
| **Declarative revision:** tuple structs, multi-input calls, multi-output destructuring | **PASS** | `tests/module_system.rs`, compile-pass `tests/ui/pass/multi_io.rs` |
| **Declarative revision:** reusable `RmsNorm`/`GatedMlp`/RoPE/mask/GQA components | **PASS** | `tests/transformer_components.rs` (8) |
| **Declarative revision:** Gemma on generic components | **PASS** | dense `2.999e-7`, streaming `3.427e-7` unchanged |
| **Declarative revision:** streaming sessions + prefill/decode + isolation | **PASS** | `crates/oxide-ml-gemma/tests/streaming_sessions.rs` |
| **Declarative revision:** CPU suite | **PASS** | `cargo test --workspace`: 438 passed, 0 failed |
| **Gemma layer/attention declarative migration** | **PASS** | tiny dense `2.999e-7`, streaming `3.427e-7`, MoE tiny `1.639e-7` unchanged |
| **Declarative Gemma on CUDA (fresh T4, current source)** | **PASS** | dense `1.609e-6`, MoE `1.162e-6` ([log](t5-declarative-cuda.log)) |
| **Declarative revision:** CUDA rebuild (fresh T4 session, current source) | **PASS** | Rust 1.99.1, cargo-oxide 0.4.0, `sm_75`; dense `1.609e-6`, MoE `1.162e-6` ([log](t4-declarative-verify.log)) |
| **Declarative revision:** CUDA model/training parity on T4 | **PASS** | forward `7.749e-7`, 20-step loss `2.325e-6`, params `2.623e-6` ([log](t4-gpu-bench.log)) |
| **Declarative revision:** `gpu_check` (T4, current source) | **PASS** | execution safety, reclamation, MLP backward, dynamic batch, training checks |
| **Declarative revision:** Compute Sanitizer memcheck | **PASS** | `gemma_cuda_check` dense, 0 errors |
| **Declarative revision:** GPU benchmark suite | **PASS** | 30 cases, 100 samples each ([JSON](phase7f19-benchmark-results-t4.json)) |
| Concurrent streaming sessions (`&self` model, per-session KV) | **PASS** | `streaming_sessions.rs::concurrent_sessions_share_the_model_but_not_state` |
| Paged KV cache (block table, gather, truncate, sharing) | **PASS** | `tests/paged_kv_cache.rs` |
| F16/BF16 low-level execution | **PASS** | `tests/half_precision.rs` |
| Generic differentiable MoE (soft routing) | **PASS** | `tests/transformer_components.rs` (forward + finite differences) |

## 11.3 Gemma 4 inference milestone

| Item | Verdict | Evidence |
|---|---|---|
| safetensors reader (mmap, F32/F16/BF16) | **PASS** | `crates/oxide-ml-safetensors` tests |
| Dense Gemma 4 text architecture | **PASS** | hidden 8/4-layer dense checkpoint |
| PLE, shared KV, double-wide MLP | **PASS** | tiny E2B-feature checkpoint, max diff `2.999e-7` |
| MoE text block (fused `Op::MoE`) | **PASS** | tiny 4-expert/top-2 checkpoint, max diff `1.639e-7` |
| Official `gemma-4-E2B-it` config parsing | **PASS** | 35 layers, 28 sliding/7 full, 20 shared-KV, PLE 256, double-wide |
| Layer-streamed E2B inference | **PASS** | 9.29 GB BF16 text, peak RSS ≈6.2 GB, CPU |
| Rust tokenizer (`tokenizer.json`) | **PASS** | `Hello` → `[9259]` → `Hello` |
| E2B logits vs transformers fp16 | **PASS** | identical top-5/order (raw and chat prompts) |
| Streaming vs graph (tiny E2B) | **PASS** | max diff `3.427e-7` |
| Weight mapping / config parser | **PASS** | 87 text tensors, 2,136,588 parameters |
| HuggingFace logits parity | **PASS** | max abs diff **2.682e-7** |
| Greedy generation parity | **PASS** | `[186294, 51247, 59626, 246881]` identical |
| `google/gemma-4-E2B-it` full inference | **PASS (CPU streaming)** | 9.29 GB BF16; one layer at a time, peak RSS ≈6.2 GB; fp16 top-5 parity |
| Gemma 4 on CUDA | **PASS (A100)** | tiny parity `1.609e-6`; full E2B f32 top-5 matches CPU |
| E2B layer streaming | **PASS** | peak RSS 6.2 GB; fp16 reference top-5 identical |
| KV cache (`prefill`/`decode`) | **PASS** | tiny E2B-feature model, cached greedy `[224,397,812,508]` matches |

Evidence: [gemma4-e2b-cuda.log](gemma4-e2b-cuda.log), [gemma4-e2b-cuda-chat.log](gemma4-e2b-cuda-chat.log), [gemma4-e2b-tiny.log](gemma4-e2b-tiny.log),
[gemma4-e2b-tiny-streaming.log](gemma4-e2b-tiny-streaming.log),
[gemma4-e2b-fp16-reference.log](gemma4-e2b-fp16-reference.log),
[gemma4-e2b-run.log](gemma4-e2b-run.log),
[gemma4-e2b-chat.log](gemma4-e2b-chat.log),
[gemma4-e2b-memory.log](gemma4-e2b-memory.log),
[gemma4-tiny-reference.json](gemma4-tiny-reference.json),
[t4-gemma_rust3.log](t4-gemma_rust3.log), and [gemma4.md](gemma4.md).

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
- Gemma 4 KV cache, general `TopK`/gather, CUDA execution and mixed precision.
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
- `docs/gemma4.md` — Gemma 4 inference support, verification and the E2B blocker.
- `docs/model-api.md`, `docs/proc-macros.md`, `docs/custom-kernels.md`,
  `docs/training-and-checkpoints.md`, `docs/migration-0.1.md`,
  `docs/known-limitations.md` — new guides.
- New examples: `examples/model_xor.rs`; new benchmark `benches/model.rs`.
- `docs/architecture.md`, `docs/cuda-backend.md` — architecture and CUDA model
  integration notes.
- This report.

## Vision, safetensors and GPU training

| Item | Verdict | Evidence |
|---|---|---|
| Safetensors writer (F32/F16/BF16, metadata, atomic write) | **PASS** | `oxide-ml-safetensors/tests/writer.rs` |
| Module export/import by stable names | **PASS** | `tests/safetensors_export.rs`; 173-tensor MobileNet checkpoint round trip |
| Tied parameter aliases in export | **PASS** | `tests/safetensors_export.rs` (`alias.head.weight`) |
| BatchNorm running-stat calibration | **PASS** | `tests/batch_norm_calibration.rs`; probes + `calibrate_batch_norm` |
| MobileNetV4 (UIB, depthwise, SE, global pool) | **PASS** | `tests/vision_models.rs`; forward `[2,3,64,64] → [2,2]`, static-batch training step; CPU example 1 epoch + export |
| GPU cats/dogs training (PyTorch mirror) | **PASS** | Tesla T4; `conv_small` 890,474 trainable parameters; best val `0.7042` ([log](cats-dogs-gpu-training.log)) |
| Rust load/eval of the GPU checkpoint | **PASS** | 173 tensors; test subset 100 images `0.7400`; logits `2.879e-3` max abs |
| CUDA MobileNetV4 training | **FAIL (not implemented)** | CUDA lacks Conv2d/BatchNorm/CrossEntropy training kernels; training uses PyTorch |
| CNN symbolic-batch training | **FAIL (limitation)** | `ReduceMean` autograd needs static shapes; use a static batch |
| External photos | **PARTIAL** | dogs correct; two close-up cat photos misclassified; checkpoint is a demo, not production |

## CUDA convolution training

| Item | Verdict | Evidence |
|---|---|---|
| CUDA `Conv2d` / depthwise forward | **PASS** | MobileNetV4 eval forward `8.3e-17` vs CPU |
| CUDA `Conv2dBackward` / depthwise backward | **PASS** | one-step loss `6.6e-7`; parameter update `~2e-3` |
| CUDA `CrossEntropyLoss` forward/backward | **PASS** | same loss to `6.6e-7` |
| CUDA BatchNorm training | **PASS** | train-mode forward diff `2.0e-4`, isolated BN exact |
| `ReduceMean` keep-dims axis mapping | **PASS (fixed)** | previously mapped output channel to axis 0; now correct |
| CUDA AdamW state initialization | **PASS (fixed)** | `CudaTensor::zeros` was uninitialized memory; now zeroed |
| `train_mobilenet` end-to-end CUDA training | **PASS (functional)** | 1 epoch 125.4 s, finite loss, safetensors export |
| Speed vs PyTorch cuDNN | **FAIL (12.3× slower)** | 125.4 s vs 10.22 s per epoch on T4 |
| CUDA training accuracy (1–3 epochs) | **FAIL (not converged)** | val 0.525 (majority class); PyTorch checkpoint 0.7042 |
| Generic grouped conv (1 < groups < C) | **NOT IMPLEMENTED** | groups=1 and depthwise only |
