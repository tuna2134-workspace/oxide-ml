# Verification report — integrated model API, proc macros and CUDA integration

This report records exactly what was implemented, what was executed, and what
remains unverified. Every item uses **PASS / FAIL / NOT RUN / BLOCKED**. A
feature is only PASS when it was executed in this environment.

## 1. Environment

| Item | Value |
|---|---|
| Host | Linux x86_64 workstation, no NVIDIA GPU |
| Rust | `rustc 1.99.1 (58cf8f9c0 2026-10-13)`, `cargo 1.99.1` (stable channel) |
| CUDA toolkit | **not installed** (`nvcc` absent, `/usr/local/cuda*` absent) |
| `cargo oxide` | **not installed** (`no such command: oxide`) |
| GPU | none (`nvidia-smi` absent) |
| Colab session | **expired** (`/api/kernels` returned HTTP 404); re-authentication requires an interactive OAuth flow |
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
| `oxide-ml-cuda` | `prelude`, `custom` (device kernel registry), `model` (`CudaModel`, `CudaTrainingModel`), `Op::Custom` dispatch, dependency on `oxide-ml-model` — **not compiled here** |

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

**Status: NOT RUN.** No CUDA toolkit, no GPU, no `cargo oxide`, expired Colab
session. The CUDA crate cannot be type-checked locally because `cuda-bindings`
requires `CUDA_HOME`/`CUDA_TOOLKIT_PATH` at build-script time. The host-side
additions (`src/model.rs`, `src/custom.rs`) were type-checked against stub
CUDA types in a temporary workspace crate, which found and fixed a missing
`oxide_ml_model` re-export; the `CudaExecutor` dispatch arm and all device code
were validated for syntax only (`rustfmt --check`).

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
| `cargo test --workspace` | **PASS** — 288 passed, 0 failed, 3 ignored (50 test binaries including doc-tests) |
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
| `crates/oxide-ml-graph/tests/new_ops.rs` | 10 | inference/errors for new ops |
| `crates/oxide-ml-autograd/tests/new_ops_grad.rs` | 8 | finite differences for new ops |
| `crates/oxide-ml-macros` unit tests | 15 | macro expansion and diagnostics |

Existing regression suites (graph IR, ops, f64, compiler, memory, training,
gradient check, event/execution safety) all pass unchanged.

## 8. CUDA test results

| Test | Result | Reason |
|---|---|---|
| `cargo oxide build --arch sm_75` | **NOT RUN** | `cargo oxide` and CUDA toolkit unavailable |
| `gpu_check`, `linear_regression_check`, `mixed_graph_check`, ... | **NOT RUN** | same; Colab session expired |
| CPU/CUDA numerical parity for the new model API | **BLOCKED** | requires a CUDA device |
| CUDA custom kernel registration / launch | **NOT RUN** | device code not compiled |
| CUDA new-op kernels (`Sqrt`...`Embedding`) | **FAIL (not implemented)** | explicit `CudaError::Unsupported` |

Historical T4 results from earlier phases remain in `docs/phase6.md` and
`docs/phase7f*.md`; they were **not re-run** and do not cover this milestone's
code.

## 9. Sanitizer results

| Tool | Result |
|---|---|
| `compute-sanitizer --tool memcheck` | **NOT RUN** |
| `compute-sanitizer --tool racecheck` | **NOT RUN** |
| `compute-sanitizer --tool synccheck` | **NOT RUN** |

No GPU was available. CPU tests were run under the ordinary test harness; no
address/UB sanitizer build was performed (Rust toolchain sanitizers require a
nightly toolchain and a separate rebuild, and the environment cannot install
one because `$RUSTUP_HOME` is read-only).

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
| 7 | cuda-oxide kernels registered in the graph | **PASS (CPU mechanism) / NOT RUN (device)** | `Op::Custom` + descriptor + registry; device launch unverified |
| 8 | CUDA kernel executed from `CudaExecutor` | **NOT RUN** | dispatch code exists; no device available |
| 9 | AccessGuard / event safety preserved | **FAIL for custom kernels / PASS for existing kernels** | standard CUDA kernels unchanged; the custom registry uses the stream/pool but does not acquire `AccessGuard` |
| 10 | Model forward and backward | **PASS** | training tests + finite-difference checks |
| 11 | Same model on CPU and CUDA | **PASS (API) / NOT RUN (device)** | `CudaModel`/`CudaTrainingModel` consume the same `BuiltModel` contract |
| 12 | Training API trains | **PASS (CPU)** | XOR MSE and 2-class cross-entropy converge; gradient accumulation, scheduler, validation |
| 13 | Checkpoints save/restore | **PASS** | model + AdamW state round trip, unknown version rejected |
| 14 | Major NN operators available | **PASS with documented gaps** | 23 layers + `Sequential` + 5 losses; Conv3d, sequence outputs, stochastic dropout missing |
| 15 | Tests, docs, examples | **PASS** | 288 tests, new guides + examples + 2 benchmarks |
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
- Mixed precision (F16/BF16) and distributed training.
- `graph!` outside the `oxide-ml` facade.
- f64 model API (the low-level graph API remains generic over `Scalar`).
- Module architecture serialization beyond checkpoint metadata (module type,
  parameter names/shapes/roles).

## 13. Verification debt

1. Compile and run the CUDA workspace on a T4 (`cargo oxide build --arch sm_75`)
   including the new `model`/`custom` modules and `Op::Custom`.
2. Run the full CUDA acceptance suite and Compute Sanitizer (memcheck,
   racecheck, synccheck) on the new code paths.
3. Verify CPU/CUDA numerical parity for every model-API layer.
4. Add CUDA kernels for the new IR ops, or document per-op support in the CUDA
   README.
5. Integrate `AccessGuard` into the custom kernel launch path.
6. Implement BatchNorm running statistics and stochastic dropout (the latter
   needs a per-run random seed in `ExecutionContext`, which is a kernel-ABI
   change and must bump `KERNEL_ABI_VERSION`).
7. Optimize the backward pass and generic matmul layouts; add a model-level
   performance regression benchmark with thresholds.

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
