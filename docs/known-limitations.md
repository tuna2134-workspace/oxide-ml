# Known limitations

This page records the gaps in the 0.1 milestone honestly. The CUDA crate was
**not** compiled or run for this milestone because this environment has no CUDA
toolkit, GPU, or cuda-oxide, and the Colab session is unavailable.

## Model API and dtype

- **Model API is F32.** `Module`, `CompiledModel`, `Trainer`, `Tensor`,
  datasets, losses and checkpoints are F32. The low-level graph API is generic
  over `Scalar` and now supports F64, F16 and BF16 execution
  (`half::f16`/`bf16` implement `Scalar`); there is no autocast, mixed-precision
  training or loss scaling, and CUDA remains F32.

## Training-time layers

- **Dropout is not stochastic.** The graph IR has no per-step random source.
  `Dropout::new(p)` accepts `p` in `[0, 1)`, but building or running a training
  graph with `p > 0` returns `Error::Unsupported` instead of silently applying
  no mask. Eval mode is the identity; `p = 0` is a no-op.
- **BatchNorm running statistics are not updated.** Training normalizes with
  batch statistics. Eval uses `running_mean`/`running_var`, which are saved and
  restored as buffers but never written by training. Without externally supplied
  values, eval after training still uses the initial mean 0 / variance 1.
  `without_running_stats()` makes the behavior explicit.
- **Conv3d is unsupported.** `Conv3d::new` exists for API compatibility, but the
  type owns no parameters and `forward` returns `Error::Unsupported`. Use
  `Conv2d` over frames or add a 3-D convolution op to the graph stack.
- **Recurrent layers are minimal.** `Rnn`, `Gru`, and `Lstm` accept `[B, T, D]`
  with a static sequence length, return only the last hidden state `[B, H]`, and
  accept no initial state. There is no full-sequence output, bidirectionality, or
  packed-sequence support.
- **Attention needs three inputs.** `Attention` implements `forward_multi`;
  calling `forward` on it returns an error. `MultiHeadAttention` is
  self-attention over one `[B, T, E]` input and supports a causal mask.

## Macros

- **`.forward()` works on fields, tuple fields and module-collection loop
  variables.** Inside `#[model]`, `self.<field>.forward(x)`, `self.0.forward(x)`
  and `for layer in &self.layers { layer.forward(x) }` are rewritten. Multiple
  arguments dispatch to `forward_multi`, and `let (a, b) = ...` destructures
  multiple outputs with an arity check. Calling `.forward()` on an arbitrary
  local (for example one created by `let layer = self.fc;`) is rejected rather
  than guessed; store it in a field or iterate the field. `self.forward(...)` is
  rejected to avoid recursion.
- **Structured named outputs are not generated.** Multi-output modules use tuple
  destructuring (2–4 outputs) or an explicit `forward_multi` call; a module
  returning a struct whose fields map to graph values is not supported.
- **Enum variants have one forward body.** A module enum forwards to its only
  module field or to the field marked `#[module(body)]`; custom per-variant
  control flow needs a hand-written `Module` implementation (the derive emits a
  compile error instead of guessing).
- **`#[model]` rejects async/unsafe functions, closures, `.await`, async blocks,
  non-identifier parameters, empty bodies, and bodies without an output
  expression.** A bare `return;` is also rejected.
- **Multi-output returns must be a literal tuple.** Returning a tuple variable
  from a `#[model]` function with several declared outputs is rejected; return
  `(a, b)` directly.
- **`graph!` is facade-only.** It expands to `::oxide_ml::__private` and has no
  crate override, so a crate that renames the facade dependency cannot use it.
- **`graph!` has no constants.** Every value is a graph input, a graph parameter,
  or the result of a listed operation. It is also a fixed statement list: no
  `if`, loops, closures, or arbitrary expressions.
- **Reserved identifiers.** Generated code uses names beginning with
  `__oxide_` (argument temporaries, context, output/loss value names). Avoid
  those prefixes for locals, fields, and graph value names.

## Parameters and write-back

- **Shared `Arc<M>` parameters cannot be written back into the module tree.**
  When an `Arc` is shared, `visit_tensors_mut` skips it. The `ParameterStore`
  remains authoritative, and `WriteBackReport::skipped_aliases` counts the
  skipped entries. Training still updates the single canonical parameter.
- **Aliases are canonicalized.** Two module paths that share tensor storage
  produce one canonical parameter and one optimizer slot; the second path is an
  alias. Do not expect two independent checkpoints entries for aliased paths.

## Devices and datasets

- **The CPU `Trainer` rejects `Device::Cuda`.** It returns an explicit error
  directing callers to `oxide-ml-cuda`'s trainer/model. The CUDA crate is a
  standalone workspace and is excluded from the root workspace.
- **One target tensor.** `Trainer::fit` and `Trainer::evaluate` support exactly
  one target tensor from the dataset; multi-target datasets are rejected.
- **`TensorDataset` needs a leading sample dimension.** Scalar inputs/targets
  without a batch dimension are rejected, and all input/target tensors must
  agree on the sample count.
- **`predict` recompiles.** The generated `predict` helper compiles an unbatched
  graph per call. Use `CompiledModel` for repeated inference.

## Checkpoints and optimizer state

- **Optimizer state requires matching parameter names.** `state_dict` is keyed by
  stable name, but `load_state_dict` resolves those names through a supplied
  `ParameterStore`. The store must be built from the same graph/name set; a
  missing name is an error.
- **`Trainer::load_checkpoint` does not refresh the supplied store.** It applies
  parameter values to the model and restores optimizer slots using the store's
  name-to-id mapping. Rebuild the store from the restored model before continuing
  training.
- **Checkpoints are JSON.** They contain raw F32 data and are not a binary or
  memory-mapped format. Optimizer slots are stored as `f64`.
- **Optimizer hyperparameters are not restored.** Only slots and (for AdamW) the
  step are persisted; rebuild the optimizer with the desired learning rate and
  betas.
- **Model-type mismatch is reported, not fatal.** `LoadReport::model_type_matches`
  is `false` when the checkpoint was produced by a different `module_type`, but
  matching names/shapes are still applied. `is_exact()` only reflects missing
  names.

## CUDA status

- **CUDA was verified on a Tesla T4 in this milestone** (CUDA 13.0, driver
  580.82.07, `cargo oxide 0.4.0`, `--arch sm_75`). `model_check` and
  `custom_kernel_check` pass with maximum differences of `7.7e-7`, `2.6e-6` and
  `0.0`; memcheck, racecheck and synccheck report zero errors.
- **Coverage is targeted, not exhaustive.** Only the model/custom checks and
  `gpu_check` were run; the other acceptance binaries and the Phase 7F
  benchmark matrix were not re-run.
- **One pre-existing concurrency assertion is timing sensitive.** The first
  `gpu_check` run failed `concurrent4.stats().slot_set_count >= 4`; a rerun
  passed. Treat that assertion as flaky until stabilized.
- **Remaining IR ops without a CUDA kernel.** `SliceBackward` and
  `EmbeddingBackward` return an explicit `CudaError::Unsupported`. `Slice`,
  `Embedding`, `Concat`, `Split`, `LogSoftmax`, `MoE`, `Sqrt`, `Exp`, `Log`,
  `Abs`, `AbsBackward`, `Cast` (same dtype), `MulAdd` and `MatMulBias` are
  verified on A100/T4 (see [verification-report.md](verification-report.md)).
- **CUDA custom kernels are F32-only.** `CudaKernelRegistry::register` rejects
  other dtypes. Launch functions must use the executor's stream and buffer pool;
  the registry does not generate PTX. The registry also does not acquire
  `AccessGuard` for a launch; use `launch_with_access` in the closure (as
  `custom_check.rs` does).

## Compiler framework

- **Control flow is host-side in `#[model]`.** `if`/`match`/loops are executed
  while building the graph. Tensor-dependent control flow is not representable;
  the intended Region/Block/Cond/While design is in
  [control-flow-ir.md](control-flow-ir.md) and is **not implemented**.
- **Mixed precision is not executable.** The IR now carries F16/BF16/integer
  dtypes, but the reference executor is single-dtype. `Cast` to a different
  dtype is rejected; there is no autocast, loss scaling or accumulation dtype.
- **Kernel selection is structural, not performance-based.** It prefers
  specialized/fused providers by capability and specificity; there is no cost
  model, autotuning or measured choice.
- **Distributed execution is single-process or in-process only.** `DeviceMesh`,
  `ProcessGroup`, `DataParallel` and `LocalGroup` exist; `LocalGroup` implements
  collectives across threads, but multi-process launch, NCCL/Gloo, gradient
  all-reduce and sharding are not implemented.
- **CPU matmul is multithreaded; other kernels are not.** The contiguous
  output buffer is split across up to eight threads, but there is no explicit
  SIMD and elementwise/reduction kernels remain single-threaded.
- **Macro diagnostics are expansion-level.** Unit tests cover accepted and
  rejected syntax, and `trybuild` compile-fail fixtures cover seven rejected
  forms. There is still no generated-code snapshot test.
- **Property tests are randomized, not a property-testing framework.** The
  generator is a small deterministic xorshift; failures are reproducible by
  seed.
- **CUDA coverage is partial for the new operators.** Cast/MulAdd/MatMulBias
  and Sqrt/Exp/Log/Abs/AbsBackward are T4-verified; Concat/Split/LogSoftmax/
  Slice/Embedding are not. Conv2d/pooling/CrossEntropy/LayerNorm-backward remain
  CPU-only.

## Gemma 4 inference

- **MoE text blocks are supported.** PLE, shared KV, double-wide MLP and MoE
  blocks are implemented and verified on a tiny 4-expert, top-k 2 checkpoint
  (CPU max diff `1.639e-7`, CUDA `1.162e-6`). Expert execution is fused
  (`Op::MoE`) and inference-only: there is no autograd rule and no general
  `TopK`/gather IR op, so expert-parallel training is not available.
- **`google/gemma-4-E2B-it` runs through the CPU layer-streaming path and on
  A100 CUDA.** It fits a 12 GB host via one-layer-at-a-time f32 conversion
  (peak ≈6.2 GB), but not the full f32 parameter store. The A100 path loads the
  full f32 text model (18.6 GB) and has no KV cache, so it re-runs the prompt
  for each generation step. See [gemma4.md](gemma4.md).
- **Streaming repeats host-side orchestration, not layer math.** The
  layer-streamed path reimplements embedding lookup, PLE packing, final norm and
  the tied head on the host because the full-model graph does not fit in memory;
  the decoder-layer math is the same `GemmaLayer` code as the graph path. Both
  paths are pinned to the same logits on the tiny model (`2.999e-7` vs
  `3.427e-7`), which is the regression guard for this duplication.
- **KV cache is implemented for layer streaming through the reusable
  framework `KvCache`.** `prefill`/`decode` avoid recomputing the prompt and
  support shared-KV links; the graph path still recompiles per sequence length,
  and each decode step re-reads layer weights (no resident f32 model). The CUDA
  executor has no KV-cache integration yet.
- **Sparse top-k MoE remains inference-only.** `Op::MoE` performs routing and
  expert computation inside one fused kernel (CPU and CUDA) and has no autograd
  rule. The generic `MixtureOfExperts` module uses differentiable soft routing
  over ordinary expert modules and trains with the standard autograd path.
  There is no general `TopK`/gather IR op.
- **Streaming sessions are state-isolated and concurrent.** `StreamingGemma`
  shares a mutex-protected cache of compiled layer templates, and
  `prefill`/`decode` take `&self`; each `StreamingSession` owns its KV cache.
  Concurrent sessions are tested on threads. Compilation of a new shape happens
  under the cache lock, so many distinct prompt lengths serialize compilation.
- **`grouped_query_attention` is single-batch.** Batch is the caller's loop
  (`[1, length, heads, head_dim]` tensors).
- **Paged KV cache is host-side.** `PagedKvCache` implements block tables,
  shared KV, truncate/reset and `gather_*`; a device runtime must upload the
  physical pools and block table and consume pages directly.
- **CUDA Gemma requires A100-class memory.** The full f32 E2B text model is
  18.6 GB; a T4 (15.36 GB) cannot hold it, so T4 uses CPU streaming. CUDA
  supports F32 only and batch size 1. Verified generation re-runs the prompt
  (≈29.8 s/token); fixed-length resident decode removes the per-token re-upload
  (0.40 s/token, still O(T²)); a graph-level KV decode (`GemmaDecode`) is
  implemented and A100-verified at ≈0.30 s/token after a CPU prefill
  ([log](gemma4-e2b-paris-kvcache.log)).
- **The verified checkpoint is a tiny random dense model**, not a quality model;
  the result proves architecture/loader parity, not generation quality.

## Related documentation

- [model-api.md](model-api.md)
- [proc-macros.md](proc-macros.md)
- [custom-kernels.md](custom-kernels.md)
- [training-and-checkpoints.md](training-and-checkpoints.md)
- [migration-0.1.md](migration-0.1.md)
