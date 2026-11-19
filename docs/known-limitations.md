# Known limitations

This page records the gaps in the 0.1 milestone honestly. The CUDA crate was
**not** compiled or run for this milestone because this environment has no CUDA
toolkit, GPU, or cuda-oxide, and the Colab session is unavailable.

## Model API and dtype

- **F32 only.** `Module`, `CompiledModel`, `Trainer`, `Tensor`, datasets, losses,
  and checkpoints are F32. The low-level graph API remains generic over
  `Scalar`, but the model path does not expose F64 or integer tensors.
- **No mixed precision.** There is no autocast, loss scaling, or FP16/BF16 model
  path.

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

- **`.forward()` is limited to `self.<field>` chains.** Inside `#[model]`,
  calling `.forward()` on a local variable is rejected; store the module in a
  field. `self.forward(...)` is rejected to avoid recursion. Use
  `forward_multi` for modules with multiple inputs.
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
- **Some IR ops still have no CUDA kernel.** `LogSoftmax`, `Slice`,
  `Embedding`, `Concat`, `Split`, `SliceBackward`, and `EmbeddingBackward`
  return an explicit `CudaError::Unsupported`. `Sqrt`, `Exp`, `Log`, `Abs`,
  `AbsBackward`, `Cast` (same dtype), `MulAdd` and `MatMulBias` are verified on
  the T4.
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
- **No distributed abstractions.** Device mesh, sharding and collectives are not
  implemented.
- **CPU execution is single-threaded.** No thread pool or explicit SIMD; the
  matmul fast path relies on four independent accumulators.
- **Macro diagnostics are expansion-level.** 21 unit tests cover accepted and
  rejected syntax, but there is no `trybuild` compile-fail fixture suite and no
  generated-code snapshot test.
- **Property tests are randomized, not a property-testing framework.** The
  generator is a small deterministic xorshift; failures are reproducible by
  seed.
- **CUDA coverage is partial for the new operators.** Cast/MulAdd/MatMulBias
  and Sqrt/Exp/Log/Abs/AbsBackward are T4-verified; Concat/Split/LogSoftmax/
  Slice/Embedding are not. Conv2d/pooling/CrossEntropy/LayerNorm-backward remain
  CPU-only.

## Related documentation

- [model-api.md](model-api.md)
- [proc-macros.md](proc-macros.md)
- [custom-kernels.md](custom-kernels.md)
- [training-and-checkpoints.md](training-and-checkpoints.md)
- [migration-0.1.md](migration-0.1.md)
