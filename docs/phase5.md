# Phase 5 survey and implementation notes

This is the pre-implementation survey required by the Phase 5 brief, updated to
match the code as implemented. The README is a summary; the source and tests are
authoritative.

## 1. Current implementation structure

```
oxide-ml-core
  DType/Device/Scalar, symbolic Shape/Dim, DenseTensor<T>, ParameterStore<T>,
  RNG, MemoryPlan data types

oxide-ml-graph
  SSA Graph, Op/OpAttrs, GraphBuilder, shape inference, GraphVerifier,
  constant tensors, JSON serialization/versioning

oxide-ml-kernel
  KernelCall<T>, TensorSpec, aliasing/workspace/context descriptors,
  KernelBackend<T>, KERNEL_ABI_VERSION / BACKEND_CAPABILITY_VERSION

oxide-ml-cpu
  generic F32/F64 kernels, CpuBackend<T>, full Executor<T>, planned
  run_planned executor and ExecutionStats

oxide-ml-autograd
  GradientRegistry, GradientRule/FunctionRule, built-in rules,
  TrainingGraph and backward graph construction

oxide-ml-compiler
  PassManager, ConstantFolding, DCE, CSE, OperatorFusion,
  MemoryPlanner, Compiler/CompileOptions, graph rewrite utilities

oxide-ml-nn / oxide-ml-optim / oxide-ml
  F32/F64-capable layers and optimizers; facade, prelude and Trainer
```

Dependency direction is one-way: core ← graph/kernel ← cpu/autograd ← nn/optim
← facade; compiler ← graph/kernel and is consumed by the facade. The graph crate
has no backend or CUDA dependency.

## 2. Implemented vs. not implemented

Implemented:

- typed F32/F64 tensors and parameters;
- symbolic shapes, shape inference, broadcasting and batched matmul;
- SSA verifier with dense-id, SSA, cycle, shape, dtype and operation checks;
- static graph execution on CPU;
- reverse-mode autograd with explicit backward graph and gradient accumulation;
- built-in gradient rules plus a registration API and preflight missing-rule
  errors;
- constant folding, DCE, CSE, Add+ReLU fusion, pass manager and pass toggles;
- liveness analysis, aligned buffer slots, slot reuse and planned execution;
- separated serialization/IR/kernel-ABI/capability versions;
- F64 forward/backward/optimizer support;
- JSON/file round-trip and legacy envelope migration;
- finite-difference, convergence, compiler, memory, and IR tests;
- a std-only benchmark harness.

Not implemented in Phase 5:

- CUDA/cuda-oxide kernels, CUDA Graph, streams/events;
- proc macros (`#[derive(Module)]`, `graph!`);
- RNN/GRU/LSTM, attention, embedding, BatchNorm;
- F16/BF16 execution;
- true in-place execution and workspace allocation;
- cost-model-driven fusion, kernel selection, compile cache;
- distributed training.

## 3. Changes made for the compiler

- Added `oxide-ml-compiler` as a separate crate. It depends on graph/kernel but
  not on the CPU backend; constant folding receives a `KernelBackend`.
- Added `GraphVerifier` and cycle detection; executor/compiler use it.
- Added `Constant` IR node and `ConstantTensor` payload so folded expressions can
  be represented without abusing scalar constants.
- Added `AddRelu`/`AddReluBackward` and all end-to-end support (inference, CPU
  kernel, autograd rule).
- Split version domains and added v1→v2 envelope migration.
- Added `MemoryPlan` to core (data types) and `MemoryPlanner` to compiler
  (algorithm), preserving the dependency direction.
- Extended `KernelCall` with specs, aliasing, workspace and execution context;
  made `KernelBackend` generic over `Scalar`.
- Added `GradientRegistry`; existing built-ins remain in `rules.rs` but backward
  construction resolves registered rules.
- Made training graphs output loss and parameter gradients so planned execution
  has explicit roots.
- Genericized tensors, kernels, executors, optimizers and layers over `Scalar`
  without changing the default F32 API spelling.

## 4. Compatibility risks

- `DenseTensor` gained a defaulted scalar parameter. Existing `DenseTensor` and
  `Tensor<R>` spellings still mean F32; code using `DenseTensor::new` etc. works.
- `Executor`, `ExecutionInputs`, `ParameterStore`, `Optimizer`, and `Trainer`
  gained defaulted parameters. Type inference can require an explicit
  `::<f32>` in tests that construct an object before a concrete tensor is seen.
  The public F32 examples use the familiar spelling.
- `Graph.format_version` changed from 1 to 2. Old JSON is migrated, but code
  asserting the literal old value must change; that is intentional.
- Training graph outputs now include explicit loss/gradient roots. Full
  execution still exposes every value, but `run_planned` only exposes roots.
- The memory plan is a logical CPU allocation plan. It reduces retained arena
  bytes, not the temporary `Vec` that a kernel returns. Benchmarks must not be
  read as a zero-copy allocator claim.
- Custom gradient rules are keyed by operation name. A custom IR operation still
  needs IR/kernel support; the registry does not invent semantics for an unknown
  `Op`.
- Fusion is intentionally narrow (`Add + ReLU`). It skips patterns whose output
  ownership or use counts are ambiguous.

## 5. Recommended implementation order (actual)

1. Fix IR contracts and verifier.
2. Genericize scalar/tensor/executor and add F64.
3. Extend the kernel ABI.
4. Add gradient registry without changing built-in math.
5. Add compiler crate and pass manager.
6. Add constant folding, CSE, DCE, then fusion.
7. Add liveness and memory planner, then planned executor.
8. Add tests and benchmarks; update the README last.

This order kept every stage executable and allowed the existing convergence
tests to act as a regression gate throughout.

## 6. Verification status

The local CPU suite and release build are the authoritative verification for
this phase. Benchmarks are sampled with the std-only harness in
`benches/compiler.rs`; timings are machine-dependent. GPU work is intentionally
out of scope for Phase 5 and will be run through Colab only when CUDA kernels
exist.
