# Architecture

oxide-ml is layered so that the graph IR, the compiler, autograd, the model API
and the CUDA backend can evolve independently.

```text
oxide-ml-core          dtypes, devices, shapes, DenseTensor, ParameterStore
      |
oxide-ml-graph         SSA IR, GraphBuilder, inference, verifier, canonical hash
      |
oxide-ml-kernel        backend ABI + custom kernel descriptors
      |
      +--> oxide-ml-cpu        reference kernels + Executor
      +--> oxide-ml-autograd   TrainingGraph + gradient registry
      +--> oxide-ml-compiler   passes + memory planner
      +--> oxide-ml-optim      optimizers + portable state
      +--> oxide-ml-model      Module, CompiledModel, Trainer, checkpoints
      +--> oxide-ml-nn         layers/losses (model API) + legacy builder layers
      +--> oxide-ml            facade, prelude, legacy compatibility
      +--> oxide-ml-cuda       standalone cuda-oxide backend
```

`oxide-ml-graph` never sees a backend type. `oxide-ml-model` has no CUDA
dependency: the CUDA crate consumes its graph and parameter-name contracts.
`oxide-ml-macros` is an independent proc-macro crate that generates ordinary
Rust calls; it never re-implements the type checker.

## Graph IR

The IR is SSA with dense `ValueId` / `NodeId` tables:

- each value is defined exactly once (input, parameter, constant, or node
  output);
- nodes are topologically ordered;
- shapes are symbolic (`Dim::Static`, `Dim::Symbol`, `Dim::Dynamic`);
- operation semantics live in `Op`/`OpAttrs`; constants are values, not tensor
  bytes on `ValueInfo`;
- `GraphVerifier::verify` re-infers every node, checks definition-before-use,
  cycles, duplicate names, parameter metadata and graph inputs/outputs.

Version domains are independent: serialization envelope (`2`), semantic IR
(`1`), kernel ABI (`2`), backend capability (`1`). Adding an `Op` variant is
additive; existing graphs remain valid. Canonical hashing encodes every operator
tag and attribute so a new variant cannot collide with an old hash.

New in this milestone: `Sqrt`, `Exp`, `Log`, `Abs`, `LogSoftmax`, `Slice`,
`Embedding`, fused backward `AbsBackward`, `SliceBackward`,
`EmbeddingBackward`, and `Custom` (a user-registered kernel with declared output
shapes).

## Autograd

`build_training_graph` clones the forward graph and appends backward nodes in
reverse topological order. The backward pass is a normal executable graph, so it
uses the same kernels, verifier, memory planner and backend.

- `GradientRegistry` resolves a rule per forward operation before construction
  starts; a missing rule is an autograd error naming the operation.
- Gradient accumulation is centralized: a second contribution becomes an `Add`
  node, with `Unbroadcast` when shapes differ.
- `CustomKernelRule` dispatches `Op::Custom` backward by kernel name. A kernel
  registered without a backward rule fails at training-graph construction, not
  during execution.
- Numerical gradient checks live in `tests/gradient_check.rs`,
  `tests/model_gradients.rs`, `crates/oxide-ml-autograd/tests/new_ops_grad.rs`.

## Compiler and memory planning

`PassManager` runs constant folding, CSE, operator fusion (`Add` + `ReLU` →
`AddRelu`) and DCE. `MemoryPlanner` assigns aligned, non-overlapping slots from
liveness. `CompiledModel` runs both for every model graph it compiles
(`ModelCompileOptions { optimize, memory_plan, .. }`) and records the resulting
node/value counts and peak bytes in `ModelReport`.

CSE/DCE preserve `ValueId`s, which keeps inputs, outputs and parameter handles
valid across optimization. Fusion preserves numerical meaning; it never
reassociates floating-point reductions.

## Model layer

```text
Module tree
  -> visit_parameters: ParameterRegistry (stable names, aliases)
  -> forward through ModelContext: GraphBuilder + path stack
  -> Graph -> CompiledModel (explicit `forward`/`loss`/`training` graphs)
  -> ParameterStore bound by name -> Executor
  -> write_back into the module tree
```

`ModulePath` is a dot-separated path (`fc1.weight`). Modules ask the context for
parameters by a path-relative name; the derive macro enters a path scope around
every child call. Two module paths that reach the same tensor storage become an
alias of one canonical parameter, so shared parameters receive one gradient and
one optimizer slot.

The compiled model owns a forward graph and (optionally) a training graph. It is
specialized for one input signature; batch dimensions are symbolic so one graph
serves every batch size. `input`/`output`/parameter values survive optimization
and are addressed by name at the model boundary.

## Safety

Phase 7F is unchanged and still applies to standard CUDA kernels:

- event failure state machine and quarantine;
- canonical graph hash + memory-plan hash + device/context identity for arenas;
- async execution lifecycle and deferred reclamation;
- `AccessGuard` read/write hazard tracking;
- dynamic-shape fallback and concurrent stream safety.

Custom CUDA kernel launches use the executor's stream and pool but currently do
not acquire `AccessGuard` automatically; see `docs/known-limitations.md`.
