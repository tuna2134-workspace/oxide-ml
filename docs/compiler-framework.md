# Compiler framework: cache, kernel selection and reports

This milestone turns the existing pass pipeline into an explicit compiler
front end for the model API.

## Compile cache

`oxide_ml_model::cache` compiles each graph once per identity.

```text
structural key (fast, no graph build)
  cache version + compiler version + model type + module descriptor
  + parameter names/shapes/roles/aliases + input signature
  + train/eval mode + optimize/memory-plan flags + backend + arch + ABI

canonical key (authoritative, after graph build)
  BLAKE3 canonical graph hash + the same metadata
```

- A structural hit returns the compiled model without building the graph.
- A structural miss builds the graph, computes the canonical graph hash, and
  looks it up under the canonical key. Structurally identical models (even of
  different Rust types) share the artifact.
- Compilation runs outside the lock. A per-key in-flight set plus a condition
  variable makes concurrent callers wait for one compilation.
- Entries are LRU-evicted (default capacity 32); aliases are dropped when their
  target is evicted.
- `clear`, `invalidate`, `set_capacity`, `stats` and explicit-cache APIs are
  public. The process-wide cache is used by generated `predict` and by
  `Runtime::compile`.
- The key includes `COMPILE_CACHE_VERSION`, `COMPILER_VERSION` and
  `KERNEL_ABI_VERSION`; a codegen change must bump one of them.

Measured on the reference CPU (MLP 784-256-10, batch 64):

| Operation | Time |
|---|---:|
| cold compile | 250 µs |
| cached compile | 10.1 µs |
| compiled run | 5.0 ms |

## Kernel selection

`oxide_ml_kernel::KernelCatalog` holds `KernelProvider` candidates with
capabilities: op, dtypes, layouts, rank range, architectures, workspace,
autograd support and ABI.

Selection filters by capability and then orders by **specificity**
(`Generic < Vectorized < Tiled < TensorCore < Fused < Specialized < Custom`),
then explicit priority, then provider name. It never estimates performance, so
selection is deterministic and reproducible. `KernelRequest` can set a workspace
budget, target architecture and whether custom providers are allowed.

The compiler builds a `KernelPlan` (`node -> provider`, alternatives, reason)
and exposes it in `CompileReport::kernel_plan` and
`CompiledModel::kernel_plan()`. A node without a provider is a diagnostic by
default and an error with `strict_kernel_selection`.

## Compiler report

`CompileReport` now records:

- before/after verification summaries, applied and rejected passes;
- fusion count, kernel count, nodes removed;
- peak planned bytes and compile time in microseconds;
- backend name and per-node selection diagnostics;
- optional graph dumps.

`ModelReport` mirrors the relevant fields (`kernels`, `fusions`,
`compile_time_micros`, `diagnostics`, `memory_plan_bytes`).

## Fusion

`OperatorFusion` now performs:

| Pattern | Result | Condition |
|---|---|---|
| `Add + ReLU` | `AddRelu` | Add result single-use, not a graph output |
| `Mul + Add` | `MulAdd` | Mul result single-use, not a graph output |
| `MatMul + Add` | `MatMulBias` | MatMul result single-use, not a graph output, shapes equal |

Fused nodes keep the consumer's output `ValueId`, so graph outputs and
downstream handles stay valid. No floating-point reassociation is performed.
Tests compare fused and unfused execution exactly.

## Runtime

```rust
let runtime = Runtime::builder()
    .device(Device::Cpu)
    .cache_capacity(32)
    .build()?;
let compiled = runtime.compile(&model, &inputs, ModelCompileOptions::default())?;
let output = compiled.run_with(&runtime, &parameters, &[("input0", &input)])?;
```

`Device::cuda(0)` constructs a logical CUDA device. A CUDA runtime is rejected
with an explicit error because execution is owned by `oxide-ml-cuda`'s
`CudaModel`/`CudaTrainingModel`; there is no hidden CPU fallback.

## Training additions

- `clip_grad_norm(max)` clips the global L2 norm before each optimizer step and
  records the pre-clipping norm.
- `metrics(true)` computes classification accuracy from rank-2 logits and
  rank-1 class-index targets.
- `early_stopping(patience, min_delta)` stops `fit` and sets
  `TrainReport::stopped_early`; the state machine is reusable
  (`EarlyStopping`).

## New operations

`Cast`, `Concat`, `Split`, `MulAdd` and `MatMulBias` have shape inference, CPU
kernels, gradient rules, canonical encoding and serialization. `Concat` also
enables RNN/GRU/LSTM sequence outputs (`with_sequences()` → `[B, T, H]`).

`Cast` is executable only when the target dtype equals the executor scalar
type; the reference executor is single-dtype, so mixed precision is not
implemented. A dtype-changing cast is rejected with `DTypeMismatch`.
