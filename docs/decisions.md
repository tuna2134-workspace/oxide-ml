# Design decisions locked in Phase 5

This document records contracts that must remain stable while the compiler,
executor, and future backends evolve.

## 1. Version domains

Four versions are deliberately independent:

| Domain | Constant | Current | Changes when |
|---|---|---:|---|
| JSON envelope | `SERIALIZATION_FORMAT_VERSION` | 2 | field layout/encoding changes |
| Semantic IR | `IR_VERSION` | 1 | node/value meaning changes |
| Kernel ABI | `KERNEL_ABI_VERSION` | 2 | `KernelCall`/`KernelBackend` contract changes |
| Backend capability | `BACKEND_CAPABILITY_VERSION` | 1 | optional capability descriptor changes |

`GRAPH_FORMAT_VERSION` remains as a backwards-compatible alias for the envelope
version. It is **not** the IR version. A v1 envelope (no `ir_version` field) is
migrated on read to the current envelope and IR v1. Unknown envelope or IR
versions are rejected before execution.

The graph stores only semantics: symbols, input/output value ids, nodes with
`Op`/`OpAttrs`, dense value metadata, and parameter metadata. Constant payloads
are values in `OpAttrs::Constant`; tensor bytes are never stored on `ValueInfo`.

**Invariants verified by `GraphVerifier`:**

- dense value, node, parameter, and symbol tables;
- every node input is defined before the node;
- no value has two definitions;
- dependency-cycle detection;
- every graph input has `Input` kind and every output is defined;
- every node is re-inferred, so serialized shapes/dtypes must match;
- parameter metadata agrees with its value and has a static shape;
- duplicate names are rejected.

`Graph::from_json` calls the verifier. `Executor` and `Compiler` call it again
at their boundaries. A graph that fails verification never executes.

## 2. Gradient-rule registry

Built-in rules remain in `oxide-ml-autograd/src/rules.rs`, but backward graph
construction resolves them through `GradientRegistry`:

```rust
registry.register("my_op", Arc::new(FunctionRule::new("my_op", rule_fn)));
let training = build_training_graph_with_registry(&forward, loss, &registry)?;
```

`GradientRegistry::ensure_supported` runs before backward construction. A
missing rule is an `Autograd` error naming the operation and node, not a panic
halfway through execution. Resolution happens once per forward node while the
backward graph is built; there is no per-kernel dynamic lookup.

The rule contract returns one optional gradient per input. Accumulation is
centralized in `build_training_graph`: a second contribution becomes an `Add`
node, with `Unbroadcast` inserted when shapes differ.

Fused backward ops (`ActivationBackward`, `AddReluBackward`, `SoftmaxBackward`,
`LayerNormBackward`, `Conv2dBackward`, `MSELossBackward`,
`CrossEntropyLossBackward`, `Unbroadcast`) are first-class IR operations. This
keeps backward graphs serializable and executable by the same executor.

Training graphs explicitly output the loss and parameter gradients because the
planned executor retains only graph outputs at the end. `forward_outputs` keeps
the inference boundary used by `Trainer::inference_executor`.

## 3. Kernel ABI

`oxide-ml-kernel` defines a generic, versioned boundary:

```rust
pub struct KernelCall<'a, T: Scalar = f32> {
    pub op: &'a Op,
    pub attrs: &'a OpAttrs,
    pub inputs: &'a [DenseTensor<T>],
    pub input_specs: &'a [TensorSpec],
    pub output_specs: &'a [TensorSpec],
    pub aliasing: &'a [AliasingConstraint],
    pub workspace: WorkspaceSpec,
    pub context: ExecutionContext,
}

pub trait KernelBackend<T: Scalar = f32>: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> BackendCapabilities;
    fn execute(&self, call: KernelCall<'_, T>) -> Result<Vec<DenseTensor<T>>>;
}
```

`TensorSpec` carries shape, dtype, strides, alignment, layout, and device.
`AliasingConstraint` and `WorkspaceSpec` are placeholders for future in-place
and scratch allocation; the current CPU backend reports no aliasing and no
workspace. `ExecutionContext` carries device and stream id without introducing a
CUDA type into the IR.

**F64.** `DenseTensor<T>` and every kernel are generic over `core::Scalar`
(`f32`, `f64`). Backends advertise their supported dtypes through
`BackendCapabilities`. F16/BF16 are rejected, never silently converted.

**Memory planning.** The compiler produces `MemoryPlan` (slot sizes, alignment,
per-value assignments, liveness, planned bytes). The CPU `run_planned` executor
writes each result into its assigned slot and reuses a slot when the lifetime
has ended. Only graph outputs are retained. Kernels still allocate their
temporary output; the plan controls the retained activation arena. This is
explicitly a CPU reference implementation, not a zero-copy allocator.

**Why not compile CUDA in the proc macro?** Proc macros only generate IR and Rust
code. Kernel module compilation belongs to the backend build step. The ABI above
is where that boundary is enforced.

## 4. Pass semantics

- **Constant folding** evaluates nodes whose operands are constants through the
  `KernelBackend` supplied to the pass, then rewrites the node in place to
  `Op::Constant`. Symbolic outputs or dtype-mismatched constants are skipped and
  recorded in the pass notes.
- **DCE** treats graph outputs as roots and removes unreachable nodes. Current
  operations are pure; a future side-effecting operation must be added as an
  explicit root.
- **CSE** compares op, attrs, and resolved ordered inputs. It does not
  reassociate arithmetic or merge nodes with different values.
- **Fusion** currently implements `Add + ReLU -> AddRelu` only when the Add has a
  single ReLU consumer and the Add result is not a graph output. The fused node
  keeps stable ValueIds where possible.
- Every pass boundary is verified. `PassManager` records node counts, timings,
  notes, and optional before/after dumps.

## 5. Other invariants

**Static graph, not a tape.** The executor runs `Graph` by id. `TrainingGraph`
contains forward nodes followed by backward nodes; inference plans truncate the
node list.

**Symbolic shapes.** `Dim::Symbol` is bound from concrete inputs at each run.
Parameters must be static. Values with symbolic shapes are not assigned to memory
slots; planned execution falls back to owned tensors for them.

**No hidden global state.** A `GraphBuilder` owns its graph; a `ParameterStore`
owns typed tensors; an `Executor` owns a verified graph plus a backend. The
compiler never mutates a graph owned by another component.

**Performance.** Kernels are readable reference loops. The compiler reduces node
count and retained bytes; it does not yet provide tiled/SIMD kernels or a cost
model. Optimization is allowed to be slower in wall-clock compile time; benchmark
output keeps that visible.
