# oxide-ml

Rust-native, graph-based deep learning framework. This repository now implements
the **Phase 1–6 stack**: typed tensors, a symbolic-shape SSA graph IR, a
reference CPU executor, reverse-mode autograd, a graph compiler with
optimization/memory planning, and a CUDA backend built with **NVIDIA
cuda-oxide** kernels and `cuda-core` host runtime.

The CPU reference remains the numerical oracle. The CUDA backend never falls
back to CPU for an unsupported operation; it returns an explicit error. The graph
IR remains backend-independent.

## Phase 6 CUDA status

`crates/oxide-ml-cuda/` is a standalone workspace because NVIDIA cuda-oxide is a
custom rustc backend (`cargo oxide`), not a normal crate. It was exercised on a
Colab Tesla T4 (sm_75, CUDA 13.0):

- device/context/stream/device-memory/pool: implemented
- device-resident graph executor: implemented
- F32 Constant/Add/Sub/Mul/Div/AddRelu/Neg/ReLU/GELU/Sigmoid/Tanh/Transpose/
  Reshape/Broadcast/Reduce/MatMul/MSE/Softmax/LayerNorm: implemented
- backward: Add/Sub/Mul/Div, MatMul, activation, MSE, AddRelu, Unbroadcast
- GEMM: naive and 16×16 shared-memory tiled
- GPU training: linear regression and XOR MLP with CUDA SGD/AdamW
- not implemented: Conv2d, pooling, CrossEntropy, batched MatMul, F16/BF16,
  CUDA Graph, Tensor Core, CPU MemoryPlan reuse on CUDA

GPU results (actual T4 runs):

- MLP CPU vs CUDA forward: PASS
- linear regression: loss `2.2e-13`, `w=2.0000005`, `b=0.9999992`
- XOR MLP: loss `4.29e-4`
- 512×512 GEMM: CPU 27.67 ms, CUDA naive 1.32 ms, CUDA tiled 0.46 ms

See [crates/oxide-ml-cuda/README.md](crates/oxide-ml-cuda/README.md) and
[docs/phase6.md](docs/phase6.md).

## Phase 7F-13 event failure safety

`crates/oxide-ml-event-safety/` is a CUDA-independent crate that owns the
event state machine, the free/pending/quarantine pool, destroy diagnostics and
the deferred-reclamation predicate. The CUDA backend supplies a generic
production factory and a raw `CUevent` wrapper; CPU tests supply an injectable
mock backend. Failed events are never returned to a free list, and buffers are
reclaimed only when every attached event is explicitly `Completed`.

Verified on the Colab Tesla T4:

- Event state machine, create/reuse, record/query/synchronize: PASS
- Pending-event retention and deferred reclamation: PASS
- EventPool Drop and context lifetime: PASS
- Normal / planned / arena numerical equality: PASS
- CPU fault-injection suite: 22 new tests (109 workspace tests total)
- Compute Sanitizer memcheck: 0 errors

See [docs/phase7f13-final-report.md](docs/phase7f13-final-report.md).

## Phase 7F-14 arena identity

Persistent Planned Arena lookup now uses a canonical, construction-order
independent identity chain:

```text
Graph -> CanonicalGraph -> GraphHash (BLAKE3)
MemoryPlan -> CanonicalPlan -> PlanHash (BLAKE3)
GraphHash + PlanHash + DeviceId + ContextId + DType + PlanVersion -> ArenaKey
ArenaKey + exact ArenaDescriptor -> CudaArenaRegistry
```

- `Graph::canonicalize()` removes NodeId/ValueId and construction-order
  dependence. Operator attributes and constant tensors are hashed from an
  explicit binary encoding (constants are bitwise, including NaN payloads and
  signed zero).
- `canonicalize_memory_plan()` validates slot capacity/alignment, assignments,
  lifetimes, dtype/shape consistency, and maps ValueId/NodeId to canonical refs.
- `CudaArenaRegistry` reuses an arena only when the key and the exact descriptor
  match. Hash collisions create a separate arena instead of sharing.
- Device identity uses ordinal + UUID + name + compute capability. Context
  identity uses the raw `CUcontext` with a monotonically allocated process-local
  id (never the `Arc` address).
- The generic `ArenaRegistry` supports per-key concurrent lookup, a registry
  capacity limit separate from SlotSet capacity, mutex poisoning recovery, and
  creation-failure isolation.

Verified on the Colab Tesla T4: same graph/plan reuse, different graph/plan
separation, 100-run persistent arena reuse, collision rejection, EventPool
regression, and Normal/Planned/Arena numerical equality.

See [docs/phase7f14-architecture.md](docs/phase7f14-architecture.md) and
[docs/phase7f14-final-report.md](docs/phase7f14-final-report.md).

## Phase 7F-15 async execution lifetime

Asynchronous execution now has an explicit resource bundle and lifecycle:

```text
CudaExecution
  -> ExecutionResources { run, stream, context, completion EventLease, arena lease }
  -> ExecutionState: Submitted / Running / Completed / Collected / Failed
```

- `execute_async_planned` and `execute_async_arena` no longer synchronize the host.
- `is_complete` uses the execution completion Event query (non-blocking).
- `wait` synchronizes the completion event; `collect` downloads outputs and rejects a
  second collect.
- `CudaExecution` Drop defers arena slot sets to `PendingCompletion` and lets buffers
  enter pending reclamation/quarantine. It does not synchronize.
- Executor Drop, Arena Registry removal, External Output ownership, multiple output
  owners, and Context lifetime are covered by CPU Mock tests and T4 checks.

CPU Mock/ownership tests live in `crates/oxide-ml-execution-safety/`.

## Phase 7F-16 AccessGuard / buffer hazards

CUDA kernel launches now go through a per-buffer AccessGuard:

```text
BufferIdentity { id, generation, context_id }
BufferAccessState { readers, writer, waits, stats }
CudaKernels launch wrapper
  -> plan cross-stream waits
  -> launch
  -> record completion EventLease
  -> attach event to buffers
  -> record access state
```

- Read/Read does not wait; Write/ReadWrite waits for all readers on other
  streams; same-stream hazards are elided.
- Buffer reuse from the pool or an arena slot starts a new generation and clears
  the old access state.
- Tensor clones/aliases share the same BufferId and Access State.
- Kernel/event failures invalidate participating buffers so they enter
  quarantine rather than the free list.

See [docs/phase7f16-audit.md](docs/phase7f16-audit.md) and
[docs/phase7f16-final-report.md](docs/phase7f16-final-report.md).

## What works

| Area | Status |
|---|---|
| `DenseTensor<T>` / rank-typed `Tensor<R, T>` | F32 and F64, contiguous row-major |
| Symbolic shapes | `Dim::{Static, Symbol, Dynamic}`, per-run binding |
| SSA Graph IR | dense `ValueId`/`NodeId` tables, op attrs, explicit verifier |
| Versioning | serialization, IR, kernel ABI, backend capability versions separated |
| Shape inference | broadcasting, batched matmul, reduction, conv, pooling |
| CPU executor | static graph, reusable across batches, F32/F64 |
| Reverse-mode autograd | backward graph appended to forward graph |
| Gradient rules | extensible registry; built-ins for all forward ops |
| Graph compiler | pass manager, constant folding, DCE, CSE, Add+ReLU fusion |
| Memory planning | liveness, aligned buffer slots, reusable slots, planned executor |
| NN layers | `Linear`, `LayerNorm`, `Mlp` (F32/F64 constructors) |
| Optimizers | SGD, SGD+momentum, AdamW, RMSProp (generic over scalar) |
| Serialization | versioned JSON round-trip + structural hash |
| Tests | verifier, passes, memory planning, F64 finite differences, convergence |
| Benchmarks | std-only compiler benchmark + T4 CUDA GEMM benchmark |
| CUDA backend | F32 cuda-oxide kernels, device executor, pooling, SGD/AdamW |
| CUDA MemoryPlan | adapter + planned slot writes (sync/default stream); events/async pending |

Not implemented yet: Conv2d/pooling CUDA kernels, CrossEntropy CUDA, batched
MatMul CUDA, F16/BF16, CUDA Graph, Tensor Core, proc macros, RNN/GRU/LSTM,
attention, embedding, BatchNorm, distributed training. See
[docs/phase5.md](docs/phase5.md), [docs/phase6.md](docs/phase6.md), and
[docs/decisions.md](docs/decisions.md).

## Workspace

```
crates/
  oxide-ml-core/       dtype, Scalar, symbolic shape, DenseTensor<T>, MemoryPlan
  oxide-ml-graph/      SSA IR, builder, shape inference, verifier, JSON
  oxide-ml-kernel/     versioned backend ABI (KernelCall / KernelBackend)
  oxide-ml-cpu/        reference kernels, full and memory-planned executors
  oxide-ml-autograd/   reverse-mode graph construction, gradient registry/rules
  oxide-ml-compiler/   pass manager, passes, constant folding, memory planner
  oxide-ml-nn/         Linear / LayerNorm / MLP and parameter initialization
  oxide-ml-optim/      SGD / momentum / AdamW / RMSProp
  oxide-ml-event-safety/ CUDA-independent event state machine/quarantine policy
  oxide-ml-execution-safety/ async execution lifecycle / ownership policy
  oxide-ml-cuda/       standalone cuda-oxide kernels + cuda-core executor
examples/               linear regression, XOR, MNIST MLP
tests/                  gradient_check, graph_ir, ops, training, compiler, memory, f64
benches/                compiler benchmark (no external dependency)
crates/oxide-ml-cuda/src/bin/gpu_check.rs   GPU acceptance test binary
crates/oxide-ml-cuda/src/bin/gemm_bench.rs  CPU/naive/tiled GEMM benchmark
```

Dependency direction is one-way:

```
oxide-ml-core
   ↑        ↑
graph     kernel
   ↑        ↑
autograd  cpu        compiler
   ↑        ↑           ↑
 nn       optim        graph + kernel
      \   /
     oxide-ml
```

`oxide-ml-graph` never sees CUDA or a backend type. `oxide-ml-compiler` depends on
the graph and the `KernelBackend` trait, not on a concrete CUDA backend.

## Quick start

`GraphBuilder` is the graph-construction API: operations append SSA nodes and
return `ValueId`s. `Tensor<R, T>` is a rank-checked data container for concrete
inputs and outputs; it is not a graph node.

```rust
use oxide_ml::prelude::*;

fn main() -> Result<()> {
    let mut g = GraphBuilder::new();
    let x = g.input("x", Shape::from_static([16, 1]), DType::F32);
    let y = g.input("y", Shape::from_static([16, 1]), DType::F32);
    let w = g.try_parameter("linear.weight", Shape::from_static([1, 1]), DType::F32, true)?;
    let b = g.try_parameter("linear.bias", Shape::from_static([1]), DType::F32, true)?;
    let product = g.matmul(x, w)?;
    let prediction = g.add(product, b)?;
    let loss = g.mse_loss(prediction, y, Reduction::Mean)?;
    let forward = g.finish(&[prediction, loss])?;

    let training = build_training_graph(&forward, loss)?;
    let parameters = init_parameters(&forward, 7)?;
    let optimizer = AdamW::with_config(1e-2, 0.9, 0.999, 1e-8, 0.0)?;
    let mut trainer = Trainer::new(training, parameters, Box::new(optimizer))?;

    let mut inputs = ExecutionInputs::new();
    inputs.insert(forward.inputs()[0], DenseTensor::from_vec([16, 1], vec![0.0; 16])?);
    inputs.insert(forward.inputs()[1], DenseTensor::from_vec([16, 1], vec![1.0; 16])?);
    let loss = trainer.step(&inputs)?;
    println!("loss = {loss}");
    Ok(())
}
```

Compile and run with a memory plan:

```rust
let compiler = Compiler::<f32>::new(
    Box::new(CpuBackend::<f32>::new()),
    CompileOptions::default(),
);
let compiled = compiler.compile(graph)?;
let plan = compiled.memory_plan.as_ref().unwrap();
let executor = Executor::<f32>::new(compiled.graph.clone())?;
let (result, stats) = executor.run_planned(&params, &inputs, plan)?;
println!("peak planned bytes = {}", stats.planned_peak_bytes);
```

`run_planned` retains only graph outputs. A training graph explicitly outputs its
loss and parameter gradients, so `collect_gradients` and `Trainer::step_planned`
work with the planned executor. Use `Executor::run` when every intermediate value
must remain inspectable.

Run the checked examples:

```bash
cargo test --workspace
cargo run --example linear_regression
cargo run --example xor
python3 scripts/fetch_mnist.py data/mnist
cargo run --release --example mnist_mlp -- data/mnist 2 5000
cargo bench -p oxide-ml --bench compiler
```

## Graph IR and versions

`Graph::to_json()` uses a versioned envelope (`format_version: 2`) with a separate
semantic `ir_version: 1`. Version 1 envelopes are migrated on read; unknown
envelope or IR versions are rejected. `GraphVerifier::verify` is the gate used
by both the executor and compiler; it checks dense ids, SSA uniqueness,
definition-before-use, cycles, shapes, dtypes, operation arity/attrs, and graph
inputs/outputs. A graph that fails verification never reaches optimization or
execution.

Version constants:

| Constant | Current | Meaning |
|---|---:|---|
| `SERIALIZATION_FORMAT_VERSION` | 2 | JSON envelope layout/encoding |
| `IR_VERSION` | 1 | semantic node/value meaning |
| `KERNEL_ABI_VERSION` | 2 | `KernelCall`/`KernelBackend` contract |
| `BACKEND_CAPABILITY_VERSION` | 1 | optional capability descriptor |

## Graph compiler

`PassManager` runs independent passes in registration order, verifies the graph
before/after every pass, records node-count changes and timings, and can be
configured per pass. Default order:

1. `ConstantFolding` evaluates constant-only nodes through the backend and
   rewrites them to `Op::Constant`.
2. `CSE` merges identical single-output operations. It compares serialized op,
   attrs, and resolved inputs; it never reassociates floating-point arithmetic.
3. `OperatorFusion` fuses `Add + ReLU` into `AddRelu` when the Add result has one
   use and ownership/root conditions are safe.
4. `DeadCodeElimination` removes nodes unreachable from graph outputs. Current
   IR operations are pure; a future side-effecting op must be explicitly treated
   as a root.

`CompileOptions` can disable optimization globally or toggle individual passes
through `PassToggles`; `PassManager::set_enabled` is available for custom pass
lists. `set_dump(true)` records before/after textual graph dumps in the report.

## Memory planning

`MemoryPlanner` computes each value's definition/last-use interval, excludes
caller-owned inputs and parameters, and assigns static-size values to aligned
buffer slots. Slots are reused when lifetimes do not overlap. Graph outputs are
live to the end. Symbolic-shape values are left unplanned and are materialized
normally, so planning never guesses a batch dimension.

The planner is intentionally conservative: in-place aliasing is not used, each
value owns a whole slot, and a planned execution retains only roots. Kernels
still allocate their temporary result tensor; the plan controls the retained
activation arena. This is enough to validate reuse semantics on the CPU and gives
the CUDA backend a concrete plan format, but it is not yet a zero-copy allocator.

## F64 reference

`DenseTensor<T>` and `ParameterStore<T>` default to F32, so existing code keeps
using the short type names. F64 uses `DenseTensor::<f64>`,
`ParameterStore::<f64>`, `Executor::<f64>`, and `init_parameters::<f64>`. The
kernel body is generic over `Scalar`; F16/BF16 are rejected rather than silently
converted. F64 gradient checks use `eps = 1e-6` and match within `1e-7` in the
test suite.

## Testing and benchmarks

Current suite (local CPU):

- `tests/graph_ir.rs`: invalid ids, definition-before-use, shape mismatch,
  JSON/file round-trip, legacy version migration, unsupported versions
- `tests/compiler.rs`: constant folding, DCE, CSE (including transitive use),
  Add+ReLU fusion, pass order, pass toggles, optimized backward, planned run
- `tests/memory.rs`: liveness, aligned slots, buffer reuse, planned/unplanned
  equality, input/output safety, training roots
- `tests/gradient_check.rs`: F32 finite differences for MLP/MSE, softmax cross
  entropy, LayerNorm, Conv2d, broadcast bias
- `tests/f64.rs`: F64 finite differences, F64 AdamW, F64 MLP/compiler/planned run,
  F16 rejection
- `tests/ops.rs`: transposed matmul, activation/pooling gradients, shared-param
  accumulation, AddRelu gradient, custom gradient rule, invalid bindings
- `tests/training.rs`: linear regression, XOR, symbolic batches, inference plan
- `tests/graph_ir.rs` also serializes optimized/training graphs

Run:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p oxide-ml --bench compiler
```

Benchmark sample (release, this workstation; timings vary by run):

| Graph | Scenario | Compile µs | Nodes | Non-const | Exec µs | Peak B | Allocs |
|---|---|---:|---:|---:|---:|---:|---:|
| small | no-opt/no-plan | 76.7 | 16 | 14 | 41.0 | 6,724 | 16 |
| small | full-opt/plan | 146.3 | 11 | 10 | 32.1 | 1,024 | 2 |
| small | constant-fold | 46.1 | 16 | 13 | 34.2 | 1,024 | 2 |
| small | fusion | 52.4 | 13 | 11 | 37.2 | 1,536 | 3 |
| large | no-opt/no-plan | 161.4 | 55 | 53 | 4,232.2 | 426,244 | 55 |
| large | full-opt/plan | 750.0 | 37 | 36 | 3,952.8 | 16,384 | 2 |
| large | constant-fold | 257.5 | 55 | 52 | 4,051.1 | 16,384 | 2 |
| large | fusion | 156.8 | 39 | 37 | 4,093.5 | 24,576 | 3 |

Optimization increases compile time. Execution-time deltas at this scale are
small and run-dependent; the harness reports every scenario so regressions stay
visible rather than claiming optimization is always a runtime win. The memory
plan has a clear effect: the large graph drops from an estimated 426,244 bytes
and 55 allocations to 16,384 bytes and 2 allocations.

## Colab CLI and GPU scope

CPU tests and examples run locally; they do not need a GPU runtime. This
workstation has no GPU, so Colab is reserved for GPU-specific work. A probe
confirmed that the available runtime is a Tesla T4 with CUDA enabled, but the
current milestone has no CUDA kernels to run. Phase 6 CUDA tests and benchmarks
will be run through the Colab CLI; CPU parity tests stay local.

GPU kernel results from the T4 Phase 6 runs are recorded in
[crates/oxide-ml-cuda/README.md](crates/oxide-ml-cuda/README.md).
Phase 7's MemoryPlan adapter, planned slot execution, async execution, CUDA
Event wrapper, deferred release and pool statistics are GPU-verified on T4.
EventPool/EventState, deferred release, persistent arena reuse, Arena Key
registry, capacity/concurrent leasing, external-output async Drop and a GPU
Read/Read/Write test are verified. Event-failure quarantine, AccessGuard kernel
integration, dynamic-shape fallback and the full 5-mode benchmark remain
incomplete; see [docs/phase7f-final-report.md](docs/phase7f-final-report.md).

## Next phases

1. Phase 6: CUDA backend behind the `KernelBackend` ABI using cuda-oxide, tiled
   GEMM, and the compiled memory plan.
2. Proc macros: `#[derive(Module)]`, `#[model]`, `graph!`.
3. More fusion patterns (MatMul+Bias, Mul+Add) and a cost model.
4. True in-place buffer assignment and workspace planning.
5. RNN/GRU/LSTM, attention, embedding, BatchNorm, checkpoint serialization.
