# oxide-ml

Rust-native, graph-based machine-learning framework. Define models as plain
Rust structs, compile them into a symbolic-shape SSA graph with a custom graph
compiler, and run training/inference on CPU or CUDA.

```text
Module definition -> parameter discovery -> graph IR -> verification
                  -> optimization -> memory planning -> CPU / CUDA execution
```

## Quick start

```rust
use oxide_ml::prelude::*;

#[derive(Module)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = tanh(self.fc1.forward(x));
        self.fc2.forward(hidden)
    }
}

let mut model = Mlp { fc1: Linear::new(2, 8), fc2: Linear::new(8, 1) };
let dataset = TensorDataset::from_tensors(inputs, targets)?;
let mut trainer = Trainer::builder()
    .optimizer(AdamW::new(1e-2)?)
    .loss(MSELoss::new())
    .batch_size(4)
    .build()?;
trainer.fit(&mut model, &dataset, 100)?;
let prediction = model.predict(input)?;
```

Run the complete example:

```bash
cargo run -p oxide-ml --example model_xor
```

## Integrated model API (this milestone)

| Area | Status |
|---|---|
| `#[derive(Module)]`, `#[model]`, `graph!` | implemented, CPU tested |
| Module hierarchy, stable parameter names, shared-parameter aliases | implemented, CPU tested |
| `Sequential`, train/eval state, module descriptors | implemented, CPU tested |
| `CompiledModel` over the existing graph compiler + memory planner | implemented, CPU tested |
| `Trainer` (datasets, DataLoader, scheduler, metrics-lite, validation) | implemented, CPU tested |
| Checkpoints (model + optimizer state, versioned) | implemented, CPU tested |
| Custom kernels as graph operators (`Op::Custom`) | implemented, CPU tested |
| Compile cache (canonical hash + structural fast path, LRU, single-flight) | implemented, tested |
| Kernel provider catalog + deterministic selection | implemented, tested |
| Compiler report (passes, fusions, kernels, memory, timing, diagnostics) | implemented, tested |
| Fusion: Add+ReLU, Mul+Add, MatMul+Bias | implemented, tested |
| `Runtime` API (`builder`, `Device::cuda`, `run_with`) | implemented, CPU tested; CUDA is explicit |
| Training: gradient clipping, accuracy metric, early stopping | implemented, tested |
| New ops: Cast/Concat/Split/MulAdd/MatMulBias | implemented, CPU tested |
| Control-flow IR (Region/Cond/While) | **design only** ([control-flow-ir.md](docs/control-flow-ir.md)) |
| Mixed precision / distributed | **not implemented** |
| 23 NN layers + `Sequential` + 5 losses | implemented, CPU tested |
| CUDA model integration (`CudaModel`, `CudaTrainingModel`, custom kernel registry) | implemented; **PASS on T4** (`model_check`, `custom_kernel_check`, `new_ops_check`, `gpu_check`, sanitizers) |

The CPU reference remains the numerical oracle. The CUDA backend never falls
back to CPU for an unsupported operation; new IR ops such as `Sqrt`, `Exp`,
`LogSoftmax`, `Slice` and `Embedding` return an explicit `CudaError::Unsupported`
until device kernels are added.

Documentation:

- [docs/compiler-framework.md](docs/compiler-framework.md)
- [docs/operator-support-matrix.md](docs/operator-support-matrix.md)
- [docs/benchmark-report.md](docs/benchmark-report.md)
- [docs/control-flow-ir.md](docs/control-flow-ir.md)
- [docs/model-api.md](docs/model-api.md)
- [docs/proc-macros.md](docs/proc-macros.md)
- [docs/custom-kernels.md](docs/custom-kernels.md)
- [docs/training-and-checkpoints.md](docs/training-and-checkpoints.md)
- [docs/cuda-backend.md](docs/cuda-backend.md)
- [docs/migration-0.1.md](docs/migration-0.1.md)
- [docs/known-limitations.md](docs/known-limitations.md)
- [docs/verification-report.md](docs/verification-report.md)
- [docs/architecture.md](docs/architecture.md) (graph/compiler/autograd internals)

## Workspace layout

| Crate | Responsibility |
|---|---|
| `oxide-ml-core` | dtypes, devices, symbolic shapes, dense tensors, parameter store |
| `oxide-ml-graph` | SSA graph IR, shape inference, verifier, canonical hashing |
| `oxide-ml-kernel` | backend ABI, custom kernel descriptors and registry |
| `oxide-ml-cpu` | reference kernels and the CPU executor |
| `oxide-ml-autograd` | reverse-mode autograd, `TrainingGraph`, gradient registry |
| `oxide-ml-compiler` | constant folding, CSE, fusion, DCE, memory planning |
| `oxide-ml-optim` | SGD, momentum, AdamW, RMSProp, portable optimizer state |
| `oxide-ml-model` | `Module`, `CompiledModel`, `Trainer`, datasets, checkpoints, `Tensor` |
| `oxide-ml-macros` | `#[derive(Module)]`, `#[model]`, `graph!` |
| `oxide-ml-nn` | model-API layers/losses and the original builder layers (`legacy`) |
| `oxide-ml` | facade, prelude, `GraphTrainer`, `legacy` compatibility module |
| `oxide-ml-event-safety`, `oxide-ml-execution-safety` | Phase 7F safety state machines |
| `oxide-ml-cuda` | standalone CUDA workspace (cuda-oxide kernels + cuda-core runtime) |

## CUDA verification (this milestone, Tesla T4)

`cargo oxide build --arch sm_75` succeeded with CUDA 13.0, driver 580.82.07,
Rust 1.99.1 and `cargo oxide 0.4.0`:

| Check | Result |
|---|---|
| CPU/CUDA forward parity (`model_check`) | PASS, max abs diff `7.749e-7` |
| CPU/CUDA 20-step training parity | PASS, `0.274796` vs `0.274798` |
| Post-training parameter parity | PASS, max diff `2.623e-6` |
| cuda-oxide kernel as `Op::Custom` | PASS, diff `0.000e0` |
| Existing `gpu_check` suite | PASS, 40 checks (capacity test now holds four async leases) |
| `Sqrt`/`Exp`/`Log`/`Abs`/`AbsBackward` on device | PASS, max diff `<= 1.2e-7` |
| BatchNorm forward (training statistics) | PASS, diff `0.0` |
| Compute Sanitizer memcheck/racecheck/synccheck | PASS, 0 errors |

The sections below record earlier milestones; their numbers were not re-run and
remain historical.

## Historical phase results

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

## Phase 7F final status

Phase 7F is **frozen**. The implementation scope is complete; verification is
**partial** and tracked as explicit debt.

| Area | Implementation | Verification |
|---|---|---|
| 7F-13 Event failure safety | COMPLETE | PASS (Mock + T4 normal path) |
| 7F-14 Arena identity / canonical hash | COMPLETE | PASS (T4) |
| 7F-15 Async execution lifecycle | COMPLETE | PASS (T4 + Mock) |
| 7F-16 AccessGuard / buffer hazards | COMPLETE | PASS (T4 + Mock) |
| 7F-17 Dynamic shape / bounded capacity | COMPLETE | PASS (T4); `plan_bounded` does not accept `Dim::Dynamic` |
| 7F-18 Concurrent execution | COMPLETE | PASS (T4); 4-stream matrix 144 PASS / 30 CPU ref unavailable / 3 N/A |
| 7F-19 Benchmark + constant copy recovery | COMPLETE | PARTIAL (sanitizer gaps; mixed synccheck application assertion) |

- Phase 7F Implementation: **COMPLETE / FROZEN**
- Phase 7F Verification: **PARTIAL**
- Phase 8 readiness: **READY** with explicit verification debt.

See [docs/phase7f-final-closure.md](docs/phase7f-final-closure.md),
[docs/phase7f-verification-debt.md](docs/phase7f-verification-debt.md),
[docs/phase7f-operator-backlog.md](docs/phase7f-operator-backlog.md), and
[docs/phase7f-performance-backlog.md](docs/phase7f-performance-backlog.md).

No new CUDA operators or benchmarks are added under the Phase 7F freeze.

## What works (low-level stack)

| Area | Status |
|---|---|
| `DenseTensor<T>` / rank-typed `RankedTensor<R, T>` | F32 and F64, contiguous row-major |
| Symbolic shapes | `Dim::{Static, Symbol, Dynamic}`, per-run binding |
| SSA Graph IR | dense `ValueId`/`NodeId` tables, op attrs, explicit verifier |
| Versioning | serialization, IR, kernel ABI, backend capability versions separated |
| Shape inference | broadcasting, batched matmul, reduction, conv, pooling, new elementwise ops |
| CPU executor | static graph, reusable across batches, F32/F64 |
| Reverse-mode autograd | backward graph appended to forward graph |
| Gradient rules | extensible registry; built-ins for every forward op, including custom kernels |
| Graph compiler | pass manager, constant folding, DCE, CSE, Add+ReLU fusion |
| Memory planning | liveness, aligned buffer slots, reusable slots, planned executor |
| Model API | `#[derive(Module)]`, `#[model]`, `Sequential`, `CompiledModel`, `Trainer` |
| NN layers | 30 model-API modules; the original builder layers remain in `legacy` |
| Optimizers | SGD, SGD+momentum, AdamW, RMSProp; portable state dicts |
| Serialization | versioned graph JSON + versioned checkpoints |
| Custom kernels | `Op::Custom`, descriptors, specialization keys, CPU reference execution |
| CUDA backend | existing cuda-oxide kernels + new CUDA model/custom-kernel APIs (unverified this milestone) |

The CUDA backend still lacks device kernels for `Conv2d`/pooling/CrossEntropy,
batched MatMul, the ten new IR ops, F16/BF16, CUDA Graph and Tensor Core support.
See [docs/known-limitations.md](docs/known-limitations.md).

## Workspace

See the crate table above. Dependency direction is one-way:

```
oxide-ml-core -> oxide-ml-graph -> oxide-ml-kernel -> oxide-ml-cpu/autograd
              -> oxide-ml-compiler -> oxide-ml-optim -> oxide-ml-model
              -> oxide-ml-nn -> oxide-ml (facade)
```

`oxide-ml-graph` never sees CUDA or a backend type. `oxide-ml-compiler` depends on
the graph and the `KernelBackend` trait, not on a concrete CUDA backend.
`oxide-ml-model` has no CUDA dependency; the CUDA crate consumes its graph and
parameter-name contracts.

## Low-level graph API

`GraphBuilder` is the graph-construction API: operations append SSA nodes and
return `ValueId`s. `RankedTensor<R, T>` is a rank-checked data container for concrete
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
    let mut trainer = GraphTrainer::new(training, parameters, Box::new(optimizer))?;

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
cargo run --example model_xor
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

Compiler benchmark (`cargo bench -p oxide-ml --bench compiler`):

| Graph | Scenario | Compile µs | Nodes | Non-const | Exec µs | Peak B | Allocs |
|---|---|---:|---:|---:|---:|---:|---:|
| small | no-opt/no-plan | 68.2 | 16 | 14 | 55.3 | 6,724 | 16 |
| small | full-opt/plan | 172.0 | 11 | 10 | 48.1 | 1,024 | 2 |
| small | constant-fold | 67.6 | 16 | 13 | 48.9 | 1,024 | 2 |
| small | fusion | 55.5 | 13 | 11 | 53.8 | 1,536 | 3 |
| large | no-opt/no-plan | 227.4 | 55 | 53 | 5,775.1 | 426,244 | 55 |
| large | full-opt/plan | 879.3 | 37 | 36 | 4,618.4 | 16,384 | 2 |
| large | constant-fold | 325.5 | 55 | 52 | 3,606.5 | 16,384 | 2 |
| large | fusion | 212.3 | 39 | 37 | 3,826.3 | 24,576 | 3 |

Model benchmark (`cargo bench -p oxide-ml --bench model`), MLP 784-256-10,
batch 64, reference CPU backend:

| Path | Time |
|---|---:|
| compile (cold) | 250 µs |
| compile (cached structural hit) | 10.1 µs |
| compiled inference | 5.0 ms/pass |
| `predict` (cached graph + parameter binding + run) | 6.1 ms/call |
| training step (forward + backward + SGD) | 71–73 ms/batch |

The CPU backend is a correctness reference, not a tuned BLAS; the matmul fast
path uses four independent accumulators but the backward pass and non-Linear
layouts remain naive. Compiler optimization increases compile time and reduces
peak memory; execution-time deltas at this scale are small and run-dependent.
The memory plan drops the large graph from an estimated 426,244 bytes and 55
allocations to 16,384 bytes and 2 allocations.

## CPU and GPU scope

The CPU workspace is the numerical oracle and runs without a GPU
(`cargo test --workspace`). CUDA work runs through the Colab CLI on a Tesla T4
with `cargo oxide build --arch sm_75`. For this milestone, the model integration
(`model_check`), custom-kernel path (`custom_kernel_check`) and the existing
`gpu_check` acceptance suite were executed on the T4, plus Compute Sanitizer.
The remaining acceptance binaries and the Phase 7F benchmark matrix were not
re-run; see [docs/verification-report.md](docs/verification-report.md) for the
exact scope.

## Remaining work

1. Control-flow IR: Region/Block/Cond/While representation, verifier,
   interpreter, autograd and `#[model]` lowering (`if_tensor!`/`while_tensor!`).
2. Mixed precision (F16/BF16 execution, accumulation dtype, loss scaling) and
   the `Cast` execution path.
3. CUDA kernels for the remaining IR ops (`LogSoftmax`, `Slice`, `Embedding`,
   `Concat`, `Split`, `EmbeddingBackward`, `SliceBackward`), plus Conv2d,
   pooling, CrossEntropy and LayerNorm backward.
4. Stochastic dropout (needs a per-run random source in the execution context)
   and BatchNorm running-stat updates.
5. Conv3d, MaxPool1d backward, initial hidden states for RNN/GRU/LSTM.
6. Distributed abstractions (device mesh, collectives) and CPU thread-pool/SIMD.
7. `trybuild` compile-fail UI fixtures and generated-code snapshot tests.
8. Performance: the CPU backend is a correctness reference; the backward pass
   and non-Linear matmul layouts remain unoptimized.
