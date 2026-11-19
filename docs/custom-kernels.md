# Custom kernels

A custom kernel is a user-provided operation described by a `KernelDescriptor`
and executed through a `CustomKernelRegistry`. The graph IR stores only the
kernel name, input arity, declared output shapes, and dtype (`Op::Custom`), so a
graph remains backend-independent. Bodies are supplied by the caller: on CPU a
reference implementation, on CUDA a `#[cuda_module]`/`#[kernel]` launch.

## Descriptor

```rust
use std::sync::Arc;
use oxide_ml::prelude::*;

fn activation_descriptor() -> KernelDescriptor {
    KernelDescriptor::new(
        "custom_activation",
        DType::F32,
        vec![TensorConstraint::any().rank(2)],
        vec![TensorConstraint::any().rank(2)],
    )
    .with_launch(LaunchConfig::one_dimensional(256))
    .with_architecture("sm_75")
    .with_autograd()
}
```

| Type | Purpose |
| --- | --- |
| `KernelDescriptor` | Name, `abi_version` (currently 1), dtype, input/output constraints, launch config, optional architecture, autograd metadata, side effects. |
| `TensorConstraint` | `any()`, `contiguous(dims, dtype)`, `.rank(n)`, `.dtype(dtype)`. |
| `Dimension` | `Any`, `Fixed(n)`, or `SameAs { tensor, axis }`. |
| `LaunchConfig` | `one_dimensional(block)`, `grid_for(elements)`, shared memory, elements per thread. |
| `SideEffect` | `None`, `ReadsInput`, `WritesWorkspace`, `AtomicAccumulation`. |
| `SpecializationKey` | Kernel, ABI version, architecture, dtype, input/output shapes, layout, constants. |

`KernelDescriptor::validate` rejects an empty name, a wrong ABI version, no
outputs, or a zero launch dimension. `CustomKernel::new` additionally requires
the descriptor dtype to match the kernel's scalar type parameter (`T::DTYPE`).
`KernelOperator::new(descriptor)` wraps a descriptor and exposes its name and
optional static output shapes.

## Register and run on CPU

```rust
fn activation_forward(call: &CustomCall<'_, f32>) -> Result<Vec<DenseTensor<f32>>> {
    let input = &call.inputs[0];
    Ok(vec![input.map(|value| value / (1.0 + value.abs()))])
}

fn activation_backward(
    builder: &mut GraphBuilder,
    node: &Node,
    gradients: &[Option<ValueId>],
) -> Result<Vec<Option<ValueId>>> {
    let dy = gradients
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| Error::Autograd("custom activation has no upstream gradient".into()))?;
    let x = node.inputs[0];
    let absolute = builder.abs(x)?;
    let one = builder.constant_scalar(1.0);
    let denominator = builder.add(absolute, one)?;
    let squared = builder.mul(denominator, denominator)?;
    let dx = builder.div(dy, squared)?;
    Ok(vec![Some(dx)])
}

fn registry() -> Arc<CustomKernelRegistry<f32>> {
    let mut registry = CustomKernelRegistry::new();
    let kernel = CustomKernel::new(activation_descriptor(), Arc::new(activation_forward))
        .unwrap()
        .with_backward(Arc::new(activation_backward));
    registry.register(kernel).unwrap();
    Arc::new(registry)
}
```

`CustomKernelRegistry` provides `new`, `register` (duplicate names are
rejected), `get`, `contains`, `names`, `len`, `specialize`, `execute`,
`backward`, `backward_rules`, and `has_backward`. `execute` validates input
constraints and output shapes against the descriptor, then calls the registered
forward function.

```rust
let mut builder = GraphBuilder::new();
let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
let y = builder
    .custom(
        "custom_activation",
        &[x],
        vec![Shape::from_static([2, 2])],
    )
    .unwrap()[0];
let graph = builder.finish(&[y]).unwrap();

let executor = Executor::with_custom(graph, registry()).unwrap();
let mut inputs = ExecutionInputs::<f32>::new();
inputs.insert(
    x,
    DenseTensor::from_vec([2, 2], vec![1.0, -1.0, 2.0, -2.0]).unwrap(),
);
let result = executor.run(&ParameterStore::new(), &inputs).unwrap();
assert_eq!(result.output(0).unwrap().dims(), vec![2, 2]);
```

`GraphBuilder::custom(name, &[ValueId], Vec<Shape>) -> Result<Vec<ValueId>>`
adds `Op::Custom`; shape inference returns exactly the declared output shapes.
If a graph contains `Op::Custom` but the executor was created with
`Executor::new`, CPU execution returns an explicit error: the executor has no
custom kernel registry.

## Use from a model

`#[model]` bodies can call `apply_kernel(x, "name")` when the output shape equals
the input shape, or `apply_kernel_shaped(x, "name", [dims])` for an explicit
static shape. Kernels that change arity or need several inputs should be wired
through a hand-written `Module::forward` and `GraphBuilder::custom`.

```rust
#[derive(Module)]
struct CustomModel {
    linear: Linear,
}

impl CustomModel {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let x = self.linear.forward(x);
        apply_kernel(x, "custom_activation")
    }
}

let model = CustomModel {
    linear: Linear::with_seed(2, 2, true, 3),
};
let compiled = CompiledModel::compile(
    &model,
    &[InputSpec::unbatched("input0", [2, 2])],
    ModelCompileOptions {
        custom_kernels: Some(registry()),
        ..Default::default()
    },
)?;
```

`Trainer::builder().custom_kernels(registry)` installs the registry for both
forward and training graphs.

## Autograd

A backward rule is registered with `CustomKernel::with_backward`. The autograd
registry dispatches it through `CustomKernelRule`:

```rust
let gradient_registry = GradientRegistry::with_custom_kernels(registry().as_ref());
let training = build_training_graph_with_registry(&graph, loss, &gradient_registry)?;
```

When `CompiledModel` is given `custom_kernels` and no explicit
`gradient_registry`, it builds `GradientRegistry::with_custom_kernels`
automatically. If a custom kernel has no backward rule, the error is reported
while **building** the training graph, not during execution:
`custom kernel "..." has no backward rule; register one with
CustomKernel::with_backward`. `KernelDescriptor::with_autograd()` only marks the
descriptor; it does not synthesize a backward implementation.

## CUDA

The CUDA implementation lives in the standalone `crates/oxide-ml-cuda`
workspace (excluded from the root workspace) and is built with the `cargo oxide`
toolchain, for example `cargo oxide build --arch sm_75`.

```rust
// Not runnable in this checkout: no CUDA toolkit, GPU, or cuda-oxide.
use oxide_ml_cuda::prelude::*;

let device = CudaDevice::new(0)?;
let mut registry = CudaKernelRegistry::new();
registry.register(activation_descriptor(), |executor, stream, inputs, outputs| {
    // Launch a #[cuda_module] #[kernel] wrapper. Allocate outputs from
    // executor.pool() so access tracking and deferred reclamation work.
    let _ = (executor, stream, inputs, outputs);
    todo!("launch the device kernel and return Vec<CudaTensor>")
})?;
let executor = CudaExecutor::new(device.clone())?
    .with_custom_kernels(std::sync::Arc::new(registry));
```

`CudaKernelRegistry::register(descriptor, launch)` takes a launch function with
the signature `Fn(&CudaExecutor, &CudaStream, &[CudaTensor], &[Vec<usize>]) ->
Result<Vec<CudaTensor>>`; only F32 is accepted. `CudaExecutor::with_custom_kernels`
dispatches `Op::Custom`; without a registry it returns
`CudaError::Unsupported`.

The CUDA model wrappers reuse the same names and graph:

| API | Use |
| --- | --- |
| `CudaModel::compile(device, model, inputs, options)` | Forward-only, device-resident parameters. |
| `CudaModel::run(&[(name, DenseTensor)])` | Download only the declared outputs. |
| `CudaModel::host_parameters()` | Download a host `ParameterStore`. |
| `CudaModel::sync_to_model(&mut model)` | Write trained parameters back. |
| `CudaTrainingModel::new(device, model, inputs, target, loss, optimizer)` | Device training graph. |
| `CudaTrainingModel::step(inputs, targets)` | One optimizer step; returns the loss. |

`cargo oxide build --arch sm_75` is the documented build command. **These CUDA
APIs were not compiled or run for this milestone**: this environment has no CUDA
toolkit, GPU, or cuda-oxide, and the Colab session is unavailable. The
pre-existing Phase 7F CUDA results (T4, sm_75, CUDA 13.0) remain in
[phase7f.md](phase7f.md) and the README but were not re-run. The new IR ops
`Sqrt`, `Exp`, `Log`, `Abs`, `LogSoftmax`, `Slice`, `Embedding`, `AbsBackward`,
`SliceBackward`, and `EmbeddingBackward` have no CUDA kernels and return an
explicit `CudaError::Unsupported`.

## Limitations

- The graph records only the kernel name, arity, output shapes, and dtype. A
  shape-changing kernel must be described at graph construction time
  (`GraphBuilder::custom` or `apply_kernel_shaped`).
- `apply_kernel` assumes the output shape equals the input shape.
- A CPU graph with `Op::Custom` requires `Executor::with_custom`; otherwise
  execution fails with an explicit "no custom kernel registry" error.
- Custom autograd needs an explicit `CustomBackward`; the descriptor's autograd
  flag is metadata only.
- Only F32 custom kernels are supported by the CUDA registry.
- CUDA custom launches must use the executor's stream and buffer pool; the
  registry never generates PTX or bypasses `AccessGuard`/event handling.
- No CUDA verification was performed in this milestone.
