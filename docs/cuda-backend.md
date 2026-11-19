# CUDA backend and model integration

`crates/oxide-ml-cuda` is a standalone workspace because NVIDIA cuda-oxide is a
custom rustc backend driven by `cargo oxide`, not a normal crate. It keeps the
Phase 6/7F device-resident executor and safety machinery and adds:

- `oxide_ml_cuda::prelude` — host types, kernel descriptor types and
  re-exports of `cuda_core`, `cuda_device`, `cuda_host`;
- `CudaKernelRegistry` — register a device launch under a
  `KernelDescriptor`; `CudaExecutor` dispatches `Op::Custom` to it;
- `CudaModel` — forward-only compiled model with device-resident parameters;
- `CudaTrainingModel` — training graph + `CudaTrainer`, with
  `sync_to_model` write-back.

## Building on a T4 (reference commands)

```bash
cd crates/oxide-ml/crates/oxide-ml-cuda
export CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cargo oxide build --arch sm_75 -- --bin gpu_check --release
./target/release/gpu_check
```

Toolchain/toolkit used by the pinned environment: Rust 1.99.1 (per
`rust-toolchain.toml`), CUDA 13.x, `--arch sm_75`.

## Same model, CPU and CUDA

```rust
use oxide_ml::prelude::*;
use oxide_ml_cuda::prelude::*;

let device = CudaDevice::new(0)?;
let compiled = CudaModel::compile(
    device,
    &model,
    &[InputSpec::unbatched("input0", [64, 784])],
    ModelCompileOptions::default(),
)?;
let outputs = compiled.run(&[("input0", host_tensor)])?;
```

Training:

```rust
let mut training = CudaTrainingModel::new(
    CudaDevice::new(0)?,
    &model,
    &[InputSpec::batched("input0", [784])],
    InputSpec::batched("target0", [10]),
    &CrossEntropyLoss::new(),
    Box::new(CudaAdamW::new(AdamWParams {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    })?),
)?;
let loss = training.step(&[("input0", batch)], &targets)?;
training.sync_to_model(&mut model)?;
```

The CPU and CUDA paths use the same `BuiltModel` graph, parameter names and
checkpoint names; only the executor differs. CUDA never falls back to CPU: an
operation without a device kernel returns `CudaError::Unsupported`.

## Custom kernels

```rust
let descriptor = KernelDescriptor::new(
    "custom_activation",
    DType::F32,
    vec![TensorConstraint::any().rank(2)],
    vec![TensorConstraint::any().rank(2)],
)
.with_launch(LaunchConfig::one_dimensional(256))
.with_architecture("sm_75")
.with_autograd();

fn register(descriptor: KernelDescriptor) -> oxide_ml_cuda::Result<CudaExecutor> {
    let mut registry = CudaKernelRegistry::new();
    registry.register(descriptor, |executor, stream, inputs, output_shapes| {
        // Launch the `#[cuda_module]` / `#[kernel]` wrapper here.
        // Allocate every output from `executor.pool()` so access tracking and
        // deferred reclamation stay active.
        let _ = (stream, inputs);
        let mut outputs = Vec::with_capacity(output_shapes.len());
        for shape in output_shapes {
            outputs.push(CudaTensor::zeros(executor.pool(), shape.clone())?);
        }
        Ok(outputs)
    })?;
    Ok(CudaExecutor::new(CudaDevice::new(0)?)?.with_custom_kernels(std::sync::Arc::new(registry)))
}
```

The graph side is backend-independent:

```rust
let y = ops::apply_kernel(cx, x, "custom_activation")?;
// or, outside a model body:
let outputs = builder.custom("custom_activation", &[x], vec![shape])?;
```

## Current verification status

**NOT RUN in this milestone.** This environment has no CUDA toolkit, no GPU, no
`cargo oxide`, and the Colab session is expired. The CUDA additions are provided
as API-complete source but were not compiled or executed here. Device kernels
for the ten new IR ops (`Sqrt`, `Exp`, `Log`, `Abs`, `LogSoftmax`, `Slice`,
`Embedding` and their backward ops) are not implemented and return an explicit
unsupported error.

Before relying on these paths, run the CUDA acceptance suite and Compute
Sanitizer on a T4. See `docs/verification-report.md` for the exact commands and
the list of unverified items, including `AccessGuard` integration for custom
kernel launches.
