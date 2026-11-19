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

## Verification status (Tesla T4, sm_75)

Verified in this milestone with CUDA 13.0, driver 580.82.07, Rust 1.99.1 and
`cargo oxide 0.4.0`:

| Check | Result |
|---|---|
| `cargo oxide build --arch sm_75` | PASS (library + model/custom/gpu bins) |
| `model_check` forward parity | PASS, max abs diff `7.749e-7` |
| `model_check` 20 SGD steps parity | PASS, loss `0.274796` vs `0.274798` |
| `model_check` post-training parameters | PASS, max diff `2.623e-6` |
| `custom_kernel_check` | PASS, diff `0.000e0` |
| `gpu_check` (existing suite) | PASS, 40 checks; the capacity-4 test now holds four async leases |
| Fused `MatMulBias` / `AddRelu` (compiler output) | PASS inside `model_check` |
| `Sqrt` / `Exp` / `Log` / `Abs` / `AbsBackward` | PASS, max diff `<= 1.2e-7` |
| BatchNorm forward (training statistics) | PASS, diff `0.0` |
| Compute Sanitizer memcheck/racecheck/synccheck (new ops) | PASS, 0 errors |
| `gpu_check` concurrency capacity test | now uses four async leases; 40 PASS |

The custom check launches through `launch_with_access`, so AccessGuard and the
completion event are exercised. `CudaKernelRegistry` does not acquire the guard
itself; a third-party launch closure must do it.

Still not implemented: `LogSoftmax`, `Slice`, `Embedding`, `Concat`, `Split`,
`Embedding`/`Slice` backward ops, Conv2d/pooling, CrossEntropy, and
LayerNorm backward. They return an explicit `CudaError::Unsupported` (no CPU
fallback). `Cast`, `MulAdd` and `MatMulBias` are executed on the device by
composing existing kernels (copy / elementwise / GEMM), not by a host fallback.
