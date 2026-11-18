# Phase 6: CUDA backend survey and implementation

## 1. cuda-oxide investigation

There are two unrelated projects named `cuda-oxide`:

1. **crates.io `cuda-oxide 0.4.0`** (Protryon/cuda-oxide): a safe CUDA Driver
   API wrapper. It does not compile Rust to PTX, is GPL-3.0-or-later, and
   explicitly lists events, memory pools, and graphs as unsupported. It is not
   NVIDIA's codegen project.
2. **NVIDIA `cuda-oxide`** (github.com/NVIDIA/cuda-rust): a rustc codegen
   backend plus `cargo-oxide`; device code is written in Rust with
   `#[cuda_module]`/`#[kernel]`, compiled to PTX, and launched through the
   `cuda-core`/`cuda-host` host runtime.

The environment has no local GPU or CUDA toolkit, so NVIDIA cuda-oxide was
investigated and exercised on Colab. The actual NVIDIA runtime is usable there:

- `cargo-oxide 0.4.0` builds and runs the stock vecadd probe on a T4.
- Device kernels in this crate use `#[cuda_module]`, `#[kernel]`, `DisjointSlice`,
  `SharedArray`, and `thread` APIs from NVIDIA `cuda-device`.
- Host launches are generated typed methods from `cuda-host`.

NVIDIA cuda-oxide is **not** a normal library dependency; it is a custom rustc
backend. Therefore `oxide-ml-cuda` is a standalone workspace, excluded from the
CPU workspace, and built with `cargo oxide`, not `cargo build`.

## 2. ABI connection

The existing `KernelBackend<f32>` ABI is implemented by `CudaBackend`, but the
main execution path is a device-resident `CudaExecutor`:

- Graph input `DenseTensor<f32>` → `DeviceBuffer<f32>` H2D.
- All intermediate values stay in `CudaTensor` device buffers.
- Graph outputs are copied D2H only at the end.
- Parameters live in `CudaParameterStore` across steps.
- The existing `TrainingGraph` supplies the backward graph; no autograd rewrite.

`KernelCall` per-op adapter does H2D/D2H because the ABI takes host tensors. It
is for conformance and one-op comparisons; it is not used for training.

## 3. Runtime

- `CudaDevice` owns the primary context and exposes device name, compute
  capability, total/free memory, launch limits, shared memory, warp size.
- Streams: default stream plus `new_stream`; kernel launches are stream-ordered.
- Device memory: RAII `DeviceBuffer`; a size-class pool tracks allocations,
  reuses, current/peak bytes, and releases.
- Errors: typed `CudaError`; PTX/JIT and launch errors propagate; unsupported
  operations are explicit, never CPU fallback.

## 4. Implemented operations

Constant, ConstantScalar, Add/Sub/Mul/Div/AddRelu, Neg, ReLU/GELU/Sigmoid/Tanh,
Transpose, Reshape, BroadcastTo, ReduceSum/Mean, MatMul (naive + 16×16 shared
memory tiled), MSE forward/backward, ActivationBackward, AddReluBackward,
Unbroadcast, Softmax, LayerNorm, SGD, AdamW.

Not implemented: Conv2d, pooling, CrossEntropy, SoftmaxBackward,
LayerNormBackward, Conv2dBackward, batched MatMul, F64 CUDA, CUDA Graph,
Tensor Core.

## 5. Verification

GPU checks were actually executed on Colab Tesla T4 (logs in the session):

- Device info: Tesla T4, cc 7.5, 15.6 GB, warp 32.
- Device buffer round-trip: PASS.
- CPU vs CUDA MLP forward: PASS.
- CPU vs CUDA MLP backward gradients: PASS.
- GPU linear regression (SGD, 300 steps): loss `2.2e-13`, learned
  `w=2.0000005`, `b=0.9999992`.
- GPU XOR MLP (2→8→1, SGD 2500 steps): loss `4.29e-4`.
- Device pool during training: 32 allocations, 69,972 reuses, peak 1,536 B.

GEMM benchmark (T4, `sm_75`, 20 iterations, max error vs CPU 0):

| Size | CPU | CUDA naive | CUDA tiled |
|---:|---:|---:|---:|
| 64 | 119.7 µs | 35.3 µs | 12.2 µs |
| 128 | 556.4 µs | 58.9 µs | 19.3 µs |
| 256 | 3,617.2 µs | 200.6 µs | 73.2 µs |
| 512 | 27,672.6 µs | 1,323.9 µs | 459.4 µs |

## 6. Constraints

- Build requires `cargo oxide --arch sm_75` on T4. Without `--arch`, cuda-oxide
  emits `sm_80` PTX and the T4 driver rejects it with JIT error 218.
- `cargo oxide test --arch sm_75` hit an upstream “embedded artifact payload is
  empty” issue; GPU acceptance is therefore a runnable binary (`gpu_check`)
  rather than `cargo test`.
- CPU `MemoryPlan` is not yet wired into CUDA buffer reuse. The device pool
  provides allocation reuse, but executor values are still per-node buffers.
- No cuBLAS comparison; only CPU vs naive/tiled.
