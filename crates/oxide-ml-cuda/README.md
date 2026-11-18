# oxide-ml-cuda

CUDA backend for oxide-ml, implemented with **NVIDIA cuda-oxide** for device
kernels and `cuda-core` for host runtime/memory/streams.

## Toolchain

Verified on the Colab Tesla T4 runtime:

- NVIDIA `cuda-oxide` 0.4.0 (git `860375a1`)
- Rust 1.99.1 with `rust-src`, `rustc-dev`, `llvm-tools`
- CUDA Toolkit 13.0, driver 580.82.07
- clang/libclang (Ubuntu clang 18 was sufficient once installed)
- GPU compute capability 7.5 (`sm_75`)

Install the Rust toolchain and `cargo-oxide`:

```bash
rustup toolchain install 1.99.1 --profile minimal
rustup component add rust-src rustc-dev llvm-tools --toolchain 1.99.1
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide --locked
sudo apt install clang libclang-dev
```

The crate is a standalone workspace and is excluded from the CPU workspace
because a normal `cargo build` cannot generate cuda-oxide device artifacts.

## Run GPU checks

```bash
cd crates/oxide-ml-cuda
export CUDA_HOME=/usr/local/cuda
export CUDA_TOOLKIT_PATH=/usr/local/cuda

cargo oxide run --bin gpu_check --arch sm_75
cargo oxide run --bin gemm_bench --arch sm_75
```

`gpu_check` verifies device info, device buffer round-trip, CPU-vs-CUDA MLP
forward, GPU linear regression training, and GPU XOR training. `gemm_bench`
compares CPU GEMM against naive and shared-memory tiled CUDA GEMM.

## Architecture

- `device` — `CudaDevice` / `CudaDeviceInfo`, context and stream ownership.
- `buffer` — `DeviceBuffer` RAII wrappers, typed `CudaTensor`, size-class pool
  with allocation/reuse/peak statistics.
- `kernels` — `#[cuda_module]` device kernels compiled by cuda-oxide and typed
  launch wrappers using `cuda-core`.
- `executor` — device-resident graph execution. Only graph inputs are uploaded
  and graph outputs are downloaded; intermediates stay in device memory.
- `param` — persistent device-resident parameter store.
- `optimizer` — `CudaSgd` and `CudaAdamW` device kernels.
- `trainer` — reuses the existing autograd `TrainingGraph`; backward is the same
  graph as CPU, executed by the CUDA executor.
- `backend` — `KernelBackend<f32>` adapter for ABI conformance. It is a
  one-op H2D/execute/D2H adapter; the graph executor is the device-resident path.

## Current limits

- F32 only; F64 CUDA is not implemented.
- Rank-2 MatMul only; batched MatMul/transpose dispatch is not implemented.
- Conv2d, pooling, CrossEntropy, SoftmaxBackward, LayerNormBackward, and the
  remaining backward rules are rejected with clear `Unsupported` errors.
- Tiled GEMM is shared-memory 16x16, no Tensor Core, no cuBLAS.
- The CPU `MemoryPlan` is not yet used to reuse CUDA buffers; the device pool
  reuses allocations by size but executor outputs are still allocated per value.
- No CUDA Graph capture.
