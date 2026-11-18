#!/usr/bin/env bash
# Reproducible Colab setup for the Phase 6 CUDA backend.
# Run on a Colab GPU runtime (Tesla T4 verified).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
  sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain 1.99.1
fi
rustup toolchain install 1.99.1 --profile minimal
rustup component add rust-src rustc-dev llvm-tools --toolchain 1.99.1

if ! command -v clang >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq clang libclang-dev
fi

if ! command -v cargo-oxide >/dev/null 2>&1; then
  cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide --locked
fi

cd "$(dirname "$0")/../crates/oxide-ml-cuda"
cargo oxide doctor
cargo oxide run --bin gpu_check --arch sm_75
cargo oxide run --bin gemm_bench --arch sm_75
