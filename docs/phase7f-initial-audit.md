# Phase 7F Initial Audit

日付: 2026-11-18  
対象: oxide-ml workspace / CUDA backend / safety crates / T4 environment

## 1. Workspace

CPU workspace:

- oxide-ml-core
- oxide-ml-graph
- oxide-ml-compiler
- oxide-ml-cpu
- oxide-ml-autograd
- oxide-ml-nn
- oxide-ml-event-safety
- oxide-ml-execution-safety

Standalone CUDA crate:

- `crates/oxide-ml-cuda`（workspace 外。`cargo oxide build` でビルド）

## 2. 実装済み

- CPU: DenseTensor / Symbolic Shape / Graph IR / Verifier / Shape Inference /
  CPU Executor / Autograd / NN / Optimizers / MemoryPlan / Planned Executor
- CUDA: Device / Context / Stream / Buffer / Pool / Tensor / Kernel / Event /
  EventPool / Deferred Reclamation / Persistent Arena / Arena Registry /
  Async Execution Lifecycle
- Event Safety: EventState / Pending / Quarantine / Reclamation Decision
- Execution Safety: ExecutionState / ExecutionResources / Mock Failure Injection
- Phase 7F-16: BufferIdentity / AccessGuard / Hazard Tracking / 全 CudaKernels
  Launch Wrapper 統合

## 3. 未実装・不足

1. Dynamic Shape は CPU では Symbol を実行時に bind できるが、CUDA は
   Phase 7F-16 時点で `static_dims()` 必須だった。
2. `Dim::Dynamic`（unbounded）の GPU 実行は未対応。
3. Bounded Dynamic の Capacity 予約 / Capacity Overflow は未実装。
4. AccessGuard の CUDA adapter への実 Fault Injection は未実施。
5. 過去 Benchmark は Host Wall Clock 中心で、GPU Event Timing が不足。
6. GitHub Actions に GPU Runner はない。
7. `cargo check --manifest-path crates/oxide-ml-cuda/Cargo.toml` は
   cuda-oxide standalone 構成のため単独では成立しない。既存方式の
   `cargo oxide build` / `cargo clippy` を使う。

## 4. 環境

- Tesla T4 / sm_75 / CUDA 13.0
- Rust 1.99.1 / cuda-oxide 0.4.0
- Compute Sanitizer 利用可能
- T4 は Colab CLI の trainer session 経由。token は約 1 時間で失効する。
- GPU は 1 台。Cross-GPU / Device Lost の実機試験はできない。

## 5. 既存テスト基準

- CPU workspace: 184 tests（Phase 7F-16 完了時点）
- CUDA `gpu_check`: 37 PASS
- AccessGuard `access_check`: PASS
- Event Safety 22 / Execution Safety 27（Failure Mock 追加前）

## 6. 作業方針

- 既存 Crate 境界を維持。
- Event Safety / Arena Identity / Execution Lifecycle を再実装しない。
- Dynamic Shape は既存 `Shape` / `Dim` / `SymbolId` を再利用。
- Benchmark は CudaEvent elapsed と Host Submission を分離。
- 検証不能な項目は NOT RUN / BLOCKED / SOURCE ONLY と明記。
