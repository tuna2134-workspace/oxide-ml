# Phase 7F Closure Audit

> **Status note (2026-11-19):** final status is frozen in `phase7f-final-closure.md`.
> This audit describes the mid-closure state and is retained as history.

## 現状

- Phase 7F-16: AccessGuard / BufferIdentity / 全 CudaKernels Launch 統合済み。
- CUDA Executor は Symbolic Shape を runtime bind するが、`static_shape` 依存が残っていた。
- MemoryPlanner は Symbol value を unplanned にしていた。
- Shape Signature / Graph Specialization は実装済み。
- Async API は Host Synchronize なし。2 Stream Overlap 実測済み。
- Sanitizer は access_check のみ。gpu_check 全体 / 4 Stream / Dynamic は未実施。
- Benchmark は Small MLP + Dynamic のみ。

## 今回の変更点

1. `runtime_shapes()` を追加し、実入力から全 Node 出力 shape を
   `infer_outputs` で具体化。`Dim::Dynamic` 出力も Pool で実行可能。
2. `CudaExecutor::plan_bounded(graph, max_bindings)` を追加。
   max shape で graph を specialize し、max capacity の MemoryPlan を作る。
3. `validate_runtime_capacity()` を追加。
   symbol max / runtime bytes / alignment / offset / slot capacity を Kernel Launch
   前に検証。
4. `CudaError::Capacity` を追加。Capacity Overflow は明示エラー。
5. `CudaMemoryPlan` に capacity bindings / max_capacity_bytes を追加。
6. 4 Stream Overlap Test / 4 Stream Shared Read/Write / Arena capacity 4 を追加。
7. Dynamic / 4 Stream / full gpu_check の Sanitizer を実行。
8. Benchmark に MatMul / Activation / Reduction / Large MLP / XOR を追加。

## T4

- Tesla T4 / sm_75 / CUDA 13.0
- gpu_check: 40 PASS
- concurrency4_check: 4 Stream Overlap PASS
- dynamic_check: PASS
- race4_check: PASS
- benchmark_check: PASS

## 残課題

- Backward Graph / Linear Regression の benchmark。
- 4 Stream benchmark matrix（benchmark binary 内）。
- Circular Event Wait の明示的 deadlock test は API 上構築しない方針。
- CUDA Driver 実 Fault の注入は SOURCE ONLY。