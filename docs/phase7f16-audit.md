# Phase 7F-16-A 既存コード監査 — AccessGuard / Hazard Tracking

監査日: 2026-11-18  
対象: AccessMode / BufferAccessState / CudaTensor / PooledBuffer / CudaBufferPool /
CudaKernels / CudaPlannedArena / SlotSet / SlotSetLease / CudaExecution /
ExecutionResources / CudaExecutor / CudaStream / CudaEvent / EventLease

## 1. 既存 AccessMode / BufferAccessState

既存 `crates/oxide-ml-cuda/src/access.rs` は次のみだった。

- `BufferAccess::{Read, Write, ReadWrite}`
- `BufferAccessState { last_write, last_write_stream, pending_reads }`
- `wait_for(stream, access)` と `record(stream, access, event)`

問題:

- Buffer Identity / Generation を持たない。
- CudaTensor / PooledBuffer と接続されていない。
- `gpu_check` の単発テスト以外、Kernel Launch 経路に統合されていない。
- Same-stream elision はあるが、Buffer 単位の一貫した状態管理ではない。
- AccessGuard / Drop / EventLease の所有がない。

新しい State Machine を重複実装せず、このルールを純粋な
`oxide_ml_execution_safety::BufferAccessState` へ発展させ、CUDA 側は
Event Wait/Record と EventLease の所有だけを担当する。

## 2. Buffer の実体識別

- `CudaTensor` は `Arc<PooledBuffer>` を保持する。
- Clone / `from_storage` / `clone` は同じ `Arc<PooledBuffer>`。
- Reshape は同じ storage を共有する。
- Transpose は executor では新しい buffer にコピーする（view ではない）。
- Slice / Broadcast View は現状の CUDA Tensor API にはない。
- Pool は size class ごとの `Vec<DeviceBuffer<f32>>` free list。Buffer の
  generation / identity は持っていなかった。
- Arena Slot は `CudaTensor` を SlotSet 内に保持し、Execution 間で再利用する。
  SlotSet 再利用時に古い Access State をリセットする仕組みはなかった。

## 3. Kernel Launch 経路

- すべての GPU kernel launch は `CudaKernels` の host wrapper を経由する。
- executor / optimizer / trainer / gpu_check / gemm_bench は全て `CudaKernels`
  メソッドを呼ぶ。
- `CudaKernels` は EventPool を持っていなかった。
- AccessGuard を Kernel Launch に統合するには、`CudaKernels` が Access State と
  completion event を管理するのが最小の変更点。

対象 launch wrapper:

`unary`, `binary`, `add_relu_backward`, `activation_backward`, `transpose`,
`broadcast`, `reduce`, `softmax`, `layer_norm`, `mse`, `mse_backward`, `scale`,
`gemm_naive`, `gemm_tiled`, `sgd`, `adamw`

## 4. Event / Buffer / Stream

- `CudaEvent` は raw CUevent、Context、state machine、diagnostics を保持。
- `EventLease` は EventPool の pending/free/quarantine に接続する。
- Buffer は completion events を `Vec<Arc<CudaEvent>>` で保持し、Drop 時に
  Event Safety の deferred reclamation を通る。
- AccessGuard の completion event は EventLease として Access State が保持し、
  buffer の completion events にも追加する必要がある。
- 同一 Stream は CUDA stream order で保護できる。異なる Stream のみ
  `CudaEvent::wait_on(stream)` を使う。

## 5. 監査結論

1. Buffer に物理 ID + generation + context ID を付与する。
2. `PooledBuffer` に Access State を持たせ、CudaTensor Clone/Alias で共有する。
3. Pool acquire / Arena lease で generation を進め、Access State を reset する。
4. `CudaKernels` の全 launch wrapper を AccessGuard で包む。
5. AccessRequest は PooledBuffer の strong/weak reference を持たない
   （`CudaTensor::buffer_mut()` の一意性を壊さないため）。
6. Launch 後、kernel wrapper が completion event を buffer に attach する。
7. Launch / Record / Wait failure 時は対象 buffer を invalidate し、
   quarantine 経路に載せる。
