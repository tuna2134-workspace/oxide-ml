# Phase 7F-15-A 既存コード監査 — Async Execution Ownership

監査日: 2026-11-18  
対象: CudaExecutor / CudaExecution / CudaRun / CudaPlannedArena / SlotSetLease /
PooledBuffer / CudaBufferPool / CudaEvent / EventLease / ContextIdentity / CudaArenaRegistry

## 1. CudaExecutor

所有:

- `CudaDevice`（`Arc<CudaContext>` と info）
- `CudaBufferPool`（`Arc<PoolInner>`。PoolInner が `Arc<CudaContext>` と
  `Arc<CudaStream>` を保持）
- `CudaKernels`（cuda-host module）
- `EventPool`（`Arc<EventPoolInner>`。`Arc<CudaContext>` を保持）

Drop:

- カスタム Drop なし。フィールドが宣言順に drop される。
- executor 自身は GPU 処理の完了を待たない。

## 2. CudaExecution（Phase 7F-14 まで）

所有:

- `CudaRun`（graph values / outputs の `CudaTensor`）
- `Arc<CudaStream>`

所有していない:

- `CudaExecutor`
- `Arc<CudaContext>`（stream と DeviceBuffer 経由で間接保持）
- Arena lease
- 明示的な completion event

問題:

1. `execute_async_planned` は `run_planned_on` を呼び、その末尾で
   `stream.synchronize()` していた。**実質同期実行**であり async API ではない。
2. `CudaExecution::is_complete` は stream query を使用。execution 固有の
   completion event を持たず、Event Failure と execution Failure を区別できない。
3. `wait(&self)` が `&self` で、lifecycle state を持たない。
4. `collect` は二重 Collect を拒否しない。
5. Arena の async execution 経路がなく、`SlotSetLease` を Execution が保持できない。
6. 明示的な `Arc<CudaContext>` を Execution が持たないため、Context Lifetime が
   Buffer/Event の間接保持に依存している。

## 3. CudaRun / PooledBuffer / CudaBufferPool

- `CudaRun` は `CudaTensor` の HashMap。`CudaTensor` は `Arc<PooledBuffer>`。
- `PooledBuffer` は `DeviceBuffer<f32>`、`Arc<PoolInner>`、completion events を保持。
- `PooledBuffer::drop` は `release_deferred`:
  - 全 event が Completed → free list
  - 未完了 / Unrecorded → pending
  - Failed / query error → quarantine
- `PoolInner` は `Arc<CudaContext>` を保持するため、buffer が生存する限り
  Context は生存する。
- Phase 7F-13 の Event Safety / Deferred Reclamation をそのまま再利用できる。

## 4. CudaEvent / EventLease

- `CudaEvent` は raw `CUevent`、`Arc<CudaContext>`、state machine、diagnostics を保持。
- `EventLease` は `EventPool` の pending/free/quarantine 管理に接続する。
- 新しい execution completion event は `EventLease` として Execution が保持するのが
  適切。Drop 時に Failed event を EventPool の quarantine に戻せる。
- Buffer に attach 済みの node event は `Arc<CudaEvent>` として buffer が保持する。

## 5. CudaPlannedArena / SlotSetLease

- `CudaPlannedArena` は `Arc<ArenaInner>`。ArenaInner は plan / pool / slot sets /
  stats を保持。
- `SlotSetLease` は `Arc<ArenaInner>`、slot tensors、events、streams を保持。
- `take()` で slot tensors / events / streams を取り出せる。
- `finish()` は slot set を `PendingCompletion` に戻す。
- 旧 `finish()` はエラー時に引数の buffers を drop していた。Phase 7F-15 で
  `finish_or_return()` を追加し、失敗時に resources を返せるようにした。
- Arena は `SlotSetLease` が保持する限り Registry drop 後も生存する。

## 6. External Output

- `CudaExecution::output_tensor(index)` は `CudaTensor::clone` を返す。
- `CudaTensor` は `Arc<PooledBuffer>` の clone。外部 output は Buffer の
  独立した owner になる。
- Buffer は completion events を保持するため、Execution Drop 後も
  deferred reclamation/quarantine の対象になる。

## 7. Context Lifetime

- `CudaContext` は `Arc` で参照される。
- Stream / Buffer / Event / Pool / Arena が `Arc<CudaContext>` を保持。
- Execution が `Arc<CudaContext>` を明示保持すれば、Executor Drop 後も Context が
  生存することを保証できる。
- Event / Buffer の Drop は `bind_to_thread()` の失敗を diagnostics に記録し、
  destroy を skip する既存 Phase 7F-13 契約を維持する。

## 8. CudaArenaRegistry

- Registry は `ArenaKey` + `ArenaDescriptor` で `Arc<CudaPlannedArena>` を保持。
- Entry を drop しても、Execution が `SlotSetLease` を保持していれば
  `Arc<ArenaInner>` により Arena は生存する。
- Registry の context token が drop されても、Execution/Arena/Pool が
  `Arc<CudaContext>` を保持する。

## 9. 監査結論

既存の Event Safety / Buffer Deferred Reclamation / Arena の `Arc` 所有権は
正しく、作り直す必要はない。Phase 7F-15 で必要な変更は次に限定する。

1. `execute_async_planned` / `execute_async_arena` を **host synchronize なし**にする。
2. Execution 固有の completion `EventLease` を記録・保持する。
3. `ExecutionResources` に run / stream / context / completion event / arena lease を
   一括保持する。
4. `ExecutionState` lifecycle を明示する。
5. `wait` / `collect` / `is_complete` / Drop を state と completion event に接続する。
6. Arena lease を completion まで保持し、Drop 時は PendingCompletion へ移す。
7. Executor / Registry Drop 後も Execution が動くことを T4 で確認する。
