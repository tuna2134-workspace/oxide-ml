# Phase 7F-15 Async Execution Lifecycle Safety / Resource Ownership — Final Report

## 1. 最終判定

**PASS.**

Async Execution の Resource Bundle、Execution lifecycle、Drop Safety、External
Output Ownership、Arena Lease Lifetime、Context Lifetime を実装し、CPU Mock Test と
T4 Integration で確認した。Phase 7F-16 以降には進んでいない。

| Area | Judgement |
|---|---|
| ExecutionResources | PASS |
| ExecutionState / Lifecycle | PASS |
| Non-blocking is_complete | PASS |
| wait / collect / double collect | PASS |
| Execution Drop | PASS |
| Executor Drop | PASS |
| Arena Lease / Registry Removal | PASS |
| External Output Ownership | PASS |
| Context Lifetime | PASS |
| Error Injection | PASS（CPU Mock） |
| CPU Tests | PASS 171 / 0 failed |
| Clippy | PASS |
| T4 Integration | PASS 28 / EXIT 0 |
| Sanitizer | PASS |

## 2. 既存コード監査

詳細: [phase7f15-audit.md](phase7f15-audit.md)

- `execute_async_planned` は `run_planned_on` の末尾で `stream.synchronize()` して
  おり、実質同期実行だった。
- `CudaExecution` は `CudaRun` と `Arc<CudaStream>` のみ。completion event、
  arena lease、明示 `Arc<CudaContext>` を持っていなかった。
- `is_complete` は stream query、`wait` は `&self`、`collect` は二重 Collect を
  拒否しなかった。
- `PooledBuffer` は completion events を保持し、Drop 時に Event Safety の
  deferred reclamation/quarantine を通る。この機構は正しく、再利用した。
- `SlotSetLease::finish` はエラー時に buffers を drop していたため、
  `finish_or_return` を追加して安全側に変更した。

## 3. Execution Ownership Model

```text
CudaExecution
  |
  +-- ExecutionResources
        +-- CudaRun (all CudaTensor / PooledBuffer / completion events)
        +-- Arc<CudaStream>
        +-- Arc<CudaContext>          <- explicit context lifetime
        +-- EventLease                <- final completion barrier
        +-- Option<ArenaLeaseState>
              +-- SlotSetLease
              +-- slot buffers
              +-- slot events
              +-- slot streams
        +-- Arc<ExecutionDiagnostics>
  +-- SharedExecutionLifecycle
```

- `CudaExecutor` は Execution から参照されない。
- Input/Output buffer は `CudaRun` の `CudaTensor`（`Arc<PooledBuffer>`）が保持する。
- External Output は `CudaTensor::clone` により独立 owner になる。
- 各 Buffer は自身の completion events を保持する。
- ExecutionResources の Drop は host synchronize を行わない。

## 4. Execution State Machine

実装: `crates/oxide-ml-execution-safety/src/lib.rs`

```rust
pub enum ExecutionState {
    Submitted,
    Running,
    Completed,
    Collected,
    Failed,
}
```

| Current | Event | Next |
|---|---|---|
| Submitted | poll not complete | Running |
| Running | poll not complete | Running |
| Submitted / Running | completion confirmed | Completed |
| Submitted / Running | query/sync/D2H/binding error | Failed |
| Completed | collect begins | Completed |
| Completed | collect succeeds | Collected |
| Collected | collect again | Error（AlreadyCollected） |
| Failed | collect / wait | Error |

- CUDA Event の Failed と Execution の Failed は別概念。
- Execution Failed でも GPU がリソースを使用中の可能性があるため、Buffer は解放しない。
- `is_complete` は Event Query のみ。Host Synchronization は行わない。
- `wait` は completion `EventLease::synchronize`、成功後 Completed。
- `collect` は未完了なら `None`、完了なら D2H → Collected。
- `begin_collect` は D2H 成功まで Collected にしない。

## 5. Drop Safety

### CudaExecution Drop

1. `ExecutionResources::drop` が arena lease を `PendingCompletion` へ移す。
2. `CudaRun` の Buffer が Drop され、Event Safety の `release_deferred` に入る。
3. 未完了 Buffer は pending、Failed/query error は quarantine。
4. completion `EventLease` が Drop され、Failed event は EventPool の quarantine へ。
5. `Arc<CudaContext>` は Buffer/Stream/Event が保持する限り生存。
6. Host Synchronization は行わない。
7. Arena finish 失敗は `ExecutionDiagnostics::cleanup_failures` に記録し、panic しない。

### Executor Drop

- `CudaExecutor` は Execution を参照しない。
- Execution が `Arc<CudaStream>` / `Arc<CudaContext>` / `CudaRun` / EventLease /
  Arena lease を保持するため、Executor Drop 後も wait/collect できる。
- T4 で Executor を Drop してから Execution を wait するテストが PASS。

## 6. Executor Lifetime

- Execution は `Arc<CudaContext>` を明示保持する。
- `CudaKernels` は submit 後の実行に不要。Executor Drop で module が drop されても
  既に enqueue 済みの kernel は stream 上で完了する。
- T4 実測:

```text
executor drop before completion: PASS
execution drop before completion: PASS
  pending_release_count=4, pending_release_bytes=160
```

Executor Drop 後も Execution の wait が成功したことを確認した。

## 7. Arena Lease Lifetime

- `CudaExecution` は `SlotSetLease` と slot buffers/events/streams を保持する。
- 完了時 (`wait` / `collect`) に `SlotSetLease::finish_or_return` で
  `PendingCompletion` へ移す。
- Execution Drop 時も同じく PendingCompletion へ移す。
- `SlotSetLease::finish_or_return` は失敗時に resources を返すため、
  premature free しない。
- Registry から Arena entry が消えても、Execution の lease が `Arc<ArenaInner>` を
  保持する。
- T4:

```text
arena registry removal / async lease: PASS
```

Registry drop 後に Execution.wait し、Arena が slot set を reclaim することを確認。

## 8. External Output Ownership

- `CudaExecution::output_tensor` / `collect_output` は `CudaTensor` を clone する。
- `CudaTensor` は `Arc<PooledBuffer>` の独立 owner。
- Execution Drop 後も External Output は Buffer を保持する。
- 複数 owner の最後の Drop まで Buffer は解放されない。
- T4:

```text
external output/multiple owners: PASS
```

Execution より Output が長く生存し、Clone の片方 Drop 後も他方が使用できた。

## 9. Context Lifetime

- `ExecutionResources.context: Arc<CudaContext>`。
- `CudaRun` / `PooledBuffer` / `CudaEvent` / `CudaStream` / `SlotSetLease` も
  Context を保持する。
- 最終参照が drop されるまで Context は生存する。
- Drop 時の Event destroy / Buffer free は既存 Phase 7F-13 の
  `bind_to_thread()` 方針に従う。
- Binding failure は Event destroy を skip し、diagnostics に記録する。
- CPU Mock で context binding failure → Execution Failed、資源保持を確認。

## 10. Error Injection

### Event Layer

既存 `oxide-ml-event-safety` の Generic Backend / Mock をそのまま使用。
Query / Synchronize / Record / Destroy failure、Event Quarantine、Pool Drop は
Phase 7F-13 の 22 tests で PASS。

### Execution Layer

`oxide-ml-execution-safety` の Mock で注入:

| Failure | Result |
|---|---|
| Query Failure | Execution Failed / 資源保持 / panic なし |
| Synchronize Failure | Execution Failed / 資源保持 |
| D2H Failure | Execution Failed / Output ownership 保持 / diagnostics |
| Context Binding Failure | Execution Failed / diagnostics |
| Arena Lease Finish Failure | cleanup_failures 記録 / panic なし |
| Executor Drop | Execution wait 成功 |
| Execution Drop While Running | Buffer pending / Arena pending |
| External Output Drop | Deferred reclamation |
| Double Release | 最後の owner のみ解放 |
| Poisoned Mutex | SharedExecutionLifecycle が回復 |

## 11. CPU Unit Tests

```text
cargo test --workspace
passed=171 failed=0
```

| Package | Tests |
|---|---:|
| oxide-ml-execution-safety | 14 |
| oxide-ml-event-safety | 22 |
| その他 CPU workspace | 135 |
| **合計** | **171** |

Execution-safety 内訳:

- ExecutionLifecycle state transitions（Submitted/Running/Completed/Collected/Failed）
- Submitted → Failed、Running → Failed
- Double Collect Rejection
- Poisoned SharedExecutionLifecycle
- Executor Drop While Running
- Execution Drop While Running（Buffer pending / Arena pending）
- External Output Outlives Execution
- Multiple Output Owners
- Arena Lease Outlives Registry Entry
- Query Failure
- Synchronize Failure
- D2H Failure
- Context Binding Failure
- Arena Finish Failure
- Double Release Prevention

既存 Test の削除は行っていない。

## 12. T4 GPU Integration

`gpu_check`（Tesla T4 / sm_75 / CUDA 13.0）:

```text
28 PASS / EXIT 0
```

Phase 7F-15 追加確認:

| Check | Result |
|---|---|
| async planned completion / wait / collect | PASS |
| double collect rejection | PASS |
| ExecutionState Completed / Collected | PASS |
| Executor Drop before completion | PASS |
| Execution Drop before completion | PASS（pending 4 buffers/160 B を観測） |
| External Output outlives execution | PASS |
| Multiple External Output owners | PASS |
| Arena Registry Removal during execution | PASS |
| Arena reclaim after completion | PASS |
| Context lifetime | PASS |
| EventPool reuse | PASS（allocation 2 / reuse 842） |
| Normal / Planned / Arena numerical equality | PASS |
| Persistent Arena 100 runs / 2 slot allocations | PASS |
| SlotSet capacity / busy | PASS |
| MLP backward | PASS |
| Linear regression | PASS |
| XOR | PASS |

`execution drop before completion` は Device Synchronize の成功だけではなく、
Drop 直後に `pending_release_count=4` を観測し、その後の poll で 0 になることを
確認している。

## 13. Benchmark / Sanitizer

T4 release `memplan_bench`:

| Mode | Median | P95 |
|---|---:|---:|
| Normal + Pool | 140.4 us | 189.9 us |
| Planned + Pool | 115.8 us | 131.6 us |
| Persistent Arena | 127.2 us | 154.7 us |
| Event acquire+record+release | 4.77 us | 5.48 us |

Phase 7F-14 の範囲（Normal ~139–151 us / Planned ~107–124 us / Arena ~122–130 us）と
比較して、Execution Lifecycle による有意な退行はない。Event overhead は VM ノイズの
範囲内。

Compute Sanitizer:

| Tool | Target | Result |
|---|---|---|
| memcheck | `gpu_check` | PASS 0 errors |
| racecheck | `memplan_bench` | PASS 0 hazards |
| synccheck | `memplan_bench` | PASS 0 errors |

## 14. 変更ファイル一覧

新規:

- `crates/oxide-ml-execution-safety/Cargo.toml`
- `crates/oxide-ml-execution-safety/src/lib.rs`
- `crates/oxide-ml-execution-safety/tests/ownership.rs`
- `docs/phase7f15-audit.md`
- `docs/phase7f15-final-report.md`

変更:

- `crates/oxide-ml-cuda/Cargo.toml`（execution-safety dependency）
- `crates/oxide-ml-cuda/src/executor.rs`（ExecutionResources / async / lifecycle /
  drop / collect / async arena）
- `crates/oxide-ml-cuda/src/arena.rs`（`finish_or_return` / `is_finished`）
- `crates/oxide-ml-cuda/src/error.rs`（`CudaError::Execution`）
- `crates/oxide-ml-cuda/src/lib.rs`（Execution 型 re-export）
- `crates/oxide-ml-cuda/src/bin/gpu_check.rs`（Phase 7F-15 T4 checks）
- `Cargo.lock`
- `crates/oxide-ml-cuda/Cargo.lock`
- `README.md`
- `docs/gpu-p7f15-*.log`

## 15. 既知の制約

- D2H Failure / Context Binding Failure / Arena Finish Failure は CPU Mock で検証。
  実 GPU への意図的な故障注入は行っていない。
- `CudaExecution::wait` は既存 API 互換のため
  `Result<HashMap<ValueId, DenseTensor<f32>>>` を返す。state-only の API は
  `is_complete` / `collect` で提供する。
- `CudaExecution` の Cross-thread 共有 API は追加していない。`&mut self` による
  collect/wait の直列化を前提とする。
- Arena `finish_or_return` 失敗時は resources を保持するが、Arena SlotSet が
  Leased のまま残る可能性がある。premature free を避けるための安全側のリークで、
  diagnostics に記録する。
- Execution の completion event は 1 本の barrier event。Node ごとの Event は
  Buffer 側に保持され、Buffer Reclamation は従来通り各 Event を評価する。
- `ExecutionResources` は `CudaRun` の全値（入力・中間・出力）を保持するため、
  Execution 生存中の peak memory は従来より長く保持される。これは安全性優先の
  設計であり、性能改善は目的外。
- T4 は 1 GPU のため、別 GPU / 別 Context の物理故障テストは NOT RUN。

## 16. 未達項目

なし。以下の完了条件をすべて満たした。

| 完了条件 | 状態 |
|---|---|
| Execution Resource Bundle | PASS |
| Execution State | PASS |
| Non-blocking Query | PASS |
| Wait | PASS |
| Collect / Double Collect 防止 | PASS |
| Execution Drop | PASS |
| Executor Drop | PASS |
| Arena Registry Removal | PASS |
| External Output Ownership | PASS |
| Multiple Output Owners | PASS |
| Context Lifetime | PASS |
| Error Injection | PASS（CPU Mock） |
| Event Failure Safety / Quarantine | PASS（Phase 7F-13 Regression） |
| CPU Tests | PASS 171 / 0 failed |
| Clippy | PASS |
| T4 Integration | PASS 28 / EXIT 0 |
| Existing Regression | PASS |

**Phase 7F-15 完了。**
