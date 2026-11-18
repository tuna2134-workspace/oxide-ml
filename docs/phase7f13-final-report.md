# Phase 7F-13 Event Failure Safety — Final Report

## 1. 実装概要

Phase 7F の EventPool / EventState / Deferred Reclamation を維持したまま、
CUDA Runtime の Event 操作異常時にも DeviceBuffer を誤って解放・再利用しない
仕組みを追加した。

- 新規 `crates/oxide-ml-event-safety/` は CUDA 非依存。
  - Event 状態機械
  - free / pending / quarantine を持つ Generic EventPool
  - structured EventError と destroy diagnostics
  - 「全 Event が明示的に Completed のときだけ reclaim」する判定関数
- `oxide-ml-cuda` 側は Generic Backend の本番実装のみを持つ。
  - `CUevent` の create / record / query / synchronize / destroy / stream wait を
    最小限ラップ
  - Context は `Arc<CudaContext>` で Event / Pool / Buffer が保持
  - CUDA の生 API 全体をラップし直してはいない
- CPU の Fault Injection は Mock Backend を compile-time で差し替える。
  Dynamic Dispatch は使用していない。

T4 実機では EventPool の create/reuse、record/query/synchronize、pending 保持、
deferred reclamation、pool Drop、context lifetime、buffer の context 不一致拒否、
Normal / Planned / Arena の数値一致を確認した。

## 2. Event State Machine

`EventState` は既存の 4 状態を維持した。

| Current | Operation | Success | Failure |
|---|---|---|---|
| Unrecorded | record | Recorded | Failed |
| Unrecorded | query | Unrecorded（未完了） | - |
| Unrecorded | synchronize | 拒否（Error） | - |
| Recorded | record（再記録） | Recorded | Failed |
| Recorded | query | Recorded / Completed | Failed |
| Recorded | synchronize | Completed | Failed |
| Completed | query | Completed | - |
| Completed | synchronize | Completed | - |
| Completed | reuse | Unrecorded | - |
| Failed | record / query / synchronize / reuse | すべて拒否 | - |

保証:

- `Failed` は terminal。再 Record・再 reuse・Completed 扱いをすべて拒否する。
- `Unrecorded` は `is_complete() == false`。query が成功しても Completed にしない。
- State 更新は CUDA API 成功確認後。Record 失敗時は generation を進めない。
- `CudaEvent::is_complete` は query の `Ok(true)` だけを完了とし、
  `Err` は Failed に遷移させて呼び出し元へ返す。
- `synchronize` は Unrecorded を明示エラーにする。Recorded の成功のみ Completed。
- `wait_on` は Failed / 異 Context を拒否し、stream wait 失敗時は Event を Failed に
  して Buffer 側の完了推定を成立させない。

## 3. Quarantine 設計

`EventPool` は `free` / `pending` / `quarantine` を単一 Mutex 配下で管理する。

- Failed Event は release / poll のどちらでも `quarantine` に移動する。
- quarantine から free へ戻す経路は存在しない。acquire は free のみを見る。
- Quarantine された Event の `EventError`（op / CUDA code / message）を保持する。
- `quarantine_diagnostics()` で event_id / generation / state / error を取得できる。
- `quarantine_count` は現在数、`quarantine_total` は累積数。
- Mutex poisoning は `lock_recover` で回復し、`poison_recoveries` に記録する。
  Drop 中も panic しない。
- `Destroy` 失敗は shared `EventDiagnostics` に記録し、再試行しない。
  Context が current にできない場合は destroy を skip して記録する。

追加した `EventPoolStats`:

```rust
pub struct EventPoolStats {
    pub allocation_count: usize,
    pub reuse_count: usize,
    pub pending_count: usize,
    pub quarantine_count: usize,
    pub quarantine_total: usize,
    pub destroy_failure_count: usize,
    pub destroy_attempt_count: usize,
    pub destroy_skipped_count: usize,
    pub poison_recovery_count: usize,
}
```

## 4. Fault Injection 設計

単一の `EventBackend` に押し込めず、型で責務を分離した。これは Stream / Context の
型を Event ごとに固定し、Dynamic Dispatch を避けるため。

```rust
pub trait CompletionEvent: Send + Sync + 'static {
    fn state(&self) -> EventState;
    fn generation(&self) -> u64;
    fn last_error(&self) -> Option<EventError>;
    fn context_id(&self) -> ContextId;
    fn is_complete(&self) -> Result<bool, EventError>;
    fn synchronize(&self) -> Result<(), EventError>;
}

pub trait ReusableEvent: CompletionEvent {
    fn reset_for_reuse(&self) -> Result<(), EventError>;
}

pub trait RecordEvent<Stream: ?Sized>: CompletionEvent {
    fn record(&self, stream: &Stream) -> Result<(), EventError>;
}

pub trait EventFactory<E>: Send + Sync + 'static {
    fn context_id(&self) -> ContextId;
    fn create(&self, diagnostics: Arc<EventDiagnostics>) -> Result<E, EventError>;
}
```

- 本番: `CudaEvent` + `CudaEventFactory`（`CUevent` を直接所有）
- テスト: `MockEvent` + `MockFactory`
  - create / record / query / synchronize / destroy を個別に注入
  - query の ready/not-ready を制御
  - context mismatch を注入
  - destroy failure / context lost を注入

Mock の Event 破棄は `Drop` で 1 回だけ実行し、結果を `EventDiagnostics` に記録する。
本番も `RawCudaEvent::Drop` で 1 回だけ `cuEventDestroy_v2` を呼び、失敗を記録して
再試行しない。

## 5. Buffer Reclamation Safety

`evaluate_completion` を唯一の判定経路として `CudaBufferPool` に統合した。

```text
Buffer is reclaimable
    iff all completion events are confirmed Completed
```

- Unrecorded Event: Pending（解放禁止）
- Recorded Event: 完了確認まで Pending
- Completed Event: 他 Event も完了していれば Reclaim
- Failed Event: Quarantine
- Query Error: Quarantine
- Synchronize Error: Quarantine

`CudaBufferPool` の変更:

- `quarantine: Vec<QuarantinedBuffer>` を追加。通常の free list には絶対に戻さない。
- Quarantine した Buffer は `DeviceBuffer` を保持し続け、Pool Drop まで再利用しない。
- `BufferQuarantineRecord` に event_states / event_errors / error を保存。
- `BufferPoolStats` を追加:

```rust
pub struct BufferPoolStats {
    pub quarantined_count: usize,
    pub quarantined_bytes: usize,
    pub quarantine_total: usize,
}
```

- `mark_used` は Event の ContextId と Buffer Pool の ContextId を比較し、
  不一致または Failed Event を拒否する。戻り値を `Result<()>` に変更した。
- pending → quarantine 遷移で `current_bytes` を二重減算しないよう修正した。
- Mutex poisoning は回復し、`lock_poison_recoveries` に記録する。
- Pool Drop では pending / quarantine を free に戻さず、保持したまま破棄する。
  cuda-core の `DeviceBuffer` は access lease 完了まで deallocation を遅延する。

## 6. EventPool Drop / Context Lifetime

- `CudaEvent` は `Arc<CudaContext>` を保持する。`EventPool` / `CudaBufferPool` /
  `PooledBuffer` / `CudaTensor` も Context を保持するため、Event / Buffer が
  Context より長く生存することは構造的にない。
- Event は作成元 ContextId を保持する。`EventFactory` が別 Context の Event を
  返した場合、Pool は登録せずに拒否する。
- `RawCudaEvent::Drop` は最初に `context.bind_to_thread()` を確認する。
  失敗した場合は CUDA API を呼ばず、destroy skipped として記録する。
- CUDA Driver API の `cuEventDestroy` は「未完了 Event でも破棄可能。完了前に
  破棄した場合、API は完了を待たず、関連リソースは完了時に非同期解放される」と
  明記されている。この契約に基づき、Pending / Quarantined Event の Drop は
  synchronous wait をせずに destroy する。これは未完了を「完了」と推定する
  処理ではない。
- Destroy 失敗は統計と diagnostics に記録し、再 Destroy はしない。
- `EventPool::shutdown()` は free / pending / quarantine を drain し、
  `CleanupReport` を返す。Buffer などが Event を保持している場合は最後の Arc の
  Drop まで破棄を遅延する。
- `EventPoolInner::Drop` は `get_mut` で poisoning を回復し、Vec の drop に任せる。
  Drop 中に panic する経路はない。

## 7. 変更ファイル一覧

新規:

- `crates/oxide-ml-event-safety/Cargo.toml`
- `crates/oxide-ml-event-safety/src/lib.rs`
- `crates/oxide-ml-event-safety/src/error.rs`
- `crates/oxide-ml-event-safety/src/state.rs`
- `crates/oxide-ml-event-safety/src/diagnostics.rs`
- `crates/oxide-ml-event-safety/src/pool.rs`
- `crates/oxide-ml-event-safety/src/buffer.rs`
- `crates/oxide-ml-event-safety/tests/fault_injection.rs`

変更:

- `crates/oxide-ml-cuda/Cargo.toml`
- `crates/oxide-ml-cuda/src/event.rs`（全面的に置換）
- `crates/oxide-ml-cuda/src/buffer.rs`
- `crates/oxide-ml-cuda/src/executor.rs`
- `crates/oxide-ml-cuda/src/arena.rs`
- `crates/oxide-ml-cuda/src/access.rs`
- `crates/oxide-ml-cuda/src/error.rs`
- `crates/oxide-ml-cuda/src/lib.rs`
- `crates/oxide-ml-cuda/src/bin/gpu_check.rs`
- `crates/oxide-ml-cuda/src/bin/memplan_bench.rs`
- `crates/oxide-ml-cuda/tests/gpu.rs`（既存の GPU 統合テストを repo に取り込み、
  clippy 指摘の未使用変数を修正）
- `crates/oxide-ml-cuda/src/kernels.rs`（`buffer()` の Result 化に伴う `?`、
  clippy 指摘の `div_ceil` / 定数 / 未使用 ABI 引数名を修正）
- `README.md`

検証ログ:

- `docs/gpu-phase7f13-run.log`
- `docs/gpu-p7f13-bench.log`
- `docs/gpu-p7f13-bench-multi.log`
- `docs/gpu-p7f13-san-memcheck.log`
- `docs/gpu-p7f13-san-racecheck-bench.log`
- `docs/gpu-p7f13-san-synccheck-bench.log`
- `docs/gpu-p7f13-clippy.log`

## 8. CPU Unit Test 結果

`oxide-ml-event-safety` に 22 テストを追加。すべて PASS。

- Unrecorded は完了しない
- Record 成功で Recorded
- Query 未完了で Recorded 維持
- Query 完了で Completed
- Synchronize 成功で Completed
- Record / Query / Synchronize 失敗で Failed
- Failed の再 Record 拒否
- Failed Event の Quarantine
- Quarantine から Free List へ戻らない
- Pending Event の非再利用
- Create 失敗で統計不整合なし
- Destroy 失敗の統計反映
- Context lost 時の Destroy skip
- Buffer の全 Event 完了前 Reclaim 禁止
- Failed / Query Error Buffer の Quarantine 判定
- Context 不一致の拒否
- Pool Drop の非 panic（failed / pending / quarantined 混在）
- Mutex poisoning からの回復

実行結果:

```text
cargo test -p oxide-ml-event-safety --test fault_injection
test result: ok. 22 passed; 0 failed

cargo test --workspace
passed=109 failed=0
```

既存テストの削除は行っていない。CUDA 無効の CPU workspace から実行可能。

Clippy:

```text
# CPU workspace (CUDA crate is excluded by design)
cargo clippy --workspace --all-targets -- -D warnings
Finished (no warnings)

# CUDA crate on T4
cargo clippy --all-targets -- -D warnings
Finished (no warnings), EXIT 0
```

CUDA 側の `too_many_arguments` / `type_complexity` は、CUDA ABI をそのまま反映する
kernel launch wrapper と複数リストを返す内部 helper に限定して crate レベルで
allow した。それ以外の指摘は修正した。

## 9. CUDA T4 Integration Test 結果

`gpu_check`（T4 / sm_75 / CUDA 13.0）: **EXIT 0**、22 個の PASS。

Phase 7F-13 追加分:

| Check | Result |
|---|---|
| EventPool create | PASS |
| EventPool reuse | PASS |
| Event Record / Query / Synchronize | PASS |
| Pending Event retention | PASS（`pending_count=1` を観測） |
| EventPool destroy statistics | PASS（attempt=1, failure=0） |
| EventPool Drop safety | PASS |
| Event Context lifetime（Pool drop 後も Event が query 可能） | PASS |
| Buffer の Context 不一致拒否 | PASS |
| Event/Buffer quarantine stats normal path | PASS（quarantine 0） |
| Normal / Planned 数値一致 | PASS |
| Arena 数値一致 | PASS |
| Async / Deferred Reclamation | PASS |
| MLP backward CPU vs CUDA | PASS |
| Linear regression / XOR training | PASS |

通常系 EventPool stats:

```text
EventPoolStats {
  allocation_count: 2,
  reuse_count: 842,
  pending_count: 0,
  quarantine_count: 0,
  quarantine_total: 0,
  destroy_failure_count: 0,
  destroy_attempt_count: 0,
  destroy_skipped_count: 0,
  poison_recovery_count: 0,
}
```

Buffer quarantine stats は正常系で 0。Deferred Reclamation は
`pending_release_count=1` / `1,048,576 B` を観測後、poll で 0 になることを確認。

## 10. Benchmark 結果

T4、`memplan_bench`、100 iterations、warm-up 5。3 回反復した中央値の範囲:

| Mode | Median | P95 | 参考: Phase 7F 以前 |
|---|---:|---:|---:|
| Normal + Pool | 139.0 – 150.8 us | 156.0 – 188.9 us | 138.9 us |
| Planned + Pool | 106.9 – 123.8 us | 129.7 – 147.2 us | 125.0 us |
| Persistent Arena | 121.6 – 130.1 us | 143.4 – 159.3 us | 133.1 us |

EventPool の host overhead（acquire + record + release、1000 iterations）:

```text
median = 3.42 – 3.61 us
p95    = 4.74 – 6.19 us
```

- 正常系の Event allocation は 2–7 回、reuse は 2,923–2,927 回。
- Normal / Planned / Arena の追加 Synchronize は増えていない。
- 初回計測は VM ノイズで Normal 210.7 us だったが、反復計測では
  Phase 7F 以前の範囲に収まった。性能改善ではなく、退行がないことの確認。

## 11. Sanitizer 結果

| Tool | Target | Result |
|---|---|---|
| Compute Sanitizer memcheck | `gpu_check` 全体 | **PASS: 0 errors** |
| Compute Sanitizer racecheck | `memplan_bench`（EventPool / Normal / Planned / Arena） | **PASS: 0 hazards, 0 errors, 0 warnings** |
| Compute Sanitizer synccheck | `memplan_bench`（EventPool / Normal / Planned / Arena） | **PASS: 0 errors** |

注: `gpu_check` 全体の racecheck / synccheck は XOR 2500 step の途中で
実行時間が過大になったため完走していない。memcheck は `gpu_check` 全体で完走。
racecheck / synccheck は EventPool と Normal / Planned / Arena を含む
`memplan_bench` に対して完走させた。未完走の `gpu_check` racecheck は
PASS として扱っていない。

## 12. 未実装・未検証項目

- 実 CUDA Runtime に対する意図的な Create / Record / Query / Synchronize /
  Destroy 故障注入は行っていない（危険なため）。Fault Injection はすべて
  Mock Backend の CPU テスト。
- Failed Event に関連する Buffer の Quarantine は、共有判定関数の CPU テストと
  T4 正常系（quarantine 0）で確認した。実 GPU 故障による Buffer Quarantine は
  未実施。
- `gpu_check` 全体の racecheck / synccheck は未完走（上記）。
- Context 不一致テストは、同一 primary context に対する別 `Arc<CudaContext>`
  で identity 不一致を確認した。別物理 GPU 間のテストではない。
- Quarantined Buffer は Pool Drop まで保持し、その後 cuda-core の deallocation
  遅延に委ねる。明示的な Buffer destroy API は追加していない。
- API 変更:
  - `PooledBuffer::buffer()` / `CudaTensor::buffer()` / `pointer()` は `Result` 化。
  - `mark_used()` は `Result<()>` 化。
  - `EventPoolStats` は `usize` と追加フィールド。
  - `EventPool::shutdown()` / `quarantine_diagnostics()` を追加。

## 13. Phase 7F-13 Acceptance Criteria

| Criterion | State |
|---|---|
| Event State Machine 全遷移テスト | **PASS** |
| Create Failure Fault Injection | **PASS**（Mock） |
| Record Failure Fault Injection | **PASS**（Mock） |
| Query Failure Fault Injection | **PASS**（Mock） |
| Synchronize Failure Fault Injection | **PASS**（Mock） |
| Destroy Failure Fault Injection / 統計 | **PASS**（Mock） |
| Failed Event Quarantine | **PASS** |
| Failed Event の Free List 混入防止 | **PASS** |
| Buffer Quarantine | **PASS**（policy unit + T4 normal path） |
| Deferred Reclamation 安全性 | **PASS** |
| EventPool Drop Safety | **PASS** |
| Context Lifetime Test | **PASS**（T4） |
| CPU Unit Tests | **PASS** 109 / 0 failed |
| CUDA T4 Integration Tests | **PASS** EXIT 0 |
| Normal / Planned / Arena 数値一致 | **PASS** |
| Clippy | **PASS**（CPU workspace all-targets / CUDA crate all-targets） |
| Compute Sanitizer | memcheck PASS / racecheck PASS / synccheck PASS（対象は §11） |

**Phase 7F-13 は完了。** ただし §11 の通り、racecheck / synccheck の対象は
`memplan_bench` であり、`gpu_check` 全体の racecheck は未完走である。
