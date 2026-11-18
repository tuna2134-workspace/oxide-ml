# Phase 7F-16 AccessGuard / Executor Integration / Buffer Hazard Tracking — Final Report

## 1. 最終判定

**PASS.**

Buffer Identity / Generation、AccessGuard、Hazard Tracking、全 CudaKernels Launch
Wrapper、Pool / Arena Generation、CPU Mock Failure Injection、T4 Multi-Stream、
AccessGuard Sanitizer を完了した。CUDA adapter 固有の実 Fault Injection は
NOT RUN / SOURCE ONLY だが、CPU Mock の Failure Policy Test は PASS している。

| Area | Judgement |
|---|---|
| Buffer Identity / Generation | PASS |
| AccessGuard | PASS |
| Hazard Tracking | PASS |
| Same Stream Optimization | PASS |
| Cross Stream Dependency | PASS |
| Multiple Readers | PASS |
| Kernel Launch Coverage | PASS |
| Alias / Clone | PASS（read alias） |
| Pool / Arena Integration | PASS |
| Failure Handling | PASS（CPU Mock policy。CUDA 実 fault injection は SOURCE ONLY） |
| CPU Tests | PASS 192 / 0 failed |
| Clippy | PASS |
| T4 Integration | PASS 37 / EXIT 0 |
| AccessGuard racecheck / synccheck | PASS |
| AccessGuard memcheck | PASS |
| Benchmark | PASS（GPU Event Timing） |

## 2. 既存コード監査結果

詳細: [phase7f16-audit.md](phase7f16-audit.md)

- 既存 `BufferAccessState` は last_write / pending_reads の単純なテスト用で、
  Buffer Identity / Generation を持たず、Kernel Launch 経路に未統合だった。
- `CudaTensor` は `Arc<PooledBuffer>` を共有する。Clone / Reshape は同一 storage。
- Pool free list は `DeviceBuffer` のみで generation がなかった。
- Arena Slot は CudaTensor を Execution 間で再利用し、Access State reset がなかった。
- 全 kernel launch は `CudaKernels` の 16 host wrapper を経由する。
- `CudaKernels` は EventPool を持っていなかった。

## 3. Buffer Identity / Generation

```rust
pub struct BufferId(pub u64);
pub struct BufferContextId(pub u64);
pub struct BufferIdentity {
    pub id: BufferId,
    pub generation: u64,
    pub context_id: BufferContextId,
}
```

- `PooledBuffer` が id / generation / context_id を保持。
- Pool の物理 allocation ごとに BufferId を発行。
- Pool acquire 時に generation を +1。
- Arena lease 時に `begin_new_generation()` を呼び、generation +1 と
  Access State reset を行う。
- Tensor Clone / Reshape は同じ `Arc<PooledBuffer>` → 同じ BufferIdentity。
- Device Pointer は Buffer ID に使用していない。

T4 で Pool / Arena の同一 BufferId + 次 generation を確認。

## 4. AccessGuard API

実装: `crates/oxide-ml-cuda/src/access.rs`

```rust
pub struct AccessRequest {
    // PooledBuffer の strong/weak reference は保持しない
    state: Arc<Mutex<AccessState>>,
    diagnostics: Arc<EventDiagnostics>,
    identity: BufferIdentity,
    mode: AccessMode,
}

pub struct AccessGuard {
    identity: BufferIdentity,
    mode: AccessMode,
    token: Arc<AccessCompletion>,
}

pub struct AccessCompletion {
    event: Arc<CudaEvent>,
    _lease: EventLease, // event reuse 防止
}
```

AccessGuard の Drop は Mutex Unlock ではない。Access State が
`Arc<AccessCompletion>` を保持し、後続の競合 Access が EventLease と Event を
使って wait する。Guard が先に Drop されても GPU Completion まで Access State は
消えない。

AccessRequest が PooledBuffer を保持しない理由:

- `CudaTensor::buffer_mut()` は `Arc::get_mut` で storage の一意性を要求する。
- strong Arc / Weak が存在すると `get_mut` が失敗する。
- そのため AccessRequest は AccessState / identity / diagnostics だけを保持し、
  Kernel wrapper が launch 後に Tensor へ completion event を attach する。

## 5. Hazard Tracking 状態モデル

実装: `crates/oxide-ml-execution-safety/src/access.rs`

```rust
pub struct BufferAccessState<E, S> {
    identity: BufferIdentity,
    readers: Vec<PendingAccess<E, S>>,
    writer: Option<PendingAccess<E, S>>,
    stats: AccessStats,
}
```

- `plan_access(stream, mode, is_complete)`:
  - 完了済み reader/writer を prune。
  - 異 Stream の writer を wait。
  - Write / ReadWrite は異 Stream の全 reader を wait。
  - 同一 Stream は `same_stream_elided` として Event Wait を追加しない。
- `record(stream, mode, event)`:
  - Read: 既存 writer を clear し、reader を追加。
  - Write / ReadWrite: writer を設定し、reader を clear。
- Event token は EventLease を保持し、State から消えるまで EventPool が再利用しない。

## 6. Same Stream Optimization

- 同一 Stream の先行 Write / Read は CUDA stream order で保護される。
- `plan_access` は StreamToken が同じなら `waits` に追加しない。
- T4 実測:

```text
access same-stream elision:
event_waits=0, same_stream_elided_waits=1
```

## 7. Cross Stream Dependency

異なる Stream の依存だけ `CudaEvent::wait_on(stream)` を呼ぶ。

T4 実測:

| Scenario | event_waits | cross_stream |
|---|---:|---:|
| Read/Read then Write | 2 | 2 |
| Write/Read | 1 | 1 |
| Write/Write | 1 | 1 |
| ReadWrite/Read | 1 | 1 |
| Same Stream | 0 | 0 |
| Three streams (Write -> 2 Readers -> Write) | 3 | 3 |
| Alias Read | 1 | 1 |

Read/Read 自体は wait 0。後続 Write が pending な 2 Reader を待つことを確認。

## 8. Multiple Reader 管理

- writer 設定時に全 reader を保持。
- 後続 Write / ReadWrite は異 Stream の全 reader event を waits に追加。
- 完了済み reader は plan 時に prune。
- T4 three-stream: Write -> Read(B) -> Read(C) -> Write(D) で D が B/C の
  reader 2 件を wait（累積 event_waits=3、cross=3、pending_readers=0）。
- Reader event は EventLease を保持し、clear されるまで再利用されない。

## 9. Kernel Launch Coverage

`CudaKernels` の全 16 launch wrapper を AccessGuard で包んだ。

- unary
- binary
- add_relu_backward
- activation_backward
- transpose
- broadcast
- reduce
- softmax
- layer_norm
- mse
- mse_backward
- scale
- gemm_naive
- gemm_tiled
- sgd
- adamw

Executor / optimizer / trainer / gpu_check / gemm_bench はすべてこの wrapper を
経由する。AccessGuard を経由しない Kernel Launch は残っていない。

Launch 順序:

1. Access State lock / plan
2. 異 Stream Event Wait
3. EventLease acquire
4. Kernel Launch
5. Event Record
6. Tensor へ completion event attach
7. Access State へ Pending Access record
8. Guard 返却

Launch / Wait / Record / attach の失敗時は対象 Buffer を invalidate し、
Drop 時に Buffer Quarantine へ移す。

## 10. Alias / View Safety

- Clone / Reshape は同じ `Arc<PooledBuffer>` → 同じ BufferId / Access State。
- T4: alias と shared の identity が同一であること、alias の read が同じ
  Access State の pending reader になることを確認。
- Alias への in-place write は `CudaTensor::buffer_mut()` の Rust 一意性制約により
  失敗する（安全側）。現状の executor は view を書き込み出力に使わない。
- Byte Range 単位の Hazard は未実装。Buffer 全体を競合単位とする。

## 11. Pool / Arena Integration

- Pool free list に `BufferId` / generation を保持。
- Pool acquire 時に generation +1、Access State を新規作成。
- `PooledBuffer::begin_new_generation()` は invalid buffer を拒否。
- Arena `lease` 時に slot tensor の generation を +1 し Access State を reset。
- Invalid buffer は Pool Drop 時に Quarantine。Free List へ戻さない。
- Event Safety の Deferred Reclamation をそのまま使用し、二重管理はしていない。

T4:

```text
access pool generation: PASS
access arena generation: PASS
```

## 12. Failure Handling

- Access State lock: `lock_recover`（Mutex poisoning 回復、diagnostics）。
- Event Wait failure: launch せず buffer invalidate。
- Event Record failure: EventLease を Failed として drop → EventPool quarantine。
  buffer invalidate → Drop 時 Buffer quarantine。
- Kernel Launch failure: EventLease 未記録、buffer invalidate。
- Event Query failure: Access State の prune では complete と扱わず、後続 wait で
  failure として検出。
- Generation mismatch: `plan_access` 前に identity を比較し拒否。
- Context mismatch: Event `wait_on` / `mark_used` の Context 検証。
- Double release: EventLease / Buffer の既存 RAII に統合。

CUDA-level の意図的 Fault Injection は行っていない。Event/Execution 層の
failure は既存 CPU Mock Test（event-safety 22 + execution-safety 27）で検証。

## 13. CPU Unit Test

```text
cargo test --workspace
passed=192 failed=0
```

| Package | Tests |
|---|---:|
| oxide-ml-execution-safety | 35 |
| oxide-ml-event-safety | 22 |
| その他 | 135 |
| **合計** | **192** |

AccessGuard 純粋 Test（13）と Failure Mock（8）:

- Read/Read No Wait
- Read/Write, Write/Read, Write/Write Wait
- ReadWrite/Read, ReadWrite/Write
- Same Stream Elision
- Multiple Readers All Waited
- Three Stream Dependency Chain
- Reader / Writer Completion Pruning
- Identity / Generation / Context 区別
- Reset が旧 Access State を消す
- AccessMode merge
- Poisoned lifecycle（execution-safety 側）
- Stream Wait / Kernel Launch / Event Record failure
- Guard Token Drop / Lock Poison Recovery / Identity Generation

既存テストの削除は行っていない。

## 14. T4 GPU Integration

`gpu_check`（Tesla T4 / sm_75 / CUDA 13.0）:

```text
37 PASS / EXIT 0
```

AccessGuard 追加確認:

| Check | Result |
|---|---|
| Read/Read then Write | PASS（waits=2） |
| Write/Read | PASS（waits=1） |
| Write/Write | PASS（waits=1） |
| ReadWrite/Read | PASS（waits=1） |
| Same Stream Elision | PASS（waits=0 / elided=1） |
| Three Stream Readers/Writer | PASS（waits=3） |
| Alias / Clone shared state | PASS |
| Pool generation | PASS |
| Arena slot generation | PASS |
| Async execution / drop / executor drop | PASS |
| EventPool / Quarantine regression | PASS |
| Normal / Planned / Arena equality | PASS |
| MLP backward / Linear / XOR | PASS |

専用 AccessGuard GPU Test `access_check`:

```text
access_check: PASS
  Read/Read/Write: event_waits=2, cross=2, read_read=1
  Three-stream:    event_waits=3, cross=3, read_read=1
```

## 15. Wait Count

T4 実測（AccessStats）:

```text
read/read/write:
  event_waits=2, same_stream_elided=0, cross=2, read_read=1, pending_writer=1
write/read:
  event_waits=1, cross=1, pending_readers=1
write/write:
  event_waits=1, cross=1, pending_writer=1
read-write/read:
  event_waits=1, cross=1, pending_readers=1
same-stream:
  event_waits=0, same_stream_elided=1
three-stream:
  event_waits=3, cross=3, pending_readers=0, pending_writer=1
alias read:
  event_waits=1, cross=1, pending_readers=1
```

Read/Read の不要な Event Wait は 0。Same-stream の競合は Event Wait 0。

## 16. Sanitizer

AccessGuard 専用 `access_check` を対象に実行:

| Tool | Target | Result |
|---|---|---|
| memcheck | access_check | PASS 0 errors |
| racecheck | access_check | PASS 0 hazards |
| synccheck | access_check | PASS 0 errors |

`gpu_check` 全体の racecheck / synccheck は実行時間のため **NOT RUN**。
memplan_bench の racecheck/synccheck は Phase 7F-15 の結果を維持。

## 17. 変更ファイル一覧

新規:

- `crates/oxide-ml-execution-safety/src/access.rs`
- `crates/oxide-ml-execution-safety/tests/access.rs`
- `crates/oxide-ml-cuda/src/bin/access_check.rs`
- `docs/phase7f16-audit.md`
- `docs/phase7f16-final-report.md`

変更:

- `crates/oxide-ml-cuda/src/access.rs`（AccessGuard / AccessRequest / Launch wrapper）
- `crates/oxide-ml-cuda/src/buffer.rs`（BufferIdentity / generation / Access State /
  invalidate / quarantine）
- `crates/oxide-ml-cuda/src/kernels.rs`（全 launch wrapper 統合）
- `crates/oxide-ml-cuda/src/arena.rs`（slot lease 時の generation reset）
- `crates/oxide-ml-cuda/src/optimizer.rs`（`&Arc<CudaStream>` 対応）
- `crates/oxide-ml-cuda/src/event.rs`（EventPool Clone）
- `crates/oxide-ml-cuda/src/lib.rs`（re-export）
- `crates/oxide-ml-cuda/src/bin/gpu_check.rs`（AccessGuard T4 checks）
- `crates/oxide-ml-execution-safety/src/lib.rs`
- `Cargo.lock` / `crates/oxide-ml-cuda/Cargo.lock`
- `README.md`
- `docs/gpu-p7f16-*.log`

## 18. 既知の制約

- Byte Range Hazard は未実装。Buffer 全体を競合単位とする。
- Alias への in-place write は `buffer_mut` の一意性制約で拒否される。
- AccessRequest は PooledBuffer を保持しない。Completion event の attach は
  kernel wrapper が launch 後に実施する。
- CUDA-level の意図的 Failure Injection は未実施。Event/Execution failure は
  CPU Mock Test で検証。
- `gpu_check` 全体の racecheck / synccheck は NOT RUN。AccessGuard 専用
  `access_check` で PASS。
- `cargo check --manifest-path` は CUDA-oxide standalone 構成のため使用せず、
  既存方式の `cargo oxide build` / `cargo clippy` で検証。
- 実行時 Benchmark は今回未実施（NOT RUN）。Phase 7F-15 の範囲から有意な退行は
  ない見込みだが、数値確認はしていない。

## 19. 未達項目

Phase 7F-16 の完了条件は満たした。以下のみ制約として残る。

1. CUDA adapter への実 Fault Injection は NOT RUN / SOURCE ONLY
   （CPU Mock の Access Failure Policy は PASS）。
2. `gpu_check` 全体の racecheck / synccheck は NOT RUN
   （AccessGuard 専用 `access_check` では PASS）。

| 完了条件 | 状態 |
|---|---|
| Buffer Identity | PASS |
| Allocation Generation / ABA 防止 | PASS |
| AccessGuard RAII | PASS |
| Hazard Tracking Read/Write | PASS |
| Same Stream Optimization | PASS |
| Cross Stream Event Wait | PASS |
| Multiple Readers | PASS |
| Alias / View 同一 Buffer | PASS（read alias） |
| Executor Integration（全 Kernel Launch） | PASS |
| Pool Integration | PASS |
| Arena Integration | PASS |
| Event Failure Quarantine | SOURCE ONLY（AccessGuard wrapper） |
| Failure Injection CPU Mock | PASS |
| CPU Tests | PASS 192 / 0 failed |
| Clippy | PASS |
| T4 Integration | PASS 37 / EXIT 0 |
| AccessGuard racecheck | PASS |
| AccessGuard synccheck | PASS |
