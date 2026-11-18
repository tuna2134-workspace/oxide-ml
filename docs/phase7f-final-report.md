# Phase 7F Final Report

## 1. Phase 7F 最終判定

**PARTIAL.**

Phase 7F-16 / 7F-18 / 7F-19 の主要機能は実装・T4 検証済み。
Phase 7F-17 は Symbolic Dynamic Shape の実行と Shape Specialization / Arena を
実装したが、**Bounded Dynamic Capacity** と **Unbounded `Dim::Dynamic` 出力の
Pool Fallback** が未達のため、Phase 7F 全体は COMPLETED ではない。

| Phase | Status |
|---|---|
| 7F-16 AccessGuard | COMPLETED |
| 7F-17 Dynamic Shape | PARTIAL |
| 7F-18 Concurrent Execution | PARTIAL（2 Stream 実 Overlap PASS / 4 Stream NOT RUN） |
| 7F-19 Benchmark | PARTIAL（主要 Benchmark PASS / 全 Matrix NOT RUN） |
| Final CPU / CUDA / Clippy / Docs | PASS |

## 2. アーキテクチャ変更

- BufferIdentity（id / generation / context）を PooledBuffer に導入。
- AccessRequest / AccessGuard / AccessCompletion を CUDA に追加。
- BufferAccessState を純粋 Hazard State として execution-safety に配置。
- 全 CudaKernels Launch Wrapper を AccessGuard 経由に変更。
- CUDA Executor に runtime Symbol binding / Shape Signature / Graph
  Specialization を追加。
- CudaEvent に GPU Event elapsed_ms を追加。
- T4 用 access_check / concurrency_check / benchmark_check を追加。

## 3. AccessGuard

- Read/Read: Wait なし。
- Read/Write, Write/Read, Write/Write: 異 Stream Event Wait。
- ReadWrite: Write と同等。
- Same Stream は Stream Order で elide。
- Multiple Reader は writer が全 reader を wait。
- AccessRequest は PooledBuffer の strong/weak ref を持たず、
  `buffer_mut` の一意性を壊さない。
- Kernel wrapper が launch 後に completion event を Tensor へ attach。

T4 `gpu_check`: **37 PASS / EXIT 0**。

Wait Count:

```text
read/read/write: waits=2 cross=2 read_read=1
write/read:      waits=1 cross=1
write/write:     waits=1 cross=1
read-write/read: waits=1 cross=1
same-stream:     waits=0 elided=1
three-stream:    waits=3 cross=3
```

## 4. Dynamic Shape

- `bind_inputs`: graph input から Symbol を runtime bind。
- `resolve_shape`: Static / Symbol を解決、Dynamic は明示エラー。
- `CudaShapeSignature`: SymbolId + runtime dim の sorted identity。
- `specialize_graph`: Symbol を Static に置換した graph を生成。
- Specialized graph で static MemoryPlan / Persistent Arena を利用。
- T4: batch 1 / 4 / 8 の Normal / Planned / Specialized Arena PASS。

未達:

- Bounded Dynamic Capacity（max capacity 予約 / runtime required bytes 分離）
- Unbounded `Dim::Dynamic` 出力の Pool Fallback
- Capacity Overflow Test

## 5. MemoryPlan

- Static plan は既存どおり slot / alignment / logical bytes を保持。
- Symbolic value は `unplanned_values` として Pool Fallback。
- `CudaMemoryPlan` は dynamic value が slot に割り当てられた場合のみ拒否。
- Shape specialization により shape ごとの static plan を生成可能。
- Bounded capacity の plan API は未実装。

## 6. Persistent Arena

- ArenaKey / GraphHash / PlanHash / DeviceIdentity / ContextIdentity を維持。
- SlotSet lease 時に generation を進め、Access State を reset。
- Capacity 1 の Busy、Capacity 2 の並行 lease を検証。
- 2 threads / 2 streams / 同一 Arena の `run_arena` を検証。
- Dynamic shape は specialized graph ごとに Arena を生成（Registry 統合は未実施）。

## 7. Multi-Stream Execution

- `execute_async_planned` / `execute_async_arena` は内部 Host Synchronize なし。
- `run` / `run_planned` / `run_arena` は同期 API。
- Execution Drop は Host Synchronize なし。
- AccessGuard が Hazard を解決。
- Arena lease は Event Wait で reclaim。

## 8. Actual GPU Overlap

`concurrency_check`（N=1024 GEMM × 4 / stream）:

```text
serial_span_ms=103.001
parallel_span_ms=90.894
host_submit_ms=0.077 / 0.072
a_ms=81.834
b_ms=90.891
overlap_ms=81.831
speedup=1.133
PASS actual GPU overlap
```

GPU Event Timestamp で 81.8 ms の重なりを確認。Host Submission の短さだけを
根拠にしていない。

## 9. Execution Lifecycle

- ExecutionState: Submitted / Running / Completed / Collected / Failed。
- `is_complete` は Event Query のみ。
- `wait` / `collect` は明示的完了待ち。
- Execution Drop は Buffer を Pending / Quarantine へ。
- Executor Drop 後も Execution が Context / Stream / Event / Arena lease を保持。
- Double Collect はエラー。

## 10. Buffer / Event Lifetime

- EventLease が EventPool の free 再利用を防ぐ。
- AccessCompletion が EventLease を保持。
- Buffer は completion events を保持し、Event Safety の deferred reclamation へ。
- AccessGuard の Drop だけでは Access State を消さない。
- Failed Event は Quarantine。Buffer invalidate は Buffer Quarantine。

## 11. Memory Statistics

論理:

- Logical Tensor Bytes
- Planned Logical Bytes
- Runtime Required Bytes（dynamic bounded は未定義）

物理:

- Reserved / Allocated / Active / Free / Pending / Quarantined Bytes

Arena:

- Capacity / Used / Slot Capacity / Slot Allocation / Slot Reuse / Arena Count

Event:

- Allocation / Reuse / Pending / Failed / Quarantined

Access:

- Read / Write / ReadWrite / Same-Stream Elided / Cross-Stream Wait /
  Read-Read Concurrent

`logical_bytes` と `physical_reserved_bytes` は定義が異なる。前者は compiler plan、
後者は allocator が保持する DeviceBuffer 容量。

## 12. Benchmark

`benchmark_check`（T4, Small MLP, Median）:

| Mode | GPU ms | Host Submit ms |
|---|---:|---:|
| normal | 0.1294 | 0.1292 |
| planned | 0.1117 | 0.1115 |
| arena | 0.1368 | 0.1367 |
| async_planned | 0.1267 | 0.1169 |
| async_arena | 0.1570 | 0.1469 |
| dynamic_normal (batch 4) | 0.0393 | 0.0391 |
| dynamic_planned (batch 4) | 0.0385 | 0.0383 |

Pool: allocations=45, reuses=563, reserved=1760 B。  
Arena: slot_sets=10, slot_reuse=54, physical_reserved=1280 B。  
Event: allocation=11, reuse=821, pending=0, quarantine=0。

## 13. CPU Tests

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

AccessGuard:
- 13 hazard/identity tests
- 8 failure injection tests（Stream Wait / Kernel Launch / Event Record /
  Guard Drop / Lock Poison / Generation / Context / Pruning）

## 14. CUDA Tests

```text
gpu_check: 37 PASS / EXIT 0
access_check: PASS
concurrency_check: PASS actual overlap
benchmark_check: PASS
```

T4: Tesla T4 / sm_75 / CUDA 13.0 / Rust 1.99.1 / cuda-oxide 0.4.0。

## 15. Compute Sanitizer

| Target | Tool | Result |
|---|---|---|
| access_check | memcheck | PASS 0 |
| access_check | racecheck | PASS 0 hazards |
| access_check | synccheck | PASS 0 |
| concurrency_check | memcheck | PASS 0 |
| concurrency_check | synccheck | PASS 0 |
| benchmark_check | memcheck | PASS 0 |
| concurrency_check | racecheck | NOT RUN |
| gpu_check full | racecheck / synccheck | NOT RUN |

## 16. Clippy / Documentation

- CPU `cargo clippy --workspace --all-targets -D warnings`: PASS
- CUDA `cargo clippy --all-targets -D warnings`: PASS
- `cargo doc --workspace --no-deps`: PASS

## 17. 変更ファイル一覧

新規:

- `crates/oxide-ml-execution-safety/src/access.rs`
- `crates/oxide-ml-execution-safety/tests/access.rs`
- `crates/oxide-ml-execution-safety/tests/access_failure.rs`
- `crates/oxide-ml-cuda/src/bin/access_check.rs`
- `crates/oxide-ml-cuda/src/bin/concurrency_check.rs`
- `crates/oxide-ml-cuda/src/bin/benchmark_check.rs`
- `docs/phase7f-initial-audit.md`
- `docs/phase7f-master-progress.md`
- `docs/phase7f16-*.md`
- `docs/phase7f17-final-report.md`
- `docs/phase7f18-*.md`
- `docs/phase7f19-*.md`
- `docs/phase7f-final-report.md`

変更:

- `crates/oxide-ml-cuda/src/access.rs`
- `crates/oxide-ml-cuda/src/buffer.rs`
- `crates/oxide-ml-cuda/src/kernels.rs`
- `crates/oxide-ml-cuda/src/arena.rs`
- `crates/oxide-ml-cuda/src/event.rs`
- `crates/oxide-ml-cuda/src/executor.rs`
- `crates/oxide-ml-cuda/src/optimizer.rs`
- `crates/oxide-ml-cuda/src/lib.rs`
- `crates/oxide-ml-cuda/src/bin/gpu_check.rs`
- `crates/oxide-ml-execution-safety/src/lib.rs`
- `Cargo.lock` / `crates/oxide-ml-cuda/Cargo.lock`
- `README.md`
- `docs/gpu-p7f17-*.log`, `docs/gpu-p7f18-*.log`, `docs/gpu-p7f19-*.log`

## 18. 既知の制約

- Bounded Dynamic Capacity 未実装。
- Unbounded `Dim::Dynamic` 出力の Pool Fallback 未実装。
- 4 Stream / 4 Execution Matrix 未実施。
- concurrency racecheck 未実施（timeout）。
- gpu_check 全体の racecheck / synccheck 未実施。
- CUDA adapter への実 Fault Injection は SOURCE ONLY。
  CPU Mock policy は PASS。
- T4 は 1 GPU。Cross-GPU / Device Lost は NOT RUN。
- GitHub Actions に GPU Runner はない。CPU CI のみ可能。
- `cargo check --manifest-path crates/oxide-ml-cuda/Cargo.toml` は
  cuda-oxide standalone 構成のため使用しない。

## 19. 未達項目

1. Dynamic Shape Bounded Capacity / Capacity Overflow
2. Unbounded Dynamic Shape 出力 Fallback
3. 4 Streams / 4 Executions Matrix
4. concurrency_check racecheck
5. gpu_check 全体 racecheck / synccheck
6. CUDA adapter 実 Fault Injection
7. Full Benchmark Graph / Stream Matrix

## 20. 次の開発候補

- Phase 8 には着手しない。
- 次候補:
  - Bounded Dynamic Capacity Planner / ShapeSignature Arena Registry
  - 4 Stream / 4 Execution Matrix と Deadlock Detector
  - CUDA fault injection backend
  - Full benchmark suite（Large MLP / Backward / Reduction / Activation）
