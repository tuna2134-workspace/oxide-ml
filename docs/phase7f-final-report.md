# Phase 7F Final Report (pre-closure snapshot)

> **Status note (2026-11-19):** this file predates the final freeze. The authoritative
> final decision is `phase7f-final-closure.md` (Implementation COMPLETE / Verification PARTIAL).

## 1. Phase 7F 最終判定

**PARTIAL.**

Phase 7F-16 / 7F-17 / 7F-18 は Closure 完了。Phase 7F-19 は主要 Benchmark を
実行したが、Backward Graph / Linear Regression / 4 Stream Benchmark /
Full Shape Matrix が NOT RUN のため PARTIAL。

| Phase | Status |
|---|---|
| 7F-16 AccessGuard | COMPLETED |
| 7F-17 Dynamic Shape | COMPLETED |
| 7F-18 Concurrent Execution | COMPLETED |
| 7F-19 Benchmark | PARTIAL |
| Final CPU / CUDA / Clippy / Docs | PASS |

## 2. Bounded Dynamic Capacity

```rust
let (plan, signature) = executor.plan_bounded(&graph, &max_bindings)?;
```

- max shape で Graph を specialize。
- MemoryPlanner が max capacity の slot を作る。
- `CudaMemoryPlan::capacity_bindings()` / `max_capacity_bytes()`。
- Runtime Required Bytes / max bytes / alignment / offset / slot capacity を
  Kernel Launch 前に検証。
- Capacity 超過は `CudaError::Capacity`。Kernel は起動しない。

T4: batch 1 / 4 / 8 / max 8 PASS、batch 9 は Capacity Error。

## 3. Runtime Required Bytes

`checked_numel` が usize overflow を検出。dtype size を乗算。
`runtime bytes <= assignment capacity` と `offset + bytes <= slot capacity` を
実行時検証。Symbol actual <= maximum も検証。

## 4. Capacity Overflow

- 明示エラー。
- Buffer Overflow なし。
- 既存 Arena / Lease を破壊しない。
- 自動再確保はしない。
- 不正 Plan は Cache に登録しない（呼び出し側がエラー）。

## 5. Unbounded Pool Fallback

- `runtime_shapes` が実入力から全 Node 出力 shape を `infer_outputs` で具体化。
- `Dim::Dynamic` 出力も runtime で確定。
- Plan assignment がない value は Pool allocation。
- Static / Bounded は Arena、Unbounded は Pool。
- T4: batch 3 / 5 の Unbounded Output を Normal / Planned で PASS。

## 6. Mixed Graph

- Static Parameter / Bounded Symbol / Unbounded Output を同一 Graph で実行可能。
- 専用 Mixed Benchmark Graph は NOT RUN。
- Allocation は value ごとに plan assignment の有無で分岐。

## 7. 4 Stream / 4 Execution

`concurrency4_check`:

```text
serial_span_ms=192.874
parallel_span_ms=131.385
serial_host_ms=147.710
parallel_host_ms=9.444
total_overlap_ms=122.080
speedup=1.468
PASS actual GPU overlap
```

- 4 independent streams / 4 executions。
- Shared Read: 4 Reader + Writer waits=4、read_read=3。
- Arena capacity 4 / 4 threads PASS。

## 8. GPU Overlap

CUDA Event Timestamp で各 Stream の開始・終了を記録。Union overlap 122.08 ms。
Host Submission 9.44 ms。Host の短さだけを根拠にしていない。

## 9. Deadlock / Starvation

- Arena capacity 1 Busy / drop 後再 lease。
- Capacity 2 / 4 並行 lease。
- AccessGuard BufferId lock ordering。
- Multiple Reader → Writer waits all readers。
- Reader prune。
- 循環 Event Wait の意図的 test は NOT RUN（API が循環を作らない）。

## 10. CUDA Fault Injection

- Event Create/Record/Query/Sync/Destroy: event-safety Mock PASS。
- Stream Wait / Kernel Launch / Event Record: access_failure Mock PASS。
- Context Binding / Device Lost / Buffer Alloc/Release:
  execution-safety Mock PASS。
- 実 CUDA Driver Fault は **SOURCE ONLY / NOT RUN**。
- Launch 後 Record failure は Event quarantine + Buffer invalidate。

## 11. Event / Buffer Safety

- EventLease が EventPool 再利用を防止。
- AccessCompletion が EventLease を保持。
- Failed Event は Quarantine。
- Invalid Buffer は Buffer Quarantine。
- AccessGuard Drop だけでは Access State を消さない。
- Buffer Generation は Pool / Arena 再利用で更新。

## 12. Compute Sanitizer

| Target | memcheck | racecheck | synccheck |
|---|---|---|---|
| access_check | PASS 0 | PASS 0 | PASS 0 |
| concurrency_check | PASS 0 | NOT RUN | PASS 0 |
| gpu_check（全体） | PASS 0 | PASS 0 | PASS 0 |
| dynamic_check | PASS 0 | PASS 0 | PASS 0 |
| race4_check（4 Stream） | PASS 0 | PASS 0 | PASS 0 |
| benchmark_check | PASS 0 | NOT RUN | NOT RUN |

`gpu_check` 全体 racecheck / synccheck は完走。4 Stream は race4_check で PASS。

## 13. Benchmark Matrix

Graph:

- Small MLP / Large MLP / MatMul / Activation / Reduction / XOR: PASS
- Dynamic Batch: PASS
- Linear Regression / Backward / Mixed: NOT RUN

Execution:

- Normal / Planned / Arena / Async Planned / Async Arena / Dynamic: PASS
- 4 Stream Benchmark Matrix: NOT RUN

Shape:

- Small / Large MLP / Dynamic batch 1/4/8 / Bounded max 8 / Unbounded output: PASS
- Full Shape Sweep: NOT RUN

Stream:

- 1 / 2 / 4 (raw GEMM): PASS
- 4 Stream graph benchmark: NOT RUN

## 14. GPU / Host Timing

`benchmark_check`:

- Start Event: before submission loop。
- End Event: after submission loop（async は wait 後）。
- Host Timer: submission loop のみ（async wait を含めない）。
- Warmup 2 / Rounds 3 / Iterations 10。
- Async Host Submit < GPU Time を確認（例 async_planned 0.1203 vs 0.1312 ms）。

P90/P95/P99/min/max/stddev は Rounds=3 のため NOT RUN。

## 15. Memory Statistics

論理: Logical / Planned Logical / Runtime Required（bounded は検証）。
物理: Reserved / Active / Free / Pending / Quarantined。
Arena: Capacity / Used / Slot Allocation / Slot Reuse。
Event: Allocation / Reuse / Pending / Quarantine。
Access: Read / Write / ReadWrite / Same-Stream Elided / Cross-Stream / Read-Read。
Driver 実メモリは未取得。Allocator 値と区別。

## 16. CPU Tests

```text
cargo test --workspace
passed=192 failed=0
```

execution-safety 35 / event-safety 22 / その他 135。

## 17. CUDA Tests

```text
gpu_check: 40 PASS / EXIT 0
access_check: PASS
concurrency_check: PASS
concurrency4_check: PASS
dynamic_check: PASS
race4_check: PASS
benchmark_check: PASS
```

CUDA Clippy: PASS。CPU Clippy: PASS。Documentation: PASS。

## 18. 変更ファイル一覧

- `crates/oxide-ml-cuda/src/executor.rs`
  - runtime_shapes / plan_bounded / specialize_graph_with_bindings /
    validate_runtime_capacity / CudaShapeSignature
- `crates/oxide-ml-cuda/src/memplan.rs`
  - capacity_bindings / max_capacity_bytes / assignment offset
- `crates/oxide-ml-cuda/src/error.rs`
  - CudaError::Capacity
- `crates/oxide-ml-cuda/src/bin/gpu_check.rs`
  - bounded / unbounded / capacity4 tests
- `crates/oxide-ml-cuda/src/bin/concurrency4_check.rs`
- `crates/oxide-ml-cuda/src/bin/dynamic_check.rs`
- `crates/oxide-ml-cuda/src/bin/race4_check.rs`
- `crates/oxide-ml-cuda/src/bin/benchmark_check.rs`
- `docs/phase7f-closure-*.md`
- `docs/phase7f17-bounded-capacity-report.md`
- `docs/phase7f17-dynamic-output-report.md`
- `docs/phase7f18-four-stream-report.md`
- `docs/phase7f18-deadlock-report.md`
- `docs/phase7f-fault-injection-report.md`
- `docs/phase7f-sanitizer-final-report.md`
- `docs/phase7f19-full-benchmark-report.md`
- `docs/phase7f19-benchmark-results.json` / `.csv`
- `docs/phase7f-final-report.md`

## 19. 既知の制約

- 実 CUDA Driver Fault は SOURCE ONLY / NOT RUN。
- 循環 Event Wait の意図的 deadlock test は NOT RUN（source-level 保証のみ）。
- Backward は normal path のみ PASS。planned/arena/async は executor の
  host-to-device copy panic で BLOCKED。LayerNorm backward は NOT APPLICABLE。
- Linear Regression は BLOCKED（同じ panic、JSON 未生成）。
- Mixed Graph / 4 Stream Graph Matrix / Full Shape Matrix は実測済み
  （`phase7f19-*-report.md`）。
- P90/P95/P99 は 100 samples/case で取得済み。
- 新規バイナリの Compute Sanitizer は NOT RUN。
- T4 は 1 GPU。Cross-GPU / Device Lost は NOT RUN。
- GitHub Actions GPU Runner はない。

## 20. 未達項目

1. Backward planned / arena / async モード
2. Linear Regression Benchmark
3. 新規バイナリの Compute Sanitizer
4. 実 CUDA Driver Fault Injection
5. Circular Event Wait の timeout 付き hostile test

## 21. 次の開発候補

Phase 8 には着手しない。次候補は executor の planned constant-copy panic 修正、
Linear Regression 再実行、Backward planned/arena/async、新規バイナリの Sanitizer。