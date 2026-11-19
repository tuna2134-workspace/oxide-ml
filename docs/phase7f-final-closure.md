# Phase 7F Final Closure

最終更新: 2026-11-19
状態: **FROZEN**（Phase 7F の実装範囲を凍結）

## 1. Executive Summary

Phase 7F の実装範囲を凍結し、Phase 7F-13〜7F-19 と Constant Copy Recovery の
成果を最終整理した。

- **Phase 7F Implementation: COMPLETE**
- **Phase 7F Verification: PARTIAL**
- **Closure Status: CLOSED WITH VERIFICATION DEBT**
- **Phase 8 Readiness: READY**

実装した安全性機構（Event Safety / AccessGuard / Async Lifecycle /
Arena Identity / Dynamic Shape / Concurrent Execution）は凍結する。
検証不足は `phase7f-verification-debt.md` に独立した追跡項目として移し、
Phase 8 の進行を止めない。未実装演算子は
`phase7f-operator-backlog.md` に移す。

本 Closure では新機能を追加していない。既存コードの変更は、既に報告済みの
Constant Copy 修正と、その回帰テスト・ベンチマーク復旧に限定される。
復旧の詳細は `phase7f19-recovery-final-report.md`、
不具合解析は `phase7f19-constant-copy-investigation.md` を参照。

**「すべての検証が完了した」「完全に安全性が証明された」とは主張しない。**

## 2. Phase 7F Scope

| Phase | Scope | Implementation | Verification |
|---|---|---|---|
| 7F-13 | Event Failure Safety / quarantine / deferred reclamation | COMPLETE | PASS（Mock + T4 正常系） |
| 7F-14 | Arena Identity / Canonical Hash / Registry | COMPLETE | PASS（T4） |
| 7F-15 | Async Execution Lifecycle | COMPLETE | PASS（T4 + Mock） |
| 7F-16 | AccessGuard / Buffer Hazards | COMPLETE | PASS（T4 + Mock） |
| 7F-17 | Dynamic Shape / Bounded Capacity / Pool Fallback | COMPLETE | PASS（T4） |
| 7F-18 | Concurrent Execution / 2・4 Stream | COMPLETE | PASS（T4） |
| 7F-19 | Benchmark / Statistics / Constant Copy Recovery | COMPLETE | PARTIAL（Sanitizer / Toolchain の Debt あり） |

In scope:

- CUDA backend の安全性・実行ライフサイクル・メモリ計画・動的 shape・並行実行。
- CPU reference との数値比較。
- T4 実機での Benchmark と Sanitizer。

Out of scope（Phase 8 以降）:

- 新しい CUDA 演算子、性能最適化、Cross-GPU、分散学習。
- 本番運用向けの fault injection framework。
- Phase 7F の完了条件の追加・変更。

## 3. Implemented Features

### 3.1 Event Failure Safety

- Event 状態: Unrecorded / Recorded / Completed / Failed。
- Failed は終端。再 Record・再利用しない。
- Failed Event / Buffer は free list ではなく quarantine へ。
- Buffer は全 completion event が明示的に Completed の場合のみ再利用。
- CPU Mock backend で create / record / query / synchronize / destroy failure を検証。
- T4 では正常系の EventPool / pending / drop safety を検証。

### 3.2 Arena Identity / Canonical Hash

- `Graph::canonicalize()` → GraphHash（BLAKE3）。
- `MemoryPlan` → CanonicalPlan → PlanHash。
- `ArenaKey = GraphHash + PlanHash + DeviceId + ContextId + DType + PlanVersion`。
- Hash collision 時は `ArenaDescriptor` の完全比較。一致しなければ別 Arena。
- Device は ordinal + UUID + name + compute capability、Context は raw CUcontext +
  process-local id。
- MemoryPlan の slot capacity / alignment / assignment / lifetime を検証。

### 3.3 Async Execution Lifecycle

- 状態: Submitted / Running / Completed / Collected / Failed。
- `is_complete` は Event Query で非ブロッキング確認。
- `wait` / `collect` / 二重 collect 拒否。
- Drop は host synchronize しない。Arena は PendingCompletion、Buffer は
  deferred reclamation / quarantine。
- Executor Drop 後も Execution が保持する context / stream / event / buffer は存続。

### 3.4 AccessGuard

- `BufferIdentity { id, generation, context_id }`。
- Read/Read は並行。Write/ReadWrite は他 Stream の reader を待つ。
- 同一 Stream の hazard は CUDA stream order により elide。
- Cross-stream は Event Wait。
- Launch failure / Event failure は Buffer を invalidate し quarantine。
- 4 Stream Shared Read / Shared Write を実機検証。

### 3.5 Dynamic Shape

- Runtime shape binding と shape signature。
- Bounded symbol の max capacity plan。
- Runtime capacity validation（kernel launch 前）。
- Unbounded `Dim::Dynamic` は generic plan の unplanned value として Pool Fallback。
- `CudaError::Capacity` で明示拒否。

**API 制約（重要）**: `plan_bounded()` は `Dim::Dynamic` を含む Graph を受け付けない。
Mixed Static / Bounded / Unbounded Graph は Generic Planner と Pool Fallback で動作するが、
Mixed Graph をそのまま `plan_bounded()` に渡せるわけではない。この制約は未解消であり、
解消したとは報告しない。

### 3.6 Concurrent Execution

- Multi-stream execution、1 / 2 / 4 Stream。
- AccessGuard による cross-stream hazard 制御。
- Shared Read / Shared Parameters / Shared Arena / Pool+Arena Mixed。
- 4 Stream Graph Matrix は **177 cases**:
  - 144 PASS
  - 30 CPU Reference unavailable（mixed graph の `Dim::Dynamic` を CPU が specialize できない）
  - 3 NOT APPLICABLE（Shared Write は functional graph では表現不能）
- 「177 ケースすべて成功」ではない。

### 3.7 Benchmark / Statistics

- Forward 6 graph × 5 mode = 30 cases、100 samples/case。
- Backward 7 graph（LayerNorm は forward のみ）× forward/training × 5 mode = 70 cases。
- Linear Regression 122 cases（99 PASS / 23 N/A）。
- Mixed Graph 5 cases。
- Shape Matrix 44 cases（42 PASS / 2 N/A）。
- Warmup 10 / Rounds 10 / Iterations 10。
- Min / Max / Mean / Median / P90 / P95 / P99 / StdDev、nearest-rank。
- GPU Event Time と Host Wall-clock を分離。同期 API の implicit wait は
  host_submit に含め、async の wait とは分ける。

### 3.8 Constant Copy Recovery

Planned Slot の物理 Capacity と Tensor の論理要素数が異なる場合の H2D Copy を修正。

- 論理要素数と Host Data 長を検証。
- Device Capacity >= 論理要素数を検証。
- Capacity 一致時は exact copy。
- Capacity が大きい場合は **論理 Prefix のみ** copy。
- Host Data 過剰 / Capacity 不足は Error。
- Broadcast / 不要な展開は行わない。
- Benchmark 専用分岐・固定長回避策はない。

## 4. Architecture Summary

```text
Graph
  -> GraphVerifier
  -> CanonicalGraph / GraphHash
  -> MemoryPlanner -> MemoryPlan -> CanonicalPlan / PlanHash
  -> CudaMemoryPlan
       -> Normal execution (pool)
       -> Planned execution (planned slots, pool fallback)
       -> Arena execution (SlotSetLease)
       -> Async planned / arena (CudaExecution)
  -> CudaKernels
       -> AccessGuard (per BufferIdentity)
       -> EventPool (completion / quarantine)
       -> CudaBufferPool (free / pending / quarantine)
```

Crates:

- `oxide-ml-core`: dtype, shape, tensor, MemoryPlan.
- `oxide-ml-graph`: SSA IR, builder, shape inference, verifier, canonical hash.
- `oxide-ml-compiler`: passes, memory planner, canonical plan.
- `oxide-ml-cpu`: reference executor / planner.
- `oxide-ml-autograd`: training graph construction.
- `oxide-ml-event-safety`: event state machine / pool / quarantine policy.
- `oxide-ml-execution-safety`: BufferIdentity / AccessGuard state / execution lifecycle.
- `oxide-ml-cuda`: cuda-oxide kernels, executor, arena, pool, event adapter.

Safety invariants:

1. Failed event is terminal and never reused.
2. Buffer is reusable only after all completion events are explicitly Completed.
3. AccessGuard orders conflicting cross-stream accesses; Read/Read is concurrent.
4. Async submit never synchronizes the host; Drop never synchronizes.
5. Capacity overflow is rejected before kernel launch.
6. Canonical identity + exact descriptor prevents unsafe arena sharing.

## 5. Test Results

Classification: **PASS** = executed and succeeded; **FAIL** = executed and failed;
**NOT RUN** = not executed; **BLOCKED** = environment/toolchain prevented execution.
SOURCE ONLY is never counted as a real-machine PASS.

| Item | Command | Result |
|---|---|---|
| CPU workspace | `cargo test --workspace` | **PASS 198 / FAIL 0** |
| CPU clippy | `cargo clippy --workspace --all-targets -- -D warnings` | **PASS** |
| CPU docs | `cargo doc --workspace --no-deps` | **PASS** |
| CUDA clippy | `cargo clippy --all-targets -- -D warnings` | **PASS**（`p7f19-recovery-cuda-clippy.log`, EXIT 0） |
| CUDA binary build | `cargo oxide build --arch sm_75 -- --bin <name> --release` | **PASS** |
| CUDA test harness | `cargo oxide test --arch sm_75` | **FAIL / BLOCKED**（Toolchain Debt） |

T4 verification binaries:

| Binary | Result |
|---|---|
| `constant_copy_check` | PASS |
| `backward_check_full` | 65 PASS / 0 FAIL / 5 N/A |
| `linear_regression_check` | 99 PASS / 0 FAIL / 23 N/A |
| `mixed_graph_check` | 5/5 PASS（plan_bounded の API 制約は FAIL として記録） |
| `concurrency_graph_check` | 144 PASS / 30 CPU ref unavailable / 3 N/A |
| `shape_matrix_check` | 42 PASS / 2 N/A / 0 FAIL |
| `benchmark_check` | 30/30 PASS |
| `gpu_check` | 40 PASS（既存 Closure 記録） |

Logs: `p7f19-final-cpu-tests.log`, `p7f19-final-cpu-clippy.log`,
`p7f19-final-cpu-doc.log`, `p7f19-final-cuda-build.log`,
`p7f19-recovery-cuda-clippy.log`, `p7f19-recovery-cuda-test.log`, and the
per-binary result JSON/CSV in `docs/`.

## 6. Benchmark Results

### Forward

| Graph | normal | planned | arena | async_planned | async_arena |
|---|---:|---:|---:|---:|---:|
| small_mlp | 0.1209 | 0.1068 | 0.1390 | 0.1103 | 0.1455 |
| large_mlp | 0.3129 | 0.3092 | 0.3297 | 0.2994 | 0.2640 |
| matmul | 0.0437 | 0.0435 | 0.0740 | 0.0382 | 0.0751 |
| activation | 0.0418 | 0.0423 | 0.0635 | 0.0365 | 0.0692 |
| reduction | 0.0477 | 0.0480 | 0.0614 | 0.0412 | 0.0707 |
| dynamic_batch | 0.0395 | 0.0389 | 0.0534 | 0.0348 | 0.0683 |

Median GPU event ms. Full statistics are in
`phase7f19-benchmark-results.json` / `.csv`.

### Backward（training graph, median ms）

| Graph | normal | planned | arena | async_planned | async_arena |
|---|---:|---:|---:|---:|---:|
| linear_mse | 0.3277 | 0.2698 | 0.3210 | 0.2437 | 0.3546 |
| mlp2_mse | 0.6042 | 0.4937 | 0.6061 | 0.4607 | 0.6224 |
| mse_only | 0.1576 | 0.1244 | 0.1564 | 0.1059 | 0.1391 |
| relu_activation | 0.2747 | 0.2113 | 0.2396 | 0.1980 | 0.2793 |
| add_relu | 0.1927 | 0.1516 | 0.1896 | 0.1428 | 0.2068 |
| unbroadcast_bias | 0.1561 | 0.1447 | 0.1745 | 0.1302 | 0.1964 |
| layernorm | N/A | N/A | N/A | N/A | N/A |

LayerNorm forward is measured; LayerNorm backward is N/A（未実装）。

### Linear Regression

99 PASS / 23 N/A / 0 FAIL。Normal / Planned / Arena は全 combo PASS。
Async は forward-only を実測し、async training は N/A（optimizer が host-visible な
unique device buffer を要求するため）。

### Mixed / Shape / Concurrency

- Mixed: 5 modes PASS。batch 9 は Capacity Error で launch 前に拒否。
- Shape: 44 cases、42 PASS、2 N/A（unbounded arena / async_arena）。
- Concurrency: 177 cases、144 PASS、30 CPU ref unavailable、3 N/A。
  4 Stream の GPU overlap は実測されているが、小 graph のため speedup は約 1.0。

## 7. Sanitizer Results

| Target | memcheck | racecheck | synccheck |
|---|---|---|---|
| `constant_copy_check` | PASS (0) | PASS (0 hazards) | PASS (0) |
| `backward_check_full` | PASS (0) | PASS (0 hazards) | PASS (0) |
| `linear_regression_check` | PASS (0) | **NOT RUN**（timeout） | PASS (0) |
| `mixed_graph_check` | PASS (0) | PASS (0 hazards) | **FAIL**（application AccessGuard assertion; sanitizer 0 errors） |
| `concurrency_graph_check` | PASS (0) | **NOT RUN**（timeout） | **NOT RUN** |
| `shape_matrix_check` | PASS (0) | PASS (0 hazards) | PASS (0) |
| `benchmark_check` | PASS (0) | PASS (0 hazards) | PASS (0) |

Mixed synccheck は sanitizer 自体が 0 errors でも、`same_stream_elided` の期待値と
instrumented 実行時の値が一致せず、アプリケーション側の assertion が失敗している。
**PASS に変更しない。** 原因は未確定であり、推測で解決済みにしない。

## 8. Known Limitations

1. `plan_bounded()` は `Dim::Dynamic` を含む Graph を受け付けない。Mixed Graph は
   generic planner + pool fallback。
2. Linear Regression の async training は N/A（optimizer semantics）。
3. LayerNorm Backward は CUDA 未実装。
4. Constant prefix copy は lifetime safety のため同期を維持。
5. Sanitizer の一部セルは NOT RUN。Mixed synccheck は application FAIL。
6. `cargo oxide test` は test harness の codegen で失敗。
7. 実 CUDA Driver Fault Injection は SOURCE ONLY / NOT RUN。
8. Circular Event Wait は source-level audit のみ。hostile timeout test は NOT RUN。
9. T4 1 GPU。Cross-GPU / Device Lost は NOT RUN。
10. Benchmark は単一セッションの median。cross-run / cross-GPU の再現性は未検証。

性能改善候補は `phase7f-performance-backlog.md` に分離した。

## 9. Verification Debt

`phase7f-verification-debt.md` を参照。主な項目:

- VD-01 Real CUDA driver fault injection（SOURCE ONLY / NOT RUN）
- VD-02 Linear Regression racecheck（NOT RUN）
- VD-03/04 Concurrency racecheck / synccheck（NOT RUN）
- VD-05 Mixed Graph synccheck（FAIL）
- VD-06 Full `gpu_check` racecheck / synccheck（未完了範囲あり）
- VD-07 Circular Event Wait hostile test（NOT RUN）
- VD-08 Real GPU driver failure recovery（NOT RUN）
- VD-09 `cargo oxide test` harness（BLOCKED / Toolchain Debt）
- VD-10 Cross-GPU / Device Lost（NOT RUN）

## 10. Toolchain Debt

`cargo oxide test --arch sm_75` が cuda-oxide test harness の
`DisjointSlice` MIR verification で失敗する。

```text
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
Verification failed ... DisjointSlice::get_unchecked_mut ...
MirFieldAddrOp pointer must point to a struct, tuple, union or enum type
```

- 発生コマンド: `cargo oxide test --arch sm_75`
- 影響: CUDA test harness 全体。個別 benchmark binary の build / run は成功。
- 回避できた検証: `cargo oxide build` + 個別 binary 実行、`constant_copy_check` 等。
- 断定しないこと: 「CUDA 実装の機能不具合」とも「完全に test harness だけの問題」とも
  断定しない。再検証条件は cuda-oxide / rustc_codegen_cuda の更新時。

## 11. Operator Backlog

`phase7f-operator-backlog.md` を参照。

- LayerNorm Backward
- Conv2d / MaxPool2d / AvgPool2d（forward/backward）
- CrossEntropyLoss（forward/backward）
- F64 CUDA
- Batched MatMul
- F16 / BF16
- CUDA Graph / Tensor Core
- RNN / GRU / LSTM / Attention / Embedding / BatchNorm
- Mixed Graph の `plan_bounded()` 対応（設計 backlog）

これらは Phase 7F の完了を妨げない。

## 12. Final Decision

### Critical Findings

レビューしたコードと既存レポートの範囲では、Phase 7F のクローズを止める
**重大な安全性・正当性の欠陥は確認されなかった**。

- Mixed synccheck の AccessGuard assertion は検証 FAIL であり、instrumented
  実行の scheduling 依存統計に関する問題として追跡する。実ランタイムの
  安全性欠陥と断定しない。
- `cargo oxide test` の失敗は cuda-oxide test harness の codegen であり、
  個別 benchmark binary の build/run 成功とは分離して扱う。実装の機能不具合と
  断定しない。
- `plan_bounded()` の `Dim::Dynamic` 非対応は既知の API 制約であり、
  Mixed Graph は generic planner + pool fallback で動作する。

| Item | Decision |
|---|---|
| Phase 7F Implementation | **COMPLETE / FROZEN** |
| Phase 7F Verification | **PARTIAL** |
| Closure Status | **CLOSED WITH VERIFICATION DEBT** |
| Phase 8 Readiness | **READY** |

Phase 8 を開始する条件:

- 実装範囲が本文書で確定している。→ 満たす。
- 既知の重大な安全性問題が残っていない。→ レビュー範囲では重大な安全性欠陥は
  見つかっていない。Mixed synccheck の AccessGuard assertion は検証 FAIL として
  追跡し、安全性欠陥と断定しない。
- PASS / FAIL / NOT RUN / BLOCKED が区別されている。→ 満たす。
- 未解決事項が追跡可能である。→ Verification Debt / Toolchain Debt /
  Operator Backlog / Performance Backlog に記録。
- Phase 8 が Phase 7F の未完了検証に暗黙依存しない。→ 本 Closure の範囲では
  依存しない。将来の本番運用前には Verification Debt の解消が必要。

Phase 8 の推奨テーマ候補:

1. CUDA operator coverage（LayerNorm Backward / Conv / CrossEntropy）。
2. Verification Debt の解消（Sanitizer 全セル、real driver fault）。
3. Performance work（小 graph の submit overhead、arena の直列化）。
4. Cross-GPU / device lost / production fault recovery。

本 Closure は Phase 8 の機能実装には着手していない。
