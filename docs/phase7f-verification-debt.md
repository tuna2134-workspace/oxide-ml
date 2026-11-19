# Phase 7F Verification Debt

最終更新: 2026-11-19
状態: Phase 7F は実装凍結済み。以下は未完了の検証項目であり、実装済み機能を
未実装扱いに戻すものではない。

## 1. 分類

- **Verification Debt**: 検証が未実施・失敗・タイムアウト。
- **Toolchain Debt**: 環境・ツールチェーン由来で検証できない。
- **Operator Backlog**: 未実装演算子。
- **Performance Backlog**: 性能改善候補。

Phase 8 の開発開始は、下記の非 Blocking 項目によって停止しない。
本番運用前に必須な項目には **Production Gate** を付ける。

## 2. Verification Debt 一覧

| ID | 対象機能 | 現在の状態 | これまでの検証 | 未実施の検証 | 再検証環境 | 完了条件 | Phase 8 依存 |
|---|---|---|---|---|---|---|---|
| VD-01 | Real CUDA Driver Fault Injection | SOURCE ONLY / NOT RUN | CPU Mock（event-safety / execution-safety）、T4 正常系 | Context Lost / Device Lost / Event create-record-query-sync failure / Buffer alloc-release failure / Kernel launch failure の実注入 | T4 または専用 GPU、隔離プロセス、timeout | 各 fault が Failed / quarantine / deferred release に遷移し、プロセスが安全に停止 | 非 Blocking（Production Gate） |
| VD-02 | Linear Regression racecheck | NOT RUN（timeout） | memcheck PASS、synccheck PASS、通常 run 99 PASS | 122 cases 全体の racecheck | T4、`timeout` 付き、必要なら matrix 縮小 | 全 case で 0 hazards、または縮小 matrix の根拠を記録 | 非 Blocking |
| VD-03 | Concurrency Graph racecheck | NOT RUN（timeout） | memcheck PASS、通常 run 177 cases | 177 cases の racecheck | T4、縮小 matrix 可 | Cross-stream shared read/arena の hazard 0 | 非 Blocking |
| VD-04 | Concurrency Graph synccheck | NOT RUN | memcheck PASS | 177 cases の synccheck | T4 | 0 errors、または縮小 matrix の根拠 | 非 Blocking |
| VD-05 | Mixed Graph synccheck | **FAIL**（application AccessGuard assertion） | sanitizer 0 errors、memcheck/racecheck PASS | `same_stream_elided` の期待値と instrumented 値の一致原因の確定、assertion の scheduling-independent 化 | T4 + synccheck | 原因を説明でき、assertion が安定して PASS | 非 Blocking（ただし Production Gate） |
| VD-06 | Full `gpu_check` racecheck / synccheck | 一部 NOT RUN | 既存 Closure の memcheck / 一部 race・sync | `gpu_check` 全体の racecheck / synccheck | T4 | 0 hazards / 0 errors | 非 Blocking |
| VD-07 | Circular Event Wait hostile test | NOT RUN（source-level audit のみ） | `CudaEvent::wait_on` の Unrecorded no-op、AccessGuard / Arena の backwards-only edge 解析 | timeout 付き隔離プロセスでの循環構成試行 | T4 + timeout / 隔離プロセス | 循環が構成できないこと、または検出・拒否を実証 | 非 Blocking |
| VD-08 | Real GPU Driver Failure Recovery | NOT RUN | Mock の lifecycle / quarantine | 実 driver failure 後の Execution / Arena / Pool 回復 | T4 / 専用 GPU | Failed 遷移、resource 保持、再実行不能の明示 | 非 Blocking（Production Gate） |
| VD-09 | `cargo oxide test --arch sm_75` | BLOCKED（Toolchain） | 個別 binary build/run、clippy | test harness 全体 | cuda-oxide / rustc_codegen_cuda 更新環境 | test harness が build / run 完了 | 非 Blocking |
| VD-10 | Cross-GPU / Device Lost | NOT RUN | 単一 T4 のみ | 複数 GPU、device lost、context lost | 複数 GPU 環境 | 各 GPU で identity / arena / event が分離、lost 時安全停止 | 非 Blocking（Production Gate） |
| VD-11 | Async training (linear regression) | NOT APPLICABLE（設計） | forward-only async 実測 | async optimizer step の安全な実現 | 設計変更が必要 | unique device buffer を保った async update | 非 Blocking |
| VD-12 | Mixed graph `plan_bounded` | FAIL（API 制約） | generic planner + pool fallback で動作 | `Dim::Dynamic` を含む bounded plan | 設計変更が必要 | mixed graph の bounded plan が安全に動作 | 非 Blocking |
| VD-13 | LayerNorm Backward | NOT APPLICABLE（Operator 未実装） | LayerNorm forward 5 modes | backward kernel / gradient 検証 | CUDA 実装後 | CPU との勾配一致、Sanitizer PASS | 非 Blocking（Operator Backlog） |

## 3. Toolchain Debt

### TD-01: `cargo oxide test --arch sm_75` device codegen failure

- 発生コマンド:

```bash
cd crates/oxide-ml-cuda
CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda \
  cargo oxide test --arch sm_75
```

- エラー:

```text
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
Verification failed ... DisjointSlice::get_unchecked_mut ...
MirFieldAddrOp pointer must point to a struct, tuple, union or enum type,
got: mir.disjoint_slice <builtin.fp32 ,[]>
```

- 影響範囲: CUDA test harness 全体（lib test / integration test）。
  個別 benchmark binary の `cargo oxide build` と実行は成功している。
- 回避できた検証: `cargo oxide build` + 個別 binary、`constant_copy_check`、
  `backward_check_full`、`linear_regression_check`、Sanitizer。
- 断定しないこと: 「oxide-ml の CUDA 実装の機能不具合」とも
  「完全に test harness だけの問題」とも断定しない。
- 再検証条件: cuda-oxide / rustc_codegen_cuda / Rust nightly の更新時に
  `cargo oxide test` を再実行し、同じ `DisjointSlice` エラーが出ないこと。

## 4. Phase 8 との関係

Phase 8 の開発開始を Blocking する項目はない。ただし次は Production Gate とする:

- VD-01 / VD-05 / VD-08 / VD-10。

理由: 実 driver fault、mixed synccheck の統計 assertion、cross-GPU / device lost は
本番運用の安全性説明に必要だが、Phase 8 の機能開発を止めるものではない。

## 5. 更新ルール

- Phase 7F の実装は凍結済み。検証の追加は行ってよいが、新機能追加は行わない。
- 各項目を PASS に変更するときは、実行コマンド・ログ・終了コードを記録する。
- NOT RUN を PASS に読み替えない。
- Mixed synccheck は sanitizer 0 errors でも application assertion が失敗している限り
  FAIL のままとする。
