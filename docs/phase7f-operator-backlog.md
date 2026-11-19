# Phase 7F Operator Backlog

最終更新: 2026-11-19
状態: Phase 7F 実装凍結。以下は Phase 7F の完了を妨げない未実装・将来拡張項目。

## 1. 分類

| 分類 | 扱い |
|---|---|
| CUDA Operator | CUDA backend に未実装。CPU reference は存在する場合がある |
| Design Extension | `plan_bounded` / mixed shape など API 設計の拡張 |
| Performance | 演算子・実行経路の性能改善 |

## 2. CUDA Operators

| ID | Operator | 現在の状態 | CPU Reference | 依存関係 | 優先度 |
|---|---|---|---|---|---|
| OP-01 | LayerNorm Backward | **未実装**（`layer_norm_backward` は executor が unsupported） | あり（`LayerNorm` の autograd rule） | LayerNorm forward / gamma-beta gradient / CUDA kernel | High（Backward 完備） |
| OP-02 | Conv2d forward | 未実装 | あり | im2col または direct kernel、shape inference はあり | Medium |
| OP-03 | Conv2d backward | 未実装 | あり | OP-02、gradient kernels | Medium |
| OP-04 | MaxPool2d forward / backward | 未実装 | あり | pooling index、kernel | Medium |
| OP-05 | AvgPool2d forward / backward | 未実装 | あり | pooling kernel | Medium |
| OP-06 | CrossEntropyLoss forward / backward | 未実装（CUDA） | あり | log-softmax / NLL、gradient | High（分類タスク） |
| OP-07 | Batched MatMul | 未実装（rank-2 のみ） | あり | batched GEMM、broadcast | Medium |
| OP-08 | F64 CUDA | 未実装（F32 のみ） | あり | dtype-generic kernel / ABI | Low |
| OP-09 | F16 / BF16 | 未実装 | 一部 | dtype / Tensor Core / conversion | Low |
| OP-10 | Softmax / LayerNorm の追加軸・大規模 shape | 一部 | あり | shape specialization / kernel tiling | Low |
| OP-11 | RNN / GRU / LSTM | 未実装 | なし | 演算子群 + state management | Low |
| OP-12 | Attention | 未実装 | なし | batched matmul / softmax / mask | Low |
| OP-13 | Embedding | 未実装 | なし | gather / scatter-add | Low |
| OP-14 | BatchNorm | 未実装 | なし | reduction / running stats | Low |
| OP-15 | CUDA Graph / Tensor Core | 未実装 | N/A | CUDA Graph API / MMA | Low |

## 3. Design Extensions

| ID | Item | 現在の状態 | 制約 | 必要な作業 | Phase 7F への影響 |
|---|---|---|---|---|---|
| DX-01 | `plan_bounded()` with `Dim::Dynamic` | **未対応**。`CudaError::Capacity` で拒否 | Mixed Static/Bounded/Unbounded graph は generic planner + pool fallback で動作 | specializer が `Dim::Dynamic` を許容し、capacity binding と unplanned value を共存させる設計 | なし（既知の API 制約として記録） |
| DX-02 | Async optimizer update | 未対応 | `CudaSgd` は unique `DeviceBuffer` を要求。`CudaExecution` が parameter alias を保持 | optimizer 用の buffer ownership / event 設計 | Linear の async training は N/A |
| DX-03 | Constant upload の完全 async 化 | 同期を維持 | host slice lifetime | pinned host / ownership tracking | Prefix copy は同期のまま |
| DX-04 | Cross-GPU arena / event identity | 単一 GPU のみ | Context / Device identity | multi-GPU registry / peer access | なし |

## 4. Performance Backlog

`phase7f-performance-backlog.md` を参照。

- 小 graph の per-stream submit / event overhead。
- Arena の slot-set 直列化。
- Constant upload の同期。
- 4 Stream の speedup が約 1.0 の領域。
- Cross-run / cross-GPU の再現性測定。

## 5. Roadmap Rule

- Phase 7F の完了条件にこれらの演算子を追加しない。
- LayerNorm Backward は Backward の完全性のため High だが、Phase 7F の
  Verification Debt とは別管理とする。
- Phase 8 で最初に扱う候補は OP-01 / OP-06 / DX-01。
