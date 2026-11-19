# Phase 7F Performance Backlog

最終更新: 2026-11-19
状態: Phase 7F 実装凍結。性能改善は Phase 8 以降の候補であり、Phase 7F の
完了条件ではない。

## 1. 観測された傾向

| 項目 | 観測 | 制約 |
|---|---|---|
| 4 Stream speedup | 約 1.0〜1.16。小 graph では per-stream submit / event が支配 | 単一 T4、単一セッション |
| Arena | independent/shared より一貫して遅い | slot-set 排他と cross-stream wait |
| Async planned | host submit が sync より小さい | wait を別計測 |
| Constant upload | prefix copy は同期を維持 | lifetime safety |
| Forward small graph | 0.04–0.15 ms 程度 | 計測分解能・event overhead |
| Large MLP | 0.26–0.33 ms | T4、単一 run |

## 2. Backlog

| ID | Item | 現状 | 改善候補 | リスク | 優先度 |
|---|---|---|---|---|---|
| PB-01 | Per-stream submit overhead | 小 graph で speedup がほぼ 1.0 | submission batching / CUDA Graph | stream ordering / event lifetime | Medium |
| PB-02 | Arena slot-set serialization | arena が最も遅い | slot-set pool 拡張 / lease 最適化 | AccessGuard / safety | Medium |
| PB-03 | Constant upload synchronization | prefix copy が stream sync | pinned host + lifetime tracking / async copy | host slice lifetime | Low |
| PB-04 | 4 Stream overlap | overlap はあるが speedup 小 | graph size / stream affinity / kernel fusion | 数値一致・hazard | Low |
| PB-05 | Cross-run / cross-GPU stability | 単一 T4 のみ | 複数 GPU / 複数セッション | 環境差 | Medium |
| PB-06 | Benchmark noise | 100 samples だが session 依存 | 複数 run の集約、clock/temp 記録の拡充 | なし | Low |
| PB-07 | Allocator fragmentation | Pool reserved / peak は記録済み | block splitting / size classes | quarantine / safety | Low |

## 3. 計測上の注意

- Phase 7F の benchmark は **単一 T4 / 単一セッション**の median。
  cross-run / cross-GPU の性能主張には使わない。
- Sanitizer 実行中の性能値は使わない。
- GPU Event Time と Host Wall-clock を混同しない。
- 同期 API の host_submit は implicit wait を含む。async の wait と比較しない。

## 4. 優先順位

Phase 8 の機能開発を止めない。性能改善は次の順で候補:

1. PB-01 / PB-02（実行経路の overhead）。
2. PB-05（再現性）。
3. PB-03（constant upload）。
4. その他。
