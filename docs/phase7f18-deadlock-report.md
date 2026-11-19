# Phase 7F-18 Deadlock / Starvation — Report

## 検証済み

- Arena capacity 1: 2 番目の lease が Busy。lease drop 後に再 lease 可能。
- Arena capacity 2: 2 threads / 2 streams で並行 `run_arena`、数値一致。
- Arena capacity 4: 4 threads / 4 streams で並行 `run_arena`、数値一致。
- AccessGuard Lock Ordering: BufferId + generation の昇順。
- Multiple Reader: 4 Reader の Writer が 4 event を wait。
- Writer 後に Reader を clear。完了 Reader は prune。
- Execution Drop / Executor Drop は Host Synchronize なし。
- Pool は pending / quarantine を free list に戻さない。

## NOT RUN

- 循環 Event Wait を意図的に構築するテスト。
  API は循環依存を作らない設計であり、循環を作ると実際に deadlock するため
  安全に実行できない。
- Pool exhaustion 専用 test。
- Writer starvation の長時間 test。

## 診断

- Arena Busy は `CudaError::ArenaBusy`。
- Capacity overflow は `CudaError::Capacity`。
- Access State は BufferId 順で lock。
