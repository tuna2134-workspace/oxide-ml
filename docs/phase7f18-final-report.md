# Phase 7F-18 True GPU Concurrent Execution — Final Report

## 1. 最終判定

**PASS（2 Stream / 2 Thread の実 GPU Overlap）。**  
4 Stream / 4 Execution の全 Matrix と concurrency racecheck は NOT RUN。

## 2. Async API 監査

- `execute_async_planned` / `execute_async_arena` は内部で
  `stream.synchronize()` を呼ばない。
- `run` / `run_planned` / `run_arena` は同期 API として `synchronize` する。
- `CudaExecution::wait` / `collect` は明示的な完了待ち。
- Execution Drop は Host Synchronization を行わない（Phase 7F-15）。
- Arena lease は Event Wait で slot を回収する。
- AccessGuard は Kernel 単位で Hazard を解決する。

## 3. Actual GPU Overlap

`concurrency_check`:

- `N=1024` GEMM × 4 回 / Stream
- 2 Streams / 独立 Buffer
- CUDA Event Timestamp で区間を測定
- Serial: Stream B が Stream A の終了 Event を wait
- Parallel: 相互 wait なし

結果（最終実行）:

```text
serial_span_ms=103.001
parallel_span_ms=90.894
serial_submit_ms=0.077
parallel_submit_ms=0.072
a_ms=81.834
b_ms=90.891
overlap_ms=81.831
speedup=1.133
PASS actual GPU overlap
```

GPU Timeline 上で 81.8 ms の重なりを確認した。Host Submission の短さだけを
根拠にしていない。

## 4. Multi-Execution / Multi-Thread

`gpu_check`:

- 2 threads / 2 streams / 同一 Persistent Arena capacity 2
- 各 thread が独立 `run_arena`
- 両方の出力を D2H して数値検証
- `concurrent arena execution: PASS`

`gpu_check` には Async Execution / Executor Drop / Execution Drop / External
Output Ownership の既存検証も含まれる。

## 5. Deadlock / Starvation

- Access State は BufferId 順に lock（lock ordering）。
- Event Wait は先行アクセスの完了 Event のみ。
- Reader は writer 後に clear。完了 Reader は prune。
- Arena capacity 1 の busy、capacity 2 の並行 lease を検証。
- Mutex poisoning は `lock_recover`。

循環 Event Wait の専用検出器は追加していない。

## 6. 未達 / NOT RUN

- 4 Streams / 4 Executions の Matrix
- concurrency_check の racecheck（実行時間超過のため）
- Cross-GPU / Device Lost

## 7. 既知の制約

- T4 は 1 GPU。Cross-GPU は検証不能。
- GEMM は resource 依存で overlap 率が変動する。
- `run` 系同期 API は意図的に同期する。Async API のみ非同期。
