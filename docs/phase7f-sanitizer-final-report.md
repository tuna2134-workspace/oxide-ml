# Phase 7F Sanitizer Final Report

> **Status note (2026-11-19):** current sanitizer classification is in
> `phase7f19-recovery-sanitizer-report.md` and `phase7f-verification-debt.md`.
> NOT RUN / FAIL items are not PASS.

## 実行環境

Tesla T4 / sm_75 / CUDA 13.0。Compute Sanitizer。

## 結果

| Target | memcheck | racecheck | synccheck |
|---|---|---|---|
| access_check | PASS 0 | PASS 0 hazards | PASS 0 |
| concurrency_check | PASS 0 | NOT RUN（2-stream 大規模） | PASS 0 |
| gpu_check（全体） | PASS 0 | PASS 0 hazards | PASS 0 |
| dynamic_check | PASS 0 | PASS 0 hazards | PASS 0 |
| race4_check（4 stream / shared RW） | PASS 0 | PASS 0 hazards | PASS 0 |
| benchmark_check | PASS 0 | NOT RUN | NOT RUN |

- race4_check は 4 Stream / Shared Read-Write / AccessGuard の Sanitizer 対象。
- gpu_check 全体の racecheck / synccheck は完走し、0 hazards / 0 errors。
- concurrency_check の racecheck は大規模 GEMM のため NOT RUN。
  4 Stream Hazard は race4_check で代替検証。
- benchmark_check の racecheck/synccheck は NOT RUN（memcheck のみ）。

## 判定

主要な AccessGuard / Dynamic Shape / 4 Stream / Arena / gpu_check 全体は
Sanitizer PASS。未実行項目は上記のとおり NOT RUN。