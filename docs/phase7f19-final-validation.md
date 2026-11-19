> **Status note (2026-11-19):** this file predates the Phase 7F-19 closure run.
> See `phase7f-final-closure.md` and the recovery reports for the current,
> measured verdicts. Items marked NOT RUN here may have been executed later.

# Phase 7F-19 Final Validation

## CPU

| Command | Result |
|---|---|
| `cargo test --workspace` | PASS 192 / 0 failed |
| `cargo test -p oxide-ml-execution-safety` | PASS 35 / 0 failed |
| `cargo test -p oxide-ml-event-safety` | PASS 22 / 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS |
| `cargo doc --workspace --no-deps` | PASS |

## CUDA

| Command | Result |
|---|---|
| `cargo oxide build --arch sm_75 -- --bin gpu_check --release` | PASS |
| `gpu_check` | PASS 37 / EXIT 0 |
| `cargo clippy --all-targets -- -D warnings` | PASS |
| `access_check` | PASS |
| `concurrency_check` | PASS actual overlap |
| `benchmark_check` | PASS |

## Sanitizer

| Target | Tool | Result |
|---|---|---|
| access_check | memcheck | PASS 0 |
| access_check | racecheck | PASS 0 hazards |
| access_check | synccheck | PASS 0 |
| concurrency_check | memcheck | PASS 0 |
| concurrency_check | synccheck | PASS 0 |
| benchmark_check | memcheck | PASS 0 |
| concurrency_check | racecheck | NOT RUN |
| gpu_check | racecheck / synccheck | NOT RUN |

## 未実行

- `cargo check --manifest-path crates/oxide-ml-cuda/Cargo.toml` は
  cuda-oxide standalone 構成のため単独では成立しない。`cargo oxide build` で代替。
- 4 Stream / 4 Execution Matrix
- Dynamic Bounded Capacity / Capacity Overflow
- Unbounded `Dim::Dynamic` 出力
- Cross-GPU / Device Lost
