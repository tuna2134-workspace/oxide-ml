# Phase 7F-19 Benchmark / Statistics Report

## 1. 判定

**PARTIAL.** GPU Event Timing と Host Submission の分離、Memory/Event/Arena
統計は実装・測定した。全 Graph / 全 Stream Matrix と 4 Stream Benchmark は
NOT RUN。

## 2. 環境

- Tesla T4 / sm_75 / CUDA 13.0
- Rust 1.99.1 / cuda-oxide 0.4.0
- Warmup 2 / Rounds 3 / Iterations 10 / Median

## 3. Benchmark 結果

`benchmark_check`（Small MLP, batch 2）:

| Mode | GPU ms | Host Submit ms |
|---|---:|---:|
| normal | 0.1294 | 0.1292 |
| planned | 0.1117 | 0.1115 |
| arena | 0.1368 | 0.1367 |
| async_planned | 0.1267 | 0.1169 |
| async_arena | 0.1570 | 0.1469 |
| dynamic_normal (batch 4) | 0.0393 | 0.0391 |
| dynamic_planned (batch 4) | 0.0385 | 0.0383 |

- Async の Host Submit は wait を含めず、submit 完了までを測定。
- GPU 時間は `CudaEvent::elapsed_ms`。
- Normal / Planned は同期 API のため Host 時間に待ちを含む。

## 4. Memory Statistics

論理:

- Logical Tensor Bytes
- Planned Logical Bytes
- Runtime Required Bytes（dynamic は未定義）

物理:

- Physical Reserved Bytes / Allocated Bytes
- Active Buffer Bytes
- Free Pool Bytes（reserved - active - pending - quarantine）
- Pending Release Bytes
- Quarantined Bytes

Arena:

- Capacity / Used / Slot Capacity / Slot Allocation / Slot Reuse / Arena Count

Event:

- Allocation / Reuse / Pending / Failed / Quarantined

Access:

- Read / Write / ReadWrite / Same-Stream Elided / Cross-Stream Wait /
  Read-Read Concurrent

`logical_bytes` は compiler plan の値、`physical_reserved_bytes` は allocator が
保持する DeviceBuffer 容量。両者は同一ではない。

最終測定:

```text
pool: allocations=45 reuses=563 released=608 reserved=1760 peak=1760
arena: slot_sets=10 slot_reuse=54 physical_reserved=1280
event: allocation=11 reuse=821 pending=0 quarantine=0
```

## 5. Sanitizer

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

## 6. 未達

- 4 Stream Benchmark
- Large MLP / Backward / Reduction / Activation の全 Graph Matrix
- Dynamic Bounded Capacity Benchmark
- concurrency racecheck

## 7. 再現手順

```bash
# T4
cd /content/oxide-ml/crates/oxide-ml-cuda
cargo oxide build --arch sm_75 -- --bin benchmark_check --release
./target/release/benchmark_check
```

ログ: `docs/gpu-p7f19-bench.log`, `docs/gpu-p7f19-gpu.log`
