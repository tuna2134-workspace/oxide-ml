# Phase 7F-17 Bounded Dynamic Capacity — Report

## API

```rust
let (plan, signature) = executor.plan_bounded(&graph, &max_bindings)?;
executor.run_planned(&graph, params, inputs, &plan)?;
executor.run_arena(&graph, params, inputs, &arena, stream)?;
```

- max_bindings: SymbolId -> maximum runtime dimension。
- Graph を max shape で specialize し、MemoryPlanner で max capacity slot を作る。
- `CudaMemoryPlan::capacity_bindings()` が max を保持。
- `max_capacity_bytes()` が slot 合計。

## Runtime Required Bytes

`validate_runtime_capacity` が Kernel Launch 前に検証:

1. 全 bounded Symbol が bound されている。
2. actual <= maximum。
3. runtime elements × dtype size が usize overflow しない。
4. runtime bytes <= assignment capacity。
5. offset % alignment == 0。
6. offset + runtime bytes <= slot capacity。

違反は `CudaError::Capacity`。Slot を破壊しない。

## T4

`gpu_check`:

```text
batch = 1, 4, 8
bounded plan normal/planned: PASS
bounded plan Arena (same plan, capacity 2): PASS
batch = 9 (max 8): Capacity Error, no kernel launch
```

`dynamic_check` は bounded batch 1/4/8 を Normal / Planned / Arena で実行し、
sanitizer memcheck / racecheck / synccheck 0。

## 制約

- min dimension 制約は未実装。
- Capacity 超過時の自動再確保はしない（明示エラー）。
- Runtime Required Bytes の統計は API 値として未集計（検証には使用）。
