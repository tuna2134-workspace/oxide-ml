# Phase 7F-17 Dynamic Shape Safety — Final Report

## 1. 最終判定

**PARTIAL.**

CUDA の `Dim::Symbol` を入力から runtime bind し、Normal / Planned / Shape
Specialization / Specialized Arena で batch 1 / 4 / 8 を実行できることを T4 で
確認した。一方、Bounded Dynamic Capacity の予約と `Dim::Dynamic`（unbounded）
出力の Pool Fallback は未実装である。

## 2. 監査

- `Shape` は `Dim::{Static, Symbol, Dynamic}` を持つ。
- CPU Executor は input から `SymbolBindings` を作り、`Dim::Symbol` を bind する。
- CUDA Executor は `static_shape()` で `Shape::static_dims()` を要求し、
  Symbol を含む shape を実行できなかった。
- MemoryPlanner は `numel_static()` が `None` の value を `unplanned_values` に
  入れる。したがって symbolic value は Pool Fallback で安全に実行できる。
- `CudaMemoryPlan::from_memory_plan` は dynamic value が slot に割り当てられた
  場合のみ拒否する。

## 3. 実装

### Runtime Symbol Binding

`bind_inputs(graph, inputs)`:

- input rank / static dim を検証。
- `Dim::Symbol` を実 input dim に bind。
- 同じ Symbol が矛盾する値に bind されたらエラー。
- `Dim::Dynamic` は input では受理するが、Symbol には bind しない。

`resolve_shape(shape, bindings)`:

- Static → 値
- Symbol → binding
- Dynamic → 明示エラー（安全性優先）

### Shape Signature

```rust
pub struct CudaShapeSignature { bindings: Vec<(u32, usize)> }
```

- SymbolId と runtime dim の sorted vector。
- Canonical GraphHash / PlanHash とは分離。
- `CudaExecutor::shape_signature(graph, inputs)` で取得。

### Shape Specialization

`CudaExecutor::specialize_graph(graph, inputs)`:

- graph を clone。
- 全 value の `Dim::Symbol` を `Dim::Static(runtime)` に置換。
- 全 shape が static になった specialized graph を返す。
- `specialized.validate()` を通す。
- これにより static 用の MemoryPlan / Persistent Arena を shape ごとに再利用できる。

### Fallback

Symbolic value は planner で unplanned になるため、Normal / Planned では
Pool Fallback で実行される。Specialized graph では static plan / Arena を使える。

## 4. T4 検証

`gpu_check` の `dynamic symbol batch`:

```text
batch = 1, 4, 8
normal: PASS
planned: PASS
specialize_graph + static MemoryPlan + Arena (2 runs each): PASS
shape signature: PASS
```

全体:

```text
37 PASS / EXIT 0
```

## 5. CPU 検証

Workspace 192 tests PASS。既存 CPU の `executes_symbolic_batch` を含む
symbol binding / shape mismatch テストを維持。

## 6. 未達

1. Bounded Dynamic Shape の Capacity 予約
   （max batch を指定した Arena capacity / runtime required bytes の分離）
2. Unbounded `Dim::Dynamic` 出力の Pool Fallback
3. Capacity Overflow の GPU Test
4. Shape Signature ごとの Arena Registry 統合（現状は caller が specialized
   graph ごとに Arena を生成）

## 7. 既知の制約

- `Dim::Dynamic` は input では受理するが、出力 shape に残る graph は
  `resolve_shape` でエラーになる。
- Specialized Arena は shape ごとに別 plan / arena を作る。Cache Identity は
  `CudaShapeSignature` で表現できるが、Registry へは未統合。
- Static Shape の既存性能・動作は維持。
