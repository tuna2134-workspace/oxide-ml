# Phase 7F-17 Unbounded Dynamic Output Pool Fallback — Report

## 実装

- `runtime_shapes(graph, input_shapes)` を追加。
- 入力の実 shape から `infer_outputs` で全 Node の出力 shape を具体化。
- `Dim::Dynamic` 出力も runtime で確定。
- MemoryPlan に assignment がない value は Pool から allocation。
- Static / Bounded は Arena、Unbounded は Pool に分離。
- allocation は runtime shape 確定後。Buffer は Execution 完了まで保持。
- AccessGuard / Event Safety / Buffer Generation は既存経路。

## T4

`gpu_check` unbounded graph:

```text
input [Dynamic, 2], w [2,1], b [1]
batch 3 / 5
normal: PASS
planned: PASS
specialize_graph: Capacity Error (expected)
```

`dynamic_check` でも同じ graph を Planned で実行し、sanitizer 3 種 0。

## 制約

- Mixed Static / Bounded / Unbounded の専用 Benchmark Graph は未実施。
- Unbounded は常に Pool。Arena には収容しない。
- Pool allocation failure は既存 Pool Error に従う（専用 fault test は CPU Mock）。
