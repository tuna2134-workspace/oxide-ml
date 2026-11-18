# Phase 7F-14-A 既存コード監査

監査日: 2026-11-18  
対象: Graph IR / MemoryPlan / CudaPlannedArena / CudaArenaRegistry / Device / Context

## 1. Graph IR

- `crates/oxide-ml-graph/src/ir.rs`
  - `Graph { inputs: Vec<ValueId>, outputs: Vec<ValueId>, nodes: Vec<Node>, values: Vec<ValueInfo>, parameters: Vec<ParameterInfo>, symbols: Vec<SymbolInfo>, .. }`
  - `Node { id: NodeId, op: Op, inputs: Vec<ValueId>, outputs: Vec<ValueId>, attrs: OpAttrs }`
  - SSA。Node はトポロジカル順に `Vec` へ格納される。
  - Edge は `Node.inputs` / `Node.outputs` の `ValueId` で表現される。
- Node ID は `NodeId(nodes.len())`、Value ID は `ValueId(values.len())`。
  - **構築順序に依存する**。同じ意味の Graph を独立 Node の順序を変えて構築すると、NodeId / ValueId は変わる。
- `Graph::validate` は Value/Node/Symbol/Parameter の dense 性、use-before-def、shape/dtype 推論一致、duplicate name を検証する。
- `GraphVerifier` は `Graph::validate` に cycle 検出を追加する。
- `Graph::structural_hash` は既存では pretty JSON に対する FNV-1a で、`ValueId`/`NodeId` と HashMap 順序の影響を受ける。Phase 7F-14 で canonical BLAKE3 へ置換した。

## 2. Operator Attributes

- `OpAttrs` は enum。
  - `MatMul { transpose_a, transpose_b }`
  - `Conv2d { stride, padding, dilation, groups }`
  - `Pool2d { kernel, stride, padding, dilation }`
  - `Reduce { axes, keep_dims }`
  - `Softmax { axis }`
  - `LayerNorm { normalized_shape, eps: f32 }`
  - `Reshape { shape: Shape }`
  - `Transpose { axes }`
  - `BroadcastTo { shape: Shape }`
  - `Loss { reduction }`
  - `ActivationBackward { kind }`
  - `Unbroadcast { shape: Shape }`
  - `Constant { value: ConstantTensor }`
  - `ConstantScalar { value: f32 }`
- `Shape` は `Dim::{Static(usize), Symbol(SymbolId), Dynamic}` を持ち、`SymbolId` は Graph 内の構築順に依存する。
- `eps: f32` と `ConstantScalar { value: f32 }` はビット単位の正規化が必要。
- Attribute を `Debug` 文字列化して Hash するのは不安定。Phase 7F-14 では明示的バイナリ encoding を使用する。

## 3. Constant Tensor

- `ConstantData::{F32(Vec<f32>), F64(Vec<f64>)}`
- `ConstantTensor { shape: Shape, data: ConstantData }`
- `OpAttrs::Constant` に保持される。`ConstantScalar` は f32 1 個。
- 値の取得は安定して可能。ただし f32/f64 の `PartialEq` は NaN を等価にしないため、Hash/Descriptor は `to_bits()` のビット列で扱う必要がある。
- 形状は `ConstantTensor::validate` により fully static が保証される。

## 4. MemoryPlan

- `crates/oxide-ml-core/src/memory.rs`
  - `BufferId(u32)`
  - `BufferSlot { id, size_bytes, alignment }`
  - `BufferAssignment { slot, offset, size_bytes, dtype, shape }`
  - `ValueLifetime { value, defined_at: LifetimePoint, last_used_at: LifetimePoint, size_bytes, dtype, shape }`
  - `LifetimePoint::{Node(u32), End}`
  - `MemoryPlan { slots: Vec<BufferSlot>, assignments: HashMap<ValueId, BufferAssignment>, lifetimes: Vec<ValueLifetime>, unplanned_values: Vec<ValueId>, total_value_bytes }`
- `MemoryPlan` 自体に validator は存在しない。`MemoryPlanner` が生成する。
- `assignments` は HashMap で順序を持たない。`lifetimes` は ValueId 順に sort されるが、ValueId は構築順依存。
- `BufferId` は slot table の index と一致する dense 表現。
- `LifetimePoint::Node(u32)` は NodeId に依存する。
- `MemoryPlanner` の greedy allocation は `(bytes desc, defined node id, ValueId)` で tie-break するため、**同一意味の Graph でも構築順が異なると plan の slot assignment が異なり得る**。これは plan の構造差として扱う（Graph Hash が同じでも Plan Hash は異なる）。

## 5. CudaPlannedArena / Arena 生成

- `crates/oxide-ml-cuda/src/arena.rs`
  - `CudaPlannedArena::new(plan: CudaMemoryPlan, pool: CudaBufferPool, capacity: usize)`
  - `SlotSet { slots, events, streams, state }`
  - `SlotSetState::{Free, Leased, PendingCompletion, Reclaimable}`
  - `SlotSetLease` が slot set を貸し出す。
  - `capacity` は SlotSet 数。Registry の Arena 数上限とは別概念。
- `CudaMemoryPlan` は `MemoryPlan` を CUDA 用に検証・変換し、`ValueId -> slot` の HashMap を保持する。
  - persistent arena は `CudaMemoryPlan` の ValueId をそのまま使うため、異なる ValueId 番号の Graph に同じ Arena を渡すと assignment が一致しない危険がある。

## 6. 旧 ArenaKey / Registry

Phase 7F-14 以前の `arena.rs` 末尾:

```rust
pub struct ArenaKey {
    pub graph_hash: u64,
    pub plan_hash: u64,
    pub device_id: i32,
    pub context_id: usize,
    pub dtype: u8,
    pub plan_version: u32,
}
```

問題:

- `DefaultHasher` を使用。永続 Identity としては不安定。
- `format!("{:?}", node.attrs)` を Hash。Debug 表現依存。
- Node index / ValueId をそのまま Hash。構築順依存。
- Constant 値を含まない。
- Context は `Arc::as_ptr` のアドレス。別 Arc の同一 Context を別 ID にし、アドレス再利用の ABA も防げない。
- `HashMap<ArenaKey, Arc<CudaPlannedArena>>` で 1 Key 1 Arena。Hash 衝突時は `ArenaBusy` エラーにし、別 Arena を作らない。
- Registry 全体の Mutex を保持したまま `CudaPlannedArena::new` を呼ぶ可能性がある。
- Mutex poisoning は `unwrap_or_else` で回復するが、統計がない。

## 7. Device / Context Identity

- `CudaDeviceInfo { ordinal, name, compute_capability, total_memory, .. }`
- `CudaContext::cu_device()` / `cu_ctx()` / `ordinal()` が利用可能。
- `cuDeviceGetUuid_v2` で UUID を取得できる。
- Context は `Arc<CudaContext>`。旧コードは Arc アドレスを Context ID としていた。
- 同一 primary context への複数 `CudaContext::new(0)` は raw `CUcontext` が同じで、Arc は別。Arc アドレスは Context 実体の識別に不適切。

## 8. 監査結論

Phase 7F-14 では次が必要:

1. NodeId/ValueId に依存しない Canonical Graph と BLAKE3 GraphHash。
2. Canonical Plan / PlanHash（ValueId/NodeId を canonical ref へ変換）。
3. 明示的 validator（slot/assignment/lifetime/dtype/shape）。
4. Raw `CUcontext` をキーにした世代付き Context Identity（Arc アドレス禁止）。
5. UUID + ordinal + arch を含む Device Identity。
6. ArenaKey + 完全 Descriptor 比較。Hash 衝突時は別 Arena。
7. 純 Rust の Generic ArenaRegistry（CPU Test 可能、per-key creation lock、capacity limit、poison recovery）。
8. CUDA 側は identity 生成と Arena 生成だけを担当し、実行時 plan の raw ValueId 一致（backend layout）も確認する。
