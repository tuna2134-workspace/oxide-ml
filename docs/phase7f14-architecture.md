# Phase 7F-14 Arena Identity Architecture

## 1. 目的

Persistent Planned Arena を Graph / MemoryPlan / Device / Context の組み合わせで識別し、
異なる Graph や Plan が同じ Arena を誤って共有することを防ぐ。Hash は高速な索引に
限定し、実際の再利用は完全な Descriptor 比較で確定する。

## 2. Crate 境界

```text
oxide-ml-core
    ↑
oxide-ml-graph       CanonicalGraph / GraphHash
    ↑
oxide-ml-compiler    CanonicalPlan / PlanHash / ArenaKey / ArenaDescriptor
    ↑                Generic ArenaRegistry
oxide-ml-cuda        DeviceIdentity / ContextIdentity
                     CudaArenaRegistry / CudaPlannedArena
```

- Graph Hash を CUDA Crate に置かない。
- Compiler は CUDA 型を知らない。
- CPU Test は `oxide-ml-graph` / `oxide-ml-compiler` で完結する。
- CUDA Crate は Identity の生成と Arena の実生成だけを担当する。

## 3. Canonical Graph

`Graph::canonicalize()`:

1. `Graph::validate()` で構造・shape/dtype を検証。
2. Symbol を名前へ変換（SymbolId に依存しない）。
3. Input / Parameter を `CanonicalExternal` にし、完全 encoding の byte 順で sort。
4. 各 Node の fingerprint を計算:
   - Operator tag
   - Canonical attributes（定数はビット列）
   - 入力: external fingerprint または producer fingerprint + output index
   - 出力: shape / dtype / device
5. 同一 fingerprint の Node を hash-consing し、`occurrences` を数える。
6. 決定論的 topological sort（ready node の fingerprint 昇順 tie-break）で
   Canonical Node を並べる。
7. 元の ValueId -> `CanonicalValueRef`、NodeId -> canonical node index の map を作る。
8. Graph outputs を canonical ref 列として保持。

これにより元の NodeId/ValueId、Node の構築順、HashMap 反復順は Hash に入らない。
同一 fingerprint の重複 Node は意味的に同一な計算として統合される。

### Encoding

- 明示的 little-endian binary。`Debug` 文字列や JSON は使わない。
- 長さフィールドは u64。
- f32/f64 は `to_bits()`。NaN は payload を含むビット同一性、`0.0` と `-0.0` は別。
- Constant は DType / Shape / element count / 全ビット列を含む。
- Shape の `Dim::Symbol` は Symbol 名、`Dim::Static` は u64、`Dim::Dynamic` は tag。

### GraphHash

```text
BLAKE3("oxide-ml.graph.v1" || canonical_graph_encoding)
```

`CANONICAL_GRAPH_VERSION` と `Graph.ir_version` も encoding に含む。

## 4. Canonical Plan

`canonicalize_memory_plan(graph, plan)`:

1. `Graph::canonicalize()` で ValueId/NodeId の変換表を得る。
2. Plan を検証する。
3. Assignment の value を `CanonicalValueRef` に変換。
4. Lifetime の `LifetimePoint::Node(id)` を canonical node index に変換。
5. Assignment / Lifetime / unplanned を canonical ref 順に sort。
6. Slot は id 順のまま保持。

`CanonicalPlan` は slots / assignments / lifetimes / unplanned / total bytes / version を
含む。`PlanHash` は `BLAKE3("oxide-ml.memory-plan.v1" || encoding)`。

Raw ValueId を使う `memory_plan_hash(plan)` も提供するが、これは graph-local な
構造検証用であり、Arena には `canonical_memory_plan_hash(graph, plan)` を使う。

### Plan Validation

- Slot table の dense 性、重複 ID、alignment の power-of-two、capacity > 0、
  capacity の alignment 整合。
- Assignment の slot 存在、offset + size <= capacity、offset alignment、
  shape numel × dtype size と size_bytes の一致。
- Lifetime の重複、defined_at <= last_used_at、shape/dtype/size 整合。
- Unplanned の重複、assignment との排他。
- total_value_bytes と assignment bytes の一致。
- Graph-aware: graph value の dtype/shape 一致、assignment された value の lifetime 存在、
  全 node output の lifetime 存在。

不正 Plan は Hash 化せず `PlanHashError` を返す。

## 5. ArenaKey / ArenaDescriptor

```rust
pub struct ArenaKey {
    pub graph_hash: GraphHash,
    pub plan_hash: PlanHash,
    pub device_id: DeviceId,
    pub context_id: ArenaContextId,
    pub dtype: DType,
    pub memory_plan_version: u32,
}
```

`DeviceId` は ordinal / UUID / name / compute capability / total memory。
`ArenaContextId` はプロセス内 monotonic ID。

`ArenaDescriptor` は Key の元になった完全な構造を保持する:

- `Arc<CanonicalGraph>`
- `Arc<CanonicalPlan>`
- DeviceId / ArenaContextId / DType / MemoryPlan version
- `backend_layout: Vec<u8>`
  - CUDA の実 slot / assignment / raw ValueId binding を deterministic に符号化。
  - Persistent Arena は ValueId-keyed plan を保持するため、canonical-equal でも
    番号が違う Graph への誤再利用を防ぐ。

Arena 再利用条件:

```text
key が一致
AND descriptor が完全一致
AND backend_layout が一致
AND 既存 arena capacity >= 要求 capacity
```

## 6. Device / Context Identity

### Device

`CudaDeviceInfo` + `cuDeviceGetUuid_v2` から `DeviceId` を作る。
UUID が取得できない場合も ordinal/name/compute capability で識別する。
異なる GPU は ordinal または UUID が異なるため別 Arena。

### Context

`Arc::as_ptr` は使わない。raw `CUcontext` handle をキーにした process-local registry:

- 同一 raw `CUcontext` の複数参照は同一 `ArenaContextId`。
- ID は monotonic counter で、破棄後に raw handle が再利用されても新しい ID。
- `ContextIdentity` は `Arc<CudaContext>` を保持し、Context Lifetime を延ばす。
- `CudaArenaRegistry` は context token を保持し、Arena 存続中は ID の対応を維持。
- ID はプロセス内限定。永続化禁止。

## 7. Collision Handling

Registry は `HashMap<ArenaKey, Vec<RegistryEntry>>`。

- Key が同じでも Descriptor が違えば、既存 Arena を返さず新しい Entry を作る。
- Entry には `Arc<ArenaDescriptor>` を保持する。
- `ArenaRegistryStats::collision_count` に記録する。
- 本番コードに Collision を強制する設定は公開しない。テストは
  `get_or_create_explicit(key, descriptor, ...)` で同じ Key / 異なる Descriptor を
  注入する。Generic Registry の CPU Test でも同じ方法を使う。

## 8. Concurrency / Capacity

- 最初に Registry Mutex 下で lookup。
- Miss の場合、per-key `Arc<Mutex<()>>` を取得してから再 lookup。
- 同一 Key の同時 Lookup は per-key lock で直列化され、Arena は 1 回だけ生成される。
- 異なる Key は別 lock なので並列。
- CUDA Allocation (`CudaPlannedArena::new` 内の SlotSet allocation は lease 時) は
  Registry Mutex の外で実行する。
- `max_arenas` は Registry が保持する Arena 数。SlotSet capacity とは独立。
- 上限到達時は `ArenaRegistryError::CapacityExceeded`。既存 Arena は破棄しない。
- Mutex poisoning は回復し、`poison_recovery_count` を記録。
- Registry Drop は通常の Vec/Arc drop。panic しない。

## 9. Safety Invariants

1. Failed/不正 Plan は Hash 化しない。
2. Hash 一致だけで Arena を共有しない。Descriptor 完全一致が必要。
3. Graph の構築順・NodeId・ValueId は GraphHash に入らない。
4. Node attribute / edge / shape / dtype / constant が違えば GraphHash が変わる。
5. Plan の slot / assignment / capacity / alignment / lifetime / shape / dtype が
   違えば PlanHash が変わる。
6. Device / Context が違えば ArenaKey が変わる。
7. Persistent Arena の実行互換性は `backend_layout`（raw ValueId binding）で確認する。
8. Phase 7F-13 の Event Safety / Buffer Reclamation には変更を加えない。

## 10. Known Constraints

- Context ID はプロセス内限定。
- GraphHash の内部 Node fingerprint は BLAKE3 の衝突耐性を前提とする。
  Arena 再利用の最終判定は Descriptor 比較で行う。
- MemoryPlanner の greedy tie-break は定義順を使うため、意味的に同じ Graph でも
  independently generated Plan の slot assignment が異なり得る。その場合は
  別 Plan として別 Arena になる（安全側）。
- Persistent Arena は raw ValueId-keyed plan を保持するため、renumbered Graph への
  Arena 再利用は行わない。`source_graph_raw_hash` で拒否する。
- Slot ID のラベル正規化は行わない。Slot ID は実行 layout の一部として扱う。
- Arena Registry の eviction は未実装。上限到達は明示エラー。
