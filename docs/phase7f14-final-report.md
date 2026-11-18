# Phase 7F-14 Arena Identity / Canonical Graph Hash / MemoryPlan Hash — Final Report

## 1. 最終判定

**PASS.**

Arena Identity の実装、CPU Unit Test、CUDA T4 Integration、Phase 7F-13 Event Safety
回帰、Clippy、Compute Sanitizer まで完了した。Phase 7F-15 以降には進んでいない。

| Area | Judgement |
|---|---|
| Canonical Graph / GraphHash | PASS |
| Canonical Plan / PlanHash / Validation | PASS |
| ArenaKey / ArenaDescriptor | PASS |
| Device / Context Identity | PASS |
| Collision Handling | PASS |
| Arena Registry / Concurrent Lookup | PASS |
| T4 Integration | PASS |
| Phase 7F-13 Regression | PASS |
| CPU Tests | PASS 157 / 0 failed |
| Clippy | PASS |
| Compute Sanitizer | PASS |

## 2. 既存コード監査結果

詳細: [phase7f14-audit.md](phase7f14-audit.md)

- Graph はトポロジカル順の SSA。`NodeId = nodes.len()`、`ValueId = values.len()` で
  **構築順に依存**する。
- Edge は `Node.inputs: Vec<ValueId>` / `Node.outputs: Vec<ValueId>`。
- `OpAttrs` は enum。`Shape` は `SymbolId` を含み、f32 属性（LayerNorm eps,
  ConstantScalar）はビット正規化が必要。
- Constant は `ConstantData::{F32, F64}`。値の取得は可能だが、NaN の equality は
  bitwise で扱う必要がある。
- `MemoryPlan` は `slots: Vec<BufferSlot>`、`assignments: HashMap<ValueId, _>`、
  `lifetimes: Vec<ValueLifetime>`。既存 validator はない。
- `MemoryPlanner` の greedy tie-break は定義順を使うため、同一意味でも構築順が
  違うと slot assignment が異なり得る。
- 旧 `ArenaKey` は `DefaultHasher`、`Debug` 文字列、生 Node index、`Arc::as_ptr`
  Context ID を使用し、Constant 値を含まず、Hash 衝突時はエラーにしていた。
- Device は ordinal/name/compute capability、Context は raw `CUcontext` を取得可能。
  UUID は `cuDeviceGetUuid_v2` で取得できる。

## 3. Canonical Graph の仕様

実装: `crates/oxide-ml-graph/src/canonical.rs`

```rust
pub struct CanonicalGraph {
    pub version: u32,
    pub ir_version: u32,
    pub symbols: Vec<String>,
    pub inputs: Vec<CanonicalExternal>,
    pub parameters: Vec<CanonicalExternal>,
    pub nodes: Vec<CanonicalNode>,
    pub outputs: Vec<CanonicalValueRef>,
    // value_map / node_map は lookup 用で equality/hash 対象外
}
```

正規化:

1. `Graph::validate()`。
2. SymbolId を Symbol 名へ変換。
3. Input / Parameter を完全 encoding の byte 順に sort。
4. Node fingerprint を次から計算:
   - Operator tag
   - Canonical attributes（ビット列）
   - 入力 external fingerprint または producer fingerprint + output index
   - 出力 shape / dtype / device
5. 同一 fingerprint の Node を hash-consing し、`occurrences` を保持。
6. 決定論的 topological sort（ready node の fingerprint 昇順 tie-break）で
   Canonical Node を並べる。
7. 元 ValueId / NodeId の lookup map を作る。
8. Graph outputs を canonical ref の順序で保持。

これにより NodeId / ValueId / 構築順 / HashMap 反復順は GraphHash に入らない。
独立 Node の構築順を変えても同じ CanonicalGraph になることを CPU Test で確認した。

## 4. Graph Hash の仕様

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphHash(pub [u8; 32]);

pub fn graph_structure_hash(graph: &Graph) -> Result<GraphHash, GraphHashError>;
```

- Hash: **BLAKE3**。
- Domain: `oxide-ml.graph.v1`。
- `CANONICAL_GRAPH_VERSION = 1` と `Graph.ir_version` を encoding に含む。
- 明示的 little-endian binary encoding。`Debug` 文字列 / JSON / HashMap 順は不使用。
- Constant は DType / Shape / element count / 全ビット列。
- f32/f64 は `to_bits()`。NaN は payload を含むビット同一性、`0.0` と `-0.0` は別。
- `Graph::structural_hash() -> u64` は後方互換のため canonical GraphHash の先頭 8 byte を
  返すよう更新した。

CPU Test（`oxide-ml-graph` 24 tests 中、canonical_hash 19）:

- Same Graph Same Hash
- Different Node Attribute / Edge / Operator / Shape / DType / Constant / Output
- Output Order
- Different Node ID / Construction Order
- Parameter Insertion Order / Symbol Creation Order
- JSON Roundtrip
- NaN payload / -0.0
- Canonicalization Version
- Raw execution identity（NodeId/ValueId を追跡）

## 5. MemoryPlan Hash の仕様

実装: `crates/oxide-ml-compiler/src/plan_hash.rs`

```rust
pub struct PlanHash(pub [u8; 32]);

pub fn memory_plan_hash(plan: &MemoryPlan) -> Result<PlanHash, PlanHashError>;
pub fn canonical_memory_plan_hash(
    graph: &Graph,
    plan: &MemoryPlan,
) -> Result<PlanHash, PlanHashError>;
pub fn canonicalize_memory_plan(
    graph: &Graph,
    plan: &MemoryPlan,
) -> Result<CanonicalPlan, PlanHashError>;
```

- Domain: canonical は `oxide-ml.memory-plan.v1`、raw は
  `oxide-ml.memory-plan.raw.v1`。
- `MEMORY_PLAN_VERSION = 1` を encoding に含む。
- CanonicalPlan は slots（id, size, alignment）、assignments（canonical value,
  slot, offset, size, dtype, canonical shape）、lifetimes（canonical value,
  canonical lifetime point, size, dtype, canonical shape）、unplanned、
  total_value_bytes を含む。
- Assignment / lifetime / unplanned は canonical value ref 順に sort。
- LifetimePoint::Node は canonical node index へ変換。
- Validation:
  - Slot dense / 重複 / out-of-range
  - alignment power-of-two / capacity alignment
  - slot capacity 不足 / offset alignment
  - static shape × dtype size と assignment size の一致
  - lifetime defined <= last_used
  - lifetime の重複 / size/shape/dtype 整合
  - unplanned の重複 / assignment との排他
  - total_value_bytes と assignment bytes
  - graph value の dtype/shape 一致
  - 全 node output の lifetime 存在
- 不正 Plan は Hash 化せず `PlanHashError` を返す。

CPU Test（`oxide-ml-compiler` 30 tests 中、plan_hash 15）:

- Same Plan Same Hash
- Different Slot Assignment / Capacity / Alignment / Lifetime / Shape / DType
- Dynamic Shape / Unplanned
- Invalid Plan（duplicate slot, bad alignment, bad lifetime, capacity overrun,
  unknown value, dtype mismatch, out-of-range slot）
- Plan Version
- Graph numbering 非依存（明示的に同一構造の Plan を使用）
- Raw plan hash の graph-local validation

## 6. ArenaKey の仕様

実装: `crates/oxide-ml-compiler/src/arena_key.rs`

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

- `DeviceId`: ordinal / UUID / name / compute capability / total memory。
- `ArenaContextId`: プロセス内 monotonic ID。
- `ArenaDescriptor`:
  - `Arc<CanonicalGraph>`
  - `Arc<CanonicalPlan>`
  - DeviceId / ArenaContextId / DType / MemoryPlan version
  - `backend_layout: Vec<u8>`（CUDA の実 slot / raw ValueId binding）
- `ArenaKey` は高速索引。再利用は Key + Descriptor 完全一致で確定する。

## 7. Device / Context Identity

実装: `crates/oxide-ml-cuda/src/arena_key.rs`

### Device

- `CudaDeviceInfo` + `cuDeviceGetUuid_v2`。
- UUID が取れない場合も ordinal/name/compute capability で識別。
- T4 実機で UUID 取得成功:
  `[82, 122, 192, 104, 8, 71, 16, 124, 161, 23, 231, 141, 150, 218, 203, 184]`。
- 異なる GPU は ordinal または UUID が異なるため別 Arena。

### Context

- `Arc::as_ptr` は使用しない。
- raw `CUcontext` handle をキーにした process-local registry。
- 同一 raw Context の複数参照は同一 ID。
- ID は monotonic counter。raw handle が再利用されても ID は再利用しない。
- `ContextIdentity` は `Arc<CudaContext>` を保持し、Context Lifetime を延ばす。
- `CudaArenaRegistry` は context token を保持し、Arena 存続中は ID 対応を維持。
- ID はプロセス内限定。永続化しない。

T4 で `DeviceIdentity` / `ContextIdentity` の再取得が同一 ID を返すことを確認。

## 8. Collision Handling

- Registry は `HashMap<ArenaKey, Vec<RegistryEntry>>`。
- Key が一致しても Descriptor が完全一致しなければ既存 Arena を返さない。
- 異なる Graph Descriptor / Plan Descriptor / Device / Context / DType / Version は
  別 Entry。
- `ArenaRegistryStats::collision_count` に記録。
- 本番に Collision を強制する設定は公開しない。テストは
  `get_or_create_explicit` で同一 Key / 異なる Descriptor を注入する。
- CUDA の `backend_layout` により、canonical-equal でも raw ValueId が異なる
  Graph への誤再利用を防止。
- さらに `CudaMemoryPlan::source_graph_raw_hash` と `Graph::raw_identity_hash` を
  照合し、Plan が別 Graph 番号から作られた場合は `InvalidPlan` で拒否。

T4 collision test:

```text
arena_count=4
lookup_count=5
hit_count=1
miss_count=4
collision_count=1
```

Hash 一致（同一 Key）でも Descriptor 不一致で別 Arena になったことを確認。

## 9. Arena Registry

実装:

- Generic: `crates/oxide-ml-compiler/src/arena_registry.rs`
- CUDA wrapper: `crates/oxide-ml-cuda/src/arena_registry.rs`

```rust
pub struct ArenaRegistry<A> { ... }
pub struct CudaArenaRegistry { ... }
```

- `CudaArenaRegistry::get_or_create(graph, cuda_plan, pool, device, context, capacity)`。
- Registry 最大 Arena 数 `max_arenas` と SlotSet capacity は別。
  - `max_arenas` は Arena Entry 数。
  - SlotSet capacity は `CudaPlannedArena` の slot set 数。
- 上限到達は `CapacityExceeded` / `CudaError::ArenaBusy`。既存 Arena は破棄しない。
- Creation Failure は Registry を変更せず `Factory` エラー。
- Mutex poisoning は回復し `poison_recovery_count` を記録。
- Registry Drop は通常の Vec/Arc drop。CPU Test で panic しないことを確認。

## 10. Concurrent Lookup

- 最初に Registry Mutex 下で lookup。
- Miss 時は per-key `Arc<Mutex<()>>` を取得して再 lookup。
- 同一 Key の同時 Lookup は per-key lock で直列化し、Arena を 1 回だけ生成。
- 異なる Key は別 lock で並列。
- CUDA Arena 生成は Registry Mutex の外で実行。
- 同じ Key の 8 thread 同時 Lookup で factory 呼び出し 1 回、Registry 1 entry を
  CPU Test で確認。
- 異なる Key の 8 thread 同時 Lookup で 8 Arena を確認。
- Creation Failure / Capacity Limit / Mutex Poison / Registry Drop を Test。

## 11. 変更ファイル一覧

新規:

- `crates/oxide-ml-graph/src/canonical.rs`
- `crates/oxide-ml-graph/tests/canonical_hash.rs`
- `crates/oxide-ml-compiler/src/plan_hash.rs`
- `crates/oxide-ml-compiler/src/arena_key.rs`
- `crates/oxide-ml-compiler/src/arena_registry.rs`
- `crates/oxide-ml-compiler/tests/plan_hash.rs`
- `crates/oxide-ml-compiler/tests/arena_registry.rs`
- `crates/oxide-ml-cuda/src/arena_key.rs`
- `crates/oxide-ml-cuda/src/arena_registry.rs`
- `docs/phase7f14-audit.md`
- `docs/phase7f14-architecture.md`
- `docs/phase7f14-final-report.md`

変更:

- `Cargo.toml`（workspace `blake3`）
- `Cargo.lock`
- `crates/oxide-ml-cuda/Cargo.lock`
- `crates/oxide-ml-graph/Cargo.toml`
- `crates/oxide-ml-graph/src/lib.rs`
- `crates/oxide-ml-graph/src/ir.rs`（`Op: Eq`、`structural_hash` を BLAKE3 へ）
- `crates/oxide-ml-compiler/Cargo.toml`
- `crates/oxide-ml-compiler/src/lib.rs`
- `crates/oxide-ml-cuda/src/lib.rs`
- `crates/oxide-ml-cuda/src/arena.rs`（旧 ArenaKey/Registry を分離）
- `crates/oxide-ml-cuda/src/memplan.rs`（canonical plan / plan hash / raw graph hash /
  backend layout）
- `crates/oxide-ml-cuda/src/bin/gpu_check.rs`（Arena Identity T4 checks）
- `README.md`

検証ログ:

- `docs/gpu-p7f14-run.log`
- `docs/gpu-p7f14-bench.log`
- `docs/gpu-p7f14-clippy.log`
- `docs/gpu-p7f14-san-memcheck.log`
- `docs/gpu-p7f14-san-racecheck.log`
- `docs/gpu-p7f14-san-synccheck.log`
- `docs/gpu-p7f14-runner.log`

## 12. CPU Unit Test

```text
cargo test --workspace
passed=157 failed=0
```

Package 別:

| Package | Tests |
|---|---:|
| oxide-ml-graph | 24 |
| oxide-ml-compiler | 30 |
| oxide-ml-event-safety | 22 |
| その他 CPU workspace | 81 |
| **合計** | **157** |

既存 Test の削除は行っていない。

## 13. Clippy / Docs

```text
cargo clippy --workspace --all-targets -- -D warnings
PASS

# CUDA crate on T4
cargo clippy --all-targets -- -D warnings
PASS (CLIPPY_EXIT:0)

cargo doc --workspace --no-deps
PASS
```

CUDA 側の `too_many_arguments` / `type_complexity` は、CUDA ABI を反映する
kernel launch wrapper と複数リストを返す内部 helper に限定して allow している。

## 14. T4 Integration Test

`gpu_check`（Tesla T4 / sm_75 / CUDA 13.0 / Rust 1.99.1 / cuda-oxide 0.4.0）:

```text
23 PASS
EXIT:0
```

| Check | Result |
|---|---|
| Same Graph / Plan -> Same Arena | PASS |
| Different Graph -> Separate Arena | PASS |
| Different Plan -> Separate Arena | PASS |
| Plan from different graph numbering rejected | PASS |
| Collision: same key / different descriptor -> separate arena | PASS |
| DeviceIdentity stable / UUID | PASS |
| ContextIdentity stable | PASS |
| Persistent Arena 100 runs / 2 slot allocations | PASS |
| SlotSet capacity / busy | PASS |
| EventPool normal path | PASS |
| Normal / Planned / Arena numerical equality | PASS |
| Async / Deferred Reclamation | PASS |
| MLP backward / Linear regression / XOR | PASS |

Arena registry stats:

```text
arena_count=3
lookup_count=4
hit_count=1
miss_count=3
collision_count=0
```

Collision test:

```text
arena_count=4
lookup_count=5
hit_count=1
miss_count=4
collision_count=1
```

EventPool stats（正常系）:

```text
allocation_count=2
reuse_count=842
pending_count=0
quarantine_count=0
```

## 15. Phase 7F-13 Regression

- CPU: `oxide-ml-event-safety` 22 tests PASS。
- CPU: Buffer Reclamation / Event State / Quarantine / Drop Safety tests PASS。
- T4: `gpu_check` の Event State Machine / EventPool / Pending Retention /
  EventPool Drop / Context Lifetime / Buffer Context Mismatch / Quarantine Stats /
  Deferred Reclamation がすべて PASS。
- Phase 7F-13 の Event Safety 実装には変更を加えていない。

```text
event pool stats: EventPoolStats {
  allocation_count: 2,
  reuse_count: 842,
  pending_count: 0,
  quarantine_count: 0,
  ...
}
```

## 16. Benchmark / Sanitizer

T4 release `memplan_bench`:

| Mode | Median | P95 |
|---|---:|---:|
| Normal + Pool | 141.4 us | 172.8 us |
| Planned + Pool | 110.2 us | 128.3 us |
| Persistent Arena | 123.3 us | 141.5 us |
| Event acquire+record+release | 3.40 us | 4.85 us |

Phase 7F-13 以前の範囲（Normal 138.9 us / Planned 125.0 us / Arena 133.1 us）と
比較して、Arena Identity による有意な性能退行はない。

Compute Sanitizer:

| Tool | Target | Result |
|---|---|---|
| memcheck | `gpu_check` | PASS 0 errors |
| racecheck | `memplan_bench` | PASS 0 hazards |
| synccheck | `memplan_bench` | PASS 0 errors |

## 17. 既知の制約・未達項目

### 既知の制約

- `ArenaContextId` はプロセス内限定。永続化・プロセス間共有はしない。
- `GraphHash` の内部 Node fingerprint は BLAKE3 の衝突耐性を前提とする。
  Arena 再利用の最終判定は Descriptor 完全一致。
- Persistent Arena は raw ValueId-keyed plan を保持するため、renumbered Graph への
  Arena 再利用は行わない。`source_graph_raw_hash` で明示的に拒否する。
- `MemoryPlanner` の greedy tie-break は定義順を使う。意味的に同じ Graph でも
  独立生成した Plan の slot assignment が異なる場合は、別 Plan / 別 Arena になる
  （安全側）。
- Slot ID のラベル正規化は行わない。Slot ID は実行 layout の一部。
- Registry の eviction は未実装。上限到達は明示エラー。**NOT RUN（対象外）**。
- 別物理 GPU 間の Context Identity Test は T4 1 台のため **NOT RUN**。
  DeviceId / ContextId の差し替えによる Key 分離は CPU Test で PASS。
- Hash Provider trait は導入せず、`get_or_create_explicit` による
  Key/Descriptor 注入を Collision Test Hook とした。本番に Collision 強制設定は
  公開していない。**PASS（CPU/T4 Test）**。
- CanonicalGraph は衝突判定用に Constant のビット列を保持する。Hash 計算自体は
  canonical slice を streaming するが、canonicalization 時に Constant サイズ分の
  メモリを 1 回確保する。巨大 Constant を大量に含む Graph のための Constant cache は
  未実装。**SOURCE ONLY（既知の制約）**。

### 未達項目

なし。Phase 7F-14 の完了条件は満たしている。

## Acceptance Criteria

| 項目 | 状態 |
|---|---|
| Canonical Graph | **PASS** |
| Graph Structure Hash | **PASS** |
| Hash Determinism（構築順非依存） | **PASS** |
| Operator Attributes | **PASS** |
| Constant Tensor（bitwise） | **PASS** |
| MemoryPlan Hash | **PASS** |
| MemoryPlan Validation | **PASS** |
| ArenaKey | **PASS** |
| Device Identity | **PASS** |
| Context Identity | **PASS**（process-local、T4） |
| Collision Handling | **PASS** |
| Arena Registry | **PASS** |
| Concurrent Lookup | **PASS** |
| Existing Arena Reuse | **PASS** |
| Event Safety Regression | **PASS** |
| CPU Tests | **PASS** 157 / 0 failed |
| Clippy | **PASS** |
| T4 Integration | **PASS** 23 / EXIT 0 |
