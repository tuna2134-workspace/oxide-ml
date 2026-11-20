# Module system, macro and Gemma 4 redesign: final engineering report

This report closes the cohesive redesign requested for oxide-ml. It is written
against the code in this workspace, not against a proposal: every capability
below is implemented, covered by tests, and (where hardware allowed) verified.
Numerical claims name the log or test that produced them.

## 1. Outcome

The framework now supports declarative model definitions with:

- `#[derive(Module)]` on **structs and enums**, including generics, nested
  modules, optional modules, enum architectural variants and `ModuleList<M>`;
- `#[model]` forward bodies with nested module calls, residuals, local
  variables, tuple outputs and **loops over module collections** (statically
  unrolled while building the graph);
- first-class **tied parameters** (`TiedLinear`) and storage-based aliasing, so
  a tied embedding/LM head is one canonical parameter, not two copies;
- a reusable **execution-state abstraction** (`KvCache`, `KvLayerConfig`,
  `ModelState`, `Module::state_config`) with per-layer shapes, shared-KV links,
  capacity growth, truncation, reset and per-session ownership;
- Gemma 4 rebuilt on those APIs: `ModuleList<GemmaLayer>` decoder stack,
  `TiedLinear<Embedding>` head, framework KV cache for streaming prefill/decode;
- CUDA `LogSoftmax`, `Split` and fused `MoE` kernels implemented this turn and
  A100-verified (the only remaining unsupported IR ops are `SliceBackward` and
  `EmbeddingBackward`).

All CPU tests pass: **394 cargo test cases, 0 failures**, plus the 7-fixture
`trybuild` compile-fail suite. The tiny Gemma parity numbers are unchanged:
dense `2.999e-7`, streaming `3.427e-7`.

## 2. Architecture

The workspace keeps the crate boundaries from the earlier milestones; this turn
changed responsibilities inside `oxide-ml-model`, `oxide-ml-macros`,
`oxide-ml-nn`, `oxide-ml-gemma` and the CUDA backend.

```text
core      -> dtype/shape/tensor/parameter store
graph     -> SSA IR, inference, verifier, canonical hashing
kernel    -> backend ABI, provider catalog
cpu       -> reference kernels and executor
autograd  -> reverse-mode training graphs
compiler  -> passes, fusion, memory planning
model     -> Module/ModuleList/enum derive contract, state (KvCache),
             parameter registry + aliasing, CompiledModel, Trainer, cache
nn        -> reusable layers, TiedLinear/TiedWeightSource, losses
safetensors -> mmap reader (F32/F16/BF16), architecture independent
gemma     -> Gemma 4 text config/weights/modules/streaming
cuda      -> standalone cuda-oxide workspace (executor + kernels)
```

Model definitions never contain backend code. Modules emit IR through
`ModelContext`; CPU and CUDA execute the same graph. State is host data
(`KvCache`) owned by the session, so a runtime can reside it on a device
without changing the model definition.

## 3. Module system

### 3.1 `ModuleList<M>`

New in [module.rs](../crates/oxide-ml-model/src/module.rs):

- typed homogeneous collection with integer path segments (`layers.0.weight`);
- `new`, `with_capacity`, `from_vec`, `push`, `insert`, `remove`, `pop`, `len`,
  `is_empty`, `get`, `get_mut`, `first`, `last`, `iter`, `iter_mut`, `as_slice`,
  `as_mut_slice`, `extend`, `into_vec`;
- `Index`, `IntoIterator` (owned/`&`/`&mut`), `FromIterator`, `From<Vec<M>>`;
- `Module` implementation that applies its elements in order (paths `0..n`);
- debug output that prints length instead of requiring `M: Debug`.

`Vec<M>` remains a `Module`, so existing code keeps compiling.

### 3.2 Enum architectural variants

`#[derive(Module)]` now accepts enums. Each variant is an alternative
architecture, not an extra hierarchy level: field names (or tuple indices) are
the path segments in every variant, so a checkpoint/optimizer state stays valid
when a variant changes.

```rust
#[derive(Module)]
enum FeedForward {
    Dense { #[module(body)] mlp: GemmaMlp },
    Sparse { #[module(body)] moe: GemmaMoE, router: Linear },
}
```

- `forward`/`forward_multi` dispatch to the only module field or to the field
  marked `#[module(body)]`, entering that field's path scope;
- other module fields are still discovered, so parameters are structurally
  stable across variants;
- `#[param]`, `#[module(skip)]`, `#[module(rename = ...)]`, generics with
  `FieldType: Module` bounds, and `visit_tensors[_mut]`/`describe` all work;
- ambiguous variants, body markers on parameters, unit variants, empty enums,
  and a custom `forward_multi` option on an enum produce compile errors instead
  of a guessed graph.

### 3.3 Tied parameters

`oxide-ml-nn::TiedLinear<S>` (new
[tied.rs](../crates/oxide-ml-nn/src/tied.rs)):

- `TiedWeightSource` is implemented for `Linear` and `Embedding`;
- `TiedLinear::new(Arc<S>)` applies the shared `[out, in]` matrix to the last
  dimension; `TiedLinear::flattening(...)` flattens `[batch, seq, hidden]` to
  `[seq, hidden]` first (Gemma's LM head);
- parameter discovery registers `head.weight` with the source tensor, so
  `ParameterRegistry` records it as an alias of the canonical
  `embed_tokens.weight`: one graph parameter, one optimizer slot, one
  checkpoint entry, both names resolvable;
- write-back through the tied path is a no-op (the store is authoritative), the
  same rule as shared `Arc<M>`.

Aliasing is storage-based, so a hand-written tied structure also works without
`TiedLinear`.

### 3.4 State declaration

`Module::state_config()` is a new defaulted method returning
`Result<Option<KvCacheConfig>>`. Stateless modules return `Ok(None)` (no
breakage for manual implementations). `Gemma4Text` returns its per-layer KV
layout derived from the typed config, including shared-KV links.

## 4. Procedural macros

### 4.1 `#[derive(Module)]`

- structs: unchanged contract (nested discovery, `#[param]` roles, rename, skip,
  generics with appended `Module` bounds);
- enums: new, as described above;
- generated methods: `module_type`, `visit_parameters`, `forward`,
  `forward_multi`, `set_training`, `is_training`, `visit_tensors`,
  `visit_tensors_mut`, `describe`, and the default `state_config`;
- diagnostics name the fix, e.g. *“module variant has 2 module fields; mark the
  forward target with `#[module(body)]` …”*.

### 4.2 `#[model]`

New: module-collection loops.

```rust
for block in &self.blocks {
    for layer in &block.layers {
        hidden = layer.forward(hidden);
    }
}
```

- supported iterables: `&self.layers`, `self.layers`, `self.layers.iter()`,
  `self.layers.iter().enumerate()` (`for (index, layer) in ...`), and the same
  rooted at a module bound by an enclosing module loop;
- the macro appends the loop index to the parameter path at runtime
  (`blocks.1.layers.0.weight`) and rewrites `.forward()` on the loop variable;
- a loop is only rewritten when its body calls `forward`/`forward_multi` on the
  loop variable (or contains a nested module loop), so ordinary configuration
  loops (`for value in &self.config.heads`) keep their element binding;
- loops are statically unrolled at graph-construction time; tensor-dependent
  control flow remains unrepresentable and is documented, not silently merged;
- generated graph methods carry `#[allow(clippy::clone_on_copy)]` so the natural
  residual style (`let residual = x.clone();`) stays warning-free even though
  graph values are `Copy` ids.

### 4.3 Compile-fail tests

`tests/compile_fail.rs` runs `trybuild` over seven fixtures in `tests/ui/`, with
committed `.stderr` snapshots:

| Fixture | Diagnostic |
| --- | --- |
| tuple struct | named fields required, implement `Module` manually |
| enum unit variant | use `Option<M>` or a variant with a module body |
| enum without body marker | mark the forward field with `#[module(body)]` |
| `#[module(skip)]` + `#[param]` | cannot be combined |
| local `.forward()` | only fields / module-loop variables are rewritten |
| closure body | closures are not supported |
| missing tail expression | forward must end with an output expression |

Regenerate deliberately with
`TRYBUILD=overwrite cargo test -p oxide-ml --test compile_fail`.

## 5. Parameters, weights and memory

- Stable dot-joined names from the module hierarchy; aliases resolved by tensor
  storage address; duplicate names rejected.
- Tied embedding/head is one parameter (verified: store length 3 for
  `embed.weight` + `norm.gamma/beta`, `parameter_ids` has no `head.weight`, and
  `aliases` maps `head.weight → embed.weight`).
- Safetensors loading stays architecture independent (F32/F16/BF16, mmap).
- The existing layer-streaming loader materializes one decoder layer at a time;
  it now uses the framework KV cache instead of ad-hoc vectors.
- Lazy materialization, BF16/F16 conversion and streaming remain in
  `oxide-ml-gemma`; the framework provides the state and parameter contracts,
  not Gemma-specific loading.

## 6. Runtime and execution

- Model definition, graph compilation and execution remain separate:
  `Module` emits IR; `CompiledModel` runs it on the CPU reference backend;
  `CudaModel` runs the same IR on device.
- Compile cache, memory planning, kernel catalog and compiler reports are
  unchanged and tested.
- New state layer ([state.rs](../crates/oxide-ml-model/src/state.rs)):
  - `KvLayerConfig { kv_heads, head_dim, shared_with }`;
  - `KvCacheConfig { capacity, layers }`, with `uniform(...)` convenience;
  - `KvCache`: per-owner storage, automatic growth (`reserve`,
    `next_power_of_two`), `append`/`append_many`, `keys`/`values` slices,
    `len(layer)`, `owner`, `truncate`, `reset`, `sequence_length`;
  - `ModelState { reset, sequence_length }`, implemented by `KvCache`;
  - shared layers read the owner's rows and reject `append`; chained/forward
    shared references are rejected at construction;
  - state is `Send`, owns its buffers, and has no interior mutability, so each
    session gets an independent cache.
- `StreamingGemma` now derives its cache layout from
  `Gemma4Config` (`kv_cache_config`), appends prefill rows, reads past rows per
  decode step, and appends only the new row (the decoder emits concatenated
  K/V). Public `state()`/`reset_state()` expose session control.

## 7. Gemma 4 implementation

- Decoder stack is `ModuleList<GemmaLayer>`; the LM head is
  `TiedLinear<Embedding>` (`TiedHead` is a type alias for compatibility).
- Supported text features: embedding scaling, RMSNorm, no-scale V norm, GQA,
  Q/K norm, sliding and full attention, per-layer RoPE (default and
  proportional), GeGLU, double-wide MLP, PLE, shared KV, tied head, final
  soft-capping and MoE blocks.
- The graph path and the streaming path share the same `GemmaLayer` math
  (`apply_with` / `decode_with`); only orchestration differs, and both produce
  the same logits on the tiny model (`2.999e-7` graph, `3.427e-7` streaming).
- MoE: router RMSNorm/projection/softmax/top-k with per-expert scales and packed
  experts, executed by fused `Op::MoE` on CPU and CUDA. Inference only; no
  autograd rule or generic gather op.
- No multimodal (image/audio) towers are implemented or claimed.

## 8. CUDA and backend status

Implemented and verified this turn on an NVIDIA A100-SXM4-40GB (Compute
Sanitizer enabled):

| Check | Result |
| --- | --- |
| Fresh `cargo oxide build --arch sm_80` from the uploaded source | PASS |
| `cuda_split_softmax_check` (Split + LogSoftmax) | PASS, max diff `1.192e-7`, memcheck 0 errors |
| `gemma_cuda_check` dense tiny | PASS, `1.609e-6` |
| `gemma_cuda_check` MoE tiny (exercises fused `Op::MoE`) | PASS, `1.162e-6`, memcheck 0 errors |
| Rebuild after the module/tied/state refactor | PASS: dense `1.609e-6`, MoE `1.162e-6`, memcheck 0 errors |
| Full E2B f32 after the refactor | PASS: `"Hello"` top-5 identical to CPU (`9259 4.861705`), prefill 29.553 s |
| Fixed-length resident decode (no per-token re-upload) | PASS: 19.9 s load + 29.3 s compile/upload, then 0.40 s/token (`gemma_e2b_resident`, [log](gemma4-e2b-paris-resident.log)); O(T²) compute, not a KV cache |
| Full E2B f32 on CUDA (`gemma_e2b_cuda`) | PASS, top-5 identical to CPU, 4.63 B params |
| E2B CUDA generation (recompute per token, no KV cache) | PASS; `"Hello"` prefill 29.6 s, ≈29.8 s/token |

Not implemented / not claimed: `SliceBackward`, `EmbeddingBackward`, CUDA mixed
precision, batched (>1) Gemma decoding. CUDA KV-cache generation is still
experimental: `Gemma4Text::decode_graph` and `GemmaDecode` feed per-layer K/V
through the graph (CPU graph-decode log:
[gemma4-e2b-paris-cpu-cache.log](gemma4-e2b-paris-cpu-cache.log)), and the CUDA
runner `gemma_e2b_kvcache` compiles, but its first A100 run stopped at the
cuda-core artifact-anchor lookup after the CPU prefill, before any decode step,
so there is no end-to-end CUDA KV-cache parity or latency result. The verified
CUDA generation paths are prompt recomputation and the fixed-length resident
decode above. CPU/CUDA parity is
asserted within `1e-5` for synthetic ops and `1e-3` for tiny Gemma logits.

## 9. Numerical verification

| Scenario | Reference | Max abs diff |
| --- | --- | --- |
| Tiny dense Gemma 4 vs transformers | `reference.json` | `2.999e-7` |
| Tiny E2B-feature (PLE/shared KV/double-wide) | transformers | `2.999e-7` |
| Tiny E2B-feature streaming vs graph | graph path | `3.427e-7` |
| Tiny MoE (4 experts, top-2) | transformers | `1.639e-7` |
| Official `google/gemma-4-E2B-it` vs transformers fp16 | fp16 reference | identical top-5/order |
| CPU/CUDA tiny dense | CPU executor | `1.609e-6` |
| CPU/CUDA tiny MoE | CPU executor | `1.162e-6` |
| CPU/CUDA Split+LogSoftmax | CPU executor | `1.192e-7` |

Greedy generation on the tiny model is unchanged after the redesign
(`[224, 397, 812]` for the first three tokens), and the existing finite
difference gradient checks, shared-parameter tests and checkpoint tests all
pass.

## 10. Performance and memory

Release-build CPU numbers on the reference workspace
(`cargo bench -p oxide-ml --bench module_system`, 8 residual blocks of 2
`Linear(128,128)`, width 128, batch 8):

| Measurement | Result |
| --- | ---: |
| module construction (50 parameters) | 657 µs/model |
| `ModuleList` build (16 × 8×8 Linear) | 4.0 µs |
| parameter discovery | 63.5 µs |
| build + `set_training` propagation | 920 µs/model |
| compiled forward | 10.3 ms/pass |
| graph compile (cold) | 869 µs |
| KV append, 256 tokens × 32 layers (8×128 heads/dim) | 14.5 ms (4.6 GB/s) |
| KV slice lookup, 32 layers | 0.46 µs |
| KV footprint for that session | 64.0 MiB |

Module-system overhead is graph-construction only; inference remains dominated
by the reference matmul. The KV cache copies host f32 data at memory bandwidth;
it is the reusable reference state, not a tuned device cache.

Memory (previously measured, unchanged by this turn):

- full E2B text f32 parameter store: 18.6 GB (A100 40 GB; not a T4);
- layer-streamed E2B on CPU: peak RSS ≈6.2 GB for 9.29 GB BF16 weights;
- the new KV cache is capacity-explicit and reported per session (64 MiB in the
  benchmark above).

No performance improvement is claimed for the redesigned code; the benchmark
numbers are a new baseline measured with the new APIs.

## 11. Remaining limitations

- Control-flow IR (Region/Cond/While) remains design-only; `#[model]` executes
  Rust control flow while building the graph and cannot express
  tensor-dependent branches. Unsupported constructs are compile errors.
- Mixed precision execution and `Cast` between dtypes are still unsupported.
- CUDA backward kernels for `Slice`/`Embedding` are missing. Verified CUDA
  generation re-runs the prompt (≈29.8 s/token); fixed-length resident decode
  avoids the re-upload (0.40 s/token) but still recomputes the full sequence.
  A graph-level per-layer KV decode exists (`GemmaDecode`) and is exercised on
  CPU, but its CUDA runner is experimental and unverified. Batch size is 1.
- Gemma MoE is inference-only and fused; no autograd rule or expert-parallel
  execution. Expert execution on CUDA assumes `num_experts <= 64` and
  `top_k <= 8` (compile-time kernel capacities).
- `Dropout` is not stochastic; BatchNorm does not update running statistics;
  Conv3d and full RNN sequence outputs are limited (see
  [known-limitations.md](known-limitations.md)).
- Paged KV is not implemented. `KvCache` is capacity-based and contiguous; a
  paged extension would need a block table in the IR and a gather kernel.
- The layer-streamed Gemma path repeats host-side orchestration (embedding
  lookup, PLE packing, final norm/tied head) because the full-model graph does
  not fit in memory. The decoder-layer math is shared with the graph path and
  both are pinned to the same tiny-model logits (`2.999e-7` / `3.427e-7`), so
  the duplication is deliberate and regression-tested rather than silent.
- Distributed execution is in-process/in-process-thread only.
- The CPU backend remains a correctness reference, not a tuned BLAS.

## 12. Breaking changes and migration

| Change | Action |
| --- | --- |
| `Gemma4Text::layers` is `ModuleList<GemmaLayer>` (was `Vec<GemmaLayer>`) | `.iter()`, `len()`, indexing still work; use `.as_slice()` / `.into_vec()` where a slice/`Vec` is required |
| `Gemma4Text::lm_head` is `TiedLinear<Embedding>`; `TiedHead` is an alias | construct with `TiedLinear::flattening(embed.clone())` |
| `Module` gained a defaulted `state_config` | no change required for manual implementations |
| New `ModuleList`, `KvCache`, `KvLayerConfig`, `ModelState`, `TiedLinear`, `TiedWeightSource` exports | additive |
| `trybuild` added as a dev-dependency of `oxide-ml` | dev-only |

There are no changes to graph IR, checkpoint format, compiler cache keys or
parameter naming for existing struct models.

## 13. Reproduction

```bash
# CPU correctness (all suites, including module system, KV cache, compile-fail)
cargo test --workspace

# tiny Gemma parity (dense graph path and streaming path)
cargo run -p oxide-ml-gemma --bin gemma4_infer -- \
  --model .refs/gemma4-e2b-tiny --tokens 5,17,42 \
  --reference .refs/gemma4-e2b-tiny/reference.json --generate 3

cargo run -p oxide-ml-gemma --bin gemma4_infer -- \
  --model .refs/gemma4-e2b-tiny --tokens 5,17,42 \
  --reference .refs/gemma4-e2b-tiny/reference.json --generate 3 --streaming

# benchmarks
cargo bench -p oxide-ml --bench module_system
cargo bench -p oxide-ml --bench compiler
cargo bench -p oxide-ml --bench model

# CUDA (standalone workspace, NVIDIA toolchain)
cd crates/oxide-ml-cuda
cargo oxide build --arch sm_80 -- --bin cuda_split_softmax_check
cargo oxide build --arch sm_80 -- --bin gemma_cuda_check
```

The CUDA evidence for this report is recorded in
[gemma4-cuda-san_split.log](gemma4-cuda-san_split.log),
[gemma4-cuda-san_moe.log](gemma4-cuda-san_moe.log),
[gemma4-cuda-e2b_gen.log](gemma4-cuda-e2b_gen.log) and
[gemma4-e2b-paris.log](gemma4-e2b-paris.log) and
[gemma4-cuda-e2b-refactor.log](gemma4-cuda-e2b-refactor.log).

## 14. Files changed in this turn

- `crates/oxide-ml-model`: `module.rs` (`ModuleList`, `state_config`),
  `state.rs` (new), `lib.rs`.
- `crates/oxide-ml-macros`: `derive.rs` (enum derive), `model.rs` (module
  loops, clippy allowances), `attr.rs` (`#[module(body)]`), `Cargo.toml`.
- `crates/oxide-ml-nn`: `tied.rs` (new), `lib.rs`.
- `crates/oxide-ml-gemma`: `modules.rs` (`ModuleList`, `TiedLinear`, state
  declaration), `streaming.rs` (`KvCache`), `lib.rs` (exports).
- `crates/oxide-ml`: exports, prelude, test registration, `trybuild`
  dev-dependency.
- CUDA: `executor.rs`/`kernels.rs` (`LogSoftmax`, `Split`, `MoE`),
  `cuda_split_softmax_check.rs` (new), `gemma_e2b_cuda.rs` (timed generation).
  The tree also contains experimental decode work (`GemmaDecode`,
  `CudaModel::rebuild`, `gemma_e2b_resident`, `gemma_e2b_kvcache`); only the
  resident fixed-length decode has a recorded run, and no CUDA KV-cache run
  reached a decode step.
- Tests: `tests/module_system.rs`, `tests/kv_cache.rs`, `tests/compile_fail.rs`,
  `tests/ui/*`.
- Benchmark: `benches/module_system.rs`.
- Docs: this report plus `model-api.md`, `proc-macros.md`, `gemma4.md`,
  `known-limitations.md`, `verification-report.md`, `benchmark-report.md`,
  `migration-0.1.md`, `README.md`.
