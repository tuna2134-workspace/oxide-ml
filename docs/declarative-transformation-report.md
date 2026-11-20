# Declarative architecture transformation: engineering report

This report records the transformation requested for oxide-ml: make model
definitions declarative and composable, keep Rust type safety and ownership,
and preserve the existing graph/compiler/backend stack. It is evidence-based:
commands and results are included, and unverified claims are marked.

## 1. Outcome

oxide-ml now supports the mission's model-authoring style end to end:

```rust
#[derive(Module)]
pub struct DecoderLayer {
    attention_norm: RmsNorm,
    attention: MultiHeadAttention,
    mlp_norm: RmsNorm,
    mlp: FeedForward,
}

impl DecoderLayer {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x);
        let x = residual + x;
        let residual = x.clone();
        let x = self.mlp_norm.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}
```

The complete, runnable version is
[examples/decoder_transformer.rs](../examples/decoder_transformer.rs)
(`cargo run -p oxide-ml --example decoder_transformer`).

Concretely:

- models are ordinary Rust structs/enums composed of modules;
- `#[model]` turns ordinary tensor expressions, residuals, loops and nested
  module calls into graph construction;
- multi-input calls and multi-output destructuring are supported and checked;
- reusable transformer components (`RmsNorm`, `RmsNormNoScale`, `GatedMlp`,
  RoPE, masks, GQA math, tied head) live in `oxide-ml-nn`;
- Gemma 4 uses those components for its norms, MLP, RoPE, masks and attention
  math, keeping only architecture composition;
- inference state is separated from the model: `StreamingSession` owns a
  `KvCache`, so sessions cannot share mutable state;
- CPU, graph compiler, memory planner, KV cache and CUDA backends are preserved.

## 2. Audit: limitations found

Reading the workspace before changing it found these concrete gaps:

| Gap | Effect |
| --- | --- |
| `.forward(a, b, ...)` rejected in `#[model]` | multi-input modules could not be composed declaratively |
| multi-output module calls returned a `Vec<ValueId>` | no safe destructuring; easy to misuse |
| tuple structs rejected by the derive | only named-field modules were natural |
| parameter paths ignored tuple fields | `self.0.forward(x)` was not a module call |
| no reusable RMSNorm/RoPE/mask/GQA components | Gemma carried private copies |
| `StreamingGemma` owned the KV cache | one model = one session; sessions could not be isolated structurally |
| no compile-pass macro fixtures | only negative diagnostics were tested |
| no transformer benchmark | construction/traversal/forward cost of declarative composition was unmeasured |

These were fixed in the order above, with the working graph/compiler/backend
infrastructure reused unchanged.

## 3. Architectural changes

### 3.1 Module system

- `ModuleList<M>` (previous revision) remains the collection type; it now
  composes with tuple paths and loops.
- `#[derive(Module)]` accepts named structs, tuple structs and enums. Tuple
  fields use index paths (`0.weight`).
- Enum variants remain alternative architectures sharing parameter paths; the
  `#[module(body)]` field receives `forward`.
- `Module::state_config` remains the declaration point for execution state.

### 3.2 Declarative tensor and module API

`#[model]` rewriting now covers:

| Syntax | Lowering |
| --- | --- |
| `self.m.forward(x)` | `module_forward` (single output) |
| `self.m.forward(a, b)` | `module_forward_multi` (single output) |
| `let (a, b) = self.m.forward(x)` | multi-output call + `output_pair` arity check |
| `let (a, b, c) = self.m.forward_multi(a, b)` | `output_triple`, explicit multi-input |
| `self.0.forward(x)` | tuple field path `"0"` |
| `x.gelu()`, `x.matmul(w)`, `x.softmax(-1)` | existing tensor ops |
| `a + b`, `a * b`, `-x` | graph ops |
| `for layer in &self.layers` | statically unrolled, indexed path |
| `if` / blocks / `let` / `match` | host-side while building the graph |
| tail `(a, b)` | multi-output model |

Destructuring is limited to 2–4 outputs with an explicit runtime arity error;
larger arities must index `forward_multi`. A multi-output call used as a tensor
is an error rather than a silent drop.

### 3.3 Reusable components (`oxide-ml-nn`)

New `transformer` module:

- `RmsNorm` (learned scale) and `RmsNormNoScale`;
- `GatedMlp` + `GatedActivation` (GeGLU, SwiGLU, ReGLU);
- `RopeConfig` with `tables`, `apply_rope`, `apply_rope_at`, `rotate_half`,
  `rotate_half_value`;
- `causal_mask(length, sliding)`;
- `GqaConfig` + `grouped_query_attention` (GQA/SDPA math for
  `[1, length, heads, head_dim]`, used for prefill and decode).

### 3.4 Gemma 4 migration

| Component | Before | After |
| --- | --- | --- |
| norms | private `RmsNorm`/`NoScaleRmsNorm` | `oxide_ml_nn::{RmsNorm, RmsNormNoScale}` |
| MLP | private `GemmaMlp` | `oxide_ml_nn::GatedMlp` (`GemmaMlp` alias) |
| RoPE | private tables/application | `RopeConfig` + `apply_rope_at` |
| masks | private `causal_mask` | `oxide_ml_nn::causal_mask` |
| attention heads | private `#[model] head_attention` + loop | `grouped_query_attention` |
| decoder stack | `ModuleList<GemmaLayer>` | unchanged |
| tied head | `TiedLinear<Embedding>` | unchanged |

Gemma-specific code that remains: configuration/weight mapping, PLE residuals,
cross-layer shared-KV routing, per-layer configuration, fused MoE. All ordinary
tensor math now comes from framework components. `GemmaLayer` and `Gemma4Text`
remain manual because they thread optional PLE inputs and cached K/V across
layers (architecture composition); multi-input/multi-output calls are now
supported by the macro, so this is not a syntax limitation.

### 3.5 Runtime state and sessions

`StreamingGemma` is now the model (weights + compiled layer templates).
`StreamingSession` owns a `KvCache` and exposes `state()`, `sequence_length()`,
`reset()`, `into_state()`. `prefill(&mut session, ids)` and
`decode(&mut session, token)` cannot share mutable state across requests.
`prefill`/`decode` still take `&mut self` because the model caches compiled
per-layer graphs; concurrent sessions need a model per worker or a mutex.

## 4. New and modified public APIs

New:

- `oxide_ml::prelude::{RmsNorm, RmsNormNoScale, GatedMlp, GatedActivation,
  GqaConfig, RopeConfig}`;
- `oxide_ml::{module_forward, module_forward_multi, single_output,
  grouped_query_attention, apply_rope, apply_rope_at, causal_mask, rms_norm,
  rotate_half, rotate_half_value}`;
- `oxide_ml_nn::transformer::*` and `oxide_ml_model::{output_pair,
  output_triple, output_quad}` (macro helpers);
- `oxide_ml_gemma::StreamingSession`;
- `StreamingGemma::session()`, `StreamingSession::{state, sequence_length,
  reset, into_state}`.

Changed:

- `StreamingGemma::{prefill, decode}` take `&mut StreamingSession`;
  `StreamingGemma::{state, reset_state}` were removed;
- `GemmaMlp` is a type alias for `GatedMlp`; construct with `::geglu(...)`;
- `RmsNorm`/`NoScaleRmsNorm` are re-exports of the generic components;
- `#[model]` `.forward_multi(...)` without destructuring requires one output.

## 5. Macro capabilities added

- multi-input module calls via ordinary `.forward(a, b)`;
- multi-output destructuring with arity checks (`let (a, b) = ...`);
- tuple structs with index paths;
- tuple field access in module paths;
- compile-time diagnostics for destructuring >4 outputs;
- compile-pass fixtures for the declarative style and multi-IO modules;
- clippy-clean generated code for residual-style `x.clone()`.

Unsupported constructs remain explicit compile errors: indirect `.forward()` on
locals, `self.forward`, closures/async, missing output expressions, and
tensor-dependent control flow.

## 6. Test results

Command:

```bash
cargo test --workspace
```

Result: **424 passed, 0 failed**.

New/updated coverage:

| Suite | Tests | Covers |
| --- | ---: | --- |
| `tests/module_system.rs` | 12 | ModuleList, loops, enums, tied, tuple structs, multi-input/output |
| `tests/kv_cache.rs` | 7 | cache append/growth/sharing/reset/sessions |
| `tests/transformer_components.rs` | 8 | norms, gated MLP, RoPE, masks, rotate-half, GQA vs direct math |
| `oxide-ml-gemma/tests/streaming_sessions.rs` | 2 | prefill+decode vs full forward, session isolation/replay |
| `tests/compile_fail.rs` | 7 fail + 2 pass fixtures | diagnostics and accepted declarative code |
| macro unit tests | 39 | expansion and diagnostics |
| all earlier suites | unchanged | graph, compiler, autograd, CPU, training, checkpoints, cache |
| `tests/paged_kv_cache.rs` | 6 | paged blocks, gather, capacity, sharing, truncate |
| `tests/half_precision.rs` | 2 | F16/BF16 low-level execution |
| `transformer_components` (MoE) | +2 | generic differentiable MoE and router gradients |
| `streaming_sessions` (concurrent) | +1 | two sessions on threads |

`cargo fmt --all --check` passes; `cargo clippy --workspace --all-targets`
reports zero warnings.

## 7. Numerical correctness

After migrating Gemma to the generic components (including the GQA helper):

| Check | Result |
| --- | --- |
| tiny dense Gemma vs Transformers reference | `2.999e-7` max abs |
| tiny E2B-feature streaming vs reference | `3.427e-7` max abs |
| greedy generation (first 3 tokens) | `[224, 397, 812]` (unchanged) |
| prefill + decode vs full forward (tiny model) | `< 1e-5`, session replay bit-identical |
| GQA component vs direct Rust computation | `< 1e-6` |
| GeGLU component vs explicit composition | `< 1e-6` |

No tolerance was relaxed.

## 8. CPU / CUDA status

- CPU: all suites above pass.
- CUDA: kernels and the graph executor were **not modified** in this revision.
  The A100 results from the previous milestone remain the evidence for CUDA
  (`LogSoftmax`/`Split`/`MoE` parity `1.192e-7`/`1.162e-6`, Compute Sanitizer 0
  errors, full E2B f32 top-5 identical to CPU, KV-cache decode ≈0.30 s/token).
  This revision changed `gemma_e2b_kvcache` to use `StreamingSession`; the CUDA
  workspace was **not compiled or run** here (no CUDA toolkit/GPU in this
  environment), so that binary change is unverified. CPU-side Gemma numerics
  were re-verified after the migration.
- The CUDA crate remains a standalone workspace; no second CUDA abstraction was
  introduced.

## 9. Benchmarks

`cargo bench -p oxide-ml --bench module_system` (release, reference CPU):

| Measurement | Result |
| --- | ---: |
| transformer construction (4 layers, 68 params) | 931 µs |
| transformer parameter discovery | 119 µs |
| transformer cold compile | 2.07 ms |
| transformer compiled forward (batch 4, seq 8) | 18.2 ms |
| transformer graph size | 151 nodes |
| ModuleList stack construction (8 blocks) | 657 µs |
| stack parameter discovery | 63–65 µs |
| stack compiled forward (batch 8) | 9.9–11.1 ms |
| KV append (256 tokens × 32 layers) | 13.7–14.3 ms (4.7–4.9 GB/s) |
| KV slice lookup (32 layers) | 0.33–0.40 µs |

The declarative layer adds graph construction only; the compiled graph is the
same IR. Host timings vary by several percent between runs, so no speed-up is
claimed.

## 10. Files changed (this revision)

- `crates/oxide-ml-nn`: new `transformer.rs`; `lib.rs` exports; `Cargo.toml`
  (macros dependency).
- `crates/oxide-ml-macros`: `derive.rs` tuple structs; `model.rs` multi-input,
  multi-output, tuple paths, compile-pass-friendly diagnostics.
- `crates/oxide-ml-model`: `context.rs` output arity helpers; `lib.rs` exports.
- `crates/oxide-ml`: facade/prelude exports.
- `crates/oxide-ml-gemma`: `modules.rs` (generic components, GQA, tuple paths),
  `streaming.rs` (`StreamingSession`), `gemma4_infer` streaming usage.
- `crates/oxide-ml-cuda/src/bin/gemma_e2b_kvcache.rs`: session API (not compiled
  here).
- Tests: `tests/module_system.rs`, `tests/transformer_components.rs`,
  `tests/ui/pass/*`, `crates/oxide-ml-gemma/tests/streaming_sessions.rs`.
- Example: `examples/decoder_transformer.rs`.
- Benchmark: `benches/module_system.rs`.
- Docs: `declarative-model-architecture.md`, this report, `model-api.md`,
  `proc-macros.md`, `gemma4.md`, `known-limitations.md`, `migration-0.1.md`,
  `verification-report.md`, `benchmark-report.md`, `README.md`.

## 11. Known limitations and remaining work

- Structured named outputs (`out.hidden`) are not generated; tuple outputs are
  the supported form.
- `grouped_query_attention` is single-batch (batch is the caller's loop).
- Tensor-dependent control flow is still not representable; `if`/loops are
  evaluated while building the graph.
- `StreamingGemma` requires `&mut self` for prefill/decode; concurrency needs one
  model per worker or a mutex.
- CUDA KV-cache binary changes are unverified in this environment; the CUDA
  workspace should be rebuilt on an A100/T4 runner.
- MoE remains a fused inference-only op; there is no generic `TopK` IR op.
- Paged KV, mixed precision, distributed execution and multimodal Gemma remain
  out of scope.

## 12. Reproduction

```bash
cargo test --workspace
cargo test -p oxide-ml --test module_system --test transformer_components \
           --test kv_cache --test compile_fail
cargo test -p oxide-ml-gemma --test streaming_sessions
cargo run -p oxide-ml --example decoder_transformer
cargo run -p oxide-ml-gemma --bin gemma4_infer -- \
  --model .refs/gemma4-e2b-tiny --tokens 5,17,42 \
  --reference .refs/gemma4-e2b-tiny/reference.json --generate 3 --streaming
cargo bench -p oxide-ml --bench module_system
```

## 13. Follow-up revision: concurrency, paging, precision and generic MoE

Implemented after the declarative transformation:

| Item | Implementation | Verification |
| --- | --- | --- |
| Concurrent streaming sessions | `StreamingGemma` uses a mutex-protected compiled-template cache and `prefill`/`decode` take `&self`; each `StreamingSession` owns its KV cache | `streaming_sessions.rs`: two sessions on threads match sequential replays |
| Paged KV cache | `PagedKvCache`/`PagedKvConfig`: block tables, physical K/V pools, shared layers, truncate/reset, gather | `tests/paged_kv_cache.rs` (6) |
| Mixed precision (low-level) | `half::f16`/`bf16` implement `Scalar`; graph executor runs F16/BF16 tensors | `tests/half_precision.rs` (2) |
| Generic MoE | `MixtureOfExperts` with differentiable soft routing over arbitrary expert modules | `tests/transformer_components.rs`: forward parity and finite-difference router gradients |
| CUDA compatibility | rebuilt on a fresh T4 session with the current source | see §14 |
| GPU performance | `gemm_bench`/`benchmark_check` on T4 | see §14 |

Limits of this revision: half precision has no autocast/loss scaling and CUDA is
F32; sparse top-k MoE remains the fused inference-only `Op::MoE`; paging is
host-side (block table available for device runtimes).

## 14. CUDA and GPU re-verification (fresh T4 session)

A new Tesla T4 session was provisioned because the previous one had expired.
The current source was uploaded, Rust 1.99.1 and `cargo-oxide 0.4.0` installed,
and the CUDA workspace built with `--arch sm_75` (clang was required for
bindgen).

| Check | Result |
| --- | --- |
| `gemma_cuda_check` dense tiny | PASS, `1.609e-6` |
| `gemma_cuda_check` MoE tiny | PASS, `1.162e-6` |
| `model_check` forward / 20-step training / parameters | `7.749e-7` / `2.325e-6` / `2.623e-6` |
| `gpu_check` (execution safety, reclamation, backward, dynamic shapes, training) | PASS |
| Compute Sanitizer memcheck (`gemma_cuda_check`) | 0 errors |
| `gemm_bench` 512³ tiled / naive | 566.2 µs / 1,569.3 µs (CPU 29.8 ms) |
| `benchmark_check` | 30 cases, 100 samples each |
| Full E2B f32 on CUDA | not re-run: 18.6 GB does not fit a 15 GiB T4; the earlier A100 result stands for the kernels/full model |

Evidence: [t4-declarative-verify.log](../docs/t4-declarative-verify.log),
[t4-gpu-bench.log](../docs/t4-gpu-bench.log),
[phase7f19-benchmark-results-t4.json](../docs/phase7f19-benchmark-results-t4.json).

The T4 rebuild proves the framework changes compile and run against the CUDA
backend. GPU timings are re-measured on this session; the host differs from the
earlier runs, so comparisons are indicative, not a controlled benchmark.
