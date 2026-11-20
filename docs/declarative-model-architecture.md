# Declarative model architecture

This document describes how oxide-ml turns ordinary Rust model code into a
verified graph, how parameters and runtime state are separated, and how to add a
new architecture. It reflects the current implementation, not a proposal.

## 1. Design in one page

```text
#[derive(Module)] struct      parameter discovery + traversal + metadata
        │
        ├── fields: modules, ModuleList, Option, enums, Arc, #[param] tensors
        │
        └── #[model] fn forward(...)          ordinary Rust expressions
                    │
                    ▼
            ModelContext + GraphBuilder       typed, path-scoped graph building
                    │
                    ▼
                 Graph IR                     SSA values, symbolic shapes
                    │
        verifier → optimizer → memory plan
                    │
             ┌──────┴───────┐
          CPU executor   CUDA executor
```

Model authors write modules, tensor expressions and composition. They do not
write `ValueId`s, graph nodes or parameter registration. Advanced code can still
drop to `GraphBuilder` through `ModelContext::builder`.

## 2. Crate responsibilities

| Crate | Responsibility |
| --- | --- |
| `oxide-ml-core` | dtypes, symbolic shapes, dense tensors, `ParameterStore` |
| `oxide-ml-graph` | SSA IR, inference, verifier, canonical hashing, `GraphBuilder` |
| `oxide-ml-model` | `Module`, `ModuleList`, `ModulePath`, `ParameterRegistry`, `ModelContext`, `CompiledModel`, `KvCache`, compile cache |
| `oxide-ml-macros` | `#[derive(Module)]`, `#[model]`, `graph!` |
| `oxide-ml-nn` | reusable layers and transformer components, losses, `TiedLinear` |
| `oxide-ml-compiler` | constant folding, CSE, fusion, DCE, memory planning |
| `oxide-ml-cpu` | reference kernels and executor |
| `oxide-ml-cuda` | standalone cuda-oxide workspace (executor + kernels) |
| `oxide-ml-safetensors` | mmap reader, F32/F16/BF16 |
| `oxide-ml-gemma` | Gemma 4 configuration, weights, decoder composition, streaming |
| `oxide-ml` | facade and prelude |

There is one `Module` abstraction. `Sequential`, `ModuleList`, `Option`, `Vec`,
`Box`, `Arc` and derived enums all implement that same trait.

## 3. The Module system

### 3.1 Trait contract

```rust
pub trait Module {
    fn module_type(&self) -> &'static str;
    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()>;
    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>>;
    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>>;
    fn set_training(&mut self, training: bool);
    fn is_training(&self) -> bool;
    fn visit_tensors(&self, path, visitor) -> Result<()>;
    fn visit_tensors_mut(&mut self, path, visitor) -> Result<()>;
    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor);
    fn state_config(&self) -> Result<Option<KvCacheConfig>> { Ok(None) }
}
```

A `Module` is a *description*, not a session: it owns parameters and knows how to
append operations to a context. The same value is traversed for parameters (for a
store or checkpoint) and replayed into a graph.

### 3.2 Composition forms

| Rust form | Behaviour |
| --- | --- |
| field of a derived struct | child module at path `<field>` |
| `ModuleList<M>` | children at paths `0`, `1`, ...; optional sequential `forward` |
| `Vec<M>` | same path scheme, sequential `forward` |
| `Option<M>` | skipped when `None`; error if forwarded while absent |
| `Box<M>`, `Arc<M>` | transparent; `Arc` marks a shared instance |
| enum deriving `Module` | alternative architectures; fields share paths; forwards to the `#[module(body)]` field |
| tuple struct deriving `Module` | children at paths `0`, `1`, ... |
| `#[param]` tensor field | parameter at the field path (or `#[module(rename)]`) |
| `#[module(skip)]` | configuration only; excluded from traversal |
| `()` | identity module |

Generic modules are supported; the derive adds `Field: Module` bounds:

```rust
#[derive(Module)]
struct Block<A: Module, M: Module> {
    attention: A,
    mlp: M,
    norm: RmsNorm,
}
```

### 3.3 Paths, roles and aliases

Paths are dot-joined from the root module and never contain graph ids:

```text
embed_tokens.weight
layers.0.self_attn.q_proj.weight
layers.11.mlp.down_proj.weight
```

Each parameter has a role (`trainable`, `frozen`, `buffer`). Discovery is
deterministic (declaration order, depth first). If two paths reach the same
tensor storage, the second is recorded as an alias of the first canonical
parameter: one graph parameter, one optimizer slot, one checkpoint entry, both
names resolvable. `TiedLinear` uses this to tie an LM head to its embedding.

`Module::state_config` declares mutable execution state (for Gemma 4: one KV slot
per non-shared layer plus shared-KV links).

## 4. Declarative programming model

### 4.1 `#[model]` forward methods

```rust
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

The method is compiled twice conceptually: a graph-building method
(`forward(&self, cx, input)`) and a `predict` convenience method with the
declared signature. Inside the body:

| Syntax | Graph meaning |
| --- | --- |
| `self.mod.forward(x)` | append `mod` with a path scope; one output |
| `self.mod.forward(a, b, ...)` | multi-input call (`forward_multi`); exactly one output |
| `let (a, b) = self.mod.forward(x);` | multi-output call destructured by arity (2–4) |
| `let (a, b) = self.mod.forward_multi(x, y)?;` | same, explicit multi-input entry point |
| `layer.forward(x)` in a collection loop | indexed path `layers.<i>` |
| `a + b`, `a - b`, `a * b`, `a / b` | `add`/`sub`/`mul`/`div` ops |
| `-x` | `neg` |
| `x.gelu()`, `x.softmax(-1)`, `x.matmul(w)`, ... | tensor operations |
| `gelu(x)`, `matmul(a, b)`, ... | free-function forms |
| `for layer in &self.layers { ... }` | statically unrolled at graph-construction time |
| `if condition { ... } else { ... }` | host-side branch while building the graph |
| `(a, b)` as the tail | multi-output model |

Tensor operations are type-checked by rustc after expansion; shapes are checked
by graph inference at compile time (build time of the graph) and by the runtime
executor for dynamic dimensions. Errors name the operation and shapes.

A module call without destructuring always requires exactly one output. This is
deliberate: a multi-output module called as a tensor is a runtime error instead
of silently dropping values. Destructure it to receive all outputs.

### 4.2 Explicitly unsupported (compile error)

- `.forward()` on a local not bound by a module-collection loop
- `self.forward(...)` (would recurse)
- closures, `async`, `.await`, `unsafe`
- a body without an output expression, or bare `return;`
- destructuring more than four module outputs into a tuple (index the
  `forward_multi` result instead)
- tensor-dependent control flow: conditions and loop bounds are evaluated while
  building the graph, never per element

## 5. Macro transformation

### 5.1 Pipeline

1. Parse with `syn` (`DeriveInput` or `ImplItemFn`).
2. Validate the shape (named/tuple fields, receiver, signature, no async/unsafe).
3. `VisitMut` rewriting of expressions and statements:
   module calls receive path scopes, tensor methods become `ops::*` calls,
   binary operators become ops, collection loops become indexed lowering, tuple
   statements get output-arity checks.
4. Generate ordinary Rust that calls `Module`, `ModelContext::builder` and
   `oxide_ml_model::__private` helpers.
5. Aggregate `syn::Error`s with spans; unsupported syntax aborts expansion.

There is no string substitution: every rewrite is parsed back into `syn` nodes,
so malformed output is a macro bug that fails at compile time with a span.

### 5.2 `#[derive(Module)]` expansion

Generates `module_type`, `visit_parameters`, `forward` (delegating to the
inherent `forward_multi`), `forward_multi`, `set_training`, `is_training`,
`visit_tensors`, `visit_tensors_mut`, `describe` and the default `state_config`.

- structs: named and tuple fields;
- enums: per-variant match arms, body-field dispatch, shared parameter paths;
- generics: `FieldType: Module` predicates appended;
- diagnostics: tuple/unit issues, ambiguous enum bodies, skip+param conflicts.

### 5.3 Testing

- 39 macro unit tests (accepted syntax and diagnostics);
- 7 `trybuild` compile-fail fixtures with committed `.stderr`;
- 2 `trybuild` compile-pass fixtures: [declarative_model.rs](../tests/ui/pass/declarative_model.rs)
  and [multi_io.rs](../tests/ui/pass/multi_io.rs);
- integration tests for loops, multi-input/output, tuple structs and enums in
  [module_system.rs](../tests/module_system.rs).

## 6. Graph construction and inspection

While `#[model]` executes, `ModelContext` owns a `GraphBuilder`, a path stack
and the name → `ValueId` parameter map. `module_forward` pushes path segments,
runs the child, and truncates the stack, so parameters resolve relative to the
module (`weight` becomes `block.fc.weight`).

The resulting graph is an SSA IR with symbolic shapes. It is verified before
execution (dense tables, definitions before use, no duplicate definitions,
re-inferred shapes, parameter metadata), then optimized (constant folding, CSE,
fusion, DCE) and memory-planned.

Inspection:

```rust
let compiled = CompiledModel::compile(&model, &specs, ModelCompileOptions::default())?;
compiled.report();            // passes, fusions, memory, timings
compiled.forward_graph();     // Graph
compiled.parameter_ids();     // name -> ParameterId
compiled.aliases();           // alias -> canonical name
```

## 7. Parameters and weights

- `ParameterRegistry` records name, shape, role and module path; duplicate names
  are rejected; shared storage becomes an alias.
- `ParameterStore` is the runtime authority during training; `write_back` copies
  values into the module tree where storage is uniquely owned.
- `TiedLinear<S>` + `TiedWeightSource` tie a head to a `Linear`/`Embedding`.
  Cloning the source gap-matrix (not the `Arc`) duplicates the parameter; the
  API documents this.
- Safetensors loading is architecture independent (mmap, F32/F16/BF16).
  Layer-streamed loading materializes one layer at a time.

## 8. Runtime, sessions and state

Model definitions are immutable descriptions. Execution state is separate:

| Concept | Type | Lifetime |
| --- | --- | --- |
| structure + parameters | module value | process |
| compiled graph | `CompiledModel` | compiled once, reusable |
| parameter store | `ParameterStore<f32>` | training/inference session |
| KV state | `KvCache` | one inference session |
| streaming session | `StreamingSession` | one request |
| backend state | CPU/CUDA executor | per executor |

`KvCache` has per-layer shapes, shared-KV links, explicit capacity growth,
append/read/truncate/reset and `ModelState::{reset, sequence_length}`. It is
`Send`, owns its buffers and has no interior mutability. `PagedKvCache` adds a
block table over fixed-size physical blocks, with the same shared-KV semantics,
`gather_keys`/`gather_values` for contiguous materialization and explicit
block-capacity errors.

`StreamingGemma` owns weights and a shared, interior-mutable cache of compiled
layer templates. A `StreamingSession` owns all mutable sequence state, so
`prefill(&session)` and `decode(&session)` take `&self` on the model:

```rust
let model = Arc::new(StreamingGemma::open(dir)?);
let handles = prompts.map(|prompt| {
    let model = Arc::clone(&model);
    thread::spawn(move || {
        let mut session = model.session()?;
        let logits = model.prefill(&mut session, &prompt)?;
        model.decode(&mut session, argmax(&logits))
    })
});
```

Two sessions cannot share KV state, and the model can serve them concurrently;
a test runs two sessions on threads and compares each with a sequential replay.
Compiled layer templates are shared through a mutex (compilation is done
outside the lock).

### Prefill and decode

The graph path builds a whole sequence; the streaming path compiles one decoder
layer per shape and reuses the same `GemmaLayer` math for prefill and decode.
`prefill` appends all prompt rows to the cache; `decode` reads the cached rows,
appends the new row, and advances the position. Both use the same RoPE table
definition and the same attention component.

## 9. Backend integration

Model code never names a backend. It emits IR; `CompiledModel` runs it on the
CPU executor; `CudaModel` uploads parameters once and runs the same IR on device.
CUDA kernels live in the standalone `oxide-ml-cuda` workspace (cuda-oxide).
Unsupported operations return an explicit backend error; the CUDA executor never
falls back to CPU.

Verified CUDA coverage in this repository includes elementwise ops, matmul,
LayerNorm, Embedding, Slice, Concat, Split, LogSoftmax, fused MoE and the full
Gemma 4 text model (A100); `SliceBackward`/`EmbeddingBackward` remain
unsupported. See [verification-report.md](verification-report.md).

## 10. Reusable transformer components

| Component | Type | Notes |
| --- | --- | --- |
| `RmsNorm` | module | learned scale, `with_dim`/`new` |
| `RmsNormNoScale` | module | value normalization |
| `GatedMlp`, `GatedActivation` | module | GeGLU / SwiGLU / ReGLU, `geglu(...)` |
| `RopeConfig`, `apply_rope`, `apply_rope_at` | config + graph helper | default and proportional RoPE |
| `causal_mask` | helper | causal + sliding window |
| `GroupedQueryAttention` math (`GqaConfig`, `grouped_query_attention`) | graph helper | GQA/SDPA over `[1, L, H, D]`, prefill and decode |
| `Linear`, `Embedding`, `LayerNorm` | modules | unchanged builders |
| `MultiHeadAttention`, `Attention` | modules | causal/self-attention and 3-input SDPA |
| `TiedLinear` | module | tied embedding/head |
| `MixtureOfExperts`, `MoeRouting` | module | soft differentiable routing over any expert modules |

```rust
#[derive(Module)]
struct FeedForward {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let gate = self.gate.forward(x).gelu();
        let up = self.up.forward(x);
        self.down.forward(gate * up)
    }
}
```

## 11. Implementing a new model

1. Split the architecture into reusable modules (norms, attention, MLP).
2. Give each module concrete parameter fields; use `#[module(skip)]` for
   configuration.
3. Write `#[model] forward` with ordinary Rust composition.
4. Stack repeated layers with `ModuleList` and a `for` loop.
5. Declare state with `state_config` if the architecture is a decoder.
6. Compile with `CompiledModel` (or `CudaModel`) and inspect `report()`.
7. Add a numerical test against an independent reference and a parity test
   between prefill and decode.

A complete executable example is
[examples/decoder_transformer.rs](../examples/decoder_transformer.rs); the
Gemma decoder is the production example.

## 12. How Gemma 4 uses the framework

| Piece | Implementation |
| --- | --- |
| norms (input/post-attention/pre-FF/final) | `oxide_ml_nn::RmsNorm` |
| value norm | `oxide_ml_nn::RmsNormNoScale` |
| GeGLU MLP (double-wide) | `oxide_ml_nn::GatedMlp` |
| RoPE tables and application | `oxide_ml_nn::{RopeConfig, apply_rope_at}` |
| attention masks | `oxide_ml_nn::causal_mask` |
| GQA attention math | `oxide_ml_nn::grouped_query_attention` |
| tied LM head | `oxide_ml_nn::TiedLinear` |
| decoder stack | `ModuleList<GemmaLayer>` |
| PLE, shared-KV selection, MoE | Gemma-specific composition on generic ops |

Gemma-specific code is limited to configuration/weight mapping, PLE, shared-KV
routing and the fused MoE module. `GemmaLayer` still composes its residuals with
`module_forward` calls rather than `#[model]` because it threads optional PLE
inputs and cached K/V across layers; `#[model]` supports multi-input and
multi-output calls, so this is architecture composition, not a syntax gap.

## 13. Mixed precision

`DenseTensor<T>` and the graph executor are generic over `Scalar`. `f16` and
`bf16` now implement `Scalar` (half-precision arithmetic with f32 transcendental
helpers), so low-level graphs can be built and executed in F16/BF16:

```rust
let mut inputs = ExecutionInputs::<half::f16>::new();
inputs.insert(x, DenseTensor::<half::f16>::from_vec([2, 3], values)?);
Executor::<half::f16>::new(graph)?.run(&ParameterStore::<half::f16>::new(), &inputs)?;
```

The model API, autograd training path and CUDA backend remain F32; there is no
autocast policy or loss scaling yet. Checkpoint loading already converts
F16/BF16 safetensors to F32.

## 14. Known limitations

- Structured *named* outputs (`out.hidden`) are not generated; use tuple
  destructuring.
- `grouped_query_attention` is single-batch; batch is the caller's loop.
- Tensor-dependent control flow is not representable (no runtime `Cond`).
- `forward_multi` called without destructuring expects exactly one output.
- `StreamingGemma` is shareable across sessions, but a compiled layer template
  is compiled per shape (position/length) under a mutex; very high concurrency
  with many distinct lengths still serializes compilation.
- `MixtureOfExperts` supports differentiable soft routing; sparse top-k expert
  selection remains the fused inference-only `Op::MoE`.
- Half-precision execution is low-level (graph API); the model/training API and
  CUDA are F32.
- PagedKV is host-side; device runtimes must upload the physical pools and block
  table.

## 15. Commands

```bash
cargo test --workspace
cargo test -p oxide-ml --test module_system --test kv_cache \
           --test transformer_components --test compile_fail
cargo run -p oxide-ml --example decoder_transformer
cargo run -p oxide-ml-gemma --bin gemma4_infer -- \
  --model .refs/gemma4-e2b-tiny --tokens 5,17,42 \
  --reference .refs/gemma4-e2b-tiny/reference.json --generate 3 --streaming
cargo bench -p oxide-ml --bench module_system
```
