# Migrating to the 0.1 model API

Version 0.1 keeps the original graph-builder API and adds the model API. This
guide maps old code to the new modules, `#[derive(Module)]`/`#[model]`, `Trainer`,
and checkpoints. Nothing described here requires deleting the old code: the
graph-builder layers remain under `oxide_ml::legacy` and `oxide_ml_nn::legacy`,
and the original trainer is `oxide_ml::GraphTrainer`.

## At a glance

| Original API | 0.1 model API |
| --- | --- |
| `legacy::Linear::new(&mut builder, prefix, in, out, bias)` | `Linear::new(in, out)` stored in a `#[derive(Module)]` struct. |
| `legacy::LayerNorm::new(&mut builder, prefix, shape, eps)` | `LayerNorm::new/with_shape/with_eps`. |
| `legacy::Module` (`forward(&mut GraphBuilder, ValueId)`) | Model `Module` (`forward(&mut ModelContext, ValueId)`). |
| `legacy::Mlp::new(&mut builder, ...)` | A user struct with fields; the derive builds traversal. |
| `legacy::Trainer` (alias of `GraphTrainer`) | `oxide_ml_model::Trainer`. |
| `oxide_ml::GraphTrainer` | Still available for low-level `TrainingGraph` loops. |
| `collect_gradients` (generic) | `collect_graph_gradients` (generic); the new `collect_gradients` is f32. |
| `oxide_ml_core::Tensor<R, T>` | `oxide_ml::RankedTensor` (same type, renamed re-export). |
| `Tensor<T>` in the prelude | Host-resident model-API tensor. |
| `init_parameters(&graph, seed)` | `CompiledModel::parameter_store(&model)` (and `init_parameters` still in `legacy`). |
| Manual `ExecutionInputs` maps | `Dataset` + `Trainer`, or `CompiledModel::run` with named inputs. |
| `ParameterId`-keyed state | Stable names in `ParameterStore` and `Checkpoint`. |

## Before: graph builder

```rust
use oxide_ml::prelude::*;
use oxide_ml::legacy::{Mlp, Trainer};

fn main() -> Result<()> {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 2]), DType::F32);
    let target = builder.input("y", Shape::from_static([4, 1]), DType::F32);
    let mlp = Mlp::new(&mut builder, "mlp", &[2, 8, 1], ActivationKind::Tanh, true)?;
    let logits = mlp.forward(&mut builder, x)?;
    let prediction = builder.sigmoid(logits)?;
    let loss = builder.mse_loss(prediction, target, Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;

    let training = build_training_graph(&forward, loss)?;
    let parameters = init_parameters::<f32>(&forward, 42)?;
    let optimizer = AdamW::with_config(0.08, 0.9, 0.999, 1e-8, 0.0)?;
    let mut trainer = Trainer::new(training, parameters, Box::new(optimizer))?;

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.])?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([4, 1], vec![0., 1., 1., 0.])?,
    );
    for _ in 0..1200 {
        trainer.step(&inputs)?;
    }
    Ok(())
}
```

## After: modules and `Trainer`

```rust
use oxide_ml::prelude::*;

#[derive(Module)]
struct XorMlp {
    fc1: Linear,
    fc2: Linear,
}

impl XorMlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = tanh(self.fc1.forward(x));
        self.fc2.forward(hidden)
    }
}

fn main() -> Result<()> {
    let inputs = Tensor::<f32>::from_vec(
        [4, 2],
        vec![0., 0., 0., 1., 1., 0., 1., 1.],
    )?;
    let targets = Tensor::<f32>::from_vec([4, 1], vec![0., 1., 1., 0.])?;
    let dataset = TensorDataset::from_tensors(inputs, targets)?;

    let mut model = XorMlp {
        fc1: Linear::with_seed(2, 8, true, 21),
        fc2: Linear::with_seed(8, 1, true, 22),
    };
    let mut trainer = Trainer::builder()
        .optimizer(AdamW::new(0.05)?)
        .loss(MSELoss::new())
        .batch_size(4)
        .build()?;
    let report = trainer.fit(&mut model, &dataset, 1200)?;
    assert!(report.final_loss < 0.05);
    Ok(())
}
```

## Migration steps

1. **Keep or replace the import.** `use oxide_ml::prelude::*;` is still the
   recommended prelude. Old graph-builder layers are *not* glob-exported there;
   import them explicitly with `use oxide_ml::legacy::{Linear as BuilderLinear,
   Mlp as BuilderMlp, Trainer as GraphTrainer};`.
2. **Change layers from handles to owners.** A builder layer stored
   `ValueId`s; the new `Linear`, `LayerNorm`, `Conv2d`, and friends own
   `DenseTensor<f32>` values. Construct them without a `GraphBuilder`.
3. **Store modules in a struct and derive `Module`.** Field names become
   parameter path segments. Use `#[module(skip)]` for configuration values.
4. **Replace the forward method.** Return a tensor (or tuple) instead of a
   `ValueId`. Call `self.fc.forward(x)`; the macro supplies the
   `&mut ModelContext`. Use `forward_multi` for multi-input modules such as
   `Attention`, and `apply_kernel` for registered custom kernels.
5. **Replace manual input maps with a `Dataset`.** `TensorDataset` stores
   tensors with a leading sample dimension; `Trainer` adds the batch dimension
   and calls `collate`.
6. **Replace the hand-written loop.** `Trainer::fit` compiles once, iterates
   batches, steps the optimizer, and optionally writes parameters back. Use
   `CompiledModel::compile_training` + `run_training` for a custom loop.
7. **Replace `ParameterId` bookkeeping.** Compile a model and use
   `parameter_store`, `parameter_ids`, and `Checkpoint`; names are stable across
   graph rebuilds.
8. **Rename generic gradient collection.** Calls to the old generic
   `collect_gradients::<T>` become `collect_graph_gradients::<T>`.

## Naming and compatibility notes

- `oxide_ml::legacy::Trainer` is the original `GraphTrainer`, not the new
  `Trainer`. Import one or the other explicitly to avoid ambiguity.
- `legacy::Linear` and the model `Linear` share a name. Prefer aliases in mixed
  code, for example `BuilderLinear`.
- `oxide_ml_core::Tensor<R, T>` is re-exported as `RankedTensor`. The prelude's
  `Tensor<T>` is the host tensor used by datasets, checkpoints, and inference.
  Code that relied on the old `Tensor` name for a statically ranked tensor must
  switch to `RankedTensor` or import `oxide_ml_core::Tensor` directly.
- `legacy::init_parameters` and `legacy::parameter_ids` still exist and remain
  re-exported from the prelude/`legacy` module; new code should prefer a
  `CompiledModel` parameter store.
- `graph!` is facade-only and uses `::oxide_ml::__private`; it cannot be used
  from a crate that renames the facade dependency. See
  [proc-macros.md](proc-macros.md).
- Graph-builder `collect_gradients` was generic over `Scalar`; the new
  `collect_gradients` is the f32 model-API helper. Existing generic calls must
  use `collect_graph_gradients`.

## Limitations

- There is no automatic source migration. The two APIs coexist, so large
  codebases can move module by module.
- The model API is F32-only; the low-level graph API remains generic over
  `Scalar`. Porting an `F64` graph to modules requires the F32 path or staying on
  the graph API.
- The model API does not reproduce every graph-builder feature: `Conv3d` is
  unsupported, dropout is not stochastic, and recurrent layers return only the
  last hidden state. See [known-limitations.md](known-limitations.md).
- Old `ParameterStore` values are keyed by `ParameterId`, whose order depends on
  graph construction. Checkpoints use names instead; do not migrate IDs.
- The CUDA crate is a standalone workspace. Its model APIs were compiled and
  run on a Tesla T4 and an A100 for this milestone; see
  [verification-report.md](verification-report.md).


## Module-system and macro redesign (same 0.1 line)

This revision keeps the 0.1 model API but makes module collections, enum
variants and tied parameters first-class. It is source-compatible for ordinary
`#[derive(Module)]` structs; the items below are the intentional changes.

| Change | Migration |
| --- | --- |
| `ModuleList<M>` added (`ModuleList` in the prelude) | Replace `Vec<M>` fields when you want indexed module paths, `get`/`insert`/`remove`, or `IntoIterator`. `Vec<M>` still implements `Module`. |
| `#[derive(Module)]` accepts enums | No change for structs. An enum variant forwards to its only module field or to `#[module(body)]`. |
| `#[model]` rewrites module-collection loops | `for layer in &self.layers { layer.forward(x) }` now works; previously this required `self.<field>` chains. A local created by `let layer = self.fc;` is still rejected. |
| Linked E2B layers | `Gemma4Text::layers` is `ModuleList<GemmaLayer>` (was `Vec<GemmaLayer>`). `.iter()`, indexing and `len()` still work; use `.as_slice()`/`.into_vec()` when a `&[GemmaLayer]`/`Vec<GemmaLayer>` is required. |
| Tied LM head | `Gemma4Text::lm_head` is `TiedLinear<Embedding>`; `TiedHead` remains as a type alias. Construct it with `TiedLinear::flattening(embed.clone())` instead of a struct literal. |
| `Module::state_config` | New defaulted method returning `Result<Option<KvCacheConfig>>`; existing manual `Module` implementations need no change. |
| `KvCache` / `KvLayerConfig` / `ModelState` | New exports. Use them instead of ad-hoc `Vec<Option<(Vec<f32>, Vec<f32>)>>` session caches. |
| `TiedLinear` parameter write-back | A tied head cannot mutate the shared weight through its own path; the parameter store stays authoritative (same as `Arc<M>`). |
| Compile-fail suite | Adds `trybuild` as a dev-dependency of `oxide-ml` and `tests/ui/*.stderr` snapshots. Regenerate intentionally with `TRYBUILD=overwrite`. |

## Declarative-architecture revision

This revision moves reusable transformer math into `oxide-ml-nn`, adds
multi-input/multi-output module calls, tuple structs and explicit streaming
sessions. Existing struct models keep compiling.

| Change | Migration |
| --- | --- |
| `StreamingGemma::prefill` / `decode` take `&mut StreamingSession` | `let mut session = model.session()?; model.prefill(&mut session, &ids)?; model.decode(&mut session, token)?;` |
| `StreamingGemma::state()` / `reset_state()` | use `session.state()` / `session.reset()`; the model is now stateless |
| `GemmaMlp` is `oxide_ml_nn::GatedMlp` | construct with `GemmaMlp::geglu(gate, up, down)` instead of a struct literal |
| `RmsNorm` / `NoScaleRmsNorm` are `oxide_ml_nn` components | constructors are unchanged (`RmsNorm::new(weight, eps)`, `RmsNormNoScale { eps }`) |
| `.forward_multi(...)` without destructuring | requires exactly one output; use `let (a, b) = ...` or `.forward_multi(...)?` and index the result |
| `.forward(a, b, ...)` | now dispatches to `forward_multi` for multi-input modules |
| Tuple structs | accepted by `#[derive(Module)]` with index paths (`0.weight`) |
| New components | `RmsNorm`, `RmsNormNoScale`, `GatedMlp`, `GatedActivation`, `RopeConfig`, `apply_rope(_at)`, `causal_mask`, `GqaConfig`, `grouped_query_attention`, `StreamingSession` |
| CUDA KV-cache runner | `gemma_e2b_kvcache` uses the session API (`stream.session()`, `session.state()`) |

## Runtime, precision and MoE revision

| Change | Migration |
| --- | --- |
| `StreamingGemma::prefill` / `decode` now take `&self` | callers with `let mut model` still compile; `Arc<StreamingGemma>` can be shared across threads, one `StreamingSession` per thread |
| `StreamingSession` is the only mutable streaming state | unchanged for existing session callers |
| `PagedKvCache` / `PagedKvConfig` | new; use instead of `KvCache` when block-level allocation or explicit block tables are required |
| `half::f16` / `half::bf16` implement `Scalar` | build graphs with `DType::F16`/`BF16` and run `Executor::<f16>::new`; the model API remains F32 |
| `MixtureOfExperts` / `MoeRouting` | new generic trainable MoE with soft routing over any expert modules; the fused sparse `Op::MoE` is unchanged and inference-only |
| `oxide_ml::half` | re-export for constructing half-precision tensors |

## Declarative Gemma layers

- `GemmaAttention` and `GemmaLayer` now derive `Module`; their prefill math is a
  declarative `#[model]` method. `Gemma4Text` also derives `Module`.
- Optional state uses the framework slot extension: `Module::forward_slots`,
  `Option<Tensor<f32>>` parameters and `#[oxide(slots)]`. Callers that already
  use `apply_with`/`decode_with` keep working.
- MoE parameters are now scoped under the module (`layers.N.moe.router.*`,
  `experts.*`, `expert_norm`) instead of sharing the layer namespace. The
  safetensors loader maps the checkpoint names, so weights and logits are
  unchanged; only parameter metadata/checkpoint keys change.
- `Gemma4Text::training` was removed; the derive propagates `set_training` to
  children and `is_training` uses the trait default.
