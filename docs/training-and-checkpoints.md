# Training and checkpoints

`Trainer` owns the loop: dataset batching, a `CompiledModel` training graph,
optimizer steps, evaluation, and checkpoint save/restore. For custom loops use
`CompiledModel` directly; the parameter store is authoritative in both cases.

## Train an XOR MLP

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
        .scheduler(StepLr::new(400, 0.5)?)
        .build()?;

    let report = trainer.fit(&mut model, &dataset, 1200)?;
    println!("final loss: {:.6}", report.final_loss);
    assert!(report.final_loss < 0.05);
    Ok(())
}
```

## Datasets and batching

`Dataset` requires `len`, `is_empty` (defaulted), `input_specs`, `target_specs`,
and `get(index) -> Sample`. The specs describe **one sample** and must not
include the batch dimension; `Trainer` prepends a symbolic batch dimension when
it compiles the graph. `Sample { inputs, targets }` holds one tensor per input
and target.

`TensorDataset::new(Vec<Tensor>, Vec<Tensor>)` (or `from_tensors(input, target)`)
expects every tensor to have a leading sample dimension of the same length. A
`[N, 4]` input yields `[4]` samples; a `[N]` target yields scalar targets.

`DataLoader::new(batch_size)?` rejects zero. `.shuffle(seed)` enables a
deterministic shuffle (seed plus epoch); `.drop_last(bool)` discards an incomplete
last batch. `.batches(samples, epoch)` returns `Vec<Vec<usize>>`, and
`collate(&[Sample])` stacks a batch into `(Vec<Tensor>, Vec<Tensor>)`.

`Trainer` supports exactly one target tensor. It also rejects an empty dataset.

## Trainer configuration

| Builder method | Default | Notes |
| --- | --- | --- |
| `optimizer` | none | Required by `build()`. |
| `loss` | none | Required by `build()`. |
| `device` | `Device::Cpu` | `Device::Cuda` is rejected by this CPU trainer. |
| `batch_size` | 32 | Values below 1 are ignored by the builder. |
| `shuffle` | `Some(0)` | `None` disables shuffling; a seed makes it deterministic. |
| `drop_last` | `false` | Drop the final partial batch. |
| `sync_model` | `true` | Copy updated parameters into the module after each step. |
| `scheduler` | none | `LrScheduler`, evaluated once per epoch. |
| `custom_kernels` | none | `Arc<CustomKernelRegistry<f32>>` for `Op::Custom`. |
| `accumulation_steps` | 1 | Micro-batches accumulated before one optimizer step. |

`Trainer::new()` is an unconfigured trainer; `fit` fails until an optimizer and
loss are supplied. `Trainer::builder().build()` validates them up front.
`Trainer` also exposes the same chain methods plus `optimizer_ref`,
`optimizer_mut`, and `with_parts`.

Gradient accumulation is enabled with `.accumulation_steps(n)`: gradients from
`n` micro-batches are summed, averaged, and applied once. A partial group at the
end of an epoch is flushed with the same averaging, so the effective batch size
is `batch_size * n` without changing the graph.

`fit(&mut model, &dyn Dataset, epochs)` compiles the training graph once, builds
a `ParameterStore`, iterates batches, calls `run_training`, applies the
optimizer, and writes parameters back. `sync_model(false)` keeps the store
authoritative; a final write-back still occurs. `StepLr::new(step_size, gamma)`
multiplies the base learning rate by `gamma` every `step_size` epochs.

`TrainReport` contains:

| Field | Meaning |
| --- | --- |
| `epochs` | One `EpochReport { epoch, loss, steps, samples }` per epoch. |
| `steps`, `samples` | Totals over all epochs. |
| `final_loss` | Sample-weighted mean loss of the last epoch. |
| `parameters`, `trainable_parameters` | Counts from the compiled graph. |
| `write_back` | Last `WriteBackReport` (`updated`, `skipped_aliases`, `unexpected`). |

`TrainReport::loss()` returns `final_loss`.

`Trainer::evaluate(&model, &dataset)` compiles a forward+loss graph without
backward, runs the dataset, and returns the sample-weighted mean loss. It needs
a loss configured on the trainer.

## Training without the Trainer

```rust
let mut options = ModelCompileOptions::training();
options.optimize = true;
options.memory_plan = true;
let compiled = CompiledModel::compile_training(
    &model,
    &[InputSpec::unbatched("input0", [4, 2])],
    InputSpec::unbatched("target0", [4, 1]),
    &MSELoss::new(),
    options,
)?;
let mut store = compiled.parameter_store(&model)?;
let mut optimizer = AdamW::new(0.01)?;

let x = Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.])?;
let y = Tensor::<f32>::from_vec([4, 1], vec![0., 1., 1., 0.])?;
for _ in 0..100 {
    let output = compiled.run_training(&store, &[("input0", &x), ("target0", &y)])?;
    optimizer.step(&mut store, &output.gradients)?;
}
compiled.write_back(&mut model, &store)?;
```

`run_training` returns `TrainingOutput { loss, gradients, outputs }`; gradients
are keyed by `ParameterId`. `evaluate` returns the scalar loss for a graph
compiled with `compile_loss` or `compile_training`.

## Checkpoints

`Checkpoint::from_model(&model)?` captures every parameter as a
`CheckpointParameter { name, shape, dtype, role, data: Vec<f32> }`, sorted by
stable name. The serialized shape is:

```json
{
  "format": "oxide-ml.checkpoint",
  "version": 1,
  "model_type": "XorMlp",
  "parameters": [
    { "name": "fc1.bias", "shape": [8], "dtype": "f32", "role": "trainable", "data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
  ],
  "optimizer": { "kind": "adamw", "step": 20, "entries": {} },
  "step": 20,
  "epoch": 0,
  "metadata": {}
}
```

API:

| Call | Purpose |
| --- | --- |
| `Checkpoint::from_model(&model)` | Snapshot parameters and `model_type`. |
| `.with_optimizer(&optimizer, &store)` | Attach `optimizer.state_dict(&store)`. |
| `.save(path)`, `Checkpoint::load(path)` | JSON file round trip. |
| `.to_json()`, `Checkpoint::from_json(&str)` | In-memory round trip. |
| `.apply(&mut model)` | Copy parameters into a model; returns `LoadReport`. |
| `save_model(path, &model)`, `load_model(path, &mut model)` | Parameter-only helpers. |

`Checkpoint::validate` rejects an unknown format, an unsupported version,
duplicate parameter names, and a parameter whose `data.len()` does not equal the
product of its shape. `apply` validates every matching name and shape before
mutating any tensor, so a partially applied checkpoint is impossible; missing
names are reported in the `LoadReport` instead of aborting. Malformed JSON is
rejected by `from_json`.

`LoadReport { loaded, missing_in_model, missing_in_checkpoint, model_type_matches }`
reports differences; `is_exact()` is true only when both missing lists are
empty. A model-type mismatch is reported but does not abort the load.

`Trainer::save_checkpoint(path, model, &store, step)` snapshots the model,
attaches optimizer state from the store, sets `step`, and saves.
`Trainer::load_checkpoint(path, &mut model, &store)` applies the checkpoint to
the model and restores optimizer state. The supplied store is used to resolve
parameter names to ids; it is **not** updated with the restored parameter
values, so rebuild the store from the restored model before continuing:

```rust
let compiled = CompiledModel::compile(
    &model,
    &[InputSpec::unbatched("input0", [4, 2])],
    ModelCompileOptions::default(),
)?;
let store = compiled.parameter_store(&model)?;
trainer.save_checkpoint("run.json", &model, &store, 20)?;

let checkpoint = Checkpoint::load("run.json")?;
let report = checkpoint.apply(&mut model)?;
assert!(report.is_exact());
let store = compiled.parameter_store(&model)?;
if let Some(state) = &checkpoint.optimizer {
    trainer.optimizer_mut().unwrap().load_state_dict(&store, state)?;
}
```

Optimizer state is keyed by parameter **name**, never `ParameterId`, so it
survives graph rebuilds:

| Optimizer | `kind` | Slots |
| --- | --- | --- |
| `Sgd` | `"sgd"` | none |
| `SgdMomentum` | `"sgd_momentum"` | `velocity` |
| `AdamW` | `"adamw"` | `m`, `v`, plus `step` |
| `RmsProp` | `"rmsprop"` | `square_avg`, `momentum_buffer` |

Loading state into a different optimizer kind is rejected. Values are stored as
`f64` and restored into the target scalar type.

## Limitations

- The CPU `Trainer` rejects `Device::Cuda`; use the CUDA crate's trainer/model.
- `Trainer` and `evaluate` support exactly one target tensor.
- Checkpoints are JSON and store raw `f32` data; they are not a binary or
  memory-mapped format.
- Optimizer state restores slots and step, not learning-rate or other
  hyperparameters; rebuild the optimizer with the desired configuration.
- BatchNorm running statistics are saved but are never updated by training.
- Restoring optimizer state requires a `ParameterStore` built from the same
  parameter names as the checkpoint.
- The model API and this training path are F32-only.
