# MobileNetV4, GPU cats-vs-dogs training, and safetensors export

This report covers the vision and checkpoint deliverables added after the
declarative-framework work.

## 1. Safetensors writer and model export

New in `oxide-ml-safetensors`:

- `SafeTensorsWriter` with deterministic tensor ordering, 8-byte aligned data
  sections, padded header, standard metadata, duplicate/shape validation and
  atomic `write` (temp file + rename).
- `TensorRecord` constructors for F32/F16/BF16 plus raw-bytes tensors.
- The existing mmap reader now pairs with the writer; round-trip tests cover
  metadata, F32/F16/BF16 and malformed inputs.

Model-level helpers in `oxide-ml`:

- `save_module_safetensors(model, path)` — traverses the module tree, writes
  canonical parameters, records `role.*` metadata and tied-storage aliases.
- `save_module_safetensors_with_metadata(model, path, &[(key, value)])` — adds
  architecture/class/input-size metadata.
- `load_module_safetensors(&mut model, path)` — name/shape-checked load,
  resolves `alias.<name>` metadata, reports missing/unexpected tensors.
- `save_parameter_store_safetensors`, `module_tensors`, `verify_export`.

Verified by `tests/safetensors_export.rs` (MLP round trip, tied embedding/head
alias, parameter-store export) and `crates/oxide-ml-safetensors/tests/writer.rs`.
The cats-vs-dogs checkpoint is 173 tensors / 3.5 MiB and loads with zero missing
tensors.

## 2. BatchNorm calibration

The reference executor has no in-graph mutable state, so BatchNorm running
buffers were never updated. Added:

- `ModuleMode::Calibrate`;
- `ModelContext::probe` + graph probes retained by `CompiledModel` and read by
  `run_probes`;
- BatchNorm records its input in calibration mode (batch-stat normalization);
- `calibrate_batch_norm(model, specs, batches, momentum)` accumulates per-channel
  mean/variance from probes and applies EMA into `running_mean`/`running_var`.

Tests: `tests/batch_norm_calibration.rs` checks exact buffer updates (momentum
1.0) and eval consistency with batch-normalized training outputs.

## 3. MobileNetV4

New `oxide-ml-nn::vision`:

| Component | Notes |
| --- | --- |
| `ConvNorm`, `ConvNormAct` | Conv2d (no bias) + BatchNorm (+ ReLU/GELU) |
| `UibBlock` | optional start depthwise, optional expansion, middle depthwise, optional SE, 1x1 projection, residual |
| `SqueezeExcite` | global pool → Linear → ReLU → Linear → sigmoid → channel scale |
| `GlobalAvgPool` | static-shape average pool → `[N, C]` |
| `MobileNetBlock` | enum: plain conv block or UIB block |
| `MobileNetV4` | stem + `ModuleList` + head + pool + classifier, declarative `#[model]` forward |
| `MobileNetV4Config` | `nano(classes)` and `conv_small(classes)` tables |

Framework fixes required for CNNs:

- `Conv2d` allocates grouped/depthwise weights as `[out, in/groups, kH, kW]`
  (previously it assumed groups=1).
- Conv2d and pooling shape inference preserve a symbolic batch dimension
  (spatial dimensions remain static).
- BatchNorm/ReduceMean autograd still requires static shapes, so CNN training
  graphs use a static batch (documented).

Tests: `tests/vision_models.rs` (forward `[2,3,64,64] → [2,2]`, parameter count,
one training step with CrossEntropy + AdamW).

## 4. Cats vs dogs: data and GPU training

Data: `scripts/prepare_cats_dogs_parquet.py` builds a **balanced** raw dataset
from the Kaggle/HuggingFace parquet shards. This was essential: the shards are
sorted by class, so a naive prefix produced an all-cats dataset (the first
attempt reached “100%” by collapsing to one class).

GPU training: `scripts/train_mobilenetv4_cats_dogs.py` mirrors the Rust module
tree, parameter names and operation order in PyTorch, trains on a Tesla T4 and
exports safetensors with oxide-ml names and metadata. Oxide-ml's CUDA backend
does not yet implement Conv2d/BatchNorm/CrossEntropy training, so the training
step uses PyTorch while architecture, export, loading and inference remain
oxide-ml.

Run (T4):

- architecture `conv_small`, 890,474 parameters, 6,800 train / 1,200 val,
  128×128 balanced, batch 64, AdamW lr 1.5e-3, 20 epochs;
- augmentation: random crop/pad, horizontal flip, brightness/contrast, label
  smoothing 0.05;
- best validation accuracy **0.7042** (epoch 19); exported best checkpoint.

Full log: [cats-dogs-gpu-training.log](cats-dogs-gpu-training.log).

The earlier `nano` table (136,494 parameters, 96×96, 8,000 images) reached 0.6867
validation accuracy.

### Rust verification

```
checkpoint: mobilenetv4_conv_small.safetensors (173 tensors)
val subset (100): 0.6700
held-out test subset (100): 0.7400
PyTorch reference logits, 8 samples: max |diff| 2.879e-3
```

The held-out test split comes from rows not used for training (500 cats + 500
dogs at 128×128); the first 100 were evaluated in Rust. The logits difference is
consistent with independent f32 conv/BN implementations.

Qualitative external photos: golden retriever, Labrador and beagle-like photos
classify as dogs; two close-up/collage cat photos were misclassified. The model
is a from-scratch demonstration, not production-grade.

## 5. CPU inference performance

Release build, reference CPU kernels:

| Measurement | Result |
| --- | ---: |
| `conv_small` single image (128×128, process start incl. load) | 1.50 s |
| `conv_small` 100 images | 118 s (≈1.18 s/image) |

The reference convolution is correctness-oriented and not a tuned CPU/CUDA
implementation.

## 6. Limitations

- CUDA training for conv/BN/CrossEntropy is not implemented; GPU training uses
  the PyTorch mirror.
- Hybrid MobileNetV4 (Mobile MQA attention) is not implemented.
- Pretrained ImageNet initialization is not used; accuracy is from scratch.
- CNN training needs a static batch because `ReduceMean` autograd requires
  static shapes; `Trainer` cannot be used directly for BatchNorm CNNs.
- Running statistics for CPU-trained models require `calibrate_batch_norm`;
  PyTorch checkpoints already contain them.
- Safetensors writing assembles the file in memory before the atomic write;
  shared storage is written once and loading it through `visit_tensors_mut` is
  skipped.

## 7. Files

- `crates/oxide-ml-safetensors/src/writer.rs`, `tests/writer.rs`
- `crates/oxide-ml/src/export.rs`, `tests/safetensors_export.rs`
- `crates/oxide-ml-model/src/calibration.rs`, `context.rs`, `runtime.rs`
- `crates/oxide-ml-nn/src/vision.rs`, `modules.rs` (grouped conv)
- `crates/oxide-ml-graph/src/infer.rs` (symbolic batch for conv/pool)
- `examples/cats_vs_dogs.rs`, `classify_pet.rs`, `evaluate_pet.rs`
- `scripts/prepare_cats_dogs_parquet.py`,
  `scripts/train_mobilenetv4_cats_dogs.py`
- `tests/vision_models.rs`, `tests/batch_norm_calibration.rs`
- docs: `safetensors.md`, `vision-mobilenetv4.md`, this report, README,
  verification/benchmark/known-limitations updates

## 8. CUDA training support (follow-up)

The CUDA backend now implements the training ops used by MobileNetV4:
`Conv2d`, depthwise conv, `Conv2dBackward`, `CrossEntropyLoss`,
`CrossEntropyLossBackward`, and a corrected keep-dims `ReduceMean` mapping for
BatchNorm. The convolution is lowered as im2col + cuBLAS SGEMM, with direct
depthwise kernels.

- `vision_check`: eval forward `8.3e-17`, train-mode forward `2.0e-4`,
  BatchNorm `0.0`, one-step loss `6.6e-7`.
- `gradient_check`: one-step parameter updates agree to `~2e-3`; this also
  uncovered and fixed uninitialized `CudaTensor::zeros` (AdamW state).
- `train_mobilenet`: 1 epoch on a T4 = 125.4 s; PyTorch cuDNN = 10.22 s.
  oxide-ml is ~12.3× slower; cuBLAS replacing the tiled GEMM only improved
  131 s → 125 s, so im2col traffic and unfused layers dominate.
- Command:
  `cargo oxide build --arch sm_75 -- --bin train_mobilenet` then
  `./target/debug/train_mobilenet --data /path/to/cats_dogs --arch conv_small --size 128 --epochs 20 --batch 64 --export model.safetensors`.

## 9. CUDA retraining result (honest)

`train_mobilenet` (Rust module tree, CUDA ops, im2col + cuBLAS) was run on the same
6,800/1,200 dataset:

| Run | Epoch time | Train loss | Val accuracy |
| --- | ---: | ---: | ---: |
| 1 epoch, lr 1.5e-3 | 125.4 s | 0.707 | 0.525 |
| 3 epochs, lr 1.5e-3 | 125–132 s | 0.686 (epoch 3) | 0.525 |
| 3 epochs, lr 5e-3 | 126 s | 0.700 (epoch 3) | 0.525 |

The model did not converge to a usable cats/dogs classifier in these runs; the
0.525 validation accuracy equals the majority-class rate. The best checkpoint
remains the PyTorch-trained `conv_small` (0.7042). The CUDA ops are numerically
verified (`vision_check` / `gradient_check`), and the end-to-end training loop
runs, but the optimizer/feature learning behaviour and the 12× speed gap need
further work before the CUDA path replaces the PyTorch reference training.
