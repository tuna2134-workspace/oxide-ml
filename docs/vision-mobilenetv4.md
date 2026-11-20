# MobileNetV4 and the cats-vs-dogs example

`oxide-ml-nn` implements a MobileNetV4-style convolutional backbone built from
Universal Inverted Bottleneck (UIB) blocks. The model is fully declarative:
`ConvNormAct`, `UibBlock` and `MobileNetV4` derive `Module` and express their
forward pass with `#[model]`; parameters are discovered by the module tree, not
registered by hand.

## Architecture

| Piece | Implementation |
| --- | --- |
| `ConvNorm` | Conv2d (no bias) + BatchNorm, declarative `#[model] forward` |
| `ConvNormAct` | `ConvNorm` + ReLU/GELU/identity |
| `UibBlock` | optional start depthwise, optional 1x1 expansion, middle depthwise, optional SE, 1x1 projection, residual when stride 1 and channels match |
| `SqueezeExcite` | global average pool → Linear → ReLU → Linear → sigmoid → channel scale |
| `GlobalAvgPool` | static-shape average pool + reshape to `[N, C]` |
| `MobileNetV4` | stem, `ModuleList<MobileNetBlock>`, head, global pool, classifier |
| `MobileNetBlock` | enum of a plain conv block and a UIB block (declarative dispatch) |

`MobileNetV4Config::nano(classes)` is a compact table for CPU/Colab training;
`MobileNetV4Config::conv_small(classes)` is a MobileNetV4-Conv-Small-style table
(width 1.0). The published architecture also has hybrid variants with Mobile MQA
attention; that path is not implemented.

Depthwise convolutions use `Conv2d` with `groups = channels`; the graph
`Conv2d` op supports groups on CPU, and `Conv2d` allocates weights with the
correct `[out, in/groups, kH, kW]` shape.

## Data

`scripts/prepare_cats_dogs_parquet.py` converts the public Kaggle cats-vs-dogs
parquet shards into balanced raw RGB files. The two shards are sorted by class
(one class per shard), so the script samples both classes explicitly:

```bash
python scripts/prepare_cats_dogs_parquet.py --parquet s0.parquet s1.parquet \
  --out data/cats_dogs --count 8000 --size 128
```

Outputs: `train_images.u8`, `train_labels.u8`, `val_images.u8`,
`val_labels.u8`, `meta.json` (6,800/1,200 split for 8,000 images, 128×128 RGB,
balanced classes).

## Training

GPU training uses `scripts/train_mobilenetv4_cats_dogs.py`: a PyTorch mirror of
the exact `nano` module tree, names and operation order. It trains with AdamW +
cosine scheduling, label smoothing, horizontal flips, and exports the weights
with oxide-ml parameter names plus metadata. This is the GPU training path
because oxide-ml's CUDA backend does not yet implement Conv2d/BatchNorm
training (see [known-limitations.md](known-limitations.md)).

```bash
python scripts/train_mobilenetv4_cats_dogs.py \
  --data data/cats_dogs --arch conv_small --epochs 20 --batch 64 \
  --export data/cats_dogs/mobilenetv4_nano.safetensors
```

CPU training is available through `examples/cats_vs_dogs.rs`, but a full epoch is
impractical with the reference CPU kernels; it is useful for smoke tests and
small subsets. The example performs a static-batch training loop (CNN autograd
needs static shapes), writes the trained weights back, calibrates BatchNorm
running statistics with `calibrate_batch_norm`, evaluates, and exports
safetensors.

## Inference and verification

```bash
cargo run -p oxide-ml --release --example evaluate_pet -- \
  --data data/cats_dogs --model data/cats_dogs/mobilenetv4_nano.safetensors

cargo run -p oxide-ml --release --example classify_pet -- \
  --model data/cats_dogs/mobilenetv4_nano.safetensors --image cat.jpg
```

`evaluate_pet` loads the safetensors file into the Rust module tree, runs the
validation split and (optionally) compares the first eight logits with the
PyTorch reference JSON.

## Results

| Item | Value |
| --- | --- |
| Training device | Tesla T4 (Colab), PyTorch 2.14 cu130 |
| Architecture | `conv_small`, 890,474 parameters |
| Train / val images | 6,800 / 1,200 (128×128, balanced) |
| Epochs / batch | 20 / 64 (best epoch 19) |
| Validation accuracy (PyTorch) | **0.7042** |
| Held-out test accuracy (Rust, 100 images) | **0.7400** |
| Rust val subset (100 images) | 0.6700 |
| Rust ↔ PyTorch logits | max abs `2.879e-3` on 8 samples |
| Export | 173 tensors, 3.5 MiB, standard safetensors |
| Full log | [cats-dogs-gpu-training.log](cats-dogs-gpu-training.log) |

The 70% validation accuracy is for a from-scratch 0.9M-parameter model on 6,800
images; a pretrained ImageNet initialization was not used because the published
timm MobileNetV4 checkpoints do not match this module table exactly. External
photos are qualitative: golden retriever and Labrador photos classify as dogs,
while two close-up/collage cat photos were misclassified, so this checkpoint is
a demonstration, not a production pet classifier.

### Reproduce

```bash
# 1. download the two Kaggle shards (one class each) from
#    https://huggingface.co/datasets/Omriy123/Dogs_vs_Cats
# 2. prepare a balanced raw dataset
python scripts/prepare_cats_dogs_parquet.py --parquet s0.parquet s1.parquet   --out data/cats_dogs --count 8000 --size 128
# 3. GPU (PyTorch mirror)
python scripts/train_mobilenetv4_cats_dogs.py --arch conv_small --data data/cats_dogs   --size 128 --epochs 20 --batch 64 --lr 0.0015   --export data/cats_dogs/mobilenetv4_conv_small.safetensors
# 4. Rust evaluation
cargo run -p oxide-ml --release --example evaluate_pet -- --data data/cats_dogs   --model data/cats_dogs/mobilenetv4_conv_small.safetensors --split test --limit 100
```

The earlier `nano` table (136,494 parameters, 96×96, 8,000 images) reached
0.6867 validation accuracy; `conv_small` at 128×128 is the recommended
checkpoint.

## Notes and limitations

- GPU training is PyTorch; oxide-ml's CUDA kernels cover inference primitives
  but not Conv2d/BatchNorm/CrossEntropy training.
- CPU MobileNetV4 forward/training works (tests and smoke training), but the
  reference conv kernels are slow.
- BatchNorm running statistics are calibrated after training with
  `calibrate_batch_norm` for CPU-trained models; PyTorch-trained checkpoints
  already contain running statistics.
- Hybrid MobileNetV4 (MQA attention) and ImageNet-scale channel tables are not
  implemented beyond `conv_small`.
