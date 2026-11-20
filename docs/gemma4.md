# Gemma 4 text inference

oxide-ml can load HuggingFace Gemma 4 checkpoints through the `safetensors`
crate and execute the text decoder with its own graph compiler and CPU
executor. The official `google/gemma-4-E2B-it` text model is too large for a
full f32 parameter store, so it runs through a **layer-streamed** path: one
decoder layer is materialized, executed and freed at a time. This page records
what is supported and what was verified, including the real E2B checkpoint.

## Components

| Crate | Role |
|---|---|
| `oxide-ml-safetensors` | memory-mapped safetensors reader; F32/F16/BF16 → host f32 |
| `oxide-ml-gemma` | Gemma 4 text config, weight mapping, modules, `gemma4_infer` CLI |

The model is written with the model API (`Module`, `Linear`, `Embedding`,
`ModelContext`) and compiled with `CompiledModel`; no separate interpreter is
used.

## Macro usage

The Gemma modules use the framework macros where they fit, and hand-written
`Module` implementations only where the macro language cannot express the
structure:

| Component | Implementation |
|---|---|
| `RmsNorm` | `#[derive(Module)]` with `#[param] weight`; the reduction is a custom `forward_multi` because the last axis is rank-dependent |
| `NoScaleRmsNorm` (V-norm) | hand-written (no parameters) |
| `GemmaMlp` | `#[derive(Module)]` + `#[model] forward` (`gate.forward(x).gelu() * up.forward(x)`, then `down`) |
| per-head attention | `#[model] fn head_attention(q, k, v, mask)` — 4 inputs, `matmul_t` + mask + `softmax` + `matmul` |
| `GemmaAttention` | manual: GQA repeat, shared-KV selection, per-head loop, RoPE with `Slice`/`Concat` |
| `GemmaLayer` | manual: PLE residual block and shared K/V plumbing |
| `Gemma4Text` | manual `Module` with `ModuleList<GemmaLayer>` for the decoder stack, PLE inputs and shared-state map |
| `GemmaMoE` | manual module + fused `Op::MoE`: router (RMSNorm, projection, softmax, top-k, per-expert scale) and packed experts |
| `TiedHead` | `TiedLinear<Embedding>` (`flattening`): registers `lm_head.weight` as an alias of `embed_tokens.weight` |
| `LayerRunner` (streaming) | manual: module wrapper with multi-input/multi-output for per-layer compilation |

`#[model]` now supports loops over module collections, but the remaining glue
is explicit because it needs shared-KV selection, PLE residuals and
rank-dependent head math rather than a plain module chain. The macro-based and
streaming paths produce the same logits (`3.427e-7` on the E2B-feature tiny
model), and streaming state now uses the framework `KvCache`.

## Supported architecture

The text decoder supports the dense and E2B feature set:

- token embedding with `sqrt(hidden_size)` scaling, tied LM head;
  the tied head is `TiedLinear<Embedding>` and produces one canonical
  `embed_tokens.weight` parameter with `lm_head.weight` as an alias;
- RMSNorm (input / post-attention / pre-FF / post-FF / final);
- GQA attention with per-layer head dimension, QK RMSNorm, V RMSNorm without
  scale;
- per-layer RoPE: sliding layers use default RoPE (theta 10,000), full layers
  use Gemma 4 proportional RoPE (theta 1,000,000, partial factor 0.25);
- alternating sliding/full causal masks (sliding window from config);
- GeGLU MLP (`gelu_pytorch_tanh`);
- per-layer `layer_scalar`;
- final logit soft-capping (`tanh(logits/cap) * cap`).

Also implemented:

- **per-layer input embeddings (PLE)**: packed embedding lookup, context
  projection, RMSNorm and per-layer gate/projection;
- **shared KV layers**: the last `num_kv_shared_layers` layers reuse K/V from
  the last non-shared layer of the same attention type;
- **double-wide MLP** on shared-KV layers;
- **MoE text blocks**: router RMSNorm/projection/softmax/top-k with
  per-expert scales and packed expert tensors, executed by the fused `Op::MoE`
  kernel on CPU and CUDA (verified on a tiny 4-expert/top-2 checkpoint);
- a Rust tokenizer (`--prompt`);
- a reusable KV cache for layer streaming (`prefill`/`decode`, shared-KV links);
- CUDA execution of the full text decoder (A100 40 GB, f32).

Known limits:

- batch size other than 1;
- sequence lengths must be static; the graph path recompiles per length;
- no KV cache in the CUDA graph path (generation re-runs the prompt);
- MoE is inference-only (no autograd rule, no `TopK`/gather op);
- no image/audio towers: this is the text decoder only.

## Verified result

Model: `yujiepan/gemma-4-dense-tiny-random` (dense Gemma 4 text, hidden 8,
4 layers, 8/4 heads, head_dim 32/64, vocab 262144). Reference produced by
`transformers` 5.22.1 (`Gemma4ForConditionalGeneration`, float32, CPU).

| Check | Result |
|---|---|
| Safetensors weight mapping | 87 text tensors, 2,136,588 parameters |
| Prompt token `[9259]` top-5 | identical ids and values |
| Last-position logits | max abs diff **2.682e-7** |
| Greedy generation (4 tokens) | `[186294, 51247, 59626, 246881]`, identical |
| Forward | PASS |
| Backward/training | not implemented for Gemma 4 |

### E2B-feature verification

Model: a tiny `Gemma4ForCausalLM` built by
[scripts/gemma4_e2b_tiny.py](../scripts/gemma4_e2b_tiny.py) with the **E2B text
feature set**: PLE (`hidden_size_per_layer_input=8`), 2 shared-KV layers,
double-wide MLP, sliding window 8, per-layer head dim 8/16, proportional RoPE.
103,156 parameters, input ids `[5, 17, 42]`.

| Check | Result |
|---|---|
| Last-position logits vs transformers 5.22.1 | max abs diff **2.999e-7** |
| Greedy generation (4 tokens) | `[224, 397, 812, 508]`, identical |
| PLE / shared KV / double-wide execution | PASS |
| MoE text block (tiny 4-expert, top-k 2) | max diff **1.639e-7**, exact top-5 and greedy token |

Evidence: [gemma4-e2b-tiny.log](gemma4-e2b-tiny.log),
[gemma4-e2b-tiny-reference.json](gemma4-e2b-tiny-reference.json).

This is the architecture used by `google/gemma-4-E2B-it`; only the weight size
and dtype remain between this test and the official checkpoint.

The **official E2B config** parses through the same code:

```text
$ cargo run -p oxide-ml-gemma --bin gemma4_infer -- --model models/gemma-4-E2B-it --inspect
hidden_size: 1536
num_hidden_layers: 35
heads: 8 q / 1 kv
head_dim: 256 sliding / Some(512) full
layers: 28 sliding, 7 full, 20 shared-KV
PLE dim: 256
double-wide MLP: true
vocab: 262144
softcap: Some(30.0)
```

Evidence (dense): [gemma4-tiny-reference.json](gemma4-tiny-reference.json) and
[t4-gemma_rust3.log](t4-gemma_rust3.log).

## Full `google/gemma-4-E2B-it`: PASS (layer-streamed CPU)

The official checkpoint is multimodal and its text weights are:

| Measurement | Value |
|---|---|
| `model.safetensors` total | 10,246,621,918 bytes |
| text tensors (`model.language_model.*`) | **9,294,899,782 bytes BF16** |
| text parameters | ≈ 4.65 B |
| text weights as F32 | ≈ 18.6 GB |
| T4 device memory | 15.36 GB |
| Colab host RAM | 12 GB |

A whole-model f32 store cannot fit, so `StreamingGemma` compiles **one decoder
layer at a time**, loads that layer's BF16 weights and converts them to f32,
runs the layer, and drops them. Token and PLE embedding rows are read directly
from the mmap (`SafeTensorsFile::row`), and the tied LM head is evaluated in
vocabulary chunks. The per-layer graphs are compiled once per
`(layer type, head dim, KV heads, shared/store, PLE, intermediate, length)` and
reused with different parameter stores.

Configuration actually used on the T4:

```text
hidden_size: 1536
num_hidden_layers: 35
heads: 8 q / 1 kv
head_dim: 256 sliding / 512 full
layers: 28 sliding, 7 full, 20 shared-KV
PLE dim: 256
double-wide MLP: true
vocab: 262144
softcap: Some(30.0)
```

Command:

```bash
cargo run --release -p oxide-ml-gemma --bin gemma4_infer -- \
    --model /content/models/gemma4-e2b --tokens 9259 --streaming
```

### Verification against `transformers` (fp16, T4)

The reference is `Gemma4ForConditionalGeneration` in float16 on the T4 (not
4-bit; 4-bit NF4 changed the ordering of near-tied logits and is not a valid
numeric oracle here).

Prompt `[9259]` ("Hello"):

| Rank | oxide-ml f32 | transformers fp16 |
|---:|---:|---:|
| 1 | 9259 (4.8616) | 9259 (4.8359) |
| 2 | 107 (1.5792) | 107 (1.5547) |
| 3 | 106 (0.5460) | 106 (0.5229) |
| 4 | 108 (−0.9506) | 108 (−0.9722) |
| 5 | 1 (−1.2513) | 1 (−1.2705) |

Identical top-5 and order; per-logit difference ≈ 0.01–0.03 (fp16 vs f32).

Chat-formatted prompt (`<start_of_turn>user\nHello<end_of_turn>\n...`, 26
token ids):

| Rank | oxide-ml f32 | transformers fp16 |
|---:|---:|---:|
| 1 | 9259 (8.4147) | 9259 (8.4297) |
| 2 | 236820 (7.3924) | 236820 (7.5313) |
| 3 | 1 (7.3432) | 1 (7.4453) |
| 4 | 106 (6.8196) | 106 (6.9219) |
| 5 | 954 (5.6937) | 954 (5.7930) |

Identical top-5 and order; per-logit difference ≈ 0.02–0.14. Greedy generation
for one token returns `9259` (`Hello`).

Measured resources: peak resident set **≈ 6.2 GB** (12 GB host, 10 GB mmap
checkpoint); no CUDA; roughly **1–2 minutes per forward** for the prompts above
on the Colab CPU. Generation recomputes the full sequence per token, so it is
correct but slow.

Evidence:
[gemma4-e2b-fp16-reference.log](gemma4-e2b-fp16-reference.log),
[gemma4-e2b-run.log](gemma4-e2b-run.log),
[gemma4-e2b-chat.log](gemma4-e2b-chat.log),
[gemma4-e2b-memory.log](gemma4-e2b-memory.log).

The layer-streamed path matches the graph path on the tiny E2B-feature model
(`3.427e-7`): [gemma4-e2b-tiny-streaming.log](gemma4-e2b-tiny-streaming.log).
The graph path itself now uses `#[derive(Module)]` for `RmsNorm`/`GemmaMlp` and
`#[model]` for the per-head attention math.

### Remaining E2B limitations

- The CPU layer-streamed path is minutes per forward; it is not a serving
  implementation. CUDA needs an A100-class device for the full f32 model and
  has no KV cache, so each generated token re-runs the prompt.
- No KV cache in the graph path (the layer-streamed path has one).
- MoE is inference-only and fused; expert-parallel execution is not available.
- Multimodal (image/audio) Gemma 4 pathways are not implemented; this is the
  text decoder.

## Usage

```bash
# 1. Fetch a checkpoint and create a HuggingFace reference (Python side).
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("yujiepan/gemma-4-dense-tiny-random", local_dir="gemma4-tiny")
PY
python3 scripts/gemma4_reference.py gemma4-tiny ref "Hello"

# 2. Run the Rust runtime with the reference's token ids (or --tokens 9259).
cargo run -p oxide-ml-gemma --bin gemma4_infer -- \
    --model gemma4-tiny --reference ref/reference.json --generate 4
```

Output (T4, CPU execution):

```text
weights prefix: "model.language_model."
model: hidden=8 layers=4 heads=8 kv=4 vocab=262144
parameters: 2136588
top-5 next tokens:
  186294    0.449792
  ...
reference max |diff|: 2.682e-7 (tolerance 1.0e-2)
generated tokens: [186294, 51247, 59626, 246881]
```

The CLI now includes a Rust tokenizer (`tokenizers` 0.22) for plain text:
`--prompt "Hello"` reads `model/tokenizer.json`, prints the token ids, and
`--tokenize` round-trips them. The Python reference path still passes explicit
`input_ids`, which keeps numeric verification independent of tokenization.

## CUDA status: NOT RUN for Gemma 4

The Gemma graph needs `Embedding`, `Slice` and `Concat` on CUDA (currently
CPU-only), and its `Linear` calls use rank-3 inputs while the CUDA `MatMul`
supports rank 2. A CPU/CUDA parity run for the tiny checkpoint is therefore
**NOT RUN**; it is the next step after adding those kernels (or reshaping every
projection to rank 2).
