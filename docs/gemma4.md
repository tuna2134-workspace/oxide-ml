# Gemma 4 text inference

oxide-ml can load HuggingFace Gemma 4 checkpoints through the `safetensors`
crate and execute the text decoder with its own graph compiler and CPU
executor. This page records exactly what is supported, what was verified, and
what blocks full `google/gemma-4-E2B-it` inference on a T4.

## Components

| Crate | Role |
|---|---|
| `oxide-ml-safetensors` | memory-mapped safetensors reader; F32/F16/BF16 → host f32 |
| `oxide-ml-gemma` | Gemma 4 text config, weight mapping, modules, `gemma4_infer` CLI |

The model is written with the model API (`Module`, `Linear`, `Embedding`,
`ModelContext`) and compiled with `CompiledModel`; no separate interpreter is
used.

## Supported architecture

Dense Gemma 4 text (the `Gemma4ForCausalLM` / `gemma4_text` path):

- token embedding with `sqrt(hidden_size)` scaling, tied LM head;
- RMSNorm (input / post-attention / pre-FF / post-FF / final);
- GQA attention with per-layer head dimension, QK RMSNorm, V RMSNorm without
  scale;
- per-layer RoPE: sliding layers use default RoPE (theta 10,000), full layers
  use Gemma 4 proportional RoPE (theta 1,000,000, partial factor 0.25);
- alternating sliding/full causal masks (sliding window from config);
- GeGLU MLP (`gelu_pytorch_tanh`);
- per-layer `layer_scalar`;
- final logit soft-capping (`tanh(logits/cap) * cap`).

Unsupported (rejected by `Gemma4Config` with an explicit error):

- **per-layer input embeddings (PLE)**: packed embedding lookup, context
  projection, RMSNorm and per-layer gate/projection;
- **shared KV layers**: the last `num_kv_shared_layers` layers reuse K/V from
  the last non-shared layer of the same attention type;
- **double-wide MLP** on shared-KV layers;
- **MoE text blocks** (not used by E2B; rejected);
- batch size other than 1 and sequence lengths that are not static;
- a Rust tokenizer (token ids are produced by HuggingFace; see below);
- KV cache (each generated token recompiles and reruns the full sequence).

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

The tokenizer is intentionally out of scope for the Rust binary: the reference
script tokenizes with `AutoTokenizer` and stores `input_ids`; the Rust runtime
accepts those ids. This keeps the verification honest (same ids) and avoids a
second tokenizer implementation.

## Full `google/gemma-4-E2B-it` status: BLOCKED

The official E2B checkpoint is multimodal; its text weights alone are:

| Measurement | Value |
|---|---|
| `model.safetensors` total | 10,246,621,918 bytes |
| text tensors (`model.language_model.*`) | **9,294,899,782 bytes BF16** |
| text parameters | ≈ 4.65 B |
| text weights as F32 | **≈ 18.6 GB** |
| T4 device memory | 15.36 GB |
| Colab host RAM | 12 GB |

`oxide-ml-cuda` is F32-only, so loading the text model as host f32 does not fit
the T4 or the Colab host, and the graph parameter store would need a second
copy. In addition, the E2B text config enables features this implementation
does not support:

```text
hidden_size_per_layer_input = 256   (per-layer input embeddings)
num_kv_shared_layers       = 20    (shared K/V layers)
use_double_wide_mlp        = true  (wider MLP on shared-KV layers)
```

The architecture for all three features is implemented and verified, and the
official E2B config parses with the expected layer/head/PLE/shared-KV counts.
Running the official checkpoint therefore requires only:

1. BF16/F16 storage and kernels (or per-layer BF16→F32 conversion with
   weights streamed to the device and freed after the layer); the PLE embedding
   alone is `262144 x (35 x 256) = 2.35 B` parameters (~4.7 GB BF16);
2. a tokenizer (HuggingFace `tokenizer.json` via the `tokenizers` crate);
3. a KV cache for generation;
4. verification against `transformers` on the same T4.

The dense path above is the foundation for that work: the loader, RoPE variants,
GQA, RMSNorm and soft-capping are already exercised.

## CUDA status: NOT RUN for Gemma 4

The Gemma graph needs `Embedding`, `Slice` and `Concat` on CUDA (currently
CPU-only), and its `Linear` calls use rank-3 inputs while the CUDA `MatMul`
supports rank 2. A CPU/CUDA parity run for the tiny checkpoint is therefore
**NOT RUN**; it is the next step after adding those kernels (or reshaping every
projection to rank 2).
