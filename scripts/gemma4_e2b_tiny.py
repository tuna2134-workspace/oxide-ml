#!/usr/bin/env python3
"""Create a tiny Gemma 4 model with the E2B text features enabled.

The real `google/gemma-4-E2B-it` has per-layer input embeddings (PLE), 20 shared
KV layers and a double-wide MLP, but its text weights are ~9.3 GB BF16. This
script builds the same architecture at a tiny scale (random weights) so the Rust
implementation can be checked against `transformers` without the memory cost.

Usage: python scripts/gemma4_e2b_tiny.py OUT_DIR
"""

import json
import os
import sys

import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "gemma4-e2b-tiny"
    os.makedirs(out, exist_ok=True)
    config = Gemma4TextConfig(
        vocab_size=1024,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=2,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=1024,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
        enable_moe_block=False,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(config).eval()
    model.save_pretrained(out)

    input_ids = torch.tensor([[5, 17, 42]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1].float().cpu()
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=0,
        )[0].tolist()
    logits.numpy().astype("<f4").tofile(os.path.join(out, "ref_logits.f32"))
    values, indices = torch.topk(logits, 5)
    reference = {
        "input_ids": input_ids[0].tolist(),
        "logits_file": "ref_logits.f32",
        "generated": generated,
        "top5": [[int(i), float(v)] for v, i in zip(values, indices)],
        "tolerance": 2e-2,
    }
    with open(os.path.join(out, "reference.json"), "w", encoding="utf-8") as handle:
        json.dump(reference, handle, indent=1)
    print("params:", sum(p.numel() for p in model.parameters()))
    print("top5:", reference["top5"])
    print("generated:", generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
