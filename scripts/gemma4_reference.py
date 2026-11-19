#!/usr/bin/env python3
"""Create a HuggingFace reference for the Rust Gemma 4 runtime.

Usage: python scripts/gemma4_reference.py MODEL_DIR OUT_DIR [PROMPT]

Writes input_ids, last-position logits (raw f32) and greedy generation to
OUT_DIR/reference.json + OUT_DIR/logits.f32.
"""

import json
import os
import sys

import torch


def main() -> int:
    model_dir = sys.argv[1]
    out_dir = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Hello"
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print("input_ids:", input_ids.tolist())

    model = None
    errors = []
    for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            module = __import__("transformers", fromlist=[loader])
            cls = getattr(module, loader)
            model = cls.from_pretrained(model_dir, dtype=torch.float32)
            print("loaded with", loader, type(model).__name__)
            break
        except Exception as error:  # noqa: BLE001 - probe both loaders
            errors.append(f"{loader}: {error}")
    if model is None:
        raise RuntimeError("could not load model:\n" + "\n".join(errors))

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1].float().cpu()
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0].tolist()

    logits.numpy().astype("<f4").tofile(os.path.join(out_dir, "logits.f32"))
    top = torch.topk(logits, 5)
    reference = {
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "logits_file": "logits.f32",
        "vocab_size": int(logits.numel()),
        "top5": [[int(i), float(v)] for i, v in zip(top.indices, top.values)],
        "generated": generated,
        "tolerance": 2e-2,
    }
    with open(os.path.join(out_dir, "reference.json"), "w", encoding="utf-8") as handle:
        json.dump(reference, handle, indent=1)
    print("top5:", reference["top5"])
    print("generated:", generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
