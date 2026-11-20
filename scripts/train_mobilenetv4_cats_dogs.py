#!/usr/bin/env python3
"""Train the oxide-ml MobileNetV4 (`nano` table) on cats-vs-dogs and export.

The module tree mirrors `crates/oxide-ml-nn/src/vision.rs::MobileNetV4Config::nano`
exactly, including parameter names, so the resulting safetensors checkpoint is
loadable by `oxide_ml::load_module_safetensors`.

Prepare the balanced raw dataset first with
`scripts/prepare_cats_dogs_parquet.py` (the Kaggle parquet shards are sorted by
class, so a naive prefix is single-class).

This script is the GPU training path for the example model: oxide-ml's CUDA
backend does not yet implement Conv2d/BatchNorm training, so the training step
uses PyTorch on the GPU while the architecture, export format, loading and
inference remain oxide-ml.  See docs/vision-mobilenetv4.md.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Architecture mirror
# ---------------------------------------------------------------------------

class ConvNorm(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, groups):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, kernel // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvNormAct(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, groups):
        super().__init__()
        self.norm = ConvNorm(in_c, out_c, kernel, stride, groups)

    def forward(self, x):
        return F.relu(self.norm(x))


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.reduce = nn.Linear(channels, hidden)
        self.expand = nn.Linear(hidden, channels)

    def forward(self, x):
        n, c, h, w = x.shape
        pooled = F.avg_pool2d(x, (h, w)).reshape(n, c)
        gate = torch.sigmoid(self.expand(F.relu(self.reduce(pooled))))
        return x * gate.reshape(n, c, 1, 1)


class UibBlock(nn.Module):
    def __init__(self, in_c, out_c, expand_ratio, kernel, stride,
                 start_depthwise, se):
        super().__init__()
        hidden = in_c * max(expand_ratio, 1)
        if start_depthwise:
            self.start_dw = ConvNormAct(in_c, in_c, kernel, stride, in_c)
            middle_stride = 1
        else:
            self.start_dw = None
            middle_stride = stride
        if hidden != in_c:
            self.expand = ConvNormAct(in_c, hidden, 1, 1, 1)
        else:
            self.expand = None
        self.middle_dw = ConvNormAct(hidden, hidden, kernel, middle_stride, hidden)
        self.se = SqueezeExcite(hidden) if se else None
        self.project = ConvNorm(hidden, out_c, 1, 1, 1)
        self.residual = stride == 1 and in_c == out_c

    def forward(self, x):
        y = x
        if self.start_dw is not None:
            y = self.start_dw(y)
        if self.expand is not None:
            y = self.expand(y)
        y = self.middle_dw(y)
        if self.se is not None:
            y = self.se(y)
        y = self.project(y)
        return x + y if self.residual else y


class MobileNetV4Nano(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = ConvNormAct(3, 16, 3, 2, 1)
        specs = [
            ("conv", 16, 3, 1),
            ("uib", 24, 2, 3, 2, False, False),
            ("uib", 24, 2, 3, 1, False, False),
            ("uib", 32, 2, 5, 2, True, False),
            ("uib", 32, 2, 5, 1, True, True),
            ("uib", 48, 3, 3, 2, True, False),
            ("uib", 48, 3, 3, 1, True, True),
            ("uib", 64, 3, 5, 2, True, False),
            ("uib", 64, 3, 5, 1, True, True),
        ]
        blocks = []
        channels = 16
        for spec in specs:
            if spec[0] == "conv":
                _, out_c, kernel, stride = spec
                blocks.append(ConvNormAct(channels, out_c, kernel, stride, 1))
            else:
                _, out_c, ratio, kernel, stride, start_dw, se = spec
                blocks.append(UibBlock(channels, out_c, ratio, kernel, stride, start_dw, se))
            channels = out_c
        self.blocks = nn.ModuleList(blocks)
        self.head = ConvNormAct(channels, 96, 1, 1, 1)
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


def export_state(model):
    """Rename PyTorch parameters to oxide-ml module paths."""
    out = {}

    def conv_bn(prefix, conv, bn):
        out[f"{prefix}.conv.weight"] = conv.weight.detach().float().cpu().contiguous()
        for ours, theirs in (
            ("gamma", "weight"), ("beta", "bias"),
            ("running_mean", "running_mean"), ("running_var", "running_var"),
        ):
            out[f"{prefix}.bn.{ours}"] = getattr(bn, theirs).detach().float().cpu().contiguous()

    def conv_norm_act(prefix, module):
        conv_bn(f"{prefix}.norm", module.norm.conv, module.norm.bn)

    conv_norm_act("stem", model.stem)
    for index, block in enumerate(model.blocks):
        prefix = f"blocks.{index}.0"
        if isinstance(block, UibBlock):
            for name in ("start_dw", "expand", "middle_dw"):
                module = getattr(block, name)
                if module is not None:
                    conv_norm_act(f"{prefix}.{name}", module)
            conv_bn(f"{prefix}.project", block.project.conv, block.project.bn)
            if block.se is not None:
                out[f"{prefix}.se.reduce.weight"] = block.se.reduce.weight.detach().float().cpu().contiguous()
                out[f"{prefix}.se.reduce.bias"] = block.se.reduce.bias.detach().float().cpu().contiguous()
                out[f"{prefix}.se.expand.weight"] = block.se.expand.weight.detach().float().cpu().contiguous()
                out[f"{prefix}.se.expand.bias"] = block.se.expand.bias.detach().float().cpu().contiguous()
        else:
            conv_norm_act(prefix, block)
    conv_norm_act("head", model.head)
    out["classifier.weight"] = model.classifier.weight.detach().float().cpu().contiguous()
    out["classifier.bias"] = model.classifier.bias.detach().float().cpu().contiguous()
    return out


class MobileNetV4ConvSmall(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = ConvNormAct(3, 32, 3, 2, 1)
        specs = [
            ("conv", 32, 3, 1),
            ("uib", 64, 2, 3, 2, True, False),
            ("uib", 64, 2, 3, 1, True, True),
            ("uib", 96, 3, 3, 2, True, False),
            ("uib", 96, 3, 5, 1, True, True),
            ("uib", 128, 4, 3, 2, True, False),
            ("uib", 128, 4, 5, 1, True, True),
            ("uib", 160, 4, 3, 1, True, True),
        ]
        blocks = []
        channels = 32
        for spec in specs:
            if spec[0] == "conv":
                _, out_c, kernel, stride = spec
                blocks.append(ConvNormAct(channels, out_c, kernel, stride, 1))
            else:
                _, out_c, ratio, kernel, stride, start_dw, se = spec
                blocks.append(UibBlock(channels, out_c, ratio, kernel, stride, start_dw, se))
            channels = out_c
        self.blocks = nn.ModuleList(blocks)
        self.head = ConvNormAct(channels, 256, 1, 1, 1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


def build_model(arch, num_classes):
    if arch == "nano":
        return MobileNetV4Nano(num_classes)
    if arch == "conv_small":
        return MobileNetV4ConvSmall(num_classes)
    raise SystemExit(f"unknown arch {arch!r}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PetDataset(Dataset):
    def __init__(self, images, labels, size, augment):
        self.images = images
        self.labels = labels
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        import io
        per = 3 * self.size * self.size
        raw = self.images[index * per:(index + 1) * per]
        array = np.frombuffer(raw, dtype=np.uint8).reshape(3, self.size, self.size)
        array = array.astype(np.float32) / 255.0
        if self.augment:
            # Random translation (pad 4), horizontal flip, brightness/contrast.
            padded = np.pad(array, ((0, 0), (4, 4), (4, 4)), mode="edge")
            y = random.randrange(0, 9)
            x = random.randrange(0, 9)
            array = padded[:, y:y + self.size, x:x + self.size]
            if random.random() < 0.5:
                array = array[:, :, ::-1]
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.08, 0.08)
            array = np.clip(array * gain + bias, 0.0, 1.0)
        array = (array - MEAN[:, None, None]) / STD[:, None, None]
        return torch.from_numpy(np.ascontiguousarray(array)), int(self.labels[index])


def load_split(out, name):
    images = (Path(out) / f"{name}_images.u8").read_bytes()
    labels = np.frombuffer((Path(out) / f"{name}_labels.u8").read_bytes(), dtype=np.uint8)
    return images, labels


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cats_dogs")
    parser.add_argument("--arch", choices=["nano", "conv_small"], default="nano")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--export", default="data/cats_dogs/mobilenetv4_nano.safetensors")
    parser.add_argument("--reference", default="data/cats_dogs/val_reference.json")
    args = parser.parse_args()

    meta_path = Path(args.data) / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} is missing; prepare the balanced raw dataset first:\n"
            "  python scripts/prepare_cats_dogs_parquet.py --parquet shard0.parquet shard1.parquet "
            "--out data/cats_dogs --count 8000 --size 128"
        )
    meta = json.loads(meta_path.read_text())
    train_images, train_labels = load_split(args.data, "train")
    val_images, val_labels = load_split(args.data, "val")
    print(f"data: train={len(train_labels)} val={len(val_labels)} size={meta['size']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    torch.manual_seed(7)
    random.seed(7)
    model = build_model(args.arch, 2).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(
        PetDataset(train_images, train_labels, meta["size"], True),
        batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PetDataset(val_images, val_labels, meta["size"], False),
        batch_size=args.batch, shuffle=False, num_workers=2,
    )

    best = 0.0
    best_state = None
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        model.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        accuracy = val_correct / max(val_total, 1)
        if accuracy > best:
            best = accuracy
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{args.epochs}: loss {loss_sum / max(total, 1):.4f} "
              f"train {correct / max(total, 1):.4f} val {accuracy:.4f}"
              f"{' *' if accuracy >= best else ''}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best checkpoint from epoch {best_epoch} (val {best:.4f})")
    model.eval()
    state = export_state(model)
    metadata = {
        "format": "oxide-ml.safetensors",
        "version": "1",
        "module.type": "MobileNetV4",
        "model.architecture": f"MobileNetV4-{args.arch}",
        "model.classes": "cat,dog",
        "model.input_size": str(meta["size"]),
        "training.backend": "pytorch-gpu",
        "training.epochs": str(args.epochs),
        "training.val_accuracy": f"{best:.6f}",
        "training.best_epoch": str(best_epoch),
    }
    Path(args.export).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, args.export, metadata=metadata)
    print(f"exported {len(state)} tensors to {args.export} "
          f"({Path(args.export).stat().st_size / 1048576:.1f} MiB), best val {best:.4f}")

    # Reference logits for the first 8 validation samples (Rust parity check).
    with torch.no_grad():
        subset = PetDataset(val_images[:8 * 3 * meta["size"] ** 2],
                            val_labels[:8], meta["size"], False)
        images = torch.stack([subset[i][0] for i in range(len(subset))]).to(device)
        logits = model(images).cpu().numpy()
    reference = {
        "size": meta["size"],
        "labels": [int(value) for value in val_labels[:8]],
        "logits": logits.tolist(),
        "val_accuracy": best,
        "tensors": len(state),
    }
    Path(args.reference).write_text(json.dumps(reference))


if __name__ == "__main__":
    main()
