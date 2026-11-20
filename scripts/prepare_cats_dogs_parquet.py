#!/usr/bin/env python3
"""Convert cats-vs-dogs parquet shards to balanced raw RGB u8.

The Microsoft/Kaggle parquet shards are sorted by class (shard 0 is one class,
shard 1 the other), so taking the first N rows yields a single class. This
script collects a balanced sample per class across all given shards, shuffles
with a fixed seed and writes the raw files consumed by the Rust trainer.

Usage:
  python scripts/prepare_cats_dogs_parquet.py \
    --parquet data/cats_dogs/shard0.parquet data/cats_dogs/shard1.parquet \
    --out data/cats_dogs --count 3000 --size 96
"""
import argparse
import io
import json
import os
import random

import pyarrow.parquet as pq
from PIL import Image


def label_column(schema_names):
    for name in ("label", "labels", "target"):
        if name in schema_names:
            return name
    raise SystemExit(f"no label column in {schema_names}")


def collect(parquet_path, per_class, size, classes=2):
    file = pq.ParquetFile(parquet_path)
    column = label_column([field.name for field in file.schema_arrow])
    buckets = {index: [] for index in range(classes)}
    for group in range(file.metadata.num_row_groups):
        table = file.read_row_group(group, columns=["image", column])
        for row in table.to_pylist():
            label = int(row[column])
            if label < 0 or label >= classes:
                continue
            if len(buckets[label]) >= per_class:
                continue
            try:
                image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                image = image.resize((size, size), Image.LANCZOS)
                buckets[label].append((image.tobytes(), label))
            except Exception:  # noqa: BLE001
                continue
        if all(len(bucket) >= per_class for bucket in buckets.values()):
            break
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--out", default="data/cats_dogs")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    per_class = args.count // 2
    samples = []
    for path in args.parquet:
        print(f"reading {path}...", flush=True)
        buckets = collect(path, per_class, args.size)
        for bucket in buckets.values():
            samples.extend(bucket)
    random.Random(args.seed).shuffle(samples)
    val_count = int(len(samples) * args.val_fraction)
    splits = {"val": samples[:val_count], "train": samples[val_count:]}
    os.makedirs(args.out, exist_ok=True)
    for name, split in splits.items():
        with open(os.path.join(args.out, f"{name}_images.u8"), "wb") as images:
            for image, _ in split:
                images.write(image)
        with open(os.path.join(args.out, f"{name}_labels.u8"), "wb") as labels:
            labels.write(bytes(label for _, label in split))
    meta = {
        "size": args.size,
        "classes": ["cat", "dog"],
        "counts": {name: len(split) for name, split in splits.items()},
        "label_counts": {
            name: {
                str(label): sum(1 for _, value in split if value == label)
                for label in (0, 1)
            }
            for name, split in splits.items()
        },
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "source": "microsoft/cats_vs_dogs (balanced shards)",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta["counts"]), json.dumps(meta["label_counts"]), flush=True)


if __name__ == "__main__":
    main()
