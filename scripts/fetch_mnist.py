#!/usr/bin/env python3
"""Download and decompress the four MNIST IDX files."""
import gzip
import pathlib
import sys
import urllib.request

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def main() -> None:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/mnist")
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = out / name.removesuffix(".gz")
        if target.exists():
            print(f"exists: {target}")
            continue
        print(f"download: {name}")
        data = urllib.request.urlopen(f"{BASE}/{name}", timeout=60).read()
        (out / name).write_bytes(data)
        target.write_bytes(gzip.decompress(data))
        print(f"wrote: {target}")

if __name__ == "__main__":
    main()
