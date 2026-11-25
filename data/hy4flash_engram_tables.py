#!/usr/bin/env python3
"""hy4flash_engram_tables.py -- the Engram hash constants of a Hy4-flash release, for `rad-convert --add`.

The release ships each Engram layer's embedding table and its compressed token map
(`engram_layer.lookup_table`), but not the hash itself: transformers' HYV4FlashEngramLayer draws
each layer's odd n-gram multipliers at load (NumPy's default generator, seeded
`engram_seed + 10007 * layer_id` and bounded by the compressed vocabulary the loaded map spans) and
reads the per-head bucket moduli from the config. This tool writes them as a small safetensors,
three I64 arrays an Engram layer, named by the layer index the tree's weights use (`blk.<l>`, one
below the config's `engram_layer_ids`):

  engram.<l>.hash_multipliers   I64 [engram_max_ngram_size]
  engram.<l>.head_vocab_sizes   I64 [(max_ngram - 1) * n_head_per_ngram]   (order-major, as hashed)
  engram.<l>.head_offsets       I64 [same]   (each head's first row in the layer's joined table)

arch/hyv4flash declares them, optional, as blk.<l>.engram_mult / engram_vocab_sizes / engram_offsets.
Reads config.json and the four token maps (0.5 MB each) only, never a table.

usage: python3 data/hy4flash_engram_tables.py RELEASE_DIR OUT.safetensors
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def hash_multipliers(seed: int, layer_id: int, max_ngram_size: int, compressed_vocab_size: int) -> list[int]:
    """HYV4FlashEngramLayer.draw_hash_multipliers: odd, and small enough that a product with any
    compressed token id stays within int64."""
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab_size // 2)
    drawn = np.random.default_rng(seed + 10007 * layer_id).integers(0, half_bound, size=max_ngram_size, dtype=np.int64)
    return (drawn * 2 + 1).tolist()


def head_offsets(sizes: list[int]) -> list[int]:
    """HYV4FlashEngramEmbedding.head_offsets: the heads share one table, each from its own row on."""
    return [sum(sizes[:i]) for i in range(len(sizes))]


def read_tensor(release: Path, name: str) -> np.ndarray:
    """One tensor of a safetensors checkpoint, found through model.safetensors.index.json where the
    release is sharded."""
    index = release / "model.safetensors.index.json"
    files = [release / json.load(open(index))["weight_map"][name]] if index.exists() else sorted(release.glob("*.safetensors"))
    for path in files:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            entry = json.loads(f.read(n)).get(name)
            if entry is None:
                continue
            if entry["dtype"] not in ("I32", "I64"):
                sys.exit(f"{name}: dtype {entry['dtype']}, expected an integer token map")
            lo, hi = entry["data_offsets"]
            f.seek(8 + n + lo)
            return np.frombuffer(f.read(hi - lo), dtype="<i4" if entry["dtype"] == "I32" else "<i8")
    sys.exit(f"{name}: not in {release}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("out")
    a = ap.parse_args()
    release = Path(a.release)
    cfg = json.load(open(release / "config.json"))
    hp = {**cfg, **cfg.get("text_config", {})}
    layer_ids = list(hp.get("engram_layer_ids") or [])
    if not layer_ids:
        sys.exit("the release has no Engram layers; nothing to write")
    max_ngram, per, seed = hp["engram_max_ngram_size"], hp["engram_n_head_per_ngram"], hp.get("engram_seed", 0)

    tensors, report = [], []
    for layer_id in layer_ids:
        layer = layer_id - 1
        sizes = [s for order in hp["engram_layer_head_vocab_sizes"][str(layer_id)] for s in order]
        if len(sizes) != (max_ngram - 1) * per:
            sys.exit(f"engram layer id {layer_id}: {len(sizes)} head moduli, expected {(max_ngram - 1) * per}")
        token_map = read_tensor(release, f"model.layers.{layer}.engram_layer.lookup_table")
        cvocab = int(token_map.max()) + 1
        mults = hash_multipliers(seed, layer_id, max_ngram, cvocab)
        tensors += [(f"engram.{layer}.hash_multipliers", mults),
                    (f"engram.{layer}.head_vocab_sizes", sizes),
                    (f"engram.{layer}.head_offsets", head_offsets(sizes))]
        report.append(f"blk.{layer}: token map {len(token_map)} -> {cvocab}, {len(sizes)} heads, {sum(sizes)} rows")

    header, blobs, off = {}, [], 0
    for name, vals in tensors:
        b = struct.pack(f"<{len(vals)}q", *vals)
        header[name] = {"dtype": "I64", "shape": [len(vals)], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"format": "pt", "source": "radiance data/hy4flash_engram_tables.py"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    print(f"{a.out}: {len(layer_ids)} Engram layers")
    for line in report:
        print("  " + line)


if __name__ == "__main__":
    main()
