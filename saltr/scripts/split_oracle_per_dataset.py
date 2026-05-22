#!/usr/bin/env python3
"""split_oracle_per_dataset.py — split combined oracle NPZ into per-dataset files.

The combined reinit_oracle_dataset.npz contains frames from uav123, dtb70, and
visdrone_sot interleaved. This script splits it into:
    saltr/results/reinit_oracle_uav123.npz
    saltr/results/reinit_oracle_dtb70.npz
    saltr/results/reinit_oracle_visdrone_sot.npz

Each per-dataset file has the same schema as the combined file but contains only
rows for that dataset. The sequence_keys strip the dataset prefix so downstream
code can look up sequences by bare name (e.g. "car7" not "uav123/car7").

Usage:
    PYTHONPATH=src:saltr/src .venv/bin/python saltr/scripts/split_oracle_per_dataset.py \
        [--input  saltr/results/reinit_oracle_dataset.npz] \
        [--outdir saltr/results/]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATASETS = ["uav123", "dtb70", "visdrone_sot"]


def split(input_path: str, outdir: str) -> None:
    src = Path(input_path)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(str(src), allow_pickle=True)
    datasets_col = np.array([str(x) for x in d["datasets"]])
    keys_col = np.array([str(x) for x in d["sequence_keys"]])

    for ds in DATASETS:
        mask = datasets_col == ds
        n = mask.sum()
        if n == 0:
            print(f"  {ds}: 0 rows — skipping")
            continue

        # Strip dataset prefix from sequence_keys so bare seq names are usable
        bare_keys = np.array([
            k[len(ds) + 1:] if k.startswith(ds + "/") else k
            for k in keys_col[mask]
        ], dtype=object)

        save_kwargs: dict[str, np.ndarray] = {
            "sequence_keys": bare_keys,
            "datasets":      np.array([ds] * n, dtype=object),
        }
        for key in d.files:
            if key in ("sequence_keys", "datasets"):
                continue
            save_kwargs[key] = d[key][mask]

        out_path = out / f"reinit_oracle_{ds}.npz"
        np.savez_compressed(str(out_path), **save_kwargs)

        # Quick stats
        splits_col = save_kwargs["splits"]
        from collections import Counter
        split_counts = Counter(str(s) for s in splits_col)
        seqs = sorted(set(str(k) for k in bare_keys))
        print(
            f"  {ds}: {n} frames, {len(seqs)} seqs, splits={dict(split_counts)} → {out_path}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",  default="saltr/results/reinit_oracle_dataset.npz")
    ap.add_argument("--outdir", default="saltr/results/")
    args = ap.parse_args()
    split(args.input, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
