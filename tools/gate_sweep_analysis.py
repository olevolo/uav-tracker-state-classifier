#!/usr/bin/env python3
"""Analyze gate threshold sweep results.
Usage: python tools/gate_sweep_analysis.py [tracker]
"""
from __future__ import annotations
import sys, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    sys.path.insert(0, str(_p))
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa
import la_smoke as ls

PASSIVE = {
    'avtrack': ROOT / "outputs/baselines/avtrack/uav123/test",
    'ortrack': ROOT / "outputs/baselines/ortrack/uav123/test",
}
COMBO_0_90 = {
    'avtrack': ROOT / "outputs/eval13_avtrack/csc/avtrack/uav123/test/combo_mb",
    'ortrack': ROOT / "outputs/eval13_ortrack/csc/ortrack/uav123/test/combo_mb",
}
SWEEP_BASE = ROOT / "outputs/eval_gate_sweep"
THRESHOLDS = ["0.90", "0.95", "0.97", "0.99"]

def build_gf():
    idx = ls.build_index("uav123")
    P   = ROOT / "outputs/eval5_clamp/csc/sglatrack/uav123/test/passive"
    gf  = {}
    for n, seq in idx.items():
        ious, _ = ls.seq_iou(seq, P / "predictions")
        if ious is not None:
            fin = np.isfinite(ious)
            gf[n] = float((ious[fin] < 0.2).mean())
    return idx, gf

def seg(seqs, idx, d: pathlib.Path):
    vals = []
    for n in seqs:
        seq = idx.get(n)
        if not seq: continue
        ious, _ = ls.seq_iou(seq, d / "predictions")
        if ious is not None:
            fin = np.isfinite(ious)
            if fin.any(): vals.append(float(ious[fin].mean()))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)

def main():
    tracker = sys.argv[1] if len(sys.argv) > 1 else "avtrack"
    idx, gf = build_gf()

    EXCL = {'bird1_1'}
    B = {
        "ALL":  set(idx.keys()),
        "EASY": {n for n, g in gf.items() if g < 0.10},
        "MID":  {n for n, g in gf.items() if 0.10 <= g < 0.30},
        "HARD": {n for n, g in gf.items() if g >= 0.30 and n not in EXCL},
    }

    p_dir = PASSIVE[tracker]
    base  = {bk: seg(seqs, idx, p_dir)[0] for bk, seqs in B.items()}

    print(f"\n{tracker.upper()} — gate_lostaware sweep (combo_mb config)")
    print(f"{'thresh':<8}", end="")
    for bk in ("ALL", "EASY", "MID", "HARD"):
        print(f"  {bk:>8}", end="")
    print("  n_ALL")
    print("-" * 55)

    # passive reference
    print(f"{'passive':<8}", end="")
    for bk, seqs in B.items():
        v, n = seg(seqs, idx, p_dir)
        print(f"  {v:>8.4f}", end="")
    print(f"  {len(idx)}")

    # 0.90 baseline from eval13
    d90 = COMBO_0_90[tracker]
    if d90.exists():
        print(f"{'0.90':<8}", end="")
        for bk, seqs in B.items():
            v, n = seg(seqs, idx, d90)
            b = base[bk]
            s = f"{v-b:>+8.4f}" if v == v and b == b else "     ---"
            print(f"  {s}", end="")
        n_all, _ = seg(B["ALL"], idx, d90)
        na, nn = seg(B["ALL"], idx, d90)
        print(f"  [{nn}]")

    # sweep thresholds
    for thresh in ("0.95", "0.97", "0.99"):
        tag  = f"combo_mb_t{thresh.replace('.','')}"
        d    = SWEEP_BASE / tracker / "uav123" / "test" / tag
        if not (d / "predictions").exists():
            print(f"{'t'+thresh:<8}  (not done yet)")
            continue
        print(f"{'t'+thresh:<8}", end="")
        for bk, seqs in B.items():
            v, n = seg(seqs, idx, d)
            b = base[bk]
            s = f"{v-b:>+8.4f}" if v == v and b == b else "     ---"
            print(f"  {s}", end="")
        _, nn = seg(B["ALL"], idx, d)
        print(f"  [{nn}]")

    print("\n  Goal: EASY ≥ 0 AND HARD > 0")


if __name__ == "__main__":
    main()
