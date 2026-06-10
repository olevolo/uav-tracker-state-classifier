#!/usr/bin/env python
"""Decompose UAV123 success-AUC by CSC state to resolve the FCR-vs-AUC paradox.

Success-AUC (OPE) == mean per-frame IoU (area under the success curve = E[IoU]).
So AUC = sum_state (frame_fraction_state * mean_IoU_state). This shows exactly
why reducing FC (a ~0.3%-of-frames state) cannot move AUC, and why control's
AUC change comes from the CC<->LA mass shift, not from FC.

Runs compared (all CLAMPED sglatrack, eval5_clamp): passive / ctrl_hold / ctrl_widen.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.argv = [sys.argv[0]]  # keep arg parsers in imported libs happy
import csc_uav_tracking  # noqa: F401
from csc_uav_tracking.registry import DATASETS
from csc_lib.eval.custom_metrics.tracking_metrics import compute_per_frame_arrays

NAMES = {0: "CC", 1: "CU", 2: "LA", 3: "FC"}
ROOT = Path("outputs/eval5_clamp/csc/sglatrack/uav123/test")
RUNS = {"passive": ROOT/"passive", "ctrl_hold": ROOT/"ctrl_hold", "ctrl_widen": ROOT/"ctrl_widen"}

def gt_array(seq):
    out = []
    for bb in seq.ground_truth:
        if bb is None or not getattr(bb, "valid", True):
            out.append([0.0, 0.0, 0.0, 0.0])
        else:
            out.append([float(bb.x), float(bb.y), float(bb.w), float(bb.h)])
    return np.asarray(out, dtype=np.float64)

def read_preds(p: Path):
    rows = []
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            rows.append([0.0, 0.0, 0.0, 0.0]); continue
        v = [float(x) for x in ln.replace("\t", ",").split(",")[:4] if x]
        rows.append((v + [0.0]*4)[:4])
    return np.asarray(rows, dtype=np.float64)

def read_states(p: Path, n: int):
    st = np.full(n, -1, dtype=int)
    if not p.exists(): return st
    for ln in open(p):
        ln = ln.strip()
        if not ln: continue
        d = json.loads(ln); t = int(d.get("frame_idx", -1)); v = d.get("derived_state")
        if 0 <= t < n and v is not None: st[t] = int(v)
    return st

print("loading uav123 ...", file=sys.stderr)
ds = list(DATASETS.build("uav123", split="test"))
gtmap, diagmap = {}, {}
for seq in ds:
    gtmap[seq.name] = gt_array(seq)
    try:
        first = next(iter(seq.frames)); h, w = first.shape[:2]; diagmap[seq.name] = float(np.hypot(w, h))
    except Exception:
        diagmap[seq.name] = 1280.0
print(f"  {len(gtmap)} sequences", file=sys.stderr)

summary = {}
for tag, run in RUNS.items():
    pdir, sdir = run/"predictions", run/"states"
    if not pdir.is_dir(): print(f"SKIP {tag} (no predictions)", file=sys.stderr); continue
    # per-state accumulation
    iou_sum = {k: 0.0 for k in range(4)}; cnt = {k: 0 for k in range(4)}
    tot_iou = 0.0; tot_n = 0
    for name, gt in gtmap.items():
        pf = pdir/f"{name}.txt"
        if not pf.exists(): continue
        preds = read_preds(pf); n = min(len(preds), len(gt))
        if n == 0: continue
        ious, _, _ = compute_per_frame_arrays(preds[:n], gt[:n], image_diag=diagmap[name])
        st = read_states(sdir/f"{name}.jsonl", n)
        fin = np.isfinite(ious)
        for k in range(4):
            m = (st[:n] == k) & fin
            iou_sum[k] += float(ious[m].sum()); cnt[k] += int(m.sum())
        tot_iou += float(ious[fin].sum()); tot_n += int(fin.sum())
    frac = {k: (cnt[k]/tot_n if tot_n else 0.0) for k in range(4)}
    mean_iou = {k: (iou_sum[k]/cnt[k] if cnt[k] else 0.0) for k in range(4)}
    contrib = {k: frac[k]*mean_iou[k] for k in range(4)}
    summary[tag] = dict(frac=frac, mean_iou=mean_iou, contrib=contrib,
                        fw_auc=tot_iou/tot_n if tot_n else 0.0, n=tot_n)

# ---- report ----
L = ["# AUC-by-state decomposition — UAV123, clamped SGLATrack (eval5)\n",
     "success-AUC == mean IoU, so AUC = Σ_state (frame_fraction · mean_IoU). "
     "FC is ~0.3% of frames with low IoU -> its AUC contribution is ~0.000. "
     "Control's AUC change is the CC↔LA mass shift, NOT FC.\n"]
for tag, s in summary.items():
    L.append(f"\n## {tag}  (frame-weighted AUC = {s['fw_auc']:.4f}, n={s['n']})")
    L.append("| state | frame % | mean IoU | AUC contribution |")
    L.append("|---|---|---|---|")
    for k in range(4):
        L.append(f"| {NAMES[k]} | {100*s['frac'][k]:.2f}% | {s['mean_iou'][k]:.3f} | {s['contrib'][k]:.4f} |")
    L.append(f"| **Σ** | 100% | — | **{sum(s['contrib'].values()):.4f}** |")

# delta passive -> control
if "passive" in summary:
    base = summary["passive"]
    for tag in ("ctrl_hold", "ctrl_widen"):
        if tag not in summary: continue
        s = summary[tag]
        L.append(f"\n## Δ {tag} − passive (where the AUC actually moved)")
        L.append("| state | Δframe % | Δ AUC contribution |")
        L.append("|---|---|---|")
        for k in range(4):
            L.append(f"| {NAMES[k]} | {100*(s['frac'][k]-base['frac'][k]):+.2f}% | {s['contrib'][k]-base['contrib'][k]:+.4f} |")
        L.append(f"| **net** | — | **{s['fw_auc']-base['fw_auc']:+.4f}** |")

out = "\n".join(L)
print(out)
Path("outputs/eval6_matrix/AUC_by_state_decomp.md").write_text(out)
print("\n[saved] outputs/eval6_matrix/AUC_by_state_decomp.md", file=sys.stderr)
