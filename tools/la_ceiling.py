#!/usr/bin/env python
"""How much AUC could fixing LA recover? Ceiling analysis, UAV123 clamped SGLATrack.

AUC == mean IoU. LA frames currently sit at mean IoU ~0.14. If a re-detector
converted them to CC-quality (IoU ~0.80) the AUC would jump by
  gain = LA_fraction * (IoU_target - IoU_LA_now).
We report the ceiling overall AND per difficulty tercile (difficulty = per-seq
GT failure rate), since that locates the recoverable headroom (EASY has ~0 LA).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa
from csc_uav_tracking.registry import DATASETS
from csc_lib.eval.custom_metrics.tracking_metrics import compute_per_frame_arrays

RUN = Path("outputs/eval5_clamp/csc/sglatrack/uav123/test/passive")
CC_IOU = 0.80  # target if LA frame were recovered to CC quality

def gt_array(seq):
    out = []
    for bb in seq.ground_truth:
        ok = bb is not None and getattr(bb, "valid", True)
        out.append([float(bb.x), float(bb.y), float(bb.w), float(bb.h)] if ok else [0.0]*4)
    return np.asarray(out, float)

def read_preds(p):
    rows = []
    for ln in open(p):
        ln = ln.strip()
        v = [float(x) for x in ln.replace("\t", ",").split(",")[:4] if x] if ln else []
        rows.append((v + [0.0]*4)[:4])
    return np.asarray(rows, float)

def read_states(p, n):
    st = np.full(n, -1, int)
    if p.exists():
        for ln in open(p):
            if not ln.strip(): continue
            d = json.loads(ln); t = int(d.get("frame_idx", -1)); v = d.get("derived_state")
            if 0 <= t < n and v is not None: st[t] = int(v)
    return st

ds = list(DATASETS.build("uav123", split="test"))
seqs = []  # (name, ious, states, difficulty)
for seq in ds:
    pf = RUN/"predictions"/f"{seq.name}.txt"
    if not pf.exists(): continue
    gt = gt_array(seq); pr = read_preds(pf); n = min(len(pr), len(gt))
    if n == 0: continue
    try:
        first = next(iter(seq.frames)); diag = float(np.hypot(*first.shape[1::-1]))
    except Exception:
        diag = 1280.0
    ious, _, _ = compute_per_frame_arrays(pr[:n], gt[:n], image_diag=diag)
    st = read_states(RUN/"states"/f"{seq.name}.jsonl", n)
    fin = np.isfinite(ious)
    diff = float((ious[fin] < 0.2).mean()) if fin.any() else 0.0  # GT failure rate
    seqs.append((seq.name, ious, st, fin, diff))

def ceiling(group, target=CC_IOU):
    """Return overall stats for a list of seq tuples."""
    tot = la_n = 0; la_iou_sum = 0.0; iou_sum = 0.0
    for _, ious, st, fin, _ in group:
        n = len(ious)
        iou_sum += float(ious[fin].sum()); tot += int(fin.sum())
        m = (st[:n] == 2) & fin
        la_n += int(m.sum()); la_iou_sum += float(ious[m].sum())
    auc = iou_sum/tot if tot else 0.0
    la_frac = la_n/tot if tot else 0.0
    la_iou = la_iou_sum/la_n if la_n else 0.0
    gain_perfect = la_frac*(target - la_iou)         # all LA -> CC quality
    gain_half = 0.5*la_frac*(target - la_iou)        # recover half of LA
    gain_to_succ = la_frac*max(0.0, 0.5 - la_iou)    # LA -> just-passing IoU 0.5
    return dict(auc=auc, la_frac=la_frac, la_iou=la_iou, tot=tot,
                gain_perfect=gain_perfect, gain_half=gain_half, gain_to_succ=gain_to_succ)

# terciles by difficulty
seqs_sorted = sorted(seqs, key=lambda s: s[4]); t = len(seqs_sorted)//3
bins = {"EASY": seqs_sorted[:t], "MEDIUM": seqs_sorted[t:2*t], "HARD": seqs_sorted[2*t:], "ALL": seqs_sorted}

L = ["# LA-recovery AUC ceiling — UAV123 clamped SGLATrack (passive)\n",
     f"AUC == mean IoU. LA currently sits at low IoU; recovering it to CC-quality "
     f"({CC_IOU}) is the only large AUC lever. FC ceiling for contrast: ~±0.004.\n",
     "| bin | seqs | AUC now | LA % | LA mean IoU | +AUC if LA→CC (ceiling) | +AUC if half recovered | +AUC if LA→0.5 |",
     "|---|---|---|---|---|---|---|---|"]
for name, g in bins.items():
    s = ceiling(g)
    L.append(f"| {name} | {len(g)} | {s['auc']:.3f} | {100*s['la_frac']:.1f}% | {s['la_iou']:.3f} | "
             f"**+{s['gain_perfect']:.4f}** | +{s['gain_half']:.4f} | +{s['gain_to_succ']:.4f} |")
out = "\n".join(L)
print(out)
Path("outputs/eval6_matrix/LA_ceiling.md").write_text(out)
print("\n[saved] outputs/eval6_matrix/LA_ceiling.md", file=sys.stderr)
