#!/usr/bin/env python
"""Aggregate the control matrix: passive vs LA/FC control per (tracker, dataset).

For each control run (outputs/eval_control_matrix/<trk>/<ds>/<tag>) vs its passive
baseline (outputs/eval/<trk>/<ds>/test/<trk>_r3_passive), restricted to the control
run's hard sequences, reports:
  meanAUC, dAUC (vs passive, same seqs) — efficacy/safety
  FCR%, FCD  (control) and their passive values — FC-control goal
  LA%        (control) and passive — LA-control goal
  control fire counts (gated_la / relocate / sgla_redetect hits / fc_fired / fc_challenge)

Usage: .venv/bin/python tools/control_matrix_agg.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "salrtd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa: F401,E402
from csc_uav_tracking.registry import DATASETS  # noqa: E402
from csc_lib.eval.custom_metrics.tracking_metrics import compute_per_frame_arrays  # noqa: E402
sys.argv = _argv

MATRIX = PROJECT_ROOT / "outputs" / "eval_control_matrix"
TAGS = ["la_mb", "la_sgla", "fc_hold", "fc_chal"]
FC = 3
LA = 2


def gt_array(seq):
    out = []
    for bb in seq.ground_truth:
        ok = bb is not None and getattr(bb, "valid", True)
        out.append([float(bb.x), float(bb.y), float(bb.w), float(bb.h)] if ok else [0.0] * 4)
    return np.asarray(out, float)


def read_preds(p: Path):
    rows = []
    for ln in open(p):
        ln = ln.strip()
        v = [float(x) for x in ln.replace("\t", ",").split(",")[:4] if x] if ln else []
        rows.append((v + [0.0] * 4)[:4])
    return np.asarray(rows, float)


def seq_iou(seq, preds_dir: Path):
    pf = preds_dir / f"{seq.name}.txt"
    if not pf.exists():
        return None
    gt = gt_array(seq)
    pr = read_preds(pf)
    n = min(len(pr), len(gt))
    if n == 0:
        return None
    try:
        first = next(iter(seq.frames))
        diag = float(np.hypot(*first.shape[1::-1]))
    except Exception:
        diag = 1280.0
    ious, _, _ = compute_per_frame_arrays(pr[:n], gt[:n], image_diag=diag)
    return ious


def states_seq(states_dir: Path, name: str):
    p = states_dir / f"{name}.jsonl"
    if not p.exists():
        return []
    out = []
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if r.get("init"):
            continue
        out.append(int(r.get("derived_state", -1)))
    return out


def rate_fcd(states):
    n = len(states)
    if n == 0:
        return 0.0, 0.0, 0.0
    fc = sum(1 for s in states if s == FC)
    la = sum(1 for s in states if s == LA)
    runs = []
    r = 0
    for s in states:
        if s == FC:
            r += 1
        elif r:
            runs.append(r)
            r = 0
    if r:
        runs.append(r)
    fcd = float(np.mean(runs)) if runs else 0.0
    return fc / n, la / n, fcd


def auc_over(seqs, index, preds_dir):
    arrs = []
    for name in seqs:
        s = index.get(name)
        if s is None:
            continue
        iou = seq_iou(s, preds_dir)
        if iou is None:
            continue
        fin = np.isfinite(iou)
        if fin.any():
            arrs.append(iou[fin])
    return float(np.concatenate(arrs).mean()) if arrs else float("nan")


def states_over(seqs, states_dir):
    allst = []
    for name in seqs:
        allst.extend(states_seq(states_dir, name))
    return allst


def main():
    cache = {}
    for ds in ["uav123", "uav123_10fps"]:
        cache[ds] = {s.name: s for s in DATASETS.build(ds, split="test")}

    for trk in ["sglatrack", "avtrack", "ortrack"]:
        for ds in ["uav123", "uav123_10fps"]:
            index = cache[ds]
            passive = PROJECT_ROOT / "outputs" / "eval" / trk / ds / "test" / f"{trk}_r3_passive"
            cell = MATRIX / trk / ds
            if not cell.is_dir():
                continue
            present = [t for t in TAGS if (cell / t / "predictions").is_dir()]
            if not present:
                continue
            # hard seqs = union of the control runs' sequences (same set per cell)
            seqs = sorted({p.stem for t in present
                           for p in (cell / t / "predictions").glob("*.txt")})
            base_auc = auc_over(seqs, index, passive / "predictions")
            base_fcr, base_la, base_fcd = rate_fcd(states_over(seqs, passive / "states"))
            print(f"\n================ {trk} / {ds}  ({len(seqs)} hard seqs) ================")
            print(f"  passive: AUC={base_auc:.4f}  FCR={100*base_fcr:.2f}%  FCD={base_fcd:.2f}  LA={100*base_la:.2f}%")
            print(f"  {'config':9s} {'AUC':>7s} {'dAUC':>8s} {'FCR%':>6s}{'(d)':>7s} {'FCD':>5s}{'(d)':>7s} {'LA%':>6s}{'(d)':>7s}  fires")
            for t in present:
                run = cell / t
                auc = auc_over(seqs, index, run / "predictions")
                fcr, la, fcd = rate_fcd(states_over(seqs, run / "states"))
                m = {}
                mj = run / "metrics.json"
                if mj.exists():
                    d = json.load(open(mj))
                    for k in ("control_gated_la_frames", "control_gated_relocate_frames",
                              "control_sgla_redetect_hits", "control_fc_fired_frames",
                              "control_fc_challenge_switches", "control_fc_challenge_commits",
                              "control_fc_challenge_rollbacks", "control_fc_challenge_starts"):
                        if d.get(k):
                            m[k.replace("control_", "").replace("fc_challenge_", "chal_")] = d[k]
                fires = " ".join(f"{k}={v}" for k, v in m.items()) or "(none)"
                print(f"  {t:9s} {auc:7.4f} {auc-base_auc:+8.4f} "
                      f"{100*fcr:6.2f}{100*(fcr-base_fcr):+7.2f} {fcd:5.2f}{fcd-base_fcd:+7.2f} "
                      f"{100*la:6.2f}{100*(la-base_la):+7.2f}  {fires}")


if __name__ == "__main__":
    main()
