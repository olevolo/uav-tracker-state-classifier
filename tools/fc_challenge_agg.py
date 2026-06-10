#!/usr/bin/env python
"""Aggregate FC challenge-and-switch sweep: passive vs challenge configs.

Per config reports:
  meanAUC  — mean finite per-frame IoU over the subset (OPE success-AUC proxy,
             same definition as tools/la_smoke.py)
  dAUC     — meanAUC - passive meanAUC (catastrophic-switch detector: must not
             regress hard)
  FCR      — N(derived_state==FC) / N_total  (CLAUDE.md §Metrics)
  FCD      — mean contiguous FC-segment length
  controller counters from metrics.json (starts/switches/commits/rollbacks/aborts)

Also prints a per-seq dAUC table so a single catastrophic false switch is visible.

Usage:
  .venv/bin/python tools/fc_challenge_agg.py \
      --root outputs/eval_fc_challenge --passive passive \
      --runs chal_s2 chal_s3 chal_s5 --dataset uav123 --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa: F401,E402
from csc_uav_tracking.registry import DATASETS  # noqa: E402
from csc_lib.eval.custom_metrics.tracking_metrics import compute_per_frame_arrays  # noqa: E402
sys.argv = _argv

FC_IDX = 3  # derived_state false_confirmed


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


def states_fc(states_dir: Path, name: str):
    """Return per-frame derived_state list (non-init) for a sequence."""
    p = states_dir / f"{name}.jsonl"
    if not p.exists():
        return []
    st = []
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if r.get("init"):
            continue
        st.append(int(r.get("derived_state", -1)))
    return st


def fcr_fcd(states: list[int]):
    n = len(states)
    if n == 0:
        return 0.0, 0.0, 0
    fc = sum(1 for s in states if s == FC_IDX)
    runs = []
    r = 0
    for s in states:
        if s == FC_IDX:
            r += 1
        elif r:
            runs.append(r)
            r = 0
    if r:
        runs.append(r)
    fcr = fc / n
    fcd = float(np.mean(runs)) if runs else 0.0
    return fcr, fcd, fc


def config_metrics(root: Path, tag: str, index: dict):
    run = root / tag
    preds = run / "predictions"
    states = run / "states"
    seqs = sorted(p.stem for p in preds.glob("*.txt"))
    per_seq = {}
    all_iou = []
    all_states = []
    for name in seqs:
        seq = index.get(name)
        if seq is None:
            continue
        ious = seq_iou(seq, preds)
        st = states_fc(states, name)
        if ious is None:
            continue
        fin = np.isfinite(ious)
        auc = float(ious[fin].mean()) if fin.any() else 0.0
        per_seq[name] = {"auc": auc, "n": int(len(ious))}
        all_iou.append(ious[fin])
        all_states.extend(st)
    mean_auc = float(np.concatenate(all_iou).mean()) if all_iou else 0.0
    fcr, fcd, fc = fcr_fcd(all_states)
    # controller counters
    ctr = {}
    mj = run / "metrics.json"
    if mj.exists():
        d = json.load(open(mj))
        for k in ("control_fc_challenge_starts", "control_fc_challenge_switches",
                  "control_fc_challenge_commits", "control_fc_challenge_rollbacks",
                  "control_fc_challenge_aborts", "control_fc_challenge_redetect_calls",
                  "control_template_update_freezes"):
            ctr[k] = d.get(k, 0)
    return {"mean_auc": mean_auc, "fcr": fcr, "fcd": fcd, "fc_frames": fc,
            "per_seq": per_seq, "ctr": ctr, "n_seq": len(per_seq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--passive", default="passive")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default="uav123")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    root = Path(args.root)
    index = {s.name: s for s in DATASETS.build(args.dataset, split=args.split)}

    base = config_metrics(root, args.passive, index)
    runs = {tag: config_metrics(root, tag, index) for tag in args.runs}

    print(f"\n=== FC challenge sweep — {root} ({base['n_seq']} seqs, "
          f"{base['fc_frames']} passive FC frames) ===\n")
    hdr = (f"{'config':12s} {'meanAUC':>8s} {'dAUC':>8s} {'FCR%':>7s} {'FCD':>6s} "
           f"{'start':>6s} {'switch':>6s} {'commit':>6s} {'rollbk':>6s} {'abort':>6s} {'froze':>6s}")
    print(hdr)
    print("-" * len(hdr))

    def fmt(tag, m, base_auc):
        c = m["ctr"]
        return (f"{tag:12s} {m['mean_auc']:8.4f} {m['mean_auc']-base_auc:+8.4f} "
                f"{100*m['fcr']:7.3f} {m['fcd']:6.2f} "
                f"{c.get('control_fc_challenge_starts',0):6d} "
                f"{c.get('control_fc_challenge_switches',0):6d} "
                f"{c.get('control_fc_challenge_commits',0):6d} "
                f"{c.get('control_fc_challenge_rollbacks',0):6d} "
                f"{c.get('control_fc_challenge_aborts',0):6d} "
                f"{c.get('control_template_update_freezes',0):6d}")

    print(fmt(args.passive, base, base["mean_auc"]))
    for tag in args.runs:
        print(fmt(tag, runs[tag], base["mean_auc"]))

    # Per-seq dAUC vs passive (spot catastrophic switches / wins)
    print("\n=== per-seq meanAUC (passive | " + " | ".join(args.runs) + ") + dAUC ===")
    names = sorted(base["per_seq"])
    print(f"{'seq':16s} {'passive':>8s} " + " ".join(f"{t:>17s}" for t in args.runs))
    for name in names:
        b = base["per_seq"][name]["auc"]
        cells = []
        for tag in args.runs:
            ps = runs[tag]["per_seq"].get(name)
            if ps is None:
                cells.append(f"{'--':>17s}")
            else:
                cells.append(f"{ps['auc']:8.4f}({ps['auc']-b:+6.4f})")
        print(f"{name:16s} {b:8.4f} " + " ".join(cells))


if __name__ == "__main__":
    main()
