#!/usr/bin/env python
"""Smoke ΔAUC harness for LA control experiments (UAV123 clamped SGLATrack).

Modes:
  --list_only         : per-seq stats from the PASSIVE baseline (GT-fail rate,
                        LA%, LA mean-IoU, AUC) — used to choose a smoke set.
  --run_dir DIR [--seqs a b ...]
                      : per-seq AUC for DIR/predictions vs the passive baseline;
                        ΔAUC per seq + overall + HARD-only + the uav6 false-LA
                        guard (uav6 ΔAUC must stay ~0 — must NOT regress).

AUC == mean finite per-frame IoU (success-AUC area, OPE). No re-run needed for
the baseline; DIR is produced by tools/run_with_csc.py with a control policy.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa: F401
from csc_uav_tracking.registry import DATASETS
from csc_lib.eval.custom_metrics.tracking_metrics import compute_per_frame_arrays
sys.argv = _argv

PASSIVE = PROJECT_ROOT / "outputs/eval5_clamp/csc/sglatrack/uav123/test/passive"
HARD_GT_FAIL = 0.30   # per-seq GT-failure-rate threshold to call a seq HARD


def gt_array(seq):
    out = []
    for bb in seq.ground_truth:
        ok = bb is not None and getattr(bb, "valid", True)
        out.append([float(bb.x), float(bb.y), float(bb.w), float(bb.h)] if ok else [0.0] * 4)
    return np.asarray(out, float)


def read_preds(p):
    rows = []
    for ln in open(p):
        ln = ln.strip()
        v = [float(x) for x in ln.replace("\t", ",").split(",")[:4] if x] if ln else []
        rows.append((v + [0.0] * 4)[:4])
    return np.asarray(rows, float)


def la_fraction(states_path, n):
    if not states_path.exists():
        return None
    la = tot = 0
    for ln in open(states_path):
        if not ln.strip():
            continue
        d = json.loads(ln); t = int(d.get("frame_idx", -1)); v = d.get("derived_state")
        if 0 <= t < n and v is not None:
            tot += 1
            if int(v) == 2:
                la += 1
    return (la / tot) if tot else 0.0


def fc_stats(states_path, n):
    """Return (fc_fraction, fcd) from a states file. FCD = mean FC run length."""
    if not states_path.exists():
        return None, None
    st = []
    for ln in open(states_path):
        if not ln.strip():
            continue
        d = json.loads(ln); t = int(d.get("frame_idx", -1)); v = d.get("derived_state")
        if 0 <= t < n and v is not None:
            st.append(int(v))
    if not st:
        return 0.0, 0.0
    fc = sum(1 for s in st if s == 3)
    runs = []; r = 0
    for s in st:
        if s == 3:
            r += 1
        elif r:
            runs.append(r); r = 0
    if r:
        runs.append(r)
    return fc / len(st), (float(np.mean(runs)) if runs else 0.0)


def build_index(dataset="uav123", split="test"):
    ds = list(DATASETS.build(dataset, split=split))
    return {s.name: s for s in ds}


def seq_iou(seq, preds_dir):
    pf = preds_dir / f"{seq.name}.txt"
    if not pf.exists():
        return None, None
    gt = gt_array(seq); pr = read_preds(pf); n = min(len(pr), len(gt))
    if n == 0:
        return None, None
    try:
        first = next(iter(seq.frames)); diag = float(np.hypot(*first.shape[1::-1]))
    except Exception:
        diag = 1280.0
    ious, _, _ = compute_per_frame_arrays(pr[:n], gt[:n], image_diag=diag)
    return ious, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list_only", action="store_true")
    ap.add_argument("--run_dir", default=None, help="run dir with predictions/ + states/")
    ap.add_argument("--baseline_dir", default=str(PASSIVE),
                    help="passive baseline run dir for ΔAUC (default eval5_clamp passive; "
                         "pass an ABSOLUTE main-tree path when running inside a git worktree).")
    ap.add_argument("--seqs", nargs="*", default=None, help="restrict to these sequences")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--fc_metrics", action="store_true",
                    help="also report runtime FC%% and FCD (base vs run) — for FC-control experiments.")
    args = ap.parse_args()
    baseline = Path(args.baseline_dir)

    print("loading uav123 ...", file=sys.stderr)
    idx = build_index()

    if args.list_only:
        rows = []
        for name, seq in idx.items():
            ious, n = seq_iou(seq, baseline / "predictions")
            if ious is None:
                continue
            fin = np.isfinite(ious)
            auc = float(ious[fin].mean()) if fin.any() else 0.0
            gtfail = float((ious[fin] < 0.2).mean()) if fin.any() else 0.0
            laf = la_fraction(baseline / "states" / f"{name}.jsonl", n)
            # LA-frame mean IoU (how lost the LA frames actually are)
            la_iou = np.nan
            sp = baseline / "states" / f"{name}.jsonl"
            if sp.exists():
                st = np.full(n, -1, int)
                for ln in open(sp):
                    if not ln.strip():
                        continue
                    d = json.loads(ln); t = int(d.get("frame_idx", -1)); v = d.get("derived_state")
                    if 0 <= t < n and v is not None:
                        st[t] = int(v)
                m = (st == 2) & fin
                la_iou = float(ious[m].mean()) if m.any() else np.nan
            rows.append((gtfail, name, auc, laf or 0.0, la_iou, n))
        rows.sort(reverse=True)
        print(f"\n{'seq':<14} {'gtfail':>7} {'AUC':>6} {'LA%':>6} {'LA_IoU':>7} {'frames':>7}")
        for gtfail, name, auc, laf, la_iou, n in rows:
            li = "  nan" if np.isnan(la_iou) else f"{la_iou:5.3f}"
            print(f"{name:<14} {gtfail:7.3f} {auc:6.3f} {100*laf:5.1f}% {li:>7} {n:7d}")
        return

    assert args.run_dir, "need --run_dir (or --list_only)"
    run = Path(args.run_dir)
    names = args.seqs if args.seqs else sorted(
        p.stem for p in (run / "predictions").glob("*.txt"))
    print(f"\n=== ΔAUC: {args.tag}  ({run}) vs passive baseline ===")
    hdr = f"{'seq':<14} {'gtfail':>7} {'base':>6} {'run':>6} {'ΔAUC':>8} {'baseLA%':>8} {'runLA%':>7} {'hard':>5}"
    if args.fc_metrics:
        hdr += f" {'bFC%':>6} {'rFC%':>6} {'bFCD':>5} {'rFCD':>5}"
    print(hdr)
    deltas = []; hard_deltas = []; guard = None
    b_fcr = []; r_fcr = []; b_fcd = []; r_fcd = []
    for name in names:
        seq = idx.get(name)
        if seq is None:
            print(f"{name:<14} (not in uav123)"); continue
        b_iou, bn = seq_iou(seq, baseline / "predictions")
        r_iou, rn = seq_iou(seq, run / "predictions")
        if b_iou is None or r_iou is None:
            print(f"{name:<14} (missing preds)"); continue
        bfin = np.isfinite(b_iou); rfin = np.isfinite(r_iou)
        b_auc = float(b_iou[bfin].mean()) if bfin.any() else 0.0
        r_auc = float(r_iou[rfin].mean()) if rfin.any() else 0.0
        d = r_auc - b_auc
        gtfail = float((b_iou[bfin] < 0.2).mean()) if bfin.any() else 0.0
        is_hard = gtfail >= HARD_GT_FAIL
        base_la = la_fraction(baseline / "states" / f"{name}.jsonl", bn) or 0.0
        run_la = la_fraction(run / "states" / f"{name}.jsonl", rn)
        run_la_s = "  —" if run_la is None else f"{100*run_la:5.1f}%"
        line = (f"{name:<14} {gtfail:7.3f} {b_auc:6.3f} {r_auc:6.3f} {d:+8.4f} "
                f"{100*base_la:7.1f}% {run_la_s:>7} {'HARD' if is_hard else '':>5}")
        if args.fc_metrics:
            bfc, bfcd = fc_stats(baseline / "states" / f"{name}.jsonl", bn)
            rfc, rfcd = fc_stats(run / "states" / f"{name}.jsonl", rn)
            bfc = bfc or 0.0; bfcd = bfcd or 0.0; rfc = rfc or 0.0; rfcd = rfcd or 0.0
            line += f" {100*bfc:6.2f} {100*rfc:6.2f} {bfcd:5.1f} {rfcd:5.1f}"
            b_fcr.append(bfc); r_fcr.append(rfc); b_fcd.append(bfcd); r_fcd.append(rfcd)
        print(line)
        deltas.append(d)
        if is_hard:
            hard_deltas.append(d)
        if name == "uav6":
            guard = d
    print("-" * (78 + (26 if args.fc_metrics else 0)))
    if deltas:
        print(f"mean ΔAUC (all {len(deltas)} smoke seqs): {np.mean(deltas):+.4f}")
    if hard_deltas:
        print(f"mean ΔAUC (HARD {len(hard_deltas)} seqs, gtfail>={HARD_GT_FAIL}): {np.mean(hard_deltas):+.4f}")
    if guard is not None:
        verdict = "OK" if guard >= -0.01 else "REGRESSION"
        print(f"uav6 false-LA GUARD ΔAUC: {guard:+.4f}  [{verdict}]  (must stay >= -0.01)")
    if args.fc_metrics and b_fcr:
        print(f"mean FCR: base {100*np.mean(b_fcr):.2f}%  ->  run {100*np.mean(r_fcr):.2f}%  "
              f"(Δ {100*(np.mean(r_fcr)-np.mean(b_fcr)):+.2f}pp)")
        print(f"mean FCD: base {np.mean(b_fcd):.2f}  ->  run {np.mean(r_fcd):.2f}  "
              f"(Δ {np.mean(r_fcd)-np.mean(b_fcd):+.2f})")


if __name__ == "__main__":
    main()
