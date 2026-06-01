"""Frame-by-frame forecast timeline for ONE sequence — shows how the forecast
head fires BEFORE the reactive state and before the real (GT) failure.

Columns per frame: pFail / pFC / pLost (forecast probs), reactive derived_state,
IoU vs GT. Markers: '<<ALARM' first frame pFail>=thr, '<<REACT' first LA/FC,
'<<GTFAIL' first IoU<0.2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

STATE = ["CC", "CU", "LA", "FC"]
TAU_FAIL = 0.2


def iou_xywh_arr(pred, gt):
    px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx, gy, gw, gh = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(px, gx); iy1 = np.maximum(py, gy)
    ix2 = np.minimum(px + pw, gx + gw); iy2 = np.minimum(py + ph, gy + gh)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    return np.where(union > 0, inter / union, 0.0)


def first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--thr", type=float, default=0.693)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    import csc_uav_tracking  # noqa: F401
    from csc_uav_tracking.registry import DATASETS
    ds = DATASETS.build(args.dataset)
    gt = None
    for s in ds:
        if s.name == args.seq:
            gt = np.array([[b.x, b.y, b.w, b.h] for b in s.ground_truth], dtype=np.float64)
            break
    assert gt is not None, f"{args.seq} not found in {args.dataset}"

    rows = [json.loads(l) for l in (args.run_dir / "states" / f"{args.seq}.jsonl").open()
            if not json.loads(l).get("init")]
    st = np.array([r["derived_state"] for r in rows])
    pf = np.array([r.get("failure_next_10_prob") or 0.0 for r in rows])
    pfc = np.array([r.get("false_confirmed_next_10_prob") or 0.0 for r in rows])
    pl = np.array([r.get("lost_aware_next_10_prob") or 0.0 for r in rows])
    preds = np.array([[float(v) for v in l.split(",")] for l in
                      (args.run_dir / "predictions" / f"{args.seq}.txt").read_text().splitlines()],
                     dtype=np.float64)
    n = min(len(st), len(preds) - 1, len(gt) - 1)
    ious = iou_xywh_arr(preds[1:n + 1], gt[1:n + 1])
    st, pf, pfc, pl = st[:n], pf[:n], pfc[:n], pl[:n]

    alarm = first(pf >= args.thr)
    react = first(np.isin(st, [2, 3]))
    gtf = first(ious < TAU_FAIL)
    print(f"\n{args.seq}: ALARM(pFail>={args.thr})={alarm}  REACT(LA/FC)={react}  GTFAIL(IoU<0.2)={gtf}")
    if alarm is not None and gtf is not None:
        print(f"  → forecast lead over real failure = {gtf - alarm} frames; over reactive state = "
              f"{(react - alarm) if react is not None else 'n/a'} frames\n")

    s0 = args.start if args.start is not None else max(0, (alarm or 0) - 8)
    s1 = args.end if args.end is not None else min(n, (gtf or react or n) + 6)

    print(f"{'frame':>5} {'pFail':>6} {'pFC':>6} {'pLost':>6}  {'react':>5} {'IoU':>6}  marker")
    print("-" * 58)
    for i in range(s0, s1, args.stride):
        mk = []
        if i == alarm: mk.append("<<ALARM (forecast)")
        if i == react: mk.append("<<REACT (state→LA/FC)")
        if i == gtf:   mk.append("<<GTFAIL (IoU<0.2)")
        bar = "ALARM" if pf[i] >= args.thr else ""
        print(f"{i:>5} {pf[i]:>6.3f} {pfc[i]:>6.3f} {pl[i]:>6.3f}  {STATE[st[i]]:>5} {ious[i]:>6.3f}  "
              f"{bar:<5} {' '.join(mk)}")


if __name__ == "__main__":
    main()
