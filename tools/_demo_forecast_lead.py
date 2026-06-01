"""Demo: 'with forecast vs without forecast' diagnosis on easy/hard scenes.

For each sequence (across one or more passive-diag run dirs):
  - REACTIVE (без forecast): instantaneous derived_state (CC/CU/LA/FC).
  - FORECAST  (з forecast):  failure_next_10 / fc_next_10 probs (early warning).
  - GT:                      per-frame IoU(pred, gt) -> fail = IoU<0.2.

Threshold for the forecast is calibrated to FPR<=--fpr on the pooled
genuinely-easy frames (sequences whose GT-fail rate < --easy-max-failrate),
so lead time is measured at an honest operating point (not a hardcoded 0.5).
Per feedback_forecast_threshold_calibration.

Reports, per sequence: difficulty (GT-fail%), reactive failure onset frame,
forecast onset frame (first pFail>=thr), and LEAD = reactive_onset - forecast_onset.
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

TAU_FAIL = 0.2
RISKY_STATES = {2, 3}  # LA, FC


def iou_xywh_arr(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx, gy, gw, gh = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(px, gx); iy1 = np.maximum(py, gy)
    ix2 = np.minimum(px + pw, gx + gw); iy2 = np.minimum(py + ph, gy + gh)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    return np.where(union > 0, inter / union, 0.0)


def load_gt(dataset: str, split: str) -> dict[str, np.ndarray]:
    import csc_uav_tracking  # noqa: F401
    from csc_uav_tracking.registry import DATASETS
    ds = DATASETS.build(dataset, split=split) if dataset == "got10k" else DATASETS.build(dataset)
    return {seq.name: np.array([[b.x, b.y, b.w, b.h] for b in seq.ground_truth], dtype=np.float64)
            for seq in ds}


def first_true(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def load_seq(sfile: Path, pred_dir: Path, gt: np.ndarray) -> dict | None:
    rows = [json.loads(l) for l in sfile.open() if not json.loads(l).get("init")]
    if not rows:
        return None
    st = np.array([r["derived_state"] for r in rows])
    p_fail = np.array([r.get("failure_next_10_prob") or 0.0 for r in rows], dtype=np.float64)
    p_fc = np.array([r.get("false_confirmed_next_10_prob") or 0.0 for r in rows], dtype=np.float64)
    preds = np.array([[float(v) for v in l.split(",")] for l in
                      (pred_dir / f"{sfile.stem}.txt").read_text().splitlines()], dtype=np.float64)
    n = min(len(st), len(preds) - 1, len(gt) - 1)
    if n < 5:
        return None
    ious = iou_xywh_arr(preds[1:n + 1], gt[1:n + 1])
    return {"name": sfile.stem, "n": n, "st": st[:n],
            "p_fail": p_fail[:n], "p_fc": p_fc[:n], "ious": ious,
            "gt_fail": ious < TAU_FAIL}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", nargs=2, metavar=("RUN_DIR", "DATASET"),
                    required=True, help="repeatable: --run <dir> <dataset>")
    ap.add_argument("--split", default="val")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--easy-max-failrate", type=float, default=0.05)
    args = ap.parse_args()

    seqs: list[dict] = []
    for run_dir, dataset in args.run:
        run_dir = Path(run_dir)
        gt_by = load_gt(dataset, args.split)
        for sfile in sorted((run_dir / "states").glob("*.jsonl")):
            if sfile.stem not in gt_by:
                continue
            s = load_seq(sfile, run_dir / "predictions", gt_by[sfile.stem])
            if s:
                seqs.append(s)

    # calibrate pFail threshold on pooled genuinely-easy frames at FPR<=args.fpr
    easy_pfail = np.concatenate([s["p_fail"] for s in seqs
                                 if s["gt_fail"].mean() < args.easy_max_failrate]) \
        if any(s["gt_fail"].mean() < args.easy_max_failrate for s in seqs) else np.array([0.5])
    thr = float(np.quantile(easy_pfail, 1.0 - args.fpr))
    realized_fpr = float((easy_pfail >= thr).mean())
    print(f"\npFail threshold @ FPR<={args.fpr:.0%} on easy frames = {thr:.4f} "
          f"(realized FPR={realized_fpr:.1%}, n_easy_frames={len(easy_pfail)})\n")

    print(f"{'sequence':<14} {'N':>4} {'GTfail%':>7} {'diff':>5}  "
          f"{'react_on':>8} {'fcst_on':>7} {'LEAD':>5}  {'gt_on':>5} {'fcst-gt':>7}  "
          f"{'pFail_mx':>8} {'pFC_mx':>6}")
    print("-" * 96)
    for s in sorted(seqs, key=lambda x: x["gt_fail"].mean()):
        fr = s["gt_fail"].mean()
        diff = "HARD" if fr >= 0.15 else "easy"
        react_on = first_true(np.isin(s["st"], list(RISKY_STATES)))
        fcst_on = first_true(s["p_fail"] >= thr)
        gt_on = first_true(s["gt_fail"])
        lead = (react_on - fcst_on) if (react_on is not None and fcst_on is not None) else None
        fcst_gt = (gt_on - fcst_on) if (gt_on is not None and fcst_on is not None) else None
        f = lambda v: "  -  " if v is None else f"{v:>5}"
        print(f"{s['name']:<14} {s['n']:>4} {100*fr:>7.1f} {diff:>5}  "
              f"{f(react_on):>8} {f(fcst_on):>7} {f(lead):>5}  {f(gt_on):>5} {f(fcst_gt):>7}  "
              f"{s['p_fail'].max():>8.3f} {s['p_fc'].max():>6.3f}")

    print("\nreact_on = first frame in LA/FC (reactive failure onset, без forecast)")
    print("fcst_on  = first frame pFail>=thr (forecast onset, з forecast)")
    print("LEAD     = react_on - fcst_on  (>0 => forecast fires EARLIER than reactive)")
    print("fcst-gt  = gt_on - fcst_on     (>0 => forecast fires BEFORE the real failure)")


if __name__ == "__main__":
    main()
