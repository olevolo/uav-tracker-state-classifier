"""Validate CSC live diagnosis against ground truth.

For each sequence in a diagnosis run:
  1. Load live predictions (predictions/<seq>.txt) + CSC states (states/<seq>.jsonl)
  2. Load GT bboxes from the dataset
  3. Compute per-frame IoU(pred, gt)
  4. GT failure = IoU < tau_fail (0.2). Compare against CSC predicted state:
     - For CSC-FC frames: what is the IoU? (TRUE FC => IoU should be LOW)
     - For CSC-CC frames: what is the IoU? (correct => IoU should be HIGH)
  5. Report a CC-vs-fail confusion + mean IoU per CSC state.

This answers: are CSC's FC/LA predictions actual tracker failures, or false alarms?
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


def iou_xywh_arr(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Vectorised IoU for xywh boxes. pred/gt: (N,4)."""
    px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx, gy, gw, gh = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(px, gx); iy1 = np.maximum(py, gy)
    ix2 = np.minimum(px + pw, gx + gw); iy2 = np.minimum(py + ph, gy + gh)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    return np.where(union > 0, inter / union, 0.0)


def load_dataset_gt(dataset: str, split: str) -> dict[str, np.ndarray]:
    import csc_uav_tracking  # noqa: F401
    from csc_uav_tracking.registry import DATASETS
    if dataset == "got10k":
        ds = DATASETS.build(dataset, split=split)
    else:
        ds = DATASETS.build(dataset)
    out = {}
    for seq in ds:
        gt = np.array([[b.x, b.y, b.w, b.h] for b in seq.ground_truth], dtype=np.float64)
        out[seq.name] = gt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    gt_by_seq = load_dataset_gt(args.dataset, args.split)
    states_dir = args.run_dir / "states"
    pred_dir = args.run_dir / "predictions"

    print(f"\n{'seq':<14} {'N':>5} {'GTfail%':>7}  "
          f"{'CSC_FC':>6} {'CSC_LA':>6} {'CSC_CU':>6}  "
          f"{'IoU@FC':>7} {'IoU@LA':>7} {'IoU@CC':>7}  "
          f"{'FCprec':>6} {'FCrec':>6}  {'riskAUC≈':>8}")
    print("-" * 110)

    for sfile in sorted(states_dir.glob("*.jsonl")):
        seq = sfile.stem
        if seq not in gt_by_seq:
            continue
        rows = [json.loads(l) for l in sfile.open() if not json.loads(l).get("init")]
        if not rows:
            continue
        states = np.array([r["derived_state"] for r in rows])
        risk = np.array([r.get("risk_score", 0.0) for r in rows])
        p_fc = np.array([r.get("false_confirmed_next_10_prob") or 0.0 for r in rows])

        preds = []
        with (pred_dir / f"{seq}.txt").open() as f:
            for line in f:
                preds.append([float(v) for v in line.strip().split(",")])
        preds = np.array(preds, dtype=np.float64)  # includes frame 0
        gt = gt_by_seq[seq]

        # align: states/risk are for frames 1..N (init skipped). preds[0] is init.
        n = min(len(states), len(preds) - 1, len(gt) - 1)
        if n < 5:
            continue
        pred_arr = preds[1:n + 1]
        gt_arr = gt[1:n + 1]
        st = states[:n]
        ious = iou_xywh_arr(pred_arr, gt_arr)

        gt_fail = ious < TAU_FAIL          # tracker actually wrong
        csc_fc = st == 3
        csc_la = st == 2
        csc_cu = st == 1
        csc_cc = st == 0

        def miou(mask):
            return float(ious[mask].mean()) if mask.any() else float("nan")

        # FC precision/recall against GT failure (does CSC-FC coincide with low IoU?)
        tp = int((csc_fc & gt_fail).sum())
        fc_prec = tp / max(int(csc_fc.sum()), 1)
        fc_rec = tp / max(int(gt_fail.sum()), 1)

        # crude risk-vs-fail separation: mean risk on fail vs non-fail
        risk_sep = (float(risk[gt_fail].mean()) if gt_fail.any() else float("nan")) - \
                   (float(risk[~gt_fail].mean()) if (~gt_fail).any() else float("nan"))

        print(f"{seq:<14} {n:>5} {100*gt_fail.mean():>7.1f}  "
              f"{int(csc_fc.sum()):>6} {int(csc_la.sum()):>6} {int(csc_cu.sum()):>6}  "
              f"{miou(csc_fc):>7.3f} {miou(csc_la):>7.3f} {miou(csc_cc):>7.3f}  "
              f"{fc_prec:>6.2f} {fc_rec:>6.2f}  {risk_sep:>+8.3f}")

    print("\nGTfail% = frames with IoU<0.2 (tracker actually wrong).")
    print("IoU@FC should be LOW (CSC-FC = true failure), IoU@CC should be HIGH.")
    print("FCprec/FCrec: CSC-FC frames vs GT-fail frames. riskAUC≈: mean(risk|fail) - mean(risk|ok), >0 good.")


if __name__ == "__main__":
    main()
