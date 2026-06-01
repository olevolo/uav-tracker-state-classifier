"""EXPLORATORY: FC audit on UAVDT-SOT SGLATrack baseline.

For each sequence computes:
  - n_frames (valid GT frames)
  - AUC (success)
  - sm_top1_mean (SGLATrack score-map peak — proxy for tracker confidence)
  - true_fc_frames: IoU < 0.2 AND sm_top1 >= 0.5 (confident but wrong)
  - true_fc_pct = true_fc_frames / n_frames * 100
  - useful_for_training: true_fc_pct > 5%

NOTE: SGLATrack raw `confidence` is a response-map scalar (~0.017) that does
not represent probability.  The actual "how confident is the tracker" signal
is `sm_top1` (score-map peak value, 0-1 range) which is much more meaningful.
TAU_CONF=0.5 applied to sm_top1.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRED_DIR = PROJECT_ROOT / "outputs/baselines/sglatrack/uavdt_sot/test/predictions"
TEL_DIR  = PROJECT_ROOT / "outputs/baselines/sglatrack/uavdt_sot/test/telemetry"
ANNO_DIR = Path.home() / "uav-tracker-data/UAV-benchmark-SOT_v1.0/anno"
OUT_CSV  = PROJECT_ROOT / "outputs/fc_audit/sglatrack_uavdt.csv"

TAU_FAIL = 0.2   # IoU threshold below which tracker is "wrong"
TAU_CONF = 0.5   # sm_top1 threshold above which tracker is "confident"
# NOTE: SGLATrack `confidence` field (~0.017) is response-map raw scalar.
# We use `sm_top1` (score-map peak, 0-1) as the meaningful confidence proxy.


def parse_bbox_line(line: str) -> list[float] | None:
    parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
    if len(parts) < 4:
        return None
    try:
        return [float(x) for x in parts[:4]]
    except ValueError:
        return None


def iou_xywh(b1: list[float], b2: list[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def auc_success(ious: list[float]) -> float:
    arr = np.array(ious)
    thresholds = np.linspace(0.0, 1.0, 101)
    return float(np.mean([np.mean(arr > t) for t in thresholds]))


def main() -> None:
    seq_files = sorted(PRED_DIR.glob("*.txt"))
    print(f"Auditing {len(seq_files)} sequences...", flush=True)

    rows: list[dict] = []

    for pred_path in seq_files:
        seq_name = pred_path.stem

        # Load GT
        gt_path = ANNO_DIR / f"{seq_name}_gt.txt"
        if not gt_path.exists():
            print(f"  SKIP {seq_name}: no GT at {gt_path}", flush=True)
            continue

        # Parse GT
        gt_bboxes: list[list[float]] = []
        with open(gt_path) as f:
            for line in f:
                b = parse_bbox_line(line)
                if b:
                    gt_bboxes.append(b)

        # Parse predictions
        pred_bboxes: list[list[float]] = []
        with open(pred_path) as f:
            for line in f:
                b = parse_bbox_line(line)
                if b:
                    pred_bboxes.append(b)

        # Load telemetry confidence
        tel_path = TEL_DIR / f"{seq_name}.jsonl"
        conf_map: dict[int, float] = {}   # frame_idx -> sm_top1
        if tel_path.exists():
            with open(tel_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        fi = rec.get("frame_idx", -1)
                        # Use sm_top1 (score-map peak, 0-1) as confidence proxy.
                        # SGLATrack raw `confidence` (~0.017) is a response-map
                        # scalar that does not represent tracker certainty.
                        sm_top1 = rec.get("sm_top1")
                        if sm_top1 is not None:
                            conf_map[fi] = float(sm_top1)
                    except Exception:
                        pass

        n = min(len(pred_bboxes), len(gt_bboxes))
        if n == 0:
            continue

        ious: list[float] = []
        fc_frames = 0
        conf_values: list[float] = []

        for i in range(n):
            gt = gt_bboxes[i]
            # Skip invalid GT frames
            if gt[2] <= 0 or gt[3] <= 0:
                continue

            iou = iou_xywh(pred_bboxes[i], gt)
            ious.append(iou)

            # Confidence from telemetry (frame_idx is 0-based)
            conf = conf_map.get(i, None)
            if conf is not None:
                conf_values.append(conf)
                # FC: tracker confident but wrong
                if iou < TAU_FAIL and conf >= TAU_CONF:
                    fc_frames += 1

        n_valid = len(ious)
        if n_valid == 0:
            continue

        auc = auc_success(ious)
        conf_mean = float(np.mean(conf_values)) if conf_values else float("nan")
        true_fc_pct = 100.0 * fc_frames / n_valid if n_valid > 0 else 0.0
        useful = true_fc_pct > 5.0

        rows.append(
            {
                "sequence": seq_name,
                "n_frames": n_valid,
                "auc": round(auc, 4),
                "sm_top1_mean": round(conf_mean, 4) if not np.isnan(conf_mean) else "nan",
                "true_fc_frames": fc_frames,
                "true_fc_pct": round(true_fc_pct, 2),
                "useful_for_training": useful,
                "has_telemetry": len(conf_values) > 0,
            }
        )

    # Sort by FC pct descending
    rows.sort(key=lambda r: float(r["true_fc_pct"]), reverse=True)

    # Print table
    hdr = (
        f"{'Sequence':<12} {'Frames':>7} {'AUC':>6} {'sm_top1':>8}"
        f" {'FC_frames':>10} {'FC%':>7} {'Useful':>7}"
    )
    print()
    print(hdr)
    print("-" * 68)
    for r in rows:
        conf_str = f"{r['sm_top1_mean']:.3f}" if r["has_telemetry"] else "n/a"
        useful_str = "YES" if r["useful_for_training"] else "no"
        print(
            f"{r['sequence']:<12} {r['n_frames']:>7} {r['auc']:>6.3f}"
            f" {conf_str:>8} {r['true_fc_frames']:>10}"
            f" {float(r['true_fc_pct']):>6.1f}% {useful_str:>7}"
        )

    n_useful = sum(1 for r in rows if r["useful_for_training"])
    mean_auc = float(np.mean([r["auc"] for r in rows]))
    mean_fc_pct = float(np.mean([float(r["true_fc_pct"]) for r in rows]))

    print()
    print(f"Total sequences:         {len(rows)}")
    print(f"Sequences with FC% > 5%: {n_useful}")
    print(f"Overall mean AUC:        {mean_auc:.3f}")
    print(f"Overall mean FC%:        {mean_fc_pct:.1f}%")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(OUT_CSV, "w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
