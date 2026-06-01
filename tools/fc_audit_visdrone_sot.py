"""EXPLORATORY: FC audit on VisDrone-SOT SGLATrack baseline.

For each sequence computes:
  - n_frames (valid GT frames)
  - AUC (success)
  - sm_top1_mean (SGLATrack score-map peak — proxy for tracker confidence)
  - FC_frames: IoU < 0.2 AND sm_top1 >= 0.5 (tracker confident but wrong)
  - FC% = FC_frames / n_frames * 100
  - useful_for_training: FC% > 5%

VisDrone-SOT annotation format: x,y,w,h per line (comma-separated, no extra fields).
TAU_CONF=0.5 applied to sm_top1 (same as UAVDT audit).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRED_DIR = PROJECT_ROOT / "outputs/baselines/sglatrack/visdrone_sot/test/predictions"
TEL_DIR  = PROJECT_ROOT / "outputs/baselines/sglatrack/visdrone_sot/test/telemetry"
ANNO_DIR = Path.home() / "uav-tracker-data/VisDrone-SOT/VisDrone2019-SOT-test-dev/annotations"
OUT_CSV  = PROJECT_ROOT / "outputs/fc_audit/sglatrack_visdrone_sot.csv"

TAU_FAIL = 0.2   # IoU threshold below which tracker is "wrong"
TAU_CONF = 0.5   # sm_top1 threshold above which tracker is "confident"


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
        gt_path = ANNO_DIR / f"{seq_name}.txt"
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

        # Load telemetry confidence (sm_top1)
        tel_path = TEL_DIR / f"{seq_name}.jsonl"
        conf_map: dict[int, float] = {}   # frame_idx -> sm_top1
        if tel_path.exists():
            with open(tel_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        fi = rec.get("frame_idx", -1)
                        sm_top1 = rec.get("sm_top1")
                        if sm_top1 is not None:
                            conf_map[fi] = float(sm_top1)
                    except Exception:
                        pass

        n = min(len(pred_bboxes), len(gt_bboxes))
        if n == 0:
            print(f"  SKIP {seq_name}: empty preds or GT", flush=True)
            continue

        ious: list[float] = []
        fc_frames = 0
        conf_values: list[float] = []

        for i in range(n):
            gt = gt_bboxes[i]
            # Skip invalid GT frames (zero-size bbox)
            if gt[2] <= 0 or gt[3] <= 0:
                continue

            iou_val = iou_xywh(pred_bboxes[i], gt)
            ious.append(iou_val)

            # Confidence from telemetry (frame_idx is 0-based)
            conf = conf_map.get(i, None)
            if conf is not None:
                conf_values.append(conf)
                # FC: tracker confident but wrong
                if iou_val < TAU_FAIL and conf >= TAU_CONF:
                    fc_frames += 1

        n_valid = len(ious)
        if n_valid == 0:
            print(f"  SKIP {seq_name}: no valid frames", flush=True)
            continue

        auc_val = auc_success(ious)
        conf_mean = float(np.mean(conf_values)) if conf_values else float("nan")
        true_fc_pct = 100.0 * fc_frames / n_valid
        useful = true_fc_pct > 5.0

        rows.append(
            {
                "sequence": seq_name,
                "n_frames": n_valid,
                "auc": round(auc_val, 4),
                "sm_top1_mean": round(conf_mean, 4) if not (
                    isinstance(conf_mean, float) and np.isnan(conf_mean)
                ) else "nan",
                "fc_frames": fc_frames,
                "fc_pct": round(true_fc_pct, 2),
                "useful_for_training": useful,
                "has_telemetry": len(conf_values) > 0,
            }
        )
        print(
            f"  {seq_name:<28} frames={n_valid:>5} auc={auc_val:.3f}"
            f" sm_top1={conf_mean:.3f} fc_frames={fc_frames:>5}"
            f" fc%={true_fc_pct:>5.1f}% {'YES' if useful else ''}",
            flush=True,
        )

    # Sort by FC% descending
    rows.sort(key=lambda r: float(r["fc_pct"]), reverse=True)

    # Print final table
    hdr = (
        f"\n{'Sequence':<28} {'Frames':>7} {'AUC':>6} {'sm_top1':>8}"
        f" {'FC_frames':>10} {'FC%':>7} {'Useful':>7}"
    )
    print(hdr)
    print("-" * 80)
    for r in rows:
        conf_str = f"{r['sm_top1_mean']:.3f}" if r["has_telemetry"] else "n/a"
        useful_str = "YES" if r["useful_for_training"] else "no"
        print(
            f"{r['sequence']:<28} {r['n_frames']:>7} {r['auc']:>6.3f}"
            f" {conf_str:>8} {r['fc_frames']:>10}"
            f" {float(r['fc_pct']):>6.1f}% {useful_str:>7}"
        )

    if not rows:
        print("No sequences processed.")
        return

    n_useful = sum(1 for r in rows if r["useful_for_training"])
    mean_auc = float(np.mean([r["auc"] for r in rows]))
    mean_fc_pct = float(np.mean([float(r["fc_pct"]) for r in rows]))

    print()
    print(f"Total sequences:         {len(rows)}")
    print(f"Sequences with FC% > 5%: {n_useful}  ({100*n_useful/len(rows):.0f}%)")
    print(f"Overall mean AUC:        {mean_auc:.3f}")
    print(f"Overall mean FC%:        {mean_fc_pct:.1f}%")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
