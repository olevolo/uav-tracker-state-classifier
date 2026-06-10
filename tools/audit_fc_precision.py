"""FC Precision/Recall Audit Tool.

Computes per-sequence false_confirmed detection precision and recall
by comparing CSC state predictions against GT IoU.

Precision: of frames CSC predicted as FC, what fraction had IoU < tau_fail
Recall: of frames with true FC (IoU < tau_fail AND confidence >= tau_conf),
        what fraction did CSC correctly predict as FC

Usage:
    python tools/audit_fc_precision.py \
        --states_dir outputs/advisor_ablation/full_comparison/passive/.../states/ \
        --pred_dir   outputs/advisor_ablation/full_comparison/passive/.../predictions/ \
        --dataset dtb70 \
        [--tau_fail 0.2] [--tau_conf 0.0] [--output_csv outputs/fc_audit.csv]

Or with tracker run info:
    python tools/audit_fc_precision.py \
        --tracker sglatrack --dataset dtb70 \
        --csc_run_dir outputs/advisor_ablation/full_comparison/passive/sglatrack_dtb70_test_checkpoint_best

Outputs:
    - Per-sequence table: seq, n_fc_pred, precision, recall, true_fc_count
    - Summary CSV
    - Hard negatives: sequences where precision < threshold (CSC false alarms)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_bbox(line: str) -> list[float] | None:
    import re
    parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
    if len(parts) < 4:
        return None
    try:
        return [float(p) for p in parts[:4]]
    except ValueError:
        return None


def _iou(b1: list[float], b2: list[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _gt_root_for_dataset(dataset: str) -> Path:
    import os
    uav_root = Path(os.environ.get("UAV_DATA_ROOT", str(Path.home() / "uav-tracker-data")))
    mapping = {
        "dtb70": uav_root / "DTB70",
        "uav123": uav_root / "uav123" / "Dataset_UAV123" / "anno" / "full_anno",
        "visdrone_sot": uav_root / "VisDrone-SOT",
        "uavdt_sot": uav_root / "UAVDT",
    }
    return mapping.get(dataset, uav_root / dataset)


def _load_gt(dataset: str, seq_name: str) -> list[list[float]]:
    """Load GT bboxes for a sequence. Returns list of [x,y,w,h]."""
    gt_root = _gt_root_for_dataset(dataset)

    # DTB70 / UAVDT flat layout
    candidates = [
        gt_root / seq_name / "groundtruth_rect.txt",
        gt_root / seq_name / "groundtruth.txt",
        gt_root / "anno" / f"{seq_name}.txt",
    ]
    for p in candidates:
        if p.exists():
            bboxes = []
            for line in p.read_text().splitlines():
                b = _parse_bbox(line)
                if b:
                    bboxes.append(b)
            return bboxes
    return []


def audit_sequence(
    seq_name: str,
    states_file: Path,
    pred_file: Path,
    gt: list[list[float]],
    tau_fail: float = 0.2,
    tau_conf: float = 0.0,
) -> dict:
    """Compute FC precision/recall for one sequence."""
    rows = [json.loads(l) for l in states_file.read_text().splitlines() if l.strip()]
    preds_raw = [_parse_bbox(l) for l in pred_file.read_text().splitlines() if l.strip()]
    preds = [b for b in preds_raw if b is not None]

    n = min(len(rows), len(preds), len(gt))

    # CSC predicted FC frames (derived_state == 3)
    fc_pred_frames = []
    for r in rows:
        fi = r.get("frame_idx", 0)
        if r.get("derived_state") == 3 and fi < n:
            fc_pred_frames.append(fi)

    # True FC frames: IoU < tau_fail (ground truth failure)
    true_fc_frames = []
    for fi in range(1, n):  # skip init frame
        iou_val = _iou(preds[fi], gt[fi])
        if iou_val < tau_fail:
            true_fc_frames.append(fi)

    # Precision: of predicted FC, how many are real (IoU < tau_fail)
    if fc_pred_frames:
        correct = sum(
            1 for fi in fc_pred_frames
            if fi < n and _iou(preds[fi], gt[fi]) < tau_fail
        )
        precision = correct / len(fc_pred_frames)
    else:
        precision = float("nan")

    # Recall: of true FC, how many did CSC catch
    if true_fc_frames:
        fc_pred_set = set(fc_pred_frames)
        caught = sum(1 for fi in true_fc_frames if fi in fc_pred_set)
        recall = caught / len(true_fc_frames)
    else:
        recall = float("nan")

    # IoU distribution on predicted FC frames
    fc_ious = [_iou(preds[fi], gt[fi]) for fi in fc_pred_frames if fi < n]
    false_alarm_count = sum(1 for v in fc_ious if v > 0.5)

    return {
        "sequence": seq_name,
        "n_frames": n,
        "n_fc_pred": len(fc_pred_frames),
        "fc_pred_rate": len(fc_pred_frames) / n if n > 0 else 0.0,
        "n_true_fc": len(true_fc_frames),
        "true_fc_rate": len(true_fc_frames) / n if n > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": false_alarm_count / len(fc_ious) if fc_ious else float("nan"),
        "fc_iou_mean": float(np.mean(fc_ious)) if fc_ious else float("nan"),
        "fc_iou_min": float(np.min(fc_ious)) if fc_ious else float("nan"),
        "is_hard_negative": (
            len(fc_pred_frames) > 0
            and not np.isnan(precision)
            and precision < 0.3
            and len(true_fc_frames) == 0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit FC prediction precision/recall.")
    p.add_argument("--csc_run_dir", default=None,
                   help="Path to CSC run output dir (has states/ and predictions/ subdirs)")
    p.add_argument("--states_dir", default=None)
    p.add_argument("--pred_dir", default=None)
    p.add_argument("--dataset", required=True,
                   choices=["dtb70", "uav123", "visdrone_sot", "uavdt_sot"])
    p.add_argument("--tau_fail", type=float, default=0.2,
                   help="IoU threshold below which a frame is considered a true failure")
    p.add_argument("--tau_conf", type=float, default=0.0)
    p.add_argument("--output_csv", default=None)
    p.add_argument("--min_precision_warn", type=float, default=0.3,
                   help="Print warning for sequences with precision below this")
    args = p.parse_args(argv)

    if args.csc_run_dir:
        run_dir = Path(args.csc_run_dir)
        states_dir = run_dir / "states"
        pred_dir = run_dir / "predictions"
    else:
        if not args.states_dir or not args.pred_dir:
            print("ERROR: provide --csc_run_dir or both --states_dir and --pred_dir")
            return 1
        states_dir = Path(args.states_dir)
        pred_dir = Path(args.pred_dir)

    state_files = sorted(states_dir.glob("*.jsonl"))
    if not state_files:
        print(f"ERROR: no .jsonl files in {states_dir}")
        return 1

    results = []
    for sf in state_files:
        seq_name = sf.stem
        pf = pred_dir / f"{seq_name}.txt"
        if not pf.exists():
            print(f"  WARNING: predictions missing for {seq_name}")
            continue
        gt = _load_gt(args.dataset, seq_name)
        if not gt:
            print(f"  WARNING: GT not found for {seq_name} in {args.dataset}")
            continue

        r = audit_sequence(seq_name, sf, pf, gt, args.tau_fail, args.tau_conf)
        results.append(r)

        flag = " ← HARD NEG" if r["is_hard_negative"] else ""
        prec_str = f"{r['precision']:.2f}" if not np.isnan(r["precision"]) else "n/a "
        rec_str  = f"{r['recall']:.2f}"  if not np.isnan(r["recall"])  else "n/a "
        print(
            f"{seq_name:25s}  frames={r['n_frames']:4d}  "
            f"fc_pred={r['n_fc_pred']:3d}({r['fc_pred_rate']*100:5.1f}%)  "
            f"true_fc={r['n_true_fc']:3d}  "
            f"prec={prec_str}  rec={rec_str}  "
            f"false_alarm={r['false_alarm_count']:3d}{flag}"
        )

    if not results:
        print("No results computed.")
        return 1

    # Summary
    print()
    valid_prec = [r["precision"] for r in results if not np.isnan(r["precision"])]
    valid_rec  = [r["recall"]    for r in results if not np.isnan(r["recall"])]
    hard_negs  = [r["sequence"]  for r in results if r["is_hard_negative"]]

    print(f"Sequences audited:  {len(results)}")
    print(f"Mean precision:     {np.mean(valid_prec):.3f}" if valid_prec else "Mean precision: n/a")
    print(f"Mean recall:        {np.mean(valid_rec):.3f}"  if valid_rec  else "Mean recall: n/a")
    print(f"Hard negatives ({len(hard_negs)}): {', '.join(hard_negs) or 'none'}")

    if args.output_csv:
        import csv
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = list(results[0].keys())
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
