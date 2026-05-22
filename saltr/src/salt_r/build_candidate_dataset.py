"""build_candidate_dataset.py — collect and label per-candidate reinit events.

Runs SALTRunner over all training sequences with the CandidateEventLogger enabled.
For each proposed reinit candidate (accepted or rejected by geometry guard), records:
    source, bbox, scores, geometry ratio, cosine sim
Then labels each event offline:
    candidate_iou   = IoU(candidate_bbox, gt_bbox[frame])
    future_iou_gain = mean(iou_trace[t+1:t+20]) - iou_trace[t]
    label_good_candidate = int(candidate_iou > 0.3 and future_iou_gain > 0)

Output: saltr/data/candidate_events_labeled.npz
    keys: 'events'  — object array of dicts, one per event
          'stats'   — summary statistics (positive_rate, n_events, etc.)

Offline gate (required before BUG-26(b)/(c) training):
    positive_rate (IoU > 0.3) >= 5% of reinit events

Usage:
    PYTHONPATH=src:saltr/src python saltr/src/salt_r/build_candidate_dataset.py \\
        --config     configs/prod/salt.yaml \\
        --oracle-npz saltr/data/salt_rd_v2_labels.npz \\
        --dataset    uav123 \\
        --split      diagnostic \\
        --output     saltr/data/candidate_events_labeled.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _iou(a: tuple, b: tuple) -> float:
    """IoU between two (x,y,w,h) tuples."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


def _sequences_for_split(oracle_npz_path: str, split: str) -> list[str]:
    """Return sequence keys in the oracle NPZ for a given split."""
    data = np.load(oracle_npz_path, allow_pickle=True)
    split_key = f"splits/{split}"
    if split_key in data.files:
        return [str(s) for s in data[split_key]]
    # Fallback: return all feature keys
    return [k.replace("features/", "") for k in data.files if k.startswith("features/")]


def run(
    config_path: str,
    oracle_npz_path: str,
    dataset: str,
    split: str,
    output_path: str,
    max_seqs: int = 0,
) -> dict[str, Any]:
    """Collect and label candidate events. Returns summary stats."""
    from uav_tracker.salt_runner import SALTRunner
    from salt_r.candidate_events import CandidateEventLogger

    oracle_data = np.load(oracle_npz_path, allow_pickle=True)
    runner = SALTRunner.from_config(config_path)
    if runner.candidate_logger is None:
        runner.candidate_logger = CandidateEventLogger(enabled=True)
    else:
        runner.candidate_logger.enabled = True

    # Load dataset — uses __iter__ protocol
    if dataset == "uav123":
        from uav_tracker.datasets.uav123 import UAV123Dataset
        ds = UAV123Dataset()
    elif dataset == "dtb70":
        from uav_tracker.datasets.dtb70 import DTB70Dataset
        ds = DTB70Dataset()
    elif dataset == "visdrone_sot":
        from uav_tracker.datasets.visdrone_sot import VisDroneSOTDataset
        ds = VisDroneSOTDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Filter to sequences present in oracle NPZ (they have feature/label data)
    oracle_seq_names = {
        k.replace(f"features/{dataset}/", "")
        for k in oracle_data.files
        if k.startswith(f"features/{dataset}/")
    }

    all_events: list[dict] = []
    t0 = time.time()
    n_processed = 0

    for seq in ds:
        if max_seqs > 0 and n_processed >= max_seqs:
            break
        seq_name = seq.name
        if oracle_seq_names and seq_name not in oracle_seq_names:
            continue  # skip sequences not in oracle (e.g. diagnostic hold-outs)

        iou_key = f"iou_trace/{dataset}/{seq_name}"
        iou_trace = oracle_data[iou_key] if iou_key in oracle_data.files else None

        # GT bboxes come directly from the sequence object
        gt_bboxes = seq.ground_truth  # list or array of (x,y,w,h)

        runner.candidate_logger.reset(seq_id=seq_name)
        try:
            for _ in runner.run(seq):
                pass
        except Exception as exc:
            print(f"  [skip] {seq_name}: {exc}", file=sys.stderr)
            continue

        n_processed += 1

        # Label collected events with GT IoU and future utility
        for ev in runner.candidate_logger.events():
            d = ev.to_dict()
            t = ev.frame_idx

            if t < len(gt_bboxes):
                gt = gt_bboxes[t]
                # GT may be a BBox dataclass or array-like
                if hasattr(gt, 'x'):
                    gt_tuple = (float(gt.x), float(gt.y), float(gt.w), float(gt.h))
                else:
                    gt_tuple = (float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3]))
                d["candidate_iou"] = _iou(tuple(float(v) for v in ev.candidate_bbox), gt_tuple)
            else:
                d["candidate_iou"] = 0.0

            if iou_trace is not None and t < len(iou_trace):
                future = iou_trace[t + 1: t + 21]
                d["future_iou_gain"] = float(np.mean(future) - iou_trace[t]) if len(future) > 0 else 0.0
            else:
                d["future_iou_gain"] = 0.0

            d["label_good_candidate"] = int(
                d["candidate_iou"] > 0.3 and d["future_iou_gain"] > 0.0
            )
            all_events.append(d)

        print(f"  {seq_name}: {len(runner.candidate_logger.events())} events", flush=True)

    # Summary statistics
    n_total = len(all_events)
    n_accepted = sum(1 for e in all_events if e.get("accepted", False))
    n_good = sum(1 for e in all_events if e.get("label_good_candidate", 0))
    positive_rate = n_good / max(n_accepted, 1)

    stats = {
        "n_events": n_total,
        "n_accepted": n_accepted,
        "n_good_candidate": n_good,
        "positive_rate_of_accepted": positive_rate,
        "elapsed_s": time.time() - t0,
        "gate_pass": positive_rate >= 0.05,  # >= 5% positive rate required
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        events=np.array(all_events, dtype=object),
        stats=np.array([stats], dtype=object),
    )

    jsonl_path = output_path.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    print(f"\nSaved {n_total} events → {output_path}", flush=True)
    print(f"  accepted={n_accepted}  good_candidate={n_good}  positive_rate={positive_rate:.3f}", flush=True)
    print(f"  gate ({'PASS' if stats['gate_pass'] else 'FAIL'}): need positive_rate >= 0.05", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/prod/salt.yaml")
    ap.add_argument("--oracle-npz", default="saltr/data/salt_rd_v2_labels.npz")
    ap.add_argument("--dataset", default="uav123", choices=["uav123", "dtb70", "visdrone_sot"])
    ap.add_argument("--split", default="diagnostic")
    ap.add_argument("--output", default="saltr/data/candidate_events_labeled.npz")
    ap.add_argument("--max-seqs", type=int, default=0, help="0 = all sequences")
    args = ap.parse_args()

    stats = run(
        config_path=args.config,
        oracle_npz_path=args.oracle_npz,
        dataset=args.dataset,
        split=args.split,
        output_path=args.output,
        max_seqs=args.max_seqs,
    )
    sys.exit(0 if stats["gate_pass"] else 1)


if __name__ == "__main__":
    main()
