"""Compute tracking + failure + runtime metrics for one prediction set."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _read_predictions(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                rows.append([0.0, 0.0, 0.0, 0.0])
                continue
            parts = [p for p in line.replace("\t", ",").split(",") if p]
            try:
                vals = [float(p) for p in parts[:4]]
            except ValueError:
                vals = [0.0, 0.0, 0.0, 0.0]
            if len(vals) < 4:
                vals = vals + [0.0] * (4 - len(vals))
            rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def _gt_array(seq) -> np.ndarray:
    out: list[list[float]] = []
    for bb in seq.ground_truth:
        if bb is None or not getattr(bb, "valid", True):
            out.append([0.0, 0.0, 0.0, 0.0])
        else:
            out.append([float(bb.x), float(bb.y), float(bb.w), float(bb.h)])
    return np.asarray(out, dtype=np.float64)


def _read_telemetry(path: Path, n: int) -> np.ndarray:
    """Return per-frame latency array; missing frames = 0."""
    out = np.zeros(n, dtype=np.float64)
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line.strip())
            except Exception:
                continue
            t = int(row.get("frame_idx", -1))
            if 0 <= t < n and "latency_ms" in row:
                out[t] = float(row["latency_ms"])
    return out


def _load_dataset(name: str, split: str):
    import csc_uav_tracking  # noqa: F401
    from csc_uav_tracking.registry import DATASETS

    if name == "got10k":
        return DATASETS.build(name, split=split)
    return DATASETS.build(name)


def _image_diag_from_first_frame(seq) -> float:
    first = next(iter(seq.frames))
    h, w = first.shape[:2]
    return math.hypot(float(w), float(h))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate tracking results.")
    p.add_argument("--dataset", required=True, choices=["lasot", "got10k", "uav123", "dtb70", "visdrone_sot"])
    p.add_argument("--split", default="val")
    p.add_argument("--pred_dir", required=True, help="Directory containing <seq>.txt files.")
    p.add_argument("--telemetry_dir", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument(
        "--failure_threshold", type=float, default=0.2,
        help="IoU threshold below which a frame counts as a failure."
    )
    p.add_argument(
        "--severe_threshold", type=float, default=0.1,
    )
    return p.parse_args()


def main() -> int:
    # Defer the heavy imports so the file's top-level walrus stays valid.
    from csc_lib.eval.custom_metrics.bbox import iou_xywh_batch
    from csc_lib.eval.custom_metrics.failure_metrics import (
        failure_summary,
        hard_frame_auc,
        post_first_failure_auc,
    )
    from csc_lib.eval.custom_metrics.runtime_metrics import latency_summary
    from csc_lib.eval.custom_metrics.tracking_metrics import (
        average_overlap,
        compute_per_frame_arrays,
        frame_weighted_average,
        macro_average,
        normalized_precision_auc,
        per_sequence_metrics,
        precision_at_threshold,
        success_auc,
        success_rate,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("eval")
    args = parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        raise SystemExit(f"pred_dir not found: {pred_dir}")
    tel_dir = Path(args.telemetry_dir) if args.telemetry_dir else None

    dataset = _load_dataset(args.dataset, args.split)
    sequences = list(dataset)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    seq_results: dict[str, dict] = {}
    seq_failures: list[dict] = []
    all_latencies: list[float] = []

    per_sequence_csv = ["sequence,n_frames,auc,precision_20,norm_precision_auc,ao,sr_50,sr_75,fps,n_failures,total_failure_frames,hard_auc,post_first_failure_auc,mean_latency_ms"]

    for i, seq in enumerate(sequences):
        pred_path = pred_dir / f"{seq.name}.txt"
        if not pred_path.exists():
            log.warning("[%d/%d] %s: missing prediction file", i + 1, len(sequences), seq.name)
            continue
        preds = _read_predictions(pred_path)
        gts = _gt_array(seq)
        n = min(len(preds), len(gts))
        if n == 0:
            continue
        preds = preds[:n]
        gts = gts[:n]
        try:
            img_diag = _image_diag_from_first_frame(seq)
        except Exception:
            img_diag = 1280.0

        ious, ce, ne = compute_per_frame_arrays(preds, gts, image_diag=img_diag)

        latencies = (
            _read_telemetry(tel_dir / f"{seq.name}.jsonl", n)
            if tel_dir is not None
            else np.zeros(n, dtype=np.float64)
        )
        all_latencies.extend(latencies.tolist())

        seq_stats = {
            "n_frames": n,
            "ious": ious.tolist(),
            "center_errors": ce.tolist(),
            "normalized_center_errors": ne.tolist(),
            "time_seconds": float(np.nansum(latencies) / 1000.0),
        }
        seq_results[seq.name] = seq_stats

        fail_sum = failure_summary(
            ious,
            threshold=args.failure_threshold,
            severe_threshold=args.severe_threshold,
        )
        seq_failures.append({"sequence": seq.name, **fail_sum})

        per_sequence_csv.append(
            ",".join(
                str(x)
                for x in (
                    seq.name,
                    n,
                    f"{success_auc(ious):.4f}",
                    f"{precision_at_threshold(ce, 20.0):.4f}",
                    f"{normalized_precision_auc(ne):.4f}",
                    f"{average_overlap(ious):.4f}",
                    f"{success_rate(ious, 0.5):.4f}",
                    f"{success_rate(ious, 0.75):.4f}",
                    f"{(n / max(1e-6, seq_stats['time_seconds'])):.2f}",
                    fail_sum["n_failures"],
                    fail_sum["total_failure_frames"],
                    f"{hard_frame_auc(ious, threshold=args.failure_threshold):.4f}",
                    f"{post_first_failure_auc(ious, threshold=args.failure_threshold):.4f}",
                    f"{float(np.nanmean(latencies)) if latencies.size else 0.0:.2f}",
                )
            )
        )
        log.info("[%d/%d] %s done", i + 1, len(sequences), seq.name)

    per_seq = per_sequence_metrics(seq_results)
    macro = {k: macro_average(per_seq, k) for k in ["auc", "precision_20", "norm_precision_auc", "ao", "sr_50", "sr_75", "fps"]}
    weighted = {k: frame_weighted_average(per_seq, k) for k in ["auc", "precision_20", "norm_precision_auc", "ao", "sr_50", "sr_75"]}

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "n_sequences": len(per_seq),
        "n_frames": int(sum(v["n_frames"] for v in per_seq.values())),
        "macro": macro,
        "frame_weighted": weighted,
        "failure_threshold": args.failure_threshold,
        "severe_threshold": args.severe_threshold,
        "n_failures_total": int(sum(f["n_failures"] for f in seq_failures)),
        "total_failure_frames": int(sum(f["total_failure_frames"] for f in seq_failures)),
        "runtime": latency_summary(np.asarray(all_latencies)),
    }

    (out_root / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "metrics_per_sequence.csv").write_text("\n".join(per_sequence_csv) + "\n")
    (out_root / "failure_events.json").write_text(json.dumps(seq_failures, indent=2))

    log.info("summary: AUC=%.4f Precision@20=%.4f failures=%d -> %s",
             macro["auc"], macro["precision_20"], summary["n_failures_total"], out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
