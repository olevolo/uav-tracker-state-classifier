"""Generate weak scene-state labels from baseline predictions + GT.

Inputs (per dataset/split):
    outputs/baselines/sglatrack/<dataset>/<split>/predictions/<seq>.txt
    outputs/baselines/sglatrack/<dataset>/<split>/telemetry/<seq>.jsonl

Outputs:
    outputs/csc_labels/<dataset>/<split>/labels.jsonl
    outputs/csc_labels/<dataset>/<split>/labels_per_sequence/<seq>.jsonl
    outputs/csc_labels/<dataset>/<split>/label_stats.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from csc_lib.csc.calibration import (  # noqa: E402
    PercentileConfidenceCalibrator,
    PercentileFeatureCalibrator,
)
from csc_lib.csc.labeling import LabelingThresholds, label_sequence
from csc_lib.csc.labeling.label_schema import DERIVED_NAMES, DerivedState
from csc_lib.csc.labeling.sequence_labeler import summarize_label_distribution


def _read_predictions(path: Path) -> list[Optional[tuple[float, float, float, float]]]:
    bboxes: list[Optional[tuple[float, float, float, float]]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                bboxes.append(None)
                continue
            parts = [p for p in line.replace("\t", ",").split(",") if p]
            try:
                vals = [float(p) for p in parts[:4]]
            except ValueError:
                bboxes.append(None)
                continue
            if len(vals) < 4 or vals[2] <= 0 or vals[3] <= 0:
                bboxes.append(None)
            else:
                bboxes.append(tuple(vals))
    return bboxes


def _read_telemetry(path: Path, n_frames: int) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    confidences: list[Optional[float]] = [None] * n_frames
    apces: list[Optional[float]] = [None] * n_frames
    psrs: list[Optional[float]] = [None] * n_frames
    if not path.exists():
        return confidences, apces, psrs
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = int(row.get("frame_idx", -1))
            if t < 0 or t >= n_frames:
                continue
            if "confidence" in row:
                confidences[t] = float(row["confidence"])
            if "apce" in row:
                apces[t] = float(row["apce"])
            if "psr" in row:
                psrs[t] = float(row["psr"])
    return confidences, apces, psrs


def _gt_from_sequence(seq) -> tuple[
    list[Optional[tuple[float, float, float, float]]],
    list[bool],
    list[bool],
    list[bool],
]:
    gt: list[Optional[tuple[float, float, float, float]]] = []
    full_occ: list[bool] = []
    oov: list[bool] = []
    absent: list[bool] = []
    occlusion_arr = getattr(seq, "full_occlusion", None)
    oov_arr = getattr(seq, "out_of_view", None)
    for i, bb in enumerate(seq.ground_truth):
        if bb is None or not getattr(bb, "valid", True) or bb.w <= 0 or bb.h <= 0:
            gt.append(None)
            absent.append(True)
        else:
            gt.append((float(bb.x), float(bb.y), float(bb.w), float(bb.h)))
            absent.append(False)
        full_occ.append(bool(occlusion_arr[i]) if occlusion_arr is not None and i < len(occlusion_arr) else False)
        oov.append(bool(oov_arr[i]) if oov_arr is not None and i < len(oov_arr) else False)
    return gt, full_occ, oov, absent


def _image_size_from_first_frame(seq) -> tuple[int, int]:
    first = next(iter(seq.frames))
    h, w = first.shape[:2]
    return int(w), int(h)


def _load_dataset(name: str, split: str):
    import csc_uav_tracking  # noqa: F401
    from csc_uav_tracking.registry import DATASETS

    if name == "got10k":
        return DATASETS.build(name, split=split)
    return DATASETS.build(name)


def _thresholds_from_config(path: Optional[Path]) -> LabelingThresholds:
    if path is None or not Path(path).exists():
        return LabelingThresholds()
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    fields = LabelingThresholds.__dataclass_fields__.keys()
    return LabelingThresholds(**{k: data[k] for k in fields if k in data})


def _load_calibrators(
    calibration_dir: Optional[Path],
    tracker: str,
    dataset: str,
    log: logging.Logger,
    tag: Optional[str] = None,
) -> tuple[Optional[PercentileConfidenceCalibrator], Optional[PercentileFeatureCalibrator], Optional[PercentileFeatureCalibrator]]:
    """Load saved calibrators for confidence, apce, psr from *calibration_dir*.

    Returns ``(conf_cal, apce_cal, psr_cal)`` — any may be ``None`` if the
    file does not exist.  Errors during load are logged as warnings (not fatal)
    so that existing call-sites are not broken.
    """
    if calibration_dir is None:
        return None, None, None

    cal_dir = Path(calibration_dir)
    prefix = tag if tag else f"{tracker}_{dataset}"

    def _try_load(feat: str, klass):
        p = cal_dir / f"{prefix}_{feat}.json"
        if not p.exists():
            return None
        try:
            cal = klass.load(p)
            log.info("Loaded calibrator: %s", p)
            return cal
        except Exception as exc:
            log.warning("Failed to load calibrator %s: %s", p, exc)
            return None

    conf_cal = _try_load("confidence", PercentileConfidenceCalibrator)
    apce_cal = _try_load("apce", PercentileFeatureCalibrator)
    psr_cal  = _try_load("psr",  PercentileFeatureCalibrator)
    return conf_cal, apce_cal, psr_cal


def _apply_calibrator(values: list[Optional[float]], cal) -> list[Optional[float]]:
    """Apply a fitted calibrator to a list, preserving ``None`` entries."""
    if cal is None:
        return values
    result: list[Optional[float]] = []
    for v in values:
        if v is None:
            result.append(None)
        else:
            result.append(float(cal.transform(float(v))))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate weak CSC labels.")
    p.add_argument("--dataset", required=True, choices=["lasot", "got10k", "uav123", "dtb70", "visdrone_sot", "uavdt_sot"])
    p.add_argument("--split", default="val")
    p.add_argument("--baseline_dir", default="outputs/baselines/sglatrack")
    p.add_argument("--tracker", default="sglatrack",
                   help="Tracker name used to look up calibrator files (default: sglatrack).")
    p.add_argument("--output_dir", default="outputs/csc_labels")
    p.add_argument("--threshold_config", default="configs/csc/labeling.yaml")
    p.add_argument("--calibration_dir", default=None, type=Path,
                   help="Directory containing pre-fitted calibrator JSON files "
                        "(e.g. outputs/calibration).  When set, raw telemetry "
                        "scores are mapped to rank-percentiles before labeling. "
                        "Files must follow the naming convention "
                        "<tracker>_<dataset>_<feature>.json.")
    p.add_argument("--calibrator_tag", default=None,
                   help="Override calibrator prefix (e.g. 'sglatrack_aerial_v2'). "
                        "Replaces the default '<tracker>_<dataset>' convention. "
                        "Use when the calibrator was fitted on a different dataset "
                        "than the one being labeled (e.g. aerial_v2 for UAV123).")
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--forecast_horizon", type=int, default=10,
                   help="Lookahead horizon for V3 proactive forecast labels "
                        "(failure_next_K, false_confirmed_next_K, lost_aware_next_K). "
                        "Default 10 matches paper.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("label")
    args = parse_args()

    pred_root = Path(args.baseline_dir) / args.dataset / args.split / "predictions"
    tel_root = Path(args.baseline_dir) / args.dataset / args.split / "telemetry"
    out_root = Path(args.output_dir) / args.dataset / args.split
    seq_dir = out_root / "labels_per_sequence"
    seq_dir.mkdir(parents=True, exist_ok=True)

    if not pred_root.exists():
        raise SystemExit(
            f"baseline predictions not found at {pred_root}. "
            f"Run tools/run_sglatrack_baseline.py first."
        )

    thresholds = _thresholds_from_config(Path(args.threshold_config))
    log.info("thresholds: %s", thresholds)

    # ------------------------------------------------------------------
    # Load calibrators (optional)
    # ------------------------------------------------------------------
    conf_cal, apce_cal, psr_cal = _load_calibrators(
        args.calibration_dir, args.tracker, args.dataset, log,
        tag=args.calibrator_tag,
    )
    if conf_cal is not None:
        _tag = args.calibrator_tag or f"{args.tracker}_{args.dataset}"
        log.info(
            "Confidence calibrator active (%s_confidence.json) — "
            "raw scores will be mapped to rank-percentiles before labeling.",
            _tag,
        )
    else:
        log.info("No calibration_dir / no confidence calibrator file found — "
                 "using raw scores.")

    # When APCE / PSR are run through the percentile calibrator they end up in
    # [0, 1]. The weak_labeler's defaults (tau_high_apce=30.0, tau_high_psr=8.0)
    # were set for raw (uncalibrated) values and would never fire on [0,1] data.
    # Adjust thresholds automatically to 0.5 (above-median = high).
    if apce_cal is not None:
        thresholds = dataclasses.replace(thresholds, tau_high_apce=0.5)
        log.info("APCE calibrated → tau_high_apce adjusted to 0.5")
    if psr_cal is not None:
        thresholds = dataclasses.replace(thresholds, tau_high_psr=0.5)
        log.info("PSR calibrated → tau_high_psr adjusted to 0.5")

    dataset = _load_dataset(args.dataset, args.split)
    sequences = list(dataset)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    all_jsonl = open(out_root / "labels.jsonl", "w")
    total_rows = 0
    state_counter: Counter[str] = Counter()
    aux_counter: Counter[str] = Counter()
    seq_summary: list[dict[str, Any]] = []
    # V3 forecast counters
    forecast_valid_total = 0
    forecast_ignored_total = 0
    forecast_failure_pos = 0
    forecast_fc_pos = 0
    forecast_lost_pos = 0

    try:
        for i, seq in enumerate(sequences):
            pred_path = pred_root / f"{seq.name}.txt"
            if not pred_path.exists():
                log.warning("[%d/%d] %s: missing predictions, skipping", i + 1, len(sequences), seq.name)
                continue

            pred_bboxes = _read_predictions(pred_path)
            gt_bboxes, full_occ, oov, absent = _gt_from_sequence(seq)

            n = min(len(gt_bboxes), len(pred_bboxes))
            if n == 0:
                continue
            gt_bboxes = gt_bboxes[:n]
            pred_bboxes = pred_bboxes[:n]
            full_occ = full_occ[:n]
            oov = oov[:n]
            absent = absent[:n]

            confidences, apces, psrs = _read_telemetry(tel_root / f"{seq.name}.jsonl", n)

            # ----------------------------------------------------------
            # Apply calibrators (if loaded)
            # ----------------------------------------------------------
            if conf_cal is not None:
                confidences = _apply_calibrator(confidences, conf_cal)
            if apce_cal is not None:
                apces = _apply_calibrator(apces, apce_cal)
            if psr_cal is not None:
                psrs = _apply_calibrator(psrs, psr_cal)

            try:
                img_size = _image_size_from_first_frame(seq)
            except Exception:
                img_size = (1280, 720)

            labels = label_sequence(
                dataset=args.dataset,
                split=args.split,
                sequence=seq.name,
                gt_bboxes=gt_bboxes,
                pred_bboxes=pred_bboxes,
                image_size=img_size,
                full_occlusion=full_occ,
                out_of_view=oov,
                absent=absent,
                confidences=confidences,
                apces=apces,
                psrs=psrs,
                thresholds=thresholds,
                forecast_horizon=int(args.forecast_horizon),
            )

            with open(seq_dir / f"{seq.name}.jsonl", "w") as fh_seq:
                for lab in labels:
                    j = lab.to_jsonable()
                    fh_seq.write(json.dumps(j) + "\n")
                    all_jsonl.write(json.dumps(j) + "\n")
                    state_counter[j["derived_state_name"]] += 1
                    for k, v in lab.aux.items():
                        if v:
                            aux_counter[k] += 1
                    total_rows += 1
                    # V3 forecast counters
                    if not lab.ignore_forecast:
                        forecast_valid_total += 1
                        if lab.failure_next_10:
                            forecast_failure_pos += 1
                        if lab.false_confirmed_next_10:
                            forecast_fc_pos += 1
                        if lab.lost_aware_next_10:
                            forecast_lost_pos += 1
                    else:
                        forecast_ignored_total += 1

            seq_stats = summarize_label_distribution(labels)
            seq_summary.append({"sequence": seq.name, **seq_stats})
            log.info(
                "[%d/%d] %s: %d frames; %s",
                i + 1, len(sequences), seq.name, len(labels),
                {k: v for k, v in seq_stats["state_counts"].items() if v > 0},
            )
    finally:
        all_jsonl.close()

    stats = {
        "n_sequences": len(seq_summary),
        "n_frames": total_rows,
        "state_counts": {k: state_counter.get(k, 0) for k in DERIVED_NAMES},
        "aux_counts": dict(aux_counter),
        "calibration_dir": str(args.calibration_dir) if args.calibration_dir else None,
        "warnings": [],
    }
    # V3 forecast distribution stats
    forecast_stats = {
        "horizon": int(args.forecast_horizon),
        "n_valid": forecast_valid_total,
        "n_ignored": forecast_ignored_total,
        "failure_next_K_pos": forecast_failure_pos,
        "false_confirmed_next_K_pos": forecast_fc_pos,
        "lost_aware_next_K_pos": forecast_lost_pos,
    }
    if forecast_valid_total > 0:
        forecast_stats["failure_rate"] = forecast_failure_pos / forecast_valid_total
        forecast_stats["false_confirmed_rate"] = forecast_fc_pos / forecast_valid_total
        forecast_stats["lost_aware_rate"] = forecast_lost_pos / forecast_valid_total
    stats["forecast"] = forecast_stats

    if total_rows > 0:
        for k in DERIVED_NAMES:
            frac = state_counter.get(k, 0) / total_rows
            if frac < 0.005:
                stats["warnings"].append(
                    f"state {k} support is only {frac*100:.2f}% — consider rebalancing"
                )
    with open(out_root / "label_stats.json", "w") as fh:
        json.dump({"summary": stats, "per_sequence": seq_summary}, fh, indent=2)
    log.info("done: %d frames, %d sequences -> %s", total_rows, len(seq_summary), out_root)
    log.info(
        "forecast (horizon=%d): valid=%d ignored=%d  failure=%.2f%%  fc=%.2f%%  lost=%.2f%%",
        forecast_stats["horizon"], forecast_valid_total, forecast_ignored_total,
        100.0 * forecast_stats.get("failure_rate", 0.0),
        100.0 * forecast_stats.get("false_confirmed_rate", 0.0),
        100.0 * forecast_stats.get("lost_aware_rate", 0.0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
