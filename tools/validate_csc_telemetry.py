"""Telemetry quality validator for CSC pipeline outputs.

Validates alignment between telemetry JSONL and prediction TXT files,
checks bbox validity, feature presence/completeness, and runtime sanity.

Usage
-----
    # New-style CLI (spec-compliant):
    python tools/validate_csc_telemetry.py \\
        --predictions outputs/_fast_gate/ortrack/lasot/train/predictions \\
        --telemetry outputs/_fast_gate/ortrack/lasot/train/telemetry \\
        --out outputs/_smoke_quality

    # Legacy CLI (also supported):
    python tools/validate_csc_telemetry.py \\
        --telemetry_dir outputs/_fast_gate/ortrack/lasot/train/telemetry \\
        --predictions_dir outputs/_fast_gate/ortrack/lasot/train/predictions \\
        --tracker ortrack --out outputs/_smoke_quality

Exit codes
----------
    0  PASS — all hard checks pass
    1  FAIL — any of:
         * frame misalignment > 1% on any sequence
         * bbox invalid rate > 1% across all frames
         * confidence missing on any sequence
         * confidence constant on any sequence
         * all-NaN feature column
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csc_lib.csc.features import FEATURE_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Per-sequence checks
# ---------------------------------------------------------------------------

_TELEMETRY_FEATURE_FIELDS = ["confidence", "apce", "psr", "response_entropy"]
_RUNTIME_FIELDS = ["latency_ms"]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_predictions(path: Path) -> list[Optional[list[float]]]:
    """Load bbox predictions from a .txt file (one bbox per line, comma-sep)."""
    bboxes: list[Optional[list[float]]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                bboxes.append(None)
                continue
            try:
                vals = [float(v) for v in line.split(",")]
                bboxes.append(vals if len(vals) == 4 else None)
            except ValueError:
                bboxes.append(None)
    return bboxes


def _is_finite(v) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _validate_sequence(
    seq_name: str,
    tel_path: Path,
    pred_path: Path,
) -> dict:
    """Run all per-sequence checks. Returns a result dict."""
    result: dict = {"sequence": seq_name, "errors": [], "warnings": []}

    tel_rows = _load_jsonl(tel_path)
    pred_bboxes = _load_predictions(pred_path)

    n_tel = len(tel_rows)
    n_pred = len(pred_bboxes)

    # ---- Check 1: Frame alignment ----
    alignment_ok = n_tel == n_pred
    mismatch_frac = abs(n_tel - n_pred) / max(1, max(n_tel, n_pred))
    result["n_telemetry"] = n_tel
    result["n_predictions"] = n_pred
    result["alignment_mismatch_frac"] = round(mismatch_frac, 6)
    if not alignment_ok:
        msg = f"Frame count mismatch: tel={n_tel}, pred={n_pred}"
        if mismatch_frac > 0.01:
            result["errors"].append(msg)
        else:
            result["warnings"].append(msg)

    # Work with min length to keep iteration safe
    n = min(n_tel, n_pred)
    tel_rows = tel_rows[:n]
    pred_bboxes = pred_bboxes[:n]

    # ---- Check 2: BBox validity ----
    bbox_invalid_count = 0
    for i, bbox in enumerate(pred_bboxes):
        if i == 0:
            continue  # Frame 0 is init — zero bbox is allowed
        if bbox is None:
            bbox_invalid_count += 1
            continue
        x, y, w, h = bbox
        if not (
            math.isfinite(x) and math.isfinite(y)
            and math.isfinite(w) and math.isfinite(h)
            and w > 0 and h > 0
        ):
            bbox_invalid_count += 1

    bbox_invalid_rate = bbox_invalid_count / max(1, n - 1)  # exclude frame 0
    result["bbox_invalid_rate"] = round(bbox_invalid_rate, 6)
    result["bbox_invalid_count"] = bbox_invalid_count
    if bbox_invalid_rate > 0.01:
        result["errors"].append(
            f"bbox invalid rate > 1%: {bbox_invalid_rate:.2%} ({bbox_invalid_count}/{n-1} frames)"
        )
    elif bbox_invalid_rate > 0.001:
        result["warnings"].append(
            f"Some invalid bboxes: {bbox_invalid_rate:.2%} ({bbox_invalid_count}/{n-1} frames)"
        )

    # ---- Check 3: Confidence presence and finiteness ----
    conf_present = sum(1 for r in tel_rows if r.get("confidence") is not None)
    conf_finite = sum(
        1 for r in tel_rows if r.get("confidence") is not None and _is_finite(r["confidence"])
    )
    apce_present = sum(1 for r in tel_rows if r.get("apce") is not None)
    psr_present = sum(1 for r in tel_rows if r.get("psr") is not None)
    entropy_present = sum(1 for r in tel_rows if r.get("response_entropy") is not None)

    result["confidence_present_rate"] = round(conf_present / max(1, n), 6)
    result["confidence_finite_rate"] = round(conf_finite / max(1, n), 6)
    result["apce_present_rate"] = round(apce_present / max(1, n), 6)
    result["psr_present_rate"] = round(psr_present / max(1, n), 6)
    result["response_entropy_present_rate"] = round(entropy_present / max(1, n), 6)

    if conf_present == 0:
        result["errors"].append("confidence field entirely missing in all telemetry rows")
    elif conf_present / n < 0.5:
        result["errors"].append(
            f"confidence present in only {conf_present/n:.1%} of frames (< 50%)"
        )

    # ---- Check 4: Constant features (std < 1e-9) ----
    constant_features: list[str] = []
    feature_stddev: dict[str, float] = {}
    feature_unique: dict[str, int] = {}

    # We check confidence, apce, psr from telemetry fields
    for field in ["confidence", "apce", "psr", "response_entropy"]:
        vals = [
            float(r[field]) for r in tel_rows
            if r.get(field) is not None and _is_finite(r.get(field, None))
        ]
        if len(vals) < 2:
            continue
        arr = np.array(vals)
        std = float(arr.std())
        uniq = len(set(vals))
        feature_stddev[field] = round(std, 8)
        feature_unique[field] = uniq
        if std < 1e-9:
            constant_features.append(field)

    result["constant_features"] = constant_features
    result["feature_stddev"] = feature_stddev
    result["feature_unique_counts"] = feature_unique
    if constant_features:
        result["warnings"].append(
            f"Constant/near-constant features detected: {constant_features}"
        )

    # Special check: all-constant confidence is a hard failure
    if "confidence" in constant_features:
        result["errors"].append("confidence is constant — dead weight in model")

    # ---- Check 5: Missing / NaN rates for key telemetry features ----
    missing_rates: dict[str, float] = {}
    nan_inf_rates: dict[str, float] = {}
    for field in ["confidence", "apce", "psr", "response_entropy"]:
        n_missing = sum(
            1 for r in tel_rows if r.get(field) is None or not _is_finite(r.get(field, None))
        )
        missing_rates[field] = round(n_missing / max(1, n), 6)
        n_nan_inf = sum(
            1 for r in tel_rows if r.get(field) is not None and not _is_finite(r.get(field, None))
        )
        nan_inf_rates[field] = round(n_nan_inf / max(1, n), 6)
    result["missing_rates"] = missing_rates
    result["nan_inf_rates"] = nan_inf_rates

    # Check: all-NaN feature (every present value is NaN/inf)
    all_nan_features = [
        k for k in ["confidence", "apce", "psr"]
        if missing_rates.get(k, 0.0) >= 1.0
    ]
    if all_nan_features:
        for f in all_nan_features:
            result["errors"].append(f"feature '{f}' is all-NaN/missing — no valid values in sequence")

    high_missing = {k: v for k, v in missing_rates.items() if v > 0.10}
    if high_missing:
        result["warnings"].append(
            f"Features with >10% missing: {high_missing}"
        )

    # ---- Check 6: Runtime / latency ----
    latencies = [
        float(r["latency_ms"])
        for r in tel_rows
        if r.get("latency_ms") is not None and _is_finite(r.get("latency_ms", None))
    ]
    result["latency_present_rate"] = round(len(latencies) / max(1, n), 6)
    if latencies:
        arr_lat = np.array(latencies)
        result["latency_mean_ms"] = round(float(arr_lat.mean()), 3)
        result["latency_median_ms"] = round(float(np.median(arr_lat)), 3)
        result["latency_min_ms"] = round(float(arr_lat.min()), 3)
        result["latency_max_ms"] = round(float(arr_lat.max()), 3)
        fps = 1000.0 / max(1e-6, float(arr_lat.mean()))
        result["est_fps"] = round(fps, 2)
        if fps < 0.1:
            result["errors"].append(f"Implausibly low FPS: {fps:.3f}")
        if fps > 10000:
            result["warnings"].append(f"Implausibly high FPS: {fps:.1f}")
    else:
        result["warnings"].append("latency_ms not present in any telemetry row")
        result["latency_mean_ms"] = None
        result["est_fps"] = None

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(seq_results: list[dict]) -> dict:
    """Aggregate per-sequence results into a summary."""
    n = len(seq_results)
    if n == 0:
        return {"n_sequences": 0, "status": "NO_DATA"}

    total_tel = sum(r["n_telemetry"] for r in seq_results)
    total_pred = sum(r["n_predictions"] for r in seq_results)

    mismatch_fracs = [r["alignment_mismatch_frac"] for r in seq_results]
    global_mismatch_frac = abs(total_tel - total_pred) / max(1, max(total_tel, total_pred))

    bbox_invalid_rates = [r["bbox_invalid_rate"] for r in seq_results]
    conf_present_rates = [r["confidence_present_rate"] for r in seq_results]
    apce_present_rates = [r["apce_present_rate"] for r in seq_results]
    psr_present_rates = [r["psr_present_rate"] for r in seq_results]
    fps_list = [r["est_fps"] for r in seq_results if r.get("est_fps") is not None]

    all_constant = set()
    for r in seq_results:
        all_constant.update(r.get("constant_features", []))

    n_errors = sum(len(r["errors"]) for r in seq_results)
    n_warnings = sum(len(r["warnings"]) for r in seq_results)

    # Hard failure checks
    hard_failures: list[str] = []
    # Spec: alignment mismatch on any sequence
    seqs_with_alignment_error = [
        r["sequence"] for r in seq_results if r["alignment_mismatch_frac"] > 0.01
    ]
    if seqs_with_alignment_error:
        hard_failures.append(
            f"alignment mismatch > 1% on {len(seqs_with_alignment_error)} sequence(s): "
            f"{seqs_with_alignment_error[:3]}"
        )
    # Spec: invalid-bbox rate > 1% globally
    global_bbox_invalid = float(np.mean(bbox_invalid_rates)) if bbox_invalid_rates else 0.0
    if global_bbox_invalid > 0.01:
        hard_failures.append(
            f"mean bbox invalid rate > 1%: {global_bbox_invalid:.2%}"
        )
    mean_conf_present = float(np.mean(conf_present_rates)) if conf_present_rates else 0.0
    # Spec: confidence missing on any sequence
    seqs_no_conf = [
        r["sequence"] for r in seq_results if r["confidence_present_rate"] == 0.0
    ]
    if seqs_no_conf:
        hard_failures.append(
            f"confidence entirely missing on {len(seqs_no_conf)} sequence(s): {seqs_no_conf[:3]}"
        )
    elif mean_conf_present < 0.5:
        hard_failures.append(
            f"mean confidence presence < 50%: {mean_conf_present:.2%}"
        )
    # Spec: confidence constant on any sequence
    if "confidence" in all_constant:
        hard_failures.append("confidence feature is constant across sequences")
    # Spec: all-NaN feature
    seqs_with_all_nan_errors = [
        r["sequence"] for r in seq_results
        if any("all-NaN" in e for e in r.get("errors", []))
    ]
    if seqs_with_all_nan_errors:
        hard_failures.append(
            f"all-NaN/missing feature on {len(seqs_with_all_nan_errors)} sequence(s)"
        )

    status = "FAIL" if hard_failures else "PASS"

    return {
        "n_sequences": n,
        "status": status,
        "hard_failures": hard_failures,
        "global_frame_mismatch_frac": round(global_mismatch_frac, 6),
        "total_telemetry_rows": total_tel,
        "total_prediction_rows": total_pred,
        "mean_bbox_invalid_rate": round(float(np.mean(bbox_invalid_rates)), 6),
        "mean_confidence_present_rate": round(mean_conf_present, 6),
        "mean_apce_present_rate": round(float(np.mean(apce_present_rates)), 6),
        "mean_psr_present_rate": round(float(np.mean(psr_present_rates)), 6),
        "constant_features_any_seq": sorted(all_constant),
        "mean_fps": round(float(np.mean(fps_list)), 2) if fps_list else None,
        "n_sequences_with_errors": sum(1 for r in seq_results if r["errors"]),
        "n_sequences_with_warnings": sum(1 for r in seq_results if r["warnings"]),
        "total_errors": n_errors,
        "total_warnings": n_warnings,
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_md_report(
    summary: dict,
    seq_results: list[dict],
    out_dir: Path,
    tracker: str,
) -> Path:
    md_path = out_dir / "telemetry_quality.md"
    lines: list[str] = []
    lines.append(f"# Telemetry Quality Report — `{tracker}`\n")
    lines.append(
        f"**Status**: {'PASS' if summary['status'] == 'PASS' else 'FAIL'} | "
        f"{summary['n_sequences']} sequences | "
        f"{summary['total_telemetry_rows']:,} telemetry rows\n"
    )

    if summary["hard_failures"]:
        lines.append("## Hard Failures\n")
        for hf in summary["hard_failures"]:
            lines.append(f"- {hf}")
        lines.append("")

    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Global frame mismatch | {summary['global_frame_mismatch_frac']:.4%} |")
    lines.append(f"| Mean bbox invalid rate | {summary['mean_bbox_invalid_rate']:.4%} |")
    lines.append(f"| Mean confidence present | {summary['mean_confidence_present_rate']:.4%} |")
    lines.append(f"| Mean APCE present | {summary['mean_apce_present_rate']:.4%} |")
    lines.append(f"| Mean PSR present | {summary['mean_psr_present_rate']:.4%} |")
    fps_str = f"{summary['mean_fps']:.1f}" if summary["mean_fps"] is not None else "N/A"
    lines.append(f"| Mean FPS | {fps_str} |")
    lines.append(
        f"| Constant features (any seq) | "
        f"{', '.join(summary['constant_features_any_seq']) or 'none'} |"
    )
    lines.append(f"| Sequences with errors | {summary['n_sequences_with_errors']} |")
    lines.append(f"| Sequences with warnings | {summary['n_sequences_with_warnings']} |")
    lines.append("")

    lines.append("## Per-Sequence Results\n")
    lines.append("| Sequence | Align | BBoxInvalid | ConfPresent | FPS | Errors | Warnings |")
    lines.append("|----------|-------|-------------|-------------|-----|--------|---------|")
    for r in seq_results:
        fps = f"{r['est_fps']:.1f}" if r.get("est_fps") else "N/A"
        lines.append(
            f"| {r['sequence']} "
            f"| {r['alignment_mismatch_frac']:.4%} "
            f"| {r['bbox_invalid_rate']:.4%} "
            f"| {r['confidence_present_rate']:.4%} "
            f"| {fps} "
            f"| {'; '.join(r['errors']) or '-'} "
            f"| {'; '.join(r['warnings']) or '-'} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate CSC pipeline telemetry/prediction alignment and quality."
    )
    # New-style CLI (spec-compliant short flags)
    parser.add_argument(
        "--telemetry", type=Path, default=None,
        help="Directory with <seq>.jsonl telemetry files (new-style CLI).",
    )
    parser.add_argument(
        "--predictions", type=Path, default=None,
        help="Directory with <seq>.txt prediction files (new-style CLI).",
    )
    # Legacy flags kept for backward compatibility
    parser.add_argument(
        "--telemetry_dir", type=Path, default=None,
        help="Directory with <seq>.jsonl telemetry files (legacy alias for --telemetry).",
    )
    parser.add_argument(
        "--predictions_dir", type=Path, default=None,
        help="Directory with <seq>.txt prediction files (legacy alias for --predictions).",
    )
    parser.add_argument(
        "--gt_dir", type=Path, default=None,
        help="(Optional) GT directory — currently unused, reserved for future checks.",
    )
    parser.add_argument(
        "--tracker", type=str, default=None,
        help="Tracker name (used for labeling output). Auto-detected if not supplied.",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for quality reports.",
    )
    args = parser.parse_args()

    # Resolve new-style vs legacy args
    telemetry_dir: Path | None = args.telemetry or args.telemetry_dir
    predictions_dir: Path | None = args.predictions or args.predictions_dir

    if telemetry_dir is None:
        print("ERROR: supply --telemetry <dir> or --telemetry_dir <dir>", file=sys.stderr)
        return 1
    if predictions_dir is None:
        print("ERROR: supply --predictions <dir> or --predictions_dir <dir>", file=sys.stderr)
        return 1

    tracker_name: str = args.tracker or telemetry_dir.parent.parent.name or "unknown"
    out_dir: Path = args.out

    if not telemetry_dir.exists():
        print(f"ERROR: telemetry_dir {telemetry_dir} does not exist", file=sys.stderr)
        return 1
    if not predictions_dir.exists():
        print(f"ERROR: predictions_dir {predictions_dir} does not exist", file=sys.stderr)
        return 1

    # Match sequences by stem
    tel_files = {p.stem: p for p in sorted(telemetry_dir.glob("*.jsonl"))}
    pred_files = {p.stem: p for p in sorted(predictions_dir.glob("*.txt"))}

    all_stems = sorted(tel_files.keys() | pred_files.keys())
    if not all_stems:
        print("ERROR: no telemetry or prediction files found", file=sys.stderr)
        return 1

    print(f"[validate_csc_telemetry] tracker={tracker_name}", flush=True)
    print(f"  telemetry_dir:   {telemetry_dir}", flush=True)
    print(f"  predictions_dir: {predictions_dir}", flush=True)
    print(f"  sequences found: {len(all_stems)}", flush=True)
    print("", flush=True)

    seq_results: list[dict] = []
    for stem in all_stems:
        tel_path = tel_files.get(stem)
        pred_path = pred_files.get(stem)
        if tel_path is None:
            seq_results.append({
                "sequence": stem,
                "errors": [f"missing telemetry file for {stem}"],
                "warnings": [],
                "n_telemetry": 0,
                "n_predictions": 0,
                "alignment_mismatch_frac": 1.0,
                "bbox_invalid_rate": 0.0,
                "confidence_present_rate": 0.0,
                "apce_present_rate": 0.0,
                "psr_present_rate": 0.0,
                "constant_features": [],
                "est_fps": None,
            })
            continue
        if pred_path is None:
            seq_results.append({
                "sequence": stem,
                "errors": [f"missing prediction file for {stem}"],
                "warnings": [],
                "n_telemetry": 0,
                "n_predictions": 0,
                "alignment_mismatch_frac": 1.0,
                "bbox_invalid_rate": 0.0,
                "confidence_present_rate": 0.0,
                "apce_present_rate": 0.0,
                "psr_present_rate": 0.0,
                "constant_features": [],
                "est_fps": None,
            })
            continue
        print(f"  Checking {stem} ...", flush=True)
        r = _validate_sequence(stem, tel_path, pred_path)
        seq_results.append(r)
        status_flag = "OK" if not r["errors"] else "ERR"
        warn_flag = f" ({len(r['warnings'])} warnings)" if r["warnings"] else ""
        print(
            f"    [{status_flag}] tel={r['n_telemetry']} pred={r['n_predictions']} "
            f"conf_rate={r['confidence_present_rate']:.2%} "
            f"fps={r.get('est_fps', 'N/A')}"
            f"{warn_flag}",
            flush=True,
        )

    summary = _aggregate(seq_results)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "telemetry_quality.json"
    with open(json_path, "w") as fh:
        json.dump({"summary": summary, "sequences": seq_results}, fh, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)

    md_path = _write_md_report(summary, seq_results, out_dir, tracker_name)
    print(f"Wrote {md_path}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print(f"  STATUS: {summary['status']}", flush=True)
    print(f"  Sequences: {summary['n_sequences']}", flush=True)
    print(f"  Global frame mismatch: {summary['global_frame_mismatch_frac']:.4%}", flush=True)
    print(
        f"  Mean confidence present: {summary['mean_confidence_present_rate']:.2%}",
        flush=True,
    )
    print(f"  Mean APCE present: {summary['mean_apce_present_rate']:.2%}", flush=True)
    print(f"  Constant features: {summary['constant_features_any_seq'] or 'none'}", flush=True)
    fps_str = f"{summary['mean_fps']:.1f}" if summary["mean_fps"] is not None else "N/A"
    print(f"  Mean FPS: {fps_str}", flush=True)
    if summary["hard_failures"]:
        print("\n  HARD FAILURES:", flush=True)
        for hf in summary["hard_failures"]:
            print(f"    - {hf}", flush=True)
    print("=" * 60, flush=True)

    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
