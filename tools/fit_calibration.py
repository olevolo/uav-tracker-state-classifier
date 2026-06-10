"""Fit per-tracker percentile calibrators from raw telemetry JSONL files.

Usage
-----
::

    python tools/fit_calibration.py \\
        --tracker sglatrack \\
        --telemetry_dir outputs/baselines/sglatrack/got10k/val/telemetry \\
        --output_dir outputs/calibration \\
        --features confidence apce psr

Outputs (one file per feature with enough data, plus a manifest):

    outputs/calibration/sglatrack_got10k_confidence.json
    outputs/calibration/sglatrack_got10k_apce.json       # if apce present
    outputs/calibration/sglatrack_got10k_psr.json        # if psr present
    outputs/calibration/sglatrack_got10k.manifest.json

The dataset/split tag in the filename is derived from the last two path
components of ``telemetry_dir`` (e.g. ``got10k/val`` → ``got10k``).

Guard
-----
``--telemetry_dir`` must NOT contain ``uav123`` (case-insensitive) — UAV123
is the test-time benchmark and must never leak into calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csc_lib.csc.calibration import (  # noqa: E402
    PercentileConfidenceCalibrator,
    PercentileFeatureCalibrator,
)

_MIN_SAMPLES = 1_000  # same guard as the calibrator itself


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_feature(telemetry_dirs: list[Path], feature: str) -> np.ndarray:
    """Collect all finite values of *feature* from every *.jsonl in dirs."""
    values: list[float] = []
    for telemetry_dir in telemetry_dirs:
        for path in sorted(telemetry_dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = row.get(feature)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not (fv != fv) and fv == fv:  # basic NaN guard
                    values.append(fv)
    arr = np.array(values, dtype=np.float64)
    # drop non-finite values
    return arr[np.isfinite(arr)]


def _dataset_tag(telemetry_dir: Path) -> str:
    """Derive a short dataset label from the telemetry path.

    Examples
    --------
    ``…/got10k/val/telemetry``  → ``got10k``
    ``…/lasot/val/telemetry``   → ``lasot``
    ``…/lasot_train/telemetry`` → ``lasot_train``
    """
    parts = telemetry_dir.resolve().parts
    # Walk up from the telemetry leaf looking for a sensible dataset name
    # (skip generic suffixes like "telemetry", "val", "train", "test")
    skip = {"telemetry", "val", "train", "test"}
    for part in reversed(parts):
        if part.lower() not in skip:
            return part.lower()
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit per-tracker percentile calibrators from telemetry."
    )
    p.add_argument("--tracker", required=True, help="Tracker name (e.g. sglatrack).")
    p.add_argument(
        "--telemetry_dir",
        type=Path,
        help="Single telemetry directory (backward compat). Use --telemetry_dirs for multi.",
    )
    p.add_argument(
        "--telemetry_dirs",
        nargs="+",
        type=Path,
        help="One or more telemetry directories to pool (e.g. GOT-10k + DTB70 + VisDrone).",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Dataset tag for output filenames (auto-derived if omitted).",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/calibration",
        type=Path,
        help="Directory for saved calibrator JSON files.",
    )
    p.add_argument(
        "--features",
        nargs="+",
        default=["confidence", "apce", "psr"],
        help="Features to calibrate (default: confidence apce psr).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve telemetry directories
    if args.telemetry_dirs:
        telemetry_dirs = [Path(d).resolve() for d in args.telemetry_dirs]
    elif args.telemetry_dir:
        telemetry_dirs = [Path(args.telemetry_dir).resolve()]
    else:
        print("ERROR: provide --telemetry_dir or --telemetry_dirs", file=sys.stderr)
        return 1

    output_dir: Path = Path(args.output_dir).resolve()

    # ------------------------------------------------------------------
    # UAV123 guard
    # ------------------------------------------------------------------
    for td in telemetry_dirs:
        if "uav123" in str(td).lower():
            print(
                f"ERROR: telemetry_dir contains 'uav123': {td}. "
                "UAV123 is the test benchmark; calibration must use a "
                "non-test split (e.g. GOT-10k val, LaSOT train).",
                file=sys.stderr,
            )
            return 1
        if not td.exists():
            print(f"ERROR: telemetry_dir not found: {td}", file=sys.stderr)
            return 1

    all_jsonl: list[Path] = []
    for td in telemetry_dirs:
        all_jsonl.extend(td.glob("*.jsonl"))
    if not all_jsonl:
        print(f"ERROR: no *.jsonl files found in {telemetry_dirs}", file=sys.stderr)
        return 1

    # Dataset tag: use --tag if given, else join auto-derived tags
    if args.tag:
        dataset_tag = args.tag
    elif len(telemetry_dirs) == 1:
        dataset_tag = _dataset_tag(telemetry_dirs[0])
    else:
        dataset_tag = "_".join(_dataset_tag(td) for td in telemetry_dirs)

    tracker = args.tracker.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tracker : {tracker}")
    print(f"Dataset : {dataset_tag}")
    for td in telemetry_dirs:
        print(f"Source  : {td}")
    print(f"Total   : {len(all_jsonl)} files")
    print(f"Features: {args.features}")
    print()

    manifest: dict = {
        "tracker": tracker,
        "dataset": dataset_tag,
        "source_dirs": [str(td) for td in telemetry_dirs],
        "features": {},
    }

    for feat in args.features:
        print(f"--- {feat} ---")
        arr = _load_feature(telemetry_dirs, feat)

        if arr.size == 0:
            print(f"  No data for '{feat}' in telemetry — skipping.")
            print()
            continue

        if arr.size < _MIN_SAMPLES:
            print(
                f"  Only {arr.size} samples for '{feat}' "
                f"(need >= {_MIN_SAMPLES}) — skipping."
            )
            print()
            continue

        # Fit
        if feat == "confidence":
            cal: PercentileConfidenceCalibrator = PercentileConfidenceCalibrator()
        else:
            cal = PercentileFeatureCalibrator(name=feat)
        cal.fit(arr)

        # Save
        out_path = output_dir / f"{tracker}_{dataset_tag}_{feat}.json"
        cal.save(out_path)

        # Summary
        q05 = float(np.percentile(arr, 5))
        q50 = float(np.percentile(arr, 50))
        q95 = float(np.percentile(arr, 95))
        frac_above_065 = float((arr >= 0.65).mean())
        cal_above_065 = float((cal.transform(arr) >= 0.65).mean())

        print(f"  n_samples : {arr.size:,}")
        print(f"  min / max : {arr.min():.6f} / {arr.max():.6f}")
        print(f"  q05 / q50 / q95 (raw) : {q05:.6f} / {q50:.6f} / {q95:.6f}")
        if feat == "confidence":
            print(f"  frac raw >= 0.65      : {frac_above_065*100:.2f}%")
            print(f"  frac cal >= 0.65      : {cal_above_065*100:.2f}%")
        print(f"  saved -> {out_path}  ({out_path.stat().st_size} bytes)")
        print()

        manifest["features"][feat] = {
            "n_samples": arr.size,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "file": str(out_path),
        }

    # Save manifest
    manifest_path = output_dir / f"{tracker}_{dataset_tag}.manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest -> {manifest_path}")

    if not manifest["features"]:
        print("WARNING: no features were calibrated.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
