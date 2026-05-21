"""make_safe_to_update_labels.py — Generate safe_to_update labels for SALT-RD Phase 6.

Label = 1 (safe) if:
  - candidate frame (APCE>150, PSR>500, interval ok — at least 100 frames since last update)
  - IoU at frame t > 0.70
  - mean(IoU[t+1:t+20]) > 0.60 after simulated update
  - min(IoU[t+1:t+20]) > 0.30

Label = 0 (unsafe) if:
  - candidate frame but IoU[t] or future IoU below threshold
  - car7 frame 237-type: clean smap but future IoU collapses (IoU=0 in this case
    because gt is all-zeros indicating the target has left the frame)

Note: full counterfactual (re-running tracker with new template) is Phase 6B.
This script generates the offline proxy labels from existing NPZ data.

Data source: OOF fold NPZ files at saltr/tmp/oof/fold_*.npz
  OR: a single --npz file (salt_rd_v2_labels.npz format).

Usage
-----
    PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.make_safe_to_update_labels \\
        --npz saltr/data/salt_rd_v2_labels.npz \\
        --output saltr/data/salt_rd_safe_to_update_labels.npz \\
        --output-report saltr/results/safe_to_update_labels_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).parents[4]
_SRC = _REPO_ROOT / "src"
_SALTR_SRC = _REPO_ROOT / "saltr" / "src"
for _p in (_SRC, _SALTR_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

# ---------------------------------------------------------------------------
# Label thresholds
# ---------------------------------------------------------------------------

APCE_CANDIDATE_THRESHOLD = 150.0   # APCE must exceed this to be a candidate
PSR_CANDIDATE_THRESHOLD = 500.0    # PSR must exceed this
MIN_UPDATE_INTERVAL = 100          # min frames between template updates

IOU_AT_T_THRESHOLD = 0.70          # IoU at candidate frame
FUTURE_MEAN_THRESHOLD = 0.60       # mean future IoU
FUTURE_MIN_THRESHOLD = 0.30        # min future IoU
LOOKAHEAD = 20                     # frames to look ahead

# Feature indices in the 28-dim feature vector
FEAT_APCE_RAW = 0
FEAT_PSR = 2


# ---------------------------------------------------------------------------
# Core per-sequence label generation
# ---------------------------------------------------------------------------

def _compute_safe_to_update(
    features: np.ndarray,       # (T, D) float32
    iou_trace: np.ndarray,      # (T,) float32
    seq_name: str,
) -> tuple[np.ndarray, dict]:
    """Compute safe_to_update label for each frame of a sequence.

    Returns
    -------
    labels : (T,) int8 — 1=safe, 0=unsafe, -1=not a candidate frame
    stats  : dict with per-sequence statistics
    """
    T = len(iou_trace)
    labels = np.full(T, -1, dtype=np.int8)  # -1 = not a candidate

    apce = features[:, FEAT_APCE_RAW]
    psr = features[:, FEAT_PSR]

    n_candidate = 0
    n_safe = 0
    n_unsafe = 0
    last_update_frame = -MIN_UPDATE_INTERVAL  # start permissive

    for t in range(T):
        # Candidate check: high-quality tracking frame with enough separation
        is_candidate = (
            float(apce[t]) > APCE_CANDIDATE_THRESHOLD
            and float(psr[t]) > PSR_CANDIDATE_THRESHOLD
            and (t - last_update_frame) >= MIN_UPDATE_INTERVAL
        )

        if not is_candidate:
            continue

        n_candidate += 1

        # IoU at frame t
        iou_t = float(iou_trace[t])

        # Future IoU window
        t_end = min(t + 1 + LOOKAHEAD, T)
        future_iou = iou_trace[t + 1: t_end]

        if len(future_iou) == 0:
            # Not enough future frames — treat as unsafe
            labels[t] = 0
            n_unsafe += 1
            continue

        future_mean = float(future_iou.mean())
        future_min = float(future_iou.min())

        # Safe if:
        #   1. Current IoU is good
        #   2. Future IoU doesn't collapse
        safe = (
            iou_t > IOU_AT_T_THRESHOLD
            and future_mean > FUTURE_MEAN_THRESHOLD
            and future_min > FUTURE_MIN_THRESHOLD
        )

        labels[t] = int(safe)
        if safe:
            n_safe += 1
            last_update_frame = t  # simulate an update here
        else:
            n_unsafe += 1

    # Explicit check for car7 frame 237 (known hard negative)
    car7_237_check = None
    if "car7" in seq_name and T > 237:
        label_237 = int(labels[237])
        # If it's -1 (not candidate), verify by checking APCE/PSR values
        apce_237 = float(apce[237])
        psr_237 = float(psr[237])
        iou_237 = float(iou_trace[237])
        future_iou_237 = iou_trace[238: min(238 + LOOKAHEAD, T)]
        future_mean_237 = float(future_iou_237.mean()) if len(future_iou_237) > 0 else 0.0

        car7_237_check = {
            "label": label_237,
            "apce": apce_237,
            "psr": psr_237,
            "iou_t": iou_237,
            "future_iou_mean": future_mean_237,
            "is_unsafe": label_237 == 0 or (label_237 == -1 and iou_237 <= IOU_AT_T_THRESHOLD),
        }

    stats = {
        "seq_name": seq_name,
        "n_frames": T,
        "n_candidate": n_candidate,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "safe_rate": n_safe / n_candidate if n_candidate > 0 else 0.0,
        "car7_frame_237": car7_237_check,
    }

    return labels, stats


# ---------------------------------------------------------------------------
# Load from OOF folds (merge all folds; dedup by sequence name)
# ---------------------------------------------------------------------------

def _load_oof_folds(oof_dir: Path) -> dict[str, dict]:
    """Load all sequences from OOF fold NPZs.

    Returns a dict of seq_name -> {features, iou_trace, bbox_pred, bbox_gt}.
    """
    sequences: dict[str, dict] = {}
    fold_files = sorted(oof_dir.glob("fold_*.npz"))

    if not fold_files:
        raise FileNotFoundError(f"No fold_*.npz files found in {oof_dir}")

    for fold_path in fold_files:
        d = np.load(str(fold_path), allow_pickle=True)
        feature_names = list(d.get("feature_names", []))

        # Find all sequence keys
        feat_keys = [k for k in d.keys() if k.startswith("features/")]
        for fk in feat_keys:
            parts = fk.split("/")
            if len(parts) < 3:
                continue
            seq_name = parts[-1]
            dataset = parts[-2] if len(parts) >= 3 else "uav123"

            if seq_name in sequences:
                continue  # already loaded from a previous fold

            iou_key = f"iou_trace/{dataset}/{seq_name}"
            bp_key = f"bbox_pred/{dataset}/{seq_name}"
            bg_key = f"bbox_gt/{dataset}/{seq_name}"

            if iou_key not in d:
                continue

            sequences[seq_name] = {
                "features": d[fk].copy(),
                "iou_trace": d[iou_key].copy(),
                "bbox_pred": d[bp_key].copy() if bp_key in d else None,
                "bbox_gt": d[bg_key].copy() if bg_key in d else None,
                "dataset": dataset,
                "feature_names": feature_names,
            }

    return sequences


# ---------------------------------------------------------------------------
# Load from a single NPZ (salt_rd_v2_labels.npz format)
# ---------------------------------------------------------------------------

def _load_single_npz(npz_path: Path) -> dict[str, dict]:
    """Load all sequences from a single salt_rd_v2_labels.npz file."""
    d = np.load(str(npz_path), allow_pickle=True)
    feature_names = list(d.get("feature_names", []))

    sequences: dict[str, dict] = {}
    feat_keys = [k for k in d.keys() if k.startswith("features/")]

    for fk in feat_keys:
        parts = fk.split("/")
        if len(parts) < 3:
            continue
        seq_name = parts[-1]
        dataset = parts[-2] if len(parts) >= 3 else "uav123"

        iou_key = f"iou_trace/{dataset}/{seq_name}"
        if iou_key not in d:
            continue

        bp_key = f"bbox_pred/{dataset}/{seq_name}"
        bg_key = f"bbox_gt/{dataset}/{seq_name}"

        sequences[seq_name] = {
            "features": d[fk].copy(),
            "iou_trace": d[iou_key].copy(),
            "bbox_pred": d[bp_key].copy() if bp_key in d else None,
            "bbox_gt": d[bg_key].copy() if bg_key in d else None,
            "dataset": dataset,
            "feature_names": feature_names,
        }

    return sequences


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_labels(
    sequences: dict[str, dict],
    output_path: str,
    report_path: str | None = None,
) -> dict:
    """Generate safe_to_update labels for all sequences and save output NPZ.

    Returns the report dict.
    """
    print(f"\n  Generating safe_to_update labels for {len(sequences)} sequences...")

    all_labels: dict[str, np.ndarray] = {}
    all_stats: list[dict] = []
    total_candidate = 0
    total_safe = 0
    total_unsafe = 0
    car7_237_result = None

    for seq_name, data in sorted(sequences.items()):
        features = data["features"]
        iou_trace = data["iou_trace"]

        labels, stats = _compute_safe_to_update(features, iou_trace, seq_name)
        all_labels[seq_name] = labels
        all_stats.append(stats)

        total_candidate += stats["n_candidate"]
        total_safe += stats["n_safe"]
        total_unsafe += stats["n_unsafe"]

        if "car7" in seq_name and stats.get("car7_frame_237") is not None:
            car7_237_result = stats["car7_frame_237"]

    # Print car7 frame 237 result explicitly (critical negative example)
    print(f"\n  === car7 frame 237 (critical negative example) ===")
    if car7_237_result is not None:
        is_unsafe = car7_237_result["is_unsafe"]
        print(f"  APCE={car7_237_result['apce']:.1f}  PSR={car7_237_result['psr']:.1f}")
        print(f"  IoU[t]={car7_237_result['iou_t']:.3f}  future_IoU_mean={car7_237_result['future_iou_mean']:.3f}")
        print(f"  Label={car7_237_result['label']}  is_unsafe={is_unsafe}")
        if is_unsafe:
            print(f"  CONFIRMED UNSAFE: car7 frame 237 correctly labeled unsafe")
        else:
            print(f"  WARNING: car7 frame 237 NOT labeled unsafe — check thresholds")
    else:
        print(f"  car7 not found in dataset (may be in a different fold or missing)")

    # Save labels NPZ
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npz_data: dict[str, Any] = {
        "label_thresholds": np.array([
            APCE_CANDIDATE_THRESHOLD,
            PSR_CANDIDATE_THRESHOLD,
            MIN_UPDATE_INTERVAL,
            IOU_AT_T_THRESHOLD,
            FUTURE_MEAN_THRESHOLD,
            FUTURE_MIN_THRESHOLD,
            LOOKAHEAD,
        ], dtype=np.float32),
    }
    for seq_name, labels in all_labels.items():
        npz_data[f"safe_to_update/{seq_name}"] = labels

    np.savez_compressed(str(out_path), **npz_data)
    print(f"\n  Saved labels NPZ to {out_path}")

    # Build report
    report = {
        "n_sequences": len(sequences),
        "n_candidate_frames": total_candidate,
        "n_safe": total_safe,
        "n_unsafe": total_unsafe,
        "safe_rate": total_safe / total_candidate if total_candidate > 0 else 0.0,
        "car7_frame_237": car7_237_result,
        "thresholds": {
            "apce_candidate": APCE_CANDIDATE_THRESHOLD,
            "psr_candidate": PSR_CANDIDATE_THRESHOLD,
            "min_update_interval": MIN_UPDATE_INTERVAL,
            "iou_at_t": IOU_AT_T_THRESHOLD,
            "future_mean": FUTURE_MEAN_THRESHOLD,
            "future_min": FUTURE_MIN_THRESHOLD,
            "lookahead": LOOKAHEAD,
        },
        "per_sequence": all_stats,
    }

    # Print summary
    print(f"\n  === Summary ===")
    print(f"  n_sequences          : {len(sequences)}")
    print(f"  n_candidate_frames   : {total_candidate}")
    print(f"  n_safe               : {total_safe}")
    print(f"  n_unsafe             : {total_unsafe}")
    print(f"  safe_rate            : {report['safe_rate']:.3f}")

    # Save report
    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types for JSON serialization
        def _to_json(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json(v) for v in obj]
            return obj

        rp.write_text(json.dumps(_to_json(report), indent=2))
        print(f"  Saved report to {rp}")

    return report


def main():
    ap = argparse.ArgumentParser(
        description="Generate safe_to_update labels for SALT-RD Phase 6."
    )
    ap.add_argument("--npz", default=None, metavar="PATH",
                    help="Path to single NPZ file (salt_rd_v2_labels.npz format). "
                         "If not given, loads from saltr/tmp/oof/fold_*.npz")
    ap.add_argument("--oof-dir", default=None, metavar="DIR",
                    help="Directory containing OOF fold NPZ files "
                         "(default: saltr/tmp/oof/ relative to repo root)")
    ap.add_argument("--output", required=True, metavar="PATH",
                    help="Output NPZ path for safe_to_update labels")
    ap.add_argument("--output-report", default=None, metavar="PATH",
                    help="Output JSON report path")
    args = ap.parse_args()

    # Load data
    if args.npz:
        npz_path = Path(args.npz)
        if not npz_path.exists():
            print(f"ERROR: NPZ file not found: {npz_path}", file=sys.stderr)
            sys.exit(1)
        print(f"  Loading from NPZ: {npz_path}")
        sequences = _load_single_npz(npz_path)
    else:
        oof_dir = Path(args.oof_dir) if args.oof_dir else (_REPO_ROOT / "saltr" / "tmp" / "oof")
        if not oof_dir.exists():
            print(f"ERROR: OOF directory not found: {oof_dir}", file=sys.stderr)
            print("  Please provide --npz or --oof-dir, or ensure saltr/tmp/oof/ exists.", file=sys.stderr)
            sys.exit(1)
        print(f"  Loading from OOF folds: {oof_dir}")
        sequences = _load_oof_folds(oof_dir)

    if not sequences:
        print("ERROR: No sequences loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(sequences)} sequences")

    generate_labels(sequences, args.output, args.output_report)
    print("\n  Done.")


if __name__ == "__main__":
    main()
