"""oracle_actions.py — Phase 5 Oracle Reinit Label Generator.

Generates per-frame oracle labels for the reinit action from pre-collected
OOF fold NPZ telemetry.  For each frame t in each sequence, a utility score
is computed by comparing future IoU improvement against a recent-past baseline.
Two binary labels are derived:

    label_reinit = 1  if utility > 0.03 AND current_iou < 0.5
    label_reject  = 1  if current_iou >= 0.5 OR utility < -0.01

Frames within 10 frames of sequence start or end are skipped (no valid window).

Flow features (indices 22-27) are zeroed via ``zero_production_features`` from
``feature_schema.py`` to match the production v3 no-flow schema.

Usage::

    PYTHONPATH=saltr/src .venv/bin/python -m salt_r.oracle_actions \\
        --fold-dir saltr/tmp/oof \\
        --output saltr/results/reinit_oracle_dataset.npz \\
        --summary saltr/results/reinit_oracle_dataset_summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from salt_r.feature_schema import zero_production_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard sequences as defined in SUPER_PLAN
HARD_SEQUENCES: List[str] = [
    "uav123/bike2",
    "uav123/uav2",
    "uav123/uav4",
    "uav123/uav6",
    "dtb70/Gull2",
    "dtb70/Sheep1",
    "dtb70/StreetBasketball1",
]

# Split priority for deduplication across folds: val > diagnostic > train
_SPLIT_PRIORITY: Dict[str, int] = {"val": 2, "diagnostic": 1, "train": 0}

# Number of frames to skip at each end of a sequence
EDGE_SKIP: int = 10


# ---------------------------------------------------------------------------
# Data loading (with deduplication and split-priority)
# ---------------------------------------------------------------------------

def load_all_sequences(fold_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load unique sequences from OOF fold NPZs, preferring higher-priority splits.

    Priority: val > diagnostic > train.  For sequences that appear as 'val' in
    all folds the first fold encountered is used (they are identical).  For
    sequences appearing as 'diagnostic' in one fold and 'train' in others, the
    diagnostic fold is preferred.

    Parameters
    ----------
    fold_dir:
        Directory containing fold_00.npz … fold_04.npz.

    Returns
    -------
    Dict keyed by ``"{dataset}/{seq_name}"`` with sub-keys:
        iou_trace, bbox_pred, bbox_gt, features, labels, split, dataset.
    """
    fold_dir = Path(fold_dir)
    all_data: Dict[str, Dict[str, Any]] = {}

    for i in range(5):
        fold_path = fold_dir / f"fold_0{i}.npz"
        if not fold_path.exists():
            continue
        fold = np.load(str(fold_path), allow_pickle=True)

        for key in fold.files:
            if not key.startswith("iou_trace/"):
                continue
            seq = key[len("iou_trace/"):]
            split = str(fold[f"split/{seq}"])
            new_prio = _SPLIT_PRIORITY.get(split, 0)

            if seq in all_data:
                # Only replace if this fold has a higher-priority split
                existing_prio = _SPLIT_PRIORITY.get(all_data[seq]["split"], 0)
                if new_prio <= existing_prio:
                    continue

            # Read dataset field (may be missing in older NPZs)
            dataset_key = f"dataset/{seq}"
            if dataset_key in fold.files:
                dataset = str(fold[dataset_key])
            else:
                # Infer from seq_key prefix
                dataset = seq.split("/")[0] if "/" in seq else "unknown"

            all_data[seq] = {
                "iou_trace": fold[f"iou_trace/{seq}"].astype(np.float32),
                "bbox_pred": fold[f"bbox_pred/{seq}"].astype(np.float32),
                "bbox_gt":   fold[f"bbox_gt/{seq}"].astype(np.float32),
                "features":  fold[f"features/{seq}"].astype(np.float32),
                "labels":    fold[f"labels/{seq}"],
                "split":     split,
                "dataset":   dataset,
            }

    return all_data


# ---------------------------------------------------------------------------
# Utility computation
# ---------------------------------------------------------------------------

def _safe_mean(arr: np.ndarray, start: int, end: int) -> float:
    """Return mean of arr[start:end], or 0.0 if window is empty."""
    start = max(0, start)
    end = min(len(arr), end)
    if start >= end:
        return 0.0
    return float(arr[start:end].mean())


def compute_utility(
    iou_trace: np.ndarray,
    t: int,
) -> Tuple[float, float, float]:
    """Compute per-frame oracle utility for reinit at frame *t*.

    Parameters
    ----------
    iou_trace:
        Full IoU trace for the sequence (float32, shape (N,)).
    t:
        Target frame index.

    Returns
    -------
    (utility, future_iou_gain_20, future_iou_gain_50)
    """
    current_iou = float(iou_trace[t])
    baseline_iou_recent = _safe_mean(iou_trace, t - 20, t)

    future_iou_gain_20 = _safe_mean(iou_trace, t + 1, t + 21) - baseline_iou_recent
    future_iou_gain_50 = _safe_mean(iou_trace, t + 1, t + 51) - baseline_iou_recent

    # Wrong reinit penalty: tracking fine, forced reinit would be harmful
    wrong_reinit_penalty = (
        1.0 if (current_iou >= 0.5 and future_iou_gain_20 < 0.0) else 0.0
    )
    # Fragmentation penalty: big bbox jump with no IoU gain
    fragmentation_penalty = 0.05 if future_iou_gain_20 < 0.01 else 0.0

    utility = (
        future_iou_gain_50
        + 0.5 * future_iou_gain_20
        - wrong_reinit_penalty
        - fragmentation_penalty
    )
    return utility, future_iou_gain_20, future_iou_gain_50


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def derive_labels(
    current_iou: float,
    utility: float,
) -> Tuple[int, int]:
    """Derive binary oracle labels from current IoU and utility score.

    Returns
    -------
    (label_reinit, label_reject)
    """
    label_reinit = int(utility > 0.03 and current_iou < 0.5)
    label_reject  = int(current_iou >= 0.5 or utility < -0.01)
    return label_reinit, label_reject


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq_key: str,
    data: Dict[str, Any],
    *,
    edge_skip: int = EDGE_SKIP,
) -> List[Dict[str, Any]]:
    """Generate per-frame records for one sequence.

    Frames within *edge_skip* of sequence start or end are skipped (no valid
    future IoU window available).  Flow features are zeroed via
    ``zero_production_features``.

    Parameters
    ----------
    seq_key:
        Sequence identifier string, e.g. ``"uav123/bike1"``.
    data:
        Dict with fields: iou_trace, features, split, dataset.
    edge_skip:
        Number of frames to skip at start and end.

    Returns
    -------
    List of per-frame record dicts.
    """
    iou_trace: np.ndarray = data["iou_trace"]
    features: np.ndarray = data["features"]
    split: str = data["split"]
    dataset: str = data["dataset"]
    n = len(iou_trace)

    records: List[Dict[str, Any]] = []

    for t in range(edge_skip, n - edge_skip):
        current_iou = float(iou_trace[t])
        utility, future_gain_20, future_gain_50 = compute_utility(iou_trace, t)
        label_reinit, label_reject = derive_labels(current_iou, utility)

        # Zero flow features for production consistency
        feat = zero_production_features(features[t])

        records.append({
            "sequence_key":      seq_key,
            "frame_idx":         t,
            "current_iou":       current_iou,
            "future_iou_gain_20": future_gain_20,
            "future_iou_gain_50": future_gain_50,
            "utility":           utility,
            "label_reinit":      label_reinit,
            "label_reject":      label_reject,
            "features":          feat,
            "split":             split,
            "dataset":           dataset,
        })

    return records


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_oracle_dataset(
    all_data: Dict[str, Dict[str, Any]],
    *,
    edge_skip: int = EDGE_SKIP,
) -> Dict[str, np.ndarray]:
    """Build the oracle dataset arrays from all sequences.

    Parameters
    ----------
    all_data:
        Mapping from seq_key to sequence data dicts (from ``load_all_sequences``).
    edge_skip:
        Edge frames to skip per sequence.

    Returns
    -------
    Dict of numpy arrays suitable for ``np.savez``.
    """
    all_records: List[Dict[str, Any]] = []
    for seq_key, data in sorted(all_data.items()):
        records = process_sequence(seq_key, data, edge_skip=edge_skip)
        all_records.extend(records)

    if not all_records:
        raise ValueError("No records generated — check fold directory and data.")

    m = len(all_records)
    sequence_keys    = np.empty(m, dtype=object)
    frame_indices    = np.empty(m, dtype=np.int32)
    current_iou_arr  = np.empty(m, dtype=np.float32)
    utility_arr      = np.empty(m, dtype=np.float32)
    gain_20_arr      = np.empty(m, dtype=np.float32)
    gain_50_arr      = np.empty(m, dtype=np.float32)
    label_reinit_arr = np.empty(m, dtype=np.int8)
    label_reject_arr = np.empty(m, dtype=np.int8)
    features_arr     = np.empty((m, 28), dtype=np.float32)
    splits_arr       = np.empty(m, dtype=object)
    datasets_arr     = np.empty(m, dtype=object)

    for i, rec in enumerate(all_records):
        sequence_keys[i]    = rec["sequence_key"]
        frame_indices[i]    = rec["frame_idx"]
        current_iou_arr[i]  = rec["current_iou"]
        utility_arr[i]      = rec["utility"]
        gain_20_arr[i]      = rec["future_iou_gain_20"]
        gain_50_arr[i]      = rec["future_iou_gain_50"]
        label_reinit_arr[i] = rec["label_reinit"]
        label_reject_arr[i] = rec["label_reject"]
        features_arr[i]     = rec["features"]
        splits_arr[i]       = rec["split"]
        datasets_arr[i]     = rec["dataset"]

    return {
        "sequence_keys":      sequence_keys,
        "frame_indices":      frame_indices,
        "current_iou":        current_iou_arr,
        "utility":            utility_arr,
        "future_iou_gain_20": gain_20_arr,
        "future_iou_gain_50": gain_50_arr,
        "label_reinit":       label_reinit_arr,
        "label_reject":       label_reject_arr,
        "features":           features_arr,
        "splits":             splits_arr,
        "datasets":           datasets_arr,
    }


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    """Return current git short commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_summary(
    arrays: Dict[str, np.ndarray],
    all_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the JSON summary matching the SUPER_PLAN schema.

    Parameters
    ----------
    arrays:
        Output of ``build_oracle_dataset``.
    all_data:
        Mapping from seq_key to sequence data (used for missing seq detection).

    Returns
    -------
    Summary dict suitable for JSON serialization.
    """
    seq_keys_arr  = arrays["sequence_keys"]
    label_reinit  = arrays["label_reinit"]
    label_reject  = arrays["label_reject"]
    splits_arr    = arrays["splits"]
    datasets_arr  = arrays["datasets"]

    n_frames    = int(len(seq_keys_arr))
    unique_seqs = list(sorted(set(str(s) for s in seq_keys_arr)))
    n_sequences = len(unique_seqs)

    candidate_mask = (label_reinit == 1) | (label_reject == 1)
    n_candidate    = int(candidate_mask.sum())
    n_reinit_pos   = int((label_reinit == 1).sum())
    reinit_rate    = float(n_reinit_pos) / float(n_frames) if n_frames > 0 else 0.0

    # Per-dataset base rates
    per_dataset: Dict[str, Any] = {}
    for ds in sorted(set(str(d) for d in datasets_arr)):
        mask = datasets_arr == ds
        ds_total    = int(mask.sum())
        ds_reinit   = int(label_reinit[mask].sum())
        per_dataset[ds] = {
            "n_frames":       ds_total,
            "n_reinit_pos":   ds_reinit,
            "reinit_pos_rate": round(ds_reinit / ds_total, 6) if ds_total > 0 else 0.0,
        }

    # Per-split counts
    per_split: Dict[str, Any] = {}
    for sp in sorted(set(str(s) for s in splits_arr)):
        mask = splits_arr == sp
        sp_total  = int(mask.sum())
        sp_reinit = int(label_reinit[mask].sum())
        per_split[sp] = {
            "n_frames":     sp_total,
            "n_reinit_pos": sp_reinit,
        }

    # Missing sequences (not in any fold)
    missing_seqs = [s for s in HARD_SEQUENCES if s not in all_data]

    # Hard sequence analysis
    hard_included = [s for s in HARD_SEQUENCES if s in all_data]
    hard_missing  = missing_seqs

    # Max utility per sequence
    max_utility_by_seq: Dict[str, float] = {}
    for seq in unique_seqs:
        mask = seq_keys_arr == seq
        if mask.any():
            max_utility_by_seq[seq] = float(arrays["utility"][mask].max())

    return {
        "n_sequences":             n_sequences,
        "n_frames":                n_frames,
        "n_candidate_frames":      n_candidate,
        "n_reinit_positive":       n_reinit_pos,
        "reinit_positive_rate":    round(reinit_rate, 6),
        "per_dataset_base_rates":  per_dataset,
        "per_split_counts":        per_split,
        "missing_sequences":       missing_seqs,
        "hard_sequences_included": hard_included,
        "hard_sequences_missing":  hard_missing,
        "max_utility_by_sequence": {k: round(v, 5) for k, v in sorted(max_utility_by_seq.items())},
        "git_commit":              _get_git_commit(),
        "created_at":              datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Summary table (stdout)
# ---------------------------------------------------------------------------

def print_summary_table(summary: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    """Print a human-readable summary table to stdout."""
    print()
    print("Oracle Reinit Dataset Summary")
    print("=" * 70)
    print(f"  Sequences         : {summary['n_sequences']}")
    print(f"  Total frames      : {summary['n_frames']:,}")
    print(f"  Candidate frames  : {summary['n_candidate_frames']:,}"
          f"  ({100 * summary['n_candidate_frames'] / max(summary['n_frames'], 1):.1f}%)")
    print(f"  Reinit positives  : {summary['n_reinit_positive']:,}"
          f"  ({100 * summary['reinit_positive_rate']:.3f}%)")
    print()

    # Per-dataset breakdown
    print(f"  {'Dataset':<22} {'Frames':>10} {'Reinit+':>10} {'Rate':>8}")
    print("  " + "-" * 52)
    for ds, info in sorted(summary["per_dataset_base_rates"].items()):
        print(f"  {ds:<22} {info['n_frames']:>10,} {info['n_reinit_pos']:>10,} {100*info['reinit_pos_rate']:>7.3f}%")

    print()
    # Per-split breakdown
    print(f"  {'Split':<14} {'Frames':>10} {'Reinit+':>10}")
    print("  " + "-" * 36)
    for sp, info in sorted(summary["per_split_counts"].items()):
        print(f"  {sp:<14} {info['n_frames']:>10,} {info['n_reinit_pos']:>10,}")

    print()
    # Hard sequence availability
    print(f"  Hard sequences included : {len(summary['hard_sequences_included'])}")
    for s in summary["hard_sequences_included"]:
        print(f"    [OK] {s}")
    if summary["hard_sequences_missing"]:
        print(f"  Hard sequences MISSING  : {len(summary['hard_sequences_missing'])}")
        for s in summary["hard_sequences_missing"]:
            print(f"    [!!] {s}")

    print()
    # SUPER_PLAN stop condition check
    rate = summary["reinit_positive_rate"]
    if rate < 0.005:
        print("  STOP CONDITION: reinit_positive_rate < 0.5% -> switch to ranking loss")
    else:
        print(f"  GO: reinit_positive_rate = {100*rate:.3f}% (>= 0.5% threshold)")
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    fold_dir: str = "saltr/tmp/oof",
    output_path: str = "saltr/results/reinit_oracle_dataset.npz",
    summary_path: str = "saltr/results/reinit_oracle_dataset_summary.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run oracle action label generation.

    Parameters
    ----------
    fold_dir:
        Directory containing OOF fold NPZs.
    output_path:
        Path to save the output NPZ.
    summary_path:
        Path to save the JSON summary.
    verbose:
        If True, print a summary table to stdout.

    Returns
    -------
    Summary dict.
    """
    if verbose:
        print(f"Loading sequences from {fold_dir} ...")
    all_data = load_all_sequences(fold_dir)
    if verbose:
        print(f"  Loaded {len(all_data)} unique sequences.")

    if verbose:
        print("Computing per-frame oracle labels ...")
    arrays = build_oracle_dataset(all_data)

    if verbose:
        print(f"  Generated {len(arrays['sequence_keys']):,} frame records.")

    summary = build_summary(arrays, all_data)

    # Save NPZ
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **arrays)
    if verbose:
        print(f"  Saved oracle dataset: {out_path}")

    # Save JSON summary
    sum_path = Path(summary_path)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(sum_path), "w") as fh:
        json.dump(summary, fh, indent=2)
    if verbose:
        print(f"  Saved summary JSON  : {sum_path}")

    if verbose:
        print_summary_table(summary, arrays)

    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 5 Oracle Reinit Label Generator for SALT-RD UAV tracker."
    )
    parser.add_argument(
        "--fold-dir",
        default="saltr/tmp/oof",
        help="Directory containing OOF fold NPZs (fold_00.npz … fold_04.npz).",
    )
    parser.add_argument(
        "--output",
        default="saltr/results/reinit_oracle_dataset.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--summary",
        default="saltr/results/reinit_oracle_dataset_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary table output.",
    )
    args = parser.parse_args()
    run(
        fold_dir=args.fold_dir,
        output_path=args.output,
        summary_path=args.summary,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
