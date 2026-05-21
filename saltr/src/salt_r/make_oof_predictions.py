"""make_oof_predictions.py — Out-of-fold (OOF) prediction pipeline for SALT-RD.

Produces per-frame predictions for the *train split* that are never in-sample
for the model that generated them.  The merged JSON replaces
``preds_all_v2_retrained.json`` as the prediction source for SGLA memory
sidecar extraction.

Pipeline overview
-----------------
Phase 1 : Load the canonical NPZ, collect train sequences, assign each to one
          of N folds using a deterministic round-robin stratified by dataset.
Phase 2 : For each fold k, write a temporary NPZ with modified split labels,
          train a fold model (on train − fold_k), and eval it on fold_k.
Phase 3 : Merge the 5 fold prediction files → ``preds_train_oof_v2.json``.
Phase 4 : Generate val / diagnostic predictions from the canonical teacher
          checkpoint (no memory sidecar → no leakage).
Phase 5 : Merge everything → ``preds_all_v2_oof_teacher.json``.

Usage::

    # Full pipeline
    python -m salt_r.make_oof_predictions \\
        --npz saltr/data/salt_rd_v2_labels.npz \\
        --teacher-checkpoint saltr/checkpoints/v2_corrected/saltrd_best.pt \\
        --output-dir saltr/results/oof/ \\
        --merged-output saltr/results/preds_all_v2_oof_teacher.json \\
        --n-folds 5

    # Skip fold training (resume from existing checkpoints)
    python -m salt_r.make_oof_predictions ... --skip-train

    # Specific phases only
    python -m salt_r.make_oof_predictions ... --phases 1,2,3
    python -m salt_r.make_oof_predictions ... --phases 4,5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# Phase 1: fold assignment
# ---------------------------------------------------------------------------


def _dataset_bucket(seq_key: str) -> str:
    """Return the dataset name portion of a compound key ``dataset/seq_name``."""
    return seq_key.split("/", 1)[0]


def _make_fold_assignments(
    train_seqs: List[str],
    n_folds: int = 5,
) -> Dict[str, int]:
    """Assign each train sequence to a fold using deterministic round-robin.

    Stratification is by dataset (the prefix of the compound key before the
    first ``/``).  Within each dataset bucket sequences are sorted
    alphabetically so the assignment is reproducible without a random seed.

    Parameters
    ----------
    train_seqs:
        List of compound sequence keys, e.g. ``["uav123/bike1", ...]``.
    n_folds:
        Number of folds.

    Returns
    -------
    Dict mapping each sequence key to an integer fold index in ``[0, n_folds)``.
    """
    # Group by dataset bucket, sort within each bucket for reproducibility.
    buckets: Dict[str, List[str]] = {}
    for seq in train_seqs:
        bucket = _dataset_bucket(seq)
        buckets.setdefault(bucket, []).append(seq)

    assignments: Dict[str, int] = {}
    for bucket in sorted(buckets.keys()):
        seqs_sorted = sorted(buckets[bucket])
        for i, seq in enumerate(seqs_sorted):
            assignments[seq] = i % n_folds

    return assignments


# ---------------------------------------------------------------------------
# Phase 2: fold NPZ writing
# ---------------------------------------------------------------------------


def _write_fold_npz(
    source_npz_path: str,
    fold_npz_path: str,
    fold_k_train_seqs: Set[str],
    fold_k_held_out_seqs: Set[str],
) -> None:
    """Write a fold NPZ with modified split labels.

    Split re-assignment rules
    -------------------------
    - Original train sequences NOT in fold k  → ``"train"``
    - Original val sequences                   → ``"val"``   (unchanged)
    - Fold-k train sequences (held-out)        → ``"diagnostic"``
    - Original diagnostic sequences            → omitted entirely

    All feature/label/iou_trace/bbox arrays are copied verbatim; only the
    ``split/{seq}`` scalars change.

    Parameters
    ----------
    source_npz_path:
        Path to the original NPZ (``salt_rd_v2_labels.npz``).
    fold_npz_path:
        Destination path for the fold NPZ.
    fold_k_train_seqs:
        Compound keys of train-split sequences that are NOT held out for this
        fold (they become the fold model's training set).
    fold_k_held_out_seqs:
        Compound keys of train-split sequences held out for this fold (they
        become ``"diagnostic"`` so ``eval.py --split diagnostic`` scores them).
    """
    Path(fold_npz_path).parent.mkdir(parents=True, exist_ok=True)

    data = np.load(source_npz_path, allow_pickle=True)

    # Identify all compound keys from the source NPZ.
    all_compound_keys = [
        k[len("features/"):] for k in data.files if k.startswith("features/")
    ]

    # Determine which compound keys to include and their new split value.
    new_split: Dict[str, str] = {}
    for key in all_compound_keys:
        orig_split = str(data[f"split/{key}"])
        if key in fold_k_train_seqs:
            # Main train set for fold model.
            new_split[key] = "train"
        elif key in fold_k_held_out_seqs:
            # Held-out fold → expose as "diagnostic" for eval.py.
            new_split[key] = "diagnostic"
        elif orig_split == "val":
            new_split[key] = "val"
        else:
            # Original diagnostic sequences — omit from fold NPZ.
            assert orig_split == "diagnostic", (
                f"Unexpected split value {orig_split!r} for key {key!r}"
            )
            # Do NOT include; continue to next key.
            continue

    # Assemble the output arrays dict.
    out: Dict[str, np.ndarray] = {}

    # Copy scalar metadata arrays verbatim.
    scalar_meta_keys = [
        "feature_names", "feature_units", "label_names",
        "tracker_version", "tracker_config_hash", "created_at",
    ]
    for meta_key in scalar_meta_keys:
        if meta_key in data.files:
            out[meta_key] = data[meta_key]

    # Copy per-sequence arrays for the included sequences.
    per_seq_prefixes = [
        "features", "labels", "iou_trace", "bbox_pred", "bbox_gt",
        "sequence_name", "dataset",
    ]
    for key, new_sp in new_split.items():
        for prefix in per_seq_prefixes:
            npz_key = f"{prefix}/{key}"
            if npz_key in data.files:
                out[npz_key] = data[npz_key]
        out[f"split/{key}"] = np.array(new_sp)

    np.savez_compressed(fold_npz_path, **out)
    print(
        f"[fold_npz] Wrote {fold_npz_path}  "
        f"(train={len(fold_k_train_seqs)}  held_out={len(fold_k_held_out_seqs)}  "
        f"val={sum(1 for sp in new_split.values() if sp == 'val')})"
    )


# ---------------------------------------------------------------------------
# Phase 2B/C: subprocess helpers (train + eval)
# ---------------------------------------------------------------------------


def _run_subprocess(cmd: List[str], description: str) -> None:
    """Run a subprocess command; raise RuntimeError on non-zero exit."""
    print(f"\n[run] {description}")
    print(f"[cmd] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {result.returncode}"
        )


def _train_fold(
    fold_npz_path: str,
    fold_ckpt_dir: str,
    epochs: int,
    patience: int,
    python_bin: str = ".venv/bin/python",
) -> None:
    """Launch fold model training via ``salt_r.train``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:saltr/src"
    cmd = [
        python_bin, "-m", "salt_r.train",
        "--npz", fold_npz_path,
        "--output", fold_ckpt_dir,
        "--label-schema", "v2",
        "--epochs", str(epochs),
        "--patience", str(patience),
    ]
    print(f"\n[train] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for fold NPZ {fold_npz_path}")


def _eval_fold(
    fold_npz_path: str,
    fold_ckpt_path: str,
    eval_output: str,
    preds_output: str,
    python_bin: str = ".venv/bin/python",
) -> None:
    """Launch fold model evaluation via ``salt_r.eval`` on split=diagnostic."""
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:saltr/src"
    cmd = [
        python_bin, "-m", "salt_r.eval",
        "--npz", fold_npz_path,
        "--checkpoint", fold_ckpt_path,
        "--split", "diagnostic",
        "--output", eval_output,
        "--predictions-output", preds_output,
    ]
    print(f"\n[eval] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Eval failed for fold checkpoint {fold_ckpt_path}")


def _eval_teacher(
    npz_path: str,
    teacher_ckpt: str,
    split: str,
    preds_output: str,
    eval_output: Optional[str] = None,
    python_bin: str = ".venv/bin/python",
) -> None:
    """Run eval.py with the canonical teacher checkpoint for val/diagnostic."""
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:saltr/src"
    cmd = [
        python_bin, "-m", "salt_r.eval",
        "--npz", npz_path,
        "--checkpoint", teacher_ckpt,
        "--split", split,
        "--predictions-output", preds_output,
    ]
    if eval_output:
        cmd += ["--output", eval_output]
    print(f"\n[eval_teacher] split={split}  {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Teacher eval failed for split={split}")


# ---------------------------------------------------------------------------
# Phase 3/5: merge helpers
# ---------------------------------------------------------------------------


def _load_json(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def _merge_fold_preds(
    output_dir: str,
    n_folds: int,
) -> Dict[str, list]:
    """Load and merge per-fold prediction files into a single dict."""
    merged: Dict[str, list] = {}
    for k in range(n_folds):
        fold_path = os.path.join(output_dir, f"preds_fold_{k:02d}.json")
        if not Path(fold_path).exists():
            raise FileNotFoundError(
                f"Missing fold predictions file: {fold_path}\n"
                "Run phases 1,2 first or pass --skip-train to use existing checkpoints."
            )
        fold_preds = _load_json(fold_path)
        for key, frames in fold_preds.items():
            if key == "_meta":
                continue
            if key in merged:
                raise ValueError(
                    f"Sequence {key!r} appears in multiple fold prediction files — "
                    "fold assignment is not disjoint."
                )
            merged[key] = frames
    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _get_n_frames(npz_path: str) -> Dict[str, int]:
    """Return a mapping from compound key to frame count from the NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        k[len("features/"):]: int(data[k].shape[0])
        for k in data.files
        if k.startswith("features/")
    }


def _get_split_seqs(npz_path: str, split: str) -> Set[str]:
    """Return compound keys for sequences with the given split label."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        k[len("split/"):] for k in data.files
        if k.startswith("split/") and str(data[k]) == split
    }


def _validate_merged(
    preds_all: Dict[str, object],
    npz_path: str,
    train_seqs: Optional[Set[str]] = None,
    val_seqs: Optional[Set[str]] = None,
    diagnostic_seqs: Optional[Set[str]] = None,
) -> None:
    """Validate frame counts and split membership of the merged predictions dict.

    Parameters
    ----------
    preds_all:
        The merged predictions dict (may contain ``"_meta"`` key).
    npz_path:
        Path to the source NPZ for frame-count ground truth.
    train_seqs / val_seqs / diagnostic_seqs:
        Optional sets of expected sequence keys per split.  When provided,
        the function checks that the merged dict covers each set exactly once
        and that no original-diagnostic sequences sneak in.
    """
    n_frames_in_npz = _get_n_frames(npz_path)
    errors: List[str] = []

    prediction_seqs = {k for k in preds_all.keys() if k != "_meta"}

    # Frame count check.
    for seq, frames in preds_all.items():
        if seq == "_meta":
            continue
        if seq not in n_frames_in_npz:
            errors.append(f"{seq}: not found in NPZ keys")
            continue
        expected_T = n_frames_in_npz[seq]
        if len(frames) != expected_T:
            errors.append(
                f"{seq}: {len(frames)} preds != {expected_T} frames in NPZ"
            )

    # Train-sequence coverage.
    if train_seqs is not None:
        missing = train_seqs - prediction_seqs
        extra_train = prediction_seqs - (
            train_seqs
            | (val_seqs or set())
            | (diagnostic_seqs or set())
        )
        if missing:
            errors.append(
                f"{len(missing)} train sequence(s) missing from merged preds: "
                f"{sorted(missing)[:5]}..."
            )
        if extra_train:
            errors.append(
                f"{len(extra_train)} unexpected key(s) in merged preds: "
                f"{sorted(extra_train)[:5]}..."
            )

    # Diagnostic sequences must NOT appear in the merged output.
    if diagnostic_seqs is not None:
        leaked = prediction_seqs & diagnostic_seqs
        if leaked:
            errors.append(
                f"{len(leaked)} original-diagnostic sequence(s) leaked into merged "
                f"preds (should be sourced from teacher eval, not OOF): "
                f"{sorted(leaked)}"
            )

    if errors:
        raise ValueError(
            "Merged predictions validation failed:\n" + "\n".join(errors)
        )

    print(
        f"[validate] {len(prediction_seqs)} sequences — all frame counts match "
        f"and split membership is clean."
    )


# ---------------------------------------------------------------------------
# Phase 5: assemble metadata header
# ---------------------------------------------------------------------------


def _build_meta(
    train_seqs: List[str],
    val_seqs: List[str],
    diagnostic_seqs: List[str],
    n_folds: int,
) -> dict:
    return {
        "train_source": f"oof_{n_folds}fold",
        "val_source": "v2_corrected_teacher",
        "diagnostic_source": "v2_corrected_teacher",
        "n_folds": n_folds,
        "n_train_seqs": len(train_seqs),
        "n_val_seqs": len(val_seqs),
        "n_diagnostic_seqs": len(diagnostic_seqs),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "uses_oracle_labels": False,
        "n_oracle_fallback_sequences": 0,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_oof_pipeline(
    npz_path: str,
    teacher_checkpoint: str,
    output_dir: str,
    merged_output: str,
    n_folds: int = 5,
    fold_npz_dir: str = "saltr/tmp/oof",
    fold_ckpt_base: str = "saltr/checkpoints/oof_v2",
    epochs: int = 50,
    patience: int = 8,
    skip_train: bool = False,
    phases: Optional[List[int]] = None,
    python_bin: str = ".venv/bin/python",
) -> None:
    """Run the full OOF prediction pipeline.

    Parameters
    ----------
    npz_path:
        Path to the canonical NPZ (``salt_rd_v2_labels.npz``).
    teacher_checkpoint:
        Path to the canonical no-memory teacher checkpoint.
    output_dir:
        Directory for OOF eval JSONs and per-fold prediction JSONs.
    merged_output:
        Destination for the final merged predictions JSON.
    n_folds:
        Number of OOF folds (default 5).
    fold_npz_dir:
        Directory for temporary fold NPZ files.
    fold_ckpt_base:
        Base directory for fold model checkpoints.
    epochs:
        Max epochs for fold model training.
    patience:
        Early-stopping patience for fold model training.
    skip_train:
        If True, skip training and use existing fold checkpoints.
    phases:
        If not None, only run the listed phase numbers (1–5).
    python_bin:
        Path to the Python interpreter (must have ``salt_r`` on sys.path via
        the PYTHONPATH env variable set in subprocess helpers).
    """
    if phases is None:
        phases = [1, 2, 3, 4, 5]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load NPZ metadata (needed by several phases).
    # ------------------------------------------------------------------
    print(f"\n[pipeline] Loading NPZ metadata from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    all_compound_keys = [
        k[len("features/"):] for k in data.files if k.startswith("features/")
    ]

    train_seqs: List[str] = []
    val_seqs: List[str] = []
    diagnostic_seqs: List[str] = []
    for key in all_compound_keys:
        sp = str(data[f"split/{key}"])
        if sp == "train":
            train_seqs.append(key)
        elif sp == "val":
            val_seqs.append(key)
        elif sp == "diagnostic":
            diagnostic_seqs.append(key)

    print(
        f"[pipeline] Sequences — train={len(train_seqs)}  "
        f"val={len(val_seqs)}  diagnostic={len(diagnostic_seqs)}"
    )

    # ------------------------------------------------------------------
    # Phase 1: Fold assignment
    # ------------------------------------------------------------------
    if 1 in phases:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Fold assignment ({n_folds} folds)")
        print("=" * 60)

    fold_assignments = _make_fold_assignments(train_seqs, n_folds=n_folds)

    if 1 in phases:
        for fold_k in range(n_folds):
            fold_seqs = [s for s, f in fold_assignments.items() if f == fold_k]
            # Dataset distribution for this fold.
            by_ds: Dict[str, int] = {}
            for s in fold_seqs:
                ds = _dataset_bucket(s)
                by_ds[ds] = by_ds.get(ds, 0) + 1
            print(f"  Fold {fold_k}: {len(fold_seqs)} sequences  {by_ds}")

    # ------------------------------------------------------------------
    # Phase 2: Train/eval each fold
    # ------------------------------------------------------------------
    if 2 in phases:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Per-fold training + evaluation")
        print("=" * 60)

        for fold_k in range(n_folds):
            print(f"\n--- Fold {fold_k} ---")

            held_out: Set[str] = {
                s for s, f in fold_assignments.items() if f == fold_k
            }
            fold_train: Set[str] = {
                s for s, f in fold_assignments.items() if f != fold_k
            }

            # Step A: Write fold NPZ.
            fold_npz = os.path.join(fold_npz_dir, f"fold_{fold_k:02d}.npz")
            _write_fold_npz(
                source_npz_path=npz_path,
                fold_npz_path=fold_npz,
                fold_k_train_seqs=fold_train,
                fold_k_held_out_seqs=held_out,
            )

            fold_ckpt_dir = os.path.join(fold_ckpt_base, f"fold_{fold_k:02d}")
            fold_ckpt_path = os.path.join(fold_ckpt_dir, "saltrd_best.pt")

            # Step B: Train fold model.
            if skip_train:
                if not Path(fold_ckpt_path).exists():
                    raise FileNotFoundError(
                        f"--skip-train was set but checkpoint not found: {fold_ckpt_path}"
                    )
                print(f"[skip-train] Using existing checkpoint: {fold_ckpt_path}")
            else:
                _train_fold(
                    fold_npz_path=fold_npz,
                    fold_ckpt_dir=fold_ckpt_dir,
                    epochs=epochs,
                    patience=patience,
                    python_bin=python_bin,
                )

            # Step C: Eval fold model on held-out (split=diagnostic).
            eval_output = os.path.join(output_dir, f"eval_fold_{fold_k:02d}.json")
            preds_output = os.path.join(output_dir, f"preds_fold_{fold_k:02d}.json")
            _eval_fold(
                fold_npz_path=fold_npz,
                fold_ckpt_path=fold_ckpt_path,
                eval_output=eval_output,
                preds_output=preds_output,
                python_bin=python_bin,
            )

    # ------------------------------------------------------------------
    # Phase 3: Merge OOF train predictions
    # ------------------------------------------------------------------
    if 3 in phases:
        print(f"\n{'='*60}")
        print("PHASE 3: Merge OOF fold predictions")
        print("=" * 60)

        preds_train_oof = _merge_fold_preds(output_dir, n_folds)

        # Validation checks.
        train_seqs_set = set(train_seqs)
        assert set(preds_train_oof.keys()) == train_seqs_set, (
            f"OOF preds do not exactly cover train sequences.\n"
            f"Missing: {train_seqs_set - set(preds_train_oof.keys())}\n"
            f"Extra: {set(preds_train_oof.keys()) - train_seqs_set}"
        )
        val_seqs_set = set(val_seqs)
        diag_seqs_set = set(diagnostic_seqs)
        assert not (set(preds_train_oof.keys()) & (val_seqs_set | diag_seqs_set)), (
            "Val or diagnostic sequences found in OOF train predictions — "
            "fold assignment error."
        )

        # Frame-count validation.
        n_frames_in_npz = _get_n_frames(npz_path)
        frame_errors = []
        for seq, frames in preds_train_oof.items():
            expected = n_frames_in_npz[seq]
            if len(frames) != expected:
                frame_errors.append(
                    f"{seq}: {len(frames)} preds != {expected} frames"
                )
        if frame_errors:
            raise ValueError(
                "Frame count mismatches in OOF predictions:\n"
                + "\n".join(frame_errors)
            )

        train_oof_path = os.path.join(
            os.path.dirname(merged_output), "preds_train_oof_v2.json"
        )
        Path(train_oof_path).parent.mkdir(parents=True, exist_ok=True)
        with open(train_oof_path, "w") as fh:
            json.dump(preds_train_oof, fh)
        print(
            f"[phase3] OOF train predictions merged to {train_oof_path}  "
            f"({len(preds_train_oof)} sequences)"
        )

    # ------------------------------------------------------------------
    # Phase 4: Teacher predictions for val + diagnostic
    # ------------------------------------------------------------------
    if 4 in phases:
        print(f"\n{'='*60}")
        print("PHASE 4: Teacher predictions for val + diagnostic splits")
        print("=" * 60)

        out_base = os.path.dirname(merged_output)
        Path(out_base).mkdir(parents=True, exist_ok=True)

        _eval_teacher(
            npz_path=npz_path,
            teacher_ckpt=teacher_checkpoint,
            split="val",
            preds_output=os.path.join(out_base, "preds_val_v2_teacher.json"),
            eval_output=None,
            python_bin=python_bin,
        )
        _eval_teacher(
            npz_path=npz_path,
            teacher_ckpt=teacher_checkpoint,
            split="diagnostic",
            preds_output=os.path.join(out_base, "preds_diagnostic_v2_teacher.json"),
            eval_output=None,
            python_bin=python_bin,
        )

    # ------------------------------------------------------------------
    # Phase 5: Merge everything
    # ------------------------------------------------------------------
    if 5 in phases:
        print(f"\n{'='*60}")
        print("PHASE 5: Final merge + validation")
        print("=" * 60)

        out_base = os.path.dirname(merged_output)

        train_oof_path = os.path.join(out_base, "preds_train_oof_v2.json")
        val_teacher_path = os.path.join(out_base, "preds_val_v2_teacher.json")
        diag_teacher_path = os.path.join(out_base, "preds_diagnostic_v2_teacher.json")

        for p in [train_oof_path, val_teacher_path, diag_teacher_path]:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Required predictions file not found: {p}\n"
                    "Run phases 3 and 4 first."
                )

        preds_train_oof = _load_json(train_oof_path)
        preds_val_teacher = _load_json(val_teacher_path)
        preds_diag_teacher = _load_json(diag_teacher_path)

        # Check for key collisions between sources.
        train_keys = set(preds_train_oof.keys()) - {"_meta"}
        val_keys = set(preds_val_teacher.keys()) - {"_meta"}
        diag_keys = set(preds_diag_teacher.keys()) - {"_meta"}
        overlap = (train_keys & val_keys) | (train_keys & diag_keys) | (val_keys & diag_keys)
        if overlap:
            raise ValueError(
                f"Overlapping keys across prediction sources: {sorted(overlap)[:10]}"
            )

        preds_all: dict = {}
        preds_all.update({k: v for k, v in preds_train_oof.items() if k != "_meta"})
        preds_all.update({k: v for k, v in preds_val_teacher.items() if k != "_meta"})
        preds_all.update({k: v for k, v in preds_diag_teacher.items() if k != "_meta"})

        # Validation.
        _validate_merged(
            preds_all=preds_all,
            npz_path=npz_path,
            train_seqs=set(train_seqs),
            val_seqs=set(val_seqs),
            diagnostic_seqs=set(diagnostic_seqs),
        )

        total_expected = len(train_seqs) + len(val_seqs) + len(diagnostic_seqs)
        if len(preds_all) != total_expected:
            raise ValueError(
                f"Expected {total_expected} sequences total, got {len(preds_all)}"
            )

        # Prepend metadata key.
        meta = _build_meta(train_seqs, val_seqs, diagnostic_seqs, n_folds)
        final_dict: dict = {"_meta": meta}
        final_dict.update(preds_all)

        Path(merged_output).parent.mkdir(parents=True, exist_ok=True)
        with open(merged_output, "w") as fh:
            json.dump(final_dict, fh)

        print(
            f"\n[phase5] Merged predictions written to: {merged_output}\n"
            f"  train (OOF): {len(train_seqs)} sequences\n"
            f"  val (teacher): {len(val_seqs)} sequences\n"
            f"  diagnostic (teacher): {len(diagnostic_seqs)} sequences\n"
            f"  total: {len(preds_all)} sequences"
        )
        print(f"\n[pipeline] DONE — {merged_output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_phases(s: str) -> List[int]:
    """Parse a comma-separated list of phase integers, e.g. ``"1,2,3"``."""
    return [int(p.strip()) for p in s.split(",")]


def main() -> None:
    """CLI entry point for the OOF prediction pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Produce out-of-fold predictions for the SALT-RD train split. "
            "The merged JSON can replace preds_all_v2_retrained.json as the "
            "prediction source for SGLA memory sidecar extraction."
        )
    )
    parser.add_argument(
        "--npz",
        default="saltr/data/salt_rd_v2_labels.npz",
        help="Path to the canonical NPZ (default: saltr/data/salt_rd_v2_labels.npz).",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="saltr/checkpoints/v2_corrected/saltrd_best.pt",
        help=(
            "Path to the canonical no-memory teacher checkpoint "
            "(default: saltr/checkpoints/v2_corrected/saltrd_best.pt)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="saltr/results/oof/",
        help="Directory for OOF eval and prediction JSONs (default: saltr/results/oof/).",
    )
    parser.add_argument(
        "--merged-output",
        default="saltr/results/preds_all_v2_oof_teacher.json",
        help=(
            "Destination for the final merged predictions JSON "
            "(default: saltr/results/preds_all_v2_oof_teacher.json)."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of OOF folds.")
    parser.add_argument(
        "--fold-npz-dir",
        default="saltr/tmp/oof",
        help="Directory for temporary fold NPZ files (default: saltr/tmp/oof).",
    )
    parser.add_argument(
        "--fold-ckpt-base",
        default="saltr/checkpoints/oof_v2",
        help="Base dir for fold checkpoints (default: saltr/checkpoints/oof_v2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max training epochs per fold (default: 50).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early-stopping patience per fold (default: 8).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; use existing fold checkpoints.",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help=(
            "Comma-separated list of phases to run (1-5). "
            "Default: all phases. Example: --phases 1,2,3 or --phases 4,5."
        ),
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=".venv/bin/python",
        help="Python interpreter path for subprocess calls (default: .venv/bin/python).",
    )

    args = parser.parse_args()

    phases = _parse_phases(args.phases) if args.phases else None

    run_oof_pipeline(
        npz_path=args.npz,
        teacher_checkpoint=args.teacher_checkpoint,
        output_dir=args.output_dir,
        merged_output=args.merged_output,
        n_folds=args.n_folds,
        fold_npz_dir=args.fold_npz_dir,
        fold_ckpt_base=args.fold_ckpt_base,
        epochs=args.epochs,
        patience=args.patience,
        skip_train=args.skip_train,
        phases=phases,
        python_bin=args.python_bin,
    )


if __name__ == "__main__":
    main()
