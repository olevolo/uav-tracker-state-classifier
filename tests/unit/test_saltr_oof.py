"""Unit tests for salt_r.make_oof_predictions.

Tests cover:
1. _make_fold_assignments: all train sequences covered exactly once across 5 folds.
2. _make_fold_assignments: approximately balanced dataset distribution per fold.
3. _validate_merged: raises ValueError on frame-count mismatch.
4. _validate_merged: raises ValueError if a diagnostic sequence appears in OOF train preds.
5. _write_fold_npz: correct split labels in written NPZ.
6. _merge_fold_preds: raises FileNotFoundError on missing fold file.
7. validate_final_merged: passes when train ∪ val ∪ diagnostic are all present.
8. validate_final_merged: raises if a split is missing from the final merged dict.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from salt_r.make_oof_predictions import (
    _make_fold_assignments,
    _validate_merged,
    validate_final_merged,
    _write_fold_npz,
    _merge_fold_preds,
    _dataset_bucket,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_train_seqs(
    n_uav123: int = 15,
    n_visdrone: int = 10,
    n_dtb70: int = 10,
) -> List[str]:
    """Generate synthetic compound train sequence keys for testing."""
    seqs: List[str] = []
    for i in range(n_uav123):
        seqs.append(f"uav123/seq_{i:03d}")
    for i in range(n_visdrone):
        seqs.append(f"visdrone_sot/seq_{i:03d}")
    for i in range(n_dtb70):
        seqs.append(f"dtb70/seq_{i:03d}")
    return seqs


def _make_synthetic_npz(
    train_seqs: List[str],
    val_seqs: List[str],
    diagnostic_seqs: List[str],
    n_frames_per_seq: int = 30,
) -> str:
    """Write a minimal synthetic NPZ to a temp file and return the path."""
    arrays: Dict[str, np.ndarray] = {
        "feature_names": np.array(["f0", "f1"], dtype=object),
        "feature_units": np.array(["u0", "u1"], dtype=object),
        "label_names": np.array(["label0"], dtype=object),
        "tracker_version": np.array("test_v0"),
        "tracker_config_hash": np.array("deadbeef"),
        "created_at": np.array("2026-01-01T00:00:00"),
    }

    for seq_key, split in (
        [(s, "train") for s in train_seqs]
        + [(s, "val") for s in val_seqs]
        + [(s, "diagnostic") for s in diagnostic_seqs]
    ):
        T = n_frames_per_seq
        arrays[f"features/{seq_key}"] = np.zeros((T, 2), dtype=np.float32)
        arrays[f"labels/{seq_key}"] = np.zeros((T, 1), dtype=np.int8)
        arrays[f"iou_trace/{seq_key}"] = np.zeros(T, dtype=np.float32)
        arrays[f"bbox_pred/{seq_key}"] = np.zeros((T, 4), dtype=np.float32)
        arrays[f"bbox_gt/{seq_key}"] = np.zeros((T, 4), dtype=np.float32)
        arrays[f"sequence_name/{seq_key}"] = np.array(seq_key.split("/", 1)[1])
        arrays[f"dataset/{seq_key}"] = np.array(seq_key.split("/", 1)[0])
        arrays[f"split/{seq_key}"] = np.array(split)

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    np.savez_compressed(tmp.name, **arrays)
    return tmp.name


# ---------------------------------------------------------------------------
# Test 1: _make_fold_assignments — all sequences covered exactly once
# ---------------------------------------------------------------------------


def test_fold_assignments_cover_all_sequences():
    """Every train sequence appears in exactly one fold."""
    train_seqs = _make_synthetic_train_seqs(n_uav123=15, n_visdrone=10, n_dtb70=10)
    n_folds = 5
    assignments = _make_fold_assignments(train_seqs, n_folds=n_folds)

    # All train sequences assigned.
    assert set(assignments.keys()) == set(train_seqs), (
        "Some sequences were not assigned to a fold."
    )

    # Each sequence appears in exactly one fold.
    from collections import Counter
    fold_counts = Counter(assignments.values())
    for fold_k in range(n_folds):
        assert fold_k in fold_counts, f"Fold {fold_k} has no sequences."

    # Total count matches.
    assert len(assignments) == len(train_seqs)


def test_fold_assignments_deterministic():
    """Calling _make_fold_assignments twice gives the same result."""
    train_seqs = _make_synthetic_train_seqs(n_uav123=20, n_visdrone=8, n_dtb70=12)
    a1 = _make_fold_assignments(train_seqs, n_folds=5)
    a2 = _make_fold_assignments(train_seqs, n_folds=5)
    assert a1 == a2, "Fold assignments are not deterministic."


# ---------------------------------------------------------------------------
# Test 2: Approximately balanced dataset-stratified distribution
# ---------------------------------------------------------------------------


def test_fold_assignments_stratified_balance():
    """Each dataset should contribute at most ceil(n_seqs/n_folds)+1 sequences per fold.

    Because we use round-robin within each bucket, the maximum imbalance across
    folds for any single dataset is at most 1 sequence.
    """
    n_uav = 17
    n_vis = 8
    n_dtb = 11
    n_folds = 5
    train_seqs = _make_synthetic_train_seqs(n_uav123=n_uav, n_visdrone=n_vis, n_dtb70=n_dtb)
    assignments = _make_fold_assignments(train_seqs, n_folds=n_folds)

    datasets = {
        "uav123": n_uav,
        "visdrone_sot": n_vis,
        "dtb70": n_dtb,
    }

    for dataset, n_total in datasets.items():
        seqs_in_ds = [s for s in train_seqs if _dataset_bucket(s) == dataset]
        fold_sizes = [
            sum(1 for s in seqs_in_ds if assignments[s] == k)
            for k in range(n_folds)
        ]
        min_expected = n_total // n_folds
        max_expected = (n_total + n_folds - 1) // n_folds  # ceil
        for fold_k, sz in enumerate(fold_sizes):
            # Allow at most ceil deviation; round-robin guarantees at most 1 off.
            assert sz in (min_expected, max_expected), (
                f"Dataset {dataset!r}, fold {fold_k}: {sz} sequences, "
                f"expected {min_expected} or {max_expected} (±1 from round-robin)."
            )


# ---------------------------------------------------------------------------
# Test 3: _validate_merged raises on frame-count mismatch
# ---------------------------------------------------------------------------


def test_validate_merged_raises_on_frame_count_mismatch(tmp_path):
    """_validate_merged should raise ValueError when a sequence has wrong frame count."""
    train_seqs = ["uav123/seq_000", "uav123/seq_001"]
    val_seqs = ["dtb70/seq_val_0"]
    diagnostic_seqs = ["dtb70/Gull2"]

    npz_path = _make_synthetic_npz(
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        diagnostic_seqs=diagnostic_seqs,
        n_frames_per_seq=30,
    )

    # Create preds_all with a wrong frame count for seq_001.
    preds_all = {
        "uav123/seq_000": [{"false_confirmed": 0.1}] * 30,  # correct: 30 frames
        "uav123/seq_001": [{"false_confirmed": 0.2}] * 20,  # WRONG: should be 30
        "dtb70/seq_val_0": [{"false_confirmed": 0.3}] * 30,
        "dtb70/Gull2": [{"false_confirmed": 0.4}] * 30,
    }

    with pytest.raises(ValueError, match="frame"):
        _validate_merged(preds_all, npz_path)


# ---------------------------------------------------------------------------
# Test 4: _validate_merged raises if diagnostic sequence appears in OOF train preds
# ---------------------------------------------------------------------------


def test_validate_merged_raises_on_diagnostic_in_oof(tmp_path):
    """_validate_merged (OOF-only validator) should raise if a diagnostic
    sequence is present in preds AND is not in the expected train+val sets.
    This simulates a diagnostic seq accidentally included in OOF output."""
    train_seqs_list = ["uav123/seq_000", "uav123/seq_001"]
    val_seqs_list = ["dtb70/seq_val_0"]
    diag_seqs_list = ["dtb70/Gull2"]

    npz_path = _make_synthetic_npz(
        train_seqs=train_seqs_list,
        val_seqs=val_seqs_list,
        diagnostic_seqs=diag_seqs_list,
        n_frames_per_seq=30,
    )

    # Simulate OOF preds that accidentally include a diagnostic sequence.
    preds_oof_with_leak = {
        "uav123/seq_000": [{"false_confirmed": 0.1}] * 30,
        "uav123/seq_001": [{"false_confirmed": 0.2}] * 30,
        "dtb70/Gull2": [{"false_confirmed": 0.4}] * 30,  # WRONG: diagnostic seq in OOF
    }

    # _validate_merged with train_seqs+val_seqs but NOT diagnostic as expected:
    # Gull2 appears in preds but is not in train_seqs ∪ val_seqs → flagged as "extra".
    with pytest.raises(ValueError, match=r"(extra|unexpected|missing)"):
        _validate_merged(
            preds_all=preds_oof_with_leak,
            npz_path=npz_path,
            train_seqs=set(train_seqs_list),
            val_seqs=set(val_seqs_list),
            diagnostic_seqs=None,  # OOF validator: diagnostic seqs not expected
        )


# ---------------------------------------------------------------------------
# Tests 7-8: validate_final_merged
# ---------------------------------------------------------------------------


def test_validate_final_merged_passes_with_all_splits(tmp_path):
    """validate_final_merged should NOT raise when train ∪ val ∪ diagnostic
    are all present — this is the expected shape of preds_all_v2_oof_teacher.json."""
    train_seqs_list = ["uav123/seq_000", "uav123/seq_001"]
    val_seqs_list = ["dtb70/seq_val_0"]
    diag_seqs_list = ["dtb70/Gull2"]

    npz_path = _make_synthetic_npz(
        train_seqs=train_seqs_list,
        val_seqs=val_seqs_list,
        diagnostic_seqs=diag_seqs_list,
        n_frames_per_seq=30,
    )

    preds_all = {
        "uav123/seq_000": [{"false_confirmed": 0.1}] * 30,
        "uav123/seq_001": [{"false_confirmed": 0.2}] * 30,
        "dtb70/seq_val_0": [{"false_confirmed": 0.3}] * 30,
        "dtb70/Gull2": [{"false_confirmed": 0.4}] * 30,  # diagnostic — expected here
    }

    # Must not raise
    validate_final_merged(
        preds_all=preds_all,
        npz_path=npz_path,
        train_seqs=set(train_seqs_list),
        val_seqs=set(val_seqs_list),
        diagnostic_seqs=set(diag_seqs_list),
    )


def test_validate_final_merged_raises_on_missing_split(tmp_path):
    """validate_final_merged should raise if any split is absent from preds."""
    train_seqs_list = ["uav123/seq_000", "uav123/seq_001"]
    val_seqs_list = ["dtb70/seq_val_0"]
    diag_seqs_list = ["dtb70/Gull2"]

    npz_path = _make_synthetic_npz(
        train_seqs=train_seqs_list,
        val_seqs=val_seqs_list,
        diagnostic_seqs=diag_seqs_list,
        n_frames_per_seq=30,
    )

    # Missing diagnostic sequences entirely
    preds_missing_diag = {
        "uav123/seq_000": [{"false_confirmed": 0.1}] * 30,
        "uav123/seq_001": [{"false_confirmed": 0.2}] * 30,
        "dtb70/seq_val_0": [{"false_confirmed": 0.3}] * 30,
        # dtb70/Gull2 missing
    }

    with pytest.raises(ValueError, match=r"missing"):
        validate_final_merged(
            preds_all=preds_missing_diag,
            npz_path=npz_path,
            train_seqs=set(train_seqs_list),
            val_seqs=set(val_seqs_list),
            diagnostic_seqs=set(diag_seqs_list),
        )


# ---------------------------------------------------------------------------
# Test 5: _write_fold_npz — correct split labels
# ---------------------------------------------------------------------------


def test_write_fold_npz_split_labels(tmp_path):
    """The fold NPZ must assign 'train', 'val', 'diagnostic' correctly and
    must omit original-diagnostic sequences entirely."""
    train_seqs = ["uav123/seq_000", "uav123/seq_001", "uav123/seq_002"]
    val_seqs = ["dtb70/val_seq"]
    diag_seqs = ["dtb70/Gull2"]
    n_frames = 10

    npz_path = _make_synthetic_npz(
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        diagnostic_seqs=diag_seqs,
        n_frames_per_seq=n_frames,
    )

    fold_npz_path = str(tmp_path / "fold_00.npz")

    # Held out: seq_001 only; seq_000 and seq_002 are fold model's train set.
    held_out = {"uav123/seq_001"}
    fold_train = {"uav123/seq_000", "uav123/seq_002"}

    _write_fold_npz(
        source_npz_path=npz_path,
        fold_npz_path=fold_npz_path,
        fold_k_train_seqs=fold_train,
        fold_k_held_out_seqs=held_out,
    )

    result = np.load(fold_npz_path, allow_pickle=True)

    # Check split labels.
    assert str(result["split/uav123/seq_000"]) == "train", (
        "Non-held-out train seq should have split='train'."
    )
    assert str(result["split/uav123/seq_001"]) == "diagnostic", (
        "Held-out seq should have split='diagnostic'."
    )
    assert str(result["split/uav123/seq_002"]) == "train", (
        "Non-held-out train seq should have split='train'."
    )
    assert str(result["split/dtb70/val_seq"]) == "val", (
        "Val seq should keep split='val'."
    )

    # Original diagnostic sequence must be absent entirely.
    assert "split/dtb70/Gull2" not in result.files, (
        "Original diagnostic sequence should be omitted from fold NPZ."
    )
    assert "features/dtb70/Gull2" not in result.files, (
        "Original diagnostic sequence features should be omitted."
    )


# ---------------------------------------------------------------------------
# Test 6: _merge_fold_preds raises on missing file
# ---------------------------------------------------------------------------


def test_merge_fold_preds_raises_on_missing_file(tmp_path):
    """_merge_fold_preds should raise FileNotFoundError if a fold file is absent."""
    # Write only fold 0 and fold 1; fold 2 is missing.
    for k in [0, 1]:
        path = tmp_path / f"preds_fold_{k:02d}.json"
        path.write_text(json.dumps({f"uav123/seq_{k:03d}": [{"h": 0.5}]}))

    with pytest.raises(FileNotFoundError, match="preds_fold_02.json"):
        _merge_fold_preds(str(tmp_path), n_folds=3)


# ---------------------------------------------------------------------------
# Test 7: _merge_fold_preds raises on duplicate sequence across folds
# ---------------------------------------------------------------------------


def test_merge_fold_preds_raises_on_duplicate_sequence(tmp_path):
    """_merge_fold_preds should raise ValueError if a sequence appears in two folds."""
    for k in range(3):
        # All three folds contain the same sequence → duplicate.
        path = tmp_path / f"preds_fold_{k:02d}.json"
        path.write_text(json.dumps({"uav123/seq_000": [{"h": 0.5}]}))

    with pytest.raises(ValueError, match="uav123/seq_000"):
        _merge_fold_preds(str(tmp_path), n_folds=3)
