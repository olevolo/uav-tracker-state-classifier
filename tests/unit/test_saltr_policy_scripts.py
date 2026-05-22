"""test_saltr_policy_scripts.py — Import-level and structural tests for policy scripts.

Tests:
1. train_policy module is importable
2. calibrate_policy module is importable
3. rollout_policy module is importable
4. No TSA imports in any of the three modules
5. All three modules have a main() function
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure salt_r is on the path before any imports
# ---------------------------------------------------------------------------

_SALT_R_SRC = str(Path(__file__).parents[2] / "saltr" / "src")
if _SALT_R_SRC not in sys.path:
    sys.path.insert(0, _SALT_R_SRC)

# Module names to test
_POLICY_MODULE_NAMES = [
    "salt_r.train_policy",
    "salt_r.calibrate_policy",
    "salt_r.rollout_policy",
]


# ---------------------------------------------------------------------------
# Test 1: train_policy is importable
# ---------------------------------------------------------------------------

def test_train_policy_importable():
    """train_policy module must be importable without side-effects."""
    mod = importlib.import_module("salt_r.train_policy")
    assert mod is not None, "salt_r.train_policy failed to import"


# ---------------------------------------------------------------------------
# Test 2: calibrate_policy is importable
# ---------------------------------------------------------------------------

def test_calibrate_policy_importable():
    """calibrate_policy module must be importable without side-effects."""
    mod = importlib.import_module("salt_r.calibrate_policy")
    assert mod is not None, "salt_r.calibrate_policy failed to import"


# ---------------------------------------------------------------------------
# Test 3: rollout_policy is importable
# ---------------------------------------------------------------------------

def test_rollout_policy_importable():
    """rollout_policy module must be importable without side-effects."""
    mod = importlib.import_module("salt_r.rollout_policy")
    assert mod is not None, "salt_r.rollout_policy failed to import"


# ---------------------------------------------------------------------------
# Test 4: No TSA imports in any of the three modules
# ---------------------------------------------------------------------------

_TSA_PATTERNS = ["tsa", "TSA", "tracker_state_annotation", "TrackerStateAnnotation"]


def _source_for_module(module_name: str) -> str:
    """Return the source code of a module."""
    mod = importlib.import_module(module_name)
    src_file = inspect.getfile(mod)
    return Path(src_file).read_text(encoding="utf-8")


@pytest.mark.parametrize("module_name", _POLICY_MODULE_NAMES)
def test_no_tsa_imports(module_name: str):
    """None of the three policy scripts must import from TSA modules."""
    source = _source_for_module(module_name)
    for pattern in _TSA_PATTERNS:
        # Check import lines only: "import tsa", "from tsa import ...", etc.
        import_lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert pattern not in line, (
                f"{module_name}: found TSA import pattern '{pattern}' in line: {line!r}"
            )


# ---------------------------------------------------------------------------
# Test 5: All three modules have a main() function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", _POLICY_MODULE_NAMES)
def test_module_has_main(module_name: str):
    """Each policy script must expose a top-level main() callable."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "main"), f"{module_name} has no 'main' attribute"
    assert callable(mod.main), f"{module_name}.main is not callable"


# ---------------------------------------------------------------------------
# Test 6: CandidateEventDataset uses candidate_correct_iou03 label (V5 schema)
# ---------------------------------------------------------------------------

def test_candidate_event_dataset_v5_label(tmp_path):
    """CandidateEventDataset must use candidate_correct_iou03, not label_good_candidate.

    Synthetic event: label_good_candidate=0 (legacy dead label) but
    candidate_correct_iou03=1 (IoU >= 0.30). Dataset must return label 1.
    """
    import numpy as np
    import torch
    from salt_r.train_policy import CandidateEventDataset

    ev = {
        "frame_idx": 10,
        "seq_id": "car7",
        "timestamp_s": 0.0,
        "candidate_bbox": [100.0, 100.0, 40.0, 40.0],
        "tracker_bbox": [90.0, 90.0, 40.0, 40.0],
        "source": "detector",
        "detector_score": 0.8,
        "score_map_score": None,
        "geometry_area_ratio": 1.0,
        "frame_area_ratio": 0.01,
        "cosine_sim": 0.7,
        "accepted": True,
        "reject_reason": None,
        "frame_h": 480,
        "frame_w": 640,
        "dist_from_last": 0.05,
        "candidate_iou": 0.45,          # labeling present
        "future_iou_gain": 0.0,         # legacy gate would yield label=0
        "label_good_candidate": 0,      # legacy: always 0 in V4 due to broken gate
        "candidate_correct_iou03": 1,   # V5 label: IoU >= 0.30 → correct
        "candidate_correct_iou05": 0,   # IoU < 0.50
    }

    npz_path = tmp_path / "test_events.npz"
    np.savez_compressed(
        npz_path,
        events=np.array([ev], dtype=object),
        stats=np.array([{}], dtype=object),
    )

    ds = CandidateEventDataset(str(npz_path), window_size=4)
    assert len(ds) == 1, "expected exactly one sample"
    _, _, _, cand_label = ds[0]
    assert cand_label.item() == 1.0, (
        f"Expected label 1 (candidate_correct_iou03=1) but got {cand_label.item()}. "
        "CandidateEventDataset is still reading label_good_candidate."
    )


def test_candidate_event_dataset_fails_on_legacy_schema(tmp_path):
    """CandidateEventDataset must raise ValueError on V4/legacy NPZ without candidate_correct_iou03."""
    import numpy as np
    import pytest
    from salt_r.train_policy import CandidateEventDataset

    legacy_ev = {
        "frame_idx": 5,
        "seq_id": "",
        "candidate_bbox": [50.0, 50.0, 20.0, 20.0],
        "tracker_bbox": [50.0, 50.0, 20.0, 20.0],
        "source": "detector",
        "detector_score": 0.6,
        "score_map_score": None,
        "geometry_area_ratio": 1.0,
        "frame_area_ratio": 0.005,
        "cosine_sim": 0.5,
        "accepted": True,
        "reject_reason": None,
        "frame_h": 0,
        "frame_w": 0,
        "dist_from_last": 0.0,
        "candidate_iou": 0.0,
        "future_iou_gain": 0.0,
        "label_good_candidate": 0,
        # candidate_correct_iou03 intentionally absent (simulates V4 artifact)
    }

    npz_path = tmp_path / "legacy_events.npz"
    np.savez_compressed(
        npz_path,
        events=np.array([legacy_ev], dtype=object),
        stats=np.array([{}], dtype=object),
    )

    with pytest.raises(ValueError, match="candidate_correct_iou03"):
        CandidateEventDataset(str(npz_path), window_size=4)
