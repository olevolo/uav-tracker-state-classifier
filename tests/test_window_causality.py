"""Window causality test for CSCDataset.

Distinct from tests/test_csctcn_causality.py (which tests the TCN model's
internal causal convolutions).  This test verifies that the **data pipeline**
— specifically the CSCDataset window-slicing logic — never includes frames
t+1, t+2, … in the window it returns for position t.

Causality guarantee being tested:
    window for anchor t  ←  uses rows[t-W+1 : t+1] only
    rows[t+1:]           are NOT part of this window

If the window accidentally included future frames (e.g., centred window
instead of past-only), the model would see GT that hasn't happened yet during
offline training AND would silently degrade to 0 recall at runtime where those
future frames don't exist.

See also: tests/test_csctcn_causality.py for model-level causal convolution tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csc_lib.csc.config import CSCFeatureConfig
from csc_lib.csc.dataset import CSCDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_DIM = 11  # must match FEATURE_DIM from csc_lib.csc.features


def _make_fake_rows(n_frames: int, seed: int = 0) -> list[dict]:
    """Build a synthetic list of FrameLabel-compatible dicts."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_frames):
        cx = float(rng.uniform(0, 1280))
        cy = float(rng.uniform(0, 720))
        w = float(rng.uniform(20, 200))
        h = float(rng.uniform(20, 200))
        rows.append({
            "dataset": "synthetic",
            "sequence": "synthetic_seq",
            "frame_idx": i,
            "pred_bbox": [cx - w / 2, cy - h / 2, w, h],
            "gt_bbox": [cx - w / 2, cy - h / 2, w, h],
            "iou": float(rng.uniform(0, 1)),
            "confidence": float(rng.uniform(0.0, 1.0)),
            "apce": None,
            "psr": None,
            "localization_state": int(rng.integers(0, 3)),
            "confidence_state": int(rng.integers(0, 2)),
            "derived_state": int(rng.integers(0, 4)),
            "aux": {
                "occlusion": False,
                "out_of_view": False,
                "fast_motion": False,
                "scale_change": False,
                "distractor_risk": False,
            },
        })
    return rows


def _build_dataset(
    rows: list[dict],
    window_size: int = 20,
    image_size: tuple[int, int] = (1280, 720),
) -> CSCDataset:
    feature_cfg = CSCFeatureConfig(window_size=window_size)
    seq_rows = {("synthetic", "synthetic_seq"): rows}
    return CSCDataset(seq_rows, feature_cfg, image_size=image_size, stride=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWindowCausality:
    """CSCDataset window for anchor t must use rows[t-W+1 : t+1] only."""

    SEQ_LEN = 64
    WINDOW_SIZE = 20
    # t=32 is well inside the sequence; verify window for anchor frame 32.
    T_ANCHOR = 32

    def test_window_matches_expected_slice(self) -> None:
        """Window index for anchor t covers exactly rows[t-W+1 : t+1].

        Note: build_sequence_features() is stateful (velocity/acceleration
        depend on prior frames in the full sequence).  The window returned by
        CSCDataset is a slice of the feature matrix built from the FULL sequence
        starting at row 0 — so we replicate that by building the full feature
        matrix and slicing it ourselves.  The key invariant is that the dataset
        slice [t-W+1 : t+1] matches the window, meaning it is strictly past-only.
        """
        np.random.seed(0)
        rows = _make_fake_rows(self.SEQ_LEN, seed=0)
        ds = _build_dataset(rows, window_size=self.WINDOW_SIZE)

        # Dataset builds windows for end in [W, T] (inclusive via range(W, T+1)).
        # Window with end=t+1 covers rows[t-W+1 : t+1].
        # For t=T_ANCHOR: end = T_ANCHOR+1, window_idx = end - W = T_ANCHOR - W + 1.
        window_idx = self.T_ANCHOR - self.WINDOW_SIZE + 1
        assert 0 <= window_idx < len(ds), (
            f"window_idx={window_idx} out of range for dataset of size {len(ds)}"
        )

        sample = ds[window_idx]
        feats = sample["features"].numpy()  # (W, F)

        # Build the full sequence feature matrix and slice it — this replicates
        # exactly what CSCDataset does internally.
        from csc_lib.csc.features import build_sequence_features
        full_feats = build_sequence_features(rows, (1280, 720))
        expected_feats = full_feats[self.T_ANCHOR - self.WINDOW_SIZE + 1 : self.T_ANCHOR + 1]

        np.testing.assert_allclose(
            feats,
            expected_feats,
            atol=1e-5,
            err_msg=(
                f"CSCDataset window for anchor t={self.T_ANCHOR} does not match "
                f"a past-only slice of the full feature matrix. "
                f"Expected rows[{self.T_ANCHOR-self.WINDOW_SIZE+1}:{self.T_ANCHOR+1}]."
            ),
        )

    def test_future_mutation_does_not_affect_window(self) -> None:
        """Mutating rows[T_ANCHOR+1:] must not change the window for anchor T_ANCHOR.

        This is the key causality property: the dataset pipeline must NOT read
        any frame after the anchor into the window.

        See also: tests/test_csctcn_causality.py for the model-level analogue.
        """
        np.random.seed(0)
        rows_original = _make_fake_rows(self.SEQ_LEN, seed=0)

        # Build dataset from original rows, extract window for T_ANCHOR.
        ds_original = _build_dataset(rows_original, window_size=self.WINDOW_SIZE)
        window_idx = self.T_ANCHOR - self.WINDOW_SIZE + 1
        feat_original = ds_original[window_idx]["features"].numpy().copy()

        # Clone rows and replace rows[T_ANCHOR+1:] with garbage.
        import copy
        rows_mutated = copy.deepcopy(rows_original)
        rng = np.random.default_rng(999)
        for i in range(self.T_ANCHOR + 1, self.SEQ_LEN):
            # Overwrite all numeric fields with extreme garbage values.
            rows_mutated[i]["pred_bbox"] = [
                float(rng.uniform(-9999, 9999)) for _ in range(4)
            ]
            rows_mutated[i]["confidence"] = float(rng.uniform(-50, 50))
            rows_mutated[i]["iou"] = float(rng.uniform(-10, 10))
            rows_mutated[i]["localization_state"] = 99   # invalid class
            rows_mutated[i]["derived_state"] = 99

        # Build dataset from mutated rows; extract same window.
        ds_mutated = _build_dataset(rows_mutated, window_size=self.WINDOW_SIZE)
        feat_mutated = ds_mutated[window_idx]["features"].numpy().copy()

        np.testing.assert_array_equal(
            feat_original,
            feat_mutated,
            err_msg=(
                f"CSCDataset window for anchor t={self.T_ANCHOR} changed after "
                f"mutating rows[{self.T_ANCHOR+1}:]. "
                "The window must contain only frames t-W+1..t — future-frame leakage detected!"
            ),
        )

    def test_window_size_is_correct(self) -> None:
        """Each window in CSCDataset must have exactly W frames."""
        np.random.seed(0)
        rows = _make_fake_rows(self.SEQ_LEN, seed=0)
        ds = _build_dataset(rows, window_size=self.WINDOW_SIZE)

        for idx in range(len(ds)):
            sample = ds[idx]
            W_actual = sample["features"].shape[0]
            assert W_actual == self.WINDOW_SIZE, (
                f"Window {idx} has {W_actual} frames, expected {self.WINDOW_SIZE}."
            )

    def test_first_window_uses_first_w_frames(self) -> None:
        """The very first window (index 0) must cover rows[0:W].

        The first window starts at row 0, so state accumulates from the very
        start — build_sequence_features(rows[:W]) will match exactly because
        there is no prior history before row 0.
        """
        np.random.seed(0)
        rows = _make_fake_rows(self.SEQ_LEN, seed=0)
        ds = _build_dataset(rows, window_size=self.WINDOW_SIZE)

        from csc_lib.csc.features import build_sequence_features
        expected_feats = build_sequence_features(rows[:self.WINDOW_SIZE], (1280, 720))
        actual_feats = ds[0]["features"].numpy()

        np.testing.assert_allclose(actual_feats, expected_feats, atol=1e-5)

    @pytest.mark.parametrize("t_anchor", [10, 20, 40, 63])
    def test_parametric_anchors(self, t_anchor: int) -> None:
        """Generalised: for several anchor positions, future mutation must not matter."""
        seq_len = 64
        window_size = 10
        rows_original = _make_fake_rows(seq_len, seed=42)

        import copy
        rows_mutated = copy.deepcopy(rows_original)
        rng = np.random.default_rng(t_anchor)
        for i in range(t_anchor + 1, seq_len):
            rows_mutated[i]["pred_bbox"] = [float(rng.uniform(-100, 100)) for _ in range(4)]
            rows_mutated[i]["confidence"] = float(rng.uniform(-5, 5))

        # Only test if anchor is reachable
        if t_anchor < window_size:
            pytest.skip(f"t_anchor={t_anchor} < window_size={window_size}")

        ds_orig = _build_dataset(rows_original, window_size=window_size)
        ds_mutated = _build_dataset(rows_mutated, window_size=window_size)

        window_idx = t_anchor - window_size + 1
        if window_idx >= len(ds_orig):
            pytest.skip(f"window_idx={window_idx} out of range (ds size {len(ds_orig)})")

        feat_orig = ds_orig[window_idx]["features"].numpy()
        feat_mutated = ds_mutated[window_idx]["features"].numpy()

        np.testing.assert_array_equal(
            feat_orig,
            feat_mutated,
            err_msg=(
                f"Future-frame leakage for t_anchor={t_anchor}: "
                "mutating frames after anchor changed the window features."
            ),
        )
