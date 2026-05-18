"""Integration test: OPE run on SyntheticDataset with SiamFCTracker (Phase 2).

Validates that the SiamFC plugin wires correctly into the OPERunner
end-to-end without requiring pre-trained weights or a GPU.

Assertions:
  - Run completes without raising.
  - Per-sequence AUC is in [0, 1].
  - Per-sequence FPS > 0.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("torch")


def test_ope_siamfc_synthetic():
    """SiamFCTracker runs one synthetic sequence without error."""
    from uav_tracker.datasets.synthetic import SyntheticDataset
    from uav_tracker.evaluation.ope import OPERunner
    from uav_tracker.trackers.siamese.siamfc import SiamFCTracker

    tracker = SiamFCTracker(device="cpu")
    dataset = SyntheticDataset(seed=42)
    runner = OPERunner(seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = runner.run(tracker=tracker, dataset=dataset, limit=1)

    assert len(result.per_sequence) == 1
    sr = result.per_sequence[0]

    # AUC must be in [0, 1] regardless of random weights
    assert 0.0 <= sr.auc <= 1.0, f"AUC out of range: {sr.auc}"
    # FPS must be positive (tracker executed at least one update)
    assert sr.fps > 0.0, f"Expected FPS > 0, got {sr.fps}"
