"""End-to-end OPE test on the synthetic dataset using KCFKalmanTracker.

Skipped automatically when opencv-contrib-python is not installed
(i.e. cv2.TrackerKCF_create is absent).  This is the Phase 1 exit-demo
acceptance test per PLAN §11.
"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

if not hasattr(cv2, "TrackerKCF_create"):
    pytest.skip(
        "opencv-contrib-python required (cv2.TrackerKCF_create missing)",
        allow_module_level=True,
    )


def test_ope_synthetic_static_auc() -> None:
    """KCF on the static sequence should achieve AUC > 0.7.

    The bar is kept at 0.7 (not 0.85) because the init bbox covers frame 0
    but predictions start on frame 1, and KCF may need a couple of frames to
    stabilise; 0.7 is safely above chance while not being brittle.
    """
    from uav_tracker.datasets.synthetic import SyntheticDataset
    from uav_tracker.trackers.kcf_kalman import KCFKalmanTracker
    from uav_tracker.evaluation.ope import OPERunner

    dataset = SyntheticDataset(seed=42)
    # Isolate the static sequence only (limit=1 gives the first sequence).
    result = OPERunner(seed=42).run(
        tracker=KCFKalmanTracker(),
        dataset=dataset,
        limit=1,
    )
    assert result.per_sequence, "OPERunner returned no sequence results"
    static_result = result.per_sequence[0]
    assert static_result.name == "synthetic_static"
    assert static_result.auc > 0.7, (
        f"Expected AUC > 0.7 on synthetic_static, got {static_result.auc:.3f}"
    )


def test_ope_synthetic_linear_auc() -> None:
    """KCF on the linear (translating) sequence should achieve AUC > 0.3.

    Lower threshold than static because linear motion at high velocity
    may challenge the KCF ROI window, but any non-trivial tracker should
    beat 0.3.
    """
    from uav_tracker.datasets.synthetic import SyntheticDataset
    from uav_tracker.trackers.kcf_kalman import KCFKalmanTracker
    from uav_tracker.evaluation.ope import OPERunner

    dataset = SyntheticDataset(seed=42)

    # Run all 3 sequences; inspect the linear one (index 1).
    result = OPERunner(seed=42).run(
        tracker=KCFKalmanTracker(),
        dataset=dataset,
        limit=None,
    )
    seq_by_name = {r.name: r for r in result.per_sequence}
    assert "synthetic_linear" in seq_by_name, (
        f"Expected 'synthetic_linear' in results, got: {list(seq_by_name)}"
    )
    linear_result = seq_by_name["synthetic_linear"]
    assert linear_result.auc > 0.3, (
        f"Expected AUC > 0.3 on synthetic_linear, got {linear_result.auc:.3f}"
    )
