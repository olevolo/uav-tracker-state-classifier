"""Unit tests for the DINOv2 identity pilot helpers.

These tests intentionally do not load the DINOv2 model.  They protect the
bbox/crop/selection logic around the heavy offline teacher path.
"""

from __future__ import annotations

import numpy as np


def test_square_context_crop_pads_near_border() -> None:
    from salt_r.dino_identity_pilot import _square_context_crop_bgr

    frame = np.full((20, 30, 3), 127, dtype=np.uint8)
    bbox = np.array([-5, -4, 8, 6], dtype=np.float32)

    crop = _square_context_crop_bgr(frame, bbox, context_scale=2.0, min_side=16)

    assert crop.shape == (16, 16, 3)
    assert crop.dtype == np.uint8
    assert crop.max() == 127
    assert crop.min() == 0, "out-of-frame area must be padded, not wrapped"


def test_choose_frame_indices_keeps_all_false_confirmed() -> None:
    from salt_r.dino_identity_pilot import _choose_frame_indices

    labels = np.zeros((20, 2), dtype=np.float32)
    labels[[3, 7, 19], 1] = 1.0

    idx = _choose_frame_indices(
        labels=labels,
        fc_idx=1,
        frame_stride=5,
        include_all_fc=True,
        max_frames_per_seq=0,
    )

    assert 0 in idx
    assert {3, 7, 19}.issubset(set(idx.tolist()))
    assert {0, 5, 10, 15}.issubset(set(idx.tolist()))


def test_safe_metric_table_uses_identity_risk_sign() -> None:
    from salt_r.dino_identity_pilot import _FEATURE_NAMES, _safe_metric_table

    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    features = np.zeros((4, len(_FEATURE_NAMES)), dtype=np.float32)
    init_idx = _FEATURE_NAMES.index("dino_init_sim")
    # Correct frames have high similarity; false-confirmed frames low similarity.
    features[:, init_idx] = np.array([0.95, 0.90, 0.30, 0.20], dtype=np.float32)

    result = _safe_metric_table(labels, features, list(_FEATURE_NAMES))

    assert result["features"]["1_minus_dino_init_sim"]["auroc"] > 0.99
    assert result["features"]["1_minus_dino_init_sim"]["auprc"] > 0.99
