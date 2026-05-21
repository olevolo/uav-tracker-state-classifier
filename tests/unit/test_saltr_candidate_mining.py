"""Unit tests for SGLATrack candidate mining helpers.

These tests avoid loading tracker weights.  They protect the score-map
candidate geometry used by the offline false-confirmed pilot.
"""

from __future__ import annotations

import numpy as np
import torch


def test_select_candidate_peak_indices_suppresses_adjacent_cells() -> None:
    from uav_tracker.trackers.sglatrack import _select_candidate_peak_indices

    score = torch.zeros(1, 1, 16, 16)
    score[0, 0, 8, 8] = 1.0
    score[0, 0, 8, 9] = 0.95  # adjacent same mode, should be suppressed
    score[0, 0, 2, 3] = 0.80
    score[0, 0, 13, 12] = 0.70

    peaks = _select_candidate_peak_indices(score, max_candidates=3, nms_radius=2)

    coords = [divmod(idx, 16) for idx in peaks]
    assert coords == [(8, 8), (2, 3), (13, 12)]


def test_candidate_bbox_from_peak_maps_search_center_to_frame_center() -> None:
    from uav_tracker.trackers.sglatrack import _extract_candidate_diagnostics
    from uav_tracker.types import BBox

    score = torch.zeros(1, 1, 16, 16)
    score[0, 0, 8, 8] = 1.0
    size = torch.full((1, 2, 16, 16), 10.0 * 2.0 / 256.0)
    offset = torch.zeros(1, 2, 16, 16)
    prev = BBox(x=80.0, y=80.0, w=40.0, h=40.0)  # center (100, 100)

    diag = _extract_candidate_diagnostics(
        score_map=score,
        size_map=size,
        offset_map=offset,
        prev_bbox=prev,
        resize_factor=2.0,
        max_candidates=1,
    )

    cand = diag["candidates"][0]
    assert cand["row"] == 8
    assert cand["col"] == 8
    assert np.allclose(cand["bbox"], [95.0, 95.0, 10.0, 10.0], atol=1e-5)


def test_candidate_frame_record_marks_oracle_topk_hit() -> None:
    from salt_r.candidate_mining_pilot import _candidate_frame_record

    candidates = [
        {"bbox": [100, 100, 20, 20]},
        {"bbox": [10, 10, 20, 20]},
        {"bbox": [50, 50, 20, 20]},
    ]
    gt = np.array([52, 52, 20, 20], dtype=np.float32)
    pred = np.array([100, 100, 20, 20], dtype=np.float32)

    rec = _candidate_frame_record(
        frame_idx=7,
        candidates=candidates,
        gt_bbox_xywh=gt,
        tracker_bbox_xywh=pred,
        false_confirmed=True,
    )

    assert rec["false_confirmed"] is True
    assert rec["tracker_iou"] == 0.0
    assert rec["best_rank"] == 2
    assert rec["top3_hit_iou03"] is True
    assert rec["top5_hit_iou03"] is True
