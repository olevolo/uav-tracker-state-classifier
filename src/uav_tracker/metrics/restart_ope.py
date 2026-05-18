"""Restart-OPE evaluation harness (PLAN §10 + OTB restart protocol).

Restart-OPE is a more honest variant of plain OPE for long sequences: when
the tracker fails on a frame (IoU < ``threshold`` for that single frame),
we record a failure, skip ``restart_gap`` frames, and re-initialise the
tracker on the next available GT bbox.  This prevents a single drift error
from contaminating the entire remainder of a long sequence.

Reference: OTB restart protocol (Wu et al. 2015).

Usage example::

    from uav_tracker.metrics.restart_ope import RestartOPE
    result = RestartOPE(threshold=0.5, restart_gap=5).run(tracker, dataset)
    print(result.mean_success_rate, result.total_restarts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from uav_tracker.metrics.success import iou
from uav_tracker.types import BBox

_log = logging.getLogger(__name__)

_UAV123_ATTRS = frozenset(
    ["FM", "OCC", "IV", "SV", "POC", "DEF", "MB", "CM", "BC", "SOB", "LR", "ARC"]
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RestartSequenceResult:
    """Per-sequence restart-OPE result.

    Fields
    ------
    name:
        Sequence name.
    success_rate:
        Fraction of frames where IoU ≥ threshold (including re-init frames,
        which are always counted as successes).
    n_restarts:
        Number of times the tracker was re-initialised after a failure.
    n_frames:
        Total number of frames in the sequence (including frame 0).
    aux:
        Free-form extra metadata (e.g. ``{"iou_series": [...]}`).
    """

    name: str
    success_rate: float
    n_restarts: int
    n_frames: int
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class RestartOPEResult:
    """Aggregate restart-OPE result across a dataset.

    Mirrors the shape of ``OPEResult`` so downstream report writers can be
    extended without deep coupling.

    Fields
    ------
    mean_success_rate:
        Mean of per-sequence success rates.
    total_restarts:
        Total number of tracker re-initialisations across all sequences.
    per_sequence:
        Individual :class:`RestartSequenceResult` objects, one per sequence.
    aux:
        Dataset-level extras (e.g. tracker name, threshold used).
    """

    mean_success_rate: float
    total_restarts: int
    per_sequence: list[RestartSequenceResult] = field(default_factory=list)
    aux: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------


def _iou_single(a: BBox, b: BBox) -> float:
    """IoU between two BBox objects (xywh)."""
    a_arr = np.array([[a.x, a.y, a.w, a.h]], dtype=np.float64)
    b_arr = np.array([[b.x, b.y, b.w, b.h]], dtype=np.float64)
    result = iou(a_arr, b_arr)
    return float(result[0])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RestartOPE:
    """Restart-based One-Pass Evaluation (OTB restart protocol).

    Parameters
    ----------
    threshold:
        IoU threshold below which a frame is counted as a failure and a
        restart is triggered.  Default: 0.5 (OTB canonical value).
    restart_gap:
        Number of frames to *skip* after a failure before re-initialising
        the tracker.  During the gap, frames are not evaluated.  Default: 5.

    Notes
    -----
    - Frame 0 is always used for initialisation (``tracker.init``).
    - After a failure on frame *t*, frames *t+1 … t+restart_gap* are
      skipped.  Frame *t+restart_gap+1* (if valid GT exists) triggers a
      re-init and is counted as a success (by convention, matching OTB).
    - If a GT bbox is invalid (``valid=False`` on ``_BBoxAnnotated``
      subclasses), that frame is skipped for both evaluation and re-init.
    """

    def __init__(self, threshold: float = 0.5, restart_gap: int = 5) -> None:
        self.threshold = threshold
        self.restart_gap = restart_gap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        tracker: Any,
        dataset: Iterable[Any],
        limit: int | None = None,
    ) -> RestartOPEResult:
        """Run restart-OPE over *dataset* (or its first *limit* sequences).

        Parameters
        ----------
        tracker:
            A :class:`~uav_tracker.trackers.base.Tracker`-protocol object.
            Must expose ``init(frame, bbox)`` and ``update(frame) -> TrackState``.
        dataset:
            Iterable of ``Sequence`` objects (each with ``frames``,
            ``ground_truth``, ``name``).
        limit:
            If set, only the first *limit* sequences are evaluated.

        Returns
        -------
        RestartOPEResult
        """
        seq_results: list[RestartSequenceResult] = []

        for seq_idx, seq in enumerate(dataset):
            if limit is not None and seq_idx >= limit:
                break

            frames = list(seq.frames)
            gt_bboxes: list[BBox] = seq.ground_truth

            if len(frames) < 2:
                _log.debug("RestartOPE: skipping %s — fewer than 2 frames", seq.name)
                continue

            result = self._run_sequence(tracker, seq.name, frames, gt_bboxes)
            seq_results.append(result)

        if not seq_results:
            return RestartOPEResult(
                mean_success_rate=0.0,
                total_restarts=0,
                per_sequence=[],
                aux={"threshold": self.threshold, "restart_gap": self.restart_gap},
            )

        mean_sr = float(np.mean([r.success_rate for r in seq_results]))
        total_restarts = sum(r.n_restarts for r in seq_results)

        return RestartOPEResult(
            mean_success_rate=mean_sr,
            total_restarts=total_restarts,
            per_sequence=seq_results,
            aux={"threshold": self.threshold, "restart_gap": self.restart_gap},
        )

    # ------------------------------------------------------------------
    # Per-sequence inner loop
    # ------------------------------------------------------------------

    def _run_sequence(
        self,
        tracker: Any,
        name: str,
        frames: list[Any],
        gt_bboxes: list[BBox],
    ) -> RestartSequenceResult:
        """Run restart-OPE on a single sequence.

        The tracker is initialised on ``frames[0]`` with ``gt_bboxes[0]``.
        Subsequent frames (1..N-1) are evaluated one at a time.
        """
        n_frames = len(frames)
        n_evaluated = 0      # frames where we actually called update + scored
        n_successful = 0     # frames where IoU >= threshold (or re-init frame)
        n_restarts = 0

        # Check if GT bbox is valid (support for _BBoxAnnotated.valid).
        def _is_valid(bbox: BBox) -> bool:
            return bool(getattr(bbox, "valid", True))

        # Initialise on frame 0.
        if not _is_valid(gt_bboxes[0]):
            _log.warning("RestartOPE: %s — frame 0 GT invalid, skipping sequence", name)
            return RestartSequenceResult(
                name=name,
                success_rate=0.0,
                n_restarts=0,
                n_frames=n_frames,
            )

        tracker.init(frames[0], gt_bboxes[0])

        skip_until: int = -1  # frame index up to which we skip after a failure
        pending_reinit: bool = False  # True when next eligible frame triggers re-init

        frame_idx = 1
        while frame_idx < n_frames:
            gt = gt_bboxes[frame_idx] if frame_idx < len(gt_bboxes) else None

            # --- skip gap after failure ---
            if frame_idx <= skip_until:
                frame_idx += 1
                continue

            # --- re-init on first valid frame after gap ---
            if pending_reinit:
                if gt is not None and _is_valid(gt):
                    tracker.init(frames[frame_idx], gt)
                    # Re-init frame counts as successful by OTB convention.
                    n_evaluated += 1
                    n_successful += 1
                    pending_reinit = False
                frame_idx += 1
                continue

            # --- skip frames with invalid GT (e.g. NaN) ---
            if gt is None or not _is_valid(gt):
                frame_idx += 1
                continue

            # --- normal update ---
            state = tracker.update(frames[frame_idx])
            pred_bbox: BBox = state.bbox
            frame_iou = _iou_single(pred_bbox, gt)

            n_evaluated += 1

            if frame_iou >= self.threshold:
                n_successful += 1
            else:
                # Failure: trigger restart.
                n_restarts += 1
                skip_until = frame_idx + self.restart_gap
                pending_reinit = True

            frame_idx += 1

        # Success rate over evaluated frames (avoid div-by-zero).
        success_rate = (n_successful / n_evaluated) if n_evaluated > 0 else 0.0

        return RestartSequenceResult(
            name=name,
            success_rate=success_rate,
            n_restarts=n_restarts,
            n_frames=n_frames,
            aux={
                "n_evaluated": n_evaluated,
                "n_successful": n_successful,
            },
        )


__all__ = [
    "RestartOPE",
    "RestartOPEResult",
    "RestartSequenceResult",
]
