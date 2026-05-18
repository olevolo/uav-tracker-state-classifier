"""Per-frame difficulty label generator for UAV123 ML pipeline (Phase 11).

Runs the Henriques-KCF baseline tracker + motion-entropy signal over a
sequence and assigns each frame a ``SceneClass`` label according to the
priority rules in ADR-0012.

Scene class derivation priority (highest wins when multiple rules fire):

  RECOVERY (4):    tracker ``status == "lost"`` for >= 10 consecutive frames
  RISK_LOSS (3):   IoU(gt, pred) < 0.2 for >= 2 consecutive frames
  CHALLENGING (2): IoU(gt, pred) < 0.5 for >= 3 consecutive frames
  LOW_RES (5):     bbox area < 400 px² (width * height < 400)
  MODERATE (1):    entropy H̄ > 0.5 sustained >= 5 frames (no IoU drop)
  CLEAR (0):       everything else

Usage
-----
    from uav_tracker.training.label_generator import LabelGenerator
    from uav_tracker.datasets.uav123 import UAV123Dataset

    gen = LabelGenerator(seed=42)
    for seq in UAV123Dataset(root="data/uav123"):
        labels = gen.generate(seq)  # uint8 array shape (n_frames,)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from uav_tracker.types import BBox

_log = logging.getLogger(__name__)

# Scene class integer codes (mirror uav123_ml.py constants).
SCENE_CLEAR = 0
SCENE_MODERATE = 1
SCENE_CHALLENGING = 2
SCENE_RISK_LOSS = 3
SCENE_RECOVERY = 4
SCENE_LOW_RES = 5


class LabelGenerator:
    """Runs KCF baseline + entropy signal to generate per-frame scene class labels.

    Parameters
    ----------
    tracker_name:
        Registry name of the tracker to use (default ``"kcf_henriques"``).
    signal_name:
        Registry name of the entropy signal (default ``"motion_entropy"``).
    iou_hard_threshold:
        IoU threshold below which a frame is considered ``CHALLENGING`` (0.5).
    iou_loss_threshold:
        IoU threshold below which a frame is considered ``RISK_LOSS`` (0.2).
    entropy_moderate_threshold:
        EMA entropy threshold above which ``MODERATE`` is triggered (0.5).
    min_window:
        Minimum run length for the CHALLENGING rule (ADR: 3 frames).
    seed:
        RNG seed passed to the ``HybridRunner`` for determinism.
    """

    def __init__(
        self,
        tracker_name: str = "kcf_henriques",
        signal_name: str = "motion_entropy",
        iou_hard_threshold: float = 0.5,
        iou_loss_threshold: float = 0.2,
        entropy_moderate_threshold: float = 0.5,
        min_window: int = 3,
        seed: int = 42,
    ) -> None:
        self.tracker_name = tracker_name
        self.signal_name = signal_name
        self.iou_hard_threshold = iou_hard_threshold
        self.iou_loss_threshold = iou_loss_threshold
        self.entropy_moderate_threshold = entropy_moderate_threshold
        self.min_window = min_window
        self.seed = seed

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate(self, sequence: Any) -> np.ndarray:
        """Run the tracker/signal pipeline over *sequence* and return scene labels.

        Parameters
        ----------
        sequence:
            Any object conforming to the ``Sequence`` Protocol (has ``frames``,
            ``ground_truth``, ``init_bbox``).

        Returns
        -------
        np.ndarray
            uint8 array of shape ``(n_frames,)`` where entry ``i`` is the
            ``SceneClass`` code for that frame (0–5).
        """
        from uav_tracker.registry import TRACKERS, SIGNALS, SCHEDULERS

        gt_bboxes: list[BBox] = sequence.ground_truth
        n_frames = len(gt_bboxes)

        if n_frames == 0:
            return np.zeros(0, dtype=np.uint8)

        # Build a minimal single-tier runner: tracker + entropy signal + stub scheduler.
        try:
            tracker = TRACKERS.build(self.tracker_name)
        except Exception as exc:
            _log.warning("LabelGenerator: cannot build tracker %r: %s", self.tracker_name, exc)
            return np.zeros(n_frames, dtype=np.uint8)

        try:
            signal = SIGNALS.build(self.signal_name, seed=self.seed)
        except Exception as exc:
            _log.warning("LabelGenerator: cannot build signal %r: %s", self.signal_name, exc)
            signal = None

        scheduler = _NoOpScheduler()

        from uav_tracker.runner import HybridRunner

        runner = HybridRunner(
            trackers={0: tracker},
            signals={self.signal_name: signal} if signal is not None else {},
            scheduler=scheduler,
            seed=self.seed,
        )

        # ------------------------------------------------------------------- #
        # Collect per-frame tracker outputs and signal values.                 #
        # ------------------------------------------------------------------- #

        # Frame 0 is the init frame — the runner doesn't yield a TelemetryEntry
        # for it (loop starts at frame 1).
        pred_bboxes: list[BBox | None] = [gt_bboxes[0]]  # init = gt
        statuses: list[str] = ["locked"]  # frame 0 always "locked"
        entropy_values: list[float] = [0.0]  # frame 0 — no signal yet

        try:
            for entry in runner.run(sequence):
                pred_bboxes.append(entry.bbox)
                # Re-derive status from confidence using same thresholds as KCF.
                status = "locked"
                if entry.confidence < 0.15:
                    status = "lost"
                elif entry.confidence < 0.6:
                    status = "uncertain"
                statuses.append(status)
                entropy_values.append(entry.signals.get(self.signal_name, 0.0))
        except Exception as exc:
            _log.warning(
                "LabelGenerator: runner.run() failed for sequence %s: %s",
                getattr(sequence, "name", "?"),
                exc,
            )
            # Fall back to all-CLEAR labels.
            return np.zeros(n_frames, dtype=np.uint8)

        # Align lengths (runner may yield fewer entries for short sequences).
        n_actual = len(pred_bboxes)
        if n_actual > n_frames:
            pred_bboxes = pred_bboxes[:n_frames]
            statuses = statuses[:n_frames]
            entropy_values = entropy_values[:n_frames]
        elif n_actual < n_frames:
            # Pad with last-known values.
            last_bbox = pred_bboxes[-1] if pred_bboxes else gt_bboxes[0]
            pred_bboxes.extend([last_bbox] * (n_frames - n_actual))
            statuses.extend(["lost"] * (n_frames - n_actual))
            entropy_values.extend([0.0] * (n_frames - n_actual))

        # ------------------------------------------------------------------- #
        # Compute IoU trace.                                                   #
        # ------------------------------------------------------------------- #
        iou_trace = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            gt = gt_bboxes[i]
            valid = getattr(gt, "valid", True)
            if valid and pred_bboxes[i] is not None:
                iou_trace[i] = self._compute_iou(gt, pred_bboxes[i])
            else:
                iou_trace[i] = 1.0 if i == 0 else 0.0

        # ------------------------------------------------------------------- #
        # Apply labelling rules (priority order: highest first).               #
        # ------------------------------------------------------------------- #
        labels = np.zeros(n_frames, dtype=np.uint8)

        # --- RECOVERY (4): status == "lost" for >= 10 consecutive frames ----
        self._apply_run_rule(
            labels,
            condition=np.array([s == "lost" for s in statuses], dtype=bool),
            run_len=10,
            label_code=SCENE_RECOVERY,
            overwrite_lower=False,  # RECOVERY is highest priority
        )

        # --- RISK_LOSS (3): IoU < 0.2 for >= 2 consecutive frames -----------
        self._apply_run_rule(
            labels,
            condition=iou_trace < self.iou_loss_threshold,
            run_len=2,
            label_code=SCENE_RISK_LOSS,
            overwrite_lower=True,
        )

        # --- CHALLENGING (2): IoU < 0.5 for >= 3 consecutive frames ---------
        self._apply_run_rule(
            labels,
            condition=iou_trace < self.iou_hard_threshold,
            run_len=self.min_window,
            label_code=SCENE_CHALLENGING,
            overwrite_lower=True,
        )

        # --- LOW_RES (5): bbox area < 400 px² --------------------------------
        for i in range(n_frames):
            if labels[i] == SCENE_CLEAR:  # only set if not already labelled higher
                gt = gt_bboxes[i]
                if gt.w * gt.h < 400.0:
                    labels[i] = SCENE_LOW_RES

        # --- MODERATE (1): entropy > threshold for >= 5 consecutive frames ---
        entropy_arr = np.array(entropy_values, dtype=np.float32)
        self._apply_run_rule(
            labels,
            condition=entropy_arr > self.entropy_moderate_threshold,
            run_len=5,
            label_code=SCENE_MODERATE,
            overwrite_lower=True,
            only_if_zero=True,  # MODERATE only applies if no higher class set
        )

        return labels

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _apply_run_rule(
        labels: np.ndarray,
        condition: np.ndarray,
        run_len: int,
        label_code: int,
        overwrite_lower: bool,
        only_if_zero: bool = False,
    ) -> None:
        """Apply a consecutive-run labelling rule in place.

        Scans *condition* (bool array) for runs of length >= *run_len*.
        For each frame in a qualifying run the label is set if:
          - ``only_if_zero`` is True  → only overwrite ``CLEAR`` (0) frames.
          - ``overwrite_lower`` is True → overwrite frames with label < label_code.
          - Otherwise → overwrite any frame (called for highest-priority rule).
        """
        n = len(condition)
        i = 0
        while i < n:
            if condition[i]:
                j = i
                while j < n and condition[j]:
                    j += 1
                run_length = j - i
                if run_length >= run_len:
                    for k in range(i, j):
                        if only_if_zero:
                            if labels[k] == SCENE_CLEAR:
                                labels[k] = label_code
                        elif overwrite_lower:
                            if labels[k] < label_code:
                                labels[k] = label_code
                        else:
                            labels[k] = label_code
                i = j
            else:
                i += 1

    @staticmethod
    def _compute_iou(gt: BBox, pred: BBox) -> float:
        """Intersection-over-Union of two (x, y, w, h) bounding boxes."""
        ax1, ay1 = gt.x, gt.y
        ax2, ay2 = gt.x + gt.w, gt.y + gt.h
        bx1, by1 = pred.x, pred.y
        bx2, by2 = pred.x + pred.w, pred.y + pred.h

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter = inter_w * inter_h

        area_a = max(0.0, gt.w) * max(0.0, gt.h)
        area_b = max(0.0, pred.w) * max(0.0, pred.h)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Stub scheduler (always stays on tier 0)
# ---------------------------------------------------------------------------


class _NoOpScheduler:
    """Minimal scheduler that keeps the runner permanently on tier 0."""

    name: str = "_noop"
    tiers: int = 1

    def decide(self, signals: Any, current_tier: int, frame_idx: int) -> Any:
        from uav_tracker.types import SchedulerDecision
        return SchedulerDecision(tier=0, reason="noop", switched=False)

    def reset(self) -> None:
        pass


__all__ = [
    "LabelGenerator",
    "SCENE_CLEAR", "SCENE_MODERATE", "SCENE_CHALLENGING",
    "SCENE_RISK_LOSS", "SCENE_RECOVERY", "SCENE_LOW_RES",
]
