"""One-Pass Evaluation (OPE) harness.

Paper / PLAN §10.1 + §11 Phase 1: init the tracker on frame 0 of each
sequence, run straight through without re-init, collect per-frame IoU
vs ground truth. Success AUC + Precision@20 + FPS are computed over the
full sequence then averaged.

The module defines a small ``OPEResult`` dataclass so downstream report
writers (``.report``) can dump CSV/markdown without deep coupling.

Phase 3 extension: if ``tracker`` is a ``HybridRunner``-shaped object
(has a ``trackers`` dict attribute), the per-frame loop is delegated to
``HybridRunner.run()``; per-sequence ``time_in_tier`` is collected and
stored in ``SequenceResult.aux``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from uav_tracker.metrics.success import compute_auc
from uav_tracker.metrics.precision import precision_at_threshold
from uav_tracker.types import BBox


@dataclass
class SequenceResult:
    """Single-sequence OPE result."""

    name: str
    auc: float
    precision_at_20: float
    fps: float
    n_frames: int
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class OPEResult:
    """Aggregate OPE result across a dataset.

    Fields match PLAN §11 Phase 1 exit demo:
        AUC, Pr@20, FPS, GFLOPs/frame (latter lives in ``aux``).
    """

    auc: float
    precision_at_20: float
    fps: float
    per_sequence: list[SequenceResult] = field(default_factory=list)
    aux: dict[str, Any] = field(default_factory=dict)


def _is_hybrid_runner(obj: Any) -> bool:
    """Return True if ``obj`` looks like a HybridRunner (duck-type check)."""
    return hasattr(obj, "trackers") and isinstance(getattr(obj, "trackers", None), dict)


class OPERunner:
    """One-Pass Evaluation driver.

    Consumes any ``Tracker`` (conforming to the Protocol) and any
    ``Dataset`` (iterable of sequences). Produces an ``OPEResult``.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(
        self,
        tracker: Any,
        dataset: Iterable[Any],
        limit: int | None = None,
    ) -> OPEResult:
        """Run OPE over ``dataset`` (or its first ``limit`` sequences).

        Per PLAN §10.1 hard contract: init on frame 0, no re-init on
        failure. Average AUC is computed as the integral of the
        Success curve over IoU thresholds [0, 1].

        Parameters
        ----------
        tracker:
            A ``Tracker``-protocol-conforming object, or a ``HybridRunner``
            instance (detected by duck-typing on ``.trackers`` dict attribute).
        dataset:
            Iterable of ``Sequence`` objects (each has ``frames``,
            ``ground_truth``, ``init_bbox``).
        limit:
            If set, only the first ``limit`` sequences are evaluated.
        """
        if _is_hybrid_runner(tracker):
            return self._run_hybrid(tracker, dataset, limit)
        return self._run_simple(tracker, dataset, limit)

    # ------------------------------------------------------------------

    def _run_simple(
        self,
        tracker: Any,
        dataset: Iterable[Any],
        limit: int | None,
    ) -> OPEResult:
        """Phase 1 / 2 single-tracker OPE path."""
        seq_results: list[SequenceResult] = []

        for seq_idx, seq in enumerate(dataset):
            if limit is not None and seq_idx >= limit:
                break

            frames = list(seq.frames)
            gt_bboxes: list[BBox] = seq.ground_truth

            if len(frames) < 2:
                continue

            # --- Init on frame 0 (not timed toward FPS) ---
            tracker.init(frames[0], gt_bboxes[0])

            pred_bboxes: list[BBox] = []
            update_times: list[float] = []

            # --- One-pass: no re-init on failure ---
            for frame in frames[1:]:
                t0 = time.perf_counter()
                state = tracker.update(frame)
                t1 = time.perf_counter()
                pred_bboxes.append(state.bbox)
                update_times.append(t1 - t0)

            # Ground-truth for frames 1..N-1 (paired with predictions).
            gt_tail = gt_bboxes[1:]

            # Convert to numpy arrays (N, 4) xywh for metric functions.
            gt_arr = np.array(
                [[b.x, b.y, b.w, b.h] for b in gt_tail], dtype=np.float64
            )
            pred_arr = np.array(
                [[b.x, b.y, b.w, b.h] for b in pred_bboxes], dtype=np.float64
            )

            auc = compute_auc(gt_arr, pred_arr)
            pr20 = precision_at_threshold(gt_arr, pred_arr, threshold=20.0)
            mean_update_s = float(np.mean(update_times)) if update_times else 1.0
            fps = 1.0 / mean_update_s if mean_update_s > 0 else 0.0

            seq_results.append(
                SequenceResult(
                    name=seq.name,
                    auc=auc,
                    precision_at_20=pr20,
                    fps=fps,
                    n_frames=len(frames),
                )
            )

        if not seq_results:
            return OPEResult(auc=0.0, precision_at_20=0.0, fps=0.0, per_sequence=[])

        mean_auc = float(np.mean([r.auc for r in seq_results]))
        mean_pr20 = float(np.mean([r.precision_at_20 for r in seq_results]))
        mean_fps = float(np.mean([r.fps for r in seq_results]))

        return OPEResult(
            auc=mean_auc,
            precision_at_20=mean_pr20,
            fps=mean_fps,
            per_sequence=seq_results,
        )

    def _run_hybrid(
        self,
        runner: Any,
        dataset: Iterable[Any],
        limit: int | None,
    ) -> OPEResult:
        """Phase 3+ HybridRunner OPE path.

        Delegates the per-frame loop to ``HybridRunner.run()`` which
        yields ``TelemetryEntry`` objects.  Collects trajectory + tier
        time from the runner's post-run attributes.
        """
        seq_results: list[SequenceResult] = []

        for seq_idx, seq in enumerate(dataset):
            if limit is not None and seq_idx >= limit:
                break

            frames = list(seq.frames)
            gt_bboxes: list[BBox] = seq.ground_truth

            if len(frames) < 2:
                continue

            # Consume the generator to drive the full sequence.
            t_start = time.perf_counter()
            entries = list(runner.run(seq))
            t_end = time.perf_counter()

            pred_bboxes: list[BBox] = [e.bbox for e in entries]
            n_frames = len(frames)
            n_update = len(entries)  # frames[1:]

            # Ground-truth for frames 1..N-1
            gt_tail = gt_bboxes[1:]

            gt_arr = np.array(
                [[b.x, b.y, b.w, b.h] for b in gt_tail], dtype=np.float64
            )
            pred_arr = np.array(
                [[b.x, b.y, b.w, b.h] for b in pred_bboxes], dtype=np.float64
            )

            auc = compute_auc(gt_arr, pred_arr) if n_update > 0 else 0.0
            pr20 = (
                precision_at_threshold(gt_arr, pred_arr, threshold=20.0)
                if n_update > 0
                else 0.0
            )
            elapsed = t_end - t_start
            fps = n_update / elapsed if elapsed > 0 else 0.0

            # time_in_tier from runner attribute (populated by run()).
            time_in_tier: dict[int, int] = dict(getattr(runner, "time_in_tier", {}))

            seq_results.append(
                SequenceResult(
                    name=seq.name,
                    auc=auc,
                    precision_at_20=pr20,
                    fps=fps,
                    n_frames=n_frames,
                    aux={"time_in_tier": time_in_tier},
                )
            )

        if not seq_results:
            return OPEResult(auc=0.0, precision_at_20=0.0, fps=0.0, per_sequence=[])

        mean_auc = float(np.mean([r.auc for r in seq_results]))
        mean_pr20 = float(np.mean([r.precision_at_20 for r in seq_results]))
        mean_fps = float(np.mean([r.fps for r in seq_results]))

        return OPEResult(
            auc=mean_auc,
            precision_at_20=mean_pr20,
            fps=mean_fps,
            per_sequence=seq_results,
        )
