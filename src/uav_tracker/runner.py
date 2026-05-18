"""HybridRunner — composes trackers / signals / scheduler (PLAN §3).

Determinism contract
--------------------
Given identical inputs (same sequence, same plugin configs, same seed),
``HybridRunner.run`` MUST yield identical ``TelemetryEntry`` sequences,
bit-for-bit. This is enforced by:

    * Every plugin exposes a ``reset()`` that restores construction state.
    * ``run`` seeds ``numpy``, ``random``, and ``torch`` from the runner's
      ``seed`` kwarg before entering the per-frame loop.
    * The runner itself holds no learned state; all learning lives in
      plugins (see ADR-0010 future work).

Phase 6 additions
-----------------
* ``detectors`` mapping (tier index → Detector) alongside ``trackers``.
* When the scheduler transitions to a tier that is a *detector* (duck-type:
  has ``.detect()`` and no ``.update()``), HybridRunner calls
  ``_recover_with_detector`` to re-initialise lower-tier trackers.
* ``recoveries`` counter incremented on each successful re-initialisation.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from uav_tracker.types import BBox, Detection, FrameContext, FrameContextV2, SchedulerDecision, TrackState


@dataclass
class TelemetryEntry:
    """Per-frame record written to a JSONL stream for later analysis.

    Fields chosen per PLAN §11 exit demos so downstream plots (entropy
    timeline, mode bands, FPS) can reconstruct the full run from disk
    without replaying video.
    """

    frame_idx: int
    bbox: Any  # BBox
    confidence: float
    tier: int
    switched: bool
    signals: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)


def _is_detector(obj: Any) -> bool:
    """Duck-type check: does ``obj`` look like a Detector (has .detect())?"""
    return callable(getattr(obj, "detect", None))


def _is_tracker(obj: Any) -> bool:
    """Duck-type check: does ``obj`` look like a Tracker (has .update())?"""
    return callable(getattr(obj, "update", None))


def _iou(a: BBox, b: BBox) -> float:
    """Compute Intersection-over-Union of two (x, y, w, h) bboxes."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, a.w) * max(0.0, a.h)
    area_b = max(0.0, b.w) * max(0.0, b.h)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class HybridRunner:
    """Composes tier trackers/detectors, signals, and a scheduler into one loop.

    Parameters
    ----------
    trackers:
        Mapping from tier index to a concrete ``Tracker`` instance.
        Tier 0 is always required; tier 1 / tier 2 are optional.
    signals:
        Mapping from signal name to a ``SwitchSignal`` instance. Order
        does not matter; each is ``step``-ed every frame.  Can also be
        a list of signals (names resolved from ``signal.name`` attribute).
    scheduler:
        A ``Scheduler`` instance that reads ``SignalReport``s and
        emits a ``SchedulerDecision`` per frame.
    detector:
        Optional ``Detector`` for backwards-compat (single detector, used
        as-is). Takes precedence only if ``detectors`` is not provided.
    detectors:
        Optional mapping from tier index to ``Detector`` instance. If tier
        N maps to a detector-shaped object, the runner uses ``.detect()``
        instead of ``.update()`` when tier N is active, and triggers
        re-initialisation of lower-tier trackers.
    seed:
        RNG seed applied at the start of every ``run`` call.
    """

    def __init__(
        self,
        trackers: dict[int, Any],
        signals: dict[str, Any] | list[Any],
        scheduler: Any,
        detector: Any | None = None,
        detectors: dict[int, Any] | None = None,
        seed: int = 42,
        start_tier: int = 0,
        warmer: Any | None = None,
        scene_classifier: Any | None = None,
        appearance_memory: Any | None = None,
        motion_predictor: Any | None = None,
    ) -> None:
        if 0 not in trackers:
            raise ValueError("HybridRunner requires at least a tier-0 tracker")
        self.trackers = trackers
        # Accept both list and dict forms of signals.
        if isinstance(signals, list):
            self.signals: dict[str, Any] = {s.name: s for s in signals}
        else:
            self.signals = dict(signals)
        self.scheduler = scheduler

        # Build unified plugin map: all tiers (trackers + detectors).
        self._tier_plugins: dict[int, Any] = dict(trackers)
        if detectors:
            for tier, det in detectors.items():
                self._tier_plugins[tier] = det
        if detector is not None and not detectors:
            # Legacy single-detector support: assign to highest tier + 1.
            top_tier = max(self._tier_plugins) + 1 if self._tier_plugins else 1
            self._tier_plugins[top_tier] = detector

        self.detector = detector  # kept for external access
        self.detectors = detectors or {}
        self.seed = seed
        self.start_tier = start_tier
        self.warmer = warmer  # optional ModelWarmer; called before frame-0 in run()

        # Phase 14 — optional ML module hooks (additive; None = no-op)
        self._scene_classifier = scene_classifier
        self._appearance_memory = appearance_memory
        self._motion_predictor = motion_predictor

        # Per-run state (populated by run / init).
        self._last_state: TrackState | None = None
        self._current_tier: int = start_tier
        self.trajectory: list[BBox] = []
        self.tier_sequence: list[int] = []
        self.time_in_tier: dict[int, int] = {t: 0 for t in self._tier_plugins}
        self._update_times: list[float] = []
        self.recoveries: int = 0  # Phase 6: re-detection recovery counter

    # ------------------------------------------------------------------
    # Public init / run interface

    def init(self, first_frame: np.ndarray, first_bbox: BBox) -> None:
        """Initialize ALL registered trackers on the first frame.

        Phase 3 requirement: all tier trackers are warm from frame 0 so
        there is no 1-frame lag when the scheduler first escalates.
        Detector-typed tiers are skipped (no ``.init()`` method).
        """
        for tier, plugin in self._tier_plugins.items():
            if _is_tracker(plugin) and hasattr(plugin, "init"):
                plugin.init(first_frame, first_bbox)

        self._last_state = TrackState(
            bbox=first_bbox, confidence=1.0, status="locked"
        )
        self._current_tier = self.start_tier

    def run(self, sequence: Any) -> Iterator[TelemetryEntry]:
        """Run the hybrid tracker over one ``Sequence`` yielding telemetry.

        Per-frame loop (PLAN §3 architecture diagram):

            1. Build ``FrameContext`` from current + previous frame.
            2. Ask every ``SwitchSignal`` for a ``SignalReport``.
            3. Scheduler emits a ``SchedulerDecision``.
            4. If tier changed, call ``on_tier_exit`` / ``on_tier_enter``.
            5. Active tier plugin updates/detects → ``TrackState``.
            6. Emit ``TelemetryEntry``.

        Resets all plugins and RNGs to enforce determinism.
        """
        # Seed RNGs for determinism.
        np.random.seed(self.seed)
        random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
        except ImportError:
            pass

        # Reset plugin state for a fresh sequence.
        self.reset()

        frames = list(sequence.frames)
        gt_bboxes: list[BBox] = sequence.ground_truth

        if not frames:
            return

        # Pre-warm tracker models if a ModelWarmer is configured.
        # Called after construction (trackers are built) but before frame 0
        # so no latency spikes hit the tracked sequence.
        if self.warmer is not None and not getattr(self.warmer, "is_warmed", True):
            self.warmer.warmup(self._tier_plugins)

        # Init on frame 0.
        self.init(frames[0], gt_bboxes[0])

        prev_frame: np.ndarray = frames[0]

        # Frames 1..N-1 are tracked.
        for frame_idx, frame in enumerate(frames[1:], start=1):
            ctx = FrameContext(
                frame=frame,
                prev_frame=prev_frame,
                frame_idx=frame_idx,
                bbox=self._last_state.bbox if self._last_state else None,
            )

            # Phase 14: run scene classifier and upgrade to FrameContextV2.
            scene_result = None
            if self._scene_classifier is not None:
                classify_interval = getattr(
                    self._scene_classifier, "classify_interval", 5
                )
                if frame_idx % classify_interval == 0 or frame_idx == 1:
                    try:
                        scene_result = self._scene_classifier.classify(
                            ctx, self._last_state
                        )
                    except Exception:
                        scene_result = None
                else:
                    # Return cached result between classification frames.
                    scene_result = getattr(
                        self._scene_classifier, "_cached_result", None
                    )

                if scene_result is not None:
                    try:
                        ctx = FrameContextV2(
                            frame=frame,
                            prev_frame=prev_frame,
                            frame_idx=frame_idx,
                            bbox=self._last_state.bbox if self._last_state else None,
                            scene_classification=scene_result,
                        )
                    except Exception:
                        pass  # stay with plain FrameContext on construction failure

            # Step each signal.
            from uav_tracker.types import SignalReport
            signal_reports: dict[str, SignalReport] = {}
            for sig_name, sig in self.signals.items():
                signal_reports[sig_name] = sig.step(ctx, self._last_state)

            # Scheduler decides tier.
            decision: SchedulerDecision = self.scheduler.decide(
                signals=signal_reports,
                current_tier=self._current_tier,
                frame_idx=frame_idx,
            )

            new_tier = decision.tier

            # Tier-change hooks.
            if decision.switched and new_tier != self._current_tier:
                old_plugin = self._tier_plugins.get(self._current_tier)
                new_plugin = self._tier_plugins.get(new_tier)
                if old_plugin is not None and _is_tracker(old_plugin):
                    on_exit = getattr(old_plugin, "on_tier_exit", None)
                    if callable(on_exit):
                        on_exit(ctx)
                if new_plugin is not None and _is_tracker(new_plugin):
                    on_enter = getattr(new_plugin, "on_tier_enter", None)
                    if callable(on_enter):
                        on_enter(ctx)
                self._current_tier = new_tier

            # Active plugin update.
            active_plugin = self._tier_plugins.get(self._current_tier)
            if active_plugin is None:
                # Fallback: use tier-0 tracker.
                active_plugin = self._tier_plugins[0]
                self._current_tier = 0

            # Phase 14: motion predictor — compute prediction before tracker update.
            motion_pred = None
            if self._motion_predictor is not None and len(self.trajectory) >= 2:
                try:
                    motion_pred = self._motion_predictor.predict_next(
                        history=self.trajectory[-10:],  # last 10 bboxes
                        timestamps=list(range(len(self.trajectory[-10:])))
                    )
                except Exception:
                    motion_pred = None

            t0 = time.perf_counter()

            if _is_detector(active_plugin) and not _is_tracker(active_plugin):
                # Detector tier — run recovery path.
                last_bbox = self._last_state.bbox if self._last_state else None
                recovered_bbox = self._recover_with_detector(frame, active_plugin, last_bbox)

                if recovered_bbox is not None:
                    # Re-init all lower-tier trackers on the recovered bbox.
                    self._reinit_lower_trackers(frame, recovered_bbox, self._current_tier)
                    self.recoveries += 1
                    state = TrackState(
                        bbox=recovered_bbox, confidence=1.0, status="locked"
                    )
                    # Step down to the tier below (tracker tier).
                    self._current_tier = max(0, self._current_tier - 1)
                else:
                    # Detector returned nothing — hold last bbox, mark unreliable.
                    bbox_to_use = last_bbox if last_bbox is not None else BBox(0, 0, 0, 0)
                    state = TrackState(
                        bbox=bbox_to_use, confidence=0.0, status="lost"
                    )
            else:
                state = active_plugin.update(frame)

            t1 = time.perf_counter()

            # Phase 14: appearance memory store (when not lost).
            if self._appearance_memory is not None and state.status != "lost":
                try:
                    self._appearance_memory.store(ctx, state)
                except Exception:
                    pass

            # Phase 14: motion predictor online update with actual bbox.
            if self._motion_predictor is not None:
                try:
                    self._motion_predictor.update(state.bbox)
                except Exception:
                    pass

            self._last_state = state
            self.trajectory.append(state.bbox)
            self.tier_sequence.append(self._current_tier)
            self.time_in_tier[self._current_tier] = (
                self.time_in_tier.get(self._current_tier, 0) + 1
            )
            self._update_times.append(t1 - t0)

            prev_frame = frame

            # Build aux dict; include scene_class if classified this frame.
            aux: dict[str, Any] = {"reliable": state.status != "lost"}
            if scene_result is not None:
                aux["scene_class"] = int(scene_result.scene_class)
                aux["scene_confidence"] = float(scene_result.confidence)
            if motion_pred is not None:
                aux["motion_pred_bbox"] = (
                    motion_pred.x, motion_pred.y, motion_pred.w, motion_pred.h
                )

            yield TelemetryEntry(
                frame_idx=frame_idx,
                bbox=state.bbox,
                confidence=state.confidence,
                tier=self._current_tier,
                switched=decision.switched,
                signals={k: v.value for k, v in signal_reports.items()},
                timings_ms={"tracker_ms": (t1 - t0) * 1000.0},
                aux=aux,
            )

    def _recover_with_detector(
        self,
        frame: np.ndarray,
        detector: Any,
        last_bbox: BBox | None,
    ) -> BBox | None:
        """Run detector on ``frame`` and return best-IoU detection vs ``last_bbox``.

        Returns ``None`` if no detections or if best IoU is zero (no overlap).
        """
        try:
            detections: list[Detection] = detector.detect(frame, hint=last_bbox)
        except Exception:
            return None

        if not detections:
            return None

        if last_bbox is None:
            # No prior bbox — return highest-confidence detection.
            best = max(detections, key=lambda d: d.score)
            return best.bbox

        # Find detection with best IoU vs last_bbox.
        best_det = max(detections, key=lambda d: _iou(d.bbox, last_bbox))
        return best_det.bbox

    def _reinit_lower_trackers(
        self, frame: np.ndarray, bbox: BBox, top_tier: int
    ) -> None:
        """Re-initialise all tracker-typed plugins below ``top_tier``."""
        for tier, plugin in self._tier_plugins.items():
            if tier < top_tier and _is_tracker(plugin) and hasattr(plugin, "init"):
                try:
                    plugin.init(frame, bbox)
                except Exception:
                    pass

    def reset(self) -> None:
        """Reset all plugins to construction state. Used between sequences."""
        for plugin in self._tier_plugins.values():
            reset_fn = getattr(plugin, "reset", None)
            if callable(reset_fn):
                reset_fn()
        for s in self.signals.values():
            s.reset()
        self.scheduler.reset()

        # Reset ML modules if present.
        for ml_module in (
            self._scene_classifier,
            self._appearance_memory,
            self._motion_predictor,
        ):
            if ml_module is not None:
                reset_fn = getattr(ml_module, "reset", None)
                if callable(reset_fn):
                    reset_fn()

        self._last_state = None
        self._current_tier = self.start_tier
        self.trajectory = []
        self.tier_sequence = []
        self.time_in_tier = {t: 0 for t in self._tier_plugins}
        self._update_times = []
        self.recoveries = 0
