"""MLSceneScheduler — scene-class-driven scheduler bridging ML classifier → tier decision.

When the SceneClassifier confidence >= confidence_threshold AND the scene
classification has been stable for ``override_frames`` consecutive frames,
the scheduler overrides the entropy-based fallback scheduler with the
ML-derived tier.  Falls back to entropy-based scheduling when ML confidence
is low or the scene is unstable.

Special-case behaviours:
  - RISK_LOSS: uses tier from ``scene_to_tier`` mapping (default 2) and
    injects ``prearm_tier3=True`` into ``SchedulerDecision.aux`` for the next
    ``risk_prearm_frames`` frames to signal the runner to pre-arm tier-3
    (detector).
  - uncertain (confidence < threshold): hold current tier, no transition,
    don't advance fallback scheduler counters.

Registration key: ``"ml_scene"``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from uav_tracker.registry import SCHEDULERS
from uav_tracker.types import SceneClass, SchedulerDecision, SignalReport


class _NullSceneClassifier:
    """No-op scene classifier for testing and fallback. Always returns CLEAR with confidence 0.0."""

    name = "_null"
    n_classes = 6
    classify_interval = 1

    def classify(self, ctx: Any, state: Any) -> Any:
        from uav_tracker.types import SceneClassification, SceneClass as _SC
        return SceneClassification(
            scene_class=_SC.CLEAR,
            probabilities=np.ones(6) / 6,
            confidence=0.0,
            frame_idx=getattr(ctx, "frame_idx", 0),
        )

    def update_online(self, ctx: Any, state: Any, feedback: Any) -> None:
        pass

    def reset(self) -> None:
        pass

logger = logging.getLogger(__name__)

# Default scene → tier mapping (see PLAN §12 and ADR-0009).
_DEFAULT_SCENE_TO_TIER: dict[int, int] = {
    SceneClass.CLEAR: 0,
    SceneClass.MODERATE: 1,
    SceneClass.CHALLENGING: 2,
    SceneClass.RISK_LOSS: 2,
    SceneClass.RECOVERY: 3,
    SceneClass.LOW_RES: 1,
}


@SCHEDULERS.register("ml_scene")
class MLSceneScheduler:
    """Scene-class-driven scheduler bridging ML classifier → tier decision.

    When the SceneClassifier confidence >= ``confidence_threshold`` AND the
    same scene class has been returned for at least ``override_frames``
    consecutive frames, this scheduler overrides the fallback scheduler's
    tier decision with the ML-derived mapping.  Otherwise it delegates to
    ``fallback_scheduler`` (usually ``multi_tier`` or ``hysteresis_binary``).

    Parameters
    ----------
    scene_classifier:
        Any object with a ``classify(frame_context)`` method that returns a
        ``SceneClassification``-duck with ``.scene_class`` (``int`` or
        ``SceneClass``) and ``.confidence`` (``float``).  For unit tests a
        mock that always returns CLEAR with confidence 1.0 is acceptable.
    difficulty_predictor:
        Optional predictor; reserved for future lookahead logic.  Currently
        used only for logging/aux telemetry.
    fallback_scheduler_name:
        Registry key of the fallback scheduler to build (default
        ``"multi_tier"``).  Built via ``SCHEDULERS.build``.
    confidence_threshold:
        Minimum classifier confidence for ML override to activate.
    scene_to_tier:
        Mapping from ``SceneClass`` int value to tier index.  Defaults to the
        PLAN §12 mapping.
    override_frames:
        Number of consecutive frames the same scene class must be reported
        (with confidence >= threshold) before the ML tier is committed.
        Prevents rapid oscillation on classifier noise.
    risk_prearm_frames:
        After RISK_LOSS is detected, inject ``prearm_tier3=True`` into
        ``SchedulerDecision.aux`` for this many additional frames to allow
        the runner to pre-warm the detector tier.
    **fallback_kwargs:
        Extra keyword arguments forwarded to the fallback scheduler constructor.
    """

    name: str = "ml_scene"
    tiers: int = 4

    def __init__(
        self,
        scene_classifier: Any = None,  # defaults to _NullSceneClassifier
        difficulty_predictor: Any | None = None,
        fallback_scheduler_name: str = "multi_tier",
        confidence_threshold: float = 0.6,
        scene_to_tier: dict[int, int] | None = None,
        override_frames: int = 15,
        risk_prearm_frames: int = 3,
        **fallback_kwargs: Any,
    ) -> None:
        if scene_classifier is None:
            scene_classifier = _NullSceneClassifier()
        self.scene_classifier = scene_classifier
        self.difficulty_predictor = difficulty_predictor
        self.confidence_threshold = float(confidence_threshold)
        self.override_frames = int(override_frames)
        self.risk_prearm_frames = int(risk_prearm_frames)

        # Build scene → tier mapping (normalise keys to int).
        if scene_to_tier is not None:
            self.scene_to_tier: dict[int, int] = {
                int(k): int(v) for k, v in scene_to_tier.items()
            }
        else:
            self.scene_to_tier = {int(k): v for k, v in _DEFAULT_SCENE_TO_TIER.items()}

        # Build fallback scheduler lazily to avoid circular imports at
        # registration time.
        self._fallback_scheduler_name = fallback_scheduler_name
        self._fallback_kwargs = fallback_kwargs
        self._fallback: Any = None  # built on first decide() call

        # Mutable state.
        self._stable_scene: int | None = None   # scene_class int currently in stable window
        self._stable_count: int = 0             # consecutive high-confidence frames for same class
        self._ml_active: bool = False            # True when ML override is committed
        self._ml_tier: int = 0                  # tier currently overriding
        self._prearm_countdown: int = 0          # frames remaining for RISK_LOSS pre-arm

    # ------------------------------------------------------------------
    # Scheduler Protocol

    def decide(
        self,
        signals: dict[str, SignalReport],
        current_tier: int,
        frame_idx: int,
    ) -> SchedulerDecision:
        """Return the tier decision for this frame.

        Decision flow:
        1. Ask scene_classifier for latest classification (if available as a
           ``last_classification`` attribute — some classifiers pre-compute and
           store; others require a frame which we don't have here, so we fall
           back to their ``last_classification`` cached result).
        2. If confidence < threshold → uncertain; hold current tier (no
           fallback advance either).
        3. Else accumulate stability counter.  If counter >= override_frames →
           commit ML override tier.
        4. Handle RISK_LOSS pre-arm injection.
        5. If ML override not yet committed → delegate to fallback scheduler.
        """
        # Ensure fallback scheduler is built.
        if self._fallback is None:
            self._build_fallback()

        # Retrieve latest scene classification.
        classification = self._get_classification()

        # Handle unavailable classifier output.
        if classification is None:
            # No classification available yet — pure fallback.
            fallback_decision = self._fallback.decide(signals, current_tier, frame_idx)
            return self._maybe_inject_prearm(fallback_decision)

        scene_class_int = int(classification.scene_class)
        confidence = float(classification.confidence)

        # --- Uncertain: confidence below threshold → hold, no counter advance ---
        if confidence < self.confidence_threshold:
            self._stable_count = 0
            self._stable_scene = None
            self._ml_active = False
            logger.debug(
                "MLSceneScheduler frame %d: uncertain (conf=%.2f < %.2f) — hold tier %d",
                frame_idx, confidence, self.confidence_threshold, current_tier,
            )
            return SchedulerDecision(
                tier=current_tier,
                reason=(
                    f"ml_scene: uncertain (conf={confidence:.2f} < "
                    f"{self.confidence_threshold:.2f}) — holding tier {current_tier}"
                ),
                switched=False,
            )

        # --- Accumulate stability counter ---
        if scene_class_int == self._stable_scene:
            self._stable_count += 1
        else:
            # New scene class; reset stability window.
            self._stable_scene = scene_class_int
            self._stable_count = 1
            self._ml_active = False  # must re-confirm with new class

        # RISK_LOSS: pre-arm tier 3 regardless of stability window.
        if scene_class_int == int(SceneClass.RISK_LOSS) and self._prearm_countdown == 0:
            self._prearm_countdown = self.risk_prearm_frames
            logger.debug(
                "MLSceneScheduler frame %d: RISK_LOSS detected — arming prearm for %d frames",
                frame_idx, self.risk_prearm_frames,
            )

        # --- Check if stability window has been reached ---
        if self._stable_count >= self.override_frames:
            self._ml_active = True

        if self._ml_active:
            ml_tier = self.scene_to_tier.get(scene_class_int, 0)
            switched = ml_tier != current_tier
            scene_name = SceneClass(scene_class_int).name if scene_class_int in [
                s.value for s in SceneClass
            ] else str(scene_class_int)
            reason = (
                f"ml_scene: {scene_name} (conf={confidence:.2f}) "
                f"stable {self._stable_count}/{self.override_frames} frames → tier {ml_tier}"
            )
            logger.debug("MLSceneScheduler frame %d: %s", frame_idx, reason)
            decision = SchedulerDecision(tier=ml_tier, reason=reason, switched=switched)
            return self._maybe_inject_prearm(decision)

        # --- Stability window not yet reached → delegate to fallback ---
        fallback_decision = self._fallback.decide(signals, current_tier, frame_idx)
        reason = (
            f"ml_scene: fallback ({self._fallback_scheduler_name}) — "
            f"scene {scene_class_int} stable {self._stable_count}/{self.override_frames}"
        )
        # Override reason but preserve tier/switched from fallback.
        decision = SchedulerDecision(
            tier=fallback_decision.tier,
            reason=reason,
            switched=fallback_decision.switched,
        )
        return self._maybe_inject_prearm(decision)

    def reset(self) -> None:
        """Restore state to construction defaults. Idempotent."""
        self._stable_scene = None
        self._stable_count = 0
        self._ml_active = False
        self._ml_tier = 0
        self._prearm_countdown = 0
        if self._fallback is not None:
            self._fallback.reset()

    # ------------------------------------------------------------------
    # Private helpers

    def _build_fallback(self) -> None:
        """Lazily construct the fallback scheduler from the registry."""
        from uav_tracker.registry import SCHEDULERS as _SCHEDULERS
        self._fallback = _SCHEDULERS.build(
            self._fallback_scheduler_name, **self._fallback_kwargs
        )

    def _get_classification(self) -> Any | None:
        """Retrieve the most recent SceneClassification from the classifier.

        Some classifiers expose ``last_classification`` (result of the most
        recent ``classify`` call); others are purely functional.  We probe for
        ``last_classification`` first so the scheduler does not need a frame
        reference.
        """
        last = getattr(self.scene_classifier, "last_classification", None)
        if last is not None:
            return last

        # If the classifier is a simple callable mock (e.g. returns a
        # SceneClassification directly), try calling with no args.
        if callable(self.scene_classifier):
            try:
                return self.scene_classifier()
            except TypeError:
                return None

        return None

    def _maybe_inject_prearm(self, decision: SchedulerDecision) -> SchedulerDecision:
        """Tick the RISK_LOSS pre-arm countdown and inject aux flag if active."""
        if self._prearm_countdown > 0:
            self._prearm_countdown -= 1
            # SchedulerDecision is a non-frozen dataclass; attach the aux dict
            # as a plain attribute so downstream code can inspect it without
            # requiring a Protocol change.
            decision_with_aux = SchedulerDecision(
                tier=decision.tier,
                reason=decision.reason,
                switched=decision.switched,
            )
            decision_with_aux.aux = {"prearm_tier3": True}  # type: ignore[attr-defined]
            return decision_with_aux
        return decision


__all__ = ["MLSceneScheduler", "_NullSceneClassifier"]
