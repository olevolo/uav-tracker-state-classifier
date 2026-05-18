"""Unit tests for APCESignal (Phase 5, Option A).

APCESignal uses tracker confidence as a proxy (Option A per Architect decision:
cv2.TrackerKCF does not expose the raw correlation response map needed for
authentic APCE computation; authentic APCE is Phase 6+).

The tests verify that the proxy behavior matches TrackerConfidenceSignal:
  value = 1 - confidence
  reliable = True iff confidence is a valid float in [0, 1]
"""

from __future__ import annotations

import math

import pytest

from uav_tracker.types import BBox, FrameContext, SignalReport, TrackState


def _make_state(confidence: float) -> TrackState:
    return TrackState(bbox=BBox(0.0, 0.0, 10.0, 10.0), confidence=confidence, status="locked")


def _make_ctx() -> FrameContext:
    import numpy as np
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    return FrameContext(frame=frame, prev_frame=None, frame_idx=0, bbox=BBox(0, 0, 10, 10))


class TestAPCESignal:

    def test_mirrors_tracker_confidence_signal(self) -> None:
        """APCESignal must produce identical value to TrackerConfidenceSignal (Option A)."""
        from uav_tracker.signals.apce import APCESignal
        from uav_tracker.signals.tracker_confidence import TrackerConfidenceSignal

        apce = APCESignal()
        conf_sig = TrackerConfidenceSignal()
        ctx = _make_ctx()

        for conf in [0.0, 0.2, 0.5, 0.8, 1.0]:
            state = _make_state(conf)
            apce_r = apce.step(ctx, state)
            conf_r = conf_sig.step(ctx, state)
            assert abs(apce_r.value - conf_r.value) < 1e-9, (
                f"conf={conf}: APCE={apce_r.value}, Confidence={conf_r.value}"
            )

    def test_none_state_unreliable(self) -> None:
        """None state → value=0.0, reliable=False."""
        from uav_tracker.signals.apce import APCESignal
        sig = APCESignal()
        ctx = _make_ctx()
        r = sig.step(ctx, None)
        assert r.value == 0.0
        assert r.reliable is False

    def test_nan_confidence_unreliable(self) -> None:
        """NaN confidence → reliable=False."""
        from uav_tracker.signals.apce import APCESignal
        sig = APCESignal()
        ctx = _make_ctx()
        state = _make_state(float("nan"))
        r = sig.step(ctx, state)
        assert r.reliable is False

    def test_high_confidence_low_signal(self) -> None:
        """High confidence → low signal (tracker is confident → no escalation)."""
        from uav_tracker.signals.apce import APCESignal
        sig = APCESignal()
        ctx = _make_ctx()
        r = sig.step(ctx, _make_state(0.9))
        assert r.value < 0.5

    def test_low_confidence_high_signal(self) -> None:
        """Low confidence → high signal (tracker uncertain → may escalate)."""
        from uav_tracker.signals.apce import APCESignal
        sig = APCESignal()
        ctx = _make_ctx()
        r = sig.step(ctx, _make_state(0.1))
        assert r.value > 0.5

    def test_value_in_unit_interval(self) -> None:
        """value must always be in [0, 1]."""
        from uav_tracker.signals.apce import APCESignal
        sig = APCESignal()
        ctx = _make_ctx()
        for conf in [0.0, 0.5, 1.0]:
            r = sig.step(ctx, _make_state(conf))
            assert 0.0 <= r.value <= 1.0

    def test_reset_is_noop(self) -> None:
        """reset() must not raise (stateless signal)."""
        from uav_tracker.signals.apce import APCESignal
        sig = APCESignal()
        sig.reset()  # must not raise

    def test_aux_contains_apce_mode(self) -> None:
        """aux must contain apce_mode='option_a' for traceability."""
        from uav_tracker.signals.apce import APCESignal
        sig = APCESignal()
        ctx = _make_ctx()
        r = sig.step(ctx, _make_state(0.5))
        assert r.aux.get("apce_mode") == "option_a"
