"""Unit tests for Phase 5 proactive recovery hint.

Tests:
  - update_recovery_hint() fires when p_fc > 0.45 AND apce_ratio5 < 0.85
  - update_recovery_hint() does NOT fire when only one condition holds
  - consume_recovery_hint() returns correct values and clears the hint
  - Hint expires after max_age frames

Note: Phase 8 TSA→SALT-RD adapter tests (saltrd_state_to_tsa_state) have been
moved to archive/tsa_removed/test_saltr_phase5_phase8_tsa_tests.py because the
TSA module (uav_tracker.ml.tsa) was removed in the SALT-RD Phase 4 cleanup.
"""
from __future__ import annotations

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeBBox:
    def __init__(self, x=100.0, y=80.0, w=30.0, h=20.0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class _FakeTrackState:
    def __init__(self, apce=200.0, psr=3000.0, entropy=1.5, bbox=None):
        self.apce = apce
        self.psr = psr
        self.response_entropy = entropy
        self.score_map_stats = {}
        self.bbox = bbox or _FakeBBox()
        self.confidence = 0.9


def _make_advisor_with_state(p_fc: float, apce_r5: float):
    """Return an SALTRDAdvisor instance with _last_p_fc and _last_apce_ratio5 set.

    Avoids loading a real model checkpoint by patching internal state directly.
    """
    import types

    class _MockModel:
        def predict_single(self, window, device="cpu"):
            return {"false_confirmed": p_fc}

    class _PatchedAdvisor:
        """Minimal stand-in that exercises only the recovery hint logic."""

        def __init__(self):
            from salt_r.advisor import _OnlinePositiveRAM
            self._last_p_fc = p_fc
            self._last_apce_ratio5 = apce_r5
            self._current_frame = 0

            # Proactive early recovery fields (mirrors SALTRDAdvisor.__init__)
            self._recovery_hint_bbox = None
            self._recovery_hint_active = False
            self._recovery_hint_frame = -1
            self._recovery_hint_max_age = 30
            self.early_recovery_pfc_threshold = 0.45
            self.early_recovery_apce_ratio_threshold = 0.85

        # Copy methods from real class by importing them bound
        def update_recovery_hint(self, bbox, apce: float) -> bool:
            from salt_r.advisor import SALTRDAdvisor
            return SALTRDAdvisor.update_recovery_hint(self, bbox, apce)

        def consume_recovery_hint(self):
            from salt_r.advisor import SALTRDAdvisor
            return SALTRDAdvisor.consume_recovery_hint(self)

    return _PatchedAdvisor()


# ---------------------------------------------------------------------------
# Test 1: hint fires when both conditions hold
# ---------------------------------------------------------------------------

def test_update_recovery_hint_fires_when_at_risk():
    """Hint activates when p_fc >= 0.45 AND apce_ratio5 < 0.85."""
    advisor = _make_advisor_with_state(p_fc=0.50, apce_r5=0.70)
    bbox = _FakeBBox(x=100.0, y=80.0, w=30.0, h=20.0)

    hint_active = advisor.update_recovery_hint(bbox, apce=80.0)

    assert hint_active is True, "Hint should be active when at risk"
    assert advisor._recovery_hint_active is True
    assert advisor._recovery_hint_bbox is not None
    cx, cy, bw, bh = advisor._recovery_hint_bbox
    assert abs(cx - 115.0) < 1e-4, f"Expected cx=115.0, got {cx}"
    assert abs(cy - 90.0) < 1e-4, f"Expected cy=90.0, got {cy}"
    assert abs(bw - 30.0) < 1e-4
    assert abs(bh - 20.0) < 1e-4


# ---------------------------------------------------------------------------
# Test 2: hint does NOT fire when p_fc is too low
# ---------------------------------------------------------------------------

def test_update_recovery_hint_no_fire_low_pfc():
    """Hint does not activate when p_fc < threshold even with falling APCE."""
    advisor = _make_advisor_with_state(p_fc=0.30, apce_r5=0.60)
    bbox = _FakeBBox()

    hint_active = advisor.update_recovery_hint(bbox, apce=80.0)

    assert hint_active is False, "Hint should NOT activate when p_fc < 0.45"
    assert advisor._recovery_hint_active is False
    assert advisor._recovery_hint_bbox is None


# ---------------------------------------------------------------------------
# Test 3: hint does NOT fire when apce_ratio5 is high (APCE stable)
# ---------------------------------------------------------------------------

def test_update_recovery_hint_no_fire_stable_apce():
    """Hint does not activate when apce_ratio5 >= threshold even with high p_fc."""
    advisor = _make_advisor_with_state(p_fc=0.60, apce_r5=0.90)
    bbox = _FakeBBox()

    hint_active = advisor.update_recovery_hint(bbox, apce=200.0)

    assert hint_active is False, "Hint should NOT activate when APCE is stable (ratio >= 0.85)"
    assert advisor._recovery_hint_active is False


# ---------------------------------------------------------------------------
# Test 4: consume_recovery_hint returns and clears hint
# ---------------------------------------------------------------------------

def test_consume_recovery_hint_returns_and_clears():
    """consume_recovery_hint() returns (cx,cy,w,h) and then clears the hint."""
    advisor = _make_advisor_with_state(p_fc=0.50, apce_r5=0.70)
    bbox = _FakeBBox(x=50.0, y=40.0, w=20.0, h=15.0)
    advisor.update_recovery_hint(bbox, apce=60.0)

    # First consume: returns hint
    hint = advisor.consume_recovery_hint()
    assert hint is not None, "Expected a hint to be returned"
    cx, cy, bw, bh = hint
    assert abs(cx - 60.0) < 1e-4, f"Expected cx=60.0, got {cx}"
    assert abs(cy - 47.5) < 1e-4, f"Expected cy=47.5, got {cy}"

    # Hint is cleared after consume
    assert advisor._recovery_hint_active is False
    assert advisor._recovery_hint_bbox is None

    # Second consume: returns None
    hint2 = advisor.consume_recovery_hint()
    assert hint2 is None, "Hint should be None after consumption"


# ---------------------------------------------------------------------------
# Test 5: consume_recovery_hint returns None when no hint
# ---------------------------------------------------------------------------

def test_consume_recovery_hint_returns_none_when_not_active():
    """consume_recovery_hint() returns None when no hint is active."""
    advisor = _make_advisor_with_state(p_fc=0.10, apce_r5=1.0)

    hint = advisor.consume_recovery_hint()
    assert hint is None, "Expected None when no hint was set"


# ---------------------------------------------------------------------------
# Test 6: hint expires after max_age frames
# ---------------------------------------------------------------------------

def test_recovery_hint_expires_after_max_age():
    """Hint expires when current_frame - hint_frame > max_age."""
    advisor = _make_advisor_with_state(p_fc=0.50, apce_r5=0.70)
    bbox = _FakeBBox()

    # Activate hint at frame 0
    advisor._current_frame = 0
    advisor.update_recovery_hint(bbox, apce=60.0)
    assert advisor._recovery_hint_active is True

    # Advance to just before expiry (max_age=30): frame 30 → age=30 → STILL active
    advisor._current_frame = 30
    still_active = advisor.update_recovery_hint(bbox, apce=60.0)
    # Age == max_age (30) → NOT yet expired (age > max_age triggers expiry)
    assert still_active is True, "Hint at exactly max_age should still be active"

    # Advance past expiry: frame 31 → age=31 > max_age=30 → expired
    advisor._current_frame = 31
    expired = advisor.update_recovery_hint(bbox, apce=60.0)
    assert expired is False, "Hint should expire after max_age frames"
    assert advisor._recovery_hint_active is False

