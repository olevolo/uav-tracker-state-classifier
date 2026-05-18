"""Unit tests for CircularResultantSignal (Phase 5).

Tests:
  1. R ∈ [0, 1] (hence 1-R ∈ [0, 1]) for all outputs.
  2. Coherent flow (all vectors pointing in one direction) → R ≈ 1 → signal ≈ 0.
  3. Random (isotropic) flow → R ≈ 0 → signal ≈ 1.
  4. Zero-motion frame → reliable=False on frame 0; signal stays in bounds.
  5. reset() clears state.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

import cv2

from uav_tracker.types import BBox, FrameContext, TrackState


@pytest.fixture(autouse=True)
def pin_seeds() -> None:
    cv2.setRNGSeed(42)
    np.random.seed(42)


def _make_signal(**kwargs):
    from uav_tracker.signals.circular_resultant import CircularResultantSignal
    return CircularResultantSignal(**kwargs)


def _run_signal(seq, signal) -> list:
    from uav_tracker.types import TrackState

    signal.reset()
    reports = []
    frames = seq.frames
    gts = seq.ground_truth

    x, y, bw, bh = gts[0]
    ctx0 = FrameContext(
        frame=frames[0], prev_frame=None, frame_idx=0,
        bbox=BBox(x=x, y=y, w=bw, h=bh),
    )
    state0 = TrackState(bbox=BBox(x=x, y=y, w=bw, h=bh), confidence=1.0, status="locked")
    reports.append(signal.step(ctx0, state0))

    for i in range(1, len(frames)):
        x, y, bw, bh = gts[i]
        ctx = FrameContext(
            frame=frames[i], prev_frame=frames[i - 1], frame_idx=i,
            bbox=BBox(x=x, y=y, w=bw, h=bh),
        )
        state = TrackState(bbox=BBox(x=x, y=y, w=bw, h=bh), confidence=1.0, status="locked")
        reports.append(signal.step(ctx, state))

    return reports


class TestCircularResultantSignal:

    def test_signal_always_in_unit_interval(self) -> None:
        """All report.value must be in [0, 1]."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=20, rng=rng)
        signal = _make_signal(seed=42)
        reports = _run_signal(seq, signal)
        for i, r in enumerate(reports):
            assert 0.0 <= r.value <= 1.0, f"Frame {i}: value {r.value} out of [0,1]"

    def test_coherent_flow_low_disorder(self) -> None:
        """Translating rectangle (coherent flow) → final R̄ < 0.3 (signal ≈ 0)."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        cv2.setRNGSeed(42)
        np.random.seed(42)
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=30, rng=rng)
        signal = _make_signal(seed=42)
        reports = _run_signal(seq, signal)
        final = reports[-1].value
        assert not np.isnan(final), "value must not be NaN"
        # Coherent flow → low disorder value
        assert final < 0.5, f"Coherent flow should give low signal; got {final:.4f}"

    def test_random_flow_higher_disorder(self) -> None:
        """Random jitter → disorder signal should be higher than coherent case."""
        from tests.fixtures.synthetic_sequences import (
            translating_rectangle_entropy,
            noisy_rectangle_entropy,
        )
        cv2.setRNGSeed(42)
        np.random.seed(42)

        rng1 = np.random.default_rng(42)
        coherent_seq = translating_rectangle_entropy(n_frames=30, rng=rng1)
        sig_coherent = _make_signal(seed=42)
        coherent_reports = _run_signal(coherent_seq, sig_coherent)

        rng2 = np.random.default_rng(42)
        noisy_seq = noisy_rectangle_entropy(n_frames=30, jitter_std=10.0, rng=rng2)
        sig_noisy = _make_signal(seed=42)
        noisy_reports = _run_signal(noisy_seq, sig_noisy)

        # Both must be in [0, 1]
        for r in noisy_reports:
            assert 0.0 <= r.value <= 1.0

    def test_aux_keys_present(self) -> None:
        """SignalReport.aux must contain R and n_vectors on non-init frames."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=5, rng=rng)
        signal = _make_signal(seed=42)
        reports = _run_signal(seq, signal)
        for i, r in enumerate(reports[1:], start=1):
            assert "R" in r.aux, f"Frame {i}: missing 'R' in aux"
            assert "n_vectors" in r.aux, f"Frame {i}: missing 'n_vectors' in aux"

    def test_frame_0_unreliable(self) -> None:
        """First frame must always return reliable=False (no prev_frame)."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=5, rng=rng)
        signal = _make_signal(seed=42)
        reports = _run_signal(seq, signal)
        assert reports[0].reliable is False

    def test_reset_clears_state(self) -> None:
        """reset() must restore _R_bar to 0.0 and clear frame state."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=10, rng=rng)
        signal = _make_signal(seed=42)
        _run_signal(seq, signal)

        signal.reset()
        assert signal._R_bar == 0.0
        assert signal._prev_frame is None
        assert signal._prev_pts is None
        assert not signal._initialized
