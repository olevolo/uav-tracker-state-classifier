"""Unit tests for FlowDivergenceSignal (Phase 5).

Tests:
  1. Zero-motion frames → div ≈ 0 (signal stays near 0).
  2. Signal always in [0, 1].
  3. aux contains div_raw.
  4. Frame 0 returns reliable=False.
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
    from uav_tracker.signals.flow_divergence import FlowDivergenceSignal
    return FlowDivergenceSignal(**kwargs)


def _run_signal(seq, signal) -> list:
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


class TestFlowDivergenceSignal:

    def test_zero_flow_low_divergence(self) -> None:
        """Identical consecutive frames → zero residual flow → div_raw ≈ 0."""
        h, w = 120, 160
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        # Paint a textured rectangle so Shi-Tomasi finds corners.
        rng = np.random.default_rng(99)
        frame[40:80, 60:100] = np.clip(
            160 + rng.normal(0, 20, (40, 40, 3)), 0, 255
        ).astype(np.uint8)

        bbox = BBox(x=60.0, y=40.0, w=40.0, h=40.0)
        state = TrackState(bbox=bbox, confidence=1.0, status="locked")
        signal = _make_signal(seed=42)
        signal.reset()

        ctx0 = FrameContext(frame=frame, prev_frame=None, frame_idx=0, bbox=bbox)
        signal.step(ctx0, state)

        ctx1 = FrameContext(frame=frame, prev_frame=frame, frame_idx=1, bbox=bbox)
        report = signal.step(ctx1, state)

        div_raw = report.aux.get("div_raw", None)
        assert div_raw is not None, "aux must contain div_raw"
        assert abs(div_raw) < 0.5, (
            f"Zero motion should yield near-zero divergence; got div_raw={div_raw:.4f}"
        )

    def test_signal_always_in_unit_interval(self) -> None:
        """All report.value must be in [0, 1]."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=20, rng=rng)
        signal = _make_signal(seed=42)
        reports = _run_signal(seq, signal)
        for i, r in enumerate(reports):
            assert 0.0 <= r.value <= 1.0, f"Frame {i}: value {r.value} out of [0,1]"

    def test_aux_div_raw_present(self) -> None:
        """aux must contain div_raw on non-init frames."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=5, rng=rng)
        signal = _make_signal(seed=42)
        reports = _run_signal(seq, signal)
        for i, r in enumerate(reports[1:], start=1):
            assert "div_raw" in r.aux, f"Frame {i}: missing div_raw in aux"

    def test_frame_0_unreliable(self) -> None:
        """Frame 0 must always return reliable=False."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=5, rng=rng)
        signal = _make_signal(seed=42)
        reports = _run_signal(seq, signal)
        assert reports[0].reliable is False

    def test_reset_clears_state(self) -> None:
        """reset() must restore _div_bar to 0.0 and clear frame state."""
        from tests.fixtures.synthetic_sequences import translating_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = translating_rectangle_entropy(n_frames=10, rng=rng)
        signal = _make_signal(seed=42)
        _run_signal(seq, signal)

        signal.reset()
        assert signal._div_bar == 0.0
        assert signal._prev_frame is None
        assert signal._prev_pts is None
        assert not signal._initialized

    def test_no_nan_values(self) -> None:
        """No report.value should be NaN."""
        from tests.fixtures.synthetic_sequences import noisy_rectangle_entropy
        rng = np.random.default_rng(42)
        seq = noisy_rectangle_entropy(n_frames=20, rng=rng)
        signal = _make_signal(seed=42)
        reports = _run_signal(seq, signal)
        for i, r in enumerate(reports):
            assert not np.isnan(r.value), f"Frame {i}: NaN signal value"
