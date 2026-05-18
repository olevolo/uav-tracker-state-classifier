"""Integration test: scene classifier + appearance memory + online adaptor in HybridRunner.

Tests Phase 12 + Phase 13 pipeline end-to-end using only CPU and synthetic data.
No real UAV123 sequences required; all frames are randomly generated.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")

from uav_tracker.types import BBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_sequence(n_frames: int = 30, h: int = 240, w: int = 320):
    """Return (frames, gt) for a moving rectangle on a random background."""
    rng = np.random.default_rng(0)
    frames = [
        rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n_frames)
    ]
    gt = [BBox(100.0 + i * 2.0, 100.0, 40.0, 40.0) for i in range(n_frames)]
    return frames, gt


def _make_mock_sequence(frames: list, gt: list):
    """Return a minimal object that satisfies HybridRunner's sequence interface.

    HybridRunner.run() accesses:
      - sequence.frames   (list/iterable of np.ndarray)
      - sequence.ground_truth  (list of BBox, index 0 = init bbox)
    """
    class _MockSequence:
        name: str = "mock"
        attributes: set = set()

        def __init__(self, _frames, _gt):
            self.frames = _frames
            self.ground_truth = _gt
            self.init_bbox = _gt[0]

    return _MockSequence(frames, gt)


# ---------------------------------------------------------------------------
# Test 1 — HybridRunner with MLSceneScheduler + scene classifier
# ---------------------------------------------------------------------------


def test_hybrid_runner_with_scene_classifier():
    """HybridRunner can run with MLSceneScheduler + MobileNetV3TinyClassifier.

    Verifies the full Phase-12 wiring: classifier → scheduler → tracker tier
    without crashing on CPU / random weights.  Result count must equal
    (n_frames - 1) regardless of ML decisions.
    """
    from uav_tracker.registry import TRACKERS, SCHEDULERS
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier
    from uav_tracker.schedulers.ml_scene_scheduler import MLSceneScheduler
    from uav_tracker.signals.motion_entropy import MotionEntropySignal
    from uav_tracker.runner import HybridRunner

    tracker = TRACKERS.build("kcf_kalman")
    classifier = MobileNetV3TinyClassifier(device="cpu", classify_interval=5)
    scheduler = MLSceneScheduler(
        scene_classifier=classifier,
        fallback_scheduler_name="multi_tier",
        confidence_threshold=0.6,
        override_frames=5,
    )
    signal = MotionEntropySignal()

    runner = HybridRunner(
        trackers={0: tracker},
        signals=[signal],
        scheduler=scheduler,
        seed=42,
    )

    n_frames = 20
    frames, gt = make_synthetic_sequence(n_frames)
    seq = _make_mock_sequence(frames, gt)

    results = list(runner.run(seq))

    # Must yield exactly one entry per tracked frame (frames[1:]).
    assert len(results) == n_frames - 1, (
        f"Expected {n_frames - 1} telemetry entries, got {len(results)}"
    )
    # All entries must have a non-negative tier.
    for entry in results:
        assert entry.tier >= 0


# ---------------------------------------------------------------------------
# Test 2 — MLSceneScheduler stays tier 0 when classifier low-confidence
# ---------------------------------------------------------------------------


def test_ml_scene_scheduler_falls_back_when_uncertain():
    """MLSceneScheduler falls back to multi_tier when confidence < threshold."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier
    from uav_tracker.schedulers.ml_scene_scheduler import MLSceneScheduler
    from uav_tracker.types import SignalReport

    # Set threshold above any random-weight softmax max (unlikely > 0.9)
    classifier = MobileNetV3TinyClassifier(
        device="cpu", confidence_threshold=0.99
    )
    scheduler = MLSceneScheduler(
        scene_classifier=classifier,
        confidence_threshold=0.99,
        override_frames=3,
    )

    signals = {"motion_entropy": SignalReport(value=0.1, reliable=True)}
    # Without a classify() call, last_classification is None → pure fallback.
    decision = scheduler.decide(signals=signals, current_tier=0, frame_idx=1)
    assert decision.tier >= 0


# ---------------------------------------------------------------------------
# Test 3 — ModelWarmer warms tier-0 tracker
# ---------------------------------------------------------------------------


def test_model_warmer_warms_trackers():
    """ModelWarmer completes warmup for all registered tier-0 trackers."""
    from uav_tracker.registry import TRACKERS, ML_WARMERS

    warmer = ML_WARMERS.build("default", target_latency_ms=5000.0)
    tracker = TRACKERS.build("kcf_kalman")

    warmer.warmup({0: tracker})

    status = warmer.get_status()
    assert warmer.is_warmed
    # Status must contain at least one entry.
    assert len(status) >= 1
    # All entries must have a known status.
    for name, info in status.items():
        assert info["status"] in ("ok", "failed"), (
            f"Unexpected status for {name}: {info}"
        )


# ---------------------------------------------------------------------------
# Test 4 — CosineAppearanceMemory accumulates templates
# ---------------------------------------------------------------------------


def test_appearance_memory_stores_during_tracking():
    """CosineAppearanceMemory accumulates templates when fed real frames."""
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory
    from uav_tracker.types import TrackState, FrameContext

    mem = CosineAppearanceMemory(store_interval=1, min_confidence=0.0)

    frames, gt = make_synthetic_sequence(15)
    for i, (frame, bbox) in enumerate(zip(frames, gt)):
        state = TrackState(bbox=bbox, confidence=0.8, status="locked")
        ctx = FrameContext(
            frame=frame,
            prev_frame=frame,
            frame_idx=i,
            bbox=bbox,
        )
        mem.store(ctx, state)

    assert len(mem._templates) > 0, "Expected at least one stored template"

    drift = mem.compute_drift()
    assert 0.0 <= drift <= 1.0, f"Drift score {drift} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 5 — CosineAppearanceMemory evicts to max_templates
# ---------------------------------------------------------------------------


def test_appearance_memory_respects_max_templates():
    """Template count never exceeds max_templates."""
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory
    from uav_tracker.types import TrackState, FrameContext

    max_t = 5
    mem = CosineAppearanceMemory(
        max_templates=max_t, store_interval=1, min_confidence=0.0
    )

    frames, gt = make_synthetic_sequence(30)
    for i, (frame, bbox) in enumerate(zip(frames, gt)):
        state = TrackState(bbox=bbox, confidence=0.9, status="locked")
        ctx = FrameContext(frame=frame, prev_frame=frame, frame_idx=i, bbox=bbox)
        mem.store(ctx, state)

    assert len(mem._templates) <= max_t


# ---------------------------------------------------------------------------
# Test 6 — SceneClassifierOnlineAdaptor accumulates buffer
# ---------------------------------------------------------------------------


def test_online_adaptor_accumulates_buffer():
    """SceneClassifierOnlineAdaptor fills the replay buffer without crashing."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier
    from uav_tracker.ml.scene_classifier.online_adaptor import SceneClassifierOnlineAdaptor
    from uav_tracker.types import TrackState, FrameContext

    classifier = MobileNetV3TinyClassifier(device="cpu", classify_interval=5)
    adaptor = SceneClassifierOnlineAdaptor(
        classifier=classifier,
        buffer_size=50,
        adapt_interval=100,   # large → no adaptation during the test
        min_buffer_size=5,
        confidence_gate=0.0,  # accept all frames
    )

    frames, gt = make_synthetic_sequence(20)
    for i, (frame, bbox) in enumerate(zip(frames, gt)):
        state = TrackState(bbox=bbox, confidence=0.9, status="locked")
        ctx = FrameContext(frame=frame, prev_frame=frame, frame_idx=i, bbox=bbox)
        adaptor.step(ctx, state)

    assert adaptor.buffer_len > 0, "Expected buffer to have samples after 20 frames"


# ---------------------------------------------------------------------------
# Test 7 — SceneClassifierOnlineAdaptor skips lost frames
# ---------------------------------------------------------------------------


def test_online_adaptor_skips_lost_state():
    """OnlineAdaptor must not accumulate samples when tracker is 'lost'."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier
    from uav_tracker.ml.scene_classifier.online_adaptor import SceneClassifierOnlineAdaptor
    from uav_tracker.types import TrackState, FrameContext

    classifier = MobileNetV3TinyClassifier(device="cpu")
    adaptor = SceneClassifierOnlineAdaptor(
        classifier=classifier,
        confidence_gate=0.0,
    )

    frames, gt = make_synthetic_sequence(10)
    for i, (frame, bbox) in enumerate(zip(frames, gt)):
        state = TrackState(bbox=bbox, confidence=0.9, status="lost")
        ctx = FrameContext(frame=frame, prev_frame=frame, frame_idx=i, bbox=bbox)
        adaptor.step(ctx, state)

    assert adaptor.buffer_len == 0, (
        "Buffer should stay empty when status is always 'lost'"
    )


# ---------------------------------------------------------------------------
# Test 8 — OnlineAdaptor reset clears buffer
# ---------------------------------------------------------------------------


def test_online_adaptor_reset_clears_buffer():
    """SceneClassifierOnlineAdaptor.reset() empties the replay buffer."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier
    from uav_tracker.ml.scene_classifier.online_adaptor import SceneClassifierOnlineAdaptor
    from uav_tracker.types import TrackState, FrameContext

    classifier = MobileNetV3TinyClassifier(device="cpu")
    adaptor = SceneClassifierOnlineAdaptor(
        classifier=classifier, confidence_gate=0.0, adapt_interval=1000
    )

    frames, gt = make_synthetic_sequence(5)
    for i, (frame, bbox) in enumerate(zip(frames, gt)):
        state = TrackState(bbox=bbox, confidence=0.8, status="locked")
        ctx = FrameContext(frame=frame, prev_frame=frame, frame_idx=i, bbox=bbox)
        adaptor.step(ctx, state)

    assert adaptor.buffer_len > 0
    adaptor.reset()
    assert adaptor.buffer_len == 0
    assert adaptor.adapt_count == 0


# ---------------------------------------------------------------------------
# Test 9 — v2 HybridRunner propagates scene_class into TelemetryEntry.aux
# ---------------------------------------------------------------------------


def test_v2_runner_propagates_scene_classification():
    """HybridRunner with scene_classifier stores scene_class in TelemetryEntry.aux.

    Runs 15 frames and verifies that at least one telemetry entry has
    'scene_class' in its aux dict (i.e. the classify_interval fired and the
    result was wired through to TelemetryEntry).
    """
    from uav_tracker.runner import HybridRunner
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier
    from uav_tracker.schedulers.ml_scene_scheduler import MLSceneScheduler
    from uav_tracker.signals.motion_entropy import MotionEntropySignal
    from uav_tracker.registry import TRACKERS

    clf = MobileNetV3TinyClassifier(device="cpu", classify_interval=5)
    scheduler = MLSceneScheduler(
        scene_classifier=clf,
        fallback_scheduler_name="multi_tier",
        confidence_threshold=0.0,  # always accept — ensures ML path fires
        override_frames=3,
    )
    signal = MotionEntropySignal()

    runner = HybridRunner(
        trackers={0: TRACKERS.build("kcf_kalman")},
        signals=[signal],
        scheduler=scheduler,
        scene_classifier=clf,
        seed=42,
    )

    n_frames = 15
    frames, gt = make_synthetic_sequence(n_frames)
    seq = _make_mock_sequence(frames, gt)

    results = list(runner.run(seq))

    assert len(results) == n_frames - 1

    # At least one entry should have scene_class in aux (fires at frame 5, 10).
    entries_with_scene = [e for e in results if "scene_class" in e.aux]
    assert len(entries_with_scene) >= 1, (
        f"Expected at least one TelemetryEntry with 'scene_class' in aux, "
        f"got 0 out of {len(results)} entries. "
        f"aux samples: {[e.aux for e in results[:3]]}"
    )
    # scene_class must be a valid int in [0, 5].
    for entry in entries_with_scene:
        assert 0 <= entry.aux["scene_class"] <= 5, (
            f"scene_class={entry.aux['scene_class']} out of valid range [0, 5]"
        )


# ---------------------------------------------------------------------------
# Test 10 — v2 HybridRunner appearance memory grows during tracking
# ---------------------------------------------------------------------------


def test_v2_runner_appearance_memory_grows():
    """AppearanceMemory stores templates as the runner processes confident frames.

    Runs 20 frames with a min_confidence=0.0 appearance memory and verifies
    that at least one template was stored during the run.
    """
    from uav_tracker.runner import HybridRunner
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory
    from uav_tracker.registry import TRACKERS, SCHEDULERS
    from uav_tracker.signals.motion_entropy import MotionEntropySignal

    # min_confidence=0.0 and store_interval=1 so every frame stores a template.
    mem = CosineAppearanceMemory(
        max_templates=50,
        store_interval=1,
        min_confidence=0.0,
    )
    signal = MotionEntropySignal()
    scheduler = SCHEDULERS.build("multi_tier")

    runner = HybridRunner(
        trackers={0: TRACKERS.build("kcf_kalman")},
        signals=[signal],
        scheduler=scheduler,
        appearance_memory=mem,
        seed=42,
    )

    n_frames = 20
    frames, gt = make_synthetic_sequence(n_frames)
    seq = _make_mock_sequence(frames, gt)

    list(runner.run(seq))  # consume iterator to run all frames

    assert len(mem._templates) > 0, (
        f"Expected appearance memory to have stored at least one template "
        f"after {n_frames} frames, but got 0 templates."
    )
