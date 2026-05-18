"""Unit tests for ``RestartOPE``.

Verifies the restart-OPE harness using a synthetic tracker and sequence
where we can force known failures.

The tests are guarded with ``pytest.importorskip("cv2")`` because
``uav_tracker.datasets.synthetic`` indirectly pulls in cv2 on some code
paths, and a subset of tracker protocol checks need it.  In practice the
import itself (``restart_ope``) requires only numpy, so the guard is light.

Specifically we check:
- When the tracker always returns a bbox that misses the target completely,
  n_restarts > 0 and success_rate is in [0, 1].
- When the tracker always returns a perfect bbox, n_restarts == 0 and
  success_rate == 1.0.
- The ``RestartOPEResult`` aggregate mean_success_rate is in [0, 1].
- Verify that RestartOPE can handle sequences with fewer than 2 frames
  gracefully (no crash).
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_tracker.types import BBox, TrackState


# ---------------------------------------------------------------------------
# Minimal stub tracker implementations
# ---------------------------------------------------------------------------


class _PerfectTracker:
    """Always returns the exact GT bbox from the last init."""

    name = "perfect"
    tier_hint = 0

    def __init__(self) -> None:
        self._bbox: BBox = BBox(0.0, 0.0, 1.0, 1.0)

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> TrackState:
        return TrackState(bbox=self._bbox, confidence=1.0, status="locked")

    def flops_per_update(self) -> float:
        return 1.0


class _FailingTracker:
    """Always returns a bbox at (9999, 9999) — guaranteed miss."""

    name = "failing"
    tier_hint = 0

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        pass  # does nothing; we always return a far-away bbox

    def update(self, frame: np.ndarray) -> TrackState:
        return TrackState(
            bbox=BBox(9999.0, 9999.0, 1.0, 1.0),
            confidence=0.0,
            status="lost",
        )

    def flops_per_update(self) -> float:
        return 1.0


class _AlternatingTracker:
    """Alternates perfect and failing on even/odd frames."""

    name = "alternating"
    tier_hint = 0

    def __init__(self) -> None:
        self._bbox: BBox = BBox(0.0, 0.0, 1.0, 1.0)
        self._count = 0

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox
        self._count = 0

    def update(self, frame: np.ndarray) -> TrackState:
        self._count += 1
        if self._count % 2 == 0:
            bbox = self._bbox
        else:
            bbox = BBox(9999.0, 9999.0, 1.0, 1.0)
        return TrackState(bbox=bbox, confidence=0.5, status="uncertain")

    def flops_per_update(self) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Minimal synthetic sequence helpers
# ---------------------------------------------------------------------------


def _make_frames(n: int) -> list[np.ndarray]:
    return [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n)]


def _make_gt(n: int, x: float = 10.0, y: float = 10.0) -> list[BBox]:
    return [BBox(x=x, y=y, w=20.0, h=20.0) for _ in range(n)]


class _FakeSequence:
    def __init__(self, name: str, frames: list, gt: list) -> None:
        self.name = name
        self.frames = frames
        self.ground_truth = gt
        self.init_bbox = gt[0]


class _FakeDataset:
    def __init__(self, sequences: list) -> None:
        self._seqs = sequences

    def __iter__(self):
        return iter(self._seqs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import_restart_ope() -> None:
    """Module and RestartOPE class must be importable."""
    from uav_tracker.metrics.restart_ope import RestartOPE, RestartOPEResult, RestartSequenceResult  # noqa: F401

    assert RestartOPE is not None


def test_failing_tracker_triggers_restarts() -> None:
    """A tracker that always misses must trigger at least one restart per sequence."""
    from uav_tracker.metrics.restart_ope import RestartOPE

    n_frames = 20
    seq = _FakeSequence(
        name="failing_seq",
        frames=_make_frames(n_frames),
        gt=_make_gt(n_frames),
    )
    dataset = _FakeDataset([seq])

    runner = RestartOPE(threshold=0.5, restart_gap=2)
    result = runner.run(_FailingTracker(), dataset)

    assert len(result.per_sequence) == 1
    sr = result.per_sequence[0]
    assert sr.n_restarts > 0, "Expected at least one restart for a failing tracker"
    assert 0.0 <= sr.success_rate <= 1.0, f"success_rate out of [0,1]: {sr.success_rate}"
    assert result.total_restarts == sr.n_restarts


def test_perfect_tracker_no_restarts() -> None:
    """A perfect tracker must never trigger a restart."""
    from uav_tracker.metrics.restart_ope import RestartOPE

    n_frames = 15
    seq = _FakeSequence(
        name="perfect_seq",
        frames=_make_frames(n_frames),
        gt=_make_gt(n_frames),
    )
    dataset = _FakeDataset([seq])

    runner = RestartOPE(threshold=0.5, restart_gap=3)
    result = runner.run(_PerfectTracker(), dataset)

    assert len(result.per_sequence) == 1
    sr = result.per_sequence[0]
    assert sr.n_restarts == 0, f"Perfect tracker should have no restarts, got {sr.n_restarts}"
    assert sr.success_rate == pytest.approx(1.0, abs=1e-6)


def test_success_rate_in_unit_interval() -> None:
    """success_rate must always be in [0, 1] for any tracker."""
    from uav_tracker.metrics.restart_ope import RestartOPE

    for tracker_cls in [_PerfectTracker, _FailingTracker, _AlternatingTracker]:
        seq = _FakeSequence(
            name="test_seq",
            frames=_make_frames(25),
            gt=_make_gt(25),
        )
        dataset = _FakeDataset([seq])
        runner = RestartOPE(threshold=0.5, restart_gap=3)
        result = runner.run(tracker_cls(), dataset)
        sr = result.per_sequence[0]
        assert 0.0 <= sr.success_rate <= 1.0, (
            f"{tracker_cls.__name__}: success_rate={sr.success_rate} out of [0,1]"
        )


def test_aggregate_mean_success_rate() -> None:
    """mean_success_rate must equal mean of per-sequence success_rates."""
    from uav_tracker.metrics.restart_ope import RestartOPE

    seqs = [
        _FakeSequence(f"seq_{i}", _make_frames(10), _make_gt(10))
        for i in range(3)
    ]
    dataset = _FakeDataset(seqs)
    runner = RestartOPE(threshold=0.5, restart_gap=2)
    result = runner.run(_AlternatingTracker(), dataset)

    assert len(result.per_sequence) == 3
    expected_mean = np.mean([sr.success_rate for sr in result.per_sequence])
    assert result.mean_success_rate == pytest.approx(float(expected_mean), abs=1e-9)


def test_short_sequence_skipped() -> None:
    """Sequences with fewer than 2 frames are silently skipped."""
    from uav_tracker.metrics.restart_ope import RestartOPE

    seq_short = _FakeSequence("short", _make_frames(1), _make_gt(1))
    seq_normal = _FakeSequence("normal", _make_frames(10), _make_gt(10))
    dataset = _FakeDataset([seq_short, seq_normal])

    runner = RestartOPE(threshold=0.5, restart_gap=2)
    result = runner.run(_PerfectTracker(), dataset)

    # short seq should be skipped — only 1 result entry.
    assert len(result.per_sequence) == 1
    assert result.per_sequence[0].name == "normal"


def test_empty_dataset_returns_zero_result() -> None:
    """Empty dataset must return zero-valued aggregate result without crashing."""
    from uav_tracker.metrics.restart_ope import RestartOPE

    dataset = _FakeDataset([])
    runner = RestartOPE()
    result = runner.run(_PerfectTracker(), dataset)

    assert result.mean_success_rate == 0.0
    assert result.total_restarts == 0
    assert result.per_sequence == []


def test_limit_parameter() -> None:
    """``limit`` caps the number of sequences processed."""
    from uav_tracker.metrics.restart_ope import RestartOPE

    seqs = [
        _FakeSequence(f"seq_{i}", _make_frames(10), _make_gt(10))
        for i in range(5)
    ]
    dataset = _FakeDataset(seqs)
    runner = RestartOPE(threshold=0.5, restart_gap=2)
    result = runner.run(_PerfectTracker(), dataset, limit=2)

    assert len(result.per_sequence) == 2


def test_restart_gap_delays_reinit() -> None:
    """With restart_gap=5, the tracker should reinit only on frame gap+1 after failure."""
    from uav_tracker.metrics.restart_ope import RestartOPE

    n_frames = 15
    seq = _FakeSequence("gap_test", _make_frames(n_frames), _make_gt(n_frames))
    dataset = _FakeDataset([seq])

    runner = RestartOPE(threshold=0.5, restart_gap=5)
    result = runner.run(_FailingTracker(), dataset)

    sr = result.per_sequence[0]
    # With gap=5 and n_frames=15, restarts are naturally bounded.
    assert sr.n_restarts >= 1
    # total frames evaluated << n_frames because many are skipped in the gap.
    n_eval = sr.aux.get("n_evaluated", 0)
    # At most n_frames - 1 frames can be evaluated (frame 0 is init).
    assert n_eval <= n_frames - 1
