"""Unit tests for core datatypes (``uav_tracker.types``).

Kept minimal; the Protocols themselves are validated by
``tests/contract/test_plugin_contract.py``. These tests pin the *shape*
of the dataclasses Architect owns.
"""

from __future__ import annotations

import pytest

from uav_tracker import BBox, TrackState, SignalReport


def test_bbox_is_immutable() -> None:
    """``BBox`` is frozen → attempted mutation must raise."""
    box = BBox(x=1.0, y=2.0, w=3.0, h=4.0)
    with pytest.raises(Exception):
        # FrozenInstanceError on frozen dataclasses; test generously.
        box.x = 10.0  # type: ignore[misc]


def test_bbox_fields_roundtrip() -> None:
    box = BBox(x=0.5, y=1.5, w=2.5, h=3.5)
    assert box.x == 0.5
    assert box.y == 1.5
    assert box.w == 2.5
    assert box.h == 3.5


def test_track_state_defaults() -> None:
    """``TrackState.aux`` must default to an empty dict (not shared)."""
    box = BBox(x=0.0, y=0.0, w=1.0, h=1.0)
    state_a = TrackState(bbox=box, confidence=0.5, status="locked")
    state_b = TrackState(bbox=box, confidence=0.1, status="uncertain")
    # Independent dicts → mutation of one doesn't leak to the other.
    state_a.aux["k"] = 1
    assert "k" not in state_b.aux


def test_signal_report_defaults() -> None:
    """``SignalReport`` reliable=True and aux={} by default."""
    report = SignalReport(value=0.5)
    assert report.reliable is True
    assert report.vector is None
    assert report.aux == {}
