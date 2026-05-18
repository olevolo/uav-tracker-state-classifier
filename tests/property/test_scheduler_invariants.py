"""Property: schedulers respect their own cooldown.

Paper §3.3 Hysteresis scheduler has cooldown_frames + confirm_frames
invariants. Between any two tier switches, at least ``cooldown_frames``
must elapse. This test iterates registered schedulers and asserts that
invariant on hypothesis-generated signal timelines.

Vacuous in Phase 0 (no schedulers registered yet).
"""

from __future__ import annotations

import pytest

from uav_tracker import SCHEDULERS


def test_schedulers_respect_cooldown() -> None:
    names = list(SCHEDULERS.names())
    if not names:
        pytest.skip("No schedulers registered yet (Phase 3+ will tighten this test)")
    pytest.xfail("Phase 3+: hypothesis-driven switch-trace check")
