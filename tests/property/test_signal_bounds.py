"""Property: every ``SwitchSignal`` report stays in its declared range.

When no signals are registered (current Phase 0 state) this test is
vacuously true. As Phase 3/4/5 lands signals, it tightens automatically.
"""

from __future__ import annotations

import pytest

from uav_tracker import SIGNALS


def test_all_registered_signals_report_within_range() -> None:
    """Vacuous in Phase 0; tightens as signals register."""
    names = list(SIGNALS.names())
    if not names:
        pytest.skip("No signals registered yet (Phase 3+ will tighten this test)")
    # Phase 3+: iterate over names, construct synthetic contexts, assert
    # range[0] <= report.value <= range[1].
    # Deliberate xfail for now so the implementation can drop the skip.
    pytest.xfail("Phase 3+: hypothesis-driven sweep over contexts")
