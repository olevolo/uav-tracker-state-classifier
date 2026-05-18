"""End-to-end integration test: KCF on a synthetic smooth sequence.

Phase 1 acceptance: tracker completes 20 frames of near-linear motion
with AUC > 0.5. Currently skipped — the KCF implementation lands in
Phase 1.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 1: KCF + OPE wiring required")
def test_kcf_on_smooth_sequence(smooth_sequence) -> None:  # pragma: no cover
    """Integration stub — reinstated in Phase 1.

    Will construct ``KCFKalmanTracker``, feed the 20-frame fixture
    through ``OPERunner``, and assert ``result.auc > 0.5`` (loose band).
    """
    pass
