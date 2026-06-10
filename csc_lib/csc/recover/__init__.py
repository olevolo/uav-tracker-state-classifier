"""CSC-recover — V3 production add-on for FC detect-verify-recover.

V3 (csc_prod = R3-fcw3) is the runtime model and stays frozen. This module is
*additive* and lives alongside ``csc_lib/csc/v4/``. It borrows three pure-logic
V4 building blocks (no V4 model dependency) to fix the FCR-vs-AUC wall:

* ``csc_lib/csc/v4/memory.py`` — PrototypeMemory (anchor + recent EMA + distractor)
* ``csc_lib/csc/v4/verifier.py`` — CandidateVerifier (K-ranking, distractor veto)
* ``csc_lib/csc/v4/sprt_gate.py`` — Wald SPRT (adaptive temporal verification)

The controller orchestrates: FC trigger -> tracker.redetect(top_k) ->
identity verifier (against memory) -> SPRT-gated switch -> abort window with
rollback. On confirmed-correct frames the recover loop seeds memory.recent;
on FC streaks it seeds memory.distractor with the incumbent's wrong-lock
embedding (so future false-FC no longer matches the same wrong object).

Public entry point: :class:`FCRecoverController` (one instance per sequence).
The runner in ``tools/run_with_csc.py`` constructs it under the
``--policy_fc_recover`` flag and applies its returned switch/rollback levers
causally before the next ``tracker.update()``.
"""
from __future__ import annotations

from csc_lib.csc.recover.candidate_generator import (
    CandidateGeneratorConfig,
    MultiSourceCandidateGenerator,
)
from csc_lib.csc.recover.recover_ctrl import (
    FCRecoverConfig,
    FCRecoverController,
    FCRecoverDecision,
)

__all__ = [
    "CandidateGeneratorConfig",
    "FCRecoverConfig",
    "FCRecoverController",
    "FCRecoverDecision",
    "MultiSourceCandidateGenerator",
]
