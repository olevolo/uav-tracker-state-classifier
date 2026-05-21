"""Adapter: SALTRDState → TargetState for backward compatibility.

Allows existing TSA-based code to receive SALT-RD states without
knowing about the learned controller internals.

TSA remains as fallback — this adapter is used only when SALT-RD primary
mode is active and a consumer needs a TargetState int (e.g. for logging or
legacy code paths).
"""
from __future__ import annotations

from uav_tracker.ml.tsa.target_state import TargetState


def saltrd_state_to_tsa_state(saltrd_state) -> int:
    """Map SALTRDState to TargetState int for downstream TSA consumers.

    Parameters
    ----------
    saltrd_state : SALTRDState
        State produced by SALTRDAdvisor.get_state().

    Returns
    -------
    int
        TargetState value compatible with existing TSA consumers.
    """
    try:
        from salt_r.advisor import SALTRDState
    except ImportError:
        # salt_r not on path — return safe default
        return TargetState.CONFIRMED.value

    mapping = {
        SALTRDState.TRUSTED_TRACKING:      TargetState.CONFIRMED.value,
        SALTRDState.LOW_EVIDENCE_TRACKING: TargetState.CONFIRMED.value,  # still tracking
        SALTRDState.FALSE_CONFIRMED_RISK:  TargetState.OCCLUDED.value,   # uncertain
        SALTRDState.PROACTIVE_DYNAMIC_RISK: TargetState.DYNAMIC.value,
        SALTRDState.REACQUIRE_NEEDED:      TargetState.LOST.value,
    }
    return mapping.get(saltrd_state, TargetState.CONFIRMED.value)
