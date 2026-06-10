"""CSC-v4 shared types — the integration backbone all v4 modules import from.

V4 = diagnosis + action-utility (NOT state-classifier + hand policy). This file is
the SINGLE SOURCE OF TRUTH for the dataclasses/enums that cross module boundaries, so
the parallel-built v4 modules stay interface-compatible. DO NOT duplicate these
elsewhere; import from `csc_lib.csc.v4.v4types`.

V3 (csc_prod) is frozen and untouched; everything here is additive under csc_lib/csc/v4/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


# ---- Runtime derived states (UNCHANGED from V3 for eval/metric compatibility) ----
class DerivedStateV4(IntEnum):
    CC = 0   # correct_confirmed
    CU = 1   # correct_uncertain
    LA = 2   # lost_aware
    FC = 3   # false_confirmed


# ---- Training-time FC subtypes (collapse to FC=3 at runtime) ----
class FCSubtype(IntEnum):
    NONE = 0        # not FC
    DISTRACTOR = 1  # FC_D: confidently localised a wrong *similar* object
    BACKGROUND = 2  # FC_B: confident peak on background


# ---- Training-time LA subtypes (collapse to LA=2 at runtime) ----
class LASubtype(IntEnum):
    NONE = 0
    FALSE = 1       # LA_FALSE: tracker actually fine, CSC over-fired (DO NOT ACT)
    SMOOTH = 2      # LA_SMOOTH: target continues smooth motion  -> motion_bridge
    ABRUPT = 3      # LA_ABRUPT: target stopped/turned           -> bridge harmful, hold/verify
    OCCLUDED = 4    # LA_OCCLUDED: absent/occluded/out-of-view   -> hold + freeze
    CANDIDATE = 5   # LA_CANDIDATE: a 2ndary/global candidate exists -> verify + relocate


# ---- Control action space (the action-utility heads predict ΔIoU per action) ----
class Action(IntEnum):
    HOLD = 0
    MOTION_BRIDGE = 1
    RELOCATE = 2
    WIDEN = 3
    GLOBAL_SEARCH = 4   # budgeted re-detector (multi-crop SGLATrack / AVTrack sidecar)
    TEMPLATE_UPDATE = 5
    FREEZE = 6


ACTION_NAMES = [a.name.lower() for a in Action]          # ['hold','motion_bridge',...]
N_ACTIONS = len(Action)


@dataclass
class Candidate:
    """A score-map peak proposed as a (re-)localisation target. xywh = pixel bbox."""
    cx: float
    cy: float
    w: float
    h: float
    score: float = 0.0
    rank: int = 0                      # 0 = top-1 peak, 1.. = secondary peaks
    peak_margin: float = 0.0
    sim_to_init: float = float("nan")        # cosine vs frame-0 template prototype
    sim_to_recent: float = float("nan")      # cosine vs recent CC prototype
    sim_to_distractor: float = float("nan")  # max cosine vs distractor memory
    motion_plausibility: float = float("nan")
    scale_plausibility: float = float("nan")
    embedding: Optional[np.ndarray] = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)


@dataclass
class Prototype:
    embedding: np.ndarray
    frame_idx: int
    kind: str  # 'anchor' (frame-0) | 'recent' (latest CC) | 'distractor'


@dataclass
class V4Prediction:
    """Output of the V4 model's predict() for one frame (Student, causal, no GT)."""
    derived_probs: np.ndarray                      # (4,) CC/CU/LA/FC softmax
    derived_state: int                             # argmax (or gated) derived state
    fc_subtype_probs: Optional[np.ndarray] = None  # (3,) NONE/DISTRACTOR/BACKGROUND
    la_subtype_probs: Optional[np.ndarray] = None  # (6,) NONE/FALSE/SMOOTH/ABRUPT/OCCLUDED/CANDIDATE
    hazard: dict = field(default_factory=dict)     # {'next_1':p,'next_3':p,'next_10':p}
    action_utility: dict = field(default_factory=dict)  # action_name -> predicted ΔIoU
    do_not_act_prob: float = 0.0
    template_update_safe_prob: float = 0.0
    risk_score: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ActionDecision:
    """What the V4 controller decided to do this frame."""
    action: int = int(Action.HOLD)
    params: dict = field(default_factory=dict)   # e.g. {'cx':..,'cy':..} or {'factor':..}
    reason: str = ""
    evidence: float = 0.0                        # SPRT accumulated evidence at decision time
    expected_gain: float = 0.0


# Canonical model output dim hints (heads in csc_lib/csc/v4/model_v4.py must match):
HEAD_DIMS = {
    "derived": 4,
    "fc_subtype": len(FCSubtype),    # 3
    "la_subtype": len(LASubtype),    # 6
    "hazard": 3,                     # next_1, next_3, next_10  (sigmoid)
    "action_utility": N_ACTIONS,     # 7  (regression, predicted ΔIoU per action)
    "do_not_act": 1,                 # sigmoid
    "template_update_safe": 1,       # sigmoid
}
