"""CSC-v4 module A7 — sequential-evidence control gate (Wald SPRT + expected-gain).

Why this exists
---------------
V3 control fired an action on a *single* frame whenever a hand-tuned conjunction
held (``sm_local_top2_ratio >= tau AND response_entropy >= tau``). That over-fired on
transient one-frame dips (the LA-precision wall: CSC over-fired LA, e.g. uav6 98% LA
at degenerate confidence). V4 replaces that one-shot rule with two cooperating gates:

1. **Sequential gate (`SPRTGate`)** — Wald's Sequential Probability Ratio Test.
   Instead of thresholding one frame, we accumulate a per-frame log-likelihood ratio
   (LLR) of "failure (H1)" vs "fine (H0)" and only declare ``'fire'`` once the running
   sum crosses the upper Wald bound ``A``. Transient evidence that reverses crosses the
   lower bound ``B`` and yields ``'clear'`` (reset). This buys *temporal* robustness:
   a single noisy frame can't trip control, but sustained evidence trips it quickly.

2. **Value gate (`expected_gain_gate`)** — even once failure is *detected*, we only act
   if some action has positive expected value: ``max_a (predicted ΔIoU[a] − cost[a])``
   above ``min_gain``; otherwise HOLD (the "do-not-act" escape). This is the decision-
   theoretic counterpart to the SPRT detection.

`llr_from_evidence` maps the V4 gate telemetry to a per-frame LLR so the two gates
compose: ``state = sprt.update(llr_from_evidence(tel))``; if ``state == 'fire'`` ask
``expected_gain_gate(pred.action_utility, costs)`` for the action.

Wald SPRT thresholds
--------------------
For target false-alert rate ``alpha`` (P[fire | H0]) and miss rate ``beta``
(P[clear | H1]) the standard Wald bounds on the cumulative LLR are::

    A = log((1 - beta) / alpha)        # upper bound  -> 'fire'   (accept H1)
    B = log(beta / (1 - alpha))        # lower bound  -> 'clear'  (accept H0)

With defaults alpha=0.05, beta=0.1:  A = log(0.9/0.05) = log(18) ≈ +2.890,
B = log(0.1/0.95) = log(0.10526) ≈ -2.251. The accumulator starts at 0 and is clamped
to ``[B - max_evidence, A + max_evidence]`` so it can't run away during a long episode.

This file is self-contained: pure python + numpy, imports only ``csc_lib.csc.v4.v4types``
(Action, ACTION_NAMES). No torch, no other v4 module deps, no dataset access.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from csc_lib.csc.v4.v4types import Action, ACTION_NAMES


__all__ = [
    "SPRTGate",
    "SPRTResult",
    "expected_gain_gate",
    "llr_from_evidence",
    "DEFAULT_ACTION_COSTS",
]


# Result strings returned by SPRTGate.update — keep as plain str for easy logging.
_ACCUMULATE = "accumulate"
_FIRE = "fire"
_CLEAR = "clear"


@dataclass
class SPRTResult:
    """Optional rich result mirroring SPRTGate.update's string (for diagnostics)."""
    decision: str            # 'accumulate' | 'fire' | 'clear'
    evidence: float          # cumulative LLR at this step (after clamping)
    upper: float             # threshold A
    lower: float             # threshold B
    n_steps: int             # frames accumulated in the current episode
    budget_remaining: int    # false-alert budget left


class SPRTGate:
    """Wald Sequential Probability Ratio Test over a stream of per-frame LLRs.

    Each :meth:`update` adds one frame's log-likelihood ratio ``llr = log p(x|H1)/p(x|H0)``
    to a running sum ``self.evidence``. The decision rule:

    * ``evidence >= A``  -> ``'fire'``   (accept H1 = failure/act); episode resets, a
      false-alert budget unit is *not* consumed here (the budget guards against firing
      too often — see ``false_alert_budget`` below).
    * ``evidence <= B``  -> ``'clear'``  (accept H0 = fine); episode resets.
    * otherwise          -> ``'accumulate'`` (keep watching).

    Parameters
    ----------
    alpha : float
        Target false-alert probability P[fire | H0]. Smaller -> harder to fire.
    beta : float
        Target miss probability P[clear | H1]. Smaller -> harder to clear.
    max_evidence : float
        Clamp half-width added beyond ``[B, A]`` so the accumulator can't diverge over a
        long episode; also caps how much "credit" a calm stretch can bank against a
        future spike. Defaults to ``2 * (A - B)`` if None.
    false_alert_budget : int | None
        Optional cap on the number of ``'fire'`` decisions allowed before
        :meth:`reset_budget`. Once exhausted, crossing ``A`` is downgraded to
        ``'accumulate'`` (the gate refuses to fire again). ``None`` = unlimited.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.1,
        max_evidence: Optional[float] = None,
        false_alert_budget: Optional[int] = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0,1), got {beta}")
        if alpha + beta >= 1.0:
            # Wald requires the bands not to cross: A>0>B needs (1-beta)/alpha>1 and
            # beta/(1-alpha)<1, i.e. alpha<1-beta. Guard so A>B always holds.
            raise ValueError(f"need alpha + beta < 1 (got {alpha + beta})")

        self.alpha = float(alpha)
        self.beta = float(beta)
        # Wald bounds on the cumulative log-likelihood ratio.
        self.upper = math.log((1.0 - beta) / alpha)   # A  (> 0)
        self.lower = math.log(beta / (1.0 - alpha))    # B  (< 0)
        self.max_evidence = (
            float(max_evidence) if max_evidence is not None
            else 2.0 * (self.upper - self.lower)
        )
        self.false_alert_budget = false_alert_budget
        self._budget_left = (
            int(false_alert_budget) if false_alert_budget is not None else None
        )

        self.evidence: float = 0.0
        self.n_steps: int = 0

    # -- thresholds (read-only convenience) ----------------------------------------
    @property
    def A(self) -> float:  # noqa: N802 — Wald's canonical name
        """Upper log-threshold (fire / accept H1)."""
        return self.upper

    @property
    def B(self) -> float:  # noqa: N802 — Wald's canonical name
        """Lower log-threshold (clear / accept H0)."""
        return self.lower

    @property
    def budget_remaining(self) -> Optional[int]:
        return self._budget_left

    def reset(self) -> None:
        """Reset the accumulated evidence and step count (NOT the false-alert budget)."""
        self.evidence = 0.0
        self.n_steps = 0

    def reset_budget(self) -> None:
        """Replenish the false-alert budget (call e.g. once per sequence)."""
        self._budget_left = (
            int(self.false_alert_budget) if self.false_alert_budget is not None else None
        )

    def update(self, llr: float) -> str:
        """Accumulate one frame's LLR and return ``'accumulate'|'fire'|'clear'``.

        Non-finite / missing ``llr`` is treated as 0 evidence (no information) and never
        moves the accumulator, so missing telemetry is safe.
        """
        if llr is None or not math.isfinite(float(llr)):
            llr = 0.0
        self.evidence += float(llr)
        self.n_steps += 1

        # Clamp so a long episode cannot diverge to ±inf.
        lo = self.lower - self.max_evidence
        hi = self.upper + self.max_evidence
        if self.evidence > hi:
            self.evidence = hi
        elif self.evidence < lo:
            self.evidence = lo

        if self.evidence >= self.upper:
            if self._budget_left is not None:
                if self._budget_left <= 0:
                    # Budget exhausted: refuse to fire, hold the accumulator at A so the
                    # next genuine reversal still produces 'clear'.
                    self.evidence = self.upper
                    return _ACCUMULATE
                self._budget_left -= 1
            self.reset()
            return _FIRE
        if self.evidence <= self.lower:
            self.reset()
            return _CLEAR
        return _ACCUMULATE

    def update_verbose(self, llr: float) -> SPRTResult:
        """Like :meth:`update` but also returns evidence/threshold diagnostics.

        Note: call this *instead of* update (it advances state once), not in addition.
        """
        # Snapshot the pre-reset evidence for reporting, since update() may reset.
        decision = self.update(llr)
        ev = self.upper if decision == _FIRE else (self.lower if decision == _CLEAR else self.evidence)
        return SPRTResult(
            decision=decision,
            evidence=ev,
            upper=self.upper,
            lower=self.lower,
            n_steps=self.n_steps,
            budget_remaining=self._budget_left if self._budget_left is not None else -1,
        )


# ---------------------------------------------------------------------------------
# Expected-gain (value) gate
# ---------------------------------------------------------------------------------

# Default per-action costs (in ΔIoU units) — acting has a price; HOLD/FREEZE are free,
# GLOBAL_SEARCH (budgeted re-detector) is the most expensive, RELOCATE risks a bad jump.
# These are sane defaults; the controller (A10) may override per its config.
DEFAULT_ACTION_COSTS: dict[str, float] = {
    Action.HOLD.name.lower(): 0.0,
    Action.MOTION_BRIDGE.name.lower(): 0.02,
    Action.RELOCATE.name.lower(): 0.05,
    Action.WIDEN.name.lower(): 0.03,
    Action.GLOBAL_SEARCH.name.lower(): 0.08,
    Action.TEMPLATE_UPDATE.name.lower(): 0.04,
    Action.FREEZE.name.lower(): 0.0,
}


def expected_gain_gate(
    action_utility: dict,
    costs: Optional[dict] = None,
    min_gain: float = 0.0,
) -> tuple[int, float]:
    """Pick the action maximising ``predicted ΔIoU − cost``, else HOLD.

    Parameters
    ----------
    action_utility : dict
        ``{action_name -> predicted ΔIoU}`` (keys must be ACTION_NAMES, the lowercase
        Action names; this is exactly ``V4Prediction.action_utility``). Missing actions
        are skipped. Non-finite utilities are skipped.
    costs : dict | None
        ``{action_name -> cost}`` in the same ΔIoU units. ``None`` -> DEFAULT_ACTION_COSTS.
        Missing keys default to 0.0 cost.
    min_gain : float
        Minimum net gain required to act. If no action clears ``min_gain`` the gate
        returns ``(Action.HOLD, 0.0)`` (the do-not-act escape). HOLD itself is never the
        argmax winner unless it is the only finite option.

    Returns
    -------
    (action:int, gain:float)
        ``action`` is an ``Action`` int value; ``gain`` is the net expected ΔIoU of the
        chosen action (0.0 when HOLD is returned as the abstain decision).
    """
    if costs is None:
        costs = DEFAULT_ACTION_COSTS

    best_action = int(Action.HOLD)
    best_gain = -math.inf
    hold_name = Action.HOLD.name.lower()

    for name, util in (action_utility or {}).items():
        if util is None or not math.isfinite(float(util)):
            continue
        if name not in ACTION_NAMES:
            # Unknown action name — ignore rather than crash (forward-compatible).
            continue
        # HOLD is the abstain baseline, not a competing "action to take".
        if name == hold_name:
            continue
        net = float(util) - float(costs.get(name, 0.0))
        if net > best_gain:
            best_gain = net
            best_action = int(Action[name.upper()])

    if best_gain < min_gain or not math.isfinite(best_gain):
        return int(Action.HOLD), 0.0
    return best_action, float(best_gain)


# ---------------------------------------------------------------------------------
# LLR proxy from gate features
# ---------------------------------------------------------------------------------

# Calibrated LLR proxy weights (documented below). These are a transparent,
# logistic-style mapping from the V4 gate features to a per-frame failure LLR. They are
# *proxy* (hand-calibrated from the A7 finding that score-map structure + appearance
# separate true-loss vs false-LA at AUROC ~0.85), NOT fitted weights — the trained V4
# model's hazard head is the eventual source; this proxy lets the SPRT run before/without
# the model and is the documented fallback.
#
# Sign convention (positive => evidence FOR failure / LA):
#   sm_local_top2_ratio  high  => a competing 2nd peak => MORE failure evidence (+)
#   response_entropy     high  => diffuse / no clear peak => MORE failure evidence (+)
#   last_cosine_sim      high  => appearance still matches target => LESS failure (−)
#   peak_margin (sm_local_peak_margin) high => one dominant confident peak => LESS (−)
#
# Each feature is centred at a neutral operating point and scaled, then weighted and
# summed; the bias sets the prior. The result is a logit-scale LLR (natural log units),
# directly summable by the SPRT.
_LLR_BIAS = -0.40   # prior leans slightly toward H0 (fine) so noise alone won't fire

_LLR_TERMS = (
    # (feature_key, neutral_center, scale, weight)   contribution = weight*(x-center)/scale
    ("sm_local_top2_ratio", 0.50, 0.25, +1.60),   # competing peak -> failure
    ("response_entropy",    0.50, 0.25, +1.20),   # diffuse map -> failure
    ("last_cosine_sim",     0.55, 0.25, -1.40),   # identity match -> fine
    ("peak_margin",         0.40, 0.25, -1.30),   # dominant peak -> fine
)

# Accepted aliases so callers can pass either the SGLATrack telemetry name or the short
# gate name for the peak-margin feature.
_PEAK_MARGIN_ALIASES = ("peak_margin", "sm_local_peak_margin", "sm_peak_margin")


def _get_feature(features: dict, key: str) -> Optional[float]:
    """Fetch a gate feature with alias fallback; None if missing/non-finite."""
    val = features.get(key)
    if val is None and key == "peak_margin":
        for alias in _PEAK_MARGIN_ALIASES:
            if features.get(alias) is not None:
                val = features.get(alias)
                break
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def llr_from_evidence(features: dict, clip: float = 6.0) -> float:
    """Map per-frame gate features to a failure-vs-fine log-likelihood ratio.

    Documented proxy (see ``_LLR_TERMS`` above)::

        llr = bias
            + 1.60 * (sm_local_top2_ratio - 0.50)/0.25   # competing 2nd peak
            + 1.20 * (response_entropy     - 0.50)/0.25   # diffuse response map
            - 1.40 * (last_cosine_sim      - 0.55)/0.25   # appearance still matches
            - 1.30 * (peak_margin          - 0.40)/0.25   # one dominant confident peak

    Positive ``llr`` is evidence the tracker is failing (LA/FC); negative is evidence it
    is fine. Missing features contribute nothing (treated as at their neutral center), so
    a frame with no telemetry yields ``llr == bias`` (slight lean toward "fine"), keeping
    the SPRT safe under missing data. The output is clipped to ``[-clip, clip]`` so one
    extreme frame cannot single-handedly cross the Wald bound.

    Parameters
    ----------
    features : dict
        Per-frame gate telemetry. Recognised keys: ``sm_local_top2_ratio``,
        ``response_entropy``, ``last_cosine_sim``, and ``peak_margin`` (aliases
        ``sm_local_peak_margin`` / ``sm_peak_margin``).
    clip : float
        Symmetric clip on the returned LLR (natural-log units).

    Returns
    -------
    float
        The per-frame LLR, ready to feed :meth:`SPRTGate.update`.
    """
    features = features or {}
    llr = _LLR_BIAS
    for key, center, scale, weight in _LLR_TERMS:
        x = _get_feature(features, key)
        if x is None:
            continue  # missing => neutral => zero contribution
        llr += weight * (x - center) / scale
    if not math.isfinite(llr):
        return 0.0
    return float(max(-clip, min(clip, llr)))


# ---------------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- 1. Thresholds are the Wald bounds and A > 0 > B ---------------------------
    g = SPRTGate(alpha=0.05, beta=0.1)
    assert abs(g.A - math.log(0.9 / 0.05)) < 1e-9, g.A
    assert abs(g.B - math.log(0.1 / 0.95)) < 1e-9, g.B
    assert g.A > 0.0 > g.B, (g.A, g.B)
    print(f"[A7] SPRT thresholds: A=+{g.A:.4f}  B={g.B:.4f}  "
          f"max_evidence={g.max_evidence:.3f}")

    # --- 2. Sustained HIGH evidence must eventually FIRE ---------------------------
    g.reset()
    fired_at = None
    for t in range(50):
        # strong positive LLR every frame (with a little noise)
        llr = 1.2 + 0.1 * rng.standard_normal()
        dec = g.update(llr)
        if dec == _FIRE:
            fired_at = t
            break
        assert dec == _ACCUMULATE, dec
    assert fired_at is not None, "sustained high LLR never fired"
    assert fired_at <= 5, f"fired too slowly (t={fired_at})"
    print(f"[A7] sustained-high-LLR stream -> FIRE after {fired_at + 1} frames "
          f"(t={fired_at}).")

    # --- 3. Sustained LOW evidence must NEVER fire ---------------------------------
    g.reset()
    ever_fired = False
    for _ in range(500):
        llr = -0.8 + 0.1 * rng.standard_normal()   # evidence FOR 'fine'
        dec = g.update(llr)
        if dec == _FIRE:
            ever_fired = True
            break
    assert not ever_fired, "low-evidence stream wrongly fired"
    print("[A7] sustained-low-LLR stream -> never fires (clears repeatedly). OK")

    # --- 3b. Transient single moderate frame must NOT fire (temporal robustness) ---
    # A single per-frame-magnitude blip (≈ one strong failing frame's LLR) sitting
    # amid 'fine' frames stays below A: control needs *sustained* evidence, not one dip.
    g.reset()
    decs = []
    for t in range(10):
        llr = 1.2 if t == 4 else -0.5   # one strong-but-realistic blip amid 'fine'
        decs.append(g.update(llr))
    assert _FIRE not in decs, f"single transient blip wrongly fired: {decs}"
    print("[A7] single transient blip amid fine frames -> no fire (needs sustained). OK")

    # --- 3c. False-alert budget caps the number of fires ---------------------------
    gb = SPRTGate(alpha=0.05, beta=0.1, false_alert_budget=2)
    fires = 0
    for _ in range(200):
        if gb.update(2.0) == _FIRE:   # always-firing stream
            fires += 1
    assert fires == 2, f"budget not enforced (fires={fires})"
    assert gb.budget_remaining == 0, gb.budget_remaining
    gb.reset_budget()
    assert gb.budget_remaining == 2
    print("[A7] false_alert_budget=2 -> exactly 2 fires, then refused; reset_budget OK")

    # --- 4. expected_gain_gate picks the right action on a toy utility dict --------
    util = {
        "hold": 0.0,
        "motion_bridge": 0.30,   # net 0.30 - 0.02 = 0.28  <- should win
        "relocate": 0.10,        # net 0.10 - 0.05 = 0.05
        "widen": 0.05,           # net 0.05 - 0.03 = 0.02
        "global_search": 0.12,   # net 0.12 - 0.08 = 0.04
        "freeze": 0.0,
    }
    act, gain = expected_gain_gate(util, DEFAULT_ACTION_COSTS, min_gain=0.0)
    assert act == int(Action.MOTION_BRIDGE), (act, ACTION_NAMES[act])
    assert abs(gain - 0.28) < 1e-9, gain
    print(f"[A7] expected_gain_gate -> {ACTION_NAMES[act]} (net gain {gain:+.3f})")

    # All-negative utilities (or below min_gain) must abstain -> HOLD.
    bad_util = {k: -0.5 for k in ACTION_NAMES}
    act2, gain2 = expected_gain_gate(bad_util, DEFAULT_ACTION_COSTS, min_gain=0.0)
    assert act2 == int(Action.HOLD) and gain2 == 0.0, (act2, gain2)
    # min_gain raises the bar: motion_bridge net 0.28 < 0.30 -> abstain.
    act3, _ = expected_gain_gate(util, DEFAULT_ACTION_COSTS, min_gain=0.30)
    assert act3 == int(Action.HOLD), ACTION_NAMES[act3]
    print("[A7] expected_gain_gate abstains (-> HOLD) on all-negative util & high min_gain. OK")

    # --- 5. llr_from_evidence: failing telemetry => +LLR, fine telemetry => −LLR ---
    failing = {
        "sm_local_top2_ratio": 0.95,   # strong competing peak
        "response_entropy": 0.90,      # diffuse map
        "last_cosine_sim": 0.15,       # appearance lost
        "sm_local_peak_margin": 0.05,  # no dominant peak (alias key)
    }
    fine = {
        "sm_local_top2_ratio": 0.10,
        "response_entropy": 0.15,
        "last_cosine_sim": 0.92,
        "peak_margin": 0.85,
    }
    llr_fail = llr_from_evidence(failing)
    llr_fine = llr_from_evidence(fine)
    assert llr_fail > 0.5, llr_fail
    assert llr_fine < -0.5, llr_fine
    assert llr_from_evidence({}) == _LLR_BIAS  # missing => neutral => bias
    print(f"[A7] llr_from_evidence: failing={llr_fail:+.3f}  fine={llr_fine:+.3f}  "
          f"empty={llr_from_evidence({}):+.3f}")

    # --- 6. End-to-end: a (moderately) failing stream fires after a few frames -----
    # Mildly-off-neutral telemetry so per-frame LLR is small (~1) and the SPRT must
    # accumulate over a few frames before crossing A.
    failing_mod = {
        "sm_local_top2_ratio": 0.60,
        "response_entropy": 0.58,
        "last_cosine_sim": 0.48,
        "peak_margin": 0.33,
    }
    llr_mod = llr_from_evidence(failing_mod)
    assert 0.3 < llr_mod < 2.0, llr_mod
    g.reset()
    e2e_fire = None
    for t in range(30):
        dec = g.update(llr_from_evidence(failing_mod))
        if dec == _FIRE:
            e2e_fire = t
            break
    assert e2e_fire is not None, "end-to-end failing stream never fired"
    assert e2e_fire >= 1, f"moderate stream should need >1 frame (fired t={e2e_fire})"
    g.reset()
    assert all(g.update(llr_from_evidence(fine)) != _FIRE for _ in range(200))
    print(f"[A7] end-to-end: moderate-failing stream FIRES after {e2e_fire + 1} frames "
          f"(t={e2e_fire}); fine-telemetry stream never fires.")

    print("[A7] sprt_gate.py smoke OK")
