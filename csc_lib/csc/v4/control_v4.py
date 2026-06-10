"""CSC-v4 module A10 — LA-triage + action selection + abort + Meta-Updater.

The V4 orchestrator. Where V3 ran a hand-tuned policy keyed on the raw derived
state (``tools/run_with_csc.py``: ``_hard_la_gate`` -> motion_bridge / widen /
relocate, ``policy_fc_control`` -> freeze, ``gated_freeze``), V4 makes the decision
*subtype-aware* and *evidence-gated*:

  diagnosis (V4Prediction: derived + LA/FC subtypes + hazard + per-action ΔIoU)
    -> LA/FC triage (this file)
    -> SPRT sequential-evidence gate (A7)  +  expected-gain gate (A7)
    -> candidate verification by identity (A3) / budgeted re-detect (A8/A9)
    -> ActionDecision

Triage (per the build contract, mirroring the LASubtype / FCSubtype taxonomy):

  LA_FALSE      -> HOLD            (CSC over-fired; tracker is fine: do-no-harm,
                                    NO freeze, NO bbox override). This is the
                                    false-LA wall fix that killed V3's net-negative
                                    widen (uav6 0.667 -> 0.085).
  LA_SMOOTH     -> MOTION_BRIDGE   (extrapolate pre-loss velocity; +displacement
                                    cap; arm an ABORT window — roll back if the
                                    telemetry does not improve in a few frames).
                                    Verified win in V3 (car9 +0.46, group3_2 +0.46).
  LA_ABRUPT     -> HOLD            (bridge is harmful on a turn/stop — bird1_1
                                    -0.15; hold/verify instead).
  LA_OCCLUDED   -> HOLD + FREEZE   (target absent: don't move the search, don't
                                    update the template onto background).
  LA_CANDIDATE  -> verify top-k, RELOCATE only if a verified candidate beats the
                  last-good appearance by a margin (the A3 guard that prevents the
                  catastrophic relocate jump, person9/car6_2 -0.5).
  LA unknown,   -> GLOBAL_SEARCH   (budgeted multi-crop re-detect + verify, with a
  persistent                        2-frame vote so a single spurious crop peak
                                    cannot hijack the track).

  FC suspected, -> FREEZE only     (block the template update; do NOT move the bbox
  not verified                      on a guess).
  FC verified   -> FREEZE + reject bbox + search (last-good, else global).

Every *acting* decision is gated through the SPRT gate (A7) AND an expected-gain
check (A7 ``expected_gain_gate`` over the model's per-action ΔIoU head): if the
sequential evidence has not fired, or no action's predicted gain clears its cost,
the controller falls back to HOLD (``do_not_act``). The Meta-Updater decides the
TEMPLATE_UPDATE action separately (state-gated identity check + short post-recovery
window) — V4's principled replacement for ``--recovery_update_window``.

This module writes against the A2/A3/A6/A7/A8/A9 *interfaces* only. Sibling modules
that don't exist yet are stubbed inline as duck-typed Protocols + a tiny dummy impl
so the ``__main__`` smoke runs standalone (no torch model, no tracker, no dataset).
All such seams are marked ``# INTEGRATION:``.

V3 (csc_prod) is frozen and untouched; this is additive under csc_lib/csc/v4/.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

# L13: when run as a standalone script (`python csc_lib/csc/v4/control_v4.py`) the
# repo root is not on sys.path, so the absolute `csc_lib...` import below would fail.
# Mirror the la_smoke header (salrtd/src, src, repo root) so the smoke runs both
# standalone AND via `-m`. Guarded on __package__ so normal imports are untouched.
if not __package__:  # executed directly, not imported / not `-m`
    import sys as _sys
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[3]  # csc_lib/csc/v4/.. -> repo root
    for _p in (_ROOT / "salrtd" / "src", _ROOT / "src", _ROOT):
        _sp = str(_p)
        if _sp not in _sys.path:
            _sys.path.insert(0, _sp)

from csc_lib.csc.v4.v4types import (
    Action,
    ACTION_NAMES,
    ActionDecision,
    Candidate,
    DerivedStateV4,
    FCSubtype,
    LASubtype,
    V4Prediction,
)


# ===========================================================================
# Cross-module interfaces (Protocols). These mirror the build CONTRACT so this
# orchestrator type-checks/runs against the real A2/A3/A6/A7/A8/A9 classes once
# they land, without importing their files. # INTEGRATION: bind concretes here.
# ===========================================================================
@runtime_checkable
class VerifierProto(Protocol):
    """A3 csc_lib/csc/v4/verifier.py :: CandidateVerifier(memory)."""

    def score(self, c: Candidate, motion_prior: Optional[tuple] = None) -> float:
        """Calibrated [0,1] identity/plausibility score for a candidate."""

    def verify(self, c: Candidate, margin: float = ...) -> bool:
        """True iff the candidate is a trustworthy (re-)localisation target."""


@runtime_checkable
class SPRTProto(Protocol):
    """A7 csc_lib/csc/v4/sprt_gate.py :: SPRTGate."""

    def update(self, llr: float) -> str:
        """Wald SPRT step -> {'accumulate','fire','clear'}."""

    def reset(self) -> None:
        ...


@runtime_checkable
class ModelProto(Protocol):
    """A6 csc_lib/csc/v4/model_v4.py :: CSCv4."""

    def predict(self, x, last_step_only: bool = True) -> V4Prediction:
        ...


@runtime_checkable
class RedetectorProto(Protocol):
    """A8 csc_lib/csc/v4/redetect.py :: MultiCropRedetector."""

    def maybe_redetect(
        self, frame, last_good, velocity, frame_idx: int
    ) -> Optional[list[Candidate]]:
        ...


@runtime_checkable
class SidecarProto(Protocol):
    """A9 csc_lib/csc/v4/avtrack_sidecar.py :: AVTrackSidecar."""

    def propose(self, frame, crops, template_hint) -> list[Candidate]:
        ...


# ===========================================================================
# Config — every threshold the controller / Meta-Updater / abort uses.
# Defaults mirror the measured V3 values (tools/run_with_csc.py arg defaults)
# so V4 starts from a known-sane operating point.
# ===========================================================================
@dataclass
class V4ControlConfig:
    # ---- SPRT / expected-gain action gate ---------------------------------
    # Minimum predicted ΔIoU (after subtracting per-action cost) for an acting
    # decision to clear the expected-gain gate. Below this => HOLD (do_not_act).
    min_expected_gain: float = 0.0
    # Per-action cost (subtracted from predicted ΔIoU in the expected-gain gate).
    # HOLD/FREEZE are free; budgeted GLOBAL_SEARCH is the most expensive.
    action_costs: dict[str, float] = field(default_factory=lambda: {
        ACTION_NAMES[int(Action.HOLD)]:            0.00,
        ACTION_NAMES[int(Action.MOTION_BRIDGE)]:   0.01,
        ACTION_NAMES[int(Action.RELOCATE)]:        0.02,
        ACTION_NAMES[int(Action.WIDEN)]:           0.01,
        ACTION_NAMES[int(Action.GLOBAL_SEARCH)]:   0.05,
        ACTION_NAMES[int(Action.TEMPLATE_UPDATE)]: 0.00,
        ACTION_NAMES[int(Action.FREEZE)]:          0.00,
    })
    # If the model emits do_not_act >= this, force HOLD regardless of triage.
    do_not_act_thresh: float = 0.5

    # ---- LA-subtype routing (when the model gives no LA-subtype head, fall
    #      back to these runtime-telemetry rules — mirrors _hard_la_gate). ----
    la_subtype_min_prob: float = 0.5   # accept argmax LA-subtype only if prob >= this
                                       # (M8: below this the head is ~uniform; fall
                                       # back to the telemetry rules instead).
    # FALSE-LA detection from telemetry (target actually fine => hold). Strong
    # discriminator on UAV123 LA frames: a HIGH search<->template cosine means the
    # tracker is still on the right content (false alarm). (gate_cosine=0.85 in V3.)
    false_la_cosine: float = 0.85
    # SMOOTH vs ABRUPT split: motion-residual ratio (residual / sqrt(area)).
    smooth_resid_ratio: float = 1.0
    # CANDIDATE availability: a secondary score-map peak this strong (ratio to
    # top-1) is a relocate candidate worth verifying (V3 relocate_min_ratio=0.30).
    candidate_min_ratio: float = 0.30

    # ---- motion_bridge ----------------------------------------------------
    bridge_max_frames: int = 30          # stop extrapolating after N lost frames
    bridge_vel_ema: float = 0.7          # EMA factor for the pre-loss velocity
    bridge_max_resid_ratio: float = 1e9  # only bridge if pre-loss motion regular
    bridge_max_disp: float = 0.0         # cap total extrapolated disp (0 = off)

    # ---- relocate / verify -----------------------------------------------
    # A verified candidate must beat the last-good appearance by this cosine
    # margin before we RELOCATE onto it (catastrophic-jump guard).
    relocate_verify_margin: float = 0.10
    # Distractor veto: reject a candidate whose distractor-similarity exceeds this.
    relocate_max_distractor_sim: float = 0.80

    # ---- WIDEN-search fallback factors ------------------------------------
    # GLOBAL_SEARCH widen factor grows with sustained LA (1.0 + 0.1*consec_la) but
    # is clamped here so a long loss cannot blow the search region up unboundedly
    # (M6: was 6.1x at consec_la=51).
    lost_widen_max: float = 1.5
    # FC-verified WIDEN fallback factor. consec_la is reset to 0 on FC so the LA
    # formula would yield 1.0 (no widen); FC widen uses this fixed factor (M7).
    fc_widen_factor: float = 1.5

    # ---- arming / persistence (episode state machine) ---------------------
    # Sustained gated-LA frames before MOTION_BRIDGE / WIDEN fire (V3 redetect_arm).
    redetect_arm_frames: int = 3
    # LA frames with no verified candidate before escalating to GLOBAL_SEARCH.
    persistent_la_frames: int = 8
    # Re-detect 2-frame vote: a global-search candidate must be corroborated for
    # this many consecutive frames before we relocate onto it.
    global_search_vote_frames: int = 2
    # Sustained gated-FC frames before an FC action fires (V3 fc_streak_frames=2).
    fc_streak_frames: int = 2

    # ---- abort window -----------------------------------------------------
    # Frames to watch telemetry after arming MOTION_BRIDGE/RELOCATE; if it has not
    # improved by then (abort_check), roll back to HOLD (V3 had no such rollback).
    abort_window_frames: int = 3
    # Per-signal deltas that count as "telemetry improved" (after vs before).
    abort_min_top2_drop: float = 0.02       # competing-peak ratio should fall
    abort_min_entropy_drop: float = 0.05    # response entropy should fall
    abort_min_cosine_gain: float = 0.02     # template<->search cosine should rise
    abort_min_targetness_gain: float = 0.0  # apce/peak_margin should rise
    abort_min_pla_drop: float = 0.0         # model p(LA) should fall
    # How many of the above signals must improve for the bridge to be "working".
    abort_min_signals: int = 2
    # After an abort rollback, suppress re-arming the same recovery action / hold
    # for this many frames so a harmful action cannot immediately re-fire and
    # oscillate bridge<->abort (M5). Decremented once per frame.
    abort_cooldown_frames: int = 5

    # ---- Meta-Updater (TEMPLATE_UPDATE gating) ----------------------------
    update_min_cc_prob: float = 0.60        # need confident CC to refresh template
    update_max_la_prob: float = 0.20        # ...and low p(LA)
    update_max_fc_prob: float = 0.20        # ...and low p(FC)
    update_min_identity: float = 0.60       # search must still match the target
    update_max_distractor_sim: float = 0.70 # ...and not match a known distractor
    update_min_targetness_apce: float = 0.0 # optional peakiness floor (0 = off)
    # Post-recovery window: after a recovery action, allow the template to refresh
    # for N frames to LOCK IN the re-acquisition (V3 recovery_update_window).
    post_recovery_update_frames: int = 5


# ===========================================================================
# Meta-Updater — the TEMPLATE_UPDATE / FREEZE decision (replaces V3's
# recovery_update_window + gated_freeze heuristics with a state-gated rule).
# ===========================================================================
class MetaUpdater:
    """Decides whether refreshing the template is *safe* this frame.

    Safe-update gate (all must hold): target present (confident CC, low p(LA)/p(FC)),
    identity match high (search still looks like the tracked target), distractor
    conflict low. A short post-recovery window after a recovery action lets the
    template re-acquire (the mechanism behind the V3 car9/car1_s lock-in).
    """

    def __init__(self, cfg: V4ControlConfig = V4ControlConfig()) -> None:
        self.cfg = cfg
        self._recovery_window = 0  # frames remaining in the post-recovery refresh

    def note_recovery_action(self) -> None:
        """Open the post-recovery refresh window (call when a recovery fires)."""
        self._recovery_window = int(self.cfg.post_recovery_update_frames)

    def reset(self) -> None:
        """Close the post-recovery window (call between episodes)."""
        self._recovery_window = 0

    def tick(self) -> None:
        """Advance the per-frame recovery-window countdown (call once per frame)."""
        if self._recovery_window > 0:
            self._recovery_window -= 1

    @property
    def in_recovery_window(self) -> bool:
        return self._recovery_window > 0

    def template_update_safe(self, pred: V4Prediction, tel: dict) -> bool:
        """True iff it is safe to refresh the template this frame.

        Trusts the model's ``template_update_safe`` head when present (>= 0.5), but
        also requires the state-gated conditions below so a mis-calibrated head
        cannot drift the template onto a distractor.
        """
        cfg = self.cfg
        probs = pred.derived_probs
        p_cc = _prob_at(probs, int(DerivedStateV4.CC))
        p_la = _prob_at(probs, int(DerivedStateV4.LA))
        p_fc = _prob_at(probs, int(DerivedStateV4.FC))

        # State gate: confident CC, low risk of LA/FC.
        if not (p_cc >= cfg.update_min_cc_prob
                and p_la <= cfg.update_max_la_prob
                and p_fc <= cfg.update_max_fc_prob):
            # Allowed exception: inside the post-recovery lock-in window we still
            # refresh (to re-acquire) — but only when NOT actively LA or FC. (H1a:
            # previously only re-checked p_fc, so a refresh could fire DURING a loss.)
            if not (self.in_recovery_window
                    and p_la <= cfg.update_max_la_prob
                    and p_fc <= cfg.update_max_fc_prob):
                return False

        # Identity gate: the current search must still match the target and NOT
        # match a known distractor. Use the strongest available identity signal.
        identity = _identity_signal(tel)
        if identity is not None and identity < cfg.update_min_identity:
            return False
        distractor = _f(tel.get("sim_to_distractor"))
        if distractor is not None and distractor > cfg.update_max_distractor_sim:
            return False

        # Optional targetness floor.
        if cfg.update_min_targetness_apce > 0.0:
            apce = _f(tel.get("apce"))
            if apce is not None and apce < cfg.update_min_targetness_apce:
                return False

        # Honour the model's own head when it explicitly votes against.
        if pred.template_update_safe_prob and pred.template_update_safe_prob < 0.5:
            return False
        return True


# ===========================================================================
# Abort check — is the recovery action actually working?
# ===========================================================================
def abort_check(
    before_tel: dict,
    after_tel: dict,
    cfg: V4ControlConfig = V4ControlConfig(),
) -> bool:
    """Return True => ABORT (roll back the action); False => keep going.

    A recovery is "working" when the telemetry moves the right way: targetness up
    (apce / sm_local_peak_margin), entropy down, competing-peak ratio (top2_ratio)
    down, template<->search cosine up, and model p(LA) down. We count how many of
    these improved by at least the configured delta; if fewer than
    ``abort_min_signals``, the action is not helping -> abort.
    """
    improved = 0

    top2_before = _f(before_tel.get("sm_local_top2_ratio"))
    top2_after = _f(after_tel.get("sm_local_top2_ratio"))
    if top2_before is not None and top2_after is not None:
        if (top2_before - top2_after) >= cfg.abort_min_top2_drop:
            improved += 1

    ent_before = _f(before_tel.get("response_entropy"))
    ent_after = _f(after_tel.get("response_entropy"))
    if ent_before is not None and ent_after is not None:
        if (ent_before - ent_after) >= cfg.abort_min_entropy_drop:
            improved += 1

    cos_before = _f(before_tel.get("last_cosine_sim"))
    cos_after = _f(after_tel.get("last_cosine_sim"))
    if cos_before is not None and cos_after is not None:
        if (cos_after - cos_before) >= cfg.abort_min_cosine_gain:
            improved += 1

    # Targetness: prefer apce, fall back to sm_local_peak_margin (both: higher=peakier).
    tgt_before = _f(before_tel.get("apce"))
    tgt_after = _f(after_tel.get("apce"))
    if tgt_before is None or tgt_after is None:
        tgt_before = _f(before_tel.get("sm_local_peak_margin"))
        tgt_after = _f(after_tel.get("sm_local_peak_margin"))
    if tgt_before is not None and tgt_after is not None:
        if (tgt_after - tgt_before) >= cfg.abort_min_targetness_gain:
            improved += 1

    # Model p(LA) should drop (passed through telemetry as 'p_la' when available).
    pla_before = _f(before_tel.get("p_la"))
    pla_after = _f(after_tel.get("p_la"))
    if pla_before is not None and pla_after is not None:
        if (pla_before - pla_after) >= cfg.abort_min_pla_drop:
            improved += 1

    return improved < cfg.abort_min_signals


# ===========================================================================
# The orchestrator
# ===========================================================================
class V4Controller:
    """LA/FC triage + evidence-gated action selection + abort + Meta-Updater.

    Parameters
    ----------
    model : ModelProto (A6 CSCv4) — used only when ``decide`` is called without a
        pre-computed ``pred`` (the runner usually passes ``pred`` directly).
    memory : A2 PrototypeMemory — appearance identity stores for verification.
    verifier : A3 CandidateVerifier — accepts/rejects relocate candidates.
    sprt : A7 SPRTGate — sequential-evidence gate for acting decisions.
    redetector : A8 MultiCropRedetector | None — budgeted global re-detect.
    sidecar : A9 AVTrackSidecar | None — independent re-detect proposals.
    cfg : V4ControlConfig.
    """

    def __init__(
        self,
        model: Optional[ModelProto],
        memory,
        verifier: VerifierProto,
        sprt: SPRTProto,
        redetector: Optional[RedetectorProto] = None,
        sidecar: Optional[SidecarProto] = None,
        cfg: V4ControlConfig = V4ControlConfig(),
    ) -> None:
        self.model = model
        self.memory = memory
        self.verifier = verifier
        self.sprt = sprt
        self.redetector = redetector
        self.sidecar = sidecar
        self.cfg = cfg
        self.meta = MetaUpdater(cfg)
        self.reset_episode()

    # ---- per-episode state machine ----------------------------------------
    def reset_episode(self) -> None:
        """Reset all per-sequence/episode state (call between sequences)."""
        self.consec_la = 0            # sustained gated-LA frames
        self.consec_fc = 0            # sustained gated-FC frames
        self.frames_since_loss = 0    # frames since the last confirmed position
        self.last_good_center: Optional[tuple[float, float]] = None
        self.last_good_size: Optional[tuple[float, float]] = None
        # Last-good appearance identity (sim_to_recent on the most recent CC frame),
        # the bar a relocate candidate must clear by a margin (L9).
        self._last_good_sim_recent: Optional[float] = None
        self.bridge_vel = (0.0, 0.0)  # EMA velocity from confirmed frames
        self.vel_resid = 1e9          # EMA velocity-residual (motion regularity)
        # Abort window: when a recovery action is armed we stash the pre-action
        # telemetry and count down; on expiry we run abort_check.
        self.abort_active = False
        self.abort_frames_left = 0
        self.abort_before_tel: Optional[dict] = None
        # Post-abort cooldown: suppress re-arming the same action for N frames after
        # a rollback so a harmful action cannot oscillate bridge<->abort (M5).
        self._abort_cooldown = 0
        self.last_action = int(Action.HOLD)
        # Global-search 2-frame vote: remember the last corroborated candidate.
        self._gs_vote = 0
        self._gs_last_center: Optional[tuple[float, float]] = None
        # Reset the Meta-Updater's post-recovery window so it does not leak across
        # sequences (L10).
        self.meta.reset()
        try:
            self.sprt.reset()
        except Exception:  # noqa: BLE001 - reset is best-effort
            pass

    # ---- main entry -------------------------------------------------------
    def decide(
        self,
        pred: Optional[V4Prediction],
        tel: dict,
        candidates: Optional[list[Candidate]],
        frame_ctx: dict,
    ) -> ActionDecision:
        """Pick this frame's control action.

        Parameters
        ----------
        pred : V4Prediction | None
            The model's per-frame diagnosis. If None and a model was supplied, it is
            computed from ``frame_ctx['features']`` (a (1,T,F) array/tensor).
        tel : dict
            Current-frame telemetry (V3 field names + optional ``sim_to_*``, ``p_la``).
        candidates : list[Candidate] | None
            Score-map candidates for this frame (from A3.extract_candidates upstream).
        frame_ctx : dict
            Loose runtime context. Recognised keys: ``frame`` (image), ``frame_idx``,
            ``bbox`` (current xywh), ``velocity_prior`` ((vx,vy)), ``features``.
        """
        cfg = self.cfg
        candidates = candidates or []
        frame_idx = int(frame_ctx.get("frame_idx", self.frames_since_loss))

        if pred is None:
            pred = self._infer(frame_ctx)
        ds = int(pred.derived_state)

        # Per-frame bookkeeping: refresh last-good + velocity on CC; advance the
        # Meta-Updater recovery window; tick down the post-abort cooldown (M5).
        self._update_motion_model(ds, frame_ctx)
        self.meta.tick()
        if self._abort_cooldown > 0:
            self._abort_cooldown -= 1

        # ---- abort handling first: if a recovery is armed, decide keep/rollback.
        abort_decision = self._maybe_abort(tel)
        if abort_decision is not None:
            return abort_decision

        # ---- triage ------------------------------------------------------
        if ds == int(DerivedStateV4.FC):
            decision = self._triage_fc(pred, tel, candidates, frame_ctx)
        elif ds == int(DerivedStateV4.LA):
            decision = self._triage_la(pred, tel, candidates, frame_ctx)
        else:  # CC / CU
            decision = self._triage_nominal(pred, tel)

        self.last_action = decision.action
        return decision

    # ======================================================================
    # Triage: nominal (CC / CU)
    # ======================================================================
    def _triage_nominal(self, pred: V4Prediction, tel: dict) -> ActionDecision:
        """No loss/FC suspected. Decay episode counters; offer TEMPLATE_UPDATE
        when the Meta-Updater says it is safe (gated by expected-gain), else HOLD."""
        self.consec_la = max(0, self.consec_la - 1)
        self.consec_fc = 0
        self._gs_vote = 0

        # L9: on a confident CC frame, remember the last-good appearance identity
        # (the strongest available "search still matches the target" signal). This is
        # the bar a relocate candidate must beat by ``relocate_verify_margin``.
        if int(pred.derived_state) == int(DerivedStateV4.CC):
            sim = _identity_signal(tel)
            if sim is not None:
                self._last_good_sim_recent = sim

        if self.meta.template_update_safe(pred, tel):
            act, gain, ev = self._gate_action(
                int(Action.TEMPLATE_UPDATE), pred, evidence_llr=0.0,
                require_sprt=False,  # template update is low-risk, gain-gated only
            )
            if act == int(Action.TEMPLATE_UPDATE):
                # NOTE (H1b): do NOT note_recovery_action() here — a plain healthy
                # CC frame must not (re-)arm the post-recovery refresh window. That
                # window is opened only by an ACTUAL recovery action (motion_bridge /
                # relocate / global_search). Otherwise the window stays permanently
                # armed and green-lights template updates during a subsequent loss.
                return ActionDecision(
                    action=int(Action.TEMPLATE_UPDATE),
                    reason="nominal:meta_updater_safe",
                    evidence=ev, expected_gain=gain,
                )
        return ActionDecision(
            action=int(Action.HOLD), reason="nominal:hold", evidence=0.0,
        )

    # ======================================================================
    # Triage: LA (lost-aware)
    # ======================================================================
    def _triage_la(
        self,
        pred: V4Prediction,
        tel: dict,
        candidates: list[Candidate],
        frame_ctx: dict,
    ) -> ActionDecision:
        cfg = self.cfg
        self.consec_la += 1
        self.consec_fc = 0   # M4: reset the FC streak — an LA frame breaks it, so the
                             # FC streak can only be satisfied by CONSECUTIVE FC frames.
        self.frames_since_loss += 1

        sub = self._la_subtype(pred, tel, candidates)

        # --- LA_FALSE: CSC over-fired, tracker is fine. Do-no-harm HOLD: no freeze,
        #     no bbox override. (The false-LA wall fix.)
        if sub == LASubtype.FALSE:
            self.consec_la = max(0, self.consec_la - 1)  # don't accumulate on a false alarm
            return ActionDecision(
                action=int(Action.HOLD), reason="la_false:hold_no_freeze", evidence=0.0,
            )

        # --- LA_OCCLUDED: target absent. HOLD + FREEZE (don't move search, don't
        #     update template onto background).
        if sub == LASubtype.OCCLUDED:
            return ActionDecision(
                action=int(Action.FREEZE), reason="la_occluded:hold_freeze", evidence=0.0,
                params={"freeze": True},
            )

        # --- LA_ABRUPT: motion bridge would overshoot a turn/stop. Hold (+ freeze
        #     so we don't learn the wrong content while we wait).
        if sub == LASubtype.ABRUPT:
            return ActionDecision(
                action=int(Action.FREEZE), reason="la_abrupt:hold_freeze", evidence=0.0,
                params={"freeze": True},
            )

        # --- LA_CANDIDATE: a secondary/global candidate exists. Verify top-k;
        #     RELOCATE only if a verified candidate beats last-good by a margin.
        if sub == LASubtype.CANDIDATE:
            relocate = self._verify_and_relocate(pred, tel, candidates, frame_ctx)
            if relocate is not None:
                return relocate
            # verified nothing -> fall through to persistence / bridge logic.

        # --- LA_SMOOTH: extrapolate pre-loss velocity (motion bridge) + arm abort.
        if sub == LASubtype.SMOOTH:
            bridge = self._motion_bridge(pred, tel, frame_ctx)
            if bridge is not None:
                return bridge
            # bridge not applicable (irregular / capped / no last-good): hold.

        # --- persistent unknown LA: escalate to budgeted GLOBAL_SEARCH (re-detect
        #     + verify, with a 2-frame vote).
        if self.consec_la >= cfg.persistent_la_frames:
            gs = self._global_search(pred, tel, frame_ctx)
            if gs is not None:
                return gs

        # default within LA: hold (search defaults, no thrash). FREEZE the template
        # so an un-diagnosed loss does not drift it (== V3 gated do-no-harm).
        return ActionDecision(
            action=int(Action.FREEZE), reason=f"la_{sub.name.lower()}:hold_freeze",
            evidence=0.0, params={"freeze": True},
        )

    # ======================================================================
    # Triage: FC (false-confirmed)
    # ======================================================================
    def _triage_fc(
        self,
        pred: V4Prediction,
        tel: dict,
        candidates: list[Candidate],
        frame_ctx: dict,
    ) -> ActionDecision:
        cfg = self.cfg
        self.consec_fc += 1
        self.consec_la = 0

        # Require a sustained FC streak before acting (V3 fc_streak_frames).
        if self.consec_fc < cfg.fc_streak_frames:
            return ActionDecision(
                action=int(Action.FREEZE), reason="fc:streak_building_freeze_only",
                evidence=0.0, params={"freeze": True},
            )

        # Is the FC *verified*? FC means a confident peak on the wrong thing. We
        # treat it as verified-FC when the locked content matches a known distractor
        # (FC_D) OR clearly fails identity to the target, AND the FC-subtype head /
        # telemetry agrees it is not pure occlusion.
        verified_fc = self._fc_verified(pred, tel)

        if not verified_fc:
            # Suspected-not-verified: FREEZE only. Do NOT move the bbox on a guess
            # (V3 fc_action=freeze_only / hold_lastgood beat block9/widen).
            return ActionDecision(
                action=int(Action.FREEZE), reason="fc_suspected:freeze_only",
                evidence=0.0, params={"freeze": True},
            )

        # Verified FC: FREEZE + reject the current bbox + search (last-good, else
        # global). Reuse the relocate/global machinery, but ALWAYS freeze + reject.
        relocate = self._verify_and_relocate(pred, tel, candidates, frame_ctx,
                                              reason_prefix="fc_verified")
        if relocate is not None:
            relocate.params["freeze"] = True
            relocate.params["reject_bbox"] = True
            return relocate

        gs = self._global_search(pred, tel, frame_ctx, reason_prefix="fc_verified",
                                  widen_factor=cfg.fc_widen_factor)  # M7
        if gs is not None:
            gs.params["freeze"] = True
            gs.params["reject_bbox"] = True
            return gs

        # Nothing better than last-good: hold the last-good center, freeze, reject.
        params = {"freeze": True, "reject_bbox": True}
        if self.last_good_center is not None:
            w, h = self.last_good_size or _bbox_wh(frame_ctx)
            params["cx"], params["cy"] = self.last_good_center
            params["w"], params["h"] = w, h
        return ActionDecision(
            action=int(Action.FREEZE), reason="fc_verified:freeze_reject_holdlastgood",
            evidence=0.0, params=params,
        )

    # ======================================================================
    # Action builders (each runs through the SPRT + expected-gain gate)
    # ======================================================================
    def _motion_bridge(
        self, pred: V4Prediction, tel: dict, frame_ctx: dict,
    ) -> Optional[ActionDecision]:
        """Extrapolate pre-loss velocity, capped; arm the abort window. Gated."""
        cfg = self.cfg
        if self._abort_cooldown > 0:
            return None  # M5: post-abort cooldown — don't re-arm the harmful action
        if self.last_good_center is None:
            return None
        if self.consec_la < cfg.redetect_arm_frames:
            return None  # not yet sustained — keep holding (== V3 redetect_arm)
        if self.frames_since_loss > cfg.bridge_max_frames:
            return None

        w, h = self.last_good_size or _bbox_wh(frame_ctx)
        scale = max(1.0, (w * h) ** 0.5)
        # Regularity gate: only extrapolate when pre-loss motion was smooth.
        if self.vel_resid / scale > cfg.bridge_max_resid_ratio:
            return None

        dx = self.bridge_vel[0] * self.frames_since_loss
        dy = self.bridge_vel[1] * self.frames_since_loss
        # Safety cap on total extrapolated displacement (V3 bridge_max_disp).
        if cfg.bridge_max_disp > 0:
            mag = (dx * dx + dy * dy) ** 0.5
            cap = cfg.bridge_max_disp * scale
            if mag > cap and mag > 1e-6:
                r = cap / mag
                dx *= r
                dy *= r
        cx = self.last_good_center[0] + dx
        cy = self.last_good_center[1] + dy

        act, gain, ev = self._gate_action(
            int(Action.MOTION_BRIDGE), pred,
            evidence_llr=self._evidence_llr(tel),
        )
        if act != int(Action.MOTION_BRIDGE):
            return None  # gate said HOLD / do_not_act

        self._arm_abort(tel)
        self.meta.note_recovery_action()
        return ActionDecision(
            action=int(Action.MOTION_BRIDGE),
            params={"cx": cx, "cy": cy, "w": w, "h": h, "freeze": True},
            reason="la_smooth:motion_bridge", evidence=ev, expected_gain=gain,
        )

    def _verify_and_relocate(
        self,
        pred: V4Prediction,
        tel: dict,
        candidates: list[Candidate],
        frame_ctx: dict,
        reason_prefix: str = "la_candidate",
    ) -> Optional[ActionDecision]:
        """Verify top-k candidates; RELOCATE only onto a verified one that beats
        last-good appearance by a margin (the catastrophic-jump guard)."""
        cfg = self.cfg
        if self._abort_cooldown > 0:
            return None  # M5: post-abort cooldown — suppress re-arming a recovery
        best = self._best_verified_candidate(candidates)
        if best is None:
            return None

        act, gain, ev = self._gate_action(
            int(Action.RELOCATE), pred, evidence_llr=self._evidence_llr(tel),
        )
        if act != int(Action.RELOCATE):
            return None

        self._arm_abort(tel)
        self.meta.note_recovery_action()
        return ActionDecision(
            action=int(Action.RELOCATE),
            params={"cx": best.cx, "cy": best.cy, "w": best.w, "h": best.h, "freeze": True},
            reason=f"{reason_prefix}:relocate_verified", evidence=ev, expected_gain=gain,
        )

    def _global_search(
        self,
        pred: V4Prediction,
        tel: dict,
        frame_ctx: dict,
        reason_prefix: str = "la_persistent",
        widen_factor: Optional[float] = None,
    ) -> Optional[ActionDecision]:
        """Budgeted re-detect (A8) + sidecar (A9) -> verify -> 2-frame vote ->
        RELOCATE. If the budget/vote is not satisfied this frame, WIDEN the search
        (cheap fallback) so the primary tracker can re-acquire on its own.

        ``widen_factor``: when given, use this fixed WIDEN-fallback factor instead of
        the consec_la-based growth (M7: the FC path passes ``cfg.fc_widen_factor``
        because ``consec_la`` has been reset to 0 on FC, which would give 1.0 = no
        widen)."""
        cfg = self.cfg
        if self._abort_cooldown > 0:
            return None  # M5: post-abort cooldown — suppress re-arming a recovery
        proposals: list[Candidate] = []

        if self.redetector is not None:
            rd = self.redetector.maybe_redetect(
                frame_ctx.get("frame"),
                self.last_good_center,
                frame_ctx.get("velocity_prior", self.bridge_vel),
                int(frame_ctx.get("frame_idx", self.frames_since_loss)),
            )
            if rd:
                proposals.extend(rd)
        if self.sidecar is not None and frame_ctx.get("frame") is not None:
            try:
                proposals.extend(self.sidecar.propose(
                    frame_ctx.get("frame"),
                    frame_ctx.get("crops"),
                    frame_ctx.get("template_hint"),
                ))
            except Exception:  # noqa: BLE001 - sidecar is best-effort  # INTEGRATION:
                pass

        best = self._best_verified_candidate(proposals)
        if best is None:
            # No verified re-detect this frame -> cheap WIDEN fallback (gated).
            act, gain, ev = self._gate_action(
                int(Action.WIDEN), pred, evidence_llr=self._evidence_llr(tel),
            )
            if act == int(Action.WIDEN):
                if widen_factor is not None:
                    # M7: FC path supplies a fixed factor (consec_la == 0 here).
                    factor = float(widen_factor)
                else:
                    # M6: clamp the growth so a long loss cannot blow the search region
                    # up unboundedly (was 6.1x at consec_la=51).
                    factor = min(cfg.lost_widen_max, 1.0 + 0.1 * self.consec_la)
                return ActionDecision(
                    action=int(Action.WIDEN),
                    params={"factor": factor, "freeze": True},
                    reason=f"{reason_prefix}:widen_no_candidate",
                    evidence=ev, expected_gain=gain,
                )
            return None

        # 2-frame vote: require the same region to be corroborated across frames
        # before relocating (a single spurious crop peak cannot hijack the track).
        if self._vote_corroborates(best.center):
            act, gain, ev = self._gate_action(
                int(Action.GLOBAL_SEARCH), pred, evidence_llr=self._evidence_llr(tel),
            )
            if act != int(Action.GLOBAL_SEARCH):
                return None
            self._arm_abort(tel)
            self.meta.note_recovery_action()
            self._gs_vote = 0
            return ActionDecision(
                action=int(Action.GLOBAL_SEARCH),
                params={"cx": best.cx, "cy": best.cy, "w": best.w, "h": best.h, "freeze": True},
                reason=f"{reason_prefix}:global_search_voted",
                evidence=ev, expected_gain=gain,
            )
        # vote not yet satisfied: hold this frame (freeze).
        return ActionDecision(
            action=int(Action.FREEZE), reason=f"{reason_prefix}:global_search_voting",
            evidence=0.0, params={"freeze": True},
        )

    # ======================================================================
    # SPRT + expected-gain gate
    # ======================================================================
    def _gate_action(
        self,
        proposed: int,
        pred: V4Prediction,
        evidence_llr: float,
        require_sprt: bool = True,
    ) -> tuple[int, float, float]:
        """Gate a proposed action through the SPRT (sequential evidence) AND the
        expected-gain check (predicted ΔIoU - cost). Returns (action, gain, evidence).

        On any gate failure the action collapses to HOLD with gain 0. This is the
        single choke-point every *acting* decision passes through.
        """
        cfg = self.cfg
        hold = int(Action.HOLD)

        # 0) Model do_not_act override.
        if pred.do_not_act_prob and pred.do_not_act_prob >= cfg.do_not_act_thresh:
            return hold, 0.0, 0.0

        # 1) Expected-gain on the PROPOSED action itself (H2/M3): the triage already
        #    chose which recovery to run; here we only veto it if its OWN net gain
        #    (predicted ΔIoU - cost) does not strictly clear min_expected_gain. We do
        #    NOT swap to the utility head's global argmax — callers reject any action
        #    other than the one they proposed, so swapping silently killed the chosen
        #    recovery (falling back to FREEZE) once the model was trained. With an
        #    untrained/zero utility head, _expected_gain() returns the neutral
        #    pass-through (== min_expected_gain), so the strict '>' is satisfied by the
        #    epsilon below and the triage choice stands.
        gain = self._expected_gain(proposed, pred)
        if pred.action_utility:
            # Strict '>' so a net-negative or exactly-break-even action (gain == cost)
            # is rejected (M3: with all-zero utility + min_gain=0, gain=0 must NOT pass).
            if not (gain > cfg.min_expected_gain):
                return hold, gain, 0.0

        # 2) SPRT sequential-evidence gate.
        evidence = 0.0
        if require_sprt:
            verdict = self.sprt.update(float(evidence_llr))
            evidence = float(evidence_llr)
            if verdict == "clear":
                return hold, gain, evidence
            if verdict != "fire":
                # 'accumulate': not enough sustained evidence yet -> hold this frame.
                return hold, gain, evidence
        return proposed, gain, evidence

    def _expected_gain(self, action: int, pred: V4Prediction) -> float:
        """Predicted ΔIoU for one action minus its cost (0 when no head)."""
        if not pred.action_utility:
            return self.cfg.min_expected_gain  # neutral pass-through (triage decides)
        name = ACTION_NAMES[action]
        util = float(pred.action_utility.get(name, 0.0))
        cost = float(self.cfg.action_costs.get(name, 0.0))
        return util - cost

    def _expected_gain_best(self, pred: V4Prediction) -> tuple[int, float]:
        """Argmax over actions of (predicted ΔIoU - cost)."""
        best_a, best_g = int(Action.HOLD), float("-inf")
        for a in Action:
            name = ACTION_NAMES[int(a)]
            util = float(pred.action_utility.get(name, 0.0))
            cost = float(self.cfg.action_costs.get(name, 0.0))
            g = util - cost
            if g > best_g:
                best_a, best_g = int(a), g
        return best_a, best_g

    # ======================================================================
    # Abort window
    # ======================================================================
    def _arm_abort(self, tel: dict) -> None:
        """Stash pre-action telemetry and open the abort countdown.

        Idempotent while a window is already open: we keep watching the ORIGINAL
        pre-action telemetry rather than re-stashing every armed frame (which would
        push the verdict out indefinitely while the action keeps re-firing)."""
        if self.abort_active:
            return
        self.abort_active = True
        self.abort_frames_left = int(self.cfg.abort_window_frames)
        self.abort_before_tel = dict(tel)

    def _maybe_abort(self, tel: dict) -> Optional[ActionDecision]:
        """If an abort window is open, count it down and, on expiry, run
        abort_check; abort -> rollback to HOLD (and clear the SPRT/episode arming)."""
        if not self.abort_active:
            return None
        self.abort_frames_left -= 1
        if self.abort_frames_left > 0:
            return None  # still watching
        # Window expired: judge whether the recovery worked.
        self.abort_active = False
        before = self.abort_before_tel or {}
        self.abort_before_tel = None
        if abort_check(before, tel, self.cfg):
            # Not improving -> roll back. Reset arming so we don't immediately re-fire,
            # AND open a cooldown so the same recovery cannot re-arm next frame and
            # oscillate bridge<->abort (M5).
            self.consec_la = 0
            self.frames_since_loss = 0
            self._gs_vote = 0
            self._abort_cooldown = int(self.cfg.abort_cooldown_frames)
            try:
                self.sprt.reset()
            except Exception:  # noqa: BLE001
                pass
            return ActionDecision(
                action=int(Action.HOLD), reason="abort:telemetry_not_improving",
                evidence=0.0, params={"rollback": True},
            )
        return None  # working — let normal triage continue this frame

    # ======================================================================
    # Subtype inference (model head first, telemetry fallback == _hard_la_gate)
    # ======================================================================
    def _la_subtype(
        self, pred: V4Prediction, tel: dict, candidates: list[Candidate],
    ) -> LASubtype:
        """Resolve the LA subtype. Trust the model's la_subtype head when present
        and confident; otherwise fall back to runtime telemetry rules that mirror
        ``tools/run_with_csc.py::_hard_la_gate`` (false-LA vs true-loss) + motion."""
        cfg = self.cfg
        probs = pred.la_subtype_probs
        if probs is not None and len(probs) == len(LASubtype):
            idx = int(np.argmax(probs))
            if float(probs[idx]) >= cfg.la_subtype_min_prob and idx != int(LASubtype.NONE):
                return LASubtype(idx)

        # ---- telemetry fallback ----
        # FALSE-LA: high search<->template cosine => tracker still on target.
        cosine = _f(tel.get("last_cosine_sim"))
        if cosine is not None and cosine > cfg.false_la_cosine:
            return LASubtype.FALSE
        # OCCLUDED: explicit occlusion / out-of-view flags in telemetry.
        if _truthy(tel.get("occlusion")) or _truthy(tel.get("out_of_view")) \
                or _truthy(tel.get("full_occlusion")):
            return LASubtype.OCCLUDED
        # CANDIDATE: a strong secondary peak exists.
        if self._best_candidate_ratio(candidates) >= cfg.candidate_min_ratio:
            return LASubtype.CANDIDATE
        # SMOOTH vs ABRUPT: motion-residual ratio (regularity of pre-loss motion).
        w, h = self.last_good_size or (1.0, 1.0)
        scale = max(1.0, (w * h) ** 0.5)
        if self.vel_resid / scale <= cfg.smooth_resid_ratio:
            return LASubtype.SMOOTH
        return LASubtype.ABRUPT

    def _fc_verified(self, pred: V4Prediction, tel: dict) -> bool:
        """Verified-FC = confident wrong lock with distractor identity (FC_D) or a
        clear identity-fail to the target, and not pure occlusion."""
        # Occlusion guard: an occluded frame is LA, not FC.
        if _truthy(tel.get("occlusion")) or _truthy(tel.get("out_of_view")) \
                or _truthy(tel.get("full_occlusion")):
            return False
        # Model FC-subtype head: DISTRACTOR/BACKGROUND => verified.
        probs = pred.fc_subtype_probs
        if probs is not None and len(probs) == len(FCSubtype):
            idx = int(np.argmax(probs))
            if idx in (int(FCSubtype.DISTRACTOR), int(FCSubtype.BACKGROUND)):
                return True
        # Telemetry fallback: locked content matches a known distractor, or clearly
        # fails identity to the target.
        distractor = _f(tel.get("sim_to_distractor"))
        if distractor is not None and distractor >= self.cfg.relocate_max_distractor_sim:
            return True
        identity = _identity_signal(tel)
        if identity is not None and identity < self.cfg.update_min_identity:
            return True
        return False

    # ======================================================================
    # Candidate verification helpers
    # ======================================================================
    def _best_verified_candidate(self, candidates: list[Candidate]) -> Optional[Candidate]:
        """Among candidates, pick the highest-scoring VERIFIED one that beats the
        last-good appearance by ``relocate_verify_margin`` and is not a distractor."""
        cfg = self.cfg
        if not candidates:
            return None
        last_good_sim = self._last_good_identity()
        best: Optional[Candidate] = None
        best_score = float("-inf")
        for c in candidates:
            # distractor veto
            if (c.sim_to_distractor == c.sim_to_distractor  # not nan
                    and c.sim_to_distractor > cfg.relocate_max_distractor_sim):
                continue
            try:
                ok = self.verifier.verify(c, margin=cfg.relocate_verify_margin)
            except TypeError:
                ok = self.verifier.verify(c)
            if not ok:
                continue
            # must beat last-good identity by margin (when we know last-good identity)
            cand_id = _candidate_identity(c)
            if last_good_sim is not None and cand_id is not None:
                if cand_id < last_good_sim + cfg.relocate_verify_margin:
                    continue
            score = self.verifier.score(c)
            if score > best_score:
                best, best_score = c, score
        return best

    def _last_good_identity(self) -> Optional[float]:
        """Identity of the last-good appearance (the bar a relocate candidate must
        clear by ``relocate_verify_margin``).

        Sourced from the most recent confident-CC frame's identity signal
        (``sim_to_recent`` / ``sim_to_init`` / ``last_cosine_sim``), captured in
        ``_triage_nominal`` (L9). Before any CC frame is seen this is None and the
        margin guard in ``_best_verified_candidate`` is skipped (verifier.verify()
        remains the safety net).

        # INTEGRATION: A2 PrototypeMemory can expose the recent-prototype self-anchor
        # similarity for a more precise, frame-exact bar; bind it here when available.
        """
        return self._last_good_sim_recent

    @staticmethod
    def _best_candidate_ratio(candidates: list[Candidate]) -> float:
        """Strongest secondary-peak score ratio (rank>=1) to the top peak."""
        if not candidates:
            return 0.0
        scores = [c.score for c in candidates if np.isfinite(c.score)]
        if not scores:
            return 0.0
        top = max(scores)
        if top <= 0:
            return 0.0
        sec = [c.score for c in candidates if c.rank >= 1 and np.isfinite(c.score)]
        return (max(sec) / top) if sec else 0.0

    def _vote_corroborates(self, center: tuple[float, float]) -> bool:
        """2-frame vote: True once the same region has been proposed for
        ``global_search_vote_frames`` consecutive frames."""
        cfg = self.cfg
        cx, cy = center
        if self._gs_last_center is not None:
            lx, ly = self._gs_last_center
            # "same region" = within one last-good box scale (fallback 1px).
            w, h = self.last_good_size or (1.0, 1.0)
            tol = max(1.0, 0.5 * (w * h) ** 0.5)
            if (abs(cx - lx) <= tol) and (abs(cy - ly) <= tol):
                self._gs_vote += 1
            else:
                self._gs_vote = 1
        else:
            self._gs_vote = 1
        self._gs_last_center = (cx, cy)
        return self._gs_vote >= cfg.global_search_vote_frames

    # ======================================================================
    # Motion model + misc
    # ======================================================================
    def _update_motion_model(self, ds: int, frame_ctx: dict) -> None:
        """On CC: trust the bbox center, refresh the EMA velocity + regularity, and
        reset frames_since_loss. (Mirrors the V3 motion_bridge bookkeeping.)"""
        bbox = frame_ctx.get("bbox")
        if bbox is None:
            return
        bx, by, bw, bh = (float(v) for v in bbox)
        cx_now, cy_now = bx + bw / 2.0, by + bh / 2.0
        if ds == int(DerivedStateV4.CC):
            if self.last_good_center is not None:
                ivx = cx_now - self.last_good_center[0]
                ivy = cy_now - self.last_good_center[1]
                a = self.cfg.bridge_vel_ema
                resid = ((ivx - self.bridge_vel[0]) ** 2 + (ivy - self.bridge_vel[1]) ** 2) ** 0.5
                self.vel_resid = resid if self.vel_resid >= 1e8 else (a * self.vel_resid + (1 - a) * resid)
                self.bridge_vel = (a * self.bridge_vel[0] + (1 - a) * ivx,
                                   a * self.bridge_vel[1] + (1 - a) * ivy)
            self.last_good_center = (cx_now, cy_now)
            self.last_good_size = (bw, bh)
            self.frames_since_loss = 0

    def _evidence_llr(self, tel: dict) -> float:
        """Calibrated log-likelihood-ratio proxy for the SPRT, from gate features.

        # INTEGRATION: A7 ``llr_from_evidence`` is the real implementation; this is a
        local fallback so the controller's SPRT still works standalone. Positive LLR
        = evidence FOR a true loss (the SPRT accumulates toward 'fire').
        """
        # Try the A7 module-level helper if it has been bound onto the gate.
        helper = getattr(self.sprt, "llr_from_evidence", None)
        if callable(helper):
            try:
                return float(helper(tel))
            except Exception:  # noqa: BLE001
                pass
        # Local proxy: sum of standardized votes (each ~ +1 toward true-loss).
        llr = 0.0
        top2r = _f(tel.get("sm_local_top2_ratio"))
        if top2r is not None:
            llr += (top2r - self.cfg.candidate_min_ratio) * 2.0
        entropy = _f(tel.get("response_entropy"))
        if entropy is not None:
            llr += (entropy - 4.0) * 0.5
        cosine = _f(tel.get("last_cosine_sim"))
        if cosine is not None:
            llr += (self.cfg.false_la_cosine - cosine) * 2.0
        return float(llr)

    def _infer(self, frame_ctx: dict) -> V4Prediction:
        """Run the model when no pred was supplied. # INTEGRATION: A6 CSCv4."""
        if self.model is None:
            raise ValueError("decide() called with pred=None and no model supplied")
        feats = frame_ctx.get("features")
        if feats is None:
            raise ValueError("frame_ctx['features'] required to run the model")
        return self.model.predict(feats, last_step_only=True)


# ===========================================================================
# small telemetry helpers
# ===========================================================================
def _f(x) -> Optional[float]:
    """Coerce to finite float or None (handles None / nan / non-numeric)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _truthy(x) -> bool:
    """Interpret an occlusion/oov flag (1/True/'1') as True; None/0 as False."""
    if x is None:
        return False
    try:
        return bool(int(x))
    except (TypeError, ValueError):
        return bool(x)


def _prob_at(probs: Optional[np.ndarray], idx: int) -> float:
    """Safe softmax-prob lookup; 0.0 when missing/out-of-range."""
    if probs is None:
        return 0.0
    try:
        if 0 <= idx < len(probs):
            return float(probs[idx])
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def _identity_signal(tel: dict) -> Optional[float]:
    """Strongest available "does the search still match the target" signal:
    prefer the memory-derived sim_to_recent/sim_to_init, else last_cosine_sim."""
    for key in ("sim_to_recent", "sim_to_init", "last_cosine_sim"):
        v = _f(tel.get(key))
        if v is not None:
            return v
    return None


def _candidate_identity(c: Candidate) -> Optional[float]:
    """Best identity signal for a candidate (recent preferred, then init)."""
    for v in (c.sim_to_recent, c.sim_to_init):
        if v is not None and v == v:  # not nan
            return float(v)
    return None


def _bbox_wh(frame_ctx: dict) -> tuple[float, float]:
    """(w, h) from frame_ctx['bbox'] xywh, or (1,1) when absent."""
    bbox = frame_ctx.get("bbox")
    if bbox is None:
        return (1.0, 1.0)
    return (float(bbox[2]), float(bbox[3]))


# ===========================================================================
# __main__ smoke — drive .decide across a synthetic LA episode with INLINE
# duck-typed stubs for A3 (verifier) and A7 (sprt). No model / tracker / data.
# ===========================================================================
def _mk_pred(
    derived: DerivedStateV4,
    la_sub: Optional[LASubtype] = None,
    fc_sub: Optional[FCSubtype] = None,
    action_utility: Optional[dict] = None,
    do_not_act: float = 0.0,
) -> V4Prediction:
    """Build a synthetic V4Prediction with a one-hot derived dist + optional heads."""
    d = np.zeros(4, dtype=np.float64)
    d[int(derived)] = 1.0
    la_probs = None
    if la_sub is not None:
        la_probs = np.zeros(len(LASubtype), dtype=np.float64)
        la_probs[int(la_sub)] = 1.0
    fc_probs = None
    if fc_sub is not None:
        fc_probs = np.zeros(len(FCSubtype), dtype=np.float64)
        fc_probs[int(fc_sub)] = 1.0
    return V4Prediction(
        derived_probs=d,
        derived_state=int(derived),
        la_subtype_probs=la_probs,
        fc_subtype_probs=fc_probs,
        action_utility=action_utility or {},
        do_not_act_prob=do_not_act,
    )


class _StubVerifier:
    """# INTEGRATION: stands in for A3 CandidateVerifier. Verifies by score>thr."""

    def __init__(self, thr: float = 0.5) -> None:
        self.thr = thr

    def score(self, c: Candidate, motion_prior=None) -> float:
        return float(c.score)

    def verify(self, c: Candidate, margin: float = 0.1) -> bool:
        return float(c.score) >= self.thr


class _StubSPRT:
    """# INTEGRATION: stands in for A7 SPRTGate. Fires once cumulative LLR>=2."""

    def __init__(self, fire_at: float = 2.0) -> None:
        self.fire_at = fire_at
        self.acc = 0.0

    def update(self, llr: float) -> str:
        self.acc += float(llr)
        if self.acc >= self.fire_at:
            return "fire"
        if self.acc <= -self.fire_at:
            self.acc = 0.0
            return "clear"
        return "accumulate"

    def reset(self) -> None:
        self.acc = 0.0


def _smoke() -> None:
    import sys

    cfg = V4ControlConfig(
        redetect_arm_frames=2,   # fire bridge quickly in the smoke
        abort_window_frames=2,
        bridge_max_disp=5.0,
    )
    ctrl = V4Controller(
        model=None,
        memory=None,               # memory optional for this smoke
        verifier=_StubVerifier(),  # # INTEGRATION: A3
        sprt=_StubSPRT(),          # # INTEGRATION: A7
        cfg=cfg,
    )

    log: list[tuple[int, str]] = []

    def step(pred, tel, cands, ctx, expect_action=None, note=""):
        dec = ctrl.decide(pred, tel, cands, ctx)
        log.append((dec.action, dec.reason))
        print(f"  f{ctx.get('frame_idx'):>2}  ds={pred.derived_state} "
              f"-> {ACTION_NAMES[dec.action]:<14} ev={dec.evidence:+.2f} "
              f"gain={dec.expected_gain:+.3f}  [{dec.reason}] {note}")
        if expect_action is not None:
            assert dec.action == int(expect_action), (
                f"frame {ctx.get('frame_idx')}: expected "
                f"{ACTION_NAMES[int(expect_action)]}, got {ACTION_NAMES[dec.action]} "
                f"({dec.reason})"
            )
        return dec

    print("V4Controller smoke — synthetic LA episode:")

    # --- 0..2: healthy CC, moving right at +6 px/frame. Establish last-good + vel.
    for i in range(3):
        step(
            _mk_pred(DerivedStateV4.CC),
            {"last_cosine_sim": 0.95, "sm_local_top2_ratio": 0.05,
             "response_entropy": 2.0, "apce": 120.0},
            [], {"frame_idx": i, "bbox": (100 + 6 * i, 50, 20, 20)},
            note="(healthy)",
        )
    assert ctrl.last_good_center is not None
    assert ctrl.bridge_vel[0] > 0.0, f"velocity should track rightward motion, got {ctrl.bridge_vel}"

    # --- 3: FALSE-LA. Telemetry says target is fine (high cosine) -> HOLD, NO freeze.
    d = step(
        _mk_pred(DerivedStateV4.LA),
        {"last_cosine_sim": 0.95, "sm_local_top2_ratio": 0.05, "response_entropy": 2.0},
        [], {"frame_idx": 3, "bbox": (118, 50, 20, 20)},
        expect_action=Action.HOLD, note="(false-LA)",
    )
    assert not d.params.get("freeze", False), "false-LA must NOT freeze"

    # --- 4..6: SMOOTH-LA. Loss with diffuse/low-cosine telemetry (the true-loss
    #     profile: competing peaks + low template<->search cosine => strong SPRT
    #     evidence), smooth pre-loss motion. After redetect_arm_frames=2 sustained
    #     LA frames AND the SPRT firing -> MOTION_BRIDGE.
    smooth_tel = {"last_cosine_sim": 0.20, "sm_local_top2_ratio": 0.50,
                  "response_entropy": 5.0}
    step(_mk_pred(DerivedStateV4.LA, LASubtype.SMOOTH), smooth_tel, [],
         {"frame_idx": 4, "bbox": (118, 50, 20, 20)},
         note="(smooth-LA, arming)")
    dbridge = step(
        _mk_pred(DerivedStateV4.LA, LASubtype.SMOOTH), smooth_tel, [],
        {"frame_idx": 5, "bbox": (118, 50, 20, 20)},
        expect_action=Action.MOTION_BRIDGE, note="(smooth-LA -> bridge)",
    )
    assert "cx" in dbridge.params and "cy" in dbridge.params, "bridge must emit a center"
    # displacement cap honoured: |center - last_good| <= bridge_max_disp * scale
    lgc = ctrl.last_good_center
    mag = ((dbridge.params["cx"] - lgc[0]) ** 2 + (dbridge.params["cy"] - lgc[1]) ** 2) ** 0.5
    assert mag <= cfg.bridge_max_disp * 20.0 + 1e-6, f"disp cap violated: {mag}"
    assert ctrl.abort_active, "bridge must arm the abort window"

    # --- 6..7: abort window expires with NO telemetry improvement -> ABORT/rollback.
    #     (Telemetry is constant frame-to-frame => abort_check sees 0 improved
    #     signals < abort_min_signals=2, so the recovery is judged not-working.)
    step(_mk_pred(DerivedStateV4.LA, LASubtype.SMOOTH), smooth_tel, [],
         {"frame_idx": 6, "bbox": (118, 50, 20, 20)}, note="(abort window -1)")
    dab = step(
        _mk_pred(DerivedStateV4.LA, LASubtype.SMOOTH), smooth_tel, [],
        {"frame_idx": 7, "bbox": (118, 50, 20, 20)},
        expect_action=Action.HOLD, note="(abort: no improvement)",
    )
    assert dab.params.get("rollback", False), "expired non-improving window must roll back"
    assert not ctrl.abort_active and ctrl.consec_la == 0, "abort must clear arming"

    # --- 8: OCCLUDED-LA -> HOLD + FREEZE.
    step(
        _mk_pred(DerivedStateV4.LA, LASubtype.OCCLUDED),
        {"occlusion": 1, "last_cosine_sim": 0.20},
        [], {"frame_idx": 8, "bbox": (118, 50, 20, 20)},
        expect_action=Action.FREEZE, note="(occluded-LA -> freeze)",
    )

    # --- CANDIDATE-LA on a FRESH episode (the prior abort left an M5 post-abort
    #     cooldown active, which correctly suppresses re-arming a recovery; reset so
    #     we exercise the relocate path in isolation). A strong verified secondary
    #     peak whose identity beats the last-good appearance by the margin -> RELOCATE.
    ctrl.reset_episode()
    # Seed last-good center + a MODERATE last-good identity (0.60) so the L9 margin
    # guard is live but the candidate (sim 0.9) can clear 0.60 + 0.10 = 0.70.
    ctrl._update_motion_model(int(DerivedStateV4.CC), {"bbox": (118, 50, 20, 20)})
    ctrl._triage_nominal(_mk_pred(DerivedStateV4.CC), {"last_cosine_sim": 0.60})
    assert ctrl._last_good_sim_recent == 0.60, "L9: CC frame must record last-good identity"
    cand = Candidate(cx=400.0, cy=300.0, w=20.0, h=20.0, score=0.95, rank=1,
                     sim_to_recent=0.9, sim_to_distractor=0.1)
    # arm via two LA frames so SPRT has accumulated evidence
    step(_mk_pred(DerivedStateV4.LA, LASubtype.CANDIDATE),
         {"last_cosine_sim": 0.25, "sm_local_top2_ratio": 0.6, "response_entropy": 5.0},
         [cand], {"frame_idx": 9, "bbox": (118, 50, 20, 20)}, note="(candidate-LA arming)")
    drel = step(
        _mk_pred(DerivedStateV4.LA, LASubtype.CANDIDATE),
        {"last_cosine_sim": 0.25, "sm_local_top2_ratio": 0.6, "response_entropy": 5.0},
        [cand], {"frame_idx": 10, "bbox": (118, 50, 20, 20)},
        expect_action=Action.RELOCATE, note="(candidate-LA -> relocate)",
    )
    assert drel.params["cx"] == 400.0, "relocate must jump to the verified candidate"

    # --- L9 catastrophic-jump guard: a candidate that does NOT beat last-good by the
    #     margin must be REJECTED (relocate suppressed -> fall through to FREEZE).
    ctrl.reset_episode()
    ctrl._update_motion_model(int(DerivedStateV4.CC), {"bbox": (118, 50, 20, 20)})
    ctrl._triage_nominal(_mk_pred(DerivedStateV4.CC), {"last_cosine_sim": 0.95})
    weak = Candidate(cx=400.0, cy=300.0, w=20.0, h=20.0, score=0.95, rank=1,
                     sim_to_recent=0.90, sim_to_distractor=0.1)  # 0.90 < 0.95 + 0.10
    step(_mk_pred(DerivedStateV4.LA, LASubtype.CANDIDATE),
         {"last_cosine_sim": 0.25, "sm_local_top2_ratio": 0.6, "response_entropy": 5.0},
         [weak], {"frame_idx": 9, "bbox": (118, 50, 20, 20)}, note="(weak-candidate arming)")
    dweak = step(
        _mk_pred(DerivedStateV4.LA, LASubtype.CANDIDATE),
        {"last_cosine_sim": 0.25, "sm_local_top2_ratio": 0.6, "response_entropy": 5.0},
        [weak], {"frame_idx": 10, "bbox": (118, 50, 20, 20)},
        expect_action=Action.FREEZE, note="(weak candidate -> rejected, freeze)",
    )
    assert "cx" not in dweak.params, "rejected candidate must NOT produce a relocate jump"

    # --- FC path on a fresh episode: suspected (1 frame) -> FREEZE only;
    #     verified distractor (>=streak) -> FREEZE + reject + (relocate/hold).
    ctrl.reset_episode()
    ctrl._update_motion_model(int(DerivedStateV4.CC),
                              {"bbox": (200, 200, 30, 30)})  # seed last-good
    step(
        _mk_pred(DerivedStateV4.FC),
        {"sim_to_distractor": 0.95, "last_cosine_sim": 0.2},
        [], {"frame_idx": 0, "bbox": (200, 200, 30, 30)},
        expect_action=Action.FREEZE, note="(FC streak building -> freeze only)",
    )
    dfc = step(
        _mk_pred(DerivedStateV4.FC, fc_sub=FCSubtype.DISTRACTOR),
        {"sim_to_distractor": 0.95, "last_cosine_sim": 0.2},
        [], {"frame_idx": 1, "bbox": (200, 200, 30, 30)},
        expect_action=Action.FREEZE, note="(FC verified -> freeze+reject+holdlastgood)",
    )
    assert dfc.params.get("freeze") and dfc.params.get("reject_bbox"), \
        "verified FC must freeze AND reject the bbox"

    # --- do_not_act override: even on smooth-LA, do_not_act>=thr forces HOLD.
    ctrl.reset_episode()
    ctrl._update_motion_model(int(DerivedStateV4.CC), {"bbox": (0, 0, 10, 10)})
    ctrl.consec_la = cfg.redetect_arm_frames  # would otherwise bridge
    ctrl.last_good_center = (5.0, 5.0); ctrl.last_good_size = (10.0, 10.0)
    ctrl.frames_since_loss = 1
    ddna = ctrl.decide(
        _mk_pred(DerivedStateV4.LA, LASubtype.SMOOTH, do_not_act=0.9),
        {"last_cosine_sim": 0.3}, [], {"frame_idx": 3, "bbox": (0, 0, 10, 10)},
    )
    assert ddna.action == int(Action.FREEZE) or ddna.action == int(Action.HOLD), \
        f"do_not_act should suppress the bridge, got {ACTION_NAMES[ddna.action]}"

    # --- MetaUpdater: safe on confident CC, unsafe under FC risk.
    mu = MetaUpdater(cfg)
    assert mu.template_update_safe(
        _mk_pred(DerivedStateV4.CC),
        {"sim_to_recent": 0.9, "sim_to_distractor": 0.1},
    ), "should allow update on healthy CC"
    assert not mu.template_update_safe(
        _mk_pred(DerivedStateV4.FC),
        {"sim_to_recent": 0.2, "sim_to_distractor": 0.95},
    ), "should block update under FC + distractor conflict"

    # --- abort_check unit: improving telemetry => keep (False).
    assert abort_check(
        {"sm_local_top2_ratio": 0.6, "response_entropy": 5.0, "last_cosine_sim": 0.2},
        {"sm_local_top2_ratio": 0.1, "response_entropy": 3.0, "last_cosine_sim": 0.9},
        cfg,
    ) is False, "clearly-improving telemetry must NOT abort"

    print("\nV4Controller smoke OK: false-LA HOLD (no freeze), smooth-LA MOTION_BRIDGE "
          "(capped, abort armed), ABORT on no-improvement, occluded FREEZE, candidate "
          "RELOCATE, FC freeze-only->verified-reject, do_not_act override, MetaUpdater "
          "+ abort_check units.")
    sys.exit(0)


if __name__ == "__main__":
    # The la_smoke-style sys.path header (salrtd/src, src, repo root) is applied at
    # module top under `if not __package__` (L13) so the absolute csc_lib import
    # resolves when this file is run standalone as well as via `-m`.
    _smoke()
