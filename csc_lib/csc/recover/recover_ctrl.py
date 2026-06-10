"""FC detect-verify-recover controller — wraps PrototypeMemory + CandidateVerifier + SPRT.

Why a separate controller (vs extending csc_lib/csc/fc_challenge.py)?
---------------------------------------------------------------------
The V3 ``FCChallengeController`` uses a fixed ``confirm_frames`` counter and
verifies a candidate against the *incumbent's* identity (sim_to_init delta),
which is anti-discriminative for true-FC (the confident distractor scores
*higher* sim_to_init than the real object). This module replaces both pieces:

* identity verification uses :class:`csc_lib.csc.v4.memory.PrototypeMemory`
  (frame-0 anchor + EMA recent CC + bounded distractor ring) consumed by
  :class:`csc_lib.csc.v4.verifier.CandidateVerifier` — so a candidate that
  *looks like a known FC distractor* is hard-vetoed and a candidate that
  identity-matches the recent confirmed track is preferred even when the
  incumbent's sim_to_init is higher (saturated identity);
* temporal verification uses :class:`csc_lib.csc.v4.sprt_gate.SPRTGate` —
  per-frame log-likelihood ratios accumulate, switch fires only after
  sustained evidence crosses Wald's bound, single-frame anomalies do not
  fire.

V3 (csc_prod) is otherwise untouched. The module is pure-logic: the
re-detect forward pass and the per-frame CC embedding source are injected as
callables so this file is testable without a tracker.

State machine (one instance per sequence, ``reset()`` between sequences)::

    IDLE ── fc_trigger ──► CHALLENGE ──► (SPRT 'fire')   ──► ABORT_WINDOW
                                  │                                 │
                                  └─ timeout / SPRT 'clear' ──► IDLE
                                                                    │
                                                  rollback / commit ┘
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from csc_lib.csc.v4.memory import PrototypeMemory
from csc_lib.csc.v4.sprt_gate import SPRTGate
from csc_lib.csc.v4.v4types import Candidate
from csc_lib.csc.v4.verifier import CandidateVerifier, VerifierConfig


# Derived-state ints (match V3 inference: 0=CC, 1=CU, 2=LA, 3=FC).
CC, CU, LA, FC = 0, 1, 2, 3

# Phase strings.
IDLE = "idle"
CHALLENGE = "challenge"
ABORT_WINDOW = "abort_window"


# Re-detect callback contract: zero-arg, returns a list[dict] of candidate
# dicts (the SGLATracker.redetect format with top_k>1) plus latency in ms.
# Each dict must have keys: bbox=[x,y,w,h], center=[cx,cy], score, sim_to_init,
# and (preferred, when available) embedding=np.ndarray(192,) for memory.sims.
RedetectFn = Callable[[], "tuple[list[dict], float]"]


@dataclass
class FCRecoverConfig:
    """Tunables for the recover controller. Conservative defaults."""

    # ---- FC trigger (composes with the runner's FC streak/precision gate) ----
    fc_streak_required: int = 2
    """Independent of the runner gate: how many consecutive FC frames are needed
    to *enter* the CHALLENGE phase. Set the runner gate higher to be stricter."""

    # ---- top-K candidate generation ----
    redetect_top_k: int = 5
    """How many spatially-distinct candidates to ask the tracker for per call."""

    # ---- temporal verification (SPRT) ----
    sprt_alpha: float = 0.05
    """Target false-switch probability P[fire | H0]. Smaller -> harder to fire."""

    sprt_beta: float = 0.10
    """Target miss probability P[clear | H1]. Smaller -> harder to clear."""

    sprt_max_evidence: Optional[float] = None
    """Clamp half-width beyond [B,A] (None = SPRTGate default = 2*(A-B))."""

    sprt_false_alert_budget: int = 3
    """Max successful 'fire' switches per sequence (catastrophic-failure cap)."""

    llr_clip: float = 4.0
    """Per-frame LLR symmetric clip — one extreme frame can't single-handedly fire."""

    # ---- memory ----
    max_recent: int = 5
    max_distractor: int = 8
    recent_ema: float = 0.7

    # ---- distractor seeding ----
    seed_distractor_on_streak: bool = True
    """On FC streak entry, add the incumbent's peak-local search embedding to
    the distractor ring so future re-detections cannot relock the same wrong
    object. The single highest-leverage signal in the V4 design."""

    # ---- recent-CC seeding (memory.update_recent) ----
    update_recent_min_confidence: float = 0.55
    """Only fold a CC frame into the recent prototype if confidence >= this."""

    # ---- verifier (forwarded to CandidateVerifier) ----
    verifier_accept_margin: float = 0.55
    verifier_distractor_veto: float = 0.85
    verifier_min_identity: float = 0.35
    verifier_min_peak_margin: float = 0.10

    # ---- challenge bounds ----
    challenge_max_frames: int = 15
    """Hard cap on CHALLENGE duration (bounds the read-only redetect cost)."""

    early_abort_clear_frames: int = 2
    """Abort early once derived_state has been non-FC for this many frames AND
    the SPRT accumulator has not crossed the upper bound."""

    # ---- abort window (post-switch monitor) ----
    abort_window: int = 5
    """Monitor the switched track for this many frames; rollback if state
    becomes LA/FC for ``rollback_la_fc_streak`` consecutive frames, else
    commit (and release the freeze)."""

    rollback_on_failure_state: bool = True

    rollback_la_fc_streak: int = 2
    """Consecutive LA/FC frames inside the abort window required to trigger
    a rollback. The first 1-2 frames after a switch often show LA while the
    tracker re-locks at the new search centre — that single transient is
    not evidence the switch was wrong. With streak=2 we tolerate one
    settling frame; raise to 3 to be even more lenient."""

    # ---- motion prior (rejects far-away wrong-target candidates) ----
    motion_max_disp: float = 0.0
    """Maximum displacement (pixels) of a candidate from the last-CC center.
    A candidate further than this is hard-rejected by the verifier's motion
    gate. 0 = disabled (no motion prior). Defaults to 0 to preserve current
    behaviour; setting ~200-400 protects against catastrophic ``car7`` style
    wrong-object switches where the candidate is a different car far from
    the true target trajectory. Total displacement allowed scales with
    ``max(motion_max_disp, motion_max_disp_per_frame * frames_since_cc)``."""

    motion_max_disp_per_frame: float = 50.0
    """Velocity-style cap added to the static motion gate: a target can only
    travel this far per frame from the last-CC position. Combined with
    ``motion_max_disp`` so short-loss recoveries get a tight gate (mostly
    static) and long-loss recoveries get a proportionally wider gate."""

    motion_max_scale_ratio: float = 2.5
    """Maximum scale ratio (candidate vs last-CC bbox) before the verifier
    rejects on plausibility. Default 2.5 = either dim cannot grow/shrink by
    more than 2.5x from the last-CC size."""


@dataclass
class FCRecoverDecision:
    """Per-frame controller output. The runner applies these levers causally."""

    phase: str = IDLE
    ran_redetect: bool = False
    redetect_ms: float = 0.0
    n_candidates: int = 0
    n_verified: int = 0
    freeze_template: bool = False
    switch_center: Optional[tuple] = None      # (cx, cy, w, h)
    rollback_center: Optional[tuple] = None    # (cx, cy, w, h)
    started: bool = False
    committed: bool = False
    aborted: bool = False
    sprt_evidence: float = 0.0
    sprt_decision: str = "accumulate"          # 'accumulate'|'fire'|'clear'|'idle'
    cand_score: float = float("nan")
    cand_sim_init: float = float("nan")
    cand_sim_recent: float = float("nan")
    cand_sim_distractor: float = float("nan")
    distractor_seeded: bool = False
    reason: str = ""


@dataclass
class FCRecoverController:
    """Detect-verify-recover state machine. One instance per sequence.

    Wire-up (per frame, in the runner)::

        # On every frame, BEFORE tracker.update():
        ctrl.maybe_seed_anchor(tracker._initial_template_embedding)

        # AFTER tracker.update(), if a CC frame:
        if derived_state == CC and confidence >= cfg.update_recent_min_confidence:
            ctrl.note_cc(tracker._last_search_peak_local, frame_idx, bbox)

        # Once per frame, drive the state machine:
        decision = ctrl.step(
            derived_state=...,
            fc_trigger=...,                   # runner's FC streak/gate
            incumbent_bbox=...,
            incumbent_emb=tracker._last_search_peak_local,  # for distractor seed
            redetect_fn=lambda: tracker.redetect(top_k=K), ...
        )
        # Apply decision.switch_center / rollback_center / freeze_template.
    """

    config: FCRecoverConfig = field(default_factory=FCRecoverConfig)

    memory: PrototypeMemory = field(init=False)
    verifier: CandidateVerifier = field(init=False)
    sprt: SPRTGate = field(init=False)

    phase: str = IDLE
    challenge_frames: int = 0
    non_fc_frames: int = 0
    incumbent_snapshot: Optional[tuple] = None      # (cx, cy, w, h) rollback fallback
    last_best_candidate: Optional[Candidate] = None  # most-recent verified candidate
    abort_remaining: int = 0
    abort_la_fc_streak: int = 0                      # consecutive LA/FC frames in abort window
    last_cc_bbox: Optional[tuple] = None
    last_cc_frame_idx: int = -1                      # frame_idx of last note_cc call

    # Sequence counters (surfaced to metrics).
    n_challenges: int = 0
    n_switches: int = 0
    n_commits: int = 0
    n_rollbacks: int = 0
    n_aborts: int = 0
    n_redetect_calls: int = 0
    n_verified_total: int = 0
    n_distractor_seeds: int = 0

    def __post_init__(self) -> None:
        cfg = self.config
        self.memory = PrototypeMemory(
            max_recent=cfg.max_recent,
            max_distractor=cfg.max_distractor,
            ema=cfg.recent_ema,
        )
        vcfg = VerifierConfig(
            accept_margin=cfg.verifier_accept_margin,
            distractor_veto=cfg.verifier_distractor_veto,
            min_identity=cfg.verifier_min_identity,
            min_peak_margin=cfg.verifier_min_peak_margin,
        )
        self.verifier = CandidateVerifier(self.memory, vcfg)
        self.sprt = SPRTGate(
            alpha=cfg.sprt_alpha,
            beta=cfg.sprt_beta,
            max_evidence=cfg.sprt_max_evidence,
            false_alert_budget=cfg.sprt_false_alert_budget,
        )

    def reset(self) -> None:
        """Reset all per-sequence state (call between sequences)."""
        self.memory.reset()
        self.sprt.reset()
        self.sprt.reset_budget()
        self.phase = IDLE
        self.challenge_frames = 0
        self.non_fc_frames = 0
        self.incumbent_snapshot = None
        self.last_best_candidate = None
        self.abort_remaining = 0
        self.abort_la_fc_streak = 0
        self.last_cc_bbox = None
        self.last_cc_frame_idx = -1
        self.n_challenges = 0
        self.n_switches = 0
        self.n_commits = 0
        self.n_rollbacks = 0
        self.n_aborts = 0
        self.n_redetect_calls = 0
        self.n_verified_total = 0
        self.n_distractor_seeds = 0

    @property
    def active(self) -> bool:
        return self.phase != IDLE

    # -- memory hooks ----------------------------------------------------------

    def maybe_seed_anchor(self, embedding) -> None:
        """Set the frame-0 anchor (idempotent; only the first valid call sticks)."""
        if embedding is not None:
            self.memory.update_anchor(embedding)

    def note_cc(self, embedding, frame_idx: int, bbox: Optional[tuple] = None) -> None:
        """Record a CC frame's appearance in the recent prototype + last_cc_bbox."""
        if embedding is not None:
            self.memory.update_recent(embedding, frame_idx=int(frame_idx))
        self.last_cc_frame_idx = int(frame_idx)
        if bbox is not None:
            self.last_cc_bbox = (
                float(bbox[0]) + float(bbox[2]) / 2.0,
                float(bbox[1]) + float(bbox[3]) / 2.0,
                float(bbox[2]),
                float(bbox[3]),
            )

    # -- core step -------------------------------------------------------------

    def step(
        self,
        *,
        derived_state: int,
        fc_trigger: bool,
        incumbent_bbox: tuple,
        incumbent_emb=None,
        redetect_fn: Optional[RedetectFn] = None,
        frame_idx: int = -1,
    ) -> FCRecoverDecision:
        """Advance the state machine by one frame.

        Parameters
        ----------
        derived_state
            CSC derived state int (0=CC, 1=CU, 2=LA, 3=FC).
        fc_trigger
            Runner-computed gate (FC streak + precision gate). Only a trigger
            starts a CHALLENGE; raw FC argmax is too noisy.
        incumbent_bbox
            Current ``(x, y, w, h)`` from ``tracker.update()``.
        incumbent_emb
            Optional peak-local search embedding for THIS frame (192-D). Used
            once on CHALLENGE start to seed the distractor memory (the
            incumbent at FC trigger is presumed to be the wrong object).
        redetect_fn
            Zero-arg callback that runs the tracker's read-only re-detect and
            returns ``(list_of_candidate_dicts, latency_ms)``. Each dict needs
            'bbox', 'center', 'score', 'sim_to_init', and (preferred)
            'embedding' (np.ndarray(C,)).
        frame_idx
            Frame index used for memory time-stamps.
        """
        cfg = self.config
        bx, by, bw, bh = incumbent_bbox
        inc_center = (bx + bw / 2.0, by + bh / 2.0)
        inc_size = (bw, bh)
        started_now = False

        # ---- IDLE: watch for an FC trigger ------------------------------------
        if self.phase == IDLE:
            if not fc_trigger:
                return FCRecoverDecision(phase=IDLE, reason="idle")

            # Enter CHALLENGE. Snapshot the incumbent (rollback) and reset SPRT
            # for this episode.
            self.phase = CHALLENGE
            self.challenge_frames = 0
            self.non_fc_frames = 0
            anchor_center = self.last_cc_bbox if self.last_cc_bbox is not None else (
                inc_center[0], inc_center[1], inc_size[0], inc_size[1])
            self.incumbent_snapshot = anchor_center
            self.sprt.reset()
            self.n_challenges += 1
            started_now = True

            # Distractor seeding: the FC-triggering frame's incumbent embedding
            # is presumed wrong-locked. Add it to the distractor ring so future
            # candidates that match it are vetoed.
            seeded = False
            if cfg.seed_distractor_on_streak and incumbent_emb is not None:
                self.memory.add_distractor(incumbent_emb, frame_idx=int(frame_idx))
                self.n_distractor_seeds += 1
                seeded = True

            # Fall through to CHALLENGE handling on this same frame.
            decision_kwargs = dict(started=True, distractor_seeded=seeded)
        else:
            decision_kwargs = dict()

        # ---- CHALLENGE: read-only redetect + verify + SPRT --------------------
        if self.phase == CHALLENGE:
            self.challenge_frames += 1
            self.non_fc_frames = 0 if derived_state == FC else self.non_fc_frames + 1

            cands_raw, redetect_ms = ([], 0.0)
            if redetect_fn is not None:
                try:
                    cands_raw, redetect_ms = redetect_fn()
                except Exception:
                    cands_raw, redetect_ms = [], 0.0
            self.n_redetect_calls += 1

            # Convert raw dicts -> Candidate dataclass for the verifier. Required
            # keys: 'bbox', 'center', 'score'. Optional: 'sim_to_init',
            # 'score_ratio', 'embedding'.
            cands = self._dicts_to_candidates(cands_raw or [])

            # Build the motion prior the verifier uses to reject far-away
            # candidates. Disabled when motion_max_disp == 0 (default in legacy
            # configs) — caller should set the disp gate when wiring v1.1+.
            motion_prior = self._build_motion_prior(frame_idx)
            best = self.verifier.best_verified(cands, motion_prior=motion_prior)
            if best is not None:
                self.n_verified_total += 1
                self.last_best_candidate = best

            # Per-frame LLR for SPRT. With a verified candidate, the centred
            # score (>= 0.5 means "evidence to switch") drives positive LLR
            # scaled to a reasonable per-frame magnitude. Without a verified
            # candidate, we accumulate slight negative evidence (don't switch).
            if best is not None:
                s = float(self.verifier.score(best, motion_prior=motion_prior))
                # accept_margin is the conservative threshold; map score above
                # accept_margin to positive LLR, below to negative.
                llr = (s - cfg.verifier_accept_margin) * 4.0
            else:
                llr = -0.5
            llr = float(max(-cfg.llr_clip, min(cfg.llr_clip, llr)))
            decision = self.sprt.update(llr)
            evidence = float(self.sprt.evidence)

            cand_score = float("nan") if best is None else float(self.verifier.score(best, motion_prior=motion_prior))
            cand_si = float("nan") if best is None else float(best.sim_to_init)
            cand_sr = float("nan") if best is None else float(best.sim_to_recent)
            cand_sd = float("nan") if best is None else float(best.sim_to_distractor)

            if decision == "fire" and best is not None:
                # Commit switch: enter ABORT_WINDOW, return the new search center.
                self.phase = ABORT_WINDOW
                self.abort_remaining = cfg.abort_window
                self.abort_la_fc_streak = 0
                self.n_switches += 1
                return FCRecoverDecision(
                    phase=ABORT_WINDOW,
                    ran_redetect=True,
                    redetect_ms=redetect_ms,
                    n_candidates=len(cands),
                    n_verified=1 if best is not None else 0,
                    freeze_template=True,
                    switch_center=(best.cx, best.cy, best.w, best.h),
                    sprt_evidence=evidence,
                    sprt_decision=decision,
                    cand_score=cand_score,
                    cand_sim_init=cand_si,
                    cand_sim_recent=cand_sr,
                    cand_sim_distractor=cand_sd,
                    reason="switch_sprt_fire",
                    **decision_kwargs,
                )

            # No fire this frame. Decide whether to continue challenging.
            timed_out = self.challenge_frames >= cfg.challenge_max_frames
            cleared = (
                decision == "clear"
                or (self.non_fc_frames >= cfg.early_abort_clear_frames and best is None)
            )
            if timed_out or cleared:
                self.phase = IDLE
                self.n_aborts += 1
                self.last_best_candidate = None
                return FCRecoverDecision(
                    phase=IDLE,
                    ran_redetect=True,
                    redetect_ms=redetect_ms,
                    n_candidates=len(cands),
                    n_verified=1 if best is not None else 0,
                    freeze_template=False,
                    aborted=True,
                    sprt_evidence=evidence,
                    sprt_decision=decision,
                    cand_score=cand_score,
                    cand_sim_init=cand_si,
                    cand_sim_recent=cand_sr,
                    cand_sim_distractor=cand_sd,
                    reason="abort_timeout" if timed_out else "abort_cleared",
                    **decision_kwargs,
                )

            # Continue accumulating evidence; freeze template until we either
            # commit or abort.
            return FCRecoverDecision(
                phase=CHALLENGE,
                ran_redetect=True,
                redetect_ms=redetect_ms,
                n_candidates=len(cands),
                n_verified=1 if best is not None else 0,
                freeze_template=True,
                sprt_evidence=evidence,
                sprt_decision=decision,
                cand_score=cand_score,
                cand_sim_init=cand_si,
                cand_sim_recent=cand_sr,
                cand_sim_distractor=cand_sd,
                reason="challenge",
                **decision_kwargs,
            )

        # ---- ABORT_WINDOW: monitor switched track; rollback or commit --------
        if self.phase == ABORT_WINDOW:
            # Track LA/FC streak inside the window: a single transient LA frame
            # right after a switch is normal (the tracker is settling at the new
            # centre); only sustained LA/FC is real regression. With
            # rollback_la_fc_streak=2 we tolerate one settling frame, which
            # eliminates the 100%-rollback failure mode observed when the rule
            # was "rollback on first LA/FC".
            if cfg.rollback_on_failure_state and derived_state in (LA, FC):
                self.abort_la_fc_streak += 1
            else:
                self.abort_la_fc_streak = 0
            degraded = (
                cfg.rollback_on_failure_state
                and self.abort_la_fc_streak >= cfg.rollback_la_fc_streak
            )
            if degraded:
                rollback = self.incumbent_snapshot
                self.phase = IDLE
                self.abort_remaining = 0
                self.abort_la_fc_streak = 0
                self.n_rollbacks += 1
                return FCRecoverDecision(
                    phase=IDLE,
                    freeze_template=True,
                    rollback_center=rollback,
                    reason="rollback_failure_state",
                )
            self.abort_remaining -= 1
            if self.abort_remaining <= 0:
                self.phase = IDLE
                self.abort_la_fc_streak = 0
                self.n_commits += 1
                return FCRecoverDecision(
                    phase=IDLE,
                    freeze_template=False,
                    committed=True,
                    reason="commit",
                )
            return FCRecoverDecision(
                phase=ABORT_WINDOW,
                freeze_template=True,
                reason="abort_window",
            )

        # Unreachable.
        return FCRecoverDecision(phase=self.phase, reason="noop")

    # -- helpers ---------------------------------------------------------------

    def _build_motion_prior(self, frame_idx: int) -> Optional[dict]:
        """Construct verifier motion prior from last_cc_bbox + per-frame velocity cap.

        Returns ``None`` if the prior is disabled (``motion_max_disp == 0`` and
        ``motion_max_disp_per_frame == 0``) or no CC frame has been seen yet —
        in those cases the verifier behaves as before (no motion gate).
        """
        cfg = self.config
        if (cfg.motion_max_disp <= 0.0 and cfg.motion_max_disp_per_frame <= 0.0) \
                or self.last_cc_bbox is None or self.last_cc_frame_idx < 0:
            return None
        cx, cy, w, h = self.last_cc_bbox
        # Total displacement budget: static gate + per-frame velocity * dt.
        dt = max(1, int(frame_idx) - int(self.last_cc_frame_idx))
        max_disp = max(
            float(cfg.motion_max_disp),
            float(cfg.motion_max_disp_per_frame) * float(dt),
        )
        return {
            "cx": float(cx), "cy": float(cy),
            "w": float(w), "h": float(h),
            "max_disp": float(max_disp),
            "max_scale_ratio": float(cfg.motion_max_scale_ratio),
        }

    def _dicts_to_candidates(self, raw: list[dict]) -> list[Candidate]:
        """Adapt SGLATracker.redetect() / generic top-K dicts to V4 Candidate."""
        out: list[Candidate] = []
        for c in raw or []:
            try:
                bbox = c.get("bbox") or [0.0, 0.0, 1.0, 1.0]
                bx, by, bw, bh = (float(bbox[0]), float(bbox[1]),
                                   float(bbox[2]), float(bbox[3]))
                center = c.get("center")
                if center is not None:
                    cx, cy = float(center[0]), float(center[1])
                else:
                    cx, cy = bx + bw / 2.0, by + bh / 2.0
                emb = c.get("embedding")
                if emb is not None:
                    emb = np.asarray(emb)
                cand = Candidate(
                    cx=cx, cy=cy, w=max(1.0, bw), h=max(1.0, bh),
                    score=float(c.get("score", 0.0)),
                    rank=int(c.get("rank", 0)),
                    peak_margin=float(c.get("score_ratio", 0.0)),
                    sim_to_init=float(c.get("sim_to_init", float("nan"))),
                    embedding=emb,
                )
                out.append(cand)
            except (TypeError, ValueError, KeyError):
                continue
        return out


__all__ = [
    "CC", "CU", "LA", "FC",
    "IDLE", "CHALLENGE", "ABORT_WINDOW",
    "FCRecoverConfig",
    "FCRecoverDecision",
    "FCRecoverController",
    "RedetectFn",
]
