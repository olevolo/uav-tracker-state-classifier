"""FC challenge-and-switch controller (MVP, single-tracker).

Motivation
----------
LA (lost-aware) and FC (false-confirmed) need *different* re-detection
policies:

* **LA** — the current localisation is weak, so a recovery candidate can be
  applied immediately (this is what ``--policy_gated_redetect`` does).
* **FC** — the current localisation *looks* convincing but may be the wrong
  object. We must NOT discard it immediately: ~86% of UAV123 FC frames are
  *false* FC (the incumbent is actually fine, see
  ``project_csc_v4_fc_detectability``). A direct ``redetect -> relocate`` here
  would cause catastrophic false switches.

So FC uses a **challenge-and-switch** state machine instead of a direct
relocate:

    FC risk (streak + precision gate)
      -> freeze template (don't learn from a confidently-wrong frame)
      -> read-only redetect each frame (the incumbent track is NOT moved)
      -> accumulate temporal evidence: candidate vs incumbent
      -> SWITCH only if a candidate is stably better for several frames
         AND reappears near the same place across redetect calls
      -> after a switch, an ABORT window with ROLLBACK if the new track
         degrades; template stays frozen until COMMIT.

Verification is **relative** (``candidate_evidence - incumbent_evidence >
margin``), not an absolute similarity threshold — absolute ``sim_to_init`` is a
weak FC discriminator on its own.

This controller is intentionally pure-logic: it owns the state machine and
decides *when* to call re-detection, but the actual re-detect forward pass is
injected as a callback so the controller stays unit-testable without a tracker.
The runner (``tools/run_with_csc.py``) applies the returned search-center
override one-shot before the next ``tracker.update()`` (causal — no
look-ahead) and freezes/refreshes the template per the returned flags.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

# Derived-state ints, matching tools/run_with_csc.py / CSC inference:
#   0=confirmed (CC), 1=uncertain (CU), 2=lost-aware (LA), 3=false-confirmed (FC)
CC, CU, LA, FC = 0, 1, 2, 3

# State-machine phases.
IDLE = "idle"
CHALLENGE = "challenge"
ABORT_WINDOW = "abort_window"

# A re-detect callback returns the best candidate dict (or None) plus the
# wall-clock cost in ms. The candidate dict matches SGLATracker.redetect():
# {"center": (cx, cy), "bbox": [x, y, w, h], "sim_to_init": float,
#  "apce": float, "quality": float, ...}.
RedetectFn = Callable[[], "tuple[Optional[dict], float]"]


@dataclass
class FCChallengeConfig:
    """Tunables for the FC challenge-and-switch controller.

    Defaults are conservative — the priority is *not* switching unless a
    genuinely better object is confirmed over several frames (avoid the
    catastrophic false switch), at the cost of missing some true FC recoveries.
    """

    confirm_frames: int = 3
    """Consecutive frames a candidate must be ``better`` (and reappear nearby)
    before we commit a switch."""

    challenge_max_frames: int = 10
    """Abandon the challenge (declare false FC) after this many frames with no
    confirmed switch. Bounds the read-only re-detect cost per FC episode."""

    abort_window: int = 5
    """After a switch, monitor the new track for this many frames. Rollback if
    it degrades; commit (and allow template refresh) if it stays healthy."""

    sim_margin: float = 0.05
    """Relative identity margin: candidate ``sim_to_init`` must beat the
    incumbent ``initial_template_sim`` by at least this much to count as
    ``better``. Also the identity-collapse threshold during the abort window."""

    switch_mode: str = "identity"
    """How a candidate qualifies as ``better`` than the incumbent:
      ``identity`` (default, safe) — candidate identity beats incumbent by
        ``sim_margin``. Never fires on saturated-identity FC (the confident
        distractor scores HIGHER sim_to_init than the real object), so it is the
        do-no-harm default.
      ``displacement`` — candidate is genuinely RELOCATED off the (wrong)
        incumbent by ``min_switch_disp`` and reappears stably. Used to attempt a
        true-FC recovery when identity is anti-discriminative; the abort window +
        rollback is the safety net for a mis-switch onto a distractor."""

    min_switch_disp: float = 0.5
    """displacement mode: candidate center must be at least this * sqrt(w*h)
    from the current incumbent center to count as a genuine relocation."""

    assoc_gate: float = 2.0
    """association mode: a redetect candidate associates to the tracked anchor
    (initialised at the last-good position, then updated to the chosen candidate
    each frame) if its center is within ``assoc_gate * sqrt(w*h)``. This tracks
    the candidate nearest the last-good trajectory across frames — the only
    signal that picks the real object over the confident distractor in a true-FC
    (the distractor is persistent but far from the target's trajectory)."""

    apce_keep_ratio: float = 0.6
    """Candidate response (APCE) must be at least this fraction of the
    incumbent's APCE — keeps the candidate response-competitive so we win on
    identity, not by switching to a fuzzy peak."""

    reappear_radius: float = 1.0
    """A candidate ``reappears near the same position`` if its center is within
    ``reappear_radius * sqrt(w*h)`` of the running candidate anchor."""

    anchor_ema: float = 0.5
    """EMA factor for the running candidate anchor (location stability)."""

    early_abort_clear_frames: int = 2
    """Abort the challenge early (release the freeze) once the state has been
    non-FC for this many frames with zero positive evidence — don't keep
    freezing a track whose FC alarm has cleared."""

    rollback_on_failure_state: bool = True
    """Roll back the switch if the post-switch state becomes LA or FC."""


@dataclass
class FCChallengeDecision:
    """Per-frame output. The runner applies these levers causally."""

    phase: str = IDLE
    ran_redetect: bool = False
    redetect_ms: float = 0.0
    freeze_template: bool = False
    switch_center: Optional[tuple] = None      # (cx, cy, w, h): relocate onto verified candidate
    rollback_center: Optional[tuple] = None    # (cx, cy, w, h): relocate back to incumbent
    started: bool = False                      # a challenge began this frame
    committed: bool = False                    # switch committed this frame -> open recovery window
    aborted: bool = False                      # challenge abandoned this frame (false FC)
    stable_frames: int = 0
    cand_center: Optional[tuple] = None
    cand_evidence: float = float("nan")        # candidate identity (sim_to_init)
    incumbent_evidence: float = float("nan")   # incumbent identity (initial_template_sim)
    reason: str = ""


@dataclass
class FCChallengeController:
    """Challenge-and-switch FC controller. One instance per sequence.

    Call :meth:`reset` between sequences (or construct a fresh instance).
    """

    config: FCChallengeConfig = field(default_factory=FCChallengeConfig)

    # --- per-sequence state ---
    phase: str = IDLE
    challenge_frames: int = 0
    stable_frames: int = 0
    non_fc_frames: int = 0
    running_anchor: Optional[tuple] = None      # (cx, cy) EMA of the candidate location
    track_anchor: Optional[tuple] = None         # (cx, cy) association tracklet position
    incumbent_snapshot: Optional[tuple] = None  # (cx, cy, w, h) fallback to roll back to
    abort_remaining: int = 0
    switch_cand_sim: float = float("nan")       # candidate identity promised at switch time

    # --- counters (sequence totals, surfaced in metrics) ---
    n_challenges: int = 0
    n_switches: int = 0
    n_commits: int = 0
    n_rollbacks: int = 0
    n_aborts: int = 0
    n_redetect_calls: int = 0

    def reset(self) -> None:
        self.phase = IDLE
        self.challenge_frames = 0
        self.stable_frames = 0
        self.non_fc_frames = 0
        self.running_anchor = None
        self.track_anchor = None
        self.incumbent_snapshot = None
        self.abort_remaining = 0
        self.switch_cand_sim = float("nan")
        self.n_challenges = 0
        self.n_switches = 0
        self.n_commits = 0
        self.n_rollbacks = 0
        self.n_aborts = 0
        self.n_redetect_calls = 0

    @property
    def active(self) -> bool:
        """True while a challenge / abort window is in progress."""
        return self.phase != IDLE

    def step(
        self,
        *,
        derived_state: int,
        fc_trigger: bool,
        bbox: tuple,
        initial_template_sim: float,
        incumbent_apce: float,
        incumbent_center: Optional[tuple] = None,
        incumbent_size: Optional[tuple] = None,
        redetect_fn: RedetectFn,
    ) -> FCChallengeDecision:
        """Advance the state machine by one frame.

        Parameters
        ----------
        derived_state
            CSC derived state int (0=CC, 1=CU, 2=LA, 3=FC) for THIS frame.
        fc_trigger
            Runner-computed gate: FC streak satisfied AND precision vote gate
            passed. Only a trigger starts a challenge (do NOT use raw FC argmax).
        bbox
            Current incumbent bbox ``(x, y, w, h)`` from ``tracker.update()``.
        initial_template_sim, incumbent_apce
            Incumbent identity / response evidence for THIS frame (telemetry).
        incumbent_center, incumbent_size
            Last trusted CONFIRMED position to roll back to. Falls back to the
            current bbox when no confirmed frame has been seen yet.
        redetect_fn
            Zero-arg callback that runs a read-only re-detect and returns
            ``(candidate_dict | None, latency_ms)``. Invoked ONLY during the
            challenge phase (bounded cost).
        """
        cfg = self.config
        bx, by, bw, bh = bbox
        inc_center = incumbent_center if incumbent_center is not None else (bx + bw / 2.0, by + bh / 2.0)
        inc_size = incumbent_size if incumbent_size is not None else (bw, bh)
        started_now = False

        # ---- IDLE: watch for an FC trigger ----
        if self.phase == IDLE:
            if not fc_trigger:
                return FCChallengeDecision(phase=IDLE, reason="idle")
            # Enter CHALLENGE. Snapshot the incumbent as the rollback fallback.
            self.phase = CHALLENGE
            self.challenge_frames = 0
            self.stable_frames = 0
            self.non_fc_frames = 0
            self.running_anchor = None
            self.track_anchor = None
            self.incumbent_snapshot = (inc_center[0], inc_center[1], inc_size[0], inc_size[1])
            self.n_challenges += 1
            started_now = True
            # fall through to CHALLENGE handling on this same frame

        # ---- CHALLENGE: read-only redetect, accumulate relative evidence ----
        if self.phase == CHALLENGE:
            self.challenge_frames += 1
            self.non_fc_frames = 0 if derived_state == FC else self.non_fc_frames + 1

            result, ms = redetect_fn()
            self.n_redetect_calls += 1

            # Normalise the redetect result into a candidate list (association
            # mode requests top-K; identity/displacement get a single best).
            cands = result if isinstance(result, list) else ([result] if result else [])

            cand_center = None
            cand_sim = float("nan")
            chosen = None
            if cands:
                if cfg.switch_mode == "association":
                    # Track the candidate nearest the last-good trajectory: the
                    # anchor starts at the last-good position and follows the
                    # chosen candidate each frame (a mini SOT over redetect
                    # detections). The confident distractor is persistent but far
                    # from the target's trajectory, so it never associates.
                    if self.track_anchor is None:
                        self.track_anchor = (inc_center[0], inc_center[1])
                    best = None
                    for c in cands:
                        cc = c.get("center")
                        if cc is None:
                            continue
                        cbb = c.get("bbox", [0.0, 0.0, float(bw), float(bh)])
                        csc = max(1.0, math.sqrt(max(1.0, float(cbb[2]) * float(cbb[3]))))
                        d = math.hypot(cc[0] - self.track_anchor[0], cc[1] - self.track_anchor[1])
                        if d <= cfg.assoc_gate * csc and (best is None or d < best[0]):
                            best = (d, c)
                    chosen = best[1] if best is not None else None
                else:
                    chosen = cands[0]

            if chosen is not None:
                cand_center = tuple(chosen.get("center", (float("nan"), float("nan"))))
                cbb = chosen.get("bbox", [0.0, 0.0, float(bw), float(bh)])
                cw, ch = float(cbb[2]), float(cbb[3])
                scale = max(1.0, math.sqrt(max(1.0, cw * ch)))
                cand_sim = float(chosen.get("sim_to_init", float("nan")))
                cand_apce = float(chosen.get("apce", 0.0))
                response_ok = cand_apce >= float(incumbent_apce) * cfg.apce_keep_ratio

                if cfg.switch_mode == "association":
                    # A candidate associated within the gate = the tracklet
                    # persists. Response is NOT gated (the real object often has a
                    # weaker peak than the distractor); proximity-to-trajectory is
                    # the signal. Follow the tracklet.
                    better = True
                    proximate = True
                    self.track_anchor = cand_center
                elif cfg.switch_mode == "displacement":
                    # Identity is anti-discriminative for FC (the confident
                    # distractor scores HIGHER sim_to_init than the real object —
                    # measured on car2_s: real 0.69-0.78 < distractor 0.87 <
                    # incumbent 0.90). Switch on a candidate genuinely RELOCATED
                    # off the (wrong) incumbent that reappears stably; abort window
                    # + rollback is the safety net for a mis-switch.
                    inc_cx, inc_cy = bx + bw / 2.0, by + bh / 2.0
                    disp = math.hypot(cand_center[0] - inc_cx, cand_center[1] - inc_cy)
                    better = bool(disp >= cfg.min_switch_disp * scale and response_ok)
                    proximate = (
                        self.running_anchor is None
                        or math.hypot(cand_center[0] - self.running_anchor[0],
                                      cand_center[1] - self.running_anchor[1]) <= cfg.reappear_radius * scale
                    )
                else:  # identity (safe default)
                    identity_win = (
                        math.isfinite(cand_sim)
                        and (cand_sim - float(initial_template_sim)) >= cfg.sim_margin
                    )
                    better = bool(identity_win and response_ok)
                    proximate = (
                        self.running_anchor is None
                        or math.hypot(cand_center[0] - self.running_anchor[0],
                                      cand_center[1] - self.running_anchor[1]) <= cfg.reappear_radius * scale
                    )

                if better and proximate:
                    self.stable_frames += 1
                    if self.running_anchor is None:
                        self.running_anchor = cand_center
                    else:
                        a = cfg.anchor_ema
                        self.running_anchor = (
                            a * self.running_anchor[0] + (1 - a) * cand_center[0],
                            a * self.running_anchor[1] + (1 - a) * cand_center[1],
                        )
                    # CONFIRMED: stably better near the same spot for confirm_frames
                    # -> commit the switch.
                    if self.stable_frames >= cfg.confirm_frames:
                        self.switch_cand_sim = cand_sim
                        self.phase = ABORT_WINDOW
                        self.abort_remaining = cfg.abort_window
                        self.n_switches += 1
                        return FCChallengeDecision(
                            phase=ABORT_WINDOW,
                            ran_redetect=True,
                            redetect_ms=ms,
                            freeze_template=True,
                            switch_center=(cand_center[0], cand_center[1], cw, ch),
                            started=started_now,
                            stable_frames=self.stable_frames,
                            cand_center=cand_center,
                            cand_evidence=cand_sim,
                            incumbent_evidence=float(initial_template_sim),
                            reason="switch",
                        )
                else:
                    # Not better / jumped elsewhere: reset the streak and re-anchor
                    # to the newest location (start fresh tracking it).
                    self.stable_frames = 0
                    self.running_anchor = cand_center
            else:
                # No usable candidate this frame (none found, or none associated
                # within the gate): the tracklet did not persist -> reset streak.
                self.stable_frames = 0

            # No switch this frame. Decide whether to keep challenging.
            timed_out = self.challenge_frames >= cfg.challenge_max_frames
            cleared = (
                self.non_fc_frames >= cfg.early_abort_clear_frames
                and self.stable_frames == 0
            )
            if timed_out or cleared:
                self.phase = IDLE
                self.n_aborts += 1
                self.running_anchor = None
                self.track_anchor = None
                return FCChallengeDecision(
                    phase=IDLE,
                    ran_redetect=True,
                    redetect_ms=ms,
                    freeze_template=False,
                    started=started_now,
                    aborted=True,
                    stable_frames=0,
                    cand_center=cand_center,
                    cand_evidence=cand_sim,
                    incumbent_evidence=float(initial_template_sim),
                    reason="abort_timeout" if timed_out else "abort_cleared",
                )

            return FCChallengeDecision(
                phase=CHALLENGE,
                ran_redetect=True,
                redetect_ms=ms,
                freeze_template=True,
                started=started_now,
                stable_frames=self.stable_frames,
                cand_center=cand_center,
                cand_evidence=cand_sim,
                incumbent_evidence=float(initial_template_sim),
                reason="challenge",
            )

        # ---- ABORT_WINDOW: monitor the switched track; rollback or commit ----
        if self.phase == ABORT_WINDOW:
            degraded = bool(cfg.rollback_on_failure_state and derived_state in (LA, FC))
            # The identity-collapse check only applies in identity mode — for
            # displacement/association the candidate identity is legitimately low
            # (the real object scores lower sim_to_init than the distractor), so
            # an identity drop is NOT evidence of a bad switch there.
            identity_collapsed = (
                cfg.switch_mode == "identity"
                and math.isfinite(self.switch_cand_sim)
                and float(initial_template_sim) < (self.switch_cand_sim - cfg.sim_margin)
            )
            if degraded or identity_collapsed:
                rollback = self.incumbent_snapshot
                self.phase = IDLE
                self.abort_remaining = 0
                self.running_anchor = None
                self.track_anchor = None
                self.n_rollbacks += 1
                return FCChallengeDecision(
                    phase=IDLE,
                    freeze_template=True,
                    rollback_center=rollback,
                    incumbent_evidence=float(initial_template_sim),
                    reason="rollback_failure_state" if degraded else "rollback_identity",
                )

            self.abort_remaining -= 1
            if self.abort_remaining <= 0:
                # COMMIT: the switched track held -> release the freeze so the
                # runner can lock in the new appearance (recovery window).
                self.phase = IDLE
                self.running_anchor = None
                self.track_anchor = None
                self.n_commits += 1
                return FCChallengeDecision(
                    phase=IDLE,
                    freeze_template=False,
                    committed=True,
                    incumbent_evidence=float(initial_template_sim),
                    reason="commit",
                )

            return FCChallengeDecision(
                phase=ABORT_WINDOW,
                freeze_template=True,
                incumbent_evidence=float(initial_template_sim),
                reason="abort_window",
            )

        # Unreachable, but keep a safe default.
        return FCChallengeDecision(phase=self.phase, reason="noop")
