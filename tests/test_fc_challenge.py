"""Unit tests for the FC challenge-and-switch controller.

The controller is the MVP safeguard for false-confirmed (FC) frames: instead of
relocating immediately (catastrophic on the ~86% false-FC majority), it runs a
read-only redetect, verifies a candidate is *stably better than the incumbent*
across several frames near the same position, switches only then, and holds an
abort window with rollback. These tests pin that state machine with a scripted
redetect callback (no tracker needed).
"""
from __future__ import annotations

import pytest

from csc_lib.csc.fc_challenge import (
    CC, CU, LA, FC,
    IDLE, CHALLENGE, ABORT_WINDOW,
    FCChallengeConfig,
    FCChallengeController,
)


def _cand(center=(200.0, 200.0), wh=(40.0, 40.0), sim=0.8, apce=80.0):
    """Build a redetect candidate dict matching SGLATracker.redetect()."""
    cx, cy = center
    w, h = wh
    return {
        "center": (float(cx), float(cy)),
        "bbox": [cx - w / 2.0, cy - h / 2.0, float(w), float(h)],
        "sim_to_init": float(sim),
        "apce": float(apce),
        "quality": float(apce),
    }


def _redetect_const(cand):
    """A redetect_fn that always returns the same candidate (5ms)."""
    return lambda: (cand, 5.0)


def _step(ctrl, *, state, trigger, cand_fn, inc_sim=0.5, inc_apce=100.0,
          inc_center=(50.0, 50.0), inc_size=(30.0, 30.0), bbox=(190.0, 190.0, 20.0, 20.0)):
    return ctrl.step(
        derived_state=state,
        fc_trigger=trigger,
        bbox=bbox,
        initial_template_sim=inc_sim,
        incumbent_apce=inc_apce,
        incumbent_center=inc_center,
        incumbent_size=inc_size,
        redetect_fn=cand_fn,
    )


def test_idle_no_trigger_is_noop():
    """No FC trigger -> idle, no redetect, no freeze."""
    ctrl = FCChallengeController()
    dec = _step(ctrl, state=CC, trigger=False, cand_fn=_redetect_const(_cand()))
    assert dec.phase == IDLE
    assert dec.ran_redetect is False
    assert dec.freeze_template is False
    assert dec.switch_center is None
    assert ctrl.active is False


def test_trigger_starts_challenge_and_freezes():
    """An FC trigger starts a challenge: redetect runs, template freezes."""
    ctrl = FCChallengeController(FCChallengeConfig(confirm_frames=3))
    dec = _step(ctrl, state=FC, trigger=True, cand_fn=_redetect_const(_cand()))
    assert dec.started is True
    assert dec.phase == CHALLENGE
    assert dec.ran_redetect is True
    assert dec.redetect_ms == pytest.approx(5.0)
    assert dec.freeze_template is True
    assert dec.switch_center is None          # not yet confirmed
    assert dec.stable_frames == 1
    assert ctrl.active is True


def test_switch_only_after_confirm_frames():
    """A stably-better, nearby candidate switches only after confirm_frames."""
    cfg = FCChallengeConfig(confirm_frames=3, sim_margin=0.05, apce_keep_ratio=0.6)
    ctrl = FCChallengeController(cfg)
    cand = _cand(sim=0.85, apce=80.0)           # beats inc_sim=0.5 by 0.35; apce 80 >= 60
    fn = _redetect_const(cand)

    d1 = _step(ctrl, state=FC, trigger=True, cand_fn=fn)
    assert d1.switch_center is None and d1.stable_frames == 1
    d2 = _step(ctrl, state=FC, trigger=False, cand_fn=fn)
    assert d2.switch_center is None and d2.stable_frames == 2
    d3 = _step(ctrl, state=FC, trigger=False, cand_fn=fn)
    # confirmed -> switch onto the candidate, enter abort window
    assert d3.switch_center is not None
    assert d3.switch_center[0] == pytest.approx(200.0)
    assert d3.switch_center[1] == pytest.approx(200.0)
    assert d3.phase == ABORT_WINDOW
    assert d3.reason == "switch"


def test_no_switch_when_identity_not_better():
    """SAFETY: a candidate that does NOT beat the incumbent identity never
    switches, even if redetect keeps returning it — it aborts on timeout."""
    cfg = FCChallengeConfig(confirm_frames=3, challenge_max_frames=4, sim_margin=0.05)
    ctrl = FCChallengeController(cfg)
    cand = _cand(sim=0.50, apce=80.0)           # == incumbent identity -> not better
    fn = _redetect_const(cand)

    aborted = False
    for _ in range(4):
        dec = _step(ctrl, state=FC, trigger=True, cand_fn=fn)
        assert dec.switch_center is None
        assert dec.stable_frames == 0
        aborted = aborted or dec.aborted
    assert aborted is True
    assert ctrl.active is False


def test_no_switch_when_candidate_jumps_around():
    """A candidate that reappears far from the running anchor resets the streak
    (location-stability / reappearance requirement)."""
    cfg = FCChallengeConfig(confirm_frames=3, reappear_radius=1.0, challenge_max_frames=20)
    ctrl = FCChallengeController(cfg)
    centers = [(200.0, 200.0), (600.0, 600.0), (100.0, 100.0), (900.0, 50.0)]
    idx = {"i": 0}

    def fn():
        c = centers[idx["i"] % len(centers)]
        idx["i"] += 1
        return _cand(center=c, wh=(40.0, 40.0), sim=0.85, apce=80.0), 5.0

    for _ in range(6):
        dec = _step(ctrl, state=FC, trigger=True, cand_fn=fn)
        # each candidate is far from the previous anchor -> streak never builds
        assert dec.stable_frames <= 1
        assert dec.switch_center is None


def _drive_to_switch(ctrl, cand_sim=0.85):
    """Helper: run a controller to a committed switch and return nothing."""
    fn = _redetect_const(_cand(sim=cand_sim, apce=80.0))
    for _ in range(ctrl.config.confirm_frames):
        dec = _step(ctrl, state=FC, trigger=True, cand_fn=fn)
    assert dec.switch_center is not None
    return dec


def test_rollback_when_post_switch_state_degrades():
    """After a switch, an LA/FC state inside the abort window rolls back to the
    pre-switch incumbent snapshot."""
    cfg = FCChallengeConfig(confirm_frames=2, abort_window=5, rollback_on_failure_state=True)
    ctrl = FCChallengeController(cfg)
    _drive_to_switch(ctrl)
    # next frame: the switched track immediately goes LOST -> rollback
    dec = _step(ctrl, state=LA, trigger=False, cand_fn=_redetect_const(_cand()),
                inc_center=(50.0, 50.0), inc_size=(30.0, 30.0))
    assert dec.rollback_center is not None
    assert dec.rollback_center[0] == pytest.approx(50.0)
    assert dec.rollback_center[1] == pytest.approx(50.0)
    assert dec.reason == "rollback_failure_state"
    assert ctrl.active is False


def test_commit_when_post_switch_track_stays_healthy():
    """After a switch, a healthy track through the whole abort window commits
    and releases the freeze (so the template can lock in the switch)."""
    cfg = FCChallengeConfig(confirm_frames=2, abort_window=2, sim_margin=0.05)
    ctrl = FCChallengeController(cfg)
    _drive_to_switch(ctrl, cand_sim=0.85)
    # abort window: stay CONFIRMED with identity >= promised (0.85)
    d1 = _step(ctrl, state=CC, trigger=False, cand_fn=_redetect_const(_cand()), inc_sim=0.85)
    assert d1.phase == ABORT_WINDOW and d1.freeze_template is True and d1.committed is False
    d2 = _step(ctrl, state=CC, trigger=False, cand_fn=_redetect_const(_cand()), inc_sim=0.85)
    assert d2.committed is True
    assert d2.freeze_template is False
    assert ctrl.active is False


def test_rollback_when_identity_collapses_in_abort_window():
    """Even with a non-failure state, a sharp identity drop after the switch
    rolls back (the switch went to the wrong object)."""
    cfg = FCChallengeConfig(confirm_frames=2, abort_window=5, sim_margin=0.05)
    ctrl = FCChallengeController(cfg)
    _drive_to_switch(ctrl, cand_sim=0.85)
    # state is CONFIRMED but identity collapsed well below the promised 0.85
    dec = _step(ctrl, state=CC, trigger=False, cand_fn=_redetect_const(_cand()), inc_sim=0.40)
    assert dec.rollback_center is not None
    assert dec.reason == "rollback_identity"


def test_early_abort_when_fc_clears():
    """If the FC alarm clears (state non-FC) with no positive evidence, the
    challenge aborts early so a recovered track is not needlessly frozen."""
    cfg = FCChallengeConfig(confirm_frames=3, challenge_max_frames=20,
                            early_abort_clear_frames=2, sim_margin=0.05)
    ctrl = FCChallengeController(cfg)
    weak = _redetect_const(_cand(sim=0.50, apce=80.0))   # never better
    # start the challenge on an FC frame
    d0 = _step(ctrl, state=FC, trigger=True, cand_fn=weak)
    assert d0.phase == CHALLENGE
    # now the state clears to CONFIRMED for early_abort_clear_frames
    _step(ctrl, state=CC, trigger=False, cand_fn=weak)
    dec = _step(ctrl, state=CC, trigger=False, cand_fn=weak)
    assert dec.aborted is True
    assert dec.reason == "abort_cleared"
    assert dec.freeze_template is False


def test_reset_clears_state_and_counters():
    ctrl = FCChallengeController(FCChallengeConfig(confirm_frames=2))
    _drive_to_switch(ctrl)
    assert ctrl.active is True
    ctrl.reset()
    assert ctrl.active is False
    assert ctrl.phase == IDLE
    assert ctrl.n_challenges == 0
    assert ctrl.n_switches == 0


def test_displacement_mode_switches_on_relocated_low_identity_candidate():
    """displacement mode: a candidate genuinely relocated off the incumbent
    switches even with LOW identity (identity is anti-discriminative for FC).
    The same candidate would NOT switch under identity mode."""
    far = _cand(center=(400.0, 400.0), wh=(40.0, 40.0), sim=0.30, apce=80.0)
    fn = _redetect_const(far)
    # incumbent bbox center = (200,200); candidate at (400,400) is far -> displaced
    cfg_d = FCChallengeConfig(confirm_frames=2, switch_mode="displacement",
                              min_switch_disp=0.5, apce_keep_ratio=0.6)
    ctrl_d = FCChallengeController(cfg_d)
    d1 = _step(ctrl_d, state=FC, trigger=True, cand_fn=fn)
    assert d1.stable_frames == 1 and d1.switch_center is None
    d2 = _step(ctrl_d, state=FC, trigger=False, cand_fn=fn)
    assert d2.switch_center is not None        # relocated + reappeared -> switch
    assert d2.switch_center[0] == pytest.approx(400.0)

    # identity mode: same low-identity candidate never switches
    cfg_i = FCChallengeConfig(confirm_frames=2, switch_mode="identity",
                              sim_margin=0.05, challenge_max_frames=4)
    ctrl_i = FCChallengeController(cfg_i)
    for _ in range(4):
        di = _step(ctrl_i, state=FC, trigger=True, cand_fn=fn)
        assert di.switch_center is None


def test_association_mode_tracks_candidate_near_lastgood_not_distractor():
    """association mode: among multiple candidates, track the one nearest the
    last-good trajectory (init = incumbent_center) and switch to it, ignoring a
    persistent-but-far distractor — the signal that picks the real object in a
    true-FC where identity/quality both favour the distractor."""
    real = _cand(center=(60.0, 60.0), wh=(40.0, 40.0), sim=0.70, apce=40.0)
    distractor = _cand(center=(600.0, 80.0), wh=(40.0, 40.0), sim=0.90, apce=220.0)

    def fn():
        return [distractor, real], 5.0   # order must not matter

    cfg = FCChallengeConfig(confirm_frames=2, switch_mode="association", assoc_gate=2.0)
    ctrl = FCChallengeController(cfg)
    d1 = _step(ctrl, state=FC, trigger=True, cand_fn=fn,
               inc_center=(50.0, 50.0), inc_size=(30.0, 30.0))
    assert d1.stable_frames == 1 and d1.switch_center is None
    d2 = _step(ctrl, state=FC, trigger=False, cand_fn=fn,
               inc_center=(50.0, 50.0), inc_size=(30.0, 30.0))
    assert d2.switch_center is not None
    assert d2.switch_center[0] == pytest.approx(60.0)   # tracked the REAL object
    assert d2.switch_center[1] == pytest.approx(60.0)   # NOT the distractor at (600,80)


def test_association_mode_resets_when_no_candidate_in_gate():
    """association mode: a frame with only a far (un-gateable) candidate breaks
    the tracklet and resets the streak (no spurious switch)."""
    far = _cand(center=(900.0, 900.0), wh=(40.0, 40.0))

    def fn():
        return [far], 5.0

    cfg = FCChallengeConfig(confirm_frames=2, switch_mode="association",
                            assoc_gate=2.0, challenge_max_frames=5)
    ctrl = FCChallengeController(cfg)
    for _ in range(5):
        d = _step(ctrl, state=FC, trigger=True, cand_fn=fn,
                  inc_center=(50.0, 50.0), inc_size=(30.0, 30.0))
        assert d.stable_frames == 0
        assert d.switch_center is None

