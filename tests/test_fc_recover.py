"""Smoke tests for csc_lib/csc/recover/recover_ctrl.py.

CPU-only, dataset-free. Validate the state-machine + verifier + SPRT integration:

* IDLE without trigger is a no-op.
* FC trigger seeds a distractor and starts CHALLENGE.
* A *clean target candidate* (high sim_to_init/recent, no distractor match)
  accumulates positive LLR and eventually FIREs a switch.
* A *distractor-shaped candidate* (high sim_to_distractor) is hard-vetoed by
  the verifier -> no fire even if the score-map looks dominant.
* In ABORT_WINDOW, a state regression to LA/FC ROLLBACKs to the snapshot.
* SPRT false-alert budget caps how many switches can fire in one sequence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "csc_uav_tracking_sdk" / "src",
           PROJECT_ROOT / "src"):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from csc_lib.csc.recover.recover_ctrl import (  # noqa: E402
    CC,
    CU,
    LA,
    FC,
    IDLE,
    CHALLENGE,
    ABORT_WINDOW,
    FCRecoverConfig,
    FCRecoverController,
)


# ---------------------------------------------------------------------------
# Test embeddings — three orthogonal-ish 192-D unit vectors so the verifier
# sees clean signals (sim ~1 for self-match, sim ~0 for orthogonal).
# ---------------------------------------------------------------------------
DIM = 192


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


@pytest.fixture
def embeddings():
    """Three unit embeddings: target / distractor / arbitrary other."""
    return {
        "target": _unit(0),
        "distractor": _unit(1),
        "other": _unit(2),
    }


def _cand_dict(*, cx, cy, w=80.0, h=80.0, score=0.5, rank=0,
                score_ratio=0.5, sim_to_init=0.7, embedding=None):
    return {
        "bbox": [cx - w / 2.0, cy - h / 2.0, w, h],
        "center": [cx, cy],
        "score": float(score),
        "rank": int(rank),
        "score_ratio": float(score_ratio),
        "sim_to_init": float(sim_to_init),
        "embedding": embedding,
    }


# ---------------------------------------------------------------------------


def test_idle_without_trigger_is_noop(embeddings):
    cfg = FCRecoverConfig()
    ctrl = FCRecoverController(config=cfg)
    ctrl.maybe_seed_anchor(embeddings["target"])

    # 100 calls without an FC trigger -> stays IDLE, no redetects.
    for t in range(100):
        d = ctrl.step(
            derived_state=CC,
            fc_trigger=False,
            incumbent_bbox=(100.0, 100.0, 80.0, 80.0),
            redetect_fn=None,
            frame_idx=t,
        )
        assert d.phase == IDLE
        assert not d.ran_redetect
        assert d.switch_center is None
    assert ctrl.n_challenges == 0
    assert ctrl.n_redetect_calls == 0


def test_fc_trigger_seeds_distractor_and_enters_challenge(embeddings):
    cfg = FCRecoverConfig()
    ctrl = FCRecoverController(config=cfg)
    ctrl.maybe_seed_anchor(embeddings["target"])
    # A few CC frames so memory.recent has something:
    for t in range(3):
        ctrl.note_cc(embeddings["target"], frame_idx=t,
                     bbox=(80.0, 80.0, 80.0, 80.0))

    # FC trigger arrives. Pass the incumbent's wrong-lock embedding to seed
    # the distractor memory.
    d = ctrl.step(
        derived_state=FC,
        fc_trigger=True,
        incumbent_bbox=(200.0, 200.0, 80.0, 80.0),
        incumbent_emb=embeddings["distractor"],
        redetect_fn=lambda: ([], 0.0),  # no candidates this frame
        frame_idx=10,
    )
    assert d.started
    assert d.distractor_seeded
    assert d.phase in (CHALLENGE, IDLE)  # may abort same frame if no candidate
    assert ctrl.n_distractor_seeds == 1
    # The distractor was seeded into memory.
    s = ctrl.memory.sims(embeddings["distractor"])
    assert s["sim_to_distractor"] > 0.99


def test_target_candidate_eventually_fires_switch(embeddings):
    cfg = FCRecoverConfig(
        # Slightly easier SPRT so the smoke test fires within reasonable frames.
        sprt_alpha=0.10,
        sprt_beta=0.10,
        # Lower verifier accept margin so a strong-target cand passes.
        verifier_accept_margin=0.40,
        verifier_min_identity=0.30,
        motion_max_disp_per_frame=0.0,  # synthetic candidate at cx=300 with no motion model
    )
    ctrl = FCRecoverController(config=cfg)
    ctrl.maybe_seed_anchor(embeddings["target"])
    for t in range(3):
        ctrl.note_cc(embeddings["target"], frame_idx=t,
                     bbox=(80.0, 80.0, 80.0, 80.0))

    # Each frame produces one strong-target candidate.
    target_cand = lambda: (
        [_cand_dict(cx=300.0, cy=300.0, score=0.6, rank=0,
                    score_ratio=0.5, sim_to_init=0.95,
                    embedding=embeddings["target"].copy())],
        1.0,
    )

    fire_at = None
    for t in range(30):
        # Frame 0 = trigger; later frames = sustained CHALLENGE.
        d = ctrl.step(
            derived_state=FC,
            fc_trigger=(t == 0),
            incumbent_bbox=(200.0, 200.0, 80.0, 80.0),
            incumbent_emb=embeddings["distractor"],
            redetect_fn=target_cand,
            frame_idx=10 + t,
        )
        if d.switch_center is not None:
            fire_at = t
            assert d.phase == ABORT_WINDOW
            assert d.cand_sim_init > 0.9
            assert d.freeze_template
            break

    assert fire_at is not None, (
        "sustained strong-target evidence should fire a switch within 30 frames"
    )
    assert ctrl.n_switches == 1


def test_distractor_candidate_never_fires_switch(embeddings):
    cfg = FCRecoverConfig(sprt_alpha=0.10, sprt_beta=0.10)
    ctrl = FCRecoverController(config=cfg)
    ctrl.maybe_seed_anchor(embeddings["target"])
    for t in range(3):
        ctrl.note_cc(embeddings["target"], frame_idx=t,
                     bbox=(80.0, 80.0, 80.0, 80.0))

    # Each frame produces a candidate that LOOKS LIKE the distractor (vetoed).
    distractor_cand = lambda: (
        [_cand_dict(cx=300.0, cy=300.0, score=0.9, rank=0,
                    score_ratio=0.9, sim_to_init=0.5,
                    embedding=embeddings["distractor"].copy())],
        1.0,
    )

    ever_fired = False
    for t in range(25):
        d = ctrl.step(
            derived_state=FC,
            fc_trigger=(t == 0),
            incumbent_bbox=(200.0, 200.0, 80.0, 80.0),
            incumbent_emb=embeddings["distractor"],
            redetect_fn=distractor_cand,
            frame_idx=10 + t,
        )
        if d.switch_center is not None:
            ever_fired = True
            break
    assert not ever_fired, (
        "candidate matching seeded distractor must be hard-vetoed; no switch"
    )
    # Should have aborted (timeout or clear), no switches.
    assert ctrl.n_switches == 0
    assert ctrl.n_aborts >= 1


def test_abort_window_rollback_on_state_regression(embeddings):
    cfg = FCRecoverConfig(
        sprt_alpha=0.10, sprt_beta=0.10,
        verifier_accept_margin=0.40, verifier_min_identity=0.30,
        abort_window=5,
        rollback_la_fc_streak=2,  # default; verify it requires SUSTAINED regression
        motion_max_disp_per_frame=0.0,  # disable motion gate; test rollback logic in isolation
    )
    ctrl = FCRecoverController(config=cfg)
    ctrl.maybe_seed_anchor(embeddings["target"])
    for t in range(3):
        ctrl.note_cc(embeddings["target"], frame_idx=t,
                     bbox=(80.0, 80.0, 80.0, 80.0))

    # Force a switch with a strong-target candidate.
    target_cand = lambda: (
        [_cand_dict(cx=300.0, cy=300.0, score=0.7, rank=0,
                    score_ratio=0.5, sim_to_init=0.95,
                    embedding=embeddings["target"].copy())],
        1.0,
    )
    fire_at = None
    for t in range(15):
        d = ctrl.step(
            derived_state=FC,
            fc_trigger=(t == 0),
            incumbent_bbox=(200.0, 200.0, 80.0, 80.0),
            incumbent_emb=embeddings["distractor"],
            redetect_fn=target_cand,
            frame_idx=t,
        )
        if d.phase == ABORT_WINDOW:
            fire_at = t
            break
    assert fire_at is not None

    # Single transient LA inside the abort window must NOT rollback (streak=1<2).
    # This is the bug fix: post-switch tracker often shows 1 settling LA frame.
    d1 = ctrl.step(
        derived_state=LA,
        fc_trigger=False,
        incumbent_bbox=(300.0, 300.0, 80.0, 80.0),
        redetect_fn=None,
        frame_idx=fire_at + 1,
    )
    assert d1.rollback_center is None, "single LA frame should not rollback (streak=1)"
    assert d1.phase == ABORT_WINDOW

    # SECOND consecutive LA -> rollback (streak=2).
    d2 = ctrl.step(
        derived_state=LA,
        fc_trigger=False,
        incumbent_bbox=(300.0, 300.0, 80.0, 80.0),
        redetect_fn=None,
        frame_idx=fire_at + 2,
    )
    assert d2.rollback_center is not None, "two consecutive LA frames should rollback"
    assert d2.phase == IDLE
    assert ctrl.n_rollbacks == 1


def test_abort_window_commits_on_clean_post_switch(embeddings):
    """Post-switch frames in CC -> commit after abort_window expires."""
    cfg = FCRecoverConfig(
        sprt_alpha=0.10, sprt_beta=0.10,
        verifier_accept_margin=0.40, verifier_min_identity=0.30,
        abort_window=3, rollback_la_fc_streak=2,
        motion_max_disp_per_frame=0.0,  # disable motion gate; isolate commit behaviour
    )
    ctrl = FCRecoverController(config=cfg)
    ctrl.maybe_seed_anchor(embeddings["target"])
    for t in range(3):
        ctrl.note_cc(embeddings["target"], frame_idx=t,
                     bbox=(80.0, 80.0, 80.0, 80.0))

    target_cand = lambda: (
        [_cand_dict(cx=300.0, cy=300.0, score=0.7, rank=0,
                    score_ratio=0.5, sim_to_init=0.95,
                    embedding=embeddings["target"].copy())],
        1.0,
    )
    fire_at = None
    for t in range(15):
        d = ctrl.step(
            derived_state=FC,
            fc_trigger=(t == 0),
            incumbent_bbox=(200.0, 200.0, 80.0, 80.0),
            incumbent_emb=embeddings["distractor"],
            redetect_fn=target_cand,
            frame_idx=t,
        )
        if d.phase == ABORT_WINDOW:
            fire_at = t
            break
    assert fire_at is not None

    # 3 frames of CC during abort window -> commit at the 3rd.
    committed = False
    for t in range(3):
        d = ctrl.step(
            derived_state=CC,
            fc_trigger=False,
            incumbent_bbox=(300.0, 300.0, 80.0, 80.0),
            redetect_fn=None,
            frame_idx=fire_at + 1 + t,
        )
        if d.committed:
            committed = True
            break
    assert committed, "clean CC post-switch should reach commit"
    assert ctrl.n_commits == 1
    assert ctrl.n_rollbacks == 0


def test_sprt_budget_caps_switch_count(embeddings):
    cfg = FCRecoverConfig(
        sprt_alpha=0.20, sprt_beta=0.20,         # easy fire for the test
        sprt_false_alert_budget=2,
        verifier_accept_margin=0.30, verifier_min_identity=0.30,
        challenge_max_frames=20, abort_window=2,
        motion_max_disp_per_frame=0.0,
    )
    ctrl = FCRecoverController(config=cfg)
    ctrl.maybe_seed_anchor(embeddings["target"])
    for t in range(3):
        ctrl.note_cc(embeddings["target"], frame_idx=t,
                     bbox=(80.0, 80.0, 80.0, 80.0))

    target_cand = lambda: (
        [_cand_dict(cx=300.0, cy=300.0, score=0.9, rank=0,
                    score_ratio=0.9, sim_to_init=0.95,
                    embedding=embeddings["target"].copy())],
        1.0,
    )

    # Drive 200 frames where every frame either triggers FC or is in CHALLENGE.
    # The SPRT budget is 2, so n_switches must NOT exceed 2.
    for t in range(200):
        ctrl.step(
            derived_state=FC,
            fc_trigger=(ctrl.phase == IDLE),
            incumbent_bbox=(200.0, 200.0, 80.0, 80.0),
            incumbent_emb=embeddings["distractor"],
            redetect_fn=target_cand,
            frame_idx=t,
        )
    assert ctrl.n_switches <= 2, f"budget should cap switches at 2, got {ctrl.n_switches}"


def test_reset_between_sequences_clears_state(embeddings):
    cfg = FCRecoverConfig(
        sprt_alpha=0.10, sprt_beta=0.10,
        verifier_accept_margin=0.40, verifier_min_identity=0.30,
        motion_max_disp_per_frame=0.0,
    )
    ctrl = FCRecoverController(config=cfg)
    ctrl.maybe_seed_anchor(embeddings["target"])
    ctrl.note_cc(embeddings["target"], frame_idx=0, bbox=(80.0, 80.0, 80.0, 80.0))
    target_cand = lambda: (
        [_cand_dict(cx=300.0, cy=300.0, score=0.6, rank=0,
                    score_ratio=0.5, sim_to_init=0.95,
                    embedding=embeddings["target"].copy())],
        1.0,
    )
    for t in range(20):
        ctrl.step(
            derived_state=FC,
            fc_trigger=(t == 0),
            incumbent_bbox=(200.0, 200.0, 80.0, 80.0),
            incumbent_emb=embeddings["distractor"],
            redetect_fn=target_cand,
            frame_idx=t,
        )
    # State accumulated.
    assert ctrl.n_redetect_calls > 0
    assert ctrl.memory.has_anchor

    ctrl.reset()
    assert ctrl.phase == IDLE
    assert ctrl.n_redetect_calls == 0
    assert ctrl.n_switches == 0
    assert ctrl.n_distractor_seeds == 0
    assert not ctrl.memory.has_anchor
    assert ctrl.memory.n_recent == 0
    assert ctrl.memory.n_distractor == 0
