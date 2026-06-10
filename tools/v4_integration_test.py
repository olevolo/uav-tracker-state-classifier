#!/usr/bin/env python
"""CSC-v4 INTEGRATION smoke — wires the REAL v4 modules together (no stubs) and runs a
synthetic frame loop through the full chain:
  build_v4_features -> CSCv4.predict -> extract_candidates -> CandidateVerifier
  -> SPRTGate -> V4Controller.decide -> ActionDecision
Catches cross-module interface mismatches the per-module duck-typed smokes hid.
CPU-only, synthetic data, no dataset/weights. Run: .venv/bin/python tools/v4_integration_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "salrtd" / "src", PROJECT_ROOT / "src", PROJECT_ROOT):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from csc_lib.csc.v4.v4types import (
    Candidate, ActionDecision, Action, ACTION_NAMES, DerivedStateV4,
)
from csc_lib.csc.v4 import features_v4 as F
from csc_lib.csc.v4.memory import PrototypeMemory
from csc_lib.csc.v4.verifier import extract_candidates, CandidateVerifier
from csc_lib.csc.v4.model_v4 import CSCv4
from csc_lib.csc.v4.sprt_gate import SPRTGate, expected_gain_gate, llr_from_evidence
from csc_lib.csc.v4.control_v4 import V4Controller, V4ControlConfig, MetaUpdater, abort_check

rng = np.random.default_rng(0)
EMB_D = 192
T = 32


def fake_row(loss: bool) -> dict:
    """Synthetic telemetry row; loss=True looks like a diffuse/true-loss frame."""
    if loss:
        return dict(confidence=0.017, apce=40.0, psr=900.0, response_entropy=4.6,
                    sm_local_top2_ratio=0.55, sm_local_peak_margin=0.18, sm_peak_distance=0.7,
                    sm_heatmap_mass_topk=0.3, sm_n_secondary=4.0, sm_peak_width=4.0,
                    sm_top1=0.4, sm_top2=0.25, last_cosine_sim=0.55, initial_template_sim=0.6,
                    appearance_drift=0.4, bbox=[300, 200, 40, 30])
    return dict(confidence=0.017, apce=180.0, psr=3000.0, response_entropy=3.2,
                sm_local_top2_ratio=0.12, sm_local_peak_margin=0.55, sm_peak_distance=0.3,
                sm_heatmap_mass_topk=0.8, sm_n_secondary=1.0, sm_peak_width=1.0,
                sm_top1=0.85, sm_top2=0.1, last_cosine_sim=0.9, initial_template_sim=0.9,
                appearance_drift=0.08, bbox=[300, 200, 40, 30])


def main() -> None:
    print("=== building real v4 components ===")
    # A1 features
    rows = [fake_row(rng.random() < 0.3) for _ in range(400)]
    cals = F.fit_v4_calibrators(rows)
    feat_dim = F.FEATURE_DIM_V4
    print(f"  features: FEATURE_DIM_V4={feat_dim}, calibrators={len(cals)}")

    # A6 model at the REAL feature dim (not the contract's example 20)
    model = CSCv4(feature_dim=feat_dim)
    model.eval()

    # A2 memory + A3 verifier + A7 sprt
    mem = PrototypeMemory()
    mem.update_anchor(rng.standard_normal(EMB_D).astype(np.float32))
    mem.update_recent(rng.standard_normal(EMB_D).astype(np.float32), 0)
    verifier = CandidateVerifier(mem)
    sprt = SPRTGate()

    # A10 controller wiring the real model/memory/verifier/sprt (redetector/sidecar optional=None)
    ctrl = V4Controller(model, mem, verifier, sprt, cfg=V4ControlConfig())
    print("  controller wired: model + memory + verifier + sprt")

    # ---- run a synthetic episode: 5 CC frames, then 8 LA frames ----
    feat_window: list[np.ndarray] = []
    prev = None
    actions_seen = []
    for t in range(13):
        loss = t >= 5
        row = fake_row(loss)
        fv = F.build_v4_features(row, cals, prev=prev); prev = row
        feat_window.append(fv)
        win = np.stack(feat_window[-T:], axis=0)
        if win.shape[0] < T:                       # left-pad causal window
            win = np.concatenate([np.repeat(win[:1], T - win.shape[0], 0), win], 0)
        x = torch.from_numpy(win[None].astype(np.float32))   # (1,T,F)

        pred = model.predict(x)                    # A6 -> V4Prediction
        assert hasattr(pred, "action_utility") and set(pred.action_utility) == set(ACTION_NAMES), \
            "model.action_utility keys must equal ACTION_NAMES"

        # A3 candidates from a synthetic score map + verify against real memory
        score_map = rng.random((16, 16)).astype(np.float32)
        score_map[4, 6] = 5.0; score_map[10, 11] = 3.0     # a dominant + a secondary peak
        cands = extract_candidates(score_map, search_bbox=(250, 160, 160, 160),
                                   image_size=(1280, 720), k=5)
        for c in cands:
            c.embedding = rng.standard_normal(EMB_D).astype(np.float32)
            verifier.annotate(c)                   # fills sim_to_* from real memory

        # A7 standalone sanity: SPRT + expected-gain on the real prediction
        verdict = sprt.update(llr_from_evidence(row))
        gate_action, gain = expected_gain_gate(pred.action_utility, min_gain=0.0)

        # A10 the real decision
        frame_ctx = dict(frame_idx=t, features=x, bbox=row["bbox"],
                         velocity_prior=(2.0, 0.0), frame=None, crops=None,
                         template_hint=row["bbox"])
        decision = ctrl.decide(pred, row, cands, frame_ctx)
        assert isinstance(decision, ActionDecision), "decide must return ActionDecision"
        actions_seen.append(Action(decision.action).name)
        if t in (0, 5, 12):
            print(f"  t={t:2d} loss={int(loss)} ds={DerivedStateV4(pred.derived_state).name:3s} "
                  f"sprt={verdict:10s} egain={ACTION_NAMES[gate_action] if isinstance(gate_action,int) else gate_action}"
                  f"  -> DECIDE={Action(decision.action).name} ({decision.reason[:42]})")

    print(f"\n  actions over episode: {actions_seen}")
    # MetaUpdater + abort_check unit wiring
    mu = MetaUpdater(V4ControlConfig())
    _ = mu.template_update_safe(pred, fake_row(False))
    _ = abort_check(fake_row(True), fake_row(False), V4ControlConfig())
    print("\nV4 INTEGRATION OK — full real-module chain runs end to end.")


if __name__ == "__main__":
    main()
