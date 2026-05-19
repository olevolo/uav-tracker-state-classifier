"""policy.py — SALT-RD runtime decision policy.

Converts SALTRD output probabilities into concrete tracker control decisions:
  - full SGLATrack inference vs. fast path
  - search-region expansion triggers
  - re-initialisation recommendations
  - dynamic scene class override

Design constraint: must run in < 0.5 ms on CPU (MBP M2) after GRU forward pass.

Usage (offline replay):
    python -m salt_r.policy --probs-json predictions.json --npz dataset.npz --output metrics.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackerAction:
    """Action recommendations output by the SALT-RD policy for one frame."""
    compute_mode: str = "full"         # "full" | "cheap"
    template_update: str = "allow"     # "allow" | "verify" | "block"
    recovery_action: str = "none"      # "none" | "run" | "abstain"
    search_mode: str = "normal"        # "normal" | "expand"
    confidence: float = 1.0           # policy confidence in this action set [0,1]
    triggered_by: list[str] = field(default_factory=list)  # which heads triggered


@dataclass
class RiskThresholds:
    """Configurable probability thresholds for each policy decision."""
    false_confirmed_block:             float = 0.70
    false_confirmed_verify:            float = 0.40
    failure_in_5_warn:                 float = 0.50
    recoverable_run:                   float = 0.60
    recoverable_abstain:               float = 0.40
    hard_dynamic_full_compute:         float = 0.65
    needs_full_compute:                float = 0.25
    # Recovery only runs when false_confirmed is below this (avoid re-init to same distractor)
    false_confirmed_max_for_recovery:  float = 0.40


DEFAULT_THRESHOLDS = RiskThresholds()


# ---------------------------------------------------------------------------
# Core policy
# ---------------------------------------------------------------------------

def apply_policy(
    probs: dict[str, float],
    thresholds: RiskThresholds | None = None,
) -> TrackerAction:
    """Map SALTRD head probabilities to a TrackerAction for this frame.

    Decision priority:
    1. false_confirmed dominates — blocks everything if high
    2. hard_dynamic_scene → full compute
    3. needs_full_compute < threshold → cheap mode
    4. recoverable → run recovery (if false_confirmed is low)

    This policy is designed for offline replay simulation.
    Runtime integration must validate regret before deployment.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    p_fc   = probs.get("false_confirmed", 0.0)
    p_fail = probs.get("failure_in_5", 0.0)
    p_rec  = probs.get("recoverable", 0.0)
    p_dyn  = probs.get("hard_dynamic_scene", 0.0)
    p_full = probs.get("needs_full_compute", 0.5)

    action = TrackerAction()
    triggered: list[str] = []

    # 1. false_confirmed dominates
    if p_fc > thresholds.false_confirmed_block:
        action.template_update = "block"
        action.recovery_action = "abstain"   # don't reinit — we don't know the right target
        action.compute_mode = "full"         # need full compute to disambiguate
        action.confidence = p_fc
        triggered.append(f"false_confirmed={p_fc:.2f}")

    elif p_fc > thresholds.false_confirmed_verify:
        action.template_update = "verify"
        triggered.append(f"false_confirmed_warn={p_fc:.2f}")

    # 2. hard_dynamic_scene → full compute + conservative search
    if p_dyn > thresholds.hard_dynamic_full_compute:
        action.compute_mode = "full"
        action.search_mode = "expand"
        if action.template_update == "allow":
            action.template_update = "verify"
        triggered.append(f"hard_dynamic={p_dyn:.2f}")

    # 3. cheap compute if low risk
    elif p_full < thresholds.needs_full_compute and p_fail < 0.20:
        action.compute_mode = "cheap"
        triggered.append(f"cheap_ok(full_p={p_full:.2f})")

    # 4. recovery
    if p_rec > thresholds.recoverable_run and p_fc < thresholds.false_confirmed_max_for_recovery:
        action.recovery_action = "run"
        triggered.append(f"recoverable={p_rec:.2f}")
    elif p_rec > thresholds.recoverable_abstain and p_fc >= thresholds.false_confirmed_max_for_recovery:
        action.recovery_action = "abstain"
        triggered.append(f"recovery_blocked_by_fc")

    action.triggered_by = triggered
    return action


# ---------------------------------------------------------------------------
# Offline replay simulation
# ---------------------------------------------------------------------------

def replay_policy(
    probs_seq: list[dict[str, float]],
    iou_trace: np.ndarray,
    thresholds: RiskThresholds | None = None,
) -> dict[str, Any]:
    """Simulate policy decisions over a full sequence offline.

    Args:
        probs_seq:  per-frame probability dicts for one sequence
        iou_trace:  GT IoU per frame, shape (T,)
        thresholds: optional custom thresholds (defaults to DEFAULT_THRESHOLDS)

    Returns:
        dict with:
          actions: list[TrackerAction]
          wrong_reinit_rate: float  — fraction of recovery attempts at IoU < 0.5
          template_blocked_rate: float — fraction of frames where update blocked
          template_corruption_rate: float — fraction of allowed updates at IoU < 0.5
          compute_cheap_rate: float — fraction of frames using cheap mode
          abstention_gain: float — mean IoU of frames where update was blocked
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    n = len(probs_seq)
    iou = np.asarray(iou_trace, dtype=float)
    assert len(iou) == n, f"probs_seq length {n} != iou_trace length {len(iou)}"

    actions: list[TrackerAction] = []

    recovery_attempts = 0
    wrong_reinits = 0

    blocked_frames = 0
    blocked_iou_sum = 0.0

    allowed_updates = 0
    corrupt_updates = 0

    cheap_frames = 0

    for t, frame_probs in enumerate(probs_seq):
        a = apply_policy(frame_probs, thresholds)
        actions.append(a)
        frame_iou = float(iou[t])

        # recovery quality
        if a.recovery_action == "run":
            recovery_attempts += 1
            if frame_iou < 0.5:
                wrong_reinits += 1

        # template update tracking
        if a.template_update == "block":
            blocked_frames += 1
            blocked_iou_sum += frame_iou
        elif a.template_update == "allow":
            allowed_updates += 1
            if frame_iou < 0.5:
                corrupt_updates += 1

        # compute mode
        if a.compute_mode == "cheap":
            cheap_frames += 1

    wrong_reinit_rate       = wrong_reinits / recovery_attempts if recovery_attempts > 0 else 0.0
    template_blocked_rate   = blocked_frames / n if n > 0 else 0.0
    template_corruption_rate = corrupt_updates / allowed_updates if allowed_updates > 0 else 0.0
    compute_cheap_rate      = cheap_frames / n if n > 0 else 0.0
    abstention_gain         = blocked_iou_sum / blocked_frames if blocked_frames > 0 else 0.0

    return {
        "actions": actions,
        "wrong_reinit_rate": wrong_reinit_rate,
        "template_blocked_rate": template_blocked_rate,
        "template_corruption_rate": template_corruption_rate,
        "compute_cheap_rate": compute_cheap_rate,
        "abstention_gain": abstention_gain,
    }


# ---------------------------------------------------------------------------
# CLI — batch offline evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    """Evaluate policy on all sequences in a predictions JSON or NPZ checkpoint."""
    parser = argparse.ArgumentParser(
        description="SALT-RD policy offline replay simulation"
    )
    parser.add_argument("--probs-json", help="JSON file with per-sequence per-frame probs")
    parser.add_argument("--npz",        help="NPZ dataset path for iou_traces")
    parser.add_argument("--thresholds", help="JSON file with custom threshold values")
    parser.add_argument("--output",     help="Output JSON path for policy metrics")
    args = parser.parse_args()

    print("Policy offline replay — run after eval.py produces predictions JSON")

    # Load custom thresholds if provided
    thresholds = DEFAULT_THRESHOLDS
    if args.thresholds:
        with open(args.thresholds) as f:
            t_dict = json.load(f)
        thresholds = RiskThresholds(**t_dict)
        print(f"Loaded custom thresholds from {args.thresholds}")

    if not args.probs_json:
        print("No --probs-json provided. Nothing to evaluate.")
        return

    with open(args.probs_json) as f:
        all_probs: dict[str, list[dict[str, float]]] = json.load(f)
    print(f"Loaded {len(all_probs)} sequences from {args.probs_json}")

    iou_data: dict[str, np.ndarray] = {}
    if args.npz:
        npz = np.load(args.npz, allow_pickle=True)
        for key in npz.files:
            iou_data[key] = npz[key]
        print(f"Loaded IoU traces for {len(iou_data)} sequences from {args.npz}")

    aggregate: dict[str, list[float]] = {
        "wrong_reinit_rate": [],
        "template_blocked_rate": [],
        "template_corruption_rate": [],
        "compute_cheap_rate": [],
        "abstention_gain": [],
    }

    for seq_name, probs_seq in all_probs.items():
        if seq_name in iou_data:
            iou_trace = iou_data[seq_name]
        else:
            # Fallback: assume perfect tracking when no GT available
            iou_trace = np.ones(len(probs_seq), dtype=float)

        metrics = replay_policy(probs_seq, iou_trace, thresholds)
        for k in aggregate:
            aggregate[k].append(metrics[k])

    summary = {k: float(np.mean(v)) for k, v in aggregate.items() if v}
    summary["num_sequences"] = len(all_probs)

    print("\n=== Policy Offline Metrics (mean across sequences) ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()
