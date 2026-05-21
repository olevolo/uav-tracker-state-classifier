"""rollout_policy.py — Phase 8 offline pseudo-rollout for SALTRDPolicyNet.

Simulates what would happen if the policy fired at each frame.  For each
sequence in the requested split, runs the policy model forward and "applies"
reinit at frames where the model outputs REINIT with high confidence.
Compares the resulting AUC to the baseline (unmodified iou_trace).

Hard subset delta >= +0.03 AND changed_bbox_rate > 0.005 → GO

Usage::

    python -m salt_r.rollout_policy \\
        --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \\
        --oracle saltr/results/reinit_oracle_dataset.npz \\
        --split val \\
        --output saltr/results/rollout_val_policy_reinit_v1.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def _ensure_salt_r_on_path() -> None:
    salt_r_src = str(Path(__file__).parents[1])
    if salt_r_src not in sys.path:
        sys.path.insert(0, salt_r_src)


_ensure_salt_r_on_path()

from salt_r.policy_model import SALTRDPolicyNet  # noqa: E402
from salt_r.train_policy import (  # noqa: E402
    OracleReinitDataset,
    N_FEATURES,
    FEATURE_SCHEMA,
    _git_commit,
    _json_safe,
    _project_root,
    _REINIT_CLASS_IDX,
)

# ---------------------------------------------------------------------------
# Hard sequences definition
# ---------------------------------------------------------------------------

HARD_SEQUENCES: List[str] = [
    "uav123/bike2",
    "uav123/uav2",
    "uav123/uav4",
    "uav123/uav6",
    "dtb70/Gull2",
    "dtb70/Sheep1",
    "dtb70/StreetBasketball1",
]

# ---------------------------------------------------------------------------
# AUC helper (matches oracle_action_audit.py)
# ---------------------------------------------------------------------------

def _compute_auc_from_iou(iou_trace: np.ndarray) -> float:
    """Success AUC: integral of success(tau) for tau in [0,1], 21-step trapz."""
    if len(iou_trace) == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, 21)
    success_rates = np.array(
        [float(np.mean(iou_trace >= t)) for t in thresholds], dtype=np.float64
    )
    return float(np.trapz(success_rates, thresholds))


# ---------------------------------------------------------------------------
# Model loading (shared with calibrate_policy)
# ---------------------------------------------------------------------------

def _load_policy_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[SALTRDPolicyNet, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    window_size = int(ckpt.get("window_size", 20))
    hidden_size = int(ckpt.get("hidden_size", 64))
    n_layers = int(ckpt.get("n_layers", 2))
    n_features = int(ckpt.get("n_features", N_FEATURES))

    model = SALTRDPolicyNet(
        n_features=n_features,
        hidden_size=hidden_size,
        n_layers=n_layers,
        window_size=window_size,
    )
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model.to(device), ckpt


# ---------------------------------------------------------------------------
# Per-sequence rollout
# ---------------------------------------------------------------------------

def _rollout_sequence(
    model: SALTRDPolicyNet,
    features: np.ndarray,
    iou_trace: np.ndarray,
    window_size: int,
    device: str,
    reinit_confidence_threshold: float = 0.5,
    reinit_iou_value: float = 0.8,
    reinit_duration_frames: int = 10,
) -> Dict[str, Any]:
    """Run offline pseudo-rollout for one sequence.

    Parameters
    ----------
    features:
        Per-frame feature matrix, shape (T, 28).
    iou_trace:
        Per-frame IoU values, shape (T,).
    window_size:
        GRU context window size.
    reinit_confidence_threshold:
        Minimum softmax probability for REINIT class to trigger intervention.
    reinit_iou_value:
        IoU value assumed after oracle reinit (default 0.8).
    reinit_duration_frames:
        Number of frames for which the reinit effect persists.

    Returns
    -------
    dict with baseline_auc, policy_auc, delta, reinit_frames,
    changed_bbox_frames.
    """
    n = len(iou_trace)
    if n == 0:
        return {
            "baseline_auc": 0.0,
            "policy_auc": 0.0,
            "delta": 0.0,
            "reinit_frames": [],
            "changed_bbox_frames": 0,
        }

    model = model.to(device)
    model.eval()

    # Build windowed feature array: (n, window_size, 28)
    windows = []
    for t in range(n):
        start = max(0, t - window_size + 1)
        window = features[start : t + 1]
        pad_len = window_size - len(window)
        if pad_len > 0:
            window = np.concatenate(
                [np.zeros((pad_len, N_FEATURES), dtype=np.float32), window],
                axis=0,
            )
        windows.append(window.astype(np.float32))

    x_batch = torch.tensor(
        np.stack(windows, axis=0), dtype=torch.float32, device=device
    )  # (n, window_size, 28)

    with torch.no_grad():
        outputs = model(x_batch)

    recovery_logits = outputs["action_logits"]["recovery"].cpu().numpy()  # (n, 4)

    # Softmax to get probabilities
    exp_shifted = np.exp(recovery_logits - recovery_logits.max(axis=1, keepdims=True))
    recovery_probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)  # (n, 4)

    reinit_probs = recovery_probs[:, _REINIT_CLASS_IDX]  # (n,)

    # Identify frames where model outputs REINIT with high confidence
    reinit_frames = [
        int(t) for t in range(n)
        if reinit_probs[t] >= reinit_confidence_threshold
    ]

    # Build policy iou_trace: at reinit frames, set next `reinit_duration_frames`
    # frames to reinit_iou_value
    policy_iou = iou_trace.copy().astype(np.float64)
    changed_bbox_frames = 0

    # Apply reinit effects in chronological order, tracking "cooldown" to avoid
    # overlapping interventions within the same burst
    reinit_cooldown = 0
    for t in range(n):
        if reinit_cooldown > 0:
            reinit_cooldown -= 1
            continue
        if t in set(reinit_frames):
            # Apply reinit: set next `reinit_duration_frames` frames to reinit_iou_value
            end = min(t + reinit_duration_frames, n)
            for f in range(t, end):
                if policy_iou[f] < reinit_iou_value:
                    policy_iou[f] = reinit_iou_value
                    changed_bbox_frames += 1
            reinit_cooldown = reinit_duration_frames - 1

    baseline_auc = _compute_auc_from_iou(iou_trace)
    policy_auc = _compute_auc_from_iou(policy_iou)
    delta = policy_auc - baseline_auc

    return {
        "baseline_auc": float(baseline_auc),
        "policy_auc": float(policy_auc),
        "delta": float(delta),
        "reinit_frames": reinit_frames,
        "changed_bbox_frames": int(changed_bbox_frames),
    }


# ---------------------------------------------------------------------------
# Load oracle dataset raw (for iou_trace access)
# ---------------------------------------------------------------------------

def _load_oracle_raw(
    oracle_path: str,
    split: str,
    window_size: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load per-sequence features and iou_trace from the oracle NPZ.

    The NPZ has flat arrays indexed by frame; we reconstruct per-sequence
    arrays using sequence_keys and frame_indices.

    Returns
    -------
    Dict[seq_key -> {"features": (T, 28), "iou_trace": (T,)}]
    """
    data = np.load(oracle_path, allow_pickle=True)
    features_all = data["features"].astype(np.float32)    # (M, 28)
    splits_all = data["splits"]                            # (M,)
    seq_keys_all = data["sequence_keys"]                   # (M,)
    frame_idx_all = data["frame_indices"].astype(np.int64) # (M,)

    # iou_trace may not be present in the oracle dataset; handle gracefully
    has_iou = "iou_trace" in data.files
    iou_all = data["iou_trace"].astype(np.float32) if has_iou else None  # (M,) or None

    # Filter to split
    mask = np.array([str(s) == split for s in splits_all])
    features_split = features_all[mask]
    seq_keys_split = seq_keys_all[mask]
    frame_idx_split = frame_idx_all[mask]
    iou_split = iou_all[mask] if has_iou else None

    # Group by sequence
    unique_keys: List[str] = []
    seen: Dict[str, int] = {}
    for k in seq_keys_split:
        k_str = str(k)
        if k_str not in seen:
            seen[k_str] = len(unique_keys)
            unique_keys.append(k_str)

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for seq_key in unique_keys:
        seq_mask = np.array([str(k) == seq_key for k in seq_keys_split])
        seq_feats = features_split[seq_mask]
        seq_frames = frame_idx_split[seq_mask]

        # Sort by frame index
        order = np.argsort(seq_frames)
        seq_feats = seq_feats[order]

        if has_iou:
            seq_iou = iou_split[seq_mask][order]
        else:
            # Fall back to zeros if iou_trace not available
            seq_iou = np.zeros(len(seq_feats), dtype=np.float32)

        result[seq_key] = {
            "features": seq_feats,
            "iou_trace": seq_iou,
        }

    return result


# ---------------------------------------------------------------------------
# Main rollout entry point
# ---------------------------------------------------------------------------

def rollout(
    checkpoint_path: str,
    oracle_path: str,
    split: str = "val",
    output_path: Optional[str] = None,
    device: str = "cpu",
    reinit_confidence_threshold: float = 0.5,
    reinit_iou_value: float = 0.8,
    reinit_duration_frames: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run offline pseudo-rollout for all sequences in the split."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    t_start = time.time()

    print(
        f"[rollout_policy] checkpoint={checkpoint_path}  split={split}", flush=True
    )

    # Load model
    model, ckpt_meta = _load_policy_model(checkpoint_path, device=device)
    window_size = int(ckpt_meta.get("window_size", 20))
    feature_schema = str(ckpt_meta.get("feature_schema", FEATURE_SCHEMA))

    print(f"[rollout_policy] window_size={window_size}", flush=True)

    # Load oracle data
    print("[rollout_policy] Loading oracle dataset ...", flush=True)
    seq_data = _load_oracle_raw(oracle_path, split=split, window_size=window_size)
    n_seqs = len(seq_data)
    print(f"[rollout_policy] {n_seqs} sequences in split '{split}'", flush=True)

    if n_seqs == 0:
        raise RuntimeError(f"Split '{split}' is empty in {oracle_path}")

    # Per-sequence rollout
    seq_results: Dict[str, Any] = {}
    total_frames = 0

    for seq_key, seq in seq_data.items():
        features = seq["features"]      # (T, 28)
        iou_trace = seq["iou_trace"]    # (T,)
        total_frames += len(features)

        res = _rollout_sequence(
            model=model,
            features=features,
            iou_trace=iou_trace,
            window_size=window_size,
            device=device,
            reinit_confidence_threshold=reinit_confidence_threshold,
            reinit_iou_value=reinit_iou_value,
            reinit_duration_frames=reinit_duration_frames,
        )

        seq_results[seq_key] = {
            "baseline_auc": round(float(res["baseline_auc"]), 5),
            "policy_auc": round(float(res["policy_auc"]), 5),
            "delta": round(float(res["delta"]), 5),
            "reinit_frames": res["reinit_frames"],
            "changed_bbox_frames": int(res["changed_bbox_frames"]),
        }

        if seq_key in HARD_SEQUENCES or len(res["reinit_frames"]) > 0:
            print(
                f"  {seq_key:<40}  "
                f"base={res['baseline_auc']:.4f}  "
                f"policy={res['policy_auc']:.4f}  "
                f"delta={res['delta']:+.4f}  "
                f"reinits={len(res['reinit_frames'])}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------
    baseline_aucs = [r["baseline_auc"] for r in seq_results.values()]
    policy_aucs = [r["policy_auc"] for r in seq_results.values()]
    deltas = [r["delta"] for r in seq_results.values()]
    total_changed = sum(r["changed_bbox_frames"] for r in seq_results.values())
    changed_bbox_rate = float(total_changed) / max(total_frames, 1)

    mean_baseline_auc = float(np.mean(baseline_aucs)) if baseline_aucs else 0.0
    mean_policy_auc = float(np.mean(policy_aucs)) if policy_aucs else 0.0
    mean_delta = float(np.mean(deltas)) if deltas else 0.0

    # Hard subset aggregation
    hard_available = [s for s in HARD_SEQUENCES if s in seq_results]
    hard_baseline_aucs = [seq_results[s]["baseline_auc"] for s in hard_available]
    hard_policy_aucs = [seq_results[s]["policy_auc"] for s in hard_available]
    hard_deltas = [seq_results[s]["delta"] for s in hard_available]

    hard_baseline_auc = float(np.mean(hard_baseline_aucs)) if hard_baseline_aucs else float("nan")
    hard_policy_auc = float(np.mean(hard_policy_aucs)) if hard_policy_aucs else float("nan")
    hard_delta = float(np.mean(hard_deltas)) if hard_deltas else float("nan")

    # ------------------------------------------------------------------
    # GO/NO-GO: hard subset delta >= +0.03 AND changed_bbox_rate > 0.005
    # ------------------------------------------------------------------
    hard_delta_ok = not np.isnan(hard_delta) and hard_delta >= 0.03
    rate_ok = changed_bbox_rate > 0.005
    go_nogo = "GO" if (hard_delta_ok and rate_ok) else "NO-GO"

    print(f"\n[rollout_policy] mean_delta={mean_delta:+.4f}", flush=True)
    print(f"[rollout_policy] hard_delta={hard_delta:+.4f}  (need >= +0.03)", flush=True)
    print(f"[rollout_policy] changed_bbox_rate={changed_bbox_rate:.5f}  (need > 0.005)", flush=True)
    print(f"[rollout_policy] GO/NO-GO: {go_nogo}", flush=True)

    wall_time = time.time() - t_start
    git_commit = _git_commit()

    aggregate: Dict[str, Any] = {
        "mean_baseline_auc": round(mean_baseline_auc, 5),
        "mean_policy_auc": round(mean_policy_auc, 5),
        "mean_delta": round(mean_delta, 5),
        "total_changed_bbox_frames": int(total_changed),
        "total_frames": int(total_frames),
        "changed_bbox_rate": round(changed_bbox_rate, 6),
        "n_sequences": int(n_seqs),
        "hard_subset_sequences": hard_available,
        "hard_subset_baseline_auc": round(float(hard_baseline_auc), 5) if not np.isnan(hard_baseline_auc) else None,
        "hard_subset_policy_auc": round(float(hard_policy_auc), 5) if not np.isnan(hard_policy_auc) else None,
        "hard_subset_delta": round(float(hard_delta), 5) if not np.isnan(hard_delta) else None,
    }

    go_criteria: Dict[str, Any] = {
        "hard_subset_delta_threshold": 0.03,
        "hard_subset_delta_actual": round(float(hard_delta), 5) if not np.isnan(hard_delta) else None,
        "hard_subset_delta_ok": bool(hard_delta_ok),
        "changed_bbox_rate_threshold": 0.005,
        "changed_bbox_rate_actual": round(changed_bbox_rate, 6),
        "changed_bbox_rate_ok": bool(rate_ok),
    }

    results: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "oracle": str(oracle_path),
        "split": split,
        "feature_schema": feature_schema,
        "sequences": seq_results,
        "aggregate": aggregate,
        "go_no_go": go_nogo,
        "go_criteria": go_criteria,
        "rollout_params": {
            "reinit_confidence_threshold": reinit_confidence_threshold,
            "reinit_iou_value": reinit_iou_value,
            "reinit_duration_frames": reinit_duration_frames,
            "window_size": window_size,
        },
        "git_commit": git_commit,
        "feature_schema_version": feature_schema,
        "created_at": datetime.utcnow().isoformat(),
        "random_seed": seed,
        "wall_time_s": round(wall_time, 2),
    }

    # Save JSON
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(results), fh, indent=2)
        print(f"\n[rollout_policy] Results written to: {output_path}", flush=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for SALT-RD policy offline rollout."""
    parser = argparse.ArgumentParser(
        description="Offline pseudo-rollout evaluation of SALTRDPolicyNet."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to saltrd_policy_best.pt checkpoint.",
    )
    parser.add_argument(
        "--oracle",
        required=True,
        help="Path to reinit_oracle_dataset.npz.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "diagnostic"],
        help="Dataset split to roll out on (default: val).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device (default: cpu)."
    )
    parser.add_argument(
        "--reinit-threshold",
        type=float,
        default=0.5,
        help="Minimum REINIT softmax probability to trigger intervention (default: 0.5).",
    )
    parser.add_argument(
        "--reinit-iou",
        type=float,
        default=0.8,
        help="Assumed IoU after oracle reinit (default: 0.8).",
    )
    parser.add_argument(
        "--reinit-duration",
        type=int,
        default=10,
        help="Frames for which reinit effect persists (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rollout(
        checkpoint_path=args.checkpoint,
        oracle_path=args.oracle,
        split=args.split,
        output_path=args.output,
        device=args.device,
        reinit_confidence_threshold=args.reinit_threshold,
        reinit_iou_value=args.reinit_iou,
        reinit_duration_frames=args.reinit_duration,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
