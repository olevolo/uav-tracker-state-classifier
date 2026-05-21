"""calibrate_policy.py — Phase 7 temperature scaling for SALTRDPolicyNet.

Post-hoc calibration of the policy model's output heads using temperature
scaling on the validation split.  Optimises Expected Calibration Error (ECE)
via grid + ternary search (no scipy dependency).

Usage::

    python -m salt_r.calibrate_policy \\
        --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \\
        --oracle saltr/results/reinit_oracle_dataset.npz \\
        --split val \\
        --output saltr/results/calibration_val_policy_reinit_v1.json
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
    _REJECT_REINIT_CLASS_IDX,
)

# ---------------------------------------------------------------------------
# ECE helper
# ---------------------------------------------------------------------------


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins).

    Parameters
    ----------
    probs:
        Predicted probabilities in [0, 1], shape (N,) — for binary heads or
        max-prob confidence for multiclass.
    labels:
        Binary ground-truth (0/1), shape (N,).
    n_bins:
        Number of equal-width bins.

    Returns
    -------
    float — weighted mean |accuracy - confidence| across bins.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(labels[mask].mean())
        bin_conf = float(probs[mask].mean())
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / max(n, 1))


def _ece_multiclass(
    softmax_probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 10
) -> float:
    """ECE for multiclass: confidence = max softmax probability, correct = argmax == true."""
    confidence = softmax_probs.max(axis=1)
    predicted = softmax_probs.argmax(axis=1)
    correct = (predicted == true_labels).astype(float)
    return _ece(confidence, correct, n_bins=n_bins)


# ---------------------------------------------------------------------------
# Temperature scaling (NLL minimisation — no scipy)
# ---------------------------------------------------------------------------

def _fit_temperature_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fit scalar temperature T for binary head by minimising NLL (ECE-based).

    Returns T=1.0 if degenerate or already calibrated.
    """
    eps = 1e-6
    p_safe = np.clip(y_pred.astype(np.float64), eps, 1.0 - eps)
    logits = np.log(p_safe / (1.0 - p_safe))

    def nll_at_T(T: float) -> float:
        T = max(T, eps)
        p_cal = 1.0 / (1.0 + np.exp(-logits / T))
        p_cal = np.clip(p_cal, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal)))

    # Grid search
    T_grid = np.linspace(0.1, 10.0, 100)
    best_T = float(T_grid[np.argmin([nll_at_T(T) for T in T_grid])])

    # Ternary search refinement
    lo, hi = max(0.05, best_T - 1.0), best_T + 1.0
    for _ in range(50):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll_at_T(m1) < nll_at_T(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


def _apply_temperature_binary(y_pred: np.ndarray, T: float) -> np.ndarray:
    """Apply binary temperature scaling: p_cal = sigmoid(logit(p) / T)."""
    eps = 1e-6
    p_safe = np.clip(y_pred.astype(np.float64), eps, 1.0 - eps)
    logits = np.log(p_safe / (1.0 - p_safe))
    p_cal = 1.0 / (1.0 + np.exp(-logits / T))
    return np.clip(p_cal, 0.0, 1.0).astype(np.float32)


def _fit_temperature_multiclass(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a single scalar temperature for a multiclass head.

    Minimises NLL: softmax(logits / T).
    Grid + ternary search.
    """
    eps = 1e-6

    def nll_at_T(T: float) -> float:
        T = max(T, eps)
        scaled = logits / T
        # log-softmax numerically stable
        log_sum_exp = np.log(np.sum(np.exp(scaled - scaled.max(axis=1, keepdims=True)), axis=1))
        log_softmax = scaled - scaled.max(axis=1, keepdims=True) - log_sum_exp[:, None]
        # Per-sample NLL: -log_softmax at true class
        nll = 0.0
        for i, c in enumerate(y_true):
            nll -= log_softmax[i, int(c)]
        return float(nll / max(len(y_true), 1))

    T_grid = np.linspace(0.1, 10.0, 100)
    best_T = float(T_grid[np.argmin([nll_at_T(T) for T in T_grid])])

    lo, hi = max(0.05, best_T - 1.0), best_T + 1.0
    for _ in range(50):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll_at_T(m1) < nll_at_T(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


def _apply_temperature_multiclass(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to logits, return softmax probabilities."""
    T = max(T, 1e-6)
    scaled = logits / T
    exp_shifted = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return (exp_shifted / exp_shifted.sum(axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_policy_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[SALTRDPolicyNet, Dict[str, Any]]:
    """Load SALTRDPolicyNet from checkpoint.

    Returns (model, checkpoint_metadata).
    """
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
# Inference pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_inference(
    model: SALTRDPolicyNet,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model forward on all batches.

    Returns
    -------
    recovery_logits : (N, 4) raw logits for recovery_action head
    candidate_probs : (N,)   sigmoid probability for candidate_score head
    true_labels     : (N,)   integer recovery action labels
    """
    model = model.to(device)
    model.eval()
    all_recovery_logits: List[np.ndarray] = []
    all_candidate_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        outputs = model(x)

        # recovery action: expect raw logits (B, 4) from action_logits["recovery"]
        recovery = outputs["action_logits"]["recovery"]  # (B, 4)
        all_recovery_logits.append(recovery.cpu().numpy())

        # candidate_score: scalar (B,) or None
        candidate = outputs.get("candidate_score")
        if candidate is not None:
            all_candidate_probs.append(candidate.cpu().numpy().reshape(-1))
        else:
            all_candidate_probs.append(np.full(x.shape[0], 0.5, dtype=np.float32))

        all_labels.append(y.numpy())

    recovery_logits = np.concatenate(all_recovery_logits, axis=0)
    candidate_probs = np.concatenate(all_candidate_probs, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    return recovery_logits, candidate_probs, true_labels


# ---------------------------------------------------------------------------
# Calibration entry point
# ---------------------------------------------------------------------------

def calibrate(
    checkpoint_path: str,
    oracle_path: str,
    split: str = "val",
    output_path: Optional[str] = None,
    window_size: Optional[int] = None,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run temperature calibration and return results dict."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    t_start = time.time()

    print(f"[calibrate_policy] checkpoint={checkpoint_path}  split={split}", flush=True)

    # Load model
    model, ckpt_meta = _load_policy_model(checkpoint_path, device=device)
    ckpt_window_size = window_size or int(ckpt_meta.get("window_size", 20))
    feature_schema = str(ckpt_meta.get("feature_schema", FEATURE_SCHEMA))

    # Load dataset
    ds = OracleReinitDataset(oracle_path, split=split, window_size=ckpt_window_size)
    if len(ds) == 0:
        raise RuntimeError(f"Split '{split}' is empty in {oracle_path}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Run inference — get raw logits/probs before calibration
    print("[calibrate_policy] Running inference ...", flush=True)
    recovery_logits, candidate_probs, true_labels = _run_inference(model, loader, device)

    # Recovery action softmax probs (before calibration)
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    recovery_probs_before = _softmax(recovery_logits)

    # ------------------------------------------------------------------
    # Head: recovery_action — ECE before
    # ------------------------------------------------------------------
    ece_recovery_before = _ece_multiclass(recovery_probs_before, true_labels)

    # Fit temperature for recovery_action
    print("[calibrate_policy] Fitting temperature for recovery_action ...", flush=True)
    T_recovery = _fit_temperature_multiclass(recovery_logits, true_labels)
    recovery_probs_after = _apply_temperature_multiclass(recovery_logits, T_recovery)
    ece_recovery_after = _ece_multiclass(recovery_probs_after, true_labels)

    # Guard: skip temperature if it degrades ECE — pre-calibration may already pass gate
    if ece_recovery_after >= ece_recovery_before:
        print(
            f"  [calibrate_policy] T={T_recovery:.4f} degrades ECE "
            f"({ece_recovery_before:.4f} → {ece_recovery_after:.4f}) — reverting to T=1.0",
            flush=True,
        )
        T_recovery = 1.0
        recovery_probs_after = recovery_probs_before
        ece_recovery_after = ece_recovery_before

    print(
        f"  recovery_action:  T={T_recovery:.4f}  "
        f"ECE {ece_recovery_before:.4f} → {ece_recovery_after:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Head: candidate_score — treat as binary
    # ------------------------------------------------------------------
    # Binary label: is REINIT or REJECT_REINIT (any non-NONE action)
    candidate_labels = (true_labels > 0).astype(np.float32)
    ece_candidate_before = _ece(candidate_probs, candidate_labels)

    print("[calibrate_policy] Fitting temperature for candidate_score ...", flush=True)
    T_candidate = _fit_temperature_binary(candidate_labels, candidate_probs)
    candidate_probs_after = _apply_temperature_binary(candidate_probs, T_candidate)
    ece_candidate_after = _ece(candidate_probs_after, candidate_labels)

    print(
        f"  candidate_score:  T={T_candidate:.4f}  "
        f"ECE {ece_candidate_before:.4f} → {ece_candidate_after:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # GO/NO-GO condition: recovery_action ECE <= 0.10
    # ------------------------------------------------------------------
    if ece_recovery_after <= 0.10:
        go_nogo = "GO"
    elif ece_recovery_after <= 0.15:
        go_nogo = "BORDERLINE"
    else:
        go_nogo = "NO-GO"

    print(f"\n[calibrate_policy] GO/NO-GO: {go_nogo}  (recovery ECE_after={ece_recovery_after:.4f})", flush=True)

    wall_time = time.time() - t_start
    git_commit = _git_commit()

    results: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "oracle": str(oracle_path),
        "feature_schema": feature_schema,
        "split": split,
        "n_samples": int(len(ds)),
        "heads": {
            "recovery_action": {
                "ece_before": float(ece_recovery_before),
                "ece_after": float(ece_recovery_after),
                "temperature": float(T_recovery),
                "method": "temperature_scaling_multiclass",
            },
            "candidate_score": {
                "ece_before": float(ece_candidate_before),
                "ece_after": float(ece_candidate_after),
                "temperature": float(T_candidate),
                "method": "temperature",
            },
        },
        "go_no_go": go_nogo,
        "go_criteria": {
            "recovery_action_ece_threshold": 0.10,
            "recovery_action_ece_after": float(ece_recovery_after),
        },
        "git_commit": git_commit,
        "feature_schema_version": feature_schema,
        "created_at": datetime.utcnow().isoformat(),
        "random_seed": seed,
        "wall_time_s": round(wall_time, 2),
    }

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(results), fh, indent=2)
        print(f"\n[calibrate_policy] Results written to: {output_path}", flush=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for SALT-RD policy model calibration."""
    parser = argparse.ArgumentParser(
        description="Temperature-scale SALTRDPolicyNet heads on the val split."
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
        help="Dataset split to calibrate on (default: val).",
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
        "--batch-size", type=int, default=256, help="Batch size for inference."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    calibrate(
        checkpoint_path=args.checkpoint,
        oracle_path=args.oracle,
        split=args.split,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
