"""train_policy.py — SALT-RD Phase 6 policy model training.

Trains SALTRDPolicyNet on the oracle reinit dataset produced by oracle_actions.py.
Uses weighted cross-entropy to handle severe class imbalance (most frames = NONE).

Usage::

    python -m salt_r.train_policy \\
        --oracle saltr/results/reinit_oracle_dataset.npz \\
        --output saltr/checkpoints/policy_reinit_v1/ \\
        --epochs 80 \\
        --batch-size 64 \\
        --lr 3e-4 \\
        --window-size 20 \\
        --device auto \\
        --lambda-recovery 1.0 \\
        --lambda-candidate 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Path setup — ensure salt_r is importable when run as a script
# ---------------------------------------------------------------------------

def _ensure_salt_r_on_path() -> None:
    salt_r_src = str(Path(__file__).parents[1])  # saltr/src
    if salt_r_src not in sys.path:
        sys.path.insert(0, salt_r_src)


_ensure_salt_r_on_path()

from salt_r.policy_model import SALTRDPolicyNet, compute_loss, RECOVERY_ACTION_ORDER  # noqa: E402
from salt_r.actions import RecoveryAction  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_SCHEMA: str = "saltrd_v3_no_tsa_no_flow"
N_FEATURES: int = 28

# Recovery action class indices derived from RECOVERY_ACTION_ORDER in policy_model
# NONE=0, SCORE_CANDIDATES=1, REINIT=2, REJECT_REINIT=3
_REINIT_CLASS_IDX: int = RECOVERY_ACTION_ORDER.index(RecoveryAction.REINIT.value)
_REJECT_REINIT_CLASS_IDX: int = RECOVERY_ACTION_ORDER.index(RecoveryAction.REJECT_REINIT.value)
_NONE_CLASS_IDX: int = RECOVERY_ACTION_ORDER.index(RecoveryAction.NONE.value)

# Class weights: REINIT and REJECT_REINIT are 5x over NONE
# Order matches RECOVERY_ACTION_ORDER: NONE, SCORE_CANDIDATES, REINIT, REJECT_REINIT
_RECOVERY_CLASS_WEIGHTS: List[float] = [1.0, 1.0, 5.0, 5.0]

# Hard sequences for reference in summary
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
# Provenance helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root (parent of saltr/)."""
    return Path(__file__).parents[3]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_project_root()),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _json_safe(obj):
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, float):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OracleReinitDataset(Dataset):
    """Windowed sequence dataset from reinit_oracle_dataset.npz.

    The NPZ has flat arrays (M,) keyed by:
        features (M, 28), label_reinit (M,), label_reject (M,),
        utility (M,), splits (M,), sequence_keys (M,), frame_indices (M,)

    Each sample is a (window_size, 28) feature window for the last frame in
    the window.  Sequences are grouped to avoid cross-sequence windows.
    """

    def __init__(
        self,
        npz_path: str,
        split: str,
        window_size: int = 20,
    ) -> None:
        assert split in {"train", "val", "diagnostic"}, f"Unknown split: {split!r}"
        self.split = split
        self.window_size = window_size

        data = np.load(npz_path, allow_pickle=True)
        features_all: np.ndarray = data["features"].astype(np.float32)  # (M, 28)
        label_reinit: np.ndarray = data["label_reinit"].astype(np.int64)  # (M,)
        label_reject: np.ndarray = data["label_reject"].astype(np.int64)  # (M,)
        splits_all: np.ndarray = data["splits"]                           # (M,) str
        seq_keys_all: np.ndarray = data["sequence_keys"]                   # (M,) str
        frame_idx_all: np.ndarray = data["frame_indices"].astype(np.int64)  # (M,)

        # Build recovery_action label:
        # REINIT=2 if label_reinit, REJECT_REINIT=3 if label_reject, else NONE=0
        recovery_label = np.zeros(len(features_all), dtype=np.int64)
        recovery_label[label_reinit == 1] = _REINIT_CLASS_IDX
        recovery_label[label_reject == 1] = _REJECT_REINIT_CLASS_IDX
        # Where both are set, prefer REINIT
        both = (label_reinit == 1) & (label_reject == 1)
        recovery_label[both] = _REINIT_CLASS_IDX

        # Filter to requested split
        mask = np.array([str(s) == split for s in splits_all])
        features_split = features_all[mask]
        recovery_split = recovery_label[mask]
        seq_keys_split = seq_keys_all[mask]
        frame_idx_split = frame_idx_all[mask]

        # Group by sequence — build per-sequence arrays for windowing
        # Use frame_indices to find the correct in-sequence offset
        self._sequences: List[Tuple[np.ndarray, np.ndarray]] = []
        self._index: List[Tuple[int, int]] = []  # (seq_idx, local_frame_idx)

        unique_keys = []
        seen: Dict[str, int] = {}
        for k in seq_keys_split:
            k_str = str(k)
            if k_str not in seen:
                seen[k_str] = len(unique_keys)
                unique_keys.append(k_str)

        # For each unique sequence, collect frames in order of frame_indices
        for seq_key in unique_keys:
            seq_mask = np.array([str(k) == seq_key for k in seq_keys_split])
            seq_feats = features_split[seq_mask]    # (T_seq, 28)
            seq_labels = recovery_split[seq_mask]   # (T_seq,)
            seq_frames = frame_idx_split[seq_mask]  # (T_seq,)

            # Sort by frame index
            order = np.argsort(seq_frames)
            seq_feats = seq_feats[order]
            seq_labels = seq_labels[order]

            n_frames = len(seq_feats)
            if n_frames == 0:
                continue

            seq_idx = len(self._sequences)
            self._sequences.append((seq_feats, seq_labels))

            # All frames are valid; left-pad with zeros for early frames
            for t in range(n_frames):
                self._index.append((seq_idx, t))

        self._unique_seq_keys = unique_keys

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        seq_idx, t = self._index[idx]
        feats, labels = self._sequences[seq_idx]
        n_frames = feats.shape[0]

        # Build window with left-zero-padding
        start = t - self.window_size + 1
        if start >= 0:
            window = feats[start : t + 1]  # (window_size, 28)
        else:
            pad_len = -start
            window = np.concatenate(
                [np.zeros((pad_len, N_FEATURES), dtype=np.float32), feats[0 : t + 1]],
                axis=0,
            )  # (window_size, 28)

        target = labels[t]  # scalar int64
        return (
            torch.from_numpy(window.astype(np.float32)),
            torch.tensor(target, dtype=torch.long),
        )

    @property
    def labels(self) -> np.ndarray:
        """Flat array of integer class labels, one per sample (in index order)."""
        result = np.empty(len(self._index), dtype=np.int64)
        for i, (seq_idx, t) in enumerate(self._index):
            result[i] = self._sequences[seq_idx][1][t]
        return result

    def compute_class_distribution(self) -> Dict[str, int]:
        """Count frames per recovery action class."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for seq_feats, seq_labels in self._sequences:
            for v in seq_labels:
                counts[int(v)] = counts.get(int(v), 0) + 1
        return {
            "NONE": counts[0],
            "SCORE_CANDIDATES": counts[1],
            "REINIT": counts[2],
            "REJECT_REINIT": counts[3],
        }


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Metrics helpers (no sklearn)
# ---------------------------------------------------------------------------

def _macro_f1_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> float:
    """Macro-averaged F1 across all classes."""
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec > 0:
            f1s.append(2 * prec * rec / (prec + rec))
        else:
            f1s.append(0.0)
    return float(np.mean(f1s))


def _recall_for_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls: int,
) -> float:
    n_pos = int((y_true == cls).sum())
    if n_pos == 0:
        return float("nan")
    tp = int(((y_pred == cls) & (y_true == cls)).sum())
    return tp / n_pos


def _precision_for_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls: int,
) -> float:
    n_pred = int((y_pred == cls).sum())
    if n_pred == 0:
        return float("nan")
    tp = int(((y_pred == cls) & (y_true == cls)).sum())
    return tp / n_pred


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model: SALTRDPolicyNet,
    loader: DataLoader,
    class_weights: Tensor,
    lambda_recovery: float,
    lambda_candidate: float,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation; return loss + key metrics."""
    model.eval()
    all_logits_recovery: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)  # (B,) int64 class index

        # Convert integer class label to binary label_reinit and label_reject
        label_reinit = (y == _REINIT_CLASS_IDX).float()
        label_reject = (y == _REJECT_REINIT_CLASS_IDX).float()

        outputs = model(x)
        targets = {
            "label_reinit": label_reinit,
            "label_reject": label_reject,
        }
        loss_dict = compute_loss(
            outputs, targets,
            lambda_recovery=lambda_recovery,
            lambda_candidate=lambda_candidate,
        )
        loss = loss_dict["total"]
        total_loss += loss.item()
        n_batches += 1

        # Collect recovery action logits and labels
        recovery_logits = outputs["action_logits"]["recovery"]  # (B, 4)
        all_logits_recovery.append(recovery_logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    if not all_logits_recovery:
        return {"val_loss": avg_loss}

    logits_np = np.concatenate(all_logits_recovery, axis=0)    # (N, 4)
    labels_np = np.concatenate(all_labels, axis=0)              # (N,)
    pred_classes = np.argmax(logits_np, axis=1)                 # (N,)

    macro_f1 = _macro_f1_multiclass(labels_np, pred_classes)
    reinit_recall = _recall_for_class(labels_np, pred_classes, _REINIT_CLASS_IDX)
    reject_precision = _precision_for_class(labels_np, pred_classes, _REJECT_REINIT_CLASS_IDX)

    # REINIT AUPRC: binary classification REINIT vs rest (pure numpy, no sklearn)
    reinit_labels = (labels_np == _REINIT_CLASS_IDX).astype(int)
    # softmax logits → REINIT prob
    logits_shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    reinit_scores = exp_logits[:, _REINIT_CLASS_IDX] / exp_logits.sum(axis=1)
    if reinit_labels.sum() > 0:
        # Sort by descending score to build PR curve
        sort_idx = np.argsort(-reinit_scores)
        sorted_labels = reinit_labels[sort_idx]
        tp_cumsum = np.cumsum(sorted_labels)
        n_pos = reinit_labels.sum()
        precision_curve = tp_cumsum / np.arange(1, len(sorted_labels) + 1)
        recall_curve = tp_cumsum / n_pos
        # Prepend (recall=0, precision=1) sentinel
        precision_curve = np.concatenate([[1.0], precision_curve])
        recall_curve = np.concatenate([[0.0], recall_curve])
        reinit_auprc = float(np.trapz(precision_curve, recall_curve))
    else:
        reinit_auprc = 0.0

    return {
        "val_loss": avg_loss,
        "macro_f1": macro_f1,
        "reinit_recall": float(reinit_recall) if not np.isnan(reinit_recall) else 0.0,
        "reject_precision": float(reject_precision) if not np.isnan(reject_precision) else 0.0,
        "reinit_auprc": float(reinit_auprc),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    oracle_path: str,
    output_dir: str,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 3e-4,
    window_size: int = 20,
    device: str = "auto",
    lambda_recovery: float = 1.0,
    lambda_candidate: float = 0.5,
    patience: int = 15,
    seed: int = 42,
    hidden_size: int = 64,
    n_layers: int = 2,
) -> None:
    """Train SALTRDPolicyNet on oracle reinit dataset."""
    t_start = time.time()

    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev = _resolve_device(device)
    print(f"[train_policy] device={dev}  oracle={oracle_path}", flush=True)

    # ------------------------------------------------------------------
    # Datasets / loaders
    # ------------------------------------------------------------------
    # Oracle dataset labels: 'diagnostic' = training pool; 'val' = held-out validation.
    # Sequences in the NPZ are stored under their fold-selection priority split.
    print("[train_policy] Loading train split (diagnostic) ...", flush=True)
    train_ds = OracleReinitDataset(oracle_path, split="diagnostic", window_size=window_size)
    print(f"[train_policy] train samples: {len(train_ds):,}", flush=True)
    dist = train_ds.compute_class_distribution()
    print(f"[train_policy] train class distribution: {dist}", flush=True)

    print("[train_policy] Loading val split ...", flush=True)
    val_ds = OracleReinitDataset(oracle_path, split="val", window_size=window_size)
    print(f"[train_policy] val samples:   {len(val_ds):,}", flush=True)

    if len(train_ds) == 0:
        raise RuntimeError("Training split is empty — check the NPZ file.")

    # Event-balanced sampler: oversample REINIT 10× to ensure ~20-30% per batch
    sample_weights = np.where(
        train_ds.labels == _REINIT_CLASS_IDX,   # _REINIT_CLASS_IDX = 2
        10.0,   # oversample REINIT 10×
        1.0     # baseline weight for others
    )
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    ) if len(val_ds) > 0 else None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = SALTRDPolicyNet(
        n_features=N_FEATURES,
        hidden_size=hidden_size,
        n_layers=n_layers,
        window_size=window_size,
    ).to(dev)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_policy] SALTRDPolicyNet  params={total_params:,}", flush=True)

    # ------------------------------------------------------------------
    # Loss weights — inverse frequency, REINIT boosted to aid recall
    # ------------------------------------------------------------------
    dist = train_ds.compute_class_distribution()
    # Class order: NONE=0, SCORE_CANDIDATES=1, REINIT=2, REJECT_REINIT=3
    class_names = ["NONE", "SCORE_CANDIDATES", "REINIT", "REJECT_REINIT"]
    n_reinit   = max(dist.get("REINIT", 1), 1)
    n_reject   = max(dist.get("REJECT_REINIT", 1), 1)
    n_none     = max(dist.get("NONE", 1), 1)
    # SCORE_CANDIDATES has 0 training examples — set weight to 0 to ignore
    w_reject   = n_reinit / n_reject        # ≈ 0.023 (down-weight majority)
    w_none     = 1.0
    w_sc       = 0.0
    w_reinit   = min(n_reject / n_reinit, 8.0)  # cap at 8× to prevent over-prediction
    weights_list = [w_none, w_sc, w_reinit, w_reject]
    class_weights = torch.tensor(weights_list, dtype=torch.float32, device=dev)
    print(
        f"[train_policy] class weights  NONE={w_none:.2f}  SC={w_sc:.2f}"
        f"  REINIT={w_reinit:.2f}  REJECT={w_reject:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "saltrd_policy_best.pt"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_reinit_auprc = -1.0
    patience_counter = 0
    best_epoch = 0
    best_val_metrics: Dict[str, float] = {}
    epoch_history = []

    print(
        f"\n{'Epoch':>5} | {'TrainLoss':>9} | {'ValLoss':>8} | "
        f"{'ReinitRecall':>12} | {'RejectPrec':>10} | {'MacroF1':>7} | {'ReinitAUPRC':>11}",
        flush=True,
    )
    print("-" * 80, flush=True)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)  # (B,) int64 class index
            optimizer.zero_grad()
            outputs = model(x)
            # Weighted recovery CE directly — bypasses compute_loss to apply class_weights
            recovery_logits = outputs["action_logits"]["recovery"]  # (B, 4)
            recovery_loss = torch.nn.functional.cross_entropy(
                recovery_logits, y, weight=class_weights
            )
            loss = lambda_recovery * recovery_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            n_train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        # Validate
        if val_loader is not None:
            val_metrics = _validate(
                model, val_loader, class_weights,
                lambda_recovery, lambda_candidate, dev,
            )
        else:
            val_metrics = {"val_loss": float("nan"), "macro_f1": float("nan")}

        val_loss = val_metrics.get("val_loss", float("nan"))
        macro_f1 = val_metrics.get("macro_f1", float("nan"))
        reinit_recall = val_metrics.get("reinit_recall", float("nan"))
        reject_prec = val_metrics.get("reject_precision", float("nan"))
        reinit_auprc = val_metrics.get("reinit_auprc", float("nan"))

        print(
            f"{epoch:>5} | {avg_train_loss:>9.4f} | {val_loss:>8.4f} | "
            f"{reinit_recall:>12.4f} | {reject_prec:>10.4f} | {macro_f1:>7.4f} | {reinit_auprc:>11.4f}",
            flush=True,
        )

        epoch_history.append({
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_loss),
            "macro_f1": float(macro_f1),
            "reinit_recall": float(reinit_recall),
            "reject_precision": float(reject_prec),
            "reinit_auprc": float(reinit_auprc),
        })

        # Early stopping on val REINIT AUPRC
        metric_val = reinit_auprc if not np.isnan(reinit_auprc) else (-val_loss)
        if metric_val > best_reinit_auprc:
            best_reinit_auprc = metric_val
            patience_counter = 0
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_reinit_auprc": float(reinit_auprc),
                    "val_macro_f1": float(macro_f1),
                    "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                    "window_size": window_size,
                    "n_features": N_FEATURES,
                    "hidden_size": hidden_size,
                    "n_layers": n_layers,
                    "lambda_recovery": lambda_recovery,
                    "lambda_candidate": lambda_candidate,
                    "feature_schema": FEATURE_SCHEMA,
                    "model_family": "saltrd_policy",
                    # BUG-26 fix (d): candidate_score head is NOT supervised in this
                    # training run — candidate_features are never passed to forward()
                    # and OracleReinitDataset returns no per-candidate labels.
                    # Remove from trained_heads to stop misleading downstream code.
                    "trained_heads": ["recovery_action"],
                    "oracle_source": str(oracle_path),
                    "git_commit": _git_commit(),
                    "created_at": datetime.utcnow().isoformat(),
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[train_policy] Early stopping at epoch {epoch} "
                    f"(patience={patience}, best_reinit_auprc={best_reinit_auprc:.4f})",
                    flush=True,
                )
                break

    print(f"\n[train_policy] Best checkpoint: {ckpt_path}  (epoch={best_epoch})", flush=True)
    print(f"[train_policy] Best val REINIT AUPRC: {best_reinit_auprc:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Save training summary
    # ------------------------------------------------------------------
    wall_time = time.time() - t_start
    git_commit = _git_commit()

    summary = {
        "model_family": "saltrd_policy",
        "feature_schema": FEATURE_SCHEMA,
        "trained_heads": ["recovery_action"],  # BUG-26: candidate_score not supervised
        "oracle_source": str(oracle_path),
        "output_dir": str(output_dir),
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_reinit_auprc": float(best_reinit_auprc),
        "best_val_metrics": {k: float(v) for k, v in best_val_metrics.items()},
        "train_class_distribution": dist,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "window_size": window_size,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "lambda_recovery": lambda_recovery,
            "lambda_candidate": lambda_candidate,
            "patience": patience,
        },
        "epoch_history": epoch_history,
        "git_commit": git_commit,
        "created_at": datetime.utcnow().isoformat(),
        "random_seed": seed,
        "wall_time_s": round(wall_time, 2),
        "feature_schema_version": FEATURE_SCHEMA,
    }

    summary_path = out_path / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, indent=2)
    print(f"[train_policy] Training summary: {summary_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for SALT-RD policy model training."""
    parser = argparse.ArgumentParser(
        description="Train SALTRDPolicyNet on oracle reinit dataset."
    )
    parser.add_argument(
        "--oracle",
        required=True,
        help="Path to reinit_oracle_dataset.npz produced by oracle_actions.py.",
    )
    parser.add_argument(
        "--output",
        default="saltr/checkpoints/policy_reinit_v1/",
        help="Output directory for checkpoint and summary.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--window-size", type=int, default=20, help="GRU window size.")
    parser.add_argument(
        "--device", default="auto", help="Device: auto | cpu | cuda | mps."
    )
    parser.add_argument(
        "--lambda-recovery",
        type=float,
        default=1.0,
        help="Weight for recovery action loss term.",
    )
    parser.add_argument(
        "--lambda-candidate",
        type=float,
        default=0.5,
        help="Weight for candidate score loss term.",
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-size", type=int, default=64, help="GRU hidden size.")
    parser.add_argument("--n-layers", type=int, default=2, help="GRU layers.")
    args = parser.parse_args()

    train(
        oracle_path=args.oracle,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        window_size=args.window_size,
        device=args.device,
        lambda_recovery=args.lambda_recovery,
        lambda_candidate=args.lambda_candidate,
        patience=args.patience,
        seed=args.seed,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
    )


if __name__ == "__main__":
    main()
