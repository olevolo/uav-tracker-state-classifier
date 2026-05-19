"""train.py — SALT-RD supervised training loop.

Trains the SALTRD GRU multi-head model on an NPZ dataset produced by
collect_features.py.  Uses focal / weighted BCE loss to handle severe
class imbalance (false_confirmed ~1-3% base rate).

Usage::

    python train.py --npz saltr/data/salt_rd_v0.npz --output saltr/checkpoints/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Import from sibling modules — using sys.path manipulation to stay
# self-contained and avoid circular imports.
# ---------------------------------------------------------------------------

def _ensure_salt_r_on_path() -> None:
    """Add saltr/src to sys.path if needed (for running as a script)."""
    salt_r_src = str(Path(__file__).parents[1])  # saltr/src
    if salt_r_src not in sys.path:
        sys.path.insert(0, salt_r_src)


_ensure_salt_r_on_path()

from salt_r.collect_features import SavedDataset, FEATURE_NAMES, LABEL_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Head names — the 7 predictive heads (excludes "correct" which is label 0)
# ---------------------------------------------------------------------------

HEAD_NAMES: List[str] = [
    "false_confirmed",
    "failure_in_5",
    "recoverable",
    "target_dynamic",
    "camera_dynamic",
    "hard_dynamic_scene",
    "needs_full_compute",
]

# Indices of HEAD_NAMES within LABEL_NAMES (label 0 = "correct" is excluded)
_HEAD_LABEL_INDICES: List[int] = [LABEL_NAMES.index(h) for h in HEAD_NAMES]

# Per-head loss weight multiplier (higher = more emphasis)
_HEAD_IMPORTANCE: dict[str, float] = {
    "false_confirmed":    3.0,   # Most critical — silent tracker failure
    "failure_in_5":       2.0,
    "recoverable":        1.5,
    "target_dynamic":     1.0,
    "camera_dynamic":     1.0,
    "hard_dynamic_scene": 1.5,
    "needs_full_compute": 1.0,
}

# ---------------------------------------------------------------------------
# Default positive-class weights (per-head, for focal BCE)
# ---------------------------------------------------------------------------

DEFAULT_POS_WEIGHTS: dict[str, float] = {
    "false_confirmed":    40.0,  # ~2% base rate
    "failure_in_5":       10.0,  # ~8% base rate estimate
    "recoverable":         8.0,
    "target_dynamic":      3.0,  # ~25% base rate by construction
    "camera_dynamic":      3.0,
    "hard_dynamic_scene":  5.0,
    "needs_full_compute":  3.0,
}

# ---------------------------------------------------------------------------
# Metrics helpers (no sklearn dependency)
# ---------------------------------------------------------------------------


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration."""
    return float(np.trapz(y, x))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC via sorted ranking.  Returns 0.5 when all-one-class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    n_pos = y_true_sorted.sum()
    n_neg = len(y_true_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr_list, fpr_list = [0.0], [0.0]
    tp = fp = 0
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    return _trapz(tpr_arr, fpr_arr)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision (area under precision-recall curve).

    Uses the step-function interpolation (same as sklearn's
    average_precision_score).  Returns nan when all labels are the same
    class.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    n_pos = y_true_sorted.sum()
    if n_pos == 0:
        return float("nan")
    tp_cumsum = np.cumsum(y_true_sorted)
    precision = tp_cumsum / (np.arange(len(y_true_sorted)) + 1)
    recall = tp_cumsum / n_pos
    # prepend (recall=0, precision=1) anchor
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    # step integral: sum(precision[i] * d_recall[i])
    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return ap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SALTRDDataset(Dataset):
    """Windowed frame dataset from a collected NPZ file.

    Each sample is a (window_size, n_features) input window and
    multi-label (n_heads,) binary target for the last frame in the window.

    Parameters
    ----------
    npz_path:
        Path to the NPZ file produced by collect_features.py.
    split:
        Which split to use: "train" | "val" | "diagnostic".
    window_size:
        Number of consecutive frames per sample (temporal context).
    head_names:
        List of head names to use as targets.  Defaults to HEAD_NAMES
        (all 7 predictive heads, excluding "correct").
    """

    def __init__(
        self,
        npz_path: str,
        split: str,
        window_size: int = 20,
        head_names: Optional[List[str]] = None,
    ) -> None:
        assert split in {"train", "val", "diagnostic"}, f"Unknown split: {split!r}"
        self.split = split
        self.window_size = window_size
        self.head_names: List[str] = head_names if head_names is not None else list(HEAD_NAMES)
        self.head_label_indices: List[int] = [LABEL_NAMES.index(h) for h in self.head_names]

        # Load NPZ
        ds = SavedDataset.load(npz_path)

        # Store per-sequence arrays; index only sequences matching the split
        self._sequences: list[tuple[np.ndarray, np.ndarray]] = []  # (features, labels)
        self._index: list[tuple[int, int]] = []  # (seq_idx, frame_idx)

        for key, seq_split in ds.split.items():
            if seq_split != split:
                continue
            feats = ds.features[key].astype(np.float32)   # (n_frames, n_features)
            labs = ds.labels[key].astype(np.float32)      # (n_frames, n_labels)
            n_frames = feats.shape[0]
            seq_idx = len(self._sequences)
            self._sequences.append((feats, labs))
            # Only frames where a full window exists (frame_idx >= window_size - 1)
            for t in range(window_size - 1, n_frames):
                self._index.append((seq_idx, t))

        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.label_names: List[str] = list(LABEL_NAMES)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        seq_idx, frame_idx = self._index[idx]
        feats, labs = self._sequences[seq_idx]
        # Window: [frame_idx - window_size + 1, frame_idx] inclusive
        start = frame_idx - self.window_size + 1
        window = feats[start : frame_idx + 1]           # (window_size, n_features)
        target = labs[frame_idx, self.head_label_indices]  # (n_heads,)
        return (
            torch.from_numpy(window),
            torch.from_numpy(target),
        )

    def compute_base_rates(self) -> dict[str, float]:
        """Compute per-head positive-class base rate across the split."""
        if not self._sequences:
            return {h: 0.0 for h in self.head_names}
        all_labels = np.concatenate(
            [labs[self.window_size - 1:, self.head_label_indices] for _, labs in self._sequences],
            axis=0,
        )  # (N, n_heads)
        return {
            h: float(all_labels[:, i].mean())
            for i, h in enumerate(self.head_names)
        }


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------


def focal_bce_loss(
    pred: Tensor,
    target: Tensor,
    pos_weight: Tensor,
    gamma: float = 2.0,
) -> Tensor:
    """Focal BCE with class imbalance weighting.

    Parameters
    ----------
    pred:
        Predicted probabilities in [0, 1], shape (..., n_heads).
    target:
        Binary targets, shape (..., n_heads).
    pos_weight:
        Per-class positive weight, shape (n_heads,).
    gamma:
        Focal exponent — higher values down-weight easy examples more.
    """
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = (1 - pt) ** gamma
    weighted = focal_weight * bce * torch.where(
        target == 1,
        pos_weight,
        torch.ones_like(pos_weight),
    )
    return weighted.mean()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SALTRD model — thin tensor-output wrapper over model.py's dict-output model
# ---------------------------------------------------------------------------

from salt_r.model import SALTRD as _SALTRDBase, HEAD_NAMES as _HEAD_NAMES


class SALTRD(_SALTRDBase):
    """SALTRD with tensor output (B, n_heads) for training convenience.

    model.py's base class returns dict[head_name → (B,)] probabilities.
    This subclass stacks them into a (B, n_heads) tensor so the training
    loop can use standard tensor indexing without per-head boilerplate.
    Head order matches HEAD_NAMES (excluding "correct").
    """

    def forward(self, x: "Tensor") -> "Tensor":  # type: ignore[override]
        probs_dict = super().forward(x)
        return torch.stack([probs_dict[h] for h in _HEAD_NAMES], dim=1)  # (B, n_heads)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _build_pos_weight_tensor(
    head_names: List[str],
    device: torch.device,
    base_rates: Optional[dict[str, float]] = None,
) -> Tensor:
    """Build per-head positive weight tensor.

    If base_rates is provided and the rate is non-zero, use
    ``(1 - rate) / rate`` (empirical).  Otherwise fall back to
    DEFAULT_POS_WEIGHTS.
    """
    weights = []
    for h in head_names:
        if base_rates and h in base_rates and 0 < base_rates[h] < 1:
            w = (1.0 - base_rates[h]) / base_rates[h]
            # Clamp to avoid explosive weights on very rare labels
            w = float(np.clip(w, 1.0, 100.0))
        else:
            w = DEFAULT_POS_WEIGHTS.get(h, 5.0)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


@torch.no_grad()
def _evaluate(
    model: SALTRD,
    loader: DataLoader,
    pos_weight: Tensor,
    head_names: List[str],
    device: torch.device,
) -> dict[str, float]:
    """Run a full pass over the loader; return loss + per-head metrics."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = focal_bce_loss(pred, y, pos_weight)
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    if not all_preds:
        return {"val_loss": avg_loss}

    preds_np = np.concatenate(all_preds, axis=0)    # (N, n_heads)
    targets_np = np.concatenate(all_targets, axis=0) # (N, n_heads)

    metrics: dict[str, float] = {"val_loss": avg_loss}
    for i, h in enumerate(head_names):
        yt = targets_np[:, i]
        yp = preds_np[:, i]
        metrics[f"auroc_{h}"] = _roc_auc(yt, yp)
        metrics[f"auprc_{h}"] = _average_precision(yt, yp)

    return metrics


def train(
    npz_path: str,
    output_dir: str,
    window_size: int = 20,
    batch_size: int = 256,
    max_epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: str = "auto",
    seed: int = 42,
) -> None:
    """Train SALTRD on the given NPZ dataset.

    Parameters
    ----------
    npz_path:
        Path to the NPZ file produced by collect_features.py.
    output_dir:
        Directory where the best checkpoint will be saved.
    window_size:
        Number of consecutive frames per sample.
    batch_size:
        Mini-batch size.
    max_epochs:
        Maximum training epochs.
    lr:
        Initial learning rate for Adam.
    weight_decay:
        L2 regularisation coefficient.
    patience:
        Early-stopping patience on val AUPRC(false_confirmed).
    device:
        "auto" | "cpu" | "cuda" | "mps".
    seed:
        Random seed for reproducibility.
    """
    # ------------------------------------------------------------------
    # 0. Seed
    # ------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev = _resolve_device(device)
    print(f"[train] device={dev}  npz={npz_path}", flush=True)

    # ------------------------------------------------------------------
    # 1. Datasets / dataloaders
    # ------------------------------------------------------------------
    print("[train] Loading train split ...", flush=True)
    train_ds = SALTRDDataset(npz_path, split="train", window_size=window_size)
    print(f"[train] train samples: {len(train_ds):,}", flush=True)

    print("[train] Loading val split ...", flush=True)
    val_ds = SALTRDDataset(npz_path, split="val", window_size=window_size)
    print(f"[train] val samples:   {len(val_ds):,}", flush=True)

    if len(train_ds) == 0:
        raise RuntimeError("Training split is empty — check the NPZ file.")

    # Compute base rates from training data for pos_weight calibration
    base_rates = train_ds.compute_base_rates()
    print("[train] Base rates (train):", flush=True)
    for h, r in base_rates.items():
        print(f"  {h}: {r:.4f} ({r*100:.2f}%)", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
    # 2. Model
    # ------------------------------------------------------------------
    n_features = len(FEATURE_NAMES)   # 28
    n_heads = len(_HEAD_NAMES)         # 7, from model.HEAD_NAMES

    model = SALTRD(n_features=n_features).to(dev)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] SALTRD  params={total_params:,}", flush=True)

    # ------------------------------------------------------------------
    # 3. Loss / optimiser / scheduler
    # ------------------------------------------------------------------
    pos_weight = _build_pos_weight_tensor(
        train_ds.head_names, dev, base_rates=base_rates
    )
    head_importance = torch.tensor(
        [_HEAD_IMPORTANCE.get(h, 1.0) for h in train_ds.head_names],
        dtype=torch.float32,
        device=dev,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_path / "saltrd_best.pt"

    best_auprc_fc = -1.0
    patience_counter = 0

    # Table header
    print(
        f"\n{'Epoch':>5} | {'TrainLoss':>9} | {'ValLoss':>8} | "
        f"{'AUPRC(fc)':>9} | {'AUROC(fc)':>9} | {'AUROC(fail5)':>11}",
        flush=True,
    )
    print("-" * 64, flush=True)

    for epoch in range(1, max_epochs + 1):
        # -- Train --
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            optimizer.zero_grad()
            pred = model(x)
            # Per-head focal loss, weighted by head importance
            per_head_losses = torch.stack([
                focal_bce_loss(
                    pred[:, i : i + 1],
                    y[:, i : i + 1],
                    pos_weight[i : i + 1],
                )
                for i in range(n_heads)
            ])  # (n_heads,)
            loss = (per_head_losses * head_importance).sum() / head_importance.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            n_train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        # -- Validate --
        val_metrics: dict[str, float] = {}
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, pos_weight, train_ds.head_names, dev)
        else:
            val_metrics = {"val_loss": float("nan")}

        val_loss = val_metrics.get("val_loss", float("nan"))
        auprc_fc = val_metrics.get("auprc_false_confirmed", float("nan"))
        auroc_fc = val_metrics.get("auroc_false_confirmed", float("nan"))
        auroc_f5 = val_metrics.get("auroc_failure_in_5", float("nan"))

        print(
            f"{epoch:>5} | {avg_train_loss:>9.4f} | {val_loss:>8.4f} | "
            f"{auprc_fc:>9.4f} | {auroc_fc:>9.4f} | {auroc_f5:>11.4f}",
            flush=True,
        )

        # -- Early stopping on val AUPRC(false_confirmed) --
        # If val set is empty or metric is nan, fall back to val loss
        improvement_metric = auprc_fc if not np.isnan(auprc_fc) else -val_loss

        if improvement_metric > best_auprc_fc:
            best_auprc_fc = improvement_metric
            patience_counter = 0

            # Save best checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auprc_false_confirmed": auprc_fc,
                    "val_metrics": val_metrics,
                    "base_rates": base_rates,
                    "window_size": window_size,
                    "feature_names": list(FEATURE_NAMES),
                    "label_names": list(LABEL_NAMES),
                    "head_names": list(train_ds.head_names),
                    "n_features": n_features,
                    "n_heads": n_heads,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[train] Early stopping at epoch {epoch} "
                    f"(patience={patience}, best_auprc_fc={best_auprc_fc:.4f})",
                    flush=True,
                )
                break

    print(f"\n[train] Best checkpoint saved to: {ckpt_path}", flush=True)
    print(f"[train] Best val AUPRC(false_confirmed): {best_auprc_fc:.4f}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for SALT-RD training."""
    parser = argparse.ArgumentParser(
        description="Train SALTRD GRU multi-head model on NPZ dataset."
    )
    parser.add_argument("--npz", required=True, help="Path to input NPZ file.")
    parser.add_argument(
        "--output",
        default="saltr/checkpoints/",
        help="Output directory for checkpoints (default: saltr/checkpoints/).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum training epochs."
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--device", default="auto", help="Device: auto | cpu | cuda | mps."
    )
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    train(
        npz_path=args.npz,
        output_dir=args.output,
        window_size=args.window_size,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
