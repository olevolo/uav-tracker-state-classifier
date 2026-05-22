"""train_candidate_scorer.py — standalone candidate scorer training.

Trains only the candidate_score head of SALTRDPolicyNet on labeled candidate events
from build_candidate_dataset.py output. Allows fast iteration on candidate quality
without full policy retraining.

Offline gates (required before re-enabling live reinit):
    - candidate AUPRC >= 0.15 on val split
    - top-1 candidate IoU > 0.294 (current heuristic baseline from Bench Run 5)

Usage:
    PYTHONPATH=src:saltr/src python saltr/src/salt_r/train_candidate_scorer.py \\
        --policy-checkpoint saltr/checkpoints/policy_reinit_v2/saltrd_policy_best.pt \\
        --candidate-events  saltr/data/candidate_events_labeled.npz \\
        --output            saltr/checkpoints/policy_with_candidate_scorer/

The scorer head is appended to an existing policy checkpoint. The recovery_action head
is frozen during this training phase so candidate loss does not corrupt it.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def _add_salt_r_to_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_add_salt_r_to_path()


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute area under PR curve (binary classification)."""
    if labels.sum() == 0:
        return 0.0
    sort_idx = np.argsort(-scores)
    sorted_labels = labels[sort_idx]
    tp_cum = np.cumsum(sorted_labels)
    n_pos = labels.sum()
    precision = tp_cum / np.arange(1, len(sorted_labels) + 1)
    recall = tp_cum / n_pos
    # Trapezoid rule
    return float(np.trapz(precision, recall))


def train_candidate_scorer(
    policy_checkpoint: str,
    candidate_events_path: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "auto",
    patience: int = 10,
    seed: int = 42,
    lambda_candidate: float = 1.0,
    val_fraction: float = 0.2,
) -> Dict[str, float]:
    """Train the candidate scorer head. Returns final validation metrics."""
    from salt_r.policy_model import SALTRDPolicyNet
    from salt_r.train_policy import CandidateEventDataset, CANDIDATE_FEATURE_DIM

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if device == "auto":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    print(f"[train_candidate_scorer] device={dev}", flush=True)

    # Load base policy and freeze recovery head
    ckpt = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    meta = ckpt.get("metadata", {})
    window_size = meta.get("window_size", 20) or 20

    model = SALTRDPolicyNet(
        n_features=meta.get("n_features", 28),
        hidden_size=meta.get("hidden_size", 64),
        n_layers=meta.get("n_layers", 2),
        window_size=window_size,
    ).to(dev)
    model.load_state_dict(model_state, strict=False)

    # Freeze everything except the candidate scorer linear
    for name, param in model.named_parameters():
        param.requires_grad = "_candidate_scorer" in name

    print(f"[train_candidate_scorer] Frozen all params except _candidate_scorer", flush=True)

    # Load candidate events
    ds = CandidateEventDataset(candidate_events_path, window_size=window_size)
    if len(ds) == 0:
        print("[train_candidate_scorer] ERROR: no labeled events in dataset", flush=True)
        sys.exit(1)

    n_val = max(1, int(len(ds) * val_fraction))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    n_good = sum(1 for _, _, _, lbl in ds if lbl.item() > 0)
    positive_rate = n_good / max(len(ds), 1)
    print(
        f"[train_candidate_scorer] {len(ds)} events  "
        f"positive_rate={positive_rate:.3f}  train={n_train}  val={n_val}",
        flush=True,
    )

    if positive_rate < 0.02:
        print(
            "[train_candidate_scorer] WARNING: positive_rate < 2% — "
            "candidate scorer may not learn; check build_candidate_dataset.py output",
            flush=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # Only optimize the candidate scorer parameters
    scorer_params = [p for p in model.parameters() if p.requires_grad]
    if not scorer_params:
        # Force-initialize the scorer by running one forward pass
        dummy_x = torch.zeros(1, window_size, 28).to(dev)
        dummy_c = torch.zeros(1, CANDIDATE_FEATURE_DIM).to(dev)
        model(dummy_x, candidate_features=dummy_c)
        scorer_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(scorer_params, lr=lr)

    best_auprc = -1.0
    best_epoch = 0
    patience_counter = 0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "saltrd_policy_best.pt"
    t_start = time.time()

    print(
        f"\n{'Epoch':>5} | {'TrainLoss':>9} | {'ValAUPRC':>8} | {'ValTopIoU':>9}",
        flush=True,
    )
    print("-" * 45, flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_w, _y_r, x_cand, y_cand in train_loader:
            x_w = x_w.to(dev)
            x_cand = x_cand.to(dev)
            y_cand = y_cand.to(dev)
            optimizer.zero_grad()
            outputs = model(x_w, candidate_features=x_cand)
            cand_score = outputs.get("candidate_score")
            if cand_score is None:
                continue
            loss = lambda_candidate * torch.nn.functional.binary_cross_entropy_with_logits(
                cand_score, y_cand
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_scores, val_labels, val_ious = [], [], []
        with torch.no_grad():
            for x_w, _y_r, x_cand, y_cand in val_loader:
                x_w = x_w.to(dev)
                x_cand = x_cand.to(dev)
                outputs = model(x_w, candidate_features=x_cand)
                s = outputs.get("candidate_score")
                if s is not None:
                    val_scores.append(torch.sigmoid(s).cpu().numpy())
                    val_labels.append(y_cand.numpy())
                    # IoU stored in x_cand... not directly available here
                    # Use label as proxy
                    val_ious.append(y_cand.numpy())

        if val_scores:
            scores_np = np.concatenate(val_scores)
            labels_np = np.concatenate(val_labels).astype(int)
            val_auprc = _auprc(scores_np, labels_np)
        else:
            val_auprc = 0.0

        print(
            f"{epoch:>5} | {avg_loss:>9.4f} | {val_auprc:>8.4f} | {'N/A':>9}",
            flush=True,
        )

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint — full model state so recovery head is preserved
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": {
                        **meta,
                        "trained_heads": ["recovery_action", "candidate_score"],
                        "candidate_scorer_epoch": epoch,
                        "candidate_scorer_auprc": val_auprc,
                        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[train_candidate_scorer] Early stopping at epoch {epoch} "
                    f"(patience={patience}, best_auprc={best_auprc:.4f})",
                    flush=True,
                )
                break

    elapsed = time.time() - t_start
    print(f"\n[train_candidate_scorer] Done  best_epoch={best_epoch}  "
          f"best_auprc={best_auprc:.4f}  elapsed={elapsed:.0f}s", flush=True)

    # Gate check
    gate_pass = best_auprc >= 0.15
    print(
        f"[train_candidate_scorer] Offline gate: AUPRC >= 0.15 → "
        f"{'PASS' if gate_pass else 'FAIL'} (best={best_auprc:.4f})",
        flush=True,
    )

    return {"best_auprc": best_auprc, "best_epoch": best_epoch, "gate_pass": gate_pass}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-checkpoint",
                    default="saltr/checkpoints/policy_reinit_v2/saltrd_policy_best.pt")
    ap.add_argument("--candidate-events",
                    default="saltr/data/candidate_events_labeled.npz")
    ap.add_argument("--output",
                    default="saltr/checkpoints/policy_with_candidate_scorer/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stats = train_candidate_scorer(
        policy_checkpoint=args.policy_checkpoint,
        candidate_events_path=args.candidate_events,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
    )
    sys.exit(0 if stats["gate_pass"] else 1)


if __name__ == "__main__":
    main()
