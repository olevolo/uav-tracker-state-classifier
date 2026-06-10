#!/usr/bin/env python3
"""Threshold-aware metrics for forecast head fc_n10 (paper-grade).

For each ckpt, runs val pass and computes:
  - AUPRC (already in val_metrics.json)
  - recall @ FPR ≤ 1% and ≤ 3% (operating point for control mode)
  - precision @ recall = 0.50 / 0.70 / 0.85
  - PR curve elbow (max F1) point

Why: AUPRC of 0.547 sounds low but pos_rate=0.023 → baseline=0.023.
0.547 / 0.023 = 24× over baseline. recall@FPR≤3% is the real operating metric
because control mode triggers on a fixed false-alarm budget, not on AUPRC area.

Usage:
  python tools/forecast_threshold_metrics.py \\
    --ckpt outputs/csc_training/sglatrack_v3fix_tcn16_stage2/checkpoint_best.pth \\
    --ckpt outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth \\
    --labels-dir outputs/csc_labels/sglatrack/v3fix_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csc_lib.csc.config import CSCTrainConfig
from csc_lib.csc.dataset import build_train_val_datasets
from csc_lib.csc.features import FEATURE_DIM, FEATURE_DIM_V2
from csc_lib.csc.model import build_model
from torch.utils.data import DataLoader


def recall_at_max_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float):
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    s_sorted = scores[order]
    cum_tp = np.cumsum(y)
    cum_fp = np.cumsum(1.0 - y)
    n_pos = float(y.sum())
    n_neg = float((1 - y).sum())
    fpr = cum_fp / max(n_neg, 1.0)
    recall = cum_tp / max(n_pos, 1.0)
    valid = fpr <= max_fpr
    if not valid.any():
        return 0.0, float(s_sorted[0])
    last = int(np.where(valid)[0].max())
    return float(recall[last]), float(s_sorted[last])


def precision_at_min_recall(scores: np.ndarray, labels: np.ndarray, min_recall: float):
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    s_sorted = scores[order]
    cum_tp = np.cumsum(y)
    cum_fp = np.cumsum(1.0 - y)
    n_pos = float(y.sum())
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)
    recall = cum_tp / max(n_pos, 1.0)
    valid = recall >= min_recall
    if not valid.any():
        return float("nan"), float("nan")
    first = int(np.where(valid)[0].min())
    return float(precision[first]), float(s_sorted[first])


def f1_max(scores: np.ndarray, labels: np.ndarray):
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    s_sorted = scores[order]
    cum_tp = np.cumsum(y)
    cum_fp = np.cumsum(1.0 - y)
    n_pos = float(y.sum())
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)
    recall = cum_tp / max(n_pos, 1.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    best = int(np.argmax(f1))
    return float(f1[best]), float(precision[best]), float(recall[best]), float(s_sorted[best])


def run_eval(ckpt_path: Path, labels_dir: Path) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Try to read the training config from sibling
    cfg_dir = ckpt_path.parent
    # Build dataset by reading config used at train time (we know labels_dir).
    # Detect feature_version + window_size per-ckpt from the path so R3 (w32/v2)
    # and R2 (w16/v2) are each evaluated with their own correct settings.
    path_str = str(ckpt_path)
    v2_markers = ("run2_scalectx", "r25", "_v2_", "fcw", "tcn32")
    feature_version = "v2" if any(m in path_str for m in v2_markers) else "v1"
    window_size = 32 if "w32" in path_str else 16
    feature_dim = FEATURE_DIM_V2 if feature_version == "v2" else FEATURE_DIM

    # Minimal cfg for dataset
    cfg_dict = {
        "seed": 42, "device": "cpu", "val_fraction": 0.15, "stratified_split": True,
        "training_stage": 2,
        "feature": {
            "feature_version": feature_version, "use_telemetry": True,
            "use_bbox_dynamics": True, "use_geometry_normalised": True,
            "window_size": window_size, "clip_value": 8.0,
        },
        "model": {"kind": "tcn", "feature_dim": feature_dim, "hidden_dim": 64,
                  "num_layers": 4, "dropout": 0.1, "bidirectional": False, "n_states": 4,
                  "enable_forecast_heads": True, "forecast_horizon": 10,
                  "tcn": {"kernel_size": 3, "num_layers": 4, "dilations": [1, 2, 4, 8],
                          "hidden_dim": 64, "dropout": 0.1}},
        "loss": {"state_weights": [1.0, 1.5, 2.0, 4.0], "risk_weight": 0.0,
                 "aux_weight": 0.3, "use_focal": True, "focal_gamma": 2.0},
        "optim": {"lr": 5e-4, "weight_decay": 1e-4, "optimizer": "adamw",
                  "epochs": 25, "batch_size": 64, "early_stopping_patience": 10,
                  "grad_clip": 1.0, "use_balanced_sampler": True,
                  "scheduler": "cosine", "min_lr_ratio": 0.02},
        "labels_dir": str(labels_dir), "output_dir": str(cfg_dir),
    }
    cfg = CSCTrainConfig.from_dict(cfg_dict)

    print(f"[{ckpt_path.name}] feature_version={feature_version} dim={feature_dim} window_size={window_size}")

    _, val_ds, info = build_train_val_datasets(
        cfg.labels_dir, cfg.feature,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        stratified_split=cfg.stratified_split,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.optim.batch_size,
                            shuffle=False, num_workers=0)
    print(f"  val windows={info['n_val_windows']}")

    model = build_model(cfg.model)
    sd = ck.get("state_dict") or ck.get("model_state") or ck
    model.load_state_dict(sd)
    model.eval()

    fc_p, fc_t, mask = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["features"]
            out = model(x)
            if out.false_confirmed_next_10_logit is None:
                continue
            prob = torch.sigmoid(out.false_confirmed_next_10_logit.squeeze(-1))
            fc_p.append(prob.flatten().cpu().numpy())
            fc_t.append(batch["false_confirmed_next_10"].flatten().cpu().numpy())
            mask.append((1 - batch["ignore_forecast"]).flatten().cpu().numpy().astype(bool))

    fc_p = np.concatenate(fc_p)
    fc_t = np.concatenate(fc_t).astype(np.int8)
    m = np.concatenate(mask)
    fc_p, fc_t = fc_p[m], fc_t[m]

    pos_rate = float(fc_t.mean())
    r_fpr01, thr01 = recall_at_max_fpr(fc_p, fc_t, 0.01)
    r_fpr03, thr03 = recall_at_max_fpr(fc_p, fc_t, 0.03)
    r_fpr05, thr05 = recall_at_max_fpr(fc_p, fc_t, 0.05)
    p_r50, _ = precision_at_min_recall(fc_p, fc_t, 0.50)
    p_r70, _ = precision_at_min_recall(fc_p, fc_t, 0.70)
    p_r85, _ = precision_at_min_recall(fc_p, fc_t, 0.85)
    f1m, p_f1m, r_f1m, thr_f1m = f1_max(fc_p, fc_t)

    return {
        "ckpt": str(ckpt_path),
        "feature_version": feature_version,
        "n_eval": int(fc_t.size),
        "pos_rate": pos_rate,
        "lift_over_baseline_at_auprc": None,  # caller fills in if needed
        "recall_at_fpr_01": r_fpr01, "thr_fpr_01": thr01,
        "recall_at_fpr_03": r_fpr03, "thr_fpr_03": thr03,
        "recall_at_fpr_05": r_fpr05, "thr_fpr_05": thr05,
        "precision_at_recall_50": p_r50,
        "precision_at_recall_70": p_r70,
        "precision_at_recall_85": p_r85,
        "best_f1": f1m, "best_f1_precision": p_f1m,
        "best_f1_recall": r_f1m, "best_f1_threshold": thr_f1m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="Stage 2 ckpt path with forecast heads (repeat for each run)")
    ap.add_argument("--labels-dir", type=Path,
                    default=ROOT / "outputs/csc_labels/sglatrack/v3fix_combined")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "logs/v3fix_full/fc_n10_threshold_aware.json")
    args = ap.parse_args()

    results = []
    for ckpt in args.ckpt:
        results.append(run_eval(Path(ckpt), args.labels_dir))

    print("\n=== fc_n10 threshold-aware (paper-grade) ===\n")
    print(f"{'Ckpt':<60} {'feat':<5} {'pos%':<6} {'R@1%':<7} {'R@3%':<7} {'R@5%':<7} {'P@R50':<7} {'P@R70':<7} {'P@R85':<7} {'F1max':<7}")
    for r in results:
        name = Path(r["ckpt"]).parent.name
        print(f"{name:<60} {r['feature_version']:<5} {r['pos_rate']*100:<6.2f} "
              f"{r['recall_at_fpr_01']:<7.3f} {r['recall_at_fpr_03']:<7.3f} {r['recall_at_fpr_05']:<7.3f} "
              f"{r['precision_at_recall_50']:<7.3f} {r['precision_at_recall_70']:<7.3f} "
              f"{r['precision_at_recall_85']:<7.3f} {r['best_f1']:<7.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
