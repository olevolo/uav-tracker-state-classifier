"""baselines.py — Rule-based feature baselines for SALT-RD heads.

Computes AUROC/AUPRC for each individual feature used as a threshold predictor
against the target labels. Provides the paper comparison: "GRU model vs best
single-feature baseline" for imminent_failure_dynamic and false_confirmed.

Usage::

    python -m salt_r.baselines \\
        --npz saltr/data/salt_rd_v1_labels.npz \\
        --split val \\
        --probs-json saltr/results/preds_val.json \\   # optional: add model to table
        --output saltr/results/baselines_val.json

Key features as predictors:
  imminent_failure_dynamic: high risk when apce low, entropy high,
      flow_consistency low, apce trend falling, speed high.
  false_confirmed: high risk when apce high (tracker overconfident),
      entropy low, peak_margin high, flow_consistency low.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Feature sign conventions
# ---------------------------------------------------------------------------
# For each feature, "higher value → target label more likely" means positive sign.
# Features where LOW value indicates risk use sign=-1 so we negate before AUROC.

_FEATURE_SIGNS: dict[str, dict[str, int]] = {
    # imminent_failure_dynamic: dynamic scene approaching IoU degradation
    "imminent_failure_dynamic": {
        "apce_raw":            -1,   # low APCE → tracker losing confidence
        "apce_norm":           -1,
        "psr":                 -1,   # low PSR → weak response
        "entropy":             +1,   # high entropy → response spreading out
        "peak_margin":         -1,   # low top1-top2 gap → ambiguous response
        "apce_ratio_5":        -1,   # falling APCE trend
        "entropy_delta_5":     +1,   # rising entropy trend
        "peak_margin_delta_5": -1,   # falling peak margin trend
        "bbox_speed_norm":     +1,   # fast target → harder to track
        "bbox_accel_norm":     +1,   # acceleration → scene getting harder
        "global_flow_mag":     +1,   # camera motion increasing difficulty
        "ego_motion_residual": +1,   # residual motion after ego compensation
        "flow_consistency":    -1,   # low flow consistency → decoupled response
    },
    # false_confirmed: tracker confident but on wrong object
    "false_confirmed": {
        "apce_raw":            +1,   # HIGH APCE = overconfident tracker
        "apce_norm":           +1,
        "psr":                 +1,   # high PSR = sharp (but wrong) peak
        "entropy":             -1,   # low entropy = narrow (but wrong) response
        "peak_margin":         +1,   # high peak margin = very confident (wrong object)
        "heatmap_mass_topk":   +1,   # mass concentrated on wrong peak
        "flow_consistency":    -1,   # low flow consistency = object moved away
        "cosine_static_template": -1,  # low cosine = appearance drifted
        "embedding_drift_static": +1,  # high drift = wrong object tracked
    },
}

_DEFAULT_TARGETS = ["imminent_failure_dynamic", "false_confirmed"]


# ---------------------------------------------------------------------------
# Metric helpers (no sklearn)
# ---------------------------------------------------------------------------

def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # Trapezoidal integration
    return float(np.trapz(tpr, fpr))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(y_true.sum())
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    recall = np.cumsum(y_sorted) / n_pos
    # Prepend (0, 1) for proper integration
    precision_full = np.concatenate([[1.0], precision])
    recall_full = np.concatenate([[0.0], recall])
    return float(np.sum((recall_full[1:] - recall_full[:-1]) * precision_full[1:]))


# ---------------------------------------------------------------------------
# Core baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_feature_baselines(
    npz_path: str,
    split: str = "val",
    target_heads: list[str] | None = None,
    probs_json_path: str | None = None,
) -> dict[str, Any]:
    """Compute AUROC/AUPRC for each feature as a threshold predictor.

    Parameters
    ----------
    npz_path:
        Path to salt_rd NPZ (v1 schema preferred for imminent_failure_dynamic).
    split:
        Which split to evaluate on.
    target_heads:
        Which label heads to use as targets. Defaults to
        ["imminent_failure_dynamic", "false_confirmed"].
    probs_json_path:
        Optional path to model predictions JSON (from eval.py --predictions-output).
        When provided, adds the GRU model row to the comparison table.

    Returns
    -------
    Dict keyed by target head name, each containing a sorted list of
    (feature, auroc, auprc, n_pos, base_rate) tuples.
    """
    if target_heads is None:
        target_heads = _DEFAULT_TARGETS

    data = np.load(npz_path, allow_pickle=True)

    try:
        feature_names: list[str] = list(data["feature_names"].tolist())
        label_names: list[str] = list(data["label_names"].tolist())
    except Exception:
        from salt_r.collect_features import FEATURE_NAMES, LABEL_NAMES
        feature_names = list(FEATURE_NAMES)
        label_names = list(LABEL_NAMES)

    keys = [k[len("features/"):] for k in data.files if k.startswith("features/")]

    feats_list, labs_list = [], []
    for key in keys:
        if str(data[f"split/{key}"]) != split:
            continue
        feats_list.append(data[f"features/{key}"].astype(np.float32))
        labs_list.append(data[f"labels/{key}"].astype(np.int8))

    if not feats_list:
        return {"error": f"No sequences found for split={split!r}"}

    all_feats = np.concatenate(feats_list, axis=0)  # (N, F)
    all_labs = np.concatenate(labs_list, axis=0)    # (N, L)

    # Load model predictions if provided
    model_preds: dict[str, np.ndarray] | None = None
    model_head_order: list[str] | None = None
    if probs_json_path and Path(probs_json_path).exists():
        with open(probs_json_path) as f:
            raw = json.load(f)
        pred_seqs = []
        for key in keys:
            if str(data[f"split/{key}"]) != split:
                continue
            if key in raw:
                frames = raw[key]
                arr = np.array([[v for v in frame.values()] for frame in frames],
                               dtype=np.float32)
                pred_seqs.append(arr)
                if model_head_order is None:
                    model_head_order = list(frames[0].keys()) if frames else []
        if pred_seqs:
            model_preds_arr = np.concatenate(pred_seqs, axis=0)
            if model_head_order:
                model_preds = {h: model_preds_arr[:, i]
                               for i, h in enumerate(model_head_order)}

    results: dict[str, Any] = {}

    for target in target_heads:
        if target not in label_names:
            results[target] = {"note": f"{target} not in label schema"}
            continue

        lbl_idx = label_names.index(target)
        y_true = all_labs[:, lbl_idx].astype(float)
        n_pos = int(y_true.sum())
        n_total = len(y_true)
        base_rate = float(y_true.mean())

        rows: list[dict[str, Any]] = []
        signs = _FEATURE_SIGNS.get(target, {})

        for feat_name in feature_names:
            if feat_name not in signs:
                continue
            fidx = feature_names.index(feat_name)
            score = all_feats[:, fidx].astype(float)
            if signs[feat_name] == -1:
                score = -score  # negate so that higher = riskier

            auroc = _roc_auc(y_true, score)
            auprc = _average_precision(y_true, score)
            rows.append({
                "feature": feat_name,
                "sign": signs[feat_name],
                "auroc": round(auroc, 4),
                "auprc": round(auprc, 4),
                "auprc_lift": round(auprc / base_rate, 2) if base_rate > 0 else None,
            })

        # Add GRU model row if available
        if model_preds and target in model_preds:
            gru_auroc = _roc_auc(y_true, model_preds[target])
            gru_auprc = _average_precision(y_true, model_preds[target])
            rows.append({
                "feature": "GRU_model",
                "sign": None,
                "auroc": round(gru_auroc, 4),
                "auprc": round(gru_auprc, 4),
                "auprc_lift": round(gru_auprc / base_rate, 2) if base_rate > 0 else None,
            })

        # Sort by AUROC descending
        rows.sort(key=lambda r: r["auroc"] if r["auroc"] == r["auroc"] else -1,
                  reverse=True)

        results[target] = {
            "n_positive": n_pos,
            "n_total": n_total,
            "base_rate": round(base_rate, 4),
            "baselines": rows,
        }

    return results


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_table(results: dict[str, Any]) -> None:
    for target, info in results.items():
        if "note" in info or "error" in info:
            print(f"\n{target}: {info.get('note', info.get('error'))}")
            continue

        base = info["base_rate"]
        print(f"\n{'='*65}")
        print(f"Target: {target}  (base_rate={base:.4f}, "
              f"n_pos={info['n_positive']}, n_total={info['n_total']})")
        print(f"{'='*65}")
        print(f"  {'feature':<30} {'AUROC':>7} {'AUPRC':>7} {'lift':>6}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*6}")

        for row in info["baselines"]:
            marker = " ◀ GRU" if row["feature"] == "GRU_model" else ""
            lift = f"{row['auprc_lift']:.1f}x" if row["auprc_lift"] else "  n/a"
            print(f"  {row['feature']:<30} {row['auroc']:>7.4f} {row['auprc']:>7.4f} {lift:>6}{marker}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare rule-based feature baselines vs GRU model for SALT-RD heads."
    )
    parser.add_argument("--npz", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--targets", nargs="*", default=None,
                        metavar="HEAD",
                        help="Label heads to evaluate (default: imminent_failure_dynamic false_confirmed)")
    parser.add_argument("--probs-json", default=None,
                        help="Model predictions JSON from eval.py --predictions-output")
    parser.add_argument("--output", default=None,
                        help="Write JSON results to this path")
    args = parser.parse_args()

    results = evaluate_feature_baselines(
        npz_path=args.npz,
        split=args.split,
        target_heads=args.targets,
        probs_json_path=args.probs_json,
    )

    _print_table(results)

    if args.output:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline comparison written to: {args.output}")


if __name__ == "__main__":
    main()
