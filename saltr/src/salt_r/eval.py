"""eval.py — SALT-RD GRU model evaluation.

Full per-head evaluation of a trained SALTRD checkpoint against the
val/diagnostic NPZ splits.  Reports AUROC, AUPRC, ECE, Brier, NLL,
Recall@5%FPR, NT2F, and bootstrap CIs per label head.

Usage::

    python -m salt_r.eval \\
        --npz  saltr/data/salt_rd_v0.npz \\
        --checkpoint saltr/checkpoints/salt_rd_best.pt \\
        --split val \\
        --output saltr/results/eval_val.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Pure numpy/scipy replacements for sklearn.metrics
# (scikit-learn is not a declared project dependency)
# ---------------------------------------------------------------------------


def _roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve (FPR, TPR, thresholds) without sklearn.

    Follows the same convention as sklearn.metrics.roc_curve:
    - Sorted by decreasing threshold.
    - A sentinel threshold above the max score is prepended so the curve
      starts at (FPR=0, TPR=0).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # Sort by descending score
    desc_idx = np.argsort(y_score)[::-1]
    sorted_scores = y_score[desc_idx]
    sorted_labels = y_true[desc_idx]

    # Unique thresholds (descending)
    thresholds = np.concatenate([[sorted_scores[0] + 1], sorted_scores])

    tp = np.concatenate([[0], np.cumsum(sorted_labels)])
    fp = np.concatenate([[0], np.cumsum(1 - sorted_labels)])

    n_pos = tp[-1]
    n_neg = fp[-1]

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)

    return fpr, tpr, thresholds


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via the trapezoidal rule."""
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    # np.trapz expects x increasing
    return float(np.trapz(tpr, fpr))


def _average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (interpolation-free, sklearn convention)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    desc_idx = np.argsort(y_score)[::-1]
    sorted_labels = y_true[desc_idx]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    precision = tp / (tp + fp)
    recall = tp / max(sorted_labels.sum(), 1)

    # Prepend (recall=0, precision=1) sentinel
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    # Sum of trapezoids (sklearn uses step interpolation: delta_recall * precision_right)
    return float(np.sum(np.diff(recall) * precision[1:]))

# ---------------------------------------------------------------------------
# GO/NO-GO gate thresholds (Phase 1b criteria from SALT-RD plan)
# ---------------------------------------------------------------------------

GO_THRESHOLDS: dict[str, float] = {
    "auprc_false_confirmed":     0.30,   # > → GO
    "auroc_false_confirmed":     0.65,
    "auroc_failure_in_5":        0.75,
    "auroc_hard_dynamic_scene":  0.75,
    "auroc_needs_full_compute":  0.70,
    "ece_false_confirmed":       0.12,   # < → GO (lower is better)
}

STOP_THRESHOLDS: dict[str, float] = {
    "auprc_false_confirmed":     0.15,
    "auroc_false_confirmed":     0.55,
    "auroc_failure_in_5":        0.65,
}

# Heads whose ECE is lower-is-better for the GO gate
_LOWER_IS_BETTER = {"ece_false_confirmed"}

# ---------------------------------------------------------------------------
# Calibration metric helpers
# ---------------------------------------------------------------------------


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins).

    Parameters
    ----------
    probs:
        Predicted probabilities in [0, 1], shape (N,).
    labels:
        Binary ground-truth labels, shape (N,).
    n_bins:
        Number of equal-width bins in [0, 1].

    Returns
    -------
    float
        Weighted mean |accuracy - confidence| across bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(probs))


# ---------------------------------------------------------------------------
# Per-head classification metrics
# ---------------------------------------------------------------------------


def compute_head_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    head_name: str,
) -> dict[str, float]:
    """Compute full metric suite for one binary classification head.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels, shape (N,), dtype float or int.
    y_pred:
        Predicted probabilities in [0, 1], shape (N,).
    head_name:
        Name used only for error messages / logging.

    Returns
    -------
    dict with keys: base_rate, auroc, auprc, ece, brier, nll,
    recall_at_5pct_fpr.  Values are nan when the class is
    degenerate (all-zero or all-one labels).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    base_rate = float(y_true.mean())
    if base_rate == 0.0 or base_rate == 1.0:
        return {
            "base_rate": base_rate,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "nll": float("nan"),
            "recall_at_5pct_fpr": float("nan"),
        }

    auroc = _roc_auc_score(y_true, y_pred)
    auprc = _average_precision_score(y_true, y_pred)

    # ECE (10 equal-width bins)
    ece = _ece(y_pred, y_true, n_bins=10)

    # Brier score
    brier = float(np.mean((y_pred - y_true) ** 2))

    # Negative log-likelihood
    eps = 1e-8
    nll = float(
        -np.mean(
            y_true * np.log(y_pred + eps)
            + (1 - y_true) * np.log(1 - y_pred + eps)
        )
    )

    # Recall @ 5% FPR — the "false_confirmed" signature metric
    fpr, tpr, _ = _roc_curve(y_true, y_pred)
    idx = int(np.searchsorted(fpr, 0.05))
    recall_at_5fpr = float(tpr[min(idx, len(tpr) - 1)])

    return {
        "base_rate": base_rate,
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "recall_at_5pct_fpr": recall_at_5fpr,
    }


# ---------------------------------------------------------------------------
# NT2F — Normalized Time to Failure
# ---------------------------------------------------------------------------


def compute_nt2f(
    iou_traces: dict[str, np.ndarray],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Normalized Time to Failure (from MATA 2026).

    NT2F_i = t_failure_i / seq_len_i
    where t_failure is the first frame (after init) where IoU drops below
    iou_threshold.  Sequences that never fail receive NT2F = 1.0.

    Parameters
    ----------
    iou_traces:
        Mapping from sequence key to per-frame IoU array, shape (n_frames,).
    iou_threshold:
        IoU level below which a frame is considered a tracking failure.

    Returns
    -------
    dict with keys: nt2f_mean, nt2f_std, n_sequences, n_never_failed.
    """
    scores: list[float] = []
    never_failed = 0

    for seq_key, iou in iou_traces.items():
        iou = np.asarray(iou, dtype=float)
        n = len(iou)
        if n < 2:
            continue
        # Look for the first failure *after* initialisation frame (index 0)
        failures = np.where(iou[1:] < iou_threshold)[0] + 1  # +1 for the slice offset
        if len(failures) == 0:
            scores.append(1.0)
            never_failed += 1
        else:
            t_failure = int(failures[0])
            scores.append(t_failure / max(n, 1))

    if not scores:
        return {
            "nt2f_mean": float("nan"),
            "nt2f_std": float("nan"),
            "n_sequences": 0,
            "n_never_failed": 0,
        }

    return {
        "nt2f_mean": float(np.mean(scores)),
        "nt2f_std": float(np.std(scores)),
        "n_sequences": len(scores),
        "n_never_failed": never_failed,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (sequence-level resampling)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    per_sequence_scores: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI via sequence-level bootstrap resampling.

    IMPORTANT: resampling is done at the sequence level, NOT the frame level.
    Adjacent frames within a sequence are correlated; frame-level bootstrap
    would yield artificially narrow CIs.

    Parameters
    ----------
    per_sequence_scores:
        One scalar per sequence (e.g. per-sequence AUPRC).
    n_bootstrap:
        Number of bootstrap resamples.
    alpha:
        Significance level (default 0.05 → 95% CI).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (lower, upper) confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(per_sequence_scores, dtype=float)
    boot_means = np.array(
        [
            rng.choice(scores, size=len(scores), replace=True).mean()
            for _ in range(n_bootstrap)
        ]
    )
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# GO/NO-GO gate
# ---------------------------------------------------------------------------


def check_go_nogo(results: dict[str, Any]) -> str:
    """Evaluate GO/NO-GO/BORDERLINE gate from eval results dict.

    Parameters
    ----------
    results:
        Dict as returned by :func:`evaluate`.  Must contain a
        ``"head_metrics"`` sub-dict with per-head entries.

    Returns
    -------
    "GO", "STOP", or "BORDERLINE".
    """
    head_metrics: dict[str, dict[str, float]] = results.get("head_metrics", {})

    def _get(key: str) -> float:
        """Resolve a flat metric key of the form '<metric>_<head>'."""
        # key examples: "auprc_false_confirmed", "auroc_failure_in_5",
        #               "ece_false_confirmed", "auroc_hard_dynamic_scene"
        parts = key.split("_", 1)  # split at first underscore only
        if len(parts) != 2:
            return float("nan")
        metric, head = parts[0], parts[1]
        return head_metrics.get(head, {}).get(metric, float("nan"))

    # Check STOP criteria first (any one → STOP)
    for key, threshold in STOP_THRESHOLDS.items():
        val = _get(key)
        if np.isnan(val):
            continue
        if key in _LOWER_IS_BETTER:
            if val > threshold:
                return "STOP"
        else:
            if val < threshold:
                return "STOP"

    # Check GO criteria (all must pass)
    go_count = 0
    go_total = 0
    for key, threshold in GO_THRESHOLDS.items():
        val = _get(key)
        if np.isnan(val):
            continue
        go_total += 1
        if key in _LOWER_IS_BETTER:
            if val < threshold:
                go_count += 1
        else:
            if val > threshold:
                go_count += 1

    if go_total == 0:
        return "BORDERLINE"
    if go_count == go_total:
        return "GO"
    return "BORDERLINE"


# ---------------------------------------------------------------------------
# NPZ + model loading helpers
# ---------------------------------------------------------------------------


def _load_npz_split(
    npz_path: str, split: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load features, labels, and iou_traces for the requested split.

    Parameters
    ----------
    npz_path:
        Path to the compressed NPZ produced by collect_features.py.
    split:
        One of "train", "val", "diagnostic".

    Returns
    -------
    (features_dict, labels_dict, iou_dict) each keyed by compound
    sequence key ``"dataset_name/seq_name"``.
    """
    data = np.load(npz_path, allow_pickle=True)

    compound_keys = [
        k[len("features/"):] for k in data.files if k.startswith("features/")
    ]

    features_dict: dict[str, np.ndarray] = {}
    labels_dict: dict[str, np.ndarray] = {}
    iou_dict: dict[str, np.ndarray] = {}

    for key in compound_keys:
        seq_split = str(data[f"split/{key}"])
        if seq_split != split:
            continue
        features_dict[key] = data[f"features/{key}"]
        labels_dict[key] = data[f"labels/{key}"]
        iou_dict[key] = data[f"iou_trace/{key}"]

    return features_dict, labels_dict, iou_dict


def _load_model(checkpoint_path: str, n_features: int, n_labels: int, device: str):
    """Load SALTRD model from checkpoint.

    Falls back gracefully when the model module has not yet been fully
    implemented (stub) — returns None in that case so callers can skip
    the inference step and still exercise the metric functions.

    Parameters
    ----------
    checkpoint_path:
        Path to a PyTorch checkpoint (.pt / .pth) saved by train.py.
    n_features:
        Number of input features (typically 28).
    n_labels:
        Number of output heads (typically 8).
    device:
        Torch device string, e.g. "cpu" or "cuda".

    Returns
    -------
    model or None.
    """
    try:
        import torch
        from salt_r.model import SALTRD

        model = SALTRD()
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:
        warnings.warn(
            f"Could not load model from {checkpoint_path}: {exc}. "
            "Skipping model inference — only dataset-level label statistics "
            "will be reported.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _run_inference(
    model,
    features_dict: dict[str, np.ndarray],
    window_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Run model inference over all sequences.

    Each sequence is processed as one batch of overlapping windows of
    length `window_size`.  When a sequence is shorter than `window_size`
    the whole sequence is used as a single window.

    Parameters
    ----------
    model:
        Loaded SALTRD model (or None → returns empty dict).
    features_dict:
        Per-sequence feature arrays keyed by compound seq key.
    window_size:
        Temporal window length fed to the GRU.
    device:
        Torch device string.

    Returns
    -------
    Dict mapping seq_key → per-frame probability array of shape
    (n_frames, n_heads).
    """
    if model is None:
        return {}

    import torch

    model = model.to(device)
    preds: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for seq_key, feats in features_dict.items():
            n = feats.shape[0]
            x = torch.tensor(feats, dtype=torch.float32, device=device)  # (n, F)

            # Build sliding windows: (n_windows, window_size, F)
            # Each position i uses frames [i-window_size+1 .. i] (causal)
            windows = []
            for t in range(n):
                start = max(0, t - window_size + 1)
                window = feats[start : t + 1]  # (<=window_size, F)
                # Left-pad with zeros to window_size
                pad_len = window_size - len(window)
                if pad_len > 0:
                    window = np.concatenate(
                        [np.zeros((pad_len, feats.shape[1]), dtype=np.float32), window],
                        axis=0,
                    )
                windows.append(window)

            x_batch = torch.tensor(
                np.stack(windows, axis=0), dtype=torch.float32, device=device
            )  # (n, window_size, F)

            out = model(x_batch)  # dict[str, Tensor(n,)] or Tensor(n, heads)
            if isinstance(out, dict):
                # dict head-name → (n,) tensor
                n_heads = len(out)
                prob_matrix = np.zeros((n, n_heads), dtype=np.float32)
                head_order = list(out.keys())
                for hi, hname in enumerate(head_order):
                    prob_matrix[:, hi] = (
                        out[hname].sigmoid().cpu().numpy()
                        if out[hname].requires_grad
                        or out[hname].dtype == torch.float32
                        else out[hname].cpu().numpy()
                    )
            else:
                # Tensor (n, n_heads)
                prob_matrix = torch.sigmoid(out).cpu().numpy()

            preds[seq_key] = prob_matrix

    return preds


# ---------------------------------------------------------------------------
# Formatted summary table
# ---------------------------------------------------------------------------


def _print_summary_table(
    head_metrics: dict[str, dict[str, float]],
    label_names: list[str],
) -> None:
    """Print a formatted table of per-head metrics to stdout."""
    col_w = 10
    header_fields = ["head", "base%", "AUROC", "AUPRC", "ECE", "Brier", "NLL", "R@5FPR"]
    header = (
        f"{'head':<22}"
        + "".join(f"{f:>{col_w}}" for f in header_fields[1:])
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for head in label_names:
        m = head_metrics.get(head, {})
        base_pct = m.get("base_rate", float("nan")) * 100

        def _fmt(v: float) -> str:
            return f"{v:>{col_w}.4f}" if not np.isnan(v) else f"{'n/a':>{col_w}}"

        row = (
            f"{head:<22}"
            f"{base_pct:>{col_w}.1f}"
            + _fmt(m.get("auroc", float("nan")))
            + _fmt(m.get("auprc", float("nan")))
            + _fmt(m.get("ece", float("nan")))
            + _fmt(m.get("brier", float("nan")))
            + _fmt(m.get("nll", float("nan")))
            + _fmt(m.get("recall_at_5pct_fpr", float("nan")))
        )
        print(row)

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Full evaluation entry point
# ---------------------------------------------------------------------------


def evaluate(
    npz_path: str,
    checkpoint_path: str,
    split: str = "val",
    window_size: int = 20,
    device: str = "cpu",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run full evaluation and return results dict.

    Computes per-head AUROC / AUPRC / ECE / Brier / NLL / Recall@5%FPR,
    NT2F (IoU thresholds 0.5 and 0.2), bootstrap CIs on per-sequence
    AUPRC(false_confirmed), and prints a formatted summary table.

    Parameters
    ----------
    npz_path:
        Path to the NPZ produced by collect_features.py.
    checkpoint_path:
        Path to the SALTRD .pt checkpoint produced by train.py.
    split:
        Which split to evaluate: "val" | "diagnostic" | "train".
    window_size:
        Temporal window fed to the GRU during inference.
    device:
        Torch device, e.g. "cpu" or "cuda".
    output_path:
        If provided, write the full results dict as JSON to this path.

    Returns
    -------
    dict containing:
    - "split": str
    - "n_sequences": int
    - "head_metrics": {head_name: {auroc, auprc, ece, brier, nll,
                                   recall_at_5pct_fpr, base_rate}}
    - "nt2f_05": nt2f result dict at IoU threshold 0.5
    - "nt2f_02": nt2f result dict at IoU threshold 0.2
    - "bootstrap_auprc_false_confirmed": {"ci_lo", "ci_hi", "n_sequences"}
    - "go_nogo": str
    """
    # ------------------------------------------------------------------
    # 1. Load NPZ
    # ------------------------------------------------------------------
    print(f"Loading NPZ: {npz_path}  split={split}")
    features_dict, labels_dict, iou_dict = _load_npz_split(npz_path, split)

    # Read label names from NPZ (fall back to collect_features default)
    try:
        _raw = np.load(npz_path, allow_pickle=True)
        label_names: list[str] = list(_raw["label_names"].tolist())
    except Exception:
        from salt_r.collect_features import LABEL_NAMES
        label_names = list(LABEL_NAMES)

    n_seqs = len(features_dict)
    print(f"  Found {n_seqs} sequences in split '{split}'")
    if n_seqs == 0:
        warnings.warn(
            f"No sequences found for split='{split}' in {npz_path}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 2. Load model and run inference
    # ------------------------------------------------------------------
    n_features = next(iter(features_dict.values())).shape[1] if features_dict else 28
    n_labels = len(label_names)

    model = _load_model(checkpoint_path, n_features, n_labels, device)
    preds_dict = _run_inference(model, features_dict, window_size, device)
    has_preds = bool(preds_dict)

    # ------------------------------------------------------------------
    # 3. Aggregate labels and predictions across sequences
    # ------------------------------------------------------------------
    # Aggregate all frames per head; also keep per-sequence AUPRC(false_confirmed)
    all_y_true: dict[str, list[np.ndarray]] = {h: [] for h in label_names}
    all_y_pred: dict[str, list[np.ndarray]] = {h: [] for h in label_names}
    per_seq_auprc_fc: list[float] = []  # for bootstrap CI

    fc_idx = label_names.index("false_confirmed") if "false_confirmed" in label_names else -1

    for seq_key in features_dict:
        lab = labels_dict[seq_key].astype(float)  # (n_frames, n_labels)
        if has_preds and seq_key in preds_dict:
            pred = preds_dict[seq_key]  # (n_frames, n_heads)
        else:
            # No model available — use label frequency as a trivial baseline
            pred = np.tile(lab.mean(axis=0), (len(lab), 1))

        for hi, head in enumerate(label_names):
            if hi >= lab.shape[1]:
                break
            all_y_true[head].append(lab[:, hi])
            # Use name-based prediction lookup (model predicts HEAD_NAMES only,
            # not "correct"). Avoids off-by-one when "correct" is label index 0.
            if has_preds and seq_key in preds_dict:
                p = preds_dict[seq_key]  # (n_frames, n_head_preds)
                try:
                    from salt_r.model import HEAD_NAMES as _HN
                    if head in _HN:
                        pred_col = _HN.index(head)
                        all_y_pred[head].append(p[:, pred_col] if pred_col < p.shape[1] else np.full(len(lab), 0.5, np.float32))
                    else:
                        all_y_pred[head].append(np.full(len(lab), 0.5, np.float32))
                except Exception:
                    all_y_pred[head].append(p[:, hi] if hi < p.shape[1] else np.full(len(lab), 0.5, np.float32))
            else:
                all_y_pred[head].append(np.full(len(lab), 0.5, dtype=np.float32))

        # Per-sequence AUPRC for false_confirmed (bootstrap input)
        if fc_idx >= 0:
            yt = lab[:, fc_idx]
            if has_preds and seq_key in preds_dict:
                try:
                    from salt_r.model import HEAD_NAMES as _HN2
                    fc_pred_idx = _HN2.index("false_confirmed")
                    yp = preds_dict[seq_key][:, fc_pred_idx]
                except Exception:
                    yp = preds_dict[seq_key][:, fc_idx] if fc_idx < preds_dict[seq_key].shape[1] else np.full(len(yt), yt.mean(), np.float32)
            else:
                yp = np.full(len(yt), yt.mean(), dtype=np.float32)
            br = float(yt.mean())
            if 0.0 < br < 1.0:
                try:
                    per_seq_auprc_fc.append(
                        float(_average_precision_score(yt, yp))
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 4. Compute per-head metrics
    # ------------------------------------------------------------------
    head_metrics: dict[str, dict[str, float]] = {}
    for head in label_names:
        y_true_cat = np.concatenate(all_y_true[head]) if all_y_true[head] else np.array([])
        y_pred_cat = np.concatenate(all_y_pred[head]) if all_y_pred[head] else np.array([])
        if len(y_true_cat) == 0:
            head_metrics[head] = {
                "base_rate": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
                "ece": float("nan"),
                "brier": float("nan"),
                "nll": float("nan"),
                "recall_at_5pct_fpr": float("nan"),
            }
        else:
            head_metrics[head] = compute_head_metrics(y_true_cat, y_pred_cat, head)

    # ------------------------------------------------------------------
    # 5. NT2F at IoU thresholds 0.5 and 0.2
    # ------------------------------------------------------------------
    nt2f_05 = compute_nt2f(iou_dict, iou_threshold=0.5)
    nt2f_02 = compute_nt2f(iou_dict, iou_threshold=0.2)

    # ------------------------------------------------------------------
    # 6. Bootstrap CI on per-sequence AUPRC(false_confirmed)
    # ------------------------------------------------------------------
    if len(per_seq_auprc_fc) >= 2:
        ci_lo, ci_hi = bootstrap_ci(per_seq_auprc_fc)
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    bootstrap_result = {
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_sequences": len(per_seq_auprc_fc),
    }

    # ------------------------------------------------------------------
    # 7. Print formatted table
    # ------------------------------------------------------------------
    _print_summary_table(head_metrics, label_names)

    print(f"NT2F (IoU>=0.5):  mean={nt2f_05['nt2f_mean']:.4f}  "
          f"std={nt2f_05['nt2f_std']:.4f}  "
          f"n={nt2f_05['n_sequences']}  "
          f"never_failed={nt2f_05['n_never_failed']}")
    print(f"NT2F (IoU>=0.2):  mean={nt2f_02['nt2f_mean']:.4f}  "
          f"std={nt2f_02['nt2f_std']:.4f}  "
          f"n={nt2f_02['n_sequences']}  "
          f"never_failed={nt2f_02['n_never_failed']}")

    if not np.isnan(ci_lo):
        print(
            f"Bootstrap AUPRC(false_confirmed) 95%CI:  "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"(n_seq={len(per_seq_auprc_fc)})"
        )

    # ------------------------------------------------------------------
    # 8. Assemble results dict
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "split": split,
        "n_sequences": n_seqs,
        "head_metrics": head_metrics,
        "nt2f_05": nt2f_05,
        "nt2f_02": nt2f_02,
        "bootstrap_auprc_false_confirmed": bootstrap_result,
    }

    # Determine GO/NO-GO
    verdict = check_go_nogo(results)
    results["go_nogo"] = verdict

    # ------------------------------------------------------------------
    # 9. Save JSON if requested
    # ------------------------------------------------------------------
    if output_path is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        def _json_safe(obj: Any) -> Any:
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_safe(v) for v in obj]
            return obj

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(results), fh, indent=2)
        print(f"\nResults written to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for SALT-RD evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SALTRD GRU checkpoint against an NPZ split."
    )
    parser.add_argument("--npz", required=True, help="Path to the NPZ dataset.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the SALTRD .pt checkpoint."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "diagnostic"],
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for JSON results.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (default: cpu).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="GRU temporal window size (default: 20).",
    )
    args = parser.parse_args()

    results = evaluate(
        npz_path=args.npz,
        checkpoint_path=args.checkpoint,
        split=args.split,
        window_size=args.window_size,
        device=args.device,
        output_path=args.output,
    )
    verdict = check_go_nogo(results)
    print(f"\nGO/NO-GO: {verdict}")


if __name__ == "__main__":
    main()
