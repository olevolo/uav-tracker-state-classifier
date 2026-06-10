"""Per-feature usefulness analyzer for CSC telemetry features.

Computes univariate AUROC / AUPRC / Pearson-r for each of the 11 telemetry
features against a binary failure label, plus a leave-one-out (LOO) model
ablation if a checkpoint is supplied, plus a feature-group aggregation.

Usage
-----
    python tools/diagnose_csc_features.py \\
        --labels_dir outputs/csc_labels/got10k \\
        --output outputs/feature_diagnosis/got10k.csv \\
        [--checkpoint outputs/csc_training/csc_gru_v2/checkpoint_best.pth] \\
        [--per_state] \\
        [--shallow_ablation]

Notes
-----
- Pure numpy — no sklearn (except for --shallow_ablation which uses
  sklearn.linear_model.LogisticRegression).
- Labels use two schemas:
    * New schema (post-refactor): has ``derived_state`` field.
      failure = derived_state in {2=LOST_AWARE, 3=FALSE_CONFIRMED}.
    * Old schema (pre-refactor): has ``state`` / ``state_name`` fields.
      failure = state_name == "LOST"  (maps to what new schema calls LOST_AWARE).
- Features are rebuilt via ``build_sequence_features`` so that velocity /
  acceleration are causal-correct (computed from pred_bbox trajectory).
- Val split: sequence_id mod 5 == 0.
- --per_state: compute per-state feature separation stats (Cohen's d, IQR, etc.)
- --shallow_ablation: group-level LogisticRegression baseline (single-frame).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csc_lib.csc.features import FEATURE_NAMES, build_sequence_features  # noqa: E402
from csc_lib.eval.custom_metrics.scene_state_metrics import (  # noqa: E402
    failure_auroc,
    failure_auprc,
)

# ---------------------------------------------------------------------------
# Feature-group definitions for Table-5 group ablation
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "confidence": ["confidence", "apce", "psr"],
    "bbox_dynamics": [
        "cx_norm", "cy_norm", "w_norm", "h_norm", "area_norm", "aspect_ratio"
    ],
    "motion": ["velocity_norm", "accel_norm"],
}

# ---------------------------------------------------------------------------
# Label loading helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_rows(labels_dir: Path) -> list[dict]:
    """Load all labels.jsonl files found recursively under labels_dir."""
    rows: list[dict] = []
    for p in sorted(labels_dir.rglob("labels.jsonl")):
        rows.extend(_load_jsonl(p))
    if not rows:
        raise FileNotFoundError(f"No labels.jsonl found under {labels_dir}")
    return rows


def _is_failure(row: dict) -> int:
    """Return 1 if the row represents a tracking failure, else 0.

    Supports both label schema versions:
    - New schema: ``derived_state`` int in {2=LOST_AWARE, 3=FALSE_CONFIRMED}.
    - Old schema: ``state_name`` == "LOST".
    """
    if "derived_state" in row:
        ds = int(row["derived_state"])
        return int(ds in (2, 3))  # LOST_AWARE=2, FALSE_CONFIRMED=3
    # Old schema fallback
    sn = row.get("state_name", "")
    return int(sn == "LOST")


def _group_by_sequence(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by (dataset, sequence) key string."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = f"{r['dataset']}/{r['sequence']}"
        groups[key].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["frame_idx"])
    return groups


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------


def _build_features_and_labels(
    groups: dict[str, list[dict]],
    image_size: tuple[int, int] = (1280, 720),
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Build pooled feature matrix (N, 11) and failure labels (N,).

    Returns
    -------
    features : shape (N, F)
    labels   : shape (N,) binary
    seq_keys : list of length N — sequence key per row (for val split)
    all_rows : list of length N — raw dicts in the same order (for NaN accounting)
    """
    feat_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    key_list: list[str] = []
    row_list: list[dict] = []

    for key, rows in sorted(groups.items()):
        feats = build_sequence_features(rows, image_size)  # (T, F)
        labels = np.array([_is_failure(r) for r in rows], dtype=np.int8)
        feat_list.append(feats)
        label_list.append(labels)
        key_list.extend([key] * len(rows))
        row_list.extend(rows)

    if not feat_list:
        raise RuntimeError("No data loaded.")

    return (
        np.concatenate(feat_list, axis=0),
        np.concatenate(label_list, axis=0),
        key_list,
        row_list,
    )


# ---------------------------------------------------------------------------
# Metric helpers (pure numpy)
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Signed Pearson r between x and y (both 1-D)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    denom = math.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _best_auroc(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Try both polarities; return the better AUROC and its polarity."""
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return 0.5, "+"
    xm = x.copy()
    xm[~mask] = 0.0  # treat NaN as zero score
    a_pos = failure_auroc(y, xm)
    a_neg = failure_auroc(y, -xm)
    if a_neg > a_pos:
        return a_neg, "-"
    return a_pos, "+"


def _best_auprc_with_polarity(
    x: np.ndarray, y: np.ndarray, polarity: str
) -> float:
    """Compute AUPRC using the given polarity sign."""
    mask = np.isfinite(x)
    xm = x.copy()
    xm[~mask] = 0.0
    score = xm if polarity == "+" else -xm
    return failure_auprc(y, score)


# ---------------------------------------------------------------------------
# LOO model ablation helpers
# ---------------------------------------------------------------------------


def _run_model_on_window_batch(
    model,          # CSCGRU in eval mode
    feat_matrix: np.ndarray,   # (N, F) full float32
    window_size: int,
    device: str,
    zeroed_feat_idx: int | None = None,
) -> np.ndarray:
    """Run model over feature matrix with a rolling window of size window_size.

    Returns risk_score (N,) where each element is P(LOST) for that frame.
    For the first window_size-1 frames, left-pad with the first frame.
    If zeroed_feat_idx is not None, that column is set to zero before inference.
    """
    import torch

    N, F = feat_matrix.shape
    feats = feat_matrix.copy()
    if zeroed_feat_idx is not None:
        feats[:, zeroed_feat_idx] = 0.0

    risks = np.zeros(N, dtype=np.float32)
    for t in range(N):
        if t < window_size:
            pad = window_size - t - 1
            window = np.concatenate(
                [np.repeat(feats[:1], pad, axis=0), feats[: t + 1]], axis=0
            )  # (window_size, F)
        else:
            window = feats[t - window_size + 1 : t + 1]
        x = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, W, F)
        with torch.no_grad():
            out = model.predict(x)
        risks[t] = float(out["risk_score"][0, -1].cpu().item())
    return risks


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def _compute_feature_stats(
    features: np.ndarray,  # (N, F)
    labels: np.ndarray,    # (N,) binary
    seq_keys: list[str],
    all_rows: list[dict],  # raw JSONL rows in same order as features
) -> list[dict]:
    """Compute per-feature stats over full set and val split."""
    N, F = features.shape
    assert len(FEATURE_NAMES) == F

    # Raw-field name → feature index for tracking original data missingness.
    # velocity_norm and accel_norm are derived; no direct raw field.
    _RAW_FIELD: dict[str, str | None] = {
        "confidence": "confidence",
        "apce": "apce",
        "psr": "psr",
        "cx_norm": "pred_bbox",   # from pred_bbox
        "cy_norm": "pred_bbox",
        "w_norm": "pred_bbox",
        "h_norm": "pred_bbox",
        "area_norm": "pred_bbox",
        "aspect_ratio": "pred_bbox",
        "velocity_norm": None,    # derived from pred_bbox trajectory
        "accel_norm": None,
    }

    # Pre-compute per-feature raw NaN fraction from the raw rows
    raw_nan_frac: dict[str, float] = {}
    for name in FEATURE_NAMES:
        field = _RAW_FIELD.get(name)
        if field is None:
            # Derived feature: NaN fraction is 0 (computed from trajectory)
            raw_nan_frac[name] = 0.0
        elif field == "pred_bbox":
            raw_nan_frac[name] = float(
                sum(1 for r in all_rows if r.get("pred_bbox") is None) / max(1, len(all_rows))
            )
        else:
            raw_nan_frac[name] = float(
                sum(1 for r in all_rows if r.get(field) is None) / max(1, len(all_rows))
            )

    # Val split: deterministic CRC32 hash, ~20% of sequences
    val_mask = np.array(
        [zlib.crc32(k.encode()) % 5 == 0 for k in seq_keys], dtype=bool
    )

    results = []
    for fi, name in enumerate(FEATURE_NAMES):
        col = features[:, fi].astype(np.float64)

        # --- full set ---
        auroc_full, polarity = _best_auroc(col, labels)
        auprc_full = _best_auprc_with_polarity(col, labels, polarity)
        pearson_full = _pearson(col, labels.astype(np.float64))

        # --- val split ---
        col_val = col[val_mask]
        lab_val = labels[val_mask]
        if lab_val.sum() == 0 or (~val_mask).sum() == 0:
            auroc_val = 0.5
            auprc_val = 0.0
        else:
            auroc_val, _ = _best_auroc(col_val, lab_val)
            auprc_val = _best_auprc_with_polarity(col_val, lab_val, polarity)

        support_pos = int(labels.sum())
        support_neg = int((labels == 0).sum())

        results.append({
            "feature": name,
            "polarity": polarity,
            "auroc_full": round(auroc_full, 6),
            "auprc_full": round(auprc_full, 6),
            "pearson_full": round(pearson_full, 6) if not math.isnan(pearson_full) else "nan",
            "auroc_val": round(auroc_val, 6),
            "auprc_val": round(auprc_val, 6),
            "support_pos": support_pos,
            "support_neg": support_neg,
            "fraction_nan": round(raw_nan_frac[name], 6),
        })

    return results


def _compute_group_stats(feature_stats: list[dict]) -> list[dict]:
    """Compute per-group max_auprc_val from feature-level results."""
    stat_by_name = {r["feature"]: r for r in feature_stats}
    group_rows = []
    for group_name, members in FEATURE_GROUPS.items():
        auprc_vals = [
            stat_by_name[m]["auprc_val"]
            for m in members
            if m in stat_by_name
            and not isinstance(stat_by_name[m]["auprc_val"], float)
            or m in stat_by_name
        ]
        # Filter out non-float values
        auprc_vals = [
            v for v in [stat_by_name[m]["auprc_val"] for m in members if m in stat_by_name]
            if isinstance(v, (int, float)) and not math.isnan(float(v))
        ]
        best_member = ""
        best_auprc = 0.0
        for m in members:
            if m not in stat_by_name:
                continue
            v = stat_by_name[m]["auprc_val"]
            if isinstance(v, (int, float)) and not math.isnan(float(v)) and float(v) > best_auprc:
                best_auprc = float(v)
                best_member = m
        group_rows.append({
            "group": group_name,
            "members": ",".join(members),
            "max_auprc_val": round(best_auprc, 6),
            "best_member": best_member,
        })
    return group_rows


def _compute_loo_stats(
    features: np.ndarray,   # (N, F)
    labels: np.ndarray,     # (N,) binary
    seq_keys: list[str],
    checkpoint_path: Path,
) -> list[dict]:
    """LOO feature ablation using a trained checkpoint."""
    import torch
    from csc_lib.csc.inference import load_runtime

    print("[LOO] Loading checkpoint...", flush=True)
    runtime = load_runtime(checkpoint_path, device="cpu")
    model = runtime.model.eval()
    window_size = runtime.feature_cfg.window_size

    val_mask = np.array([zlib.crc32(k.encode()) % 5 == 0 for k in seq_keys], dtype=bool)
    features_val = features[val_mask]
    labels_val = labels[val_mask]

    if labels_val.sum() == 0:
        print("[LOO] WARNING: no positive labels in val split — skipping LOO.", flush=True)
        return []

    print(f"[LOO] Baseline pass (no zeroing), val frames={val_mask.sum()}...", flush=True)
    # We need to run per-sequence to avoid bleeding between sequences
    val_seq_set = sorted({k for k, m in zip(seq_keys, val_mask) if m})

    # Group val sequences
    seq_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, (k, m) in enumerate(zip(seq_keys, val_mask)):
        if m:
            seq_to_idx[k].append(i)

    def _run_all_sequences(zeroed_idx: int | None) -> np.ndarray:
        """Run model on all val sequences; return pooled risk scores."""
        all_risks: list[tuple[int, float]] = []
        for seq_key in val_seq_set:
            idxs = seq_to_idx[seq_key]
            seq_feats = features[idxs]  # (T, F)
            risks = _run_model_on_window_batch(
                model, seq_feats, window_size, "cpu", zeroed_feat_idx=zeroed_idx
            )
            for idx, risk in zip(idxs, risks):
                all_risks.append((idx, risk))
        # Reorder by global index to align with labels_val
        all_risks.sort(key=lambda t: t[0])
        return np.array([r for _, r in all_risks], dtype=np.float32)

    risks_baseline = _run_all_sequences(None)
    auprc_baseline = failure_auprc(labels_val, risks_baseline)
    print(f"[LOO] Baseline val AUPRC = {auprc_baseline:.4f}", flush=True)

    loo_rows = []
    for fi, name in enumerate(FEATURE_NAMES):
        print(f"[LOO] Zeroing feature {fi+1}/{len(FEATURE_NAMES)}: {name}...", flush=True)
        risks_zeroed = _run_all_sequences(fi)
        auprc_zeroed = failure_auprc(labels_val, risks_zeroed)
        drop = round(auprc_baseline - auprc_zeroed, 6)
        # drop = baseline - zeroed:
        #   > 0  → zeroing lowered AUPRC → feature was useful (model needed it)
        #  ≈ 0   → dead weight
        #   < 0  → zeroing raised AUPRC → feature was confusing the model (flag it)
        note = ""
        if drop > 0.001:
            note = "useful"
        elif abs(drop) <= 0.001:
            note = "dead_weight"
        else:
            note = "confusing"  # rare — feature hurts the model

        loo_rows.append({
            "feature": name,
            "auprc_baseline": round(auprc_baseline, 6),
            "auprc_zeroed": round(auprc_zeroed, 6),
            "auprc_drop_when_zeroed": drop,
            "note": note,
        })
    return loo_rows


# ---------------------------------------------------------------------------
# Task 2a — Per-state feature separation (--per_state)
# ---------------------------------------------------------------------------

# State labels for LocalizationState (used as the target here per constraint)
_LOCALIZATION_STATE_NAMES = {0: "STABLE", 1: "UNCERTAIN", 2: "LOST"}
# Derived states: CORRECT_CONFIRMED=0, CORRECT_UNCERTAIN=1, LOST_AWARE=2, FALSE_CONFIRMED=3
_DERIVED_STATE_NAMES = {0: "CORRECT_CONFIRMED", 1: "CORRECT_UNCERTAIN", 2: "LOST_AWARE", 3: "FALSE_CONFIRMED"}


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d between two 1-D arrays (group a vs group b).

    Uses pooled standard deviation.  Returns NaN if either group is empty
    or total std is zero.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    ma = float(a.mean())
    mb = float(b.mean())
    na, nb = len(a), len(b)
    # Pooled variance
    if na + nb <= 2:
        return float("nan")
    pooled_var = ((na - 1) * float(a.var(ddof=1) if na > 1 else 0.0) +
                  (nb - 1) * float(b.var(ddof=1) if nb > 1 else 0.0)) / (na + nb - 2)
    if pooled_var <= 0.0:
        return float("nan")
    return (ma - mb) / math.sqrt(pooled_var)


def _get_localization_states(rows: list[dict]) -> np.ndarray:
    """Return localization_state per row (0=STABLE,1=UNCERTAIN,2=LOST)."""
    out = np.zeros(len(rows), dtype=np.int8)
    for i, r in enumerate(rows):
        # Support both label schema versions
        if "localization_state" in r:
            out[i] = int(r["localization_state"])
        elif "derived_state" in r:
            ds = int(r["derived_state"])
            # Derived: 0=CC→STABLE, 1=CU→UNCERTAIN, 2=LOST_AWARE→LOST, 3=FC→LOST
            if ds <= 0:
                out[i] = 0
            elif ds == 1:
                out[i] = 1
            else:
                out[i] = 2
        else:
            out[i] = 0
    return out


def _get_derived_states(rows: list[dict]) -> np.ndarray:
    """Return derived_state per row (0=CC,1=CU,2=LOST_AWARE,3=FC)."""
    out = np.zeros(len(rows), dtype=np.int8)
    for i, r in enumerate(rows):
        if "derived_state" in r:
            out[i] = int(r["derived_state"])
    return out


def _compute_per_state_separation(
    features: np.ndarray,   # (N, F)
    rows: list[dict],       # raw JSONL rows, same length
) -> list[dict]:
    """Compute per-state feature distribution stats and Cohen's d effect size.

    Uses derived_state (CORRECT_CONFIRMED/CORRECT_UNCERTAIN/LOST_AWARE/FALSE_CONFIRMED)
    as the primary target per the 3-head CSC architecture constraint.

    Returns list of rows for feature_state_summary.csv.
    """
    N, F = features.shape
    derived_states = _get_derived_states(rows)

    results: list[dict] = []

    for fi, fname in enumerate(FEATURE_NAMES):
        col = features[:, fi].astype(np.float64)
        valid = np.isfinite(col)
        total_valid = int(valid.sum())
        # missing_rate for this feature overall
        missing_rate_overall = 1.0 - total_valid / max(1, N)

        for state_id, state_name in _DERIVED_STATE_NAMES.items():
            in_state = valid & (derived_states == state_id)
            not_in_state = valid & (derived_states != state_id)

            g_in = col[in_state]
            g_out = col[not_in_state]

            n_samples = int(in_state.sum())
            # missing rate: frames labelled as this state but with NaN feature
            in_state_total = int((derived_states == state_id).sum())
            n_missing_in_state = in_state_total - n_samples
            missing_rate = n_missing_in_state / max(1, in_state_total)

            if n_samples == 0:
                continue

            mean_val   = float(np.mean(g_in))
            std_val    = float(np.std(g_in, ddof=1)) if len(g_in) > 1 else 0.0
            median_val = float(np.median(g_in))
            q25 = float(np.percentile(g_in, 25))
            q75 = float(np.percentile(g_in, 75))
            p10 = float(np.percentile(g_in, 10))
            p90 = float(np.percentile(g_in, 90))
            iqr = q75 - q25

            # median delta vs rest
            if len(g_out) > 0:
                median_rest = float(np.median(g_out))
                median_delta = median_val - median_rest
            else:
                median_delta = float("nan")

            d = _cohens_d(g_in, g_out)

            results.append({
                "feature": fname,
                "state": state_name,
                "count": n_samples,
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "median": round(median_val, 6),
                "iqr": round(iqr, 6),
                "p10": round(p10, 6),
                "p90": round(p90, 6),
                "missing_rate": round(missing_rate, 6),
                "median_delta_vs_rest": round(median_delta, 6) if not math.isnan(median_delta) else "nan",
                "effect_size_d": round(d, 6) if not math.isnan(d) else "nan",
            })

    return results


def _write_per_state_md(rows: list[dict], out_dir: Path) -> Path:
    """Write feature_state_summary.md from per-state rows."""
    md_path = out_dir / "feature_state_summary.md"
    lines = ["# Feature × Derived-State Summary\n"]
    lines.append(
        "Columns: feature, state, count, mean, std, median, IQR, p10, p90, "
        "missing_rate, median_delta_vs_rest, effect_size_d (Cohen's d)\n"
    )
    lines.append(
        "| feature | state | count | mean | std | median | iqr | p10 | p90 | "
        "missing_rate | median_delta | effect_d |"
    )
    lines.append(
        "|---------|-------|-------|------|-----|--------|-----|-----|-----|"
        "-------------|-------------|---------|"
    )
    for r in rows:
        lines.append(
            f"| {r['feature']} | {r['state']} | {r['count']} "
            f"| {r['mean']} | {r['std']} | {r['median']} | {r['iqr']} "
            f"| {r['p10']} | {r['p90']} | {r['missing_rate']} "
            f"| {r['median_delta_vs_rest']} | {r['effect_size_d']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return md_path


# ---------------------------------------------------------------------------
# Task 2b — Shallow model group ablation (--shallow_ablation)
# ---------------------------------------------------------------------------

# 4 groups from CSC.md §9 + "all"
ABLATION_GROUPS: dict[str, list[str]] = {
    "confidence": ["confidence", "apce", "psr"],
    "bbox_dynamics": ["cx_norm", "cy_norm", "w_norm", "h_norm", "area_norm", "aspect_ratio"],
    "motion": ["velocity_norm", "accel_norm"],
    "all": list(FEATURE_NAMES),  # all 11
}


def _macro_f1_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 across all unique classes."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    tp = int(((y_pred == cls) & (y_true == cls)).sum())
    fp = int(((y_pred == cls) & (y_true != cls)).sum())
    fn = int(((y_pred != cls) & (y_true == cls)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _compute_shallow_ablation(
    features: np.ndarray,   # (N, F) float32
    rows: list[dict],       # raw label rows (same length)
    seq_keys: list[str],    # sequence key per row
) -> list[dict]:
    """Train LogisticRegression + RandomForestClassifier on each feature group.

    Target: derived_state (4-class: 0=CC, 1=CU, 2=LA, 3=FC).
    Split: sequence-level 80/20 (deterministic CRC32 hash, same as val_mask logic).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print(
            "[shallow_ablation] WARNING: scikit-learn not installed — skipping. "
            "Install scikit-learn to enable shallow ablation.",
            flush=True,
        )
        return []

    # ---- Build target: derived_state ----
    derived_states = _get_derived_states(rows)

    # ---- Sequence-level train/val split (80/20 by CRC32) ----
    val_mask = np.array([zlib.crc32(k.encode()) % 5 == 0 for k in seq_keys], dtype=bool)

    X_train = features[~val_mask].astype(np.float32)
    y_train = derived_states[~val_mask]
    X_val = features[val_mask].astype(np.float32)
    y_val = derived_states[val_mask]

    if len(np.unique(y_train)) < 2:
        print("[shallow_ablation] WARNING: only 1 class in train split — skipping.", flush=True)
        return []

    # Majority baseline (most common derived_state in train)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    majority_class = int(unique_classes[np.argmax(class_counts)])
    y_maj = np.full_like(y_val, fill_value=majority_class)
    majority_macro_f1 = _macro_f1_from_preds(y_val, y_maj)

    # Check which derived classes appear in labels
    has_fc = int(3) in np.unique(y_val)  # FALSE_CONFIRMED

    results: list[dict] = []

    for group_name, members in ABLATION_GROUPS.items():
        feat_idxs = [FEATURE_NAMES.index(m) for m in members if m in FEATURE_NAMES]
        if not feat_idxs:
            print(f"[shallow_ablation] WARNING: no features for group {group_name}", flush=True)
            continue

        X_tr = np.nan_to_num(X_train[:, feat_idxs], nan=0.0, posinf=0.0, neginf=0.0)
        X_vl = np.nan_to_num(X_val[:, feat_idxs], nan=0.0, posinf=0.0, neginf=0.0)

        for model_name, clf in [
            ("logreg",
             LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
            ("rf",
             RandomForestClassifier(
                 max_depth=5, n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1
             )),
        ]:
            print(
                f"[shallow_ablation] group={group_name!r} model={model_name} "
                f"({len(feat_idxs)} features, train={len(X_tr)}, val={len(X_vl)}) ...",
                flush=True,
            )
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_vl)

            macro_f1 = _macro_f1_from_preds(y_val, y_pred)
            n_pred_classes = int(len(np.unique(y_pred)))

            # Per-class F1 for all 4 derived states
            f1_cc  = _class_f1(y_val, y_pred, cls=0)  # CORRECT_CONFIRMED
            f1_cu  = _class_f1(y_val, y_pred, cls=1)  # CORRECT_UNCERTAIN
            f1_la  = _class_f1(y_val, y_pred, cls=2)  # LOST_AWARE
            f1_fc  = _class_f1(y_val, y_pred, cls=3)  # FALSE_CONFIRMED

            # Balanced accuracy (mean recall per class)
            bal_acc_parts = []
            for c in np.unique(y_val):
                mask_c = y_val == c
                if mask_c.sum() > 0:
                    bal_acc_parts.append(float((y_pred[mask_c] == c).mean()))
            bal_acc = float(np.mean(bal_acc_parts)) if bal_acc_parts else 0.0

            beats_majority = (
                macro_f1 >= majority_macro_f1 + 0.10
                and n_pred_classes >= 3
                and (not has_fc or f1_fc >= 0.25)
            )

            results.append({
                "group": group_name,
                "model": model_name,
                "n_features": len(feat_idxs),
                "macro_f1": round(macro_f1, 6),
                "balanced_accuracy": round(bal_acc, 6),
                "f1_CORRECT_CONFIRMED": round(f1_cc, 6),
                "f1_CORRECT_UNCERTAIN": round(f1_cu, 6),
                "f1_LOST_AWARE": round(f1_la, 6),
                "f1_FALSE_CONFIRMED": round(f1_fc, 6),
                "n_pred_classes": n_pred_classes,
                "majority_macro_f1": round(majority_macro_f1, 6),
                "beats_gate": beats_majority,
            })

            gate_str = "GATE_PASS" if beats_majority else "GATE_FAIL"
            print(
                f"  → macro_f1={macro_f1:.4f} bal_acc={bal_acc:.4f} "
                f"f1_FC={f1_fc:.4f} n_pred_classes={n_pred_classes} {gate_str}",
                flush=True,
            )

    return results


def _write_ablation_md(rows: list[dict], out_dir: Path) -> Path:
    """Write feature_group_ablation.md including gate rule."""
    md_path = out_dir / "feature_group_ablation.md"
    lines = ["# Feature Group Ablation — Shallow Models\n"]
    lines.append("**Target**: derived_state (4-class: CORRECT_CONFIRMED / CORRECT_UNCERTAIN / LOST_AWARE / FALSE_CONFIRMED)\n")
    lines.append("**Models**: LogisticRegression (C=1, balanced, max_iter=1000) + RandomForestClassifier (max_depth=5, n_estimators=100, balanced)\n")
    lines.append("**Split**: sequence-level 80/20 GroupShuffleSplit by CRC32(sequence_id) % 5 == 0 for val\n")
    lines.append(
        "\n**Gate rule** (`all_features` group, best model): "
        "Macro-F1 must beat majority baseline by ≥ +0.10, "
        "≥ 3 distinct classes predicted, "
        "and F1_FALSE_CONFIRMED ≥ 0.25 (if FALSE_CONFIRMED exists in labels).\n"
    )
    lines.append(
        "| group | model | n_feat | macro_f1 | bal_acc | f1_CC | f1_CU | f1_LA | f1_FC "
        "| n_pred_cls | maj_f1 | gate |"
    )
    lines.append(
        "|-------|-------|--------|----------|---------|-------|-------|-------|-------|"
        "-----------|--------|------|"
    )
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['model']} | {r['n_features']} "
            f"| {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} "
            f"| {r['f1_CORRECT_CONFIRMED']:.4f} | {r['f1_CORRECT_UNCERTAIN']:.4f} "
            f"| {r['f1_LOST_AWARE']:.4f} | {r['f1_FALSE_CONFIRMED']:.4f} "
            f"| {r['n_pred_classes']} | {r['majority_macro_f1']:.4f} "
            f"| {'PASS' if r['beats_gate'] else 'FAIL'} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*70}", flush=True)
    if not rows:
        print("  (empty)", flush=True)
        return
    keys = list(rows[0].keys())
    # header
    print("  " + "  ".join(f"{k:>18}" for k in keys), flush=True)
    print("  " + "-" * (20 * len(keys)), flush=True)
    for r in rows:
        vals = []
        for k in keys:
            v = r[k]
            if isinstance(v, float):
                vals.append(f"{v:>18.6f}")
            else:
                vals.append(f"{str(v):>18}")
        print("  " + "  ".join(vals), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-feature usefulness analyzer for CSC telemetry."
    )
    parser.add_argument(
        "--labels_dir",
        required=True,
        type=Path,
        help="Root label directory (searches recursively for labels.jsonl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path (e.g. outputs/feature_diagnosis/got10k.csv).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a trained CSCGRU checkpoint (new composite schema). "
             "If supplied, also runs LOO ablation.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("W", "H"),
        help="Image size for bbox normalisation (default: 1280 720).",
    )
    parser.add_argument(
        "--per_state",
        action="store_true",
        default=False,
        help="Compute per-state feature separation stats (Cohen's d, IQR, etc.) "
             "for LocalizationState and DerivedState axes.",
    )
    parser.add_argument(
        "--shallow_ablation",
        action="store_true",
        default=False,
        help="Run shallow (single-frame LogisticRegression) group ablation on "
             "4 feature groups (confidence / bbox_dynamics / motion / all).",
    )
    args = parser.parse_args()

    image_size: tuple[int, int] = (args.image_size[0], args.image_size[1])

    print("[1/5] Loading labels...", flush=True)
    rows = _collect_rows(args.labels_dir)
    print(f"      Loaded {len(rows):,} rows from {args.labels_dir}", flush=True)

    print("[2/5] Grouping by sequence and building features...", flush=True)
    groups = _group_by_sequence(rows)
    n_seq = len(groups)
    features, labels, seq_keys, all_rows = _build_features_and_labels(groups, image_size)
    print(
        f"      {n_seq} sequences, {len(features):,} frames, "
        f"{int(labels.sum())} failures ({100*labels.mean():.1f}%)",
        flush=True,
    )

    val_mask = np.array([zlib.crc32(k.encode()) % 5 == 0 for k in seq_keys], dtype=bool)
    n_val = int(val_mask.sum())
    n_val_pos = int(labels[val_mask].sum())
    print(f"      Val split: {n_val:,} frames, {n_val_pos} failures", flush=True)

    # ---- Per-state mode (early exit if only --per_state) ----
    if args.per_state and not args.shallow_ablation and args.checkpoint is None:
        print("[per_state] Computing per-state feature separation...", flush=True)
        ps_rows = _compute_per_state_separation(features, all_rows)
        PS_FIELDS = [
            "feature", "state", "count", "mean", "std", "median", "iqr",
            "p10", "p90", "missing_rate", "median_delta_vs_rest", "effect_size_d",
        ]
        out_dir = args.output.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        ps_csv = out_dir / "feature_state_summary.csv"
        _write_csv(ps_csv, ps_rows, PS_FIELDS)
        md_path = _write_per_state_md(ps_rows, out_dir)
        print(f"Wrote {md_path}", flush=True)
        _print_table(ps_rows[:20], "Per-state separation (first 20 rows)")
        # Print top-3 LOST_AWARE effect_size_d
        la_rows = sorted(
            [r for r in ps_rows if r["state"] == "LOST_AWARE"],
            key=lambda r: abs(float(r["effect_size_d"])) if r["effect_size_d"] != "nan" else 0.0,
            reverse=True,
        )
        print("\nTop-3 features by LOST_AWARE effect_size_d:", flush=True)
        for i, r in enumerate(la_rows[:3]):
            print(f"  {i+1}. {r['feature']}: d={r['effect_size_d']}", flush=True)
        print("\nDone.", flush=True)
        return

    # ---- Shallow ablation mode (early exit if only --shallow_ablation) ----
    if args.shallow_ablation and not args.per_state and args.checkpoint is None:
        print("[shallow_ablation] Running group ablation...", flush=True)
        sa_rows = _compute_shallow_ablation(features, all_rows, seq_keys)
        SA_FIELDS = [
            "group", "model", "n_features", "macro_f1", "balanced_accuracy",
            "f1_CORRECT_CONFIRMED", "f1_CORRECT_UNCERTAIN", "f1_LOST_AWARE",
            "f1_FALSE_CONFIRMED", "n_pred_classes", "majority_macro_f1", "beats_gate",
        ]
        out_dir = args.output.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        sa_csv = out_dir / "feature_group_ablation.csv"
        _write_csv(sa_csv, sa_rows, SA_FIELDS)
        md_path = _write_ablation_md(sa_rows, out_dir)
        print(f"Wrote {md_path}", flush=True)
        _print_table(sa_rows, "Shallow ablation (feature groups × models)")
        print("\nDone.", flush=True)
        return

    print("[3/5] Computing univariate feature statistics...", flush=True)
    feature_stats = _compute_feature_stats(features, labels, seq_keys, all_rows)

    # Sort by auprc_val descending
    feature_stats_sorted = sorted(
        feature_stats, key=lambda r: -float(r["auprc_val"])
    )

    FEAT_FIELDS = [
        "feature", "polarity", "auroc_full", "auprc_full",
        "pearson_full", "auroc_val", "auprc_val",
        "support_pos", "support_neg", "fraction_nan",
    ]
    _write_csv(args.output, feature_stats_sorted, FEAT_FIELDS)
    _print_table(feature_stats_sorted, "Feature stats (sorted by auprc_val DESC)")

    print("[4/5] Computing feature-group aggregation...", flush=True)
    group_stats = _compute_group_stats(feature_stats)
    group_path = args.output.with_name(
        args.output.stem + "_groups" + args.output.suffix
    )
    GROUP_FIELDS = ["group", "members", "max_auprc_val", "best_member"]
    _write_csv(group_path, group_stats, GROUP_FIELDS)
    _print_table(group_stats, "Feature group max-AUPRC (val split)")

    print("[5/5] LOO ablation...", flush=True)
    if args.checkpoint is None:
        print("      --checkpoint not supplied; skipping LOO.", flush=True)
    elif not args.checkpoint.exists():
        print(f"      Checkpoint {args.checkpoint} not found; skipping LOO.", flush=True)
    else:
        loo_rows = _compute_loo_stats(
            features, labels, seq_keys, args.checkpoint
        )
        if loo_rows:
            loo_path = args.output.with_name(
                args.output.stem + "_loo" + args.output.suffix
            )
            LOO_FIELDS = [
                "feature", "auprc_baseline", "auprc_zeroed",
                "auprc_drop_when_zeroed", "note"
            ]
            loo_rows_sorted = sorted(
                loo_rows, key=lambda r: -r["auprc_drop_when_zeroed"]
            )
            _write_csv(loo_path, loo_rows_sorted, LOO_FIELDS)
            _print_table(loo_rows_sorted, "LOO ablation (sorted by AUPRC drop DESC)")

    # ---- Optional extra analyses (when combined with main flow) ----
    out_dir = args.output.parent
    if args.per_state:
        print("[per_state] Computing per-state feature separation...", flush=True)
        ps_rows = _compute_per_state_separation(features, all_rows)
        ps_csv = out_dir / "feature_state_summary.csv"
        PS_FIELDS = [
            "feature", "state", "count", "mean", "std", "median", "iqr",
            "p10", "p90", "missing_rate", "median_delta_vs_rest", "effect_size_d",
        ]
        _write_csv(ps_csv, ps_rows, PS_FIELDS)
        md_path = _write_per_state_md(ps_rows, out_dir)
        print(f"Wrote {md_path}", flush=True)
        # Print top-3 LOST_AWARE effect_size_d
        la_rows = sorted(
            [r for r in ps_rows if r["state"] == "LOST_AWARE"],
            key=lambda r: abs(float(r["effect_size_d"])) if r["effect_size_d"] != "nan" else 0.0,
            reverse=True,
        )
        print("  Top-3 features by LOST_AWARE effect_size_d:", flush=True)
        for i, r in enumerate(la_rows[:3]):
            print(f"    {i+1}. {r['feature']}: d={r['effect_size_d']}", flush=True)

    if args.shallow_ablation:
        print("[shallow_ablation] Running group ablation...", flush=True)
        sa_rows = _compute_shallow_ablation(features, all_rows, seq_keys)
        sa_csv = out_dir / "feature_group_ablation.csv"
        SA_FIELDS = [
            "group", "model", "n_features", "macro_f1", "balanced_accuracy",
            "f1_CORRECT_CONFIRMED", "f1_CORRECT_UNCERTAIN", "f1_LOST_AWARE",
            "f1_FALSE_CONFIRMED", "n_pred_classes", "majority_macro_f1", "beats_gate",
        ]
        _write_csv(sa_csv, sa_rows, SA_FIELDS)
        md_path = _write_ablation_md(sa_rows, out_dir)
        print(f"Wrote {md_path}", flush=True)
        _print_table(sa_rows, "Shallow ablation (feature groups × models)")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
