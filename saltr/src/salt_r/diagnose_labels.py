"""diagnose_labels.py — Audit hard_dynamic_scene label contamination.

Quantifies the overlap between hard_dynamic_scene and failure_in_5, and
shows how much of hard_dynamic_scene is driven by the future_risk term vs
the motion/ambiguity terms (peak_margin_low, flow_consistency_low).

Usage::

    python -m salt_r.diagnose_labels \\
        --npz saltr/data/salt_rd_v0.npz \\
        --output saltr/results/label_audit.json

Outputs a JSON and a console table with per-split, per-dataset statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Label column indices (must match LABEL_NAMES in collect_features.py)
_IDX = {
    "correct": 0,
    "false_confirmed": 1,
    "failure_in_5": 2,
    "recoverable": 3,
    "target_dynamic": 4,
    "camera_dynamic": 5,
    "hard_dynamic_scene": 6,
    "needs_full_compute": 7,
}


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def _recompute_hard_dyn_pure(
    labels: np.ndarray,
    features: np.ndarray,
    iou_trace: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Recompute hard_dynamic_scene WITHOUT the future_risk term.

    Uses the same peak_margin_low and flow_consistency_low thresholds as
    collect_features.compute_labels, but drops the iou lookahead.

    Returns binary array of shape (n_frames,).
    """
    n = len(iou_trace)
    is_dynamic = (labels[:, _IDX["target_dynamic"]] | labels[:, _IDX["camera_dynamic"]]).astype(bool)

    # Recover peak_margin and flow_consistency from the feature matrix
    peak_margin = np.zeros(n, dtype=np.float32)
    flow_consistency = np.full(n, 1.0, dtype=np.float32)
    for i, name in enumerate(feature_names):
        if name == "peak_margin" and i < features.shape[1]:
            peak_margin = features[:, i].astype(float)
        if name == "flow_consistency" and i < features.shape[1]:
            flow_consistency = features[:, i].astype(float)

    peak_margin_low = peak_margin < np.percentile(peak_margin, 25)
    flow_consistency_low = flow_consistency < 0.3

    return (is_dynamic & (peak_margin_low | flow_consistency_low)).astype(np.int8)


def _future_risk_mask(iou_trace: np.ndarray) -> np.ndarray:
    """Recompute the future_risk term: iou[t+1:t+6].min() < 0.3."""
    n = len(iou_trace)
    fr = np.zeros(n, dtype=bool)
    for t in range(n):
        future = iou_trace[t + 1 : t + 6]
        if len(future) > 0 and future.min() < 0.3:
            fr[t] = True
    return fr.astype(np.int8)


def _cond_prob(a: np.ndarray, b: np.ndarray) -> float:
    """P(a=1 | b=1). Returns NaN if P(b)=0."""
    n_b = int(b.sum())
    if n_b == 0:
        return float("nan")
    return float((a & b).sum()) / n_b


def audit_split(
    labels_list: list[np.ndarray],
    features_list: list[np.ndarray],
    iou_list: list[np.ndarray],
    feature_names: list[str],
    dataset_list: list[str],
) -> dict[str, Any]:
    """Run full audit for one split."""
    if not labels_list:
        return {}

    all_labels = np.concatenate(labels_list, axis=0)
    all_features = np.concatenate(features_list, axis=0)
    all_iou = np.concatenate(iou_list)
    all_datasets = np.array([ds for ds, lab in zip(dataset_list, labels_list)
                              for _ in range(len(lab))])

    n_label_cols = all_labels.shape[1]
    has_v1 = n_label_cols >= 10

    hd = all_labels[:, _IDX["hard_dynamic_scene"]].astype(bool)
    fi5 = all_labels[:, _IDX["failure_in_5"]].astype(bool)
    td = all_labels[:, _IDX["target_dynamic"]].astype(bool)
    cd = all_labels[:, _IDX["camera_dynamic"]].astype(bool)
    nfc = all_labels[:, _IDX["needs_full_compute"]].astype(bool)

    # v1 columns read directly from label array when available; else recompute
    if has_v1:
        all_pure = all_labels[:, 8].astype(bool)   # hard_dynamic_scene_v2
        ifd_col = all_labels[:, 9].astype(bool)    # imminent_failure_dynamic
        all_fr = ifd_col | (~ifd_col & np.zeros(len(all_labels), dtype=bool))  # placeholder
        # Recompute future_risk for correlation analysis (not stored in v0/v1 directly)
        fr_list = []
        for lab, iou in zip(labels_list, iou_list):
            fr_list.append(_future_risk_mask(iou))
        all_fr = np.concatenate(fr_list).astype(bool)
    else:
        # Recompute from features for v0 NPZ
        pure_list, fr_list = [], []
        for lab, feat, iou in zip(labels_list, features_list, iou_list):
            pure_list.append(_recompute_hard_dyn_pure(lab, feat, iou, feature_names))
            fr_list.append(_future_risk_mask(iou))
        all_pure = np.concatenate(pure_list).astype(bool)
        all_fr = np.concatenate(fr_list).astype(bool)
        ifd_col = None

    is_dynamic = (td | cd)
    hd_only_future_risk = hd & ~all_pure

    base_rates: dict[str, float] = {
        "P(hard_dynamic_scene)": float(hd.mean()),
        "P(hard_dynamic_scene_pure)": float(all_pure.mean()),
        "P(failure_in_5)": float(fi5.mean()),
        "P(target_dynamic)": float(td.mean()),
        "P(camera_dynamic)": float(cd.mean()),
        "P(needs_full_compute)": float(nfc.mean()),
        "P(future_risk)": float(all_fr.mean()),
    }
    if has_v1:
        base_rates["P(hard_dynamic_scene_v2)"] = float(all_pure.mean())
        base_rates["P(imminent_failure_dynamic)"] = float(ifd_col.mean())

    overlap: dict[str, Any] = {
        "P(hd ∩ fi5)": float((hd & fi5).mean()),
        "P(hd | fi5)": float((hd | fi5).mean()),
        "P(fi5 | hd)": _cond_prob(fi5.astype(np.int8), hd.astype(np.int8)),
        "P(hd | fi5)_cond": _cond_prob(hd.astype(np.int8), fi5.astype(np.int8)),
        "P(hd_pure ∩ fi5)": float((all_pure & fi5).mean()),
        "P(hd_driven_only_by_future_risk)": float(hd_only_future_risk.mean()),
        "frac_hd_explained_by_future_risk": float(
            hd_only_future_risk.sum() / max(hd.sum(), 1)
        ),
        "frac_hd_without_future_risk": float(
            all_pure.sum() / max(hd.sum(), 1)
        ),
    }
    if has_v1 and ifd_col is not None:
        overlap["P(imminent_failure_dynamic | failure_in_5)"] = _cond_prob(
            ifd_col.astype(np.int8), fi5.astype(np.int8)
        )
        overlap["P(failure_in_5 | imminent_failure_dynamic)"] = _cond_prob(
            fi5.astype(np.int8), ifd_col.astype(np.int8)
        )
        overlap["P(hds_v2 ∩ ifd)"] = float((all_pure & ifd_col).mean())
        overlap["frac_ifd_also_hds_v2"] = float(
            (all_pure & ifd_col).sum() / max(ifd_col.sum(), 1)
        )

    stats: dict[str, Any] = {
        "n_frames": int(len(all_labels)),
        "label_schema": "v1" if has_v1 else "v0",
        "base_rates": base_rates,
        "overlap": overlap,
        "per_dataset": {},
    }

    for ds in sorted(set(all_datasets)):
        mask = all_datasets == ds
        if not mask.any():
            continue
        hd_ds = hd[mask]
        fi5_ds = fi5[mask]
        pure_ds = all_pure[mask]
        fr_ds = all_fr[mask]
        per_ds: dict[str, Any] = {
            "n_frames": int(mask.sum()),
            "P(hard_dynamic_scene)": float(hd_ds.mean()),
            "P(hard_dynamic_scene_pure)": float(pure_ds.mean()),
            "P(failure_in_5)": float(fi5_ds.mean()),
            "P(future_risk)": float(fr_ds.mean()),
            "P(fi5 | hd)": _cond_prob(fi5_ds.astype(np.int8), hd_ds.astype(np.int8)),
            "frac_hd_explained_by_future_risk": float(
                (hd_ds & ~pure_ds).sum() / max(hd_ds.sum(), 1)
            ),
        }
        if has_v1 and ifd_col is not None:
            ifd_ds = ifd_col[mask]
            per_ds["P(imminent_failure_dynamic)"] = float(ifd_ds.mean())
            per_ds["P(failure_in_5 | imminent_failure_dynamic)"] = _cond_prob(
                fi5_ds.astype(np.int8), ifd_ds.astype(np.int8)
            )
        stats["per_dataset"][ds] = per_ds

    return stats


# ---------------------------------------------------------------------------
# IoU / APCE correlation analysis
# ---------------------------------------------------------------------------

def correlation_with_iou_drops(
    labels_list: list[np.ndarray],
    iou_list: list[np.ndarray],
    features_list: list[np.ndarray],
    feature_names: list[str],
) -> dict[str, float]:
    """Pearson r between hard_dynamic_scene and various IoU degradation signals."""
    hd_all, drop_all, apce_drop_all = [], [], []

    apce_idx = feature_names.index("apce_raw") if "apce_raw" in feature_names else -1

    for lab, iou, feat in zip(labels_list, iou_list, features_list):
        hd = lab[:, _IDX["hard_dynamic_scene"]].astype(float)
        n = len(iou)

        # IoU drop at t+1 relative to t
        iou_drop = np.zeros(n)
        for t in range(n - 1):
            drop = max(0.0, iou[t] - iou[t + 1])
            iou_drop[t] = drop
        iou_drop[-1] = 0.0

        # APCE drop signal
        apce_drop = np.zeros(n)
        if apce_idx >= 0:
            apce = feat[:, apce_idx].astype(float)
            for t in range(1, n):
                apce_drop[t] = max(0.0, apce[t - 1] - apce[t])

        hd_all.append(hd)
        drop_all.append(iou_drop)
        apce_drop_all.append(apce_drop)

    hd_cat = np.concatenate(hd_all)
    drop_cat = np.concatenate(drop_all)
    apce_cat = np.concatenate(apce_drop_all)

    def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
        if x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "r(hard_dynamic_scene, iou_drop_t+1)": pearsonr(hd_cat, drop_cat),
        "r(hard_dynamic_scene, apce_drop_t)": pearsonr(hd_cat, apce_cat),
        "r(failure_in_5, iou_drop_t+1)": pearsonr(
            np.concatenate([lab[:, _IDX["failure_in_5"]].astype(float)
                            for lab in labels_list]), drop_cat
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_audit(npz_path: str) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)

    try:
        feature_names: list[str] = list(data["feature_names"].tolist())
    except Exception:
        from salt_r.collect_features import FEATURE_NAMES
        feature_names = list(FEATURE_NAMES)

    keys = [k[len("features/"):] for k in data.files if k.startswith("features/")]

    by_split: dict[str, dict[str, list]] = {}

    for key in keys:
        split = str(data[f"split/{key}"])
        dataset = str(data[f"dataset/{key}"])
        features = data[f"features/{key}"].astype(np.float32)
        labels = data[f"labels/{key}"].astype(np.int8)
        iou = data[f"iou_trace/{key}"].astype(np.float32)

        if split not in by_split:
            by_split[split] = {
                "labels": [], "features": [], "iou": [], "datasets": []
            }
        by_split[split]["labels"].append(labels)
        by_split[split]["features"].append(features)
        by_split[split]["iou"].append(iou)
        by_split[split]["datasets"].append(dataset)

    results: dict[str, Any] = {"npz_path": npz_path, "per_split": {}}

    for split, buckets in by_split.items():
        split_stats = audit_split(
            buckets["labels"],
            buckets["features"],
            buckets["iou"],
            feature_names,
            buckets["datasets"],
        )
        corr = correlation_with_iou_drops(
            buckets["labels"],
            buckets["iou"],
            buckets["features"],
            feature_names,
        )
        split_stats["iou_apce_correlations"] = corr
        results["per_split"][split] = split_stats

    return results


def _print_audit(results: dict[str, Any]) -> None:
    for split, stats in results.get("per_split", {}).items():
        print(f"\n{'='*60}")
        print(f"Split: {split}  (n_frames={stats.get('n_frames', '?')})")
        print(f"{'='*60}")

        print("\n--- Base rates ---")
        for k, v in stats.get("base_rates", {}).items():
            bar = "#" * int(v * 40)
            print(f"  {k:<35} {v:.4f}  {bar}")

        print("\n--- Overlap / contamination ---")
        for k, v in stats.get("overlap", {}).items():
            if isinstance(v, float) and not (v != v):  # not nan
                print(f"  {k:<50} {v:.4f}")
            else:
                print(f"  {k:<50} {v}")

        print("\n--- Correlations with IoU/APCE drops ---")
        for k, v in stats.get("iou_apce_correlations", {}).items():
            print(f"  {k:<55} {v:.4f}")

        print("\n--- Per dataset ---")
        for ds, ds_stats in stats.get("per_dataset", {}).items():
            print(f"  [{ds}]")
            for k, v in ds_stats.items():
                if k == "n_frames":
                    continue
                print(f"    {k:<45} {v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit hard_dynamic_scene label contamination.")
    parser.add_argument("--npz", required=True, help="Path to salt_rd NPZ file")
    parser.add_argument("--output", default=None, help="Write JSON audit to this path")
    args = parser.parse_args()

    results = run_audit(args.npz)
    _print_audit(results)

    if args.output:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        def _safe(obj: Any) -> Any:
            if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
                return None
            if isinstance(obj, dict):
                return {k: _safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_safe(v) for v in obj]
            return obj

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(_safe(results), f, indent=2)
        print(f"\nAudit written to: {args.output}")


if __name__ == "__main__":
    main()
