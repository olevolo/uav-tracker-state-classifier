"""Offline memory feature collection from existing NPZ.

Processes a v2 NPZ and produces a memory sidecar NPZ with
per-frame DAM-style memory features.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .memory import DistractorAwareMemory

MEMORY_FEATURE_NAMES: List[str] = DistractorAwareMemory.FEATURE_NAMES


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_proxy_embedding(features_row: np.ndarray) -> np.ndarray:
    """Normalize a 28-dim feature vector as proxy embedding.

    This is a fallback for offline mode where SGLATrack crop embeddings
    are not available. The memory's accumulation logic remains valid;
    real DINO embeddings can be plugged in by replacing this function.
    """
    norm = np.linalg.norm(features_row) + 1e-8
    return (features_row / norm).astype(np.float32)


def compute_memory_features_for_sequence(
    features: np.ndarray,          # (T, 28) scalar features
    labels: np.ndarray,            # (T, 14) labels
    iou_trace: np.ndarray,         # (T,) GT IoU
    preds: Optional[np.ndarray],   # (T, n_heads) calibrated probabilities (optional)
    label_names: List[str],
    head_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Returns (T, 9) memory feature array.

    Parameters
    ----------
    features:
        Per-frame scalar feature matrix, shape (T, 28).
    labels:
        Per-frame label matrix, shape (T, 14).
    iou_trace:
        Ground-truth IoU per frame, shape (T,).
    preds:
        Optional per-frame model predictions (calibrated), shape (T, n_heads).
        When provided, p_fc and p_ifd are read from the model output.
        When None, labels are used as oracle signals.
    label_names:
        Ordered list of label names matching the columns of `labels`.
    head_names:
        Ordered list of head names matching the columns of `preds`.
        Required when `preds` is not None.

    Returns
    -------
    np.ndarray of shape (T, 9), float32.
    """
    T = features.shape[0]
    n_mem_features = len(MEMORY_FEATURE_NAMES)
    result = np.zeros((T, n_mem_features), dtype=np.float32)

    # Resolve column indices for signals needed by memory gates
    fc_label_idx = label_names.index("false_confirmed") if "false_confirmed" in label_names else -1
    ifd_label_idx = (
        label_names.index("imminent_failure_dynamic")
        if "imminent_failure_dynamic" in label_names else -1
    )
    apce_norm_feat_idx = 1  # apce_norm is always feature index 1 (FEATURE_NAMES)
    n_secondary_feat_idx = 6  # n_secondary is always feature index 6

    # Resolve prediction column indices (if preds provided)
    fc_pred_idx: int = -1
    ifd_pred_idx: int = -1
    if preds is not None and head_names is not None:
        if "false_confirmed" in head_names:
            fc_pred_idx = head_names.index("false_confirmed")
        if "imminent_failure_dynamic" in head_names:
            ifd_pred_idx = head_names.index("imminent_failure_dynamic")

    mem = DistractorAwareMemory()

    for t in range(T):
        feat_t = features[t]

        # Build proxy embedding from scalar features
        embedding = _make_proxy_embedding(feat_t)

        # Read apce_norm from features
        apce_norm = float(feat_t[apce_norm_feat_idx]) if features.shape[1] > apce_norm_feat_idx else 0.0

        # Distractor signal: prefer p_fc proxy when preds available (n_secondary=0 in SGLATrack).
        # When p_fc is high, the tracker is likely on a distractor — treat as distractor candidate.
        # When preds not available, fall back to n_secondary from telemetry (usually 0).
        if preds is not None and fc_pred_idx >= 0 and fc_pred_idx < preds.shape[1]:
            # p_fc > 0.4 → strong distractor signal proxy; smooth via clip
            secondary_peak_ratio = float(np.clip(float(preds[t, fc_pred_idx]) / 0.4, 0.0, 1.0))
        else:
            n_secondary = float(feat_t[n_secondary_feat_idx]) if features.shape[1] > n_secondary_feat_idx else 0.0
            secondary_peak_ratio = float(np.clip(n_secondary / 3.0, 0.0, 1.0))

        # p_fc: use model prediction if available, else oracle from label
        if preds is not None and fc_pred_idx >= 0 and fc_pred_idx < preds.shape[1]:
            p_fc = float(preds[t, fc_pred_idx])
        elif fc_label_idx >= 0:
            p_fc = float(labels[t, fc_label_idx])
        else:
            p_fc = 0.0

        # p_ifd: use model prediction if available, else oracle from label
        if preds is not None and ifd_pred_idx >= 0 and ifd_pred_idx < preds.shape[1]:
            p_ifd = float(preds[t, ifd_pred_idx])
        elif ifd_label_idx >= 0:
            p_ifd = float(labels[t, ifd_label_idx])
        else:
            p_ifd = 0.0

        iou_t = float(iou_trace[t])

        # Compute memory features BEFORE updating memory with this frame
        # → features at t reflect only frames 0..t-1 (causal, no same-frame leakage)
        feat_dict = mem.compute_features(query_emb=embedding, query_bbox=None)

        for fi, fname in enumerate(MEMORY_FEATURE_NAMES):
            result[t, fi] = float(feat_dict[fname])

        # Now update memory with frame t's information
        mem.step(
            frame_idx=t,
            embedding=embedding,
            p_fc=p_fc,
            p_ifd=p_ifd,
            apce_norm=apce_norm,
            secondary_peak_ratio=secondary_peak_ratio,
            iou=iou_t,
            bbox=None,
        )

    return result


def collect_memory_sidecar(
    npz_v2_path: str,
    preds_json_path: Optional[str],
    output_path: str,
) -> None:
    """Process v2 NPZ and write memory sidecar NPZ.

    Output NPZ:
      memory_features/{seq}: float32 (T, 9)
      memory_feature_names: list[str]
      created_at: str
      source_npz_md5: str

    Parameters
    ----------
    npz_v2_path:
        Path to the v2 labels NPZ produced by collect_features.py.
    preds_json_path:
        Optional path to per-sequence per-frame model predictions JSON
        (as written by eval.py --predictions-output). When None, oracle
        labels are used as gate signals.
    output_path:
        Destination path for the memory sidecar NPZ.
    """
    import json as _json

    data = np.load(npz_v2_path, allow_pickle=True)

    try:
        label_names: List[str] = list(data["label_names"].tolist())
    except Exception:
        from .collect_features import LABEL_NAMES_V2
        label_names = list(LABEL_NAMES_V2)

    try:
        feature_names: List[str] = list(data["feature_names"].tolist())
    except Exception:
        from .collect_features import FEATURE_NAMES
        feature_names = list(FEATURE_NAMES)

    # Load predictions if provided
    preds_by_seq: dict[str, np.ndarray] = {}
    head_names: Optional[List[str]] = None
    if preds_json_path is not None:
        raw_preds = _json.loads(Path(preds_json_path).read_text())
        # raw_preds: {seq_key: [{head: prob, ...}, ...]}
        for seq_key, frames in raw_preds.items():
            if not frames:
                continue
            head_names_local = list(frames[0].keys())
            if head_names is None:
                head_names = head_names_local
            arr = np.array(
                [[f[h] for h in head_names_local] for f in frames],
                dtype=np.float32,
            )
            preds_by_seq[seq_key] = arr

    compound_keys = [
        k[len("features/"):] for k in data.files if k.startswith("features/")
    ]

    out: dict[str, object] = {}
    print(f"Processing {len(compound_keys)} sequences...")

    for seq_key in compound_keys:
        features = data[f"features/{seq_key}"].astype(np.float32)
        labels = data[f"labels/{seq_key}"].astype(np.float32)
        iou_trace = data[f"iou_trace/{seq_key}"].astype(np.float32)

        preds = preds_by_seq.get(seq_key, None)

        mem_feats = compute_memory_features_for_sequence(
            features=features,
            labels=labels,
            iou_trace=iou_trace,
            preds=preds,
            label_names=label_names,
            head_names=head_names,
        )
        out[f"memory_features/{seq_key}"] = mem_feats.astype(np.float32)
        MARGIN_COL = MEMORY_FEATURE_NAMES.index("mem_target_minus_distractor_margin")
        out[f"memory_margin/{seq_key}"] = mem_feats[:, MARGIN_COL].astype(np.float32)

    out["memory_feature_names"] = np.array(MEMORY_FEATURE_NAMES, dtype=object)
    out["created_at"] = np.array(datetime.now(tz=timezone.utc).isoformat())
    out["source_npz_md5"] = np.array(_md5_file(npz_v2_path))
    out["uses_oracle_labels"] = np.array(preds_json_path is None)
    out["distractor_source"] = np.array(
        "fc_proxy_t0.4" if preds_json_path is not None else "oracle_fc_label"
    )
    if preds_json_path is not None:
        out["source_preds_md5"] = np.array(_md5_file(preds_json_path))
    else:
        out["source_preds_md5"] = np.array("oracle")
    out["source_split"] = np.array("val")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    print(f"Memory sidecar written to: {output_path}  ({len(compound_keys)} sequences)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> "argparse.Namespace":
    p = argparse.ArgumentParser(
        description="Collect DAM-style memory sidecar from v2 NPZ."
    )
    p.add_argument("--npz", required=True,
                   help="Path to salt_rd_v2_labels.npz")
    p.add_argument("--preds", default=None,
                   help="Optional path to preds_val_*.json (calibrated model predictions). "
                        "When omitted, oracle labels are used as gate signals.")
    p.add_argument("--output", required=True,
                   help="Output path for memory sidecar NPZ, e.g. saltr/data/salt_rd_memory_sidecar.npz")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[memory_features] NPZ:    {args.npz}", flush=True)
    print(f"[memory_features] Preds:  {args.preds or '(oracle labels)'}", flush=True)
    print(f"[memory_features] Output: {args.output}", flush=True)
    collect_memory_sidecar(
        npz_v2_path=args.npz,
        preds_json_path=args.preds,
        output_path=args.output,
    )
    print("[memory_features] Done.", flush=True)


if __name__ == "__main__":
    main()
