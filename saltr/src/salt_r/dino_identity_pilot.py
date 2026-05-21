"""DINOv2 ROI identity pilot for SALT-RD false-confirmed detection.

This is deliberately a *pilot* tool, not the final sidecar builder.

Purpose
-------
SGLATrack backbone tokens are localization-oriented and failed as an identity
representation for false-confirmed frames.  This script checks whether frozen
DINOv2 crop embeddings provide an actual identity signal before we spend hours
building a full 228-sequence sidecar.

Implementation choices
----------------------
- Model: official Meta DINOv2 Torch Hub backbone, default ``dinov2_vits14``.
- Input: predicted tracker bbox crop, with context, from ``bbox_pred/{seq}``.
- Init anchor: frame-0 GT bbox crop from ``bbox_gt/{seq}``.
- Transform: square ROI crop, resize to 224×224 with bicubic interpolation,
  ImageNet mean/std normalization.  224 is divisible by DINOv2's patch size 14.
- Readout: CLS token by default, or mean of normalized patch tokens via
  ``--embedding-mode patch_mean`` for dense/object-centric sanity checks.
- Metrics: single-feature AUROC/AUPRC for false_confirmed.  Primary pilot gate
  is ``1 - dino_init_sim``.

Example
-------
python -m salt_r.dino_identity_pilot \\
  --npz saltr/data/salt_rd_v2_labels.npz \\
  --preds saltr/results/preds_all_v2_oof_teacher.json \\
  --output saltr/results/dino_identity_pilot.json \\
  --device mps \\
  --max-val-seqs 8 \\
  --frame-stride 5
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_DIAGNOSTIC_KEYS = {
    "uav123/bike2",
    "visdrone_sot/uav0000164",
    "dtb70/Gull2",
    "dtb70/Sheep1",
    "dtb70/StreetBasketball1",
}

_FEATURE_NAMES = [
    "dino_init_sim",
    "dino_mem_max_sim",
    "dino_mem_mean_sim",
    "dino_delta_prev",
    "dino_update_age_norm",
]

_RISK_SCORE_BY_FEATURE = {
    "1_minus_dino_init_sim": "dino_init_sim",
    "1_minus_dino_mem_max_sim": "dino_mem_max_sim",
    "1_minus_dino_mem_mean_sim": "dino_mem_mean_sim",
    "dino_delta_prev": "dino_delta_prev",
    "dino_update_age_norm": "dino_update_age_norm",
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PilotSequence:
    key: str
    dataset: str
    name: str
    split: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_paths() -> None:
    root = _repo_root()
    for extra in (root / "src", root / "saltr" / "src"):
        s = str(extra)
        if s not in sys.path:
            sys.path.insert(0, s)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / max(n_pos, 1)
    fpr = fps / max(n_neg, 1)
    return float(np.trapz(tpr, fpr))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    recall = np.cumsum(y_sorted) / n_pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.sum(np.diff(recall) * precision[1:]))


def _safe_metric_table(
    labels: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "n_frames": int(len(labels)),
        "n_positive": int(labels.sum()),
        "base_rate": float(labels.mean()) if len(labels) else float("nan"),
        "features": {},
    }
    if len(labels) == 0:
        return results

    for risk_name, source_name in _RISK_SCORE_BY_FEATURE.items():
        if source_name not in feature_names:
            continue
        idx = feature_names.index(source_name)
        score = features[:, idx].astype(np.float64)
        if risk_name.startswith("1_minus_"):
            score = 1.0 - score
        results["features"][risk_name] = {
            "auroc": _roc_auc(labels, score),
            "auprc": _average_precision(labels, score),
        }
    return results


def _select_pilot_sequences(
    data: np.lib.npyio.NpzFile,
    max_val_seqs: int,
    explicit_seqs: list[str] | None = None,
) -> list[PilotSequence]:
    keys = sorted(k[len("features/"):] for k in data.files if k.startswith("features/"))
    if explicit_seqs:
        wanted = set(explicit_seqs)
        keys = [k for k in keys if k in wanted]
        missing = sorted(wanted - set(keys))
        if missing:
            raise ValueError(f"Requested pilot seqs missing from NPZ: {missing}")

    selected: list[PilotSequence] = []
    for key in keys:
        split = str(data[f"split/{key}"])
        dataset = str(data[f"dataset/{key}"]) if f"dataset/{key}" in data.files else key.split("/", 1)[0]
        name = str(data[f"sequence_name/{key}"]) if f"sequence_name/{key}" in data.files else key.split("/", 1)[-1]
        if explicit_seqs or split == "diagnostic" or key in _DIAGNOSTIC_KEYS:
            selected.append(PilotSequence(key, dataset, name, split))

    if not explicit_seqs and max_val_seqs > 0:
        val_by_dataset: dict[str, list[PilotSequence]] = collections.defaultdict(list)
        for key in keys:
            if str(data[f"split/{key}"]) != "val":
                continue
            dataset = str(data[f"dataset/{key}"]) if f"dataset/{key}" in data.files else key.split("/", 1)[0]
            name = str(data[f"sequence_name/{key}"]) if f"sequence_name/{key}" in data.files else key.split("/", 1)[-1]
            val_by_dataset[dataset].append(PilotSequence(key, dataset, name, "val"))

        # Dataset-balanced round-robin so UAV123 doesn't silently dominate.
        datasets = sorted(val_by_dataset)
        cursor = {ds: 0 for ds in datasets}
        while len([s for s in selected if s.split == "val"]) < max_val_seqs:
            made_progress = False
            for ds in datasets:
                if len([s for s in selected if s.split == "val"]) >= max_val_seqs:
                    break
                idx = cursor[ds]
                if idx < len(val_by_dataset[ds]):
                    selected.append(val_by_dataset[ds][idx])
                    cursor[ds] += 1
                    made_progress = True
            if not made_progress:
                break

    # Keep deterministic order: diagnostic first, then val.
    selected.sort(key=lambda s: (0 if s.split == "diagnostic" else 1, s.dataset, s.name))
    return selected


def _get_dataset_sequences(dataset_name: str) -> dict[str, object]:
    _ensure_paths()
    if dataset_name == "uav123":
        from uav_tracker.datasets.uav123 import UAV123Dataset
        return {seq.name: seq for seq in UAV123Dataset(root=None)}
    if dataset_name == "visdrone_sot":
        from uav_tracker.datasets.visdrone_sot import VisDroneSOTDataset
        return {seq.name: seq for seq in VisDroneSOTDataset(root=None)}
    if dataset_name == "dtb70":
        from uav_tracker.datasets.dtb70 import DTB70Dataset
        return {seq.name: seq for seq in DTB70Dataset(root=None)}
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _square_context_crop_bgr(
    frame_bgr: np.ndarray,
    bbox_xywh: np.ndarray,
    context_scale: float,
    min_side: int = 16,
) -> np.ndarray:
    """Return square BGR crop around bbox, padding outside frame with black."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR HxWx3 frame, got shape={frame_bgr.shape}")

    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = [float(v) for v in bbox_xywh[:4]]
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        # Fallback: centered minimal crop to keep feature extraction finite.
        x, y, w, h = w_img / 2 - min_side / 2, h_img / 2 - min_side / 2, min_side, min_side

    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(float(min_side), max(w, h) * float(context_scale))
    side_i = max(min_side, int(round(side)))

    x1 = int(math.floor(cx - side_i / 2))
    y1 = int(math.floor(cy - side_i / 2))
    x2 = x1 + side_i
    y2 = y1 + side_i

    crop = np.zeros((side_i, side_i, 3), dtype=frame_bgr.dtype)
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w_img, x2)
    src_y2 = min(h_img, y2)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return crop

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    crop[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = (
        frame_bgr[src_y1:src_y2, src_x1:src_x2]
    )
    return crop


def _preprocess_crop_bgr(crop_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """DINOv2 eval-style preprocessing, returned as CHW float32."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def _load_dinov2_model(model_name: str, device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model = model.to(device)
    model.eval()
    return model, device


def _embed_crops(
    model: object,
    crops_chw: list[np.ndarray],
    device: str,
    batch_size: int,
    embedding_mode: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(crops_chw), batch_size):
            batch = np.stack(crops_chw[start:start + batch_size], axis=0)
            x = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
            if embedding_mode != "cls" and hasattr(model, "forward_features"):
                y = model.forward_features(x)
            else:
                y = model(x)
            if isinstance(y, dict):
                if embedding_mode == "patch_mean" and "x_norm_patchtokens" in y:
                    y = y["x_norm_patchtokens"].mean(dim=1)
                elif "x_norm_clstoken" in y:
                    y = y["x_norm_clstoken"]
                elif "x_norm_patchtokens" in y:
                    y = y["x_norm_patchtokens"].mean(dim=1)
                else:
                    first = next(iter(y.values()))
                    y = first.mean(dim=1) if getattr(first, "ndim", 0) == 3 else first
            y = F.normalize(y.float(), dim=-1)
            outputs.append(y.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def _choose_frame_indices(
    labels: np.ndarray,
    fc_idx: int,
    frame_stride: int,
    include_all_fc: bool,
    max_frames_per_seq: int,
) -> np.ndarray:
    T = labels.shape[0]
    selected = set(range(0, T, max(1, frame_stride)))
    selected.add(0)
    if include_all_fc and fc_idx >= 0:
        selected.update(np.flatnonzero(labels[:, fc_idx] > 0.5).tolist())
    ordered = np.array(sorted(i for i in selected if 0 <= i < T), dtype=np.int64)
    if max_frames_per_seq > 0 and len(ordered) > max_frames_per_seq:
        positive = ordered[labels[ordered, fc_idx] > 0.5] if fc_idx >= 0 else np.array([], dtype=np.int64)
        remaining = ordered[labels[ordered, fc_idx] <= 0.5] if fc_idx >= 0 else ordered
        budget = max(0, max_frames_per_seq - len(positive))
        if budget < len(remaining):
            take = np.linspace(0, len(remaining) - 1, budget, dtype=np.int64) if budget else np.array([], dtype=np.int64)
            remaining = remaining[take]
        ordered = np.array(sorted(set(positive.tolist()) | set(remaining.tolist())), dtype=np.int64)
    return ordered


def _risk_gate(preds_for_seq: list[dict[str, float]] | None, t: int) -> tuple[float, float]:
    if preds_for_seq is None or t >= len(preds_for_seq):
        return 1.0, 1.0
    frame = preds_for_seq[t]
    p_fc = float(frame.get("false_confirmed", 1.0))
    p_ifd = max(
        float(frame.get("imminent_failure_dynamic", 1.0)),
        float(frame.get("imminent_failure_dynamic_10", 1.0)),
        float(frame.get("imminent_failure_dynamic_20", 1.0)),
    )
    return p_fc, p_ifd


def _extract_one_sequence(
    data: np.lib.npyio.NpzFile,
    seq: PilotSequence,
    model: object,
    device: str,
    preds_raw: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = data[f"labels/{seq.key}"].astype(np.float32)
    label_names = list(data["label_names"].tolist())
    if "false_confirmed" not in label_names:
        raise ValueError("false_confirmed label missing from NPZ")
    fc_idx = label_names.index("false_confirmed")

    bboxes_pred = data[f"bbox_pred/{seq.key}"].astype(np.float32)
    bboxes_gt = data[f"bbox_gt/{seq.key}"].astype(np.float32)
    frame_indices = _choose_frame_indices(
        labels=labels,
        fc_idx=fc_idx,
        frame_stride=args.frame_stride,
        include_all_fc=not args.no_include_all_fc,
        max_frames_per_seq=args.max_frames_per_seq,
    )

    dataset_seqs = _get_dataset_sequences(seq.dataset)
    if seq.name not in dataset_seqs:
        raise ValueError(f"{seq.key}: sequence not found in dataset loader")
    frames = list(dataset_seqs[seq.name].frames)
    if len(frames) != labels.shape[0]:
        raise ValueError(f"{seq.key}: dataset frames={len(frames)} != NPZ labels={labels.shape[0]}")

    # Init anchor uses trusted frame-0 GT crop, not tracker prediction.
    init_crop = _square_context_crop_bgr(frames[0], bboxes_gt[0], args.context_scale)
    crops = [_preprocess_crop_bgr(init_crop, args.image_size)]
    for t in frame_indices:
        crop = _square_context_crop_bgr(frames[int(t)], bboxes_pred[int(t)], args.context_scale)
        crops.append(_preprocess_crop_bgr(crop, args.image_size))

    embeddings = _embed_crops(
        model,
        crops,
        device=device,
        batch_size=args.batch_size,
        embedding_mode=args.embedding_mode,
    )
    init_emb = embeddings[0]
    curr = embeddings[1:]

    preds_for_seq = preds_raw.get(seq.key)
    memory: list[tuple[int, np.ndarray]] = [(0, init_emb.copy())]
    last_update = 0
    prev_emb: np.ndarray | None = None
    features = np.zeros((len(frame_indices), len(_FEATURE_NAMES)), dtype=np.float32)

    for row, (t_raw, emb) in enumerate(zip(frame_indices, curr)):
        t = int(t_raw)
        mem_embs = np.stack([m[1] for m in memory], axis=0)
        mem_sims = mem_embs @ emb
        init_sim = float(init_emb @ emb)
        delta_prev = 0.0 if prev_emb is None else float(1.0 - np.clip(prev_emb @ emb, -1.0, 1.0))
        age_norm = min(float(t - last_update), float(args.max_age_norm_frames)) / float(args.max_age_norm_frames)

        features[row] = np.array([
            init_sim,
            float(mem_sims.max()),
            float(mem_sims.mean()),
            delta_prev,
            age_norm,
        ], dtype=np.float32)

        p_fc, p_ifd = _risk_gate(preds_for_seq, t)
        if (
            p_fc < args.update_fc_threshold
            and p_ifd < args.update_ifd_threshold
            and (t - last_update) >= args.update_interval
        ):
            memory.append((t, emb.copy()))
            if len(memory) > args.memory_slots:
                memory = memory[-args.memory_slots:]
            last_update = t

        prev_emb = emb

    y = labels[frame_indices, fc_idx].astype(np.float32)
    return {
        "seq_key": seq.key,
        "dataset": seq.dataset,
        "split": seq.split,
        "frame_indices": frame_indices.tolist(),
        "labels_false_confirmed": y.tolist(),
        "features": features.tolist(),
        "metrics": _safe_metric_table(y, features, list(_FEATURE_NAMES)),
        "n_memory_updates": int(max(0, len(memory) - 1)),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "overall": records,
    }
    by_split: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_split[str(r["split"])].append(r)
        by_dataset[str(r["dataset"])].append(r)
    for key, val in by_split.items():
        groups[f"split:{key}"] = val
    for key, val in by_dataset.items():
        groups[f"dataset:{key}"] = val

    out: dict[str, Any] = {}
    for name, recs in groups.items():
        if not recs:
            continue
        labels = np.concatenate([np.asarray(r["labels_false_confirmed"], dtype=np.float32) for r in recs])
        feats = np.concatenate([np.asarray(r["features"], dtype=np.float32) for r in recs], axis=0)
        out[name] = _safe_metric_table(labels, feats, list(_FEATURE_NAMES))
        out[name]["n_sequences"] = len(recs)
    return out


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_paths()
    data = np.load(args.npz, allow_pickle=True)
    selected = _select_pilot_sequences(
        data=data,
        max_val_seqs=args.max_val_seqs,
        explicit_seqs=args.seqs,
    )
    if not selected:
        raise ValueError("No pilot sequences selected")

    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "npz_path": str(args.npz),
        "preds_path": str(args.preds) if args.preds else None,
        "model_name": args.model_name,
        "embedding_mode": args.embedding_mode,
        "image_size": args.image_size,
        "context_scale": args.context_scale,
        "frame_stride": args.frame_stride,
        "max_val_seqs": args.max_val_seqs,
        "feature_names": list(_FEATURE_NAMES),
        "selected_sequences": [s.__dict__ for s in selected],
        "official_dinov2_notes": {
            "source": "facebookresearch/dinov2 Torch Hub",
            "default_model": "dinov2_vits14",
            "normalization": "ImageNet mean/std",
            "patch_size_note": "224 is divisible by ViT patch size 14",
        },
    }

    if args.dry_run:
        result["dry_run"] = True
        return result

    preds_raw: dict[str, Any] = {}
    if args.preds:
        preds_raw = json.loads(Path(args.preds).read_text())

    model, device = _load_dinov2_model(args.model_name, args.device)
    result["device_resolved"] = device

    records: list[dict[str, Any]] = []
    for idx, seq in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {seq.key} split={seq.split}", flush=True)
        records.append(_extract_one_sequence(data, seq, model, device, preds_raw, args))

    result["sequences"] = records
    result["aggregate"] = _aggregate(records)

    pilot_gate = {
        "diagnostic_min_auroc": 0.65,
        "val_min_auroc": 0.60,
        "feature": "1_minus_dino_init_sim",
    }
    diag = result["aggregate"].get("split:diagnostic", {}).get("features", {}).get(pilot_gate["feature"], {})
    val = result["aggregate"].get("split:val", {}).get("features", {}).get(pilot_gate["feature"], {})
    diag_auroc = float(diag.get("auroc", float("nan")))
    val_auroc = float(val.get("auroc", float("nan")))
    result["pilot_gate"] = {
        **pilot_gate,
        "diagnostic_auroc": diag_auroc,
        "val_auroc": val_auroc,
        "passed": bool(
            math.isfinite(diag_auroc)
            and math.isfinite(val_auroc)
            and diag_auroc >= pilot_gate["diagnostic_min_auroc"]
            and val_auroc >= pilot_gate["val_min_auroc"]
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small DINOv2 ROI identity pilot.")
    parser.add_argument("--npz", required=True, help="Path to salt_rd_v2_labels.npz")
    parser.add_argument("--preds", default=None, help="OOF/teacher predictions JSON for causal memory gates")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--seqs", nargs="*", default=None, help="Explicit compound seq keys to process")
    parser.add_argument("--max-val-seqs", type=int, default=8, help="Dataset-balanced val seq sample size")
    parser.add_argument("--frame-stride", type=int, default=5, help="Sample every N frames, plus all fc positives")
    parser.add_argument("--max-frames-per-seq", type=int, default=0, help="Optional cap per sequence; 0 disables")
    parser.add_argument("--no-include-all-fc", action="store_true", help="Do not force-include all false_confirmed frames")
    parser.add_argument("--model-name", default="dinov2_vits14", help="Torch Hub DINOv2 backbone name")
    parser.add_argument(
        "--embedding-mode",
        default="cls",
        choices=["cls", "patch_mean"],
        help="DINOv2 readout: CLS token or mean of normalized patch tokens.",
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--context-scale", type=float, default=2.0)
    parser.add_argument("--memory-slots", type=int, default=6)
    parser.add_argument("--update-interval", type=int, default=5)
    parser.add_argument("--update-fc-threshold", type=float, default=0.20)
    parser.add_argument("--update-ifd-threshold", type=float, default=0.30)
    parser.add_argument("--max-age-norm-frames", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Only resolve selected sequences; do not load DINO")
    args = parser.parse_args()

    result = run_pilot(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    out_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    print(f"DINO identity pilot written to: {out_path}")
    if "pilot_gate" in result:
        print(f"Pilot gate: {result['pilot_gate']}")


if __name__ == "__main__":
    main()
