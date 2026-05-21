"""Point-teacher feasibility pilot for SALT-RD false-confirmed detection.

This script tests whether point/motion teachers can provide an independent
signal when SGLATrack is false-confirmed.  It deliberately runs on short
windows around false-confirmed frames before any full sidecar work.

Methods
-------
``cotracker3``
    Real CoTracker3 via official torch.hub.  This is an offline teacher only.

``lk``
    OpenCV pyramidal Lucas-Kanade point tracking.  Cheap local baseline.

``farneback``
    Dense Farneback optical-flow point propagation.  This is a lightweight
    stand-in for "does flow/cycle-consistency look promising before installing
    RAFT/GMFlow".

Important
---------
Windows are seeded from the GT bbox at the first frame of the window.  This is
an upper-bound teacher capability test, not a runtime feature extractor.  If a
method fails even with this oracle seed, it is not worth building a full
runtime/OOF sidecar around it.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from salt_r.collect_features import xywh_to_xyxy
from salt_r.dino_identity_pilot import _average_precision, _roc_auc
from salt_r.teachers.cotracker3_export import sample_query_points
from salt_r.teachers.point_features import POINT_FEATURE_NAMES, compute_point_features_sequence


_DEFAULT_DIAGNOSTIC = [
    "dtb70/Gull2",
    "dtb70/Sheep1",
    "dtb70/StreetBasketball1",
    "uav123/bike2",
]

_FEATURE_RISK_SIGN = {
    "pt_visible_ratio": -1.0,
    "pt_inside_pred_ratio": -1.0,
    "pt_inside_pred_weighted": -1.0,
    "pt_forward_backward_error": 1.0,
    "pt_median_motion": 1.0,
    "pt_motion_iqr": 1.0,
    "pt_affine_residual": 1.0,
    "pt_cluster_area_ratio": 1.0,
    "pt_cluster_aspect_delta": 1.0,
    "pt_flow_agreement": -1.0,
    "pt_bbox_center_disagreement": 1.0,
    "pt_survival_since_init": -1.0,
    "pt_split_score": -1.0,
}


@dataclass(frozen=True)
class Window:
    seq_key: str
    start: int
    end: int
    center: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_paths() -> None:
    root = _repo_root()
    for extra in (root / "src", root / "saltr" / "src"):
        s = str(extra)
        if s not in sys.path:
            sys.path.insert(0, s)


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


def _select_windows(
    labels: np.ndarray,
    fc_idx: int,
    seq_key: str,
    window_radius: int,
    max_windows: int,
    min_gap: int,
) -> list[Window]:
    fc = np.where(labels[:, fc_idx] > 0.5)[0]
    if len(fc) == 0:
        return []

    centers: list[int] = []
    for t in fc:
        ti = int(t)
        if not centers or ti - centers[-1] >= min_gap:
            centers.append(ti)
        if len(centers) >= max_windows:
            break

    n = len(labels)
    windows = []
    for c in centers:
        start = max(0, c - window_radius)
        end = min(n, c + window_radius + 1)
        windows.append(Window(seq_key=seq_key, start=start, end=end, center=c))
    return windows


def _resize_window(frames: list[np.ndarray], max_side: int) -> tuple[np.ndarray, float]:
    h, w = frames[0].shape[:2]
    scale = float(max_side) / float(max(h, w))
    out_w, out_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (out_w, out_h), interpolation=cv2.INTER_AREA)
        for f in frames
    ]
    return np.stack(resized, axis=0), scale


def _fill_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    if finite.any():
        arr[~finite] = float(np.nanmedian(arr[finite]))
    else:
        arr[:] = 0.0
    return arr


def _sample_flow_at_points(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    sampled = np.zeros_like(points, dtype=np.float32)
    for i, (x, y) in enumerate(points):
        if not np.isfinite(x) or not np.isfinite(y):
            sampled[i] = np.nan
            continue
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        sampled[i] = flow[yi, xi]
    return sampled


def _track_lk(video_rgb: np.ndarray, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, _, _, _ = video_rgb.shape
    p = len(query_points)
    tracks = np.full((t, p, 2), np.nan, dtype=np.float32)
    visibility = np.zeros((t, p), dtype=bool)
    tracks[0] = query_points.astype(np.float32)
    visibility[0] = True

    prev_gray = cv2.cvtColor(video_rgb[0], cv2.COLOR_RGB2GRAY)
    prev_pts = query_points.astype(np.float32).reshape(-1, 1, 2)
    alive = np.ones(p, dtype=bool)
    for i in range(1, t):
        cur_gray = cv2.cvtColor(video_rgb[i], cv2.COLOR_RGB2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            cur_gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            break
        st = status.reshape(-1).astype(bool) & alive
        h, w = cur_gray.shape[:2]
        pts = next_pts.reshape(-1, 2)
        in_bounds = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        st &= in_bounds
        tracks[i, st] = pts[st]
        visibility[i, st] = True
        alive = st
        prev_pts = pts.astype(np.float32).reshape(-1, 1, 2)
        prev_gray = cur_gray
    return tracks, visibility


def _track_farneback(video_rgb: np.ndarray, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, _, _, _ = video_rgb.shape
    p = len(query_points)
    tracks = np.full((t, p, 2), np.nan, dtype=np.float32)
    visibility = np.zeros((t, p), dtype=bool)
    tracks[0] = query_points.astype(np.float32)
    visibility[0] = True

    prev_gray = cv2.cvtColor(video_rgb[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    pts = query_points.astype(np.float32).copy()
    alive = np.ones(p, dtype=bool)
    for i in range(1, t):
        cur_gray = cv2.cvtColor(video_rgb[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            cur_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        disp = _sample_flow_at_points(flow, pts)
        pts = pts + disp
        h, w = cur_gray.shape[:2]
        in_bounds = (
            np.isfinite(pts).all(axis=1)
            & (pts[:, 0] >= 0)
            & (pts[:, 0] < w)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < h)
        )
        alive &= in_bounds
        tracks[i, alive] = pts[alive]
        visibility[i, alive] = True
        prev_gray = cur_gray
    return tracks, visibility


def _load_cotracker3(device: str):
    import torch

    model = torch.hub.load(
        "facebookresearch/co-tracker",
        "cotracker3_offline",
        trust_repo=True,
    )
    model = model.to(device).eval()
    return model


def _track_cotracker3(
    model: object,
    video_rgb: np.ndarray,
    query_points: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    video_t = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
    queries = torch.zeros(1, len(query_points), 3, device=device)
    queries[0, :, 0] = 0
    queries[0, :, 1:] = torch.from_numpy(query_points).float().to(device)

    with torch.no_grad():
        tracks, vis = model(video_t, queries=queries)
    tracks_np = tracks[0].detach().cpu().numpy().astype(np.float32)
    vis_np = vis[0].detach().cpu().numpy().astype(bool)
    tracks_np[~vis_np] = np.nan
    return tracks_np, vis_np


def _metrics_for_features(labels: np.ndarray, features: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in POINT_FEATURE_NAMES:
        idx = POINT_FEATURE_NAMES.index(name)
        raw = features[:, idx]
        base = _fill_nan(raw)
        signed = base * _FEATURE_RISK_SIGN.get(name, 1.0)
        inv = -signed
        auroc = _roc_auc(labels, signed)
        inv_auroc = _roc_auc(labels, inv)
        result[name] = {
            "auroc": auroc,
            "auprc": _average_precision(labels, signed),
            "auroc_inverted": inv_auroc,
            "best_auroc_any_sign": max(auroc, inv_auroc)
            if np.isfinite(auroc) and np.isfinite(inv_auroc)
            else float("nan"),
            "mean_correct": float(np.nanmean(raw[labels < 0.5])) if np.any(labels < 0.5) else None,
            "mean_fc": float(np.nanmean(raw[labels > 0.5])) if np.any(labels > 0.5) else None,
        }
    return result


def _process_window(
    method: str,
    model: object | None,
    device: str,
    frames: list[np.ndarray],
    gt_xyxy: np.ndarray,
    pred_xyxy: np.ndarray,
    labels: np.ndarray,
    window: Window,
    max_side: int,
) -> dict[str, Any]:
    window_frames = frames[window.start:window.end]
    video_rgb, scale = _resize_window(window_frames, max_side=max_side)
    gt_s = gt_xyxy[window.start:window.end].copy().astype(np.float32) * scale
    pred_s = pred_xyxy[window.start:window.end].copy().astype(np.float32) * scale
    query = sample_query_points(gt_s[0])

    start = time.time()
    if method == "cotracker3":
        if model is None:
            raise ValueError("cotracker3 method requires loaded model")
        tracks, visibility = _track_cotracker3(model, video_rgb, query, device=device)
    elif method == "lk":
        tracks, visibility = _track_lk(video_rgb, query)
    elif method == "farneback":
        tracks, visibility = _track_farneback(video_rgb, query)
    else:
        raise ValueError(f"Unknown method: {method}")
    elapsed = time.time() - start

    feats = compute_point_features_sequence(tracks, visibility, pred_s)
    y = labels[window.start:window.end].astype(np.float32)
    return {
        "window": {
            "seq_key": window.seq_key,
            "start": int(window.start),
            "end": int(window.end),
            "center": int(window.center),
        },
        "scale": float(scale),
        "n_points": int(len(query)),
        "elapsed_sec": float(elapsed),
        "n_frames": int(len(y)),
        "n_fc": int(y.sum()),
        "features": feats,
        "labels": y,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_paths()
    data = np.load(args.npz, allow_pickle=True)
    label_names = [str(x) for x in data["label_names"].tolist()]
    fc_idx = label_names.index("false_confirmed")

    seq_keys = args.seqs or list(_DEFAULT_DIAGNOSTIC)
    dataset_cache: dict[str, dict[str, object]] = {}
    method_models: dict[str, object | None] = {}
    if "cotracker3" in args.methods:
        method_models["cotracker3"] = _load_cotracker3(args.device)

    method_outputs: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        print(f"[point_teacher_pilot] method={method}", flush=True)
        windows_out = []
        all_features = []
        all_labels = []
        per_seq: dict[str, dict[str, Any]] = {}
        for seq_key in seq_keys:
            dataset, name = seq_key.split("/", 1)
            dataset_cache.setdefault(dataset, _get_dataset_sequences(dataset))
            seq_obj = dataset_cache[dataset][name]
            frames = list(seq_obj.frames)
            labels = data[f"labels/{seq_key}"][:, fc_idx].astype(np.float32)
            gt_xyxy = xywh_to_xyxy(data[f"bbox_gt/{seq_key}"].astype(np.float32))
            pred_xyxy = xywh_to_xyxy(data[f"bbox_pred/{seq_key}"].astype(np.float32))
            windows = _select_windows(
                data[f"labels/{seq_key}"],
                fc_idx=fc_idx,
                seq_key=seq_key,
                window_radius=args.window_radius,
                max_windows=args.max_windows_per_seq,
                min_gap=args.min_window_gap,
            )
            print(f"  {seq_key}: {len(windows)} windows", flush=True)
            seq_features = []
            seq_labels = []
            for window in windows:
                result = _process_window(
                    method=method,
                    model=method_models.get(method),
                    device=args.device,
                    frames=frames,
                    gt_xyxy=gt_xyxy,
                    pred_xyxy=pred_xyxy,
                    labels=labels,
                    window=window,
                    max_side=args.max_side,
                )
                feats = result.pop("features")
                y = result.pop("labels")
                result["feature_metrics"] = _metrics_for_features(y, feats)
                windows_out.append(result)
                all_features.append(feats)
                all_labels.append(y)
                seq_features.append(feats)
                seq_labels.append(y)
                print(
                    f"    {window.start}:{window.end} "
                    f"frames={len(y)} fc={int(y.sum())} "
                    f"sec={result['elapsed_sec']:.2f}",
                    flush=True,
                )

            if seq_features:
                sf = np.concatenate(seq_features, axis=0)
                sy = np.concatenate(seq_labels, axis=0)
                per_seq[seq_key] = {
                    "n_frames": int(len(sy)),
                    "n_fc": int(sy.sum()),
                    "fc_rate": float(sy.mean()),
                    "feature_metrics": _metrics_for_features(sy, sf),
                }

        if all_features:
            feats_all = np.concatenate(all_features, axis=0)
            labels_all = np.concatenate(all_labels, axis=0)
            overall = {
                "n_frames": int(len(labels_all)),
                "n_fc": int(labels_all.sum()),
                "fc_rate": float(labels_all.mean()),
                "feature_metrics": _metrics_for_features(labels_all, feats_all),
            }
        else:
            overall = {"n_frames": 0, "n_fc": 0, "fc_rate": None, "feature_metrics": {}}

        method_outputs[method] = {
            "overall": overall,
            "per_sequence": per_seq,
            "windows": windows_out,
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "npz": args.npz,
        "methods": args.methods,
        "seqs": seq_keys,
        "window_radius": args.window_radius,
        "max_windows_per_seq": args.max_windows_per_seq,
        "max_side": args.max_side,
        "seed_type": "gt_bbox_at_window_start",
        "note": (
            "GT window seeding is an upper-bound teacher feasibility test, "
            "not a runtime feature extractor."
        ),
        "outputs": method_outputs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default="saltr/data/salt_rd_v2_labels.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", default=["lk", "farneback", "cotracker3"],
                        choices=["lk", "farneback", "cotracker3"])
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--window-radius", type=int, default=15)
    parser.add_argument("--max-windows-per-seq", type=int, default=3)
    parser.add_argument("--min-window-gap", type=int, default=30)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    result = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[point_teacher_pilot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
