"""Point feature sidecar extractor for SALT-RD.

Extracts per-frame LK/Farneback point consistency features for all sequences
in the SALT-RD NPZ.  Uses PRED bbox seeding (causal — no GT oracle).

Strategy: sliding-window re-seeding
  - Every ``stride`` frames a fresh set of query points is seeded from
    pred_bbox[t] and tracked forward for ``window`` frames.
  - Per-frame features come from the most-recently-started window covering
    that frame (latest seed wins).  This is causal: features at t only depend
    on frames ≤ t.

Method: ``lk`` (default) or ``farneback``.  CoTracker3 is excluded — marginal
gain, extreme memory usage (crashes laptop).

Canonical extraction flags
--------------------------
  --smoke-test N   run on N sequences only; partial save is allowed
  --allow-partial  save even when some sequences failed (diagnostic only)

Without either flag the extractor raises RuntimeError if ANY sequence fails,
matching the fail-fast contract of sgla_memory_extractor.py.

Output NPZ schema
-----------------
  point_features/{seq}:          float32 (T, F_pt)
  point_method/{seq}:            str     "lk" | "farneback"
  point_feature_names:           list[str]
  extractor_method:              str
  stride:                        int
  window:                        int
  n_points:                      int
  n_sequences:                   int
  source_npz_md5:                str
  created_at:                    str
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from salt_r.collect_features import xywh_to_xyxy
from salt_r.teachers.cotracker3_export import sample_query_points
from salt_r.teachers.point_features import (
    POINT_FEATURE_NAMES,
    compute_point_features_sequence,
)


# ---------------------------------------------------------------------------
# Lightweight point trackers (no CoTracker3 dependency)
# ---------------------------------------------------------------------------

def _track_lk(
    frames: list[np.ndarray],  # list of BGR uint8
    query_xy: np.ndarray,      # (P, 2) float32
) -> tuple[np.ndarray, np.ndarray]:
    """Pyramidal Lucas-Kanade forward tracking.

    Returns tracks (T, P, 2) and visibility (T, P) bool.
    NaN in tracks where visibility is False.
    """
    T = len(frames)
    P = len(query_xy)
    tracks = np.full((T, P, 2), np.nan, dtype=np.float32)
    vis = np.zeros((T, P), dtype=bool)

    tracks[0] = query_xy.astype(np.float32)
    vis[0] = True

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts = query_xy.astype(np.float32).reshape(-1, 1, 2)
    alive = np.ones(P, dtype=bool)

    for i in range(1, T):
        cur_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
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
        vis[i, st] = True
        alive = st
        prev_pts = pts.astype(np.float32).reshape(-1, 1, 2)
        prev_gray = cur_gray

    return tracks, vis


def _sample_flow(flow: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    out = np.zeros_like(pts, dtype=np.float32)
    for i, (x, y) in enumerate(pts):
        if not np.isfinite(x) or not np.isfinite(y):
            out[i] = np.nan
            continue
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        out[i] = flow[yi, xi]
    return out


def _track_farneback(
    frames: list[np.ndarray],
    query_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense Farneback optical-flow point propagation."""
    T = len(frames)
    P = len(query_xy)
    tracks = np.full((T, P, 2), np.nan, dtype=np.float32)
    vis = np.zeros((T, P), dtype=bool)

    tracks[0] = query_xy.astype(np.float32)
    vis[0] = True

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    pts = query_xy.astype(np.float32).copy()
    alive = np.ones(P, dtype=bool)

    for i in range(1, T):
        cur_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        disp = _sample_flow(flow, pts)
        pts = pts + disp
        h, w = cur_gray.shape[:2]
        in_bounds = (
            np.isfinite(pts).all(axis=1)
            & (pts[:, 0] >= 0) & (pts[:, 0] < w)
            & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        )
        alive &= in_bounds
        tracks[i, alive] = pts[alive]
        vis[i, alive] = True
        prev_gray = cur_gray

    return tracks, vis


_TRACKERS = {"lk": _track_lk, "farneback": _track_farneback}


# ---------------------------------------------------------------------------
# Per-sequence extraction: sliding-window re-seeding
# ---------------------------------------------------------------------------

def extract_sequence_features(
    frames: list[np.ndarray],   # list of BGR uint8 full-res frames
    pred_xyxy: np.ndarray,      # (T, 4) predicted bboxes in [x1,y1,x2,y2]
    method: str = "lk",
    stride: int = 15,
    window: int = 25,
    n_points: int = 9,
    max_side: int = 320,
) -> np.ndarray:
    """Return per-frame point features (T, F_pt) for one sequence.

    Seeding is from PRED bbox (causal).  Latest-seed wins per frame.
    Frames are resized to max_side on longest edge to keep memory low.
    """
    T = len(frames)
    F = len(POINT_FEATURE_NAMES)
    out = np.full((T, F), np.nan, dtype=np.float32)

    if T == 0:
        return out

    tracker_fn = _TRACKERS[method]

    # Resize scale
    h0, w0 = frames[0].shape[:2]
    scale = float(max_side) / float(max(h0, w0))
    out_h = max(1, int(round(h0 * scale)))
    out_w = max(1, int(round(w0 * scale)))

    resized: list[np.ndarray] = [
        cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_AREA) for f in frames
    ]
    pred_s = pred_xyxy.astype(np.float32) * scale  # scaled bboxes

    for seed in range(0, T, stride):
        end = min(T, seed + window)
        if end <= seed:
            continue

        # Seed query points from pred bbox at seed frame
        bbox_seed = pred_s[seed]
        bw = max(bbox_seed[2] - bbox_seed[0], 1.0)
        bh = max(bbox_seed[3] - bbox_seed[1], 1.0)
        if bw < 1.0 or bh < 1.0:
            continue  # degenerate bbox — skip this window
        query = sample_query_points(bbox_seed, n_points=n_points)
        if len(query) == 0:
            continue

        window_frames = resized[seed:end]
        tracks, vis = tracker_fn(window_frames, query)  # (W, P, 2), (W, P)

        # Compute features for this window's frames
        feats = compute_point_features_sequence(tracks, vis, pred_s[seed:end])  # (W, F)

        # Write into output — latest seed overwrites earlier ones (latest wins)
        out[seed:end] = feats

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_paths() -> None:
    root = Path(__file__).resolve().parents[4]
    for p in (root / "src", root / "saltr" / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _get_dataset_sequences(dataset_name: str) -> dict[str, Any]:
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_paths()
    data = np.load(args.npz, allow_pickle=True)

    seq_keys = sorted(
        k.replace("bbox_pred/", "")
        for k in data.files
        if k.startswith("bbox_pred/")
    )
    if args.smoke_test:
        seq_keys = seq_keys[: args.smoke_test]

    dataset_cache: dict[str, dict[str, Any]] = {}

    arrays: dict[str, Any] = {}
    n_ok = 0
    n_skip = 0
    interrupted = False

    try:
        for seq_key in seq_keys:
            dataset_name, seq_name = seq_key.split("/", 1)
            dataset_cache.setdefault(dataset_name, _get_dataset_sequences(dataset_name))
            seq_obj = dataset_cache[dataset_name].get(seq_name)
            if seq_obj is None:
                print(f"  [skip] {seq_key} — not found in dataset", flush=True)
                n_skip += 1
                continue

            pred_xywh = data[f"bbox_pred/{seq_key}"].astype(np.float32)
            pred_xyxy = xywh_to_xyxy(pred_xywh)
            T = len(pred_xywh)

            frames = list(seq_obj.frames)
            if len(frames) != T:
                print(f"  [skip] {seq_key} — frame count mismatch {len(frames)} vs {T}", flush=True)
                n_skip += 1
                continue

            t0 = time.time()
            feats = extract_sequence_features(
                frames=frames,
                pred_xyxy=pred_xyxy,
                method=args.method,
                stride=args.stride,
                window=args.window,
                n_points=args.n_points,
                max_side=args.max_side,
            )
            elapsed = time.time() - t0

            nan_frac = float(np.isnan(feats).mean())
            arrays[f"point_features/{seq_key}"] = feats
            arrays[f"point_method/{seq_key}"] = np.array(args.method)
            n_ok += 1
            print(
                f"  {seq_key:<42}  T={T:>5}  nan={nan_frac:.2f}  {elapsed:.1f}s",
                flush=True,
            )

    except KeyboardInterrupt:
        interrupted = True
        print("\n[point_sidecar_extractor] Interrupted!", flush=True)

    if n_skip > 0 and not args.allow_partial and not args.smoke_test:
        raise RuntimeError(
            f"{n_skip} sequences skipped — canonical sidecar requires all sequences. "
            "Fix the issues above or use --allow-partial / --smoke-test."
        )
    if interrupted and not args.allow_partial and not args.smoke_test:
        raise RuntimeError(
            "KeyboardInterrupt: canonical sidecar NOT saved. "
            "Use --allow-partial to save partial results."
        )

    # Metadata
    arrays["point_feature_names"] = np.array(POINT_FEATURE_NAMES, dtype=object)
    arrays["extractor_method"] = np.array(args.method)
    arrays["stride"] = np.array(args.stride)
    arrays["window"] = np.array(args.window)
    arrays["n_points"] = np.array(args.n_points)
    arrays["n_sequences"] = np.array(n_ok)
    arrays["source_npz_md5"] = np.array(_md5(args.npz))
    arrays["created_at"] = np.array(datetime.now(timezone.utc).isoformat())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **arrays)

    print(f"\n[point_sidecar_extractor] saved {n_ok} sequences → {out_path}", flush=True)
    if n_skip:
        print(f"  WARNING: {n_skip} sequences skipped", flush=True)

    return {"n_ok": n_ok, "n_skip": n_skip, "output": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default="saltr/data/salt_rd_v2_labels.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="lk", choices=["lk", "farneback"])
    parser.add_argument("--stride", type=int, default=15,
                        help="Re-seed every N frames")
    parser.add_argument("--window", type=int, default=25,
                        help="Track forward N frames per seed")
    parser.add_argument("--n-points", type=int, default=9)
    parser.add_argument("--max-side", type=int, default=320,
                        help="Resize longest edge to this before tracking")
    parser.add_argument("--smoke-test", type=int, default=0, metavar="N",
                        help="Run on first N sequences only")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Save even if some sequences failed")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
