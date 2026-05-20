"""Offline CoTracker3 point track export for SALT-RD teacher features.

CoTracker3 is NOT a hard dependency. Import is gated.
This module provides the infrastructure to:
1. Load dataset sequences
2. Sample query points in initial GT bbox
3. Run CoTracker3 (if available) or load precomputed tracks
4. Compute point consistency features
5. Save sidecar NPZ with provenance

Sidecar NPZ schema:
  point_tracks/{seq}:     float32 (T, P, 2)  — NaN = not visible
  point_visibility/{seq}: bool    (T, P)
  point_features/{seq}:   float32 (T, F_point)
  teacher_labels/{seq}:   dict of (T,) arrays (point_consistency_good, etc.)
  point_feature_names:    list[str]
  teacher_label_names:    list[str]
  teacher_version:        str
  teacher_model:          str ("cotracker3" | "synthetic" | "precomputed")
  created_at:             str
"""

import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from salt_r.teachers.point_features import (
    POINT_FEATURE_NAMES,
    compute_point_features_sequence,
    compute_point_teacher_labels,
)

# Sidecar format version
TEACHER_VERSION = "1.0"
TEACHER_LABEL_NAMES = [
    "point_consistency_good",
    "point_identity_break",
    "point_recoverable",
]


def sample_query_points(
    gt_bbox: np.ndarray,   # [x1,y1,x2,y2] initial GT bbox
    n_points: int = 9,     # 3x3 grid by default
    include_corners: bool = True,
) -> np.ndarray:
    """Sample query points inside GT bbox for CoTracker3.

    Returns: (P, 2) array of (x, y) points.
    For small targets (area < 400px²): 3x3 grid = 9 points
    For larger targets: 4x4 or 5x5 grid
    """
    x1, y1, x2, y2 = float(gt_bbox[0]), float(gt_bbox[1]), float(gt_bbox[2]), float(gt_bbox[3])
    w = x2 - x1
    h = y2 - y1
    area = w * h

    # Determine grid size based on target area
    if area < 400.0:
        grid_size = 3
    elif area < 2500.0:
        grid_size = 4
    else:
        grid_size = 5

    # Sample grid points
    # Use linspace with small inset to avoid bbox boundary
    inset_x = w * 0.1
    inset_y = h * 0.1
    xs = np.linspace(x1 + inset_x, x2 - inset_x, grid_size)
    ys = np.linspace(y1 + inset_y, y2 - inset_y, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])  # (grid_size^2, 2)

    if include_corners and grid_size < 5:
        # Add corners of the bbox (inset slightly)
        corners = np.array([
            [x1 + inset_x, y1 + inset_y],
            [x2 - inset_x, y1 + inset_y],
            [x1 + inset_x, y2 - inset_y],
            [x2 - inset_x, y2 - inset_y],
        ])
        # Deduplicate points close to existing grid points
        combined = np.vstack([grid_pts, corners])
        # Remove near-duplicates (within 1 pixel)
        unique_pts = [combined[0]]
        for pt in combined[1:]:
            dists = np.linalg.norm(np.array(unique_pts) - pt, axis=1)
            if dists.min() > 1.0:
                unique_pts.append(pt)
        grid_pts = np.array(unique_pts)

    return grid_pts.astype(np.float32)


def run_cotracker3_on_sequence(
    frames: list,              # list of (H,W,3) uint8 arrays
    query_points: np.ndarray,  # (P, 2) initial query points
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run CoTracker3 offline on a sequence.

    Returns: (tracks (T,P,2), visibility (T,P))
    Raises ImportError if cotracker3 not installed — use precomputed instead.
    """
    try:
        import torch  # noqa: F401
        # Try to import cotracker3
        try:
            from cotracker.predictor import CoTrackerPredictor  # type: ignore
        except ImportError:
            try:
                import cotracker3  # type: ignore  # noqa: F401
                from cotracker3.predictor import CoTrackerPredictor  # type: ignore
            except ImportError:
                raise ImportError(
                    "CoTracker3 not available. Use precomputed tracks or synthetic data."
                )

        import torch

        T = len(frames)
        H, W = frames[0].shape[:2]

        # Stack frames into video tensor (1, T, 3, H, W)
        video = np.stack(frames, axis=0)  # (T, H, W, 3)
        video_t = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
        video_t = video_t.unsqueeze(0)  # (1, T, 3, H, W)

        # Prepare query points: CoTracker expects (1, P, 3) = (batch, point, [t, x, y])
        # We query from frame 0
        P = len(query_points)
        queries = torch.zeros(1, P, 3)
        queries[0, :, 0] = 0  # query at frame 0
        queries[0, :, 1:] = torch.from_numpy(query_points).float()

        model = CoTrackerPredictor(checkpoint=None)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            pred_tracks, pred_vis = model(video_t.to(device), queries=queries.to(device))

        # pred_tracks: (1, T, P, 2), pred_vis: (1, T, P)
        tracks = pred_tracks[0].cpu().numpy()  # (T, P, 2)
        visibility = pred_vis[0].cpu().numpy().astype(bool)  # (T, P)

        # Mark not-visible points as NaN
        tracks[~visibility] = float("nan")
        return tracks, visibility

    except ImportError:
        raise ImportError(
            "CoTracker3 not available. Use precomputed tracks or synthetic data."
        )


def _make_synthetic_tracks(
    T: int,
    P: int,
    gt_bboxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic point tracks that follow GT bboxes.

    Used as fallback when CoTracker3 is not available and no precomputed tracks exist.
    Points are placed at a grid within each GT bbox and perturbed slightly.
    """
    rng = np.random.default_rng(42)

    # Initialize points in first GT bbox
    x1, y1, x2, y2 = gt_bboxes[0]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    grid_side = int(np.ceil(np.sqrt(P)))
    xs = np.linspace(x1 + w * 0.1, x2 - w * 0.1, grid_side)
    ys = np.linspace(y1 + h * 0.1, y2 - h * 0.1, grid_side)
    xx, yy = np.meshgrid(xs, ys)
    init_pts = np.column_stack([xx.ravel(), yy.ravel()])[:P]

    # Relative positions within bbox (0..1)
    rel_x = (init_pts[:, 0] - x1) / w
    rel_y = (init_pts[:, 1] - y1) / h

    tracks = np.full((T, P, 2), float("nan"), dtype=np.float32)
    visibility = np.zeros((T, P), dtype=bool)

    for t in range(T):
        bx1, by1, bx2, by2 = gt_bboxes[t]
        bw = max(bx2 - bx1, 1.0)
        bh = max(by2 - by1, 1.0)
        pts_x = bx1 + rel_x * bw + rng.normal(0, 0.5, P)
        pts_y = by1 + rel_y * bh + rng.normal(0, 0.5, P)
        tracks[t, :, 0] = pts_x
        tracks[t, :, 1] = pts_y
        visibility[t] = True  # all visible in synthetic data

    return tracks, visibility


def process_sequence_to_sidecar(
    seq_key: str,
    frames_or_path,              # list of (H,W,3) uint8 frames or path to video
    gt_bboxes: np.ndarray,       # (T, 4) GT bboxes [x1,y1,x2,y2]
    pred_bboxes: np.ndarray,     # (T, 4) predicted bboxes
    iou_trace: np.ndarray,       # (T,)
    precomputed_tracks: Optional[np.ndarray] = None,  # (T, P, 2) if already computed
    precomputed_visibility: Optional[np.ndarray] = None,  # (T, P) if already computed
    device: str = "cpu",
) -> dict:
    """Process one sequence to point teacher sidecar data.

    Returns a dict with all sidecar fields for this sequence.
    """
    T = len(gt_bboxes)
    teacher_model = "unknown"

    if precomputed_tracks is not None:
        # Use precomputed tracks
        tracks = precomputed_tracks.astype(np.float32)
        if precomputed_visibility is not None:
            visibility = precomputed_visibility.astype(bool)
        else:
            # Infer visibility from NaN
            visibility = ~np.any(np.isnan(tracks), axis=-1)
        teacher_model = "precomputed"

    else:
        # Try CoTracker3
        query_pts = sample_query_points(gt_bboxes[0])

        if isinstance(frames_or_path, (str, Path)):
            # Load frames from video path
            try:
                import cv2  # type: ignore
                cap = cv2.VideoCapture(str(frames_or_path))
                frames_list = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                frames = frames_list
            except ImportError:
                frames = None
        else:
            frames = frames_or_path

        cotracker_ok = False
        if frames is not None:
            try:
                tracks, visibility = run_cotracker3_on_sequence(frames, query_pts, device=device)
                teacher_model = "cotracker3"
                cotracker_ok = True
            except ImportError:
                pass

        if not cotracker_ok:
            # Fall back to synthetic tracks following GT bboxes
            P = len(query_pts)
            tracks, visibility = _make_synthetic_tracks(T, P, gt_bboxes)
            teacher_model = "synthetic"

    # Compute point consistency features
    features = compute_point_features_sequence(tracks, visibility, pred_bboxes)

    # Compute teacher labels
    labels_dict = compute_point_teacher_labels(
        tracks, visibility, pred_bboxes, gt_bboxes, iou_trace
    )

    return {
        "point_tracks": tracks,                  # (T, P, 2)
        "point_visibility": visibility,           # (T, P)
        "point_features": features,              # (T, F)
        "teacher_labels": labels_dict,           # dict of (T,) bool arrays
        "teacher_model": teacher_model,
        "seq_key": seq_key,
    }


def save_sidecar_npz(
    sidecar_data: list[dict],
    output_path: str,
) -> None:
    """Save a list of processed sequences to a sidecar NPZ file.

    Args:
        sidecar_data: list of dicts from process_sequence_to_sidecar
        output_path: path to output NPZ file
    """
    arrays: dict[str, np.ndarray] = {}

    for entry in sidecar_data:
        seq = entry["seq_key"]
        arrays[f"point_tracks/{seq}"] = entry["point_tracks"]
        arrays[f"point_visibility/{seq}"] = entry["point_visibility"]
        arrays[f"point_features/{seq}"] = entry["point_features"]

        # Teacher labels — store each as a separate array
        for label_name, label_arr in entry["teacher_labels"].items():
            arrays[f"teacher_labels/{label_name}/{seq}"] = label_arr.astype(np.uint8)

    # Metadata
    arrays["point_feature_names"] = np.array(POINT_FEATURE_NAMES, dtype=object)
    arrays["teacher_label_names"] = np.array(TEACHER_LABEL_NAMES, dtype=object)
    arrays["teacher_version"] = np.array(TEACHER_VERSION)
    arrays["created_at"] = np.array(
        datetime.now(timezone.utc).isoformat()
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
