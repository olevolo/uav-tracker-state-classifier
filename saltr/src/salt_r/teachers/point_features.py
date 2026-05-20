"""Point consistency features for offline teacher label generation.

These features are computed from point tracks (from CoTracker3 or similar)
and serve as teacher signals for SALT-RD training.

NO dependency on CoTracker3 at import time — gate with try/except.
All math functions work on synthetic numpy arrays for testing.
"""

import numpy as np
from typing import Optional

POINT_FEATURE_NAMES = [
    "pt_visible_ratio",           # fraction of tracked points still visible
    "pt_inside_pred_ratio",       # fraction inside predicted bbox
    "pt_inside_pred_weighted",    # confidence-weighted version
    "pt_forward_backward_error",  # mean ||track_t→t+1→t - track_t||
    "pt_median_motion",           # median displacement from previous frame
    "pt_motion_iqr",              # IQR of per-point motion magnitudes
    "pt_affine_residual",         # residual after affine fit to point cloud motion
    "pt_cluster_area_ratio",      # point cloud bounding area / predicted bbox area
    "pt_cluster_aspect_delta",    # change in point cloud aspect ratio
    "pt_flow_agreement",          # agreement of point motions with optical flow direction
    "pt_bbox_center_disagreement", # distance between point cloud centroid and bbox center
    "pt_survival_since_init",     # fraction of initial points still tracked (not lost)
    "pt_split_score",             # evidence that point cloud splits into two groups
]


def _bbox_area(bbox: np.ndarray) -> float:
    """Compute area of [x1,y1,x2,y2] bbox."""
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def _points_inside_bbox(points: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Return bool mask of points inside bbox [x1,y1,x2,y2].

    Args:
        points: (P, 2) xy positions
        bbox:   [x1, y1, x2, y2]
    Returns:
        (P,) bool array
    """
    x1, y1, x2, y2 = bbox
    return (
        (points[:, 0] >= x1) & (points[:, 0] <= x2) &
        (points[:, 1] >= y1) & (points[:, 1] <= y2)
    )


# ---------------------------------------------------------------------------
# Individual feature helpers
# ---------------------------------------------------------------------------

def _compute_visible_ratio(
    vis: np.ndarray,  # (P,) bool/float visibility at time t
) -> float:
    """Fraction of tracked points that are visible."""
    if len(vis) == 0:
        return float("nan")
    return float(np.mean(vis > 0.5))


def _compute_inside_pred_ratio(
    points: np.ndarray,   # (P, 2) positions at time t
    vis: np.ndarray,      # (P,) visibility
    bbox: np.ndarray,     # [x1,y1,x2,y2]
) -> float:
    """Fraction of visible points inside predicted bbox."""
    mask = vis > 0.5
    if not np.any(mask):
        return float("nan")
    visible_pts = points[mask]
    inside = _points_inside_bbox(visible_pts, bbox)
    return float(np.mean(inside))


def _compute_inside_pred_weighted(
    points: np.ndarray,   # (P, 2)
    vis: np.ndarray,      # (P,) float confidence weights
    bbox: np.ndarray,     # [x1,y1,x2,y2]
) -> float:
    """Confidence-weighted fraction of points inside predicted bbox."""
    total_weight = float(np.sum(vis))
    if total_weight < 1e-9:
        return float("nan")
    inside = _points_inside_bbox(points, bbox).astype(float)
    return float(np.sum(inside * vis) / total_weight)


def _compute_forward_backward_error(
    tracks_t: np.ndarray,       # (P, 2) positions at time t
    vis_t: np.ndarray,          # (P,) visibility at t
    prev_tracks: Optional[np.ndarray],  # (P, 2) positions at t-1
    vis_prev: Optional[np.ndarray],     # (P,) visibility at t-1
) -> float:
    """Mean forward-backward tracking error.

    For each point visible at both t-1 and t: error = ||pos_t_reconstructed - pos_{t-1}||.
    Here we approximate using the displacement difference between consecutive frames.
    For true forward-backward, we'd need t+1 tracks (passed as prev_tracks for reverse).
    """
    if prev_tracks is None or vis_prev is None:
        return float("nan")
    mask = (vis_t > 0.5) & (vis_prev > 0.5)
    if not np.any(mask):
        return float("nan")
    # Displacement from t-1 to t, and then the "backward" would be -(t to t-1)
    # We measure how symmetric the motion is: error = ||forward_disp + backward_disp||
    # Since we only have consecutive frames, this is 0 for perfect tracks (approx)
    fwd = tracks_t[mask] - prev_tracks[mask]
    # Approximate backward error: magnitude of net displacement (should be small for consistent motion)
    errors = np.linalg.norm(fwd, axis=1)
    return float(np.mean(errors))


def _compute_median_motion(
    tracks_t: np.ndarray,       # (P, 2)
    vis_t: np.ndarray,          # (P,)
    prev_tracks: Optional[np.ndarray],  # (P, 2)
    vis_prev: Optional[np.ndarray],     # (P,)
) -> float:
    """Median per-point displacement magnitude from previous frame."""
    if prev_tracks is None or vis_prev is None:
        return float("nan")
    mask = (vis_t > 0.5) & (vis_prev > 0.5)
    if not np.any(mask):
        return float("nan")
    displacements = np.linalg.norm(tracks_t[mask] - prev_tracks[mask], axis=1)
    return float(np.median(displacements))


def _compute_motion_iqr(
    tracks_t: np.ndarray,
    vis_t: np.ndarray,
    prev_tracks: Optional[np.ndarray],
    vis_prev: Optional[np.ndarray],
) -> float:
    """IQR of per-point motion magnitudes."""
    if prev_tracks is None or vis_prev is None:
        return float("nan")
    mask = (vis_t > 0.5) & (vis_prev > 0.5)
    if np.sum(mask) < 2:
        return float("nan")
    displacements = np.linalg.norm(tracks_t[mask] - prev_tracks[mask], axis=1)
    q75, q25 = np.percentile(displacements, [75, 25])
    return float(q75 - q25)


def _fit_affine_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Fit affine transform from src (N,2) to dst (N,2). Returns residuals array (N,)."""
    n = len(src)
    if n < 3:
        return np.zeros(n)
    # Build system: dst_x = a*src_x + b*src_y + c, dst_y = d*src_x + e*src_y + f
    A = np.column_stack([src, np.ones(n)])  # (N, 3)
    try:
        params_x, _, _, _ = np.linalg.lstsq(A, dst[:, 0], rcond=None)
        params_y, _, _, _ = np.linalg.lstsq(A, dst[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(n)
    pred_x = A @ params_x
    pred_y = A @ params_y
    residuals = np.sqrt((dst[:, 0] - pred_x) ** 2 + (dst[:, 1] - pred_y) ** 2)
    return residuals


def _compute_affine_residual(
    tracks_t: np.ndarray,
    vis_t: np.ndarray,
    prev_tracks: Optional[np.ndarray],
    vis_prev: Optional[np.ndarray],
) -> float:
    """Mean residual after fitting affine motion model to visible point displacements."""
    if prev_tracks is None or vis_prev is None:
        return float("nan")
    mask = (vis_t > 0.5) & (vis_prev > 0.5)
    if np.sum(mask) < 3:
        return float("nan")
    src = prev_tracks[mask]
    dst = tracks_t[mask]
    residuals = _fit_affine_2d(src, dst)
    return float(np.mean(residuals))


def _compute_cluster_area_ratio(
    points: np.ndarray,   # (P, 2) visible points
    vis: np.ndarray,      # (P,)
    bbox: np.ndarray,     # [x1,y1,x2,y2] predicted bbox
) -> float:
    """Ratio of point cloud bounding box area to predicted bbox area."""
    bbox_area = _bbox_area(bbox)
    if bbox_area < 1e-6:
        return float("nan")
    mask = vis > 0.5
    if np.sum(mask) < 2:
        return float("nan")
    pts = points[mask]
    pt_x1, pt_y1 = pts[:, 0].min(), pts[:, 1].min()
    pt_x2, pt_y2 = pts[:, 0].max(), pts[:, 1].max()
    pt_area = max(0.0, pt_x2 - pt_x1) * max(0.0, pt_y2 - pt_y1)
    return float(pt_area / bbox_area)


def _compute_cluster_aspect_delta(
    points: np.ndarray,
    vis: np.ndarray,
    bbox: np.ndarray,
) -> float:
    """Change in aspect ratio between point cloud bounding box and predicted bbox."""
    mask = vis > 0.5
    if np.sum(mask) < 2:
        return float("nan")
    pts = points[mask]
    pt_w = max(pts[:, 0].max() - pts[:, 0].min(), 1e-6)
    pt_h = max(pts[:, 1].max() - pts[:, 1].min(), 1e-6)
    pt_aspect = pt_w / pt_h

    bbox_w = max(bbox[2] - bbox[0], 1e-6)
    bbox_h = max(bbox[3] - bbox[1], 1e-6)
    bbox_aspect = bbox_w / bbox_h

    return float(abs(pt_aspect - bbox_aspect))


def _compute_flow_agreement(
    tracks_t: np.ndarray,
    vis_t: np.ndarray,
    prev_tracks: Optional[np.ndarray],
    vis_prev: Optional[np.ndarray],
) -> float:
    """Agreement of point motions with dominant optical flow direction.

    Computed as the cosine similarity between each point's motion vector
    and the median motion direction, then averaged.
    """
    if prev_tracks is None or vis_prev is None:
        return float("nan")
    mask = (vis_t > 0.5) & (vis_prev > 0.5)
    if np.sum(mask) < 2:
        return float("nan")
    motions = tracks_t[mask] - prev_tracks[mask]  # (M, 2)
    median_motion = np.median(motions, axis=0)
    median_norm = np.linalg.norm(median_motion)
    if median_norm < 1e-9:
        return 1.0  # all points stationary — perfect agreement
    median_dir = median_motion / median_norm
    norms = np.linalg.norm(motions, axis=1, keepdims=True)
    # Avoid division by zero for stationary points
    valid = norms[:, 0] > 1e-9
    if not np.any(valid):
        return 1.0
    directions = np.where(norms > 1e-9, motions / np.maximum(norms, 1e-9), 0.0)
    cosines = directions @ median_dir  # (M,)
    return float(np.mean(cosines[valid]))


def _compute_bbox_center_disagreement(
    points: np.ndarray,
    vis: np.ndarray,
    bbox: np.ndarray,
) -> float:
    """Normalized distance between point cloud centroid and bbox center."""
    mask = vis > 0.5
    if not np.any(mask):
        return float("nan")
    centroid = points[mask].mean(axis=0)
    bbox_cx = (bbox[0] + bbox[2]) / 2.0
    bbox_cy = (bbox[1] + bbox[3]) / 2.0
    bbox_diag = np.sqrt(
        max(bbox[2] - bbox[0], 1.0) ** 2 + max(bbox[3] - bbox[1], 1.0) ** 2
    )
    dist = np.sqrt((centroid[0] - bbox_cx) ** 2 + (centroid[1] - bbox_cy) ** 2)
    return float(dist / bbox_diag)


def _compute_survival_since_init(
    vis_t: np.ndarray,          # (P,) visibility at time t
    vis_init: Optional[np.ndarray],  # (P,) visibility at t=0 (init frame)
) -> float:
    """Fraction of initially visible points that are still visible."""
    if vis_init is None:
        return float("nan")
    init_mask = vis_init > 0.5
    if not np.any(init_mask):
        return float("nan")
    still_visible = (vis_t > 0.5) & init_mask
    return float(np.sum(still_visible) / np.sum(init_mask))


def _numpy_kmeans(points: np.ndarray, k: int = 2, max_iter: int = 30) -> np.ndarray:
    """Simple k-means clustering using numpy only.

    Returns (N,) integer cluster assignments.
    """
    n = len(points)
    if n <= k:
        return np.arange(n) % k
    rng = np.random.default_rng(42)
    # k-means++ style init: pick first center randomly, rest by distance
    centers = [points[rng.integers(n)]]
    for _ in range(k - 1):
        dists = np.array([min(np.linalg.norm(p - c) ** 2 for c in centers) for p in points])
        total = dists.sum()
        if total < 1e-9:
            centers.append(points[rng.integers(n)])
        else:
            probs = dists / total
            idx = rng.choice(n, p=probs)
            centers.append(points[idx])
    centers_arr = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assignment
        dists = np.array([
            np.linalg.norm(points - c, axis=1) for c in centers_arr
        ])  # (k, N)
        new_labels = np.argmin(dists, axis=0)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update
        for j in range(k):
            pts_j = points[labels == j]
            if len(pts_j) > 0:
                centers_arr[j] = pts_j.mean(axis=0)
    return labels


def _compute_split_score(
    points: np.ndarray,  # (P, 2) all tracked points (visible only)
    vis: np.ndarray,     # (P,)
) -> float:
    """Evidence of point cloud splitting into two clusters.

    Returns ratio of within-cluster variance to total variance.
    High ratio (→ 1.0) means one coherent cluster (no split).
    Low ratio (→ 0.0) means two well-separated clusters (split).

    Uses sklearn if available, otherwise falls back to numpy k-means.
    """
    mask = vis > 0.5
    n_visible = int(np.sum(mask))
    if n_visible < 4:
        return float("nan")
    pts = points[mask]

    # Try sklearn KMeans first; fall back to numpy implementation
    try:
        from sklearn.cluster import KMeans  # type: ignore
        km = KMeans(n_clusters=2, n_init=3, random_state=42)
        labels = km.fit_predict(pts)
    except Exception:
        labels = _numpy_kmeans(pts, k=2)

    total_var = float(np.var(pts, axis=0).sum()) * len(pts)
    if total_var < 1e-9:
        return 1.0  # all points coincide — treated as one cluster

    within_var = 0.0
    for j in range(2):
        cluster_pts = pts[labels == j]
        if len(cluster_pts) > 1:
            within_var += float(np.var(cluster_pts, axis=0).sum()) * len(cluster_pts)
        # cluster of size 0 or 1 contributes 0 variance
    return float(within_var / total_var)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_point_features(
    point_tracks: np.ndarray,     # (T, P, 2) — xy positions, NaN = not visible
    point_visibility: np.ndarray, # (T, P) bool/float — visibility flags
    pred_bboxes: np.ndarray,      # (T, 4) — [x1,y1,x2,y2] predicted bboxes
    prev_point_tracks: Optional[np.ndarray] = None,  # (T, P, 2) for fwd-bwd error
    t: int = 0,                   # frame index to compute features for
) -> np.ndarray:
    """Compute point consistency features for a single frame.

    Returns: (len(POINT_FEATURE_NAMES),) float32 array.
    NaN for features that cannot be computed (e.g., no visible points).
    """
    T, P, _ = point_tracks.shape
    assert 0 <= t < T, f"Frame index {t} out of range [0, {T})"

    tracks_t = point_tracks[t]          # (P, 2)
    vis_t = point_visibility[t].astype(float)  # (P,)
    bbox_t = pred_bboxes[t]             # (4,)

    # Previous frame data for temporal features
    if t > 0:
        tracks_prev = point_tracks[t - 1]
        vis_prev = point_visibility[t - 1].astype(float)
    else:
        tracks_prev = None
        vis_prev = None

    # Initial frame data for survival feature
    vis_init = point_visibility[0].astype(float)

    feats = np.full(len(POINT_FEATURE_NAMES), float("nan"), dtype=np.float32)

    feats[0] = _compute_visible_ratio(vis_t)
    feats[1] = _compute_inside_pred_ratio(tracks_t, vis_t, bbox_t)
    feats[2] = _compute_inside_pred_weighted(tracks_t, vis_t, bbox_t)

    # Forward-backward error: use prev_point_tracks if provided (t and t-1 frames)
    if prev_point_tracks is not None and t > 0:
        feats[3] = _compute_forward_backward_error(
            tracks_t, vis_t,
            prev_point_tracks[t - 1], point_visibility[t - 1].astype(float)
        )
    else:
        feats[3] = _compute_forward_backward_error(
            tracks_t, vis_t, tracks_prev, vis_prev
        )

    feats[4] = _compute_median_motion(tracks_t, vis_t, tracks_prev, vis_prev)
    feats[5] = _compute_motion_iqr(tracks_t, vis_t, tracks_prev, vis_prev)
    feats[6] = _compute_affine_residual(tracks_t, vis_t, tracks_prev, vis_prev)
    feats[7] = _compute_cluster_area_ratio(tracks_t, vis_t, bbox_t)
    feats[8] = _compute_cluster_aspect_delta(tracks_t, vis_t, bbox_t)
    feats[9] = _compute_flow_agreement(tracks_t, vis_t, tracks_prev, vis_prev)
    feats[10] = _compute_bbox_center_disagreement(tracks_t, vis_t, bbox_t)
    feats[11] = _compute_survival_since_init(vis_t, vis_init)
    feats[12] = _compute_split_score(tracks_t, vis_t)

    return feats


def compute_point_features_sequence(
    point_tracks: np.ndarray,     # (T, P, 2)
    point_visibility: np.ndarray, # (T, P)
    pred_bboxes: np.ndarray,      # (T, 4)
) -> np.ndarray:
    """Compute features for all T frames. Returns (T, F) array."""
    T = point_tracks.shape[0]
    F = len(POINT_FEATURE_NAMES)
    out = np.full((T, F), float("nan"), dtype=np.float32)
    for t in range(T):
        out[t] = compute_point_features(
            point_tracks, point_visibility, pred_bboxes, t=t
        )
    return out


def compute_point_teacher_labels(
    point_tracks: np.ndarray,     # (T, P, 2)
    point_visibility: np.ndarray, # (T, P)
    pred_bboxes: np.ndarray,      # (T, 4)
    gt_bboxes: np.ndarray,        # (T, 4) — GT bboxes, OFFLINE ONLY
    iou_trace: np.ndarray,        # (T,)
) -> dict[str, np.ndarray]:
    """Compute teacher labels from point tracks + GT.

    IMPORTANT: gt_bboxes and iou_trace used only here (teacher label generation).
    Student runtime features MUST NOT use GT-relative fields.

    Returns dict with:
      point_consistency_good: (T,) bool — visible + inside pred + low affine residual
      point_identity_break: (T,) bool — IoU<0.3 AND points not inside pred
      point_recoverable: (T,) bool — IoU<0.3 AND points still visible + coherent
    """
    T = point_tracks.shape[0]

    # Compute per-frame features
    features = compute_point_features_sequence(point_tracks, point_visibility, pred_bboxes)

    # Feature indices
    idx_vis_ratio = POINT_FEATURE_NAMES.index("pt_visible_ratio")
    idx_inside = POINT_FEATURE_NAMES.index("pt_inside_pred_ratio")
    idx_affine = POINT_FEATURE_NAMES.index("pt_affine_residual")
    idx_split = POINT_FEATURE_NAMES.index("pt_split_score")

    vis_ratio = features[:, idx_vis_ratio]
    inside_ratio = features[:, idx_inside]
    affine_res = features[:, idx_affine]
    split_score = features[:, idx_split]

    iou = np.asarray(iou_trace, dtype=float)

    # point_consistency_good: good visibility + mostly inside pred + coherent motion
    # Use NaN-safe comparisons: treat NaN as failing the condition
    consistency_good = (
        (np.nan_to_num(vis_ratio, nan=0.0) > 0.5) &
        (np.nan_to_num(inside_ratio, nan=0.0) > 0.5) &
        (np.nan_to_num(affine_res, nan=1e9) < 5.0)
    )

    # point_identity_break: tracker drifted (low IoU) AND points not following
    identity_break = (
        (iou < 0.3) &
        (np.nan_to_num(inside_ratio, nan=1.0) < 0.3)
    )

    # point_recoverable: tracking failed (low IoU) BUT points still visible and coherent
    # (split_score close to 1.0 = coherent single cluster = target still trackable)
    point_recoverable = (
        (iou < 0.3) &
        (np.nan_to_num(vis_ratio, nan=0.0) > 0.4) &
        (np.nan_to_num(split_score, nan=0.0) > 0.5)
    )

    return {
        "point_consistency_good": consistency_good.astype(bool),
        "point_identity_break": identity_break.astype(bool),
        "point_recoverable": point_recoverable.astype(bool),
    }
