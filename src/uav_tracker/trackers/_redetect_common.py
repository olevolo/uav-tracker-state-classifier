"""Shared wide-crop re-detection helpers used by SGLATrack and AVTrack.

Both trackers share the same 16×16 score-map layout (FEAT_SZ=16) and
256×256 search window (SEARCH_SIZE=256). The helpers are tracker-agnostic;
each adapter's redetect() calls them with its own forward-pass output.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from uav_tracker.types import BBox

FEAT_SZ: int = 16        # score-map side (256px / 16px patch stride)
SEARCH_SIZE: int = 256   # search patch size in pixels


# ---------------------------------------------------------------------------
# Peak selection
# ---------------------------------------------------------------------------

def select_candidate_peak_indices(
    score_map: torch.Tensor,
    max_candidates: int = 5,
    nms_radius: int = 1,
    min_score_ratio: float = 0.05,
) -> list[int]:
    """Select spatially distinct score-map peaks with greedy NMS.

    The raw top-2 cells are often adjacent samples from the same response peak.
    For distractor analysis we need separate candidate modes, so we greedily
    suppress cells within ``nms_radius`` grid steps of already selected peaks.
    """
    flat = score_map.reshape(-1)
    if flat.numel() == 0 or max_candidates <= 0:
        return []

    order = torch.argsort(flat, descending=True)
    top_score = float(flat[order[0]])
    selected: list[int] = []

    for idx_t in order:
        idx = int(idx_t.item())
        score = float(flat[idx])
        if selected and top_score > 0.0 and score < top_score * min_score_ratio:
            break
        row, col = divmod(idx, FEAT_SZ)
        too_close = False
        for prev_idx in selected:
            prev_row, prev_col = divmod(prev_idx, FEAT_SZ)
            if max(abs(row - prev_row), abs(col - prev_col)) <= nms_radius:
                too_close = True
                break
        if too_close:
            continue
        selected.append(idx)
        if len(selected) >= max_candidates:
            break

    return selected


# ---------------------------------------------------------------------------
# Bbox decoding from peak
# ---------------------------------------------------------------------------

def candidate_bbox_from_peak(
    idx: int,
    score: float,
    top_score: float,
    size_map: torch.Tensor,
    offset_map: torch.Tensor,
    prev_bbox: "BBox",
    resize_factor: float,
    rank: int,
) -> dict:
    """Decode one score-map peak into frame-space bbox metadata."""
    row, col = divmod(int(idx), FEAT_SZ)
    size_flat = size_map.flatten(2)
    offset_flat = offset_map.flatten(2)

    gather_idx = torch.tensor([[[idx]]], device=size_map.device, dtype=torch.long)
    gather_idx = gather_idx.expand(size_flat.shape[0], 2, 1)
    size = size_flat.gather(dim=2, index=gather_idx).squeeze(-1)[0]
    offset = offset_flat.gather(dim=2, index=gather_idx).squeeze(-1)[0]

    cx_norm = (float(col) + float(offset[0])) / FEAT_SZ
    cy_norm = (float(row) + float(offset[1])) / FEAT_SZ
    w_norm = float(size[0])
    h_norm = float(size[1])

    cx_pred = cx_norm * SEARCH_SIZE / resize_factor
    cy_pred = cy_norm * SEARCH_SIZE / resize_factor
    w_pred = max(1.0, w_norm * SEARCH_SIZE / resize_factor)
    h_pred = max(1.0, h_norm * SEARCH_SIZE / resize_factor)

    cx_prev = prev_bbox.x + prev_bbox.w / 2
    cy_prev = prev_bbox.y + prev_bbox.h / 2
    half = SEARCH_SIZE / (2 * resize_factor)

    x = cx_prev + cx_pred - half - w_pred / 2
    y = cy_prev + cy_pred - half - h_pred / 2
    cx = x + w_pred / 2
    cy = y + h_pred / 2

    score_ratio = float(score / (top_score + 1e-8)) if top_score > 0 else 0.0
    grid_center_distance = float(((row - 7.5) ** 2 + (col - 7.5) ** 2) ** 0.5)

    return {
        "rank": int(rank),
        "index": int(idx),
        "row": int(row),
        "col": int(col),
        "score": float(score),
        "score_ratio": score_ratio,
        "grid_center_distance": grid_center_distance,
        "bbox": [float(x), float(y), float(w_pred), float(h_pred)],
        "center": [float(cx), float(cy)],
    }


# ---------------------------------------------------------------------------
# Ranking and deduplication
# ---------------------------------------------------------------------------

def redetect_rank_key(cand: dict, rank_by: str) -> tuple:
    """Sort key for choosing the best redetect candidate.

    ``quality`` (default) = APCE * score_ratio — the sharpest correlation peak.
    ``identity`` = peak-local cosine to the frozen initial template (sim_to_init),
    tie-broken by quality — used by the FC challenge controller.
    """
    q = float(cand.get("quality", 0.0))
    if rank_by == "identity":
        s = float(cand.get("sim_to_init", float("nan")))
        if s != s:   # NaN sorts last
            s = float("-inf")
        return (s, q)
    return (q,)


def redetect_topk(cands: list[dict], rank_by: str, top_k: int) -> list[dict]:
    """Spatially-deduplicated top-K redetect candidates (for FC association).

    Greedy NMS: sort by rank_by, keep a candidate only if its center is farther
    than 0.5*sqrt(w*h) from every already-kept one.
    """
    if not cands:
        return []
    ordered = sorted(cands, key=lambda c: redetect_rank_key(c, rank_by), reverse=True)
    kept: list[dict] = []
    for c in ordered:
        cx, cy = c.get("center", (float("nan"), float("nan")))
        bb = c.get("bbox", [0.0, 0.0, 1.0, 1.0])
        radius = 0.5 * max(1.0, (float(bb[2]) * float(bb[3])) ** 0.5)
        if any(
            ((cx - k["center"][0]) ** 2 + (cy - k["center"][1]) ** 2) ** 0.5 <= radius
            for k in kept
        ):
            continue
        kept.append(c)
        if len(kept) >= top_k:
            break
    return kept


# ---------------------------------------------------------------------------
# Identity verification
# ---------------------------------------------------------------------------

def sim_init_at_cell(
    search_tokens: torch.Tensor,
    row: int,
    col: int,
    initial_template_embedding: torch.Tensor,
) -> float:
    """Cosine of a peak-local search embedding to the frozen initial template.

    ``search_tokens``: (256, C) — the 16×16 grid of search tokens for one crop.
    Mean-pools the 3×3 neighbourhood around grid cell ``(row, col)`` and takes
    cosine similarity to the frame-0 template embedding (shape (C,)).
    """
    emb = peak_local_embedding(search_tokens, row, col)
    if emb is None:
        return float("nan")
    return float(
        F.cosine_similarity(emb.unsqueeze(0), initial_template_embedding.unsqueeze(0)).item()
    )


def peak_local_embedding(
    search_tokens: torch.Tensor,
    row: int,
    col: int,
) -> torch.Tensor | None:
    """Mean of the 3×3 search-token neighbourhood around grid cell ``(row, col)``.

    Returns the (C,) peak-local embedding, or None when the cell is out of range.
    The PrototypeMemory / verifier in csc_lib/csc/v4 consumes this directly to
    compute sim_to_init / sim_to_recent / sim_to_distractor for a candidate.
    """
    r, c = int(row), int(col)
    idxs = [
        rr * FEAT_SZ + cc
        for rr in range(max(0, r - 1), min(FEAT_SZ, r + 2))
        for cc in range(max(0, c - 1), min(FEAT_SZ, c + 2))
    ]
    if not idxs:
        return None
    return search_tokens[idxs].mean(0)


# ---------------------------------------------------------------------------
# Generic wide-crop re-detection loop (shared by AVTrack / ORTrack adapters)
# ---------------------------------------------------------------------------

def run_redetect(
    *,
    model_forward,
    sample_target,
    to_tensor,
    hann: torch.Tensor,
    device,
    state_bbox,
    frame,
    factors=(8.0, 12.0, 16.0),
    anchor_bboxes=None,
    include_current: bool = True,
    grid_size: int = 0,
    max_candidates: int = 3,
    min_apce: float = 0.0,
    rank_by: str = "quality",
    top_k: int = 1,
    initial_template_embedding=None,
):
    """Tracker-agnostic event-driven re-detection.

    Runs the tracker's template/search forward on wide crops around a set of
    anchors (current state + caller hints + optional NxN grid), decodes the
    top score-map peaks into frame-space candidates, and returns the best one
    (``top_k==1``) or a spatially-deduplicated top-K list (``top_k>1``, for FC
    association). Side-effect-light: never mutates tracker state.

    ``model_forward(search_tensor) -> dict`` must return ``score_map`` /
    ``size_map`` / ``offset_map`` (and optionally ``backbone_feat`` for the
    per-candidate identity floor). ``sample_target(frame_rgb, bbox, factor,
    out_size)`` and ``to_tensor(patch, device)`` are the adapter's own crop
    helpers. Returns None if the tracker is uninitialised or no anchor/factor
    is usable.
    """
    import cv2
    import numpy as np
    from uav_tracker.types import BBox

    if state_bbox is None:
        return None
    _factors = tuple(float(f) for f in (factors or (8.0, 12.0, 16.0)) if float(f) > 0.0)
    if not _factors:
        return None

    H, W = frame.shape[:2]
    anchors: list = []
    if include_current:
        anchors.append(state_bbox)
    if anchor_bboxes:
        for b in anchor_bboxes:
            if b is not None and b.w > 0 and b.h > 0:
                anchors.append(b)
    if grid_size and grid_size > 1:
        ref = anchor_bboxes[0] if anchor_bboxes else state_bbox
        gw = max(1.0, float(ref.w))
        gh = max(1.0, float(ref.h))
        for gy in range(int(grid_size)):
            cyc = (gy + 0.5) * float(H) / float(grid_size)
            for gx in range(int(grid_size)):
                cxc = (gx + 0.5) * float(W) / float(grid_size)
                anchors.append(BBox(x=cxc - gw / 2.0, y=cyc - gh / 2.0, w=gw, h=gh))

    # De-duplicate near-identical anchors (current + last-good often coincide).
    uniq: list = []
    seen: set = set()
    for b in anchors:
        key = (int(round(b.x + b.w / 2.0)), int(round(b.y + b.h / 2.0)),
               int(round(b.w)), int(round(b.h)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    best: dict | None = None
    collected: list[dict] = []

    for anchor in uniq:
        for factor in _factors:
            patch, resize_factor = sample_target(rgb, anchor, factor, SEARCH_SIZE)
            x_tensor = to_tensor(patch, device)
            try:
                with torch.no_grad():
                    out = model_forward(x_tensor)
                    score_map = hann * out["score_map"]
                    # Optional per-candidate identity floor (cosine of peak-local
                    # search embedding to the frozen initial template).
                    _bf = None
                    if initial_template_embedding is not None and "backbone_feat" in out:
                        _bf = out["backbone_feat"][:, -SEARCH_SIZE:, :].squeeze(0)
                    flat = score_map.reshape(-1)
                    f_max = float(flat.max())
                    f_min = float(flat.min())
                    denom = float(((flat - f_min) ** 2).mean())
                    apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))
                    if apce < min_apce:
                        continue
                    size_map = out["size_map"]
                    offset_map = out["offset_map"]
            except Exception:
                continue

            peak_idx = select_candidate_peak_indices(
                score_map, max_candidates=max(1, int(max_candidates)))
            top_score = float(flat[peak_idx[0]]) if peak_idx else 0.0
            for rank, idx in enumerate(peak_idx):
                cand = candidate_bbox_from_peak(
                    idx=idx, score=float(flat[idx]), top_score=top_score,
                    size_map=size_map, offset_map=offset_map,
                    prev_bbox=anchor, resize_factor=resize_factor, rank=rank)
                x, y, bw, bh = cand["bbox"]
                bx = min(max(float(x), 0.0), max(0.0, float(W) - 1.0))
                by = min(max(float(y), 0.0), max(0.0, float(H) - 1.0))
                bw = max(1.0, min(float(bw), float(W) - bx))
                bh = max(1.0, min(float(bh), float(H) - by))
                score_ratio = float(cand.get("score_ratio", 0.0))
                quality = float(apce) * max(0.05, score_ratio)
                sim_init = float("nan")
                if _bf is not None and cand.get("row") is not None and cand.get("col") is not None:
                    sim_init = sim_init_at_cell(
                        _bf, int(cand["row"]), int(cand["col"]), initial_template_embedding)
                out_cand = {
                    "bbox": [bx, by, bw, bh],
                    "center": [bx + bw / 2.0, by + bh / 2.0],
                    "score": float(cand.get("score", 0.0)),
                    "score_ratio": score_ratio,
                    "rank": int(rank),
                    "factor": float(factor),
                    "apce": float(apce),
                    "quality": float(quality),
                    "sim_to_init": float(sim_init),
                }
                if best is None or redetect_rank_key(out_cand, rank_by) > redetect_rank_key(best, rank_by):
                    best = out_cand
                if top_k and top_k > 1:
                    collected.append(out_cand)

    if top_k and top_k > 1:
        return redetect_topk(collected, rank_by, int(top_k))
    return best

