"""CSC-v4 A8 — budgeted multi-crop SGLATrack re-detector.

Class-agnostic *emergency* re-localisation that REUSES the SGLATrack backbone
(no new model, no retraining). When the controller (A10) requests a global
search during a persistent-unknown LA / verified-FC episode, this module:

  1. builds a small crop *pyramid* around the last-good center
     (local-expanded -> 2x -> 3x -> sparse 3x3 full-frame grid),
  2. runs the SAME template-search forward used by ``SGLATracker.update`` on the
     batched crops, **read-only** (it never mutates the tracker's ``_state`` /
     template / search factor), and
  3. decodes top-k score-map peaks into :class:`csc_lib.csc.v4.v4types.Candidate`
     objects (pixel bboxes) and hands them back.

It NEVER auto-jumps: it only *proposes* candidates; the A3 ``CandidateVerifier``
(stubbed here against its interface) decides accept/reject, and A10 applies the
move causally. A :class:`RedetectBudget` rate-limits firing so re-detect cannot
dominate the FPS budget.

Additive — V3 (csc_prod) is frozen and untouched. Integration points that couple
to the real tracker forward are marked ``# INTEGRATION:``.
"""
from __future__ import annotations

import sys as _sys

# NOTE: this package contains a `types.py` (the v4 shared-types module). When this
# file is run directly (`python .../redetect.py`), the interpreter prepends THIS
# directory to sys.path[0], which makes the local `types.py` shadow the stdlib
# `types` module and breaks `import dataclasses`/`enum`/`re`. Scrub our own dir
# from sys.path *before* any stdlib import that needs `types`. Harmless when
# imported as a package (dir not on path). Uses only `import sys` (always loaded).
_here = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
for _p in (_here, ""):
    while _p in _sys.path:
        _sys.path.remove(_p)

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# torch is required for the real forward; the __main__ smoke (dummy tracker) also
# uses it for the random score map. It is in the project venv.
import torch

# Shared v4 types — single source of truth (contract rule 2).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from csc_lib.csc.v4.v4types import Candidate  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry / decode constants — mirror src/uav_tracker/trackers/sglatrack.py.
# Kept local (not imported) so this module never triggers the heavy tracker
# import chain and the __main__ smoke stays dataset/weight-free.
# ---------------------------------------------------------------------------
_SEARCH_SIZE = 256       # SGLATrack search-crop side (px)
_FEAT_SZ = 16            # 256 / 16 patch stride -> 16x16 score grid
_SEARCH_FACTOR = 4.0     # baseline crop area factor (sqrt(w*h) * factor)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
@dataclass
class RedetectBudget:
    """Rate-limit for the emergency re-detector.

    Re-detect is expensive (a batched ViT forward over several crops), so it
    must not run every frame. The budget enforces three independent caps:

    - ``max_fps``: soft target rate of fires per second (informational; the
      caller may translate it to a min frame interval at a known fps — we also
      expose :meth:`min_interval_for_fps`).
    - ``min_interval``: hard floor — at least this many frames must elapse
      between two fires (the "sparse trigger"). This is the cap the smoke checks.
    - ``max_attempts``: hard ceiling on total fires per tracking episode
      (reset via :meth:`MultiCropRedetector.reset`).
    """
    max_fps: float = 2.0
    min_interval: int = 5
    max_attempts: int = 6

    def min_interval_for_fps(self, video_fps: float) -> int:
        """Frames between fires implied by ``max_fps`` at a given video fps.

        Returns the stricter of the configured ``min_interval`` and the
        fps-derived interval, so both caps are respected.
        """
        if self.max_fps <= 0.0 or video_fps <= 0.0:
            return int(self.min_interval)
        fps_interval = int(math.ceil(video_fps / float(self.max_fps)))
        return max(int(self.min_interval), fps_interval)


# ---------------------------------------------------------------------------
# Crop pyramid
# ---------------------------------------------------------------------------
@dataclass
class CropSpec:
    """One search crop: a square region centred at (cx, cy) with side ``size`` px.

    ``factor`` is the equivalent SGLATrack search-area factor (informational),
    ``level`` is a human-readable label ('local'/'2x'/'3x'/'grid_r_c').
    """
    cx: float
    cy: float
    size: float
    factor: float
    level: str


def make_crop_pyramid(
    last_good_center: tuple[float, float],
    velocity_prior: tuple[float, float] | None,
    image_size: tuple[int, int],
    frame: np.ndarray | None = None,
    base_size: float | None = None,
) -> list[CropSpec]:
    """Build a coarse-to-fine + sparse-global crop pyramid for re-detection.

    Order (priority): local-expanded -> 2x -> 3x -> sparse 3x3 full-frame grid.
    The local/2x/3x crops are centred on the *motion-extrapolated* last-good
    center (last_good + velocity_prior) so a smoothly-moving target stays in
    frame; the 3x3 grid tiles the whole image for an abrupt jump / re-entry.

    Args:
        last_good_center: (cx, cy) of the last confidently-tracked bbox, px.
        velocity_prior: (vx, vy) per-frame center velocity, or None (treated 0).
        image_size: (W, H) of the frame in px.
        frame: optional ndarray; if given and image_size is bogus, its shape wins.
        base_size: optional base crop side (px). Defaults to a quarter of the
            short image side — a sane "local" search box when the bbox scale is
            unknown at re-detect time.

    Returns:
        list[CropSpec] (never empty for a valid image).
    """
    W, H = int(image_size[0]), int(image_size[1])
    if frame is not None:
        fh, fw = frame.shape[:2]
        # Trust the actual array shape if image_size looks unset/inconsistent.
        if W <= 0 or H <= 0:
            W, H = int(fw), int(fh)
    W = max(1, W)
    H = max(1, H)

    short_side = float(min(W, H))
    if base_size is None or base_size <= 0:
        base_size = max(8.0, short_side / 4.0)
    base_size = float(base_size)

    cx0, cy0 = float(last_good_center[0]), float(last_good_center[1])
    vx, vy = (float(velocity_prior[0]), float(velocity_prior[1])) if velocity_prior else (0.0, 0.0)
    # Motion-extrapolated center (clamped into the frame).
    pcx = min(max(cx0 + vx, 0.0), float(W))
    pcy = min(max(cy0 + vy, 0.0), float(H))

    specs: list[CropSpec] = []
    # Coarse-to-fine local crops around the predicted center.
    for mult, label in ((1.0, "local"), (2.0, "2x"), (3.0, "3x")):
        size = min(base_size * mult, float(max(W, H)))
        specs.append(CropSpec(cx=pcx, cy=pcy, size=size, factor=_SEARCH_FACTOR * mult, level=label))

    # Sparse 3x3 full-frame grid for abrupt jumps / target re-entry. Cell-sized
    # crops cover the whole image (cells slightly overlap so peaks near borders
    # are still centred).
    grid = 3
    cell = max(short_side / float(grid), base_size)
    for r in range(grid):
        gy = (r + 0.5) * float(H) / float(grid)
        for c in range(grid):
            gx = (c + 0.5) * float(W) / float(grid)
            specs.append(
                CropSpec(cx=gx, cy=gy, size=cell, factor=_SEARCH_FACTOR, level=f"grid_{r}_{c}")
            )

    return specs


# ---------------------------------------------------------------------------
# Score-map -> Candidate decode
# ---------------------------------------------------------------------------
def _decode_peaks_fallback(
    score_map: torch.Tensor,
    spec: CropSpec,
    k: int,
    nms_radius: int = 1,
    min_score_ratio: float = 0.05,
) -> list[Candidate]:
    """Local top-k peak decoder used when the A3 verifier is unavailable.

    Mirrors ``sglatrack._select_candidate_peak_indices`` (greedy NMS on the
    flattened 16x16 grid) and maps each grid peak to a *pixel center* in frame
    coordinates given the crop ``spec``. Size is unknown here (no size_map is
    passed in the verifier-less path), so we set the candidate w/h to a fraction
    of the crop side as a placeholder — the real size comes from A3 when wired.
    """
    sm = score_map.detach().reshape(-1).float()
    n = sm.numel()
    if n == 0 or k <= 0:
        return []
    feat = int(round(math.sqrt(n)))  # 16 for a 16x16 map
    feat = max(1, feat)

    order = torch.argsort(sm, descending=True)
    top_score = float(sm[order[0]])
    selected: list[int] = []
    for idx_t in order:
        idx = int(idx_t.item())
        score = float(sm[idx])
        if selected and top_score > 0.0 and score < top_score * min_score_ratio:
            break
        row, col = divmod(idx, feat)
        too_close = False
        for prev in selected:
            pr, pc = divmod(prev, feat)
            if max(abs(row - pr), abs(col - pc)) <= nms_radius:
                too_close = True
                break
        if too_close:
            continue
        selected.append(idx)
        if len(selected) >= k:
            break

    half = spec.size / 2.0
    cands: list[Candidate] = []
    for rank, idx in enumerate(selected):
        row, col = divmod(idx, feat)
        # Grid cell -> normalised [0,1] crop coords -> pixel offset within crop ->
        # absolute frame pixel (crop is centred at spec.cx/cy with side spec.size).
        u = (col + 0.5) / float(feat)
        v = (row + 0.5) / float(feat)
        cx = spec.cx - half + u * spec.size
        cy = spec.cy - half + v * spec.size
        w = max(1.0, spec.size / float(feat))   # placeholder; A3 returns real size
        h = w
        score = float(sm[idx])
        # peak_margin: gap to the NEXT selected peak, normalised by the top score
        # -> [0,1]. MIRRORS verifier.extract_candidates (verifier.py:217-223) so the
        # fallback feeds CandidateVerifier.verify() a margin on the SAME scale as the
        # A3 path; otherwise a strong secondary/background peak (raw score, unnormalised)
        # would bypass the anti-ambiguity gate (peak_margin >= min_peak_margin).
        next_score = float(sm[selected[rank + 1]]) if rank + 1 < len(selected) else 0.0
        margin = (score - next_score) / (top_score + 1e-8)
        cands.append(
            Candidate(
                cx=float(cx), cy=float(cy), w=float(w), h=float(h),
                score=score, rank=int(rank), peak_margin=float(max(0.0, margin)),
            )
        )
    return cands


# ---------------------------------------------------------------------------
# Re-detector
# ---------------------------------------------------------------------------
class MultiCropRedetector:
    """Budgeted, read-only multi-crop re-detector over the SGLATrack backbone.

    Usage (driven by the A10 controller during a global-search request)::

        rd = MultiCropRedetector(tracker, RedetectBudget())
        cands = rd.maybe_redetect(frame, last_good_bbox, velocity, frame_idx)
        if cands is not None:               # fired -> verifier judges
            best = verifier.pick(cands)     # A3 (never auto-jump)

    ``maybe_redetect`` returns ``None`` when the budget blocks this frame and a
    ``list[Candidate]`` (possibly empty) when it fires. The tracker is touched
    only through ``tracker._model`` / ``tracker._z_tensor`` / ``tracker._hann``
    and is NEVER mutated.
    """

    def __init__(
        self,
        tracker: Any,
        budget: RedetectBudget = RedetectBudget(max_fps=2, min_interval=5, max_attempts=6),
        *,
        top_k: int = 5,
        verifier: Any | None = None,
    ) -> None:
        self.tracker = tracker
        self.budget = budget
        self.top_k = int(top_k)
        # INTEGRATION: A3 CandidateVerifier exposing extract_candidates(score_map,
        # search_bbox, image_size, k=..). Absent in parallel build -> fallback decoder.
        self.verifier = verifier
        self._last_fire_frame: int | None = None
        self._attempts: int = 0

    # -- budget -------------------------------------------------------------
    def reset(self) -> None:
        """Clear fire history (call on tracker re-init / new sequence)."""
        self._last_fire_frame = None
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    def can_fire(self, frame_idx: int) -> bool:
        """True iff the budget permits a fire at ``frame_idx`` (sparse trigger)."""
        if self._attempts >= self.budget.max_attempts:
            return False
        if self._last_fire_frame is None:
            return True
        return (frame_idx - self._last_fire_frame) >= int(self.budget.min_interval)

    # -- main entry ---------------------------------------------------------
    def maybe_redetect(
        self,
        frame: np.ndarray,
        last_good: Any,
        velocity: tuple[float, float] | None,
        frame_idx: int,
    ) -> Optional[list[Candidate]]:
        """Fire the re-detector if the budget allows; else return None.

        Args:
            frame: current BGR frame (H, W, 3) ndarray.
            last_good: last confidently-tracked bbox. Either an object with
                ``.x/.y/.w/.h`` (uav_tracker ``BBox``) or an (x, y, w, h) tuple.
            velocity: (vx, vy) per-frame center velocity prior, or None.
            frame_idx: current frame index (for budget bookkeeping).

        Returns:
            ``list[Candidate]`` when it fired (verifier decides accept), or
            ``None`` when the budget blocked this frame.
        """
        if not self.can_fire(frame_idx):
            return None

        cx, cy, bw, bh = _bbox_xywh(last_good)
        center = (cx + bw / 2.0, cy + bh / 2.0)
        H, W = frame.shape[:2]
        # Use the last-good bbox scale to size the local crop (sqrt(area)*factor),
        # falling back to the image-derived default inside make_crop_pyramid.
        base_size = math.sqrt(max(1.0, bw * bh)) * _SEARCH_FACTOR if (bw > 0 and bh > 0) else None
        specs = make_crop_pyramid(
            last_good_center=center,
            velocity_prior=velocity,
            image_size=(W, H),
            frame=frame,
            base_size=base_size,
        )

        # Record the fire BEFORE running the forward so an exception mid-forward
        # still consumes the budget slot (avoids tight retry loops).
        self._last_fire_frame = frame_idx
        self._attempts += 1

        candidates = self._run_forward_on_crops(frame, specs)
        return candidates

    # -- tracker-coupled forward (read-only) --------------------------------
    def _run_forward_on_crops(
        self, frame: np.ndarray, specs: list[CropSpec]
    ) -> list[Candidate]:
        """Run the SGLATrack template-search forward on the batched crops.

        READ-ONLY: reads ``tracker._model`` / ``tracker._z_tensor`` (and
        ``tracker._hann`` if present) and never writes any tracker attribute.
        Returns decoded candidates aggregated across all crops, de-duplicated by
        pixel center and sorted by score (best first).
        """
        # INTEGRATION: tracker must expose a built model and a template tensor.
        model = getattr(self.tracker, "_model", None)              # INTEGRATION:
        z = getattr(self.tracker, "_z_tensor", None)               # INTEGRATION:
        if model is None or z is None:
            return []

        device = z.device if isinstance(z, torch.Tensor) else torch.device("cpu")  # INTEGRATION:
        hann = getattr(self.tracker, "_hann", None)                # INTEGRATION: optional Hann window

        # Build a batched search tensor: one crop per pyramid spec. We crop with
        # mean-padding (mirrors _sample_target) and resize to the search size.
        search_batch, resize_factors = _build_search_batch(frame, specs, device)
        if search_batch is None:
            return []

        # Expand the (single) template to the crop batch so the forward is batched.
        z_batch = z
        if isinstance(z, torch.Tensor) and z.shape[0] == 1 and search_batch.shape[0] > 1:
            z_batch = z.expand(search_batch.shape[0], *z.shape[1:])  # view, no copy/mutation

        candidates: list[Candidate] = []
        with torch.no_grad():                                       # INTEGRATION: read-only forward
            # INTEGRATION: SAME signature as SGLATracker.update()'s self._model(...)
            out = model(
                template=z_batch,
                search=search_batch,
                ce_template_mask=None,
                force_layer_idx=-1,
            )
            score_maps = out["score_map"]                          # INTEGRATION: (B,1,16,16)
            size_map = out.get("size_map")                         # INTEGRATION:
            offset_map = out.get("offset_map")                     # INTEGRATION:

            B = score_maps.shape[0]
            for b in range(B):
                sm_b = score_maps[b: b + 1]                        # (1,1,16,16)
                if hann is not None and isinstance(hann, torch.Tensor):
                    try:
                        sm_b = hann * sm_b                         # INTEGRATION: post-Hann (match update)
                    except RuntimeError:
                        pass  # shape mismatch on a stub -> use raw map
                spec = specs[b]

                if self.verifier is not None and hasattr(self.verifier, "extract_candidates"):
                    # INTEGRATION: A3 verifier.extract_candidates(score_map, search_bbox,
                    # image_size, k) -> list[Candidate] with real size_map decode.
                    search_bbox = (
                        spec.cx - spec.size / 2.0, spec.cy - spec.size / 2.0,
                        spec.size, spec.size,
                    )
                    cands_b = self.verifier.extract_candidates(
                        sm_b.squeeze().cpu().numpy(),
                        search_bbox,
                        (frame.shape[1], frame.shape[0]),
                        k=self.top_k,
                    )
                else:
                    cands_b = _decode_peaks_fallback(sm_b, spec, k=self.top_k)

                for c in cands_b:
                    candidates.append(c)

        # NOTE: tracker state is untouched here. (No snapshot/restore needed —
        # we never assigned to tracker._state/_z_tensor/_search_factor.)
        return _dedup_candidates(candidates)


# ---------------------------------------------------------------------------
# Helpers (pure, tracker-free)
# ---------------------------------------------------------------------------
def _bbox_xywh(bbox: Any) -> tuple[float, float, float, float]:
    """Coerce a BBox-like object or (x,y,w,h) sequence to a float 4-tuple."""
    if hasattr(bbox, "x") and hasattr(bbox, "w"):
        return float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)
    x, y, w, h = bbox
    return float(x), float(y), float(w), float(h)


def _crop_mean_padded(frame: np.ndarray, cx: float, cy: float, size: float) -> np.ndarray:
    """Mean-padded square crop centred at (cx, cy) — mirrors _sample_target padding."""
    H, W = frame.shape[:2]
    crop_sz = max(1, int(round(size)))
    x1 = int(round(cx - crop_sz / 2.0))
    y1 = int(round(cy - crop_sz / 2.0))
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz
    x1p = max(0, -x1)
    y1p = max(0, -y1)
    x2p = max(x2 - W, 0)
    y2p = max(y2 - H, 0)
    # Explicit end indices. The old `(x2 - x2p) or None` idiom was buggy: when a
    # crop's right/bottom edge lands EXACTLY on the frame's left/top boundary,
    # `x2 - x2p == 0` and `0 or None -> None`, which slices to the END of the
    # array (a malformed, full-row crop). Compute positive end indices directly
    # and route any non-positive width/height to the flat mean-tile branch.
    xs = x1 + x1p
    ys = y1 + y1p
    xe = x2 - x2p
    ye = y2 - y2p
    if xe - xs <= 0 or ye - ys <= 0:
        # Fully out of frame (or zero-extent crop) -> a flat mean tile.
        mean_val = frame.reshape(-1, frame.shape[2]).mean(0) if frame.ndim == 3 else frame.mean()
        return np.full((crop_sz, crop_sz, frame.shape[2]) if frame.ndim == 3 else (crop_sz, crop_sz),
                       mean_val, dtype=frame.dtype)
    sub = frame[ys: ye, xs: xe]
    if sub.size == 0:
        # Fully out of frame -> a flat mean tile.
        mean_val = frame.reshape(-1, frame.shape[2]).mean(0) if frame.ndim == 3 else frame.mean()
        return np.full((crop_sz, crop_sz, frame.shape[2]) if frame.ndim == 3 else (crop_sz, crop_sz),
                       mean_val, dtype=frame.dtype)
    if x1p or x2p or y1p or y2p:
        # Pad with the frame mean (cv2-free so the smoke needs no cv2).
        pad_w = (
            (y1p, y2p), (x1p, x2p), (0, 0)
        ) if frame.ndim == 3 else ((y1p, y2p), (x1p, x2p))
        if frame.ndim == 3:
            mean_val = frame.reshape(-1, frame.shape[2]).mean(0)
            sub = np.stack([
                np.pad(sub[..., c], ((y1p, y2p), (x1p, x2p)),
                       mode="constant", constant_values=float(mean_val[c]))
                for c in range(frame.shape[2])
            ], axis=-1).astype(frame.dtype)
        else:
            sub = np.pad(sub, pad_w, mode="constant", constant_values=float(frame.mean()))
    return sub


def _build_search_batch(
    frame: np.ndarray, specs: list[CropSpec], device: "torch.device"
) -> tuple[Optional[torch.Tensor], list[float]]:
    """Crop+resize every spec to (3,256,256) and stack into a (B,3,256,256) tensor.

    Returns (batch_tensor | None, resize_factors). ImageNet normalisation matches
    SGLATrack (_to_tensor): the tracker forward expects normalised RGB. We accept
    a BGR ndarray (OpenCV order) and convert to RGB without cv2.
    """
    if not specs:
        return None, []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # BGR (OpenCV) -> RGB. If grayscale, broadcast to 3 channels.
    if frame.ndim == 3 and frame.shape[2] == 3:
        rgb = frame[:, :, ::-1]
    elif frame.ndim == 2:
        rgb = np.repeat(frame[:, :, None], 3, axis=2)
    else:
        rgb = frame[..., :3]

    tensors: list[torch.Tensor] = []
    resize_factors: list[float] = []
    for spec in specs:
        crop = _crop_mean_padded(np.ascontiguousarray(rgb), spec.cx, spec.cy, spec.size)
        # cv2-free resize via torch interpolate (keeps the smoke dependency-light).
        t = torch.from_numpy(np.ascontiguousarray(crop).astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)  # (1,3,h,w)
        t = torch.nn.functional.interpolate(
            t, size=(_SEARCH_SIZE, _SEARCH_SIZE), mode="bilinear", align_corners=False
        )
        t = (t - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        tensors.append(t)
        resize_factors.append(_SEARCH_SIZE / float(max(1, int(round(spec.size)))))

    batch = torch.cat(tensors, dim=0).to(device)
    return batch, resize_factors


def _dedup_candidates(cands: list[Candidate], radius: float = 4.0) -> list[Candidate]:
    """Sort by score (desc) and drop near-duplicate centers (within ``radius`` px)."""
    ordered = sorted(cands, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    for c in ordered:
        dup = False
        for k in kept:
            if (c.cx - k.cx) ** 2 + (c.cy - k.cy) ** 2 <= radius * radius:
                dup = True
                break
        if not dup:
            kept.append(c)
    # Re-rank after dedup AND refresh peak_margin to the NEW global ordering: the
    # per-crop margin computed at decode time is stale relative to the merged,
    # cross-crop order (a candidate's old "next peak" may now sit in another crop
    # or be gone). Recompute each margin as (score - next_global_score)/(top+1e-8),
    # the SAME [0,1] normalisation as the decode path / verifier.extract_candidates,
    # so CandidateVerifier.verify()'s anti-ambiguity gate sees a consistent scale.
    top_score = float(kept[0].score) if kept else 0.0
    for i, c in enumerate(kept):
        c.rank = i
        next_score = float(kept[i + 1].score) if i + 1 < len(kept) else 0.0
        margin = (float(c.score) - next_score) / (top_score + 1e-8)
        c.peak_margin = float(max(0.0, margin))
    return kept


# ---------------------------------------------------------------------------
# __main__ smoke — DUMMY tracker stub, no datasets / weights / cv2 / real model.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as _np

    class _DummyModel:
        """Stand-in for SGLATracker._model: returns a random 16x16 score map batch.

        Matches the dict contract consumed by _run_forward_on_crops:
        score_map (B,1,16,16), size_map (B,2,16,16), offset_map (B,2,16,16).
        """

        def __call__(self, template, search, ce_template_mask=None, force_layer_idx=-1):
            B = search.shape[0]
            g = _FEAT_SZ
            return {
                "score_map": torch.rand(B, 1, g, g),
                "size_map": torch.rand(B, 2, g, g) * 0.2,
                "offset_map": torch.rand(B, 2, g, g),
            }

    class _DummyTracker:
        """Object exposing _model + _z_tensor (read-only) like SGLATracker."""

        def __init__(self) -> None:
            self._model = _DummyModel()
            # 1-batch template (B,3,128,128); the re-detector expands it to the crops.
            self._z_tensor = torch.rand(1, 3, 128, 128)
            # No _hann attr -> raw maps are used (covers the optional-Hann path).

    class _DummyBBox:
        def __init__(self, x, y, w, h):
            self.x, self.y, self.w, self.h = x, y, w, h

    print("== A8 redetect smoke ==")

    # 1) make_crop_pyramid: ordering + sparse grid.
    specs = make_crop_pyramid(
        last_good_center=(320.0, 240.0),
        velocity_prior=(5.0, -3.0),
        image_size=(640, 480),
    )
    levels = [s.level for s in specs]
    assert levels[:3] == ["local", "2x", "3x"], levels
    assert sum(1 for l in levels if l.startswith("grid_")) == 9, levels
    assert all(0.0 <= s.cx <= 640.0 and 0.0 <= s.cy <= 480.0 for s in specs)
    print(f"  pyramid: {len(specs)} crops, levels[:3]={levels[:3]}, grid={sum(1 for l in levels if l.startswith('grid_'))}")

    # 2) Budget gating: a 2nd fire is BLOCKED before min_interval elapses.
    budget = RedetectBudget(max_fps=2, min_interval=5, max_attempts=6)
    rd = MultiCropRedetector(_DummyTracker(), budget)
    frame = (_np.random.rand(480, 640, 3) * 255).astype(_np.uint8)  # BGR-ish uint8
    last_good = _DummyBBox(300.0, 220.0, 40.0, 40.0)

    c0 = rd.maybe_redetect(frame, last_good, velocity=(5.0, -3.0), frame_idx=10)
    assert c0 is not None, "first call should FIRE"
    assert isinstance(c0, list), type(c0)
    assert all(isinstance(c, Candidate) for c in c0), "fire must return list[Candidate]"
    print(f"  fire@10 -> {len(c0)} Candidate(s); top score={c0[0].score:.3f}" if c0 else "  fire@10 -> 0 (empty)")

    # Within min_interval (frames 11..14): BLOCKED -> None.
    for fi in (11, 12, 13, 14):
        assert rd.maybe_redetect(frame, last_good, velocity=None, frame_idx=fi) is None, \
            f"frame {fi} must be blocked by min_interval"
    assert rd.attempts == 1, rd.attempts
    print(f"  frames 11-14 blocked (min_interval={budget.min_interval}); attempts={rd.attempts}")

    # At frame 15 (>= min_interval since 10): allowed again.
    c1 = rd.maybe_redetect(frame, last_good, velocity=None, frame_idx=15)
    assert c1 is not None, "frame 15 should FIRE (interval elapsed)"
    assert rd.attempts == 2, rd.attempts
    print(f"  fire@15 -> {len(c1)} Candidate(s); attempts={rd.attempts}")

    # 3) max_attempts ceiling: keep firing past the cap -> eventually blocked.
    rd2 = MultiCropRedetector(_DummyTracker(), RedetectBudget(min_interval=1, max_attempts=3))
    fires = 0
    for fi in range(0, 40, 2):  # spaced by 2 >= min_interval(1)
        if rd2.maybe_redetect(frame, last_good, velocity=None, frame_idx=fi) is not None:
            fires += 1
    assert fires == 3, fires
    assert not rd2.can_fire(100), "must be blocked once max_attempts reached"
    print(f"  max_attempts cap honoured: {fires} fires (cap=3)")

    # 4) fps-derived interval helper.
    assert RedetectBudget(max_fps=2, min_interval=5).min_interval_for_fps(30) == 15
    assert RedetectBudget(max_fps=2, min_interval=20).min_interval_for_fps(30) == 20
    print("  min_interval_for_fps OK")

    # 5) reset clears history.
    rd.reset()
    assert rd.attempts == 0 and rd.can_fire(0)
    print("  reset OK")

    print("ALL ASSERTS PASSED")
