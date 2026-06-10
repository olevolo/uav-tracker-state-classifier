"""CSC-v4 A3 — candidate extraction + verification.

Two public pieces:

1. ``extract_candidates(score_map, search_bbox, image_size, ...)`` — turn a raw
   16x16 SGLATrack score-map into a small set of spatially-distinct
   :class:`~csc_lib.csc.v4.v4types.Candidate` peaks (greedy NMS), with grid->pixel
   geometry mirroring ``sglatrack._select_candidate_peak_indices`` /
   ``_candidate_bbox_from_peak``.

2. ``CandidateVerifier(memory)`` — scores/accepts a candidate by combining
   identity evidence (sim_to_init / sim_to_recent high == good; sim_to_distractor
   high == bad), peak evidence (score rank, peak margin) and physical
   plausibility (motion / scale). This is THE guard that stops a re-detector or
   RELOCATE action from teleporting onto a wrong object/background (the
   person9 / car6_2 -0.5 catastrophic jumps observed in the V3 control work).
   It is deliberately CONSERVATIVE: on weak/absent evidence ``.verify`` returns
   ``False`` (do not jump).

Additive module (V3 frozen). Imports shared types from ``csc_lib.csc.v4.v4types``.
The prototype memory it consumes is built by A2 (``csc_lib/csc/v4/memory.py``);
see the ``# INTEGRATION:`` markers. To stay standalone this file only relies on
the duck-typed ``memory.sims(embedding) -> dict`` contract.
"""
from __future__ import annotations

# When run by PATH (`python csc_lib/csc/v4/verifier.py`) Python prepends this
# file's dir to sys.path, where the sibling `types.py` shadows the stdlib
# `types` module and breaks dataclasses/enum imports. Drop that entry and put
# the project root on the path so `import csc_lib...` resolves either way.
if __name__ == "__main__":  # pragma: no cover - import-path hygiene only
    import os as _os
    import sys as _sys

    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path[:] = [p for p in _sys.path if _os.path.abspath(p or ".") != _here]
    _root = _os.path.abspath(_os.path.join(_here, "..", "..", ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from csc_lib.csc.v4.v4types import Candidate


# ---------------------------------------------------------------------------
# SGLATrack score-map geometry (mirrors sglatrack.py constants). The map is a
# square grid over a square search crop, so only the side length matters here.
# ---------------------------------------------------------------------------
_FEAT_SZ = 16  # SGLATrack search grid is 16x16 (256/16 patch stride)


# A search crop / target box is passed as (x, y, w, h) in image pixels. We
# accept any 4-length sequence, a mapping with x/y/w/h, or an object exposing
# .x/.y/.w/.h (e.g. uav_tracker.types.BBox) so callers don't have to convert.
def _as_xywh(box: object) -> tuple[float, float, float, float]:
    if box is None:
        raise ValueError("box is required (xywh)")
    if hasattr(box, "x") and hasattr(box, "w"):
        return float(box.x), float(box.y), float(box.w), float(box.h)  # type: ignore[attr-defined]
    if isinstance(box, dict):
        return float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
    seq = list(box)  # type: ignore[arg-type]
    if len(seq) < 4:
        raise ValueError(f"box must have 4 values (xywh), got {seq!r}")
    return float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])


def _to_numpy_2d(score_map: object) -> np.ndarray:
    """Coerce a score-map (np array / torch tensor / nested list) to a 2-D float
    grid. Accepts the raw SGLATrack shapes (16,16) / (1,1,16,16) / (256,)."""
    arr = score_map
    # torch tensor duck-typing without importing torch (keeps smoke CPU-light).
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        n = arr.shape[0]
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError(f"flat score_map length {n} is not a perfect square")
        arr = arr.reshape(side, side)
    if arr.ndim != 2:
        raise ValueError(f"score_map must reduce to a 2-D grid, got shape {arr.shape}")
    return arr


def _select_peak_indices(
    flat: np.ndarray,
    side: int,
    k: int,
    nms_radius: int,
    min_score_ratio: float,
) -> list[int]:
    """Greedy NMS over a flattened square map — mirrors
    ``sglatrack._select_candidate_peak_indices`` (Chebyshev suppression radius,
    stop once a peak drops below ``min_score_ratio`` of the top peak)."""
    if flat.size == 0 or k <= 0:
        return []
    order = np.argsort(flat)[::-1]
    top_score = float(flat[order[0]])
    selected: list[int] = []
    for idx in order:
        idx = int(idx)
        score = float(flat[idx])
        if selected and top_score > 0.0 and score < top_score * min_score_ratio:
            break
        row, col = divmod(idx, side)
        too_close = False
        for prev_idx in selected:
            prev_row, prev_col = divmod(prev_idx, side)
            if max(abs(row - prev_row), abs(col - prev_col)) <= nms_radius:
                too_close = True
                break
        if too_close:
            continue
        selected.append(idx)
        if len(selected) >= k:
            break
    return selected


def extract_candidates(
    score_map: object,
    search_bbox: object,
    image_size: Sequence[float] | tuple[float, float],
    k: int = 5,
    nms_radius: int = 1,
    min_score_ratio: float = 0.05,
    default_size: Optional[tuple[float, float]] = None,
) -> list[Candidate]:
    """Decode a SGLATrack-style score-map into ranked candidate peaks.

    Parameters
    ----------
    score_map
        Raw response grid: (16,16), (1,1,16,16), or flat (256,). np array or
        torch tensor. NMS is applied on the squeezed 2-D grid.
    search_bbox
        The square SEARCH CROP region in image pixels — (x, y, w, h) — the same
        crop the score-map was computed over (centre = previous target centre,
        side ~= sqrt(w*h)*search_factor). Each grid peak's normalised centre is
        mapped linearly into this box: ``cx = sx + (col+0.5)/side * sw``.
        Accepts a BBox / dict / 4-seq.
    image_size
        ``(width, height)`` of the full frame — candidate centres are clamped to
        it so a relocate target never lands off-frame.
    k, nms_radius, min_score_ratio
        Peak-selection knobs (defaults mirror the tracker).
    default_size
        ``(w, h)`` pixel size to assign each candidate box. The score-map alone
        carries no size/offset regression (those are separate SGLATrack heads not
        passed here), so by default we reuse the search_bbox-implied target size:
        ``search_bbox.w / search_factor`` is unknown here, so we fall back to a
        fraction of the crop (``default_size`` if given, else 1/4 of the crop
        side, matching the typical target-to-crop ratio at search_factor=4). The
        box size is a coarse hint for plausibility only; localisation uses cx,cy.
        # APPROX: size from crop fraction, not the regressed size_map.

    Returns
    -------
    list[Candidate]
        Up to ``k`` candidates ordered by score (rank 0 == peak). ``peak_margin``
        on each is (its score - next-lower selected peak score), normalised by the
        top score; rank-0 margin is (top - 2nd)/top. ``embedding`` is left None
        (the caller / re-detector attaches per-candidate embeddings if available).
    """
    grid = _to_numpy_2d(score_map)
    side = grid.shape[0]
    if grid.shape[0] != grid.shape[1]:
        raise ValueError(f"score_map must be square, got {grid.shape}")
    flat = grid.reshape(-1)

    sx, sy, sw, sh = _as_xywh(search_bbox)
    img_w, img_h = float(image_size[0]), float(image_size[1])

    if default_size is not None:
        cand_w, cand_h = float(default_size[0]), float(default_size[1])
    else:
        # 1/4 of the crop side ~= target size at the nominal search_factor=4.
        cand_w = max(1.0, sw / 4.0)
        cand_h = max(1.0, sh / 4.0)

    peak_indices = _select_peak_indices(flat, side, k, nms_radius, min_score_ratio)
    if not peak_indices:
        return []

    top_score = float(flat[peak_indices[0]])
    selected_scores = [float(flat[i]) for i in peak_indices]

    candidates: list[Candidate] = []
    for rank, idx in enumerate(peak_indices):
        row, col = divmod(int(idx), side)
        score = float(flat[idx])
        # Grid cell centre -> normalised [0,1] within the crop (cell-centre
        # convention: +0.5 puts the peak at the middle of its patch, the
        # discrete analogue of the tracker's sub-pixel offset).
        cx_norm = (float(col) + 0.5) / side
        cy_norm = (float(row) + 0.5) / side
        cx = sx + cx_norm * sw
        cy = sy + cy_norm * sh
        # Clamp centre to frame so a relocate target is never off-frame.
        cx = min(max(cx, 0.0), max(0.0, img_w - 1.0))
        cy = min(max(cy, 0.0), max(0.0, img_h - 1.0))

        # peak_margin: gap to the NEXT selected peak (or 2nd peak for rank-0),
        # normalised by top score -> [0,1]. Higher == this peak dominates.
        if rank + 1 < len(selected_scores):
            margin_raw = score - selected_scores[rank + 1]
        elif rank == 0:
            margin_raw = score  # single peak: full margin
        else:
            margin_raw = 0.0
        peak_margin = float(margin_raw / (top_score + 1e-8)) if top_score > 0 else 0.0

        candidates.append(
            Candidate(
                cx=float(cx),
                cy=float(cy),
                w=float(cand_w),
                h=float(cand_h),
                score=score,
                rank=int(rank),
                peak_margin=peak_margin,
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class _MemoryLike(Protocol):
    """Duck-typed view of A2's PrototypeMemory used by the verifier.

    # INTEGRATION: real impl = csc_lib.csc.v4.memory.PrototypeMemory (A2). Only
    # .sims(embedding) -> {sim_to_init, sim_to_recent, sim_to_distractor} is
    # required here; empty stores should return nan for the missing keys.
    """

    def sims(self, embedding: np.ndarray) -> dict: ...


@dataclass
class VerifierConfig:
    """Thresholds for candidate scoring/verification. Conservative defaults:
    the cost of a false RELOCATE (catastrophic -0.5 IoU jump) far exceeds a
    missed recovery, so the bar to ACCEPT is high."""

    # --- evidence weights (sum need not be 1; score is renormalised) ---
    w_sim_init: float = 0.30        # cosine vs frame-0 template (identity anchor)
    w_sim_recent: float = 0.25      # cosine vs recent CC prototype
    w_sim_distractor: float = 0.20  # penalise similarity to known distractors
    w_peak: float = 0.15            # score rank / peak dominance
    w_motion: float = 0.05          # motion plausibility (if a prior is given)
    w_scale: float = 0.05           # scale plausibility (if a prior is given)

    # --- similarity calibration (cosine -> [0,1]) ---
    sim_lo: float = 0.30            # cosine at/below this maps to 0 evidence
    sim_hi: float = 0.80            # cosine at/above this maps to full evidence

    # --- distractor veto ---
    distractor_veto: float = 0.85   # sim_to_distractor above this -> hard reject

    # --- identity floor: a candidate with NO positive identity evidence and no
    # memory to check against cannot be verified on score alone (background-peak
    # guard). When memory is empty for BOTH init and recent, we require a strong
    # peak AND fall back to refusing relocate unless allow_blind=False overridden.
    min_identity: float = 0.35      # need sim-evidence (init or recent) >= this

    # --- acceptance ---
    accept_margin: float = 0.55     # default .verify threshold on the [0,1] score
    min_peak_margin: float = 0.10   # reject ambiguous peaks (top2 too close)


class CandidateVerifier:
    """Accept/reject a re-localisation :class:`Candidate` against memory.

    Usage::

        verifier = CandidateVerifier(memory)          # A2 PrototypeMemory
        for c in extract_candidates(...):
            c = verifier.annotate(c)                   # fills sim_* on the candidate
            if verifier.verify(c, motion_prior=mp):    # conservative gate
                relocate_to(c)
                break

    ``score`` returns a calibrated confidence in [0,1]; ``verify`` thresholds it
    AND applies hard vetoes (distractor match, ambiguous peak, no identity
    evidence). On any missing evidence it errs toward REJECT.
    """

    def __init__(self, memory: _MemoryLike, cfg: Optional[VerifierConfig] = None) -> None:
        # INTEGRATION: `memory` is A2's PrototypeMemory (csc_lib.csc.v4.memory).
        self.memory = memory
        self.cfg = cfg or VerifierConfig()

    # -- helpers ----------------------------------------------------------
    def _sim_to_unit(self, sim: float) -> float:
        """Map a cosine similarity (or nan) to [0,1] via the sim_lo..sim_hi ramp.
        nan (empty store / no embedding) -> 0 (no positive evidence)."""
        if sim is None or (isinstance(sim, float) and math.isnan(sim)):
            return 0.0
        c = self.cfg
        if c.sim_hi <= c.sim_lo:
            return float(max(0.0, min(1.0, sim)))
        return float(max(0.0, min(1.0, (sim - c.sim_lo) / (c.sim_hi - c.sim_lo))))

    def annotate(self, candidate: Candidate) -> Candidate:
        """Populate ``sim_to_init/recent/distractor`` on the candidate from
        memory using its ``embedding`` (no-op if embedding is None — leaves the
        nan defaults so the verifier treats it as no-identity-evidence)."""
        if candidate.embedding is not None and self.memory is not None:
            sims = self.memory.sims(np.asarray(candidate.embedding))
            candidate.sim_to_init = float(sims.get("sim_to_init", float("nan")))
            candidate.sim_to_recent = float(sims.get("sim_to_recent", float("nan")))
            candidate.sim_to_distractor = float(sims.get("sim_to_distractor", float("nan")))
        return candidate

    # -- scoring ----------------------------------------------------------
    def score(self, candidate: Candidate, motion_prior: Optional[dict] = None) -> float:
        """Calibrated verification confidence in [0,1].

        Combines (weighted, renormalised over the components we actually have
        evidence for):
          + identity to init / recent (high good)
          - identity to distractor    (high bad)
          + peak dominance (rank 0 + peak_margin)
          + motion plausibility, + scale plausibility (only if a prior is given).

        ``motion_prior`` (optional) example:
          {'cx':px,'cy':py,'max_disp':px,            # predicted centre + gate
           'w':px,'h':px,'max_scale_ratio':2.0}      # expected size + scale gate
        Missing pieces are simply dropped from the weighting (not penalised),
        EXCEPT distractor similarity which always penalises when present.
        """
        c = self.cfg
        # Use annotated sims if present; otherwise read live from memory.
        si = candidate.sim_to_init
        sr = candidate.sim_to_recent
        sd = candidate.sim_to_distractor
        if (math.isnan(si) and math.isnan(sr) and math.isnan(sd)
                and candidate.embedding is not None and self.memory is not None):
            sims = self.memory.sims(np.asarray(candidate.embedding))
            si = float(sims.get("sim_to_init", float("nan")))
            sr = float(sims.get("sim_to_recent", float("nan")))
            sd = float(sims.get("sim_to_distractor", float("nan")))

        terms: list[tuple[float, float]] = []  # (weight, value in [0,1])

        ev_init = self._sim_to_unit(si)
        ev_recent = self._sim_to_unit(sr)
        has_identity = not (math.isnan(si) and math.isnan(sr))
        if not math.isnan(si):
            terms.append((c.w_sim_init, ev_init))
        if not math.isnan(sr):
            terms.append((c.w_sim_recent, ev_recent))

        # Distractor similarity: present-and-high drags the score down.
        if not math.isnan(sd):
            ev_not_distractor = 1.0 - self._sim_to_unit(sd)
            terms.append((c.w_sim_distractor, ev_not_distractor))

        # Peak evidence: rank-0 peaks with a clear margin score high; lower-rank
        # or ambiguous peaks score low. Always available from the score-map.
        rank_factor = 1.0 if candidate.rank == 0 else max(0.0, 1.0 - 0.35 * candidate.rank)
        margin_factor = float(max(0.0, min(1.0, candidate.peak_margin / max(c.min_peak_margin * 2, 1e-6))))
        ev_peak = 0.5 * rank_factor + 0.5 * (rank_factor * margin_factor)
        terms.append((c.w_peak, ev_peak))

        # Motion plausibility (optional prior).
        ev_motion = self._motion_plausibility(candidate, motion_prior)
        if ev_motion is not None:
            candidate.motion_plausibility = float(ev_motion)
            terms.append((c.w_motion, ev_motion))

        # Scale plausibility (optional prior).
        ev_scale = self._scale_plausibility(candidate, motion_prior)
        if ev_scale is not None:
            candidate.scale_plausibility = float(ev_scale)
            terms.append((c.w_scale, ev_scale))

        wsum = sum(w for w, _ in terms)
        if wsum <= 0:
            return 0.0
        raw = sum(w * v for w, v in terms) / wsum

        # Conservative cap: with NO identity evidence at all (empty memory or no
        # embedding), peak/motion alone must not push past 0.5 — a sharp peak on
        # background looks identical to a sharp peak on target. This is the core
        # anti-teleport guard.
        if not has_identity:
            raw = min(raw, 0.5)
        return float(max(0.0, min(1.0, raw)))

    def _motion_plausibility(
        self, candidate: Candidate, motion_prior: Optional[dict]
    ) -> Optional[float]:
        if not motion_prior or "cx" not in motion_prior or "cy" not in motion_prior:
            return None
        dx = candidate.cx - float(motion_prior["cx"])
        dy = candidate.cy - float(motion_prior["cy"])
        disp = math.hypot(dx, dy)
        max_disp = float(motion_prior.get("max_disp", 0.0))
        if max_disp <= 0:
            return None
        # Linear decay: at the predicted centre -> 1.0, at the gate -> 0.0,
        # beyond the gate -> 0.0 (implausible jump).
        return float(max(0.0, 1.0 - disp / max_disp))

    def _scale_plausibility(
        self, candidate: Candidate, motion_prior: Optional[dict]
    ) -> Optional[float]:
        if not motion_prior or "w" not in motion_prior or "h" not in motion_prior:
            return None
        ref_w = float(motion_prior["w"])
        ref_h = float(motion_prior["h"])
        if ref_w <= 0 or ref_h <= 0 or candidate.w <= 0 or candidate.h <= 0:
            return None
        rw = candidate.w / ref_w
        rh = candidate.h / ref_h
        # Geometric area ratio, folded to >=1, then mapped through the gate.
        ratio = max(rw, 1.0 / rw) * max(rh, 1.0 / rh)
        ratio = math.sqrt(ratio)
        max_ratio = float(motion_prior.get("max_scale_ratio", 2.0))
        if max_ratio <= 1.0:
            return 1.0 if ratio <= 1.0 + 1e-6 else 0.0
        return float(max(0.0, 1.0 - (ratio - 1.0) / (max_ratio - 1.0)))

    # -- decision ---------------------------------------------------------
    def verify(
        self,
        candidate: Candidate,
        motion_prior: Optional[dict] = None,
        margin: Optional[float] = None,
    ) -> bool:
        """Conservative accept/reject. Returns ``True`` only if ALL hold:

          * score(candidate) >= ``margin`` (default ``cfg.accept_margin``),
          * candidate is NOT a strong distractor match (sim_to_distractor <
            ``cfg.distractor_veto``),
          * the peak is not ambiguous (``peak_margin >= cfg.min_peak_margin``),
          * there is positive identity evidence to the target
            (max(unit(sim_init), unit(sim_recent)) >= ``cfg.min_identity``),
          * if a motion prior is given, the centre is within the displacement
            gate (motion_plausibility > 0).

        On any missing/weak evidence it returns ``False`` (do not jump).
        """
        c = self.cfg
        thr = c.accept_margin if margin is None else float(margin)

        s = self.score(candidate, motion_prior=motion_prior)

        # Hard veto: matches a known distractor.
        sd = candidate.sim_to_distractor
        if not math.isnan(sd) and sd >= c.distractor_veto:
            return False

        # Ambiguous peak: a near-tie 2nd peak means high relocation risk.
        if candidate.peak_margin < c.min_peak_margin:
            return False

        # Identity floor: require positive evidence the candidate IS the target.
        # (nan sims -> 0 -> fails this gate, which is the safe default.)
        ev_identity = max(self._sim_to_unit(candidate.sim_to_init),
                          self._sim_to_unit(candidate.sim_to_recent))
        if ev_identity < c.min_identity:
            return False

        # Motion gate: an implausible jump (outside the prior's gate) is rejected.
        if motion_prior:
            mp = self._motion_plausibility(candidate, motion_prior)
            if mp is not None and mp <= 0.0:
                return False

        return s >= thr

    def best_verified(
        self,
        candidates: Sequence[Candidate],
        motion_prior: Optional[dict] = None,
        margin: Optional[float] = None,
    ) -> Optional[Candidate]:
        """Return the highest-scoring candidate that passes ``verify`` (after
        annotating sims from memory), or ``None`` if none qualify — the single
        call a controller makes before a RELOCATE/GLOBAL_SEARCH action."""
        best: Optional[Candidate] = None
        best_s = -1.0
        for cand in candidates:
            self.annotate(cand)
            if not self.verify(cand, motion_prior=motion_prior, margin=margin):
                continue
            s = self.score(cand, motion_prior=motion_prior)
            if s > best_s:
                best_s, best = s, cand
        return best


# ---------------------------------------------------------------------------
# Standalone smoke (no datasets, no A2 dependency, CPU-only).
# ---------------------------------------------------------------------------
def _build_synthetic_score_map(side: int = _FEAT_SZ, seed: int = 0) -> np.ndarray:
    """A 16x16 map with a dominant Gaussian peak + a weaker secondary peak."""
    rng = np.random.default_rng(seed)
    grid = rng.uniform(0.0, 0.05, size=(side, side))
    yy, xx = np.mgrid[0:side, 0:side]

    def _blob(cy: float, cx: float, amp: float, sig: float) -> np.ndarray:
        return amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)))

    grid += _blob(5.0, 6.0, 0.50, 1.0)   # dominant peak
    grid += _blob(11.0, 12.0, 0.18, 1.0)  # secondary (distractor-ish) peak
    return grid


class _StubMemory:
    """Tiny inline PrototypeMemory-like stub so the smoke does NOT depend on A2.

    Mirrors the A2 contract: ``sims(emb) -> {sim_to_init, sim_to_recent,
    sim_to_distractor}`` via cosine; empty stores -> nan.
    # INTEGRATION: replace with csc_lib.csc.v4.memory.PrototypeMemory.
    """

    def __init__(
        self,
        init_emb: Optional[np.ndarray] = None,
        recent_emb: Optional[np.ndarray] = None,
        distractor_emb: Optional[np.ndarray] = None,
    ) -> None:
        self.init_emb = init_emb
        self.recent_emb = recent_emb
        self.distractor_emb = distractor_emb

    @staticmethod
    def _cos(a: Optional[np.ndarray], b: np.ndarray) -> float:
        if a is None:
            return float("nan")
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))

    def sims(self, emb: np.ndarray) -> dict:
        return {
            "sim_to_init": self._cos(self.init_emb, emb),
            "sim_to_recent": self._cos(self.recent_emb, emb),
            "sim_to_distractor": self._cos(self.distractor_emb, emb),
        }


def _smoke() -> None:
    side = _FEAT_SZ
    sm = _build_synthetic_score_map(side)
    search_bbox = (100.0, 50.0, 256.0, 256.0)  # square crop in pixels
    image_size = (1280.0, 720.0)

    cands = extract_candidates(sm, search_bbox, image_size, k=5)
    assert cands, "expected at least one candidate"
    assert cands[0].rank == 0
    # Scores must be non-increasing (rank order) and centres on-frame.
    for prev, cur in zip(cands, cands[1:]):
        assert cur.score <= prev.score + 1e-9
    for c in cands:
        assert 0.0 <= c.cx <= image_size[0] - 1.0
        assert 0.0 <= c.cy <= image_size[1] - 1.0
        assert c.w > 0 and c.h > 0
    # Top peak at grid (5,6) -> normalised (~0.40, ~0.34) within the crop.
    top = cands[0]
    exp_cx = search_bbox[0] + (6 + 0.5) / side * search_bbox[2]
    exp_cy = search_bbox[1] + (5 + 0.5) / side * search_bbox[3]
    assert abs(top.cx - exp_cx) < 1e-6 and abs(top.cy - exp_cy) < 1e-6, (top.cx, exp_cx)
    print(f"extract_candidates: {len(cands)} candidates; "
          f"top score={top.score:.3f} center=({top.cx:.1f},{top.cy:.1f}) "
          f"margin={top.peak_margin:.3f}")

    # --- verifier with a GOOD target embedding (high sim_to_init/recent) -----
    target = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    distractor = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    good_mem = _StubMemory(init_emb=target, recent_emb=target, distractor_emb=distractor)
    ver = CandidateVerifier(good_mem)

    good_cand = cands[0]
    good_cand.embedding = target.copy()
    ver.annotate(good_cand)
    s_good = ver.score(good_cand)
    assert 0.0 <= s_good <= 1.0
    assert ver.verify(good_cand), f"expected verify(good)=True, score={s_good:.3f}"
    print(f"verify(target-like, rank0): score={s_good:.3f} -> ACCEPT")

    # --- candidate that looks like the DISTRACTOR -> must reject -------------
    bad_cand = Candidate(cx=top.cx, cy=top.cy, w=top.w, h=top.h,
                         score=top.score, rank=0, peak_margin=top.peak_margin,
                         embedding=distractor.copy())
    ver.annotate(bad_cand)
    s_bad = ver.score(bad_cand)
    assert not ver.verify(bad_cand), f"expected verify(distractor)=False, score={s_bad:.3f}"
    print(f"verify(distractor-like): score={s_bad:.3f} -> REJECT (distractor veto)")

    # --- NO identity evidence (empty memory) -> conservative reject ----------
    empty_ver = CandidateVerifier(_StubMemory())
    blind = Candidate(cx=top.cx, cy=top.cy, w=top.w, h=top.h,
                      score=top.score, rank=0, peak_margin=0.9)
    s_blind = empty_ver.score(blind)
    assert s_blind <= 0.5, f"blind score must be capped <=0.5, got {s_blind:.3f}"
    assert not empty_ver.verify(blind), "no-identity candidate must be rejected"
    print(f"verify(no-identity): score={s_blind:.3f} (capped) -> REJECT (anti-teleport)")

    # --- ambiguous peak (tiny margin) -> reject even with good identity ------
    ambig = Candidate(cx=top.cx, cy=top.cy, w=top.w, h=top.h,
                      score=top.score, rank=0, peak_margin=0.01,
                      embedding=target.copy())
    ver.annotate(ambig)
    assert not ver.verify(ambig), "ambiguous-peak candidate must be rejected"
    print(f"verify(ambiguous peak_margin={ambig.peak_margin}): -> REJECT")

    # --- motion prior gate: implausible far jump rejected --------------------
    near_prior = {"cx": good_cand.cx, "cy": good_cand.cy, "max_disp": 50.0,
                  "w": good_cand.w, "h": good_cand.h, "max_scale_ratio": 2.0}
    far_prior = {"cx": good_cand.cx + 5000.0, "cy": good_cand.cy + 5000.0,
                 "max_disp": 50.0}
    assert ver.verify(good_cand, motion_prior=near_prior), "near jump should pass"
    assert not ver.verify(good_cand, motion_prior=far_prior), "far jump must be rejected"
    print("motion gate: near-jump ACCEPT, far-jump REJECT")

    # --- best_verified picks the verified target over the distractor ---------
    pool = [good_cand,
            Candidate(cx=10.0, cy=10.0, w=top.w, h=top.h, score=0.05, rank=2,
                      peak_margin=0.5, embedding=distractor.copy())]
    best = ver.best_verified(pool)
    assert best is good_cand, "best_verified should return the target-like candidate"
    print(f"best_verified -> rank {best.rank} center=({best.cx:.1f},{best.cy:.1f})")

    print("\nverifier.py smoke OK")


if __name__ == "__main__":
    _smoke()
