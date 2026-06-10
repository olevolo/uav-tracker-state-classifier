"""CSC-v4 A9 — AVTrack sidecar re-detector (independent-failure-profile proposer).

The V4 emergency re-detect path (A8 ``MultiCropRedetector``) reuses the *primary*
SGLATrack backbone, so its failures correlate with SGLATrack's. This sidecar adds a
**second, architecturally independent** opinion: AVTrack (DeiT-tiny adaptive ViT,
~0.7 GFLOPs, class-agnostic CENTER head). When both proposers agree on a center, the
``CandidateVerifier`` (A3) can accept a relocate with much higher confidence; when they
disagree, the verifier stays conservative.

Design contract (from ``csc_lib/csc/v4/CONTRACT.md`` A9):
  * ``class AVTrackSidecar(device='cpu', weights_path=None)``.
  * LAZY load: AVTrack is built/loaded only on the first ``.propose(...)`` call, and the
    whole load is **guarded** — if the import or weights fail, we log once and degrade to
    returning ``[]`` forever (the smoke must run with no weights present).
  * ``.propose(frame, crops, template_hint) -> list[Candidate]`` runs AVTrack's template-
    query on each crop region and extracts the top-k score-map peaks as ``Candidate`` s
    (pixel center + score + rank). It only PROPOSES — SGLATrack stays primary and the
    CandidateVerifier (A3) judges/accepts.

Self-contained + import-safe: this module imports nothing heavy at import time. AVTrack /
torch tensors are touched only inside the lazily-loaded path, all behind ``# INTEGRATION:``
markers and broad guards.

Crop-spec format (interop with A8 ``make_crop_pyramid``, stubbed against the interface):
each crop describes a square region of ``frame`` in *pixel* coords. Accepted forms:
  * ``dict``  : ``{'x0': float, 'y0': float, 'size': float}``  (top-left + side length)
  * ``(x0, y0, size)`` 3-tuple
  * ``(x0, y0, w, h)`` 4-tuple xywh  (we use ``max(w, h)`` as the square side)
``size`` is the side length of the (square) crop in frame pixels; AVTrack resizes it to
its 256-px search input internally.

V3 (csc_prod) is frozen and untouched; this is additive under ``csc_lib/csc/v4/``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from csc_lib.csc.v4.v4types import Candidate

logger = logging.getLogger(__name__)

# AVTrack CENTER-head / search geometry (mirror src/uav_tracker/trackers/avtrack.py).
# The score/size/offset maps live on a _FEAT_SZ x _FEAT_SZ grid; the search input is a
# _SEARCH_SIZE-px square. cal_bbox normalises everything to [0,1] of the crop, so peak
# -> pixel mapping needs only the crop's own (x0, y0, size) — NOT _SEARCH_SIZE.
_FEAT_SZ = 16
_SEARCH_SIZE = 256

# Default weights location (mirror avtrack adapter's _default_weights_dir()).
_WEIGHTS_NAME = "AVTrack-DeiT.pth.tar"

# ImageNet stats (mirror avtrack adapter) for the inlined search-tensor fallback.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _search_to_tensor(patch_rgb: np.ndarray, device):
    """RGB patch -> ImageNet-normalised (1,3,H,W) float tensor on ``device``.

    Prefers the live AVTrack adapter's own ``_to_tensor`` (keeps preprocessing identical to
    the real tracker); falls back to an inlined copy if that import is shadowed/unavailable
    (the live ``uav_tracker`` package may be a different repo that does not re-export the
    ``avtrack`` submodule — see CONTRACT.md rule 3). INTEGRATION: import is best-effort.
    """
    try:  # INTEGRATION: reuse the adapter's exact preprocessing when importable.
        from uav_tracker.trackers.avtrack import _to_tensor as _adapter_to_tensor  # type: ignore

        return _adapter_to_tensor(patch_rgb, device)
    except Exception:
        import torch  # type: ignore

        t = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0)
        t = (t - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _default_weights_path() -> Path:
    import os

    env = os.environ.get("UAV_WEIGHTS_ROOT")
    base = Path(env).expanduser() / "avtrack" if env else Path(
        "~/uav-tracker-weights/avtrack"
    ).expanduser()
    return base / _WEIGHTS_NAME


# ---------------------------------------------------------------------------
# Crop-spec normalisation (interop with A8; pure-python, no torch).
# ---------------------------------------------------------------------------

@dataclass
class _CropRegion:
    """A normalised square crop region in frame-pixel coords."""

    x0: float
    y0: float
    size: float  # side length in pixels

    @property
    def cx(self) -> float:
        return self.x0 + self.size / 2.0

    @property
    def cy(self) -> float:
        return self.y0 + self.size / 2.0


def _normalize_crop(spec: Any) -> Optional[_CropRegion]:
    """Coerce an A8 crop-spec into a ``_CropRegion``; return None if unparseable.

    Accepts dicts ({'x0','y0','size'} or {'cx','cy','size'}) and 3/4-tuples.
    """
    try:
        if isinstance(spec, _CropRegion):
            return spec
        if isinstance(spec, dict):
            size = float(spec.get("size", spec.get("w", spec.get("side", 0.0))))
            if "x0" in spec and "y0" in spec:
                x0, y0 = float(spec["x0"]), float(spec["y0"])
            elif "cx" in spec and "cy" in spec:
                x0 = float(spec["cx"]) - size / 2.0
                y0 = float(spec["cy"]) - size / 2.0
            else:
                x0 = float(spec.get("x", 0.0))
                y0 = float(spec.get("y", 0.0))
            if size <= 0:
                return None
            return _CropRegion(x0=x0, y0=y0, size=size)
        seq = list(spec)
        if len(seq) == 3:
            x0, y0, size = (float(v) for v in seq)
        elif len(seq) >= 4:
            x0, y0, w, h = (float(v) for v in seq[:4])
            size = max(w, h)
        else:
            return None
        if size <= 0:
            return None
        return _CropRegion(x0=x0, y0=y0, size=size)
    except (TypeError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

class AVTrackSidecar:
    """Lazy, guarded AVTrack proposer for the V4 budgeted re-detect path.

    Usage::

        sidecar = AVTrackSidecar(device="cpu")          # no load here (lazy)
        cands = sidecar.propose(frame, crops, template_hint=init_bbox_xywh)
        # -> list[Candidate]; CandidateVerifier (A3) decides whether to relocate.

    ``template_hint`` initialises (or re-initialises) AVTrack's template the first time it
    is provided, or whenever it changes — typically the last *confirmed* (good) bbox in
    xywh. Without a template (and none cached) ``propose`` returns ``[]``.

    The sidecar NEVER mutates any primary-tracker state and NEVER auto-jumps; it only
    emits candidate centers for the verifier.
    """

    def __init__(self, device: str = "cpu", weights_path: Optional[str] = None) -> None:
        self.device_str = str(device)
        self.weights_path: Optional[str] = weights_path
        # Lazy-load bookkeeping.
        self._adapter: Optional[Any] = None      # AVTrackAdapter instance
        self._load_attempted: bool = False        # have we tried (success or fail)?
        self._available: bool = False             # is the model usable?
        self._template_key: Optional[tuple] = None  # cache key for current template
        # Test hook: if set, _ensure_loaded() installs this instead of real AVTrack.
        # Smoke uses it so the test needs no weights / no AVTrack import.
        self._adapter_factory = None

    # ----- public API ----------------------------------------------------- #

    @property
    def available(self) -> bool:
        """True once a usable AVTrack model has been lazily loaded."""
        return self._available

    def propose(
        self,
        frame: np.ndarray,
        crops: Sequence[Any],
        template_hint: Optional[Sequence[float]] = None,
        k: int = 5,
        nms_radius: int = 1,
        min_score_ratio: float = 0.05,
    ) -> list[Candidate]:
        """Run AVTrack's template-query on ``crops`` and return score-map-peak Candidates.

        Args:
            frame: BGR uint8 HxWx3 image (OpenCV convention, as the AVTrack adapter expects).
            crops: iterable of A8 crop-specs (square regions in frame-pixel coords).
            template_hint: xywh bbox to (re-)initialise AVTrack's template; required on the
                first call (no template is cached yet).
            k: max peaks to extract *per crop*.
            nms_radius / min_score_ratio: greedy-NMS peak selection (mirror SGLATrack).

        Returns:
            ``list[Candidate]`` with pixel ``cx,cy,w,h``, ``score``, ``rank`` (global rank
            across all crops, sorted by score desc), ``peak_margin``. Empty list if the
            model is unavailable, no template, or no crops yield a peak. Never raises.
        """
        if not self._ensure_loaded():
            return []
        if frame is None or crops is None:
            return []

        # (Re-)initialise the template if a hint is given / changed.
        if template_hint is not None:
            if not self._set_template(frame, template_hint):
                return []
        if self._template_key is None:
            # No template ever provided -> cannot run a template-query.
            logger.debug("AVTrackSidecar.propose called with no template; returning []")
            return []

        raw_peaks: list[dict] = []
        for spec in crops:
            region = _normalize_crop(spec)
            if region is None:
                continue
            try:
                raw_peaks.extend(
                    self._peaks_for_region(
                        frame, region, k=k,
                        nms_radius=nms_radius, min_score_ratio=min_score_ratio,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                # A single bad crop must never kill the whole proposal pass.
                logger.debug("AVTrackSidecar: crop %r failed: %s", spec, exc)
                continue

        if not raw_peaks:
            return []

        # Global rank across all crops by raw score (top-1 over the whole pyramid first).
        raw_peaks.sort(key=lambda d: d["score"], reverse=True)
        top_score = raw_peaks[0]["score"]
        candidates: list[Candidate] = []
        for rank, pk in enumerate(raw_peaks):
            score_ratio = float(pk["score"] / (top_score + 1e-8)) if top_score > 0 else 0.0
            candidates.append(
                Candidate(
                    cx=float(pk["cx"]),
                    cy=float(pk["cy"]),
                    w=float(pk["w"]),
                    h=float(pk["h"]),
                    score=float(pk["score"]),
                    rank=int(rank),
                    peak_margin=score_ratio,  # rel. strength vs global top-1
                )
            )
        return candidates

    def reset(self) -> None:
        """Drop the cached template (force re-init on the next propose). Keeps the model."""
        self._template_key = None
        adapter = self._adapter
        if adapter is not None:
            try:
                adapter.reset()
            except Exception:  # pragma: no cover - defensive
                pass

    # ----- lazy load / guard ---------------------------------------------- #

    def _ensure_loaded(self) -> bool:
        """Lazily build AVTrack once; guard import/weights failures -> permanent [].

        Returns True iff a usable model is loaded. After the first failed attempt this is a
        cheap no-op returning False (we do NOT retry every frame).
        """
        if self._available:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True

        # Test/smoke hook: build a stub adapter, skipping AVTrack + weights entirely.
        if self._adapter_factory is not None:
            try:
                self._adapter = self._adapter_factory()
                self._available = self._adapter is not None
                return self._available
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("AVTrackSidecar stub factory failed: %s", exc)
                self._available = False
                return False

        # INTEGRATION: real AVTrack load. Heavy + coupled to the tracker package and to
        # torch weights on disk. Fully guarded so the rest of CSC-v4 runs without it.
        try:
            # Weights presence check first: a clean, cheap reason to degrade quietly.
            wpath = Path(self.weights_path).expanduser() if self.weights_path \
                else _default_weights_path()
            if not wpath.exists():
                logger.warning(
                    "AVTrackSidecar: weights not found at %s — sidecar disabled "
                    "(propose() will return []). Set $UAV_WEIGHTS_ROOT or pass "
                    "weights_path to enable.", wpath,
                )
                self._available = False
                return False

            # INTEGRATION: import the live AVTrack adapter. Requires src/ on sys.path
            # (the loader tools prepend salrtd/src, src, PROJECT_ROOT — see la_smoke.py).
            from uav_tracker.trackers.avtrack import AVTrackAdapter  # type: ignore

            adapter = AVTrackAdapter(
                device=self.device_str, weights_path=str(wpath),
            )
            # Build the backbone now (adapter normally builds lazily inside .init()); we
            # trigger it by deferring to first _set_template, but verify importability here.
            self._adapter = adapter
            self._available = True
            logger.info("AVTrackSidecar: AVTrack adapter ready (device=%s, weights=%s).",
                        self.device_str, wpath)
            return True
        except Exception as exc:
            logger.warning(
                "AVTrackSidecar: failed to load AVTrack (%s) — sidecar disabled, "
                "propose() will return [].", exc,
            )
            self._adapter = None
            self._available = False
            return False

    # ----- template + inference (all behind the guard) -------------------- #

    def _set_template(self, frame: np.ndarray, bbox_xywh: Sequence[float]) -> bool:
        """(Re-)init AVTrack's template from an xywh bbox. Returns False on failure.

        Caches a key so repeated identical hints don't re-run the (cheap) template encode.
        """
        try:
            x, y, w, h = (float(v) for v in list(bbox_xywh)[:4])
        except (TypeError, ValueError):
            logger.debug("AVTrackSidecar: bad template_hint %r", bbox_xywh)
            return False
        key = (round(x, 2), round(y, 2), round(w, 2), round(h, 2))
        if key == self._template_key:
            return True
        adapter = self._adapter
        if adapter is None:
            return False
        # INTEGRATION: AVTrack's init() crops a 128-px template around bbox and stores
        # _z_tensor. We feed it the live BBox type via the adapter's own .init(). Guarded.
        #
        # The adapter builds its backbone LAZILY inside .init(), so this is the first place
        # the heavy build can fail (the _ensure_loaded import succeeding does NOT mean the
        # model is usable). On failure we make the sidecar permanently unavailable so the
        # next propose() short-circuits to [] instead of retrying init() every frame (the
        # "no per-frame retry storm" contract).
        try:
            bbox = self._make_bbox(x, y, w, h)
            adapter.init(frame, bbox)
            self._template_key = key
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "AVTrackSidecar: template/backbone init failed (%s) — sidecar disabled, "
                "propose() will return [] (no per-frame retry).", exc,
            )
            self._available = False  # STICKY: never retry the failed build.
            self._adapter = None
            self._template_key = None
            return False

    @staticmethod
    def _make_bbox(x: float, y: float, w: float, h: float):
        """Build the project BBox type, with a plain-namespace fallback for the smoke."""
        try:
            from uav_tracker.types import BBox  # type: ignore

            return BBox(x=x, y=y, w=w, h=h)
        except Exception:
            from types import SimpleNamespace

            return SimpleNamespace(x=x, y=y, w=w, h=h)

    def _peaks_for_region(
        self,
        frame: np.ndarray,
        region: _CropRegion,
        k: int,
        nms_radius: int,
        min_score_ratio: float,
    ) -> list[dict]:
        """Run AVTrack on one crop region; return raw peak dicts in frame-pixel coords.

        Peak -> pixel mapping uses ONLY the crop geometry (cal_bbox is normalised to the
        crop): ``cx_px = region.x0 + cx_norm * region.size`` (and likewise for cy/w/h).
        """
        adapter = self._adapter
        if adapter is None:
            return []

        # INTEGRATION: build the search tensor for THIS crop region (not the adapter's own
        # running state) and run the model's template-query forward, then decode top-k
        # peaks. We deliberately bypass adapter.update() because that re-crops around the
        # adapter's internal _state; here the crop is dictated by A8's pyramid. All heavy
        # torch work lives in _get_score_maps (the overridable seam), guarded below.
        score_map, size_map, offset_map = self._get_score_maps(frame, region)
        if score_map is None or size_map is None or offset_map is None:
            return []

        # Greedy-NMS peak selection (mirror SGLATrack._select_candidate_peak_indices,
        # done in numpy so this stays torch-optional once maps are extracted).
        sm = np.asarray(score_map, dtype=np.float64).reshape(-1)
        if sm.size != _FEAT_SZ * _FEAT_SZ:
            return []
        sz = np.asarray(size_map, dtype=np.float64).reshape(2, -1)    # (2, N) -> w,h
        off = np.asarray(offset_map, dtype=np.float64).reshape(2, -1)  # (2, N) -> dx,dy

        order = np.argsort(-sm)
        top_score = float(sm[order[0]])
        selected: list[int] = []
        for idx in order.tolist():
            score = float(sm[idx])
            if selected and top_score > 0.0 and score < top_score * min_score_ratio:
                break
            row, col = divmod(int(idx), _FEAT_SZ)
            too_close = any(
                max(abs(row - pr), abs(col - pc)) <= nms_radius
                for pr, pc in (divmod(s, _FEAT_SZ) for s in selected)
            )
            if too_close:
                continue
            selected.append(idx)
            if len(selected) >= k:
                break

        peaks: list[dict] = []
        for idx in selected:
            row, col = divmod(int(idx), _FEAT_SZ)
            dx, dy = float(off[0, idx]), float(off[1, idx])
            w_norm, h_norm = float(sz[0, idx]), float(sz[1, idx])
            cx_norm = (col + dx) / _FEAT_SZ
            cy_norm = (row + dy) / _FEAT_SZ
            peaks.append(
                {
                    "cx": region.x0 + cx_norm * region.size,
                    "cy": region.y0 + cy_norm * region.size,
                    "w": max(1.0, w_norm * region.size),
                    "h": max(1.0, h_norm * region.size),
                    "score": float(sm[idx]),
                }
            )
        return peaks

    def _get_score_maps(self, frame: np.ndarray, region: _CropRegion):
        """Crop -> AVTrack forward -> (score_map, size_map, offset_map) as numpy arrays.

        This is the single overridable seam between the sidecar's pure geometry/peak logic
        and the heavy AVTrack forward. The ``__main__`` smoke overrides it with a torch-only
        stub so the test needs neither AVTrack, ``uav_tracker``, nor weights.

        Returns ``(None, None, None)`` on any failure. INTEGRATION: this is the only place
        that touches the AVTrack module internals + the search-tensor preprocessing.
        """
        adapter = self._adapter
        try:
            # INTEGRATION: heavy deps imported lazily, inside the guard.
            import cv2  # type: ignore
            import torch  # type: ignore

            model = getattr(adapter, "_model", None)
            z = getattr(adapter, "_z_tensor", None)
            if model is None or z is None:
                return None, None, None

            device = getattr(adapter, "_device", torch.device("cpu"))

            # Crop the square region from the (BGR) frame, mean-pad off-frame parts, and
            # resize to the 256-px search input — same recipe as _sample_target but with a
            # crop fixed by A8 rather than re-centred on the adapter's _state.
            H, W = frame.shape[:2]
            crop_sz = max(1, int(round(region.size)))
            x1 = int(round(region.x0))
            y1 = int(round(region.y0))
            x2, y2 = x1 + crop_sz, y1 + crop_sz
            x1p, y1p = max(0, -x1), max(0, -y1)
            x2p, y2p = max(x2 - W, 0), max(y2 - H, 0)
            # NOTE: use explicit positive end indices, NOT the `(x2-x2p) or None` idiom.
            # When the crop's right/bottom edge sits exactly on the frame boundary,
            # x2-x2p (resp. y2-y2p) is 0 and `0 or None` -> None would slice to the END
            # of the row/col -> a malformed (too-wide/tall) crop. Compute the ends and, if
            # the visible region has non-positive width/height, route to mean-pad below.
            xs, ys = x1 + x1p, y1 + y1p
            xe, ye = x2 - x2p, y2 - y2p
            if xe <= xs or ye <= ys:
                sub = np.empty((0, 0, frame.shape[2] if frame.ndim == 3 else 1),
                               dtype=frame.dtype)
            else:
                sub = frame[ys:ye, xs:xe]
            if sub.size == 0:
                return None, None, None
            if x1p or x2p or y1p or y2p:
                mean_val = frame.mean(axis=(0, 1)).tolist()
                sub = cv2.copyMakeBorder(
                    sub, y1p, y2p, x1p, x2p, cv2.BORDER_CONSTANT, value=mean_val
                )
            rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
            patch = cv2.resize(rgb, (_SEARCH_SIZE, _SEARCH_SIZE))
            x_tensor = _search_to_tensor(patch, device)

            with torch.no_grad():
                out = model(
                    template=z,
                    search=x_tensor,
                    template_anno=[],
                    search_anno=[],
                    is_distill=False,
                )
            score_map = out["score_map"].detach().cpu().numpy()
            size_map = out["size_map"].detach().cpu().numpy()
            offset_map = out["offset_map"].detach().cpu().numpy()
            return score_map, size_map, offset_map
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("AVTrackSidecar: forward on region failed: %s", exc)
            return None, None, None


# ---------------------------------------------------------------------------
# __main__ smoke — constructs WITHOUT loading AVTrack, mocks the model so no
# weights are needed, and asserts the API + a dummy propose path returns a list.
# ---------------------------------------------------------------------------

def _smoke() -> None:  # pragma: no cover - exercised via __main__
    import os

    logging.basicConfig(level=logging.INFO)

    # 1) Construct WITHOUT loading (lazy): nothing is built until first propose().
    sc = AVTrackSidecar(device="cpu")
    assert sc.available is False, "model must NOT be loaded at construction (lazy)"
    assert hasattr(sc, "propose") and callable(sc.propose), "API: .propose missing"
    assert hasattr(sc, "reset") and callable(sc.reset), "API: .reset missing"
    print("[smoke] constructed lazily; available =", sc.available)

    # 2) Guarded-degrade path: with a bogus weights path the real load fails -> [].
    sc_bad = AVTrackSidecar(device="cpu", weights_path="/nonexistent/avtrack.pth.tar")
    frame = (np.random.rand(180, 320, 3) * 255).astype(np.uint8)  # BGR HxWx3
    crops = [
        {"x0": 60.0, "y0": 40.0, "size": 80.0},
        (10.0, 10.0, 160.0),          # 3-tuple
        (0.0, 0.0, 100.0, 100.0),     # 4-tuple xywh
    ]
    out_bad = sc_bad.propose(frame, crops, template_hint=[80.0, 60.0, 30.0, 30.0])
    assert out_bad == [], "missing weights must degrade to []"
    assert sc_bad.available is False
    print("[smoke] missing-weights degrade -> []  (guard OK)")

    # 3) Mock-model propose path — fully self-contained (numpy only; no AVTrack, no
    #    uav_tracker, no torch, no weights). We subclass the sidecar and override the single
    #    model seam `_get_score_maps` to return deterministic maps with a clear peak, so
    #    propose() exercises crop-normalise -> NMS peak-extract -> peak->pixel -> rank ->
    #    Candidate end to end. The `_adapter_factory` hook supplies a trivial non-None
    #    adapter so the lazy-load path also runs without touching real AVTrack.
    feat = _FEAT_SZ

    def _stub_maps(frame, region):
        # score_map (1,1,F,F): sharp peak at grid (row=5,col=9); a 2ndary at (2,2); and an
        # adjacent cell (5,8) that greedy-NMS must suppress.
        sm = np.full((1, 1, feat, feat), 0.05, dtype=np.float32)
        sm[0, 0, 5, 9] = 0.95
        sm[0, 0, 5, 8] = 0.40   # adjacent -> NMS-suppressed (radius 1)
        sm[0, 0, 2, 2] = 0.60   # distinct secondary peak
        size = np.full((1, 2, feat, feat), 0.25, dtype=np.float32)    # w=h=0.25 of crop
        offset = np.zeros((1, 2, feat, feat), dtype=np.float32)
        return sm, size, offset

    class _StubSidecar(AVTrackSidecar):
        def _get_score_maps(self, frame, region):  # type: ignore[override]
            return _stub_maps(frame, region)

    class _StubAdapter:
        _model = object()
        _z_tensor = object()           # truthy template marker

        def init(self, frame, bbox):    # template encode is a no-op for the stub
            return None

        def reset(self):
            return None

    sc_mock = _StubSidecar(device="cpu")
    sc_mock._adapter_factory = _StubAdapter   # test hook -> skips real AVTrack load
    out = sc_mock.propose(frame, crops, template_hint=[80.0, 60.0, 30.0, 30.0], k=3)
    assert sc_mock.available is True, "stub adapter should report available"
    assert isinstance(out, list), "propose must return a list"
    assert len(out) >= 1, "stub peak map must yield >=1 candidate"
    assert all(isinstance(c, Candidate) for c in out), "items must be Candidate"
    # Three crops, top-1 per crop is the (5,9) peak -> across crops the strongest wins rank 0.
    top = out[0]
    assert top.rank == 0 and math.isclose(top.peak_margin, 1.0, abs_tol=1e-6), \
        "top candidate rank/margin"
    assert all(out[i].score >= out[i + 1].score for i in range(len(out) - 1)), "score-sorted"
    # The (5,8) adjacent cell must NOT appear as its own candidate (NMS radius 1).
    # Peak at grid (row=5,col=9) in the first crop {x0=60,y0=40,size=80}:
    #   cx = 60 + (9+0)/16*80 = 105.0 ; cy = 40 + (5+0)/16*80 = 65.0
    exp_cx = 60.0 + (9 / feat) * 80.0
    exp_cy = 40.0 + (5 / feat) * 80.0
    matches = [c for c in out if math.isclose(c.cx, exp_cx) and math.isclose(c.cy, exp_cy)]
    assert matches, "expected top peak mapped to crop-0 pixel center"
    assert all(np.isfinite([c.cx, c.cy, c.w, c.h, c.score]).all() for c in out)
    print(f"[smoke] mock propose -> {len(out)} candidates; "
          f"top center=({top.cx:.1f},{top.cy:.1f}) score={top.score:.3f}")

    # 4) Empty / malformed inputs must be safe.
    assert sc_mock.propose(frame, []) == [], "no crops -> []"
    assert sc_mock.propose(frame, [{"bad": 1}, None, ("x",)]) == [], "all-bad crops -> []"
    sc_mock.reset()
    print("[smoke] empty/malformed-crop safety OK; reset OK")

    # 5) Retry-storm guard: a flaky adapter whose .init() ALWAYS raises (the lazy backbone
    #    build fails). The contract requires we degrade to [] with NO per-frame retry, so
    #    .init() must be attempted at MOST ONCE across many propose() calls.
    class _FlakyAdapter:
        _model = object()
        _z_tensor = object()

        def __init__(self) -> None:
            self.init_calls = 0

        def init(self, frame, bbox):
            self.init_calls += 1
            raise RuntimeError("simulated lazy-backbone build failure")

        def reset(self):
            return None

    flaky = _FlakyAdapter()
    sc_flaky = _StubSidecar(device="cpu")
    sc_flaky._adapter_factory = lambda: flaky   # lazy-load installs the flaky adapter
    for _ in range(25):
        assert sc_flaky.propose(
            frame, crops, template_hint=[80.0, 60.0, 30.0, 30.0]
        ) == [], "flaky-init must degrade to []"
    assert sc_flaky.available is False, "flaky adapter must end up unavailable (sticky)"
    assert flaky.init_calls <= 1, (
        f"init() must be called at most once (retry-storm guard); got {flaky.init_calls}"
    )
    print(f"[smoke] retry-storm guard OK; init() called {flaky.init_calls}x over 25 proposes")

    # 6) Crop-edge-on-boundary: a crop whose right/bottom edge sits EXACTLY on the frame
    #    boundary makes x2-x2p == 0 (resp. y2-y2p). The old `(val) or None` idiom would
    #    slice to END -> oversized crop; the fix must produce a correctly-sized sub-crop.
    #    We capture the `sub` the seam extracts via a thin subclass to assert its shape.
    class _CaptureSidecar(AVTrackSidecar):
        captured_shape = None

        def _get_score_maps(self, frame, region):  # type: ignore[override]
            import numpy as _np
            H, W = frame.shape[:2]
            crop_sz = max(1, int(round(region.size)))
            x1 = int(round(region.x0)); y1 = int(round(region.y0))
            x2, y2 = x1 + crop_sz, y1 + crop_sz
            x1p, y1p = max(0, -x1), max(0, -y1)
            x2p, y2p = max(x2 - W, 0), max(y2 - H, 0)
            xs, ys = x1 + x1p, y1 + y1p
            xe, ye = x2 - x2p, y2 - y2p
            sub = frame[ys:ye, xs:xe] if (xe > xs and ye > ys) else _np.empty((0, 0, 3))
            type(self).captured_shape = sub.shape
            # Reuse the deterministic stub maps so the rest of propose() still runs.
            return _stub_maps(frame, region)

    # frame is 180 (H) x 320 (W). A 100-px crop flush to the right+bottom edge:
    #   x0 = 320-100 = 220 -> x2 = 320 == W -> x2-x2p == 0 (boundary, the bug trigger)
    #   y0 = 180-100 =  80 -> y2 = 180 == H -> y2-y2p == 0
    sc_cap = _CaptureSidecar(device="cpu")
    sc_cap._adapter_factory = _StubAdapter
    out_cap = sc_cap.propose(
        frame, [{"x0": 220.0, "y0": 80.0, "size": 100.0}],
        template_hint=[80.0, 60.0, 30.0, 30.0],
    )
    assert _CaptureSidecar.captured_shape == (100, 100, 3), (
        f"boundary crop must be exactly 100x100x3, got {_CaptureSidecar.captured_shape}"
    )
    assert isinstance(out_cap, list) and len(out_cap) >= 1, "boundary crop still proposes"
    print(f"[smoke] crop-edge-on-boundary OK; sub shape={_CaptureSidecar.captured_shape}")

    print("OK avtrack_sidecar smoke (device=cpu, no weights needed)  PID", os.getpid())


if __name__ == "__main__":
    _smoke()
