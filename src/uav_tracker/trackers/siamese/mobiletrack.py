"""MobileTrack — IET Image Processing 2022 deep tracker (UAV-tuned SiamBAN).

Reference
---------
Xue, F. et al. (2022). "MobileTrack: Lightweight mobile network for UAV
single object tracking." *IET Image Processing* 16, 3300-3313.
DOI: 10.1049/ipr2.12553

Architecture
------------
* MobileNetV2 backbone, width_mult=1.4, used_layers=[3,5,7] (8 sequential
  modules named layer0..layer7; outputs at layers 3/5/7 have 44/134/448
  channels at stride 8/8/8 with dilated convolutions).
* AdjustAllLayer neck: 3 x 1x1-conv+BN adjust layers
  (44->256, 134->256, 448->256), centred 7x7 crop for template branch.
* MultiBAN head: 3-scale DepthwiseBAN branches (box2/box3/box4),
  weighted average of cls+loc outputs.

Repository / weights
--------------------
Requires the SiamBAN-derived ``MobileTrack`` source repo on disk. The
adapter reads ``$SIAMBAN_REPO`` (default ``~/projects/MobileTrack``) at
construction time and inserts it on ``sys.path`` so ``siamban.models.*``
imports work. The repo is **not** modified.

Weights default to ``$UAV_WEIGHTS_ROOT/mobiletrack/MobileTrack-UAV123-DTB70.pth``;
if missing, the adapter falls back to random init and emits a
``RuntimeWarning`` (so unit tests / smoke tests run without the file).
The shipped per-tracker config (``configs/trackers/mobiletrack.yaml``)
points at the existing checkpoint in the sibling project so no symlink
or copy is required.

Hyper-parameters
----------------
Best values from Optuna sweep (UAV123) are exposed as constructor args.
The defaults below match the upstream config.yaml TRACK section.

Telemetry
---------
``TrackState`` is populated with:
  * ``confidence`` — best-candidate score (post-window) clipped to [0, 1]
  * ``apce`` — Average Peak-to-Correlation Energy on the chosen scale's cls map
  * ``psr`` — Peak-to-Sidelobe Ratio
  * ``response_entropy`` — L1-normalised Shannon entropy of the cls map
  * ``score_map_stats`` — top-1 / top-2, peak margin, entropy, response peak
  * ``aux.raw`` — score_max/min/mean/std and best score-map index/penalty for CSC

Bbox is clamped to frame dims before being returned (project rule —
``feedback_tracker_output_clamping``).

Device compat
-------------
The upstream SiamBAN codebase historically calls ``.cuda()`` and reads
``cfg.CUDA``. We patch ``cfg.CUDA = False`` at construction time so model
construction is device-agnostic; everything is then moved to ``self._device``.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, FrameContext, TrackState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MobileTrack repo path resolution
# ---------------------------------------------------------------------------

_env_repo = os.environ.get("SIAMBAN_REPO", "").strip()
_MOBILETRACK_REPO = (
    Path(_env_repo).expanduser()
    if _env_repo
    else Path("~/projects/MobileTrack").expanduser()
)

# ---------------------------------------------------------------------------
# Weight file resolution
# ---------------------------------------------------------------------------

_WEIGHTS_SUBDIR = "mobiletrack"
_WEIGHTS_FILENAME = "MobileTrack-UAV123-DTB70.pth"

# ---------------------------------------------------------------------------
# Architecture hyperparameters (from experiments/siamban_mobilev2_l234/config.yaml)
# ---------------------------------------------------------------------------

_BACKBONE_KWARGS: dict[str, Any] = {
    "used_layers": [3, 5, 7],
    "width_mult": 1.4,
}

_NECK_KWARGS: dict[str, Any] = {
    "in_channels": [44, 134, 448],
    "out_channels": [256, 256, 256],
}

_HEAD_KWARGS: dict[str, Any] = {
    "in_channels": [256, 256, 256],
    "cls_out_channels": 2,
    "weighted": True,
}

# Tracker inference constants (from config.yaml TRACK section)
_EXEMPLAR_SIZE: int = 127
_INSTANCE_SIZE: int = 255
_BASE_SIZE: int = 8
_CONTEXT_AMOUNT: float = 0.5
_STRIDE: int = 8
_SCORE_SIZE: int = (_INSTANCE_SIZE - _EXEMPLAR_SIZE) // _STRIDE + 1 + _BASE_SIZE  # = 25

# Defaults from upstream config.yaml. Optuna best (window_influence=0.061)
# matches the upstream default; penalty_k / scale_lr below also match.
_WINDOW_INFLUENCE: float = 0.061
_PENALTY_K: float = 0.130
_SCALE_LR: float = 0.782

# Coarse FLOPs estimate when thop measurement fails (MobileNetV2 ~1.7 GFLOPs
# template + ~1.7 GFLOPs search + small neck/head ≈ 3.5 GFLOPs aggregate).
_FLOPS_FALLBACK: float = 3.5e9


# ---------------------------------------------------------------------------
# yacs shim — registers a minimal yacs.config.CfgNode if the real package is
# missing. ``yacs`` is a small pure-Python config lib; if ``import yacs`` works
# at runtime we use it, otherwise this shim keeps siamban's import path alive.
# ---------------------------------------------------------------------------


def _inject_yacs_shim() -> None:
    """Register a minimal yacs.config.CfgNode shim so siamban imports work."""
    import types

    class _CfgNode(dict):
        def __init__(self, *args, new_allowed: bool = False, **kwargs):  # noqa: ARG002
            super().__init__(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name) from None

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

    yacs_mod = types.ModuleType("yacs")
    yacs_config_mod = types.ModuleType("yacs.config")
    yacs_config_mod.CfgNode = _CfgNode  # type: ignore[attr-defined]
    sys.modules.setdefault("yacs", yacs_mod)
    sys.modules.setdefault("yacs.config", yacs_config_mod)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_mobiletrack_model(repo: Path) -> "torch.nn.Module":
    """Import siamban internals from *repo* and build the MobileTrack model.

    Returns a ``torch.nn.Module`` exposing ``template(z)`` and ``track(x)``
    methods compatible with the inference loop. Raises on any import error;
    the caller catches and falls back to random-init mode.
    """
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        import yacs  # noqa: F401
    except ImportError:
        _inject_yacs_shim()

    # Patch cfg so upstream code that checks cfg.CUDA doesn't crash on CPU/MPS.
    try:
        from siamban.core.config import cfg as _cfg  # type: ignore[import]
        _cfg.CUDA = False
    except Exception:
        pass  # cfg may not load cleanly without yacs — that's fine.

    from siamban.models.backbone.mobile_v2_eca import MobileNetV2  # type: ignore[import]
    from siamban.models.neck.neck import AdjustAllLayer  # type: ignore[import]
    from siamban.models.head.ban import MultiBAN  # type: ignore[import]

    import torch.nn as nn

    class _MobileTrackModel(nn.Module):
        """Self-contained MobileTrack model — no cfg dependency at runtime."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = MobileNetV2(**_BACKBONE_KWARGS)
            self.neck = AdjustAllLayer(**_NECK_KWARGS)
            self.head = MultiBAN(**_HEAD_KWARGS)
            self.zf: list[torch.Tensor] | None = None

        def template(self, z: torch.Tensor) -> None:
            zf = self.backbone(z)
            zf = self.neck(zf)
            self.zf = zf

        def track(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            xf = self.backbone(x)
            xf = self.neck(xf)
            cls, loc = self.head(self.zf, xf)
            return {"cls": cls, "loc": loc}

    return _MobileTrackModel()


# ---------------------------------------------------------------------------
# SiamBAN-style sub-window crop (device-agnostic). OOB regions are filled
# with the per-channel image mean (avg_chans) — matches the upstream protocol.
# ---------------------------------------------------------------------------


def _get_subwindow(
    im: np.ndarray,
    pos: tuple[float, float],
    model_sz: int,
    original_sz: int,
    avg_chans: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Crop a padded sub-window from *im* and return a (1,3,H,W) float32 tensor."""
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2.0
    cx, cy = pos

    context_xmin = int(np.floor(cx - c + 0.5))
    context_xmax = context_xmin + sz - 1
    context_ymin = int(np.floor(cy - c + 0.5))
    context_ymax = context_ymin + sz - 1

    left_pad = max(0, -context_xmin)
    top_pad = max(0, -context_ymin)
    right_pad = max(0, context_xmax - im_sz[1] + 1)
    bottom_pad = max(0, context_ymax - im_sz[0] + 1)

    context_xmin += left_pad
    context_xmax += left_pad
    context_ymin += top_pad
    context_ymax += top_pad

    r, c_w, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c_w + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad : top_pad + r, left_pad : left_pad + c_w, :] = im
        if top_pad:
            te_im[:top_pad, left_pad : left_pad + c_w, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad :, left_pad : left_pad + c_w, :] = avg_chans
        if left_pad:
            te_im[:, :left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c_w + left_pad :, :] = avg_chans
        im_patch = te_im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]
    else:
        im_patch = im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]

    if im_patch.shape[0] != model_sz or im_patch.shape[1] != model_sz:
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))

    patch = torch.from_numpy(im_patch.transpose(2, 0, 1).astype(np.float32))
    return patch.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Anchor-free point grid (same formula as SiamBANTracker)
# ---------------------------------------------------------------------------


def _generate_points(stride: int, size: int) -> np.ndarray:
    ori = -(size // 2) * stride
    x, y = np.meshgrid(
        [ori + stride * dx for dx in range(size)],
        [ori + stride * dy for dy in range(size)],
    )
    pts = np.zeros((size * size, 2), dtype=np.float32)
    pts[:, 0] = x.flatten()
    pts[:, 1] = y.flatten()
    return pts


# ---------------------------------------------------------------------------
# MobileTrack adapter
# ---------------------------------------------------------------------------


@TRACKERS.register("mobiletrack")
class MobileTrackAdapter:
    """MobileTrack Siamese tracker (Xue et al., IET Image Processing 2022).

    Conforms to ``uav_tracker.trackers.base.Tracker``: ``init``, ``update``,
    ``flops_per_update``, plus the ``capabilities`` property used by
    ``run_with_csc.py`` to gate control hooks.

    Parameters
    ----------
    device : {"auto", "cpu", "cuda", "mps"}
        Device selection. "auto" picks cuda > mps > cpu. Falls back to cpu if
        the requested accelerator is unavailable.
    dtype : {"float32", "float16"}
        Weight/inference precision. ``float16`` requires CUDA or MPS.
    weights_path : str | None
        Explicit ``.pth`` checkpoint path. ``None`` resolves via
        ``$UAV_WEIGHTS_ROOT/mobiletrack/MobileTrack-UAV123-DTB70.pth``.
    mobiletrack_repo : str | None
        Override for the SiamBAN-derived MobileTrack repo root. Reads
        ``$SIAMBAN_REPO`` if not set; default ``~/projects/MobileTrack``.
    window_influence, penalty_k, scale_lr : float
        Tracker post-processing constants (Optuna best on UAV123).
    """

    name: str = "mobiletrack"
    tier_hint: int = 1

    def __init__(
        self,
        device: str = "auto",
        dtype: str = "float32",
        weights_path: str | None = None,
        mobiletrack_repo: str | None = None,
        window_influence: float = _WINDOW_INFLUENCE,
        penalty_k: float = _PENALTY_K,
        scale_lr: float = _SCALE_LR,
    ) -> None:
        # Device resolution: "auto" picks cuda > mps > cpu.
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "MobileTrackAdapter: CUDA requested but not available — falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            warnings.warn(
                "MobileTrackAdapter: MPS requested but not available — falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            device = "cpu"
        self.device = device
        self._device = torch.device(device)

        # dtype — float16 requires CUDA or MPS.
        if dtype == "float16" and device == "cpu":
            warnings.warn(
                "MobileTrackAdapter: float16 not supported on CPU — using float32.",
                RuntimeWarning,
                stacklevel=2,
            )
            dtype = "float32"
        self.dtype = dtype
        self._torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        self.weights_path = weights_path
        self.weights_loaded: bool = False

        self._repo = (
            Path(mobiletrack_repo).expanduser()
            if mobiletrack_repo
            else _MOBILETRACK_REPO
        )

        # Tracker hyper-parameters
        self._exemplar_size: int = _EXEMPLAR_SIZE
        self._instance_size: int = _INSTANCE_SIZE
        self._base_size: int = _BASE_SIZE
        self._context_amount: float = _CONTEXT_AMOUNT
        self._stride: int = _STRIDE
        self._score_size: int = _SCORE_SIZE
        self._window_influence: float = window_influence
        self._penalty_k: float = penalty_k
        self._lr: float = scale_lr
        self._cls_out_channels: int = _HEAD_KWARGS["cls_out_channels"]

        # Runtime state (populated in init())
        self._model: Any | None = None
        self._center_pos: np.ndarray = np.zeros(2, dtype=np.float64)
        self._size: np.ndarray = np.zeros(2, dtype=np.float64)
        self._channel_average: np.ndarray = np.zeros(3, dtype=np.float32)
        self._window: np.ndarray = np.zeros(1, dtype=np.float32)
        self._points: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self._initialized: bool = False
        self._is_stub: bool = True
        self._flops_cached: float | None = None
        # CSCAdvisor / runner control hooks (no-op for this tracker, kept for
        # protocol compatibility with run_with_csc.py).
        self._update_enabled: bool = True

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _resolve_weights_path(self) -> Path:
        if self.weights_path is not None:
            return Path(self.weights_path).expanduser()
        from uav_tracker.paths import weights_root
        return weights_root() / _WEIGHTS_SUBDIR / _WEIGHTS_FILENAME

    def _try_load_weights(self, model: Any) -> bool:
        path = self._resolve_weights_path()
        if not path.exists():
            warnings.warn(
                f"MobileTrackAdapter: weights not found at {path} — using random init.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        try:
            raw = torch.load(str(path), map_location=self._device, weights_only=False)
            state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.debug("MobileTrack: missing keys (first 5): %s", missing[:5])
            if unexpected:
                logger.debug("MobileTrack: unexpected keys (first 5): %s", unexpected[:5])
            logger.info("MobileTrackAdapter: loaded weights from %s", path)
            return True
        except Exception as exc:
            warnings.warn(
                f"MobileTrackAdapter: weight loading failed ({exc}) — using random init.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False

    # ------------------------------------------------------------------
    # Tracker Protocol
    # ------------------------------------------------------------------

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise: build model, load weights, encode template crop."""
        try:
            self._model = _build_mobiletrack_model(self._repo)
        except Exception as exc:
            warnings.warn(
                f"MobileTrackAdapter: failed to build model from {self._repo}: {exc}. "
                f"Check that the SiamBAN-derived MobileTrack repo exists or set $SIAMBAN_REPO.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model = None

        if self._model is not None:
            self._model = self._model.to(self._device).to(self._torch_dtype)
            self._model.eval()
            self.weights_loaded = self._try_load_weights(self._model)
            self._is_stub = not self.weights_loaded
        else:
            self._is_stub = True

        # Target centre and size
        cx = bbox.x + bbox.w / 2.0
        cy = bbox.y + bbox.h / 2.0
        self._center_pos = np.array([cx, cy], dtype=np.float64)
        self._size = np.array([bbox.w, bbox.h], dtype=np.float64)

        # Per-channel mean for OOB padding
        self._channel_average = np.mean(frame, axis=(0, 1))

        # Template crop (SiamBAN protocol — context_amount * (w + h) padding).
        w_z = self._size[0] + self._context_amount * np.sum(self._size)
        h_z = self._size[1] + self._context_amount * np.sum(self._size)
        s_z = round(np.sqrt(w_z * h_z))

        z_crop = _get_subwindow(
            frame,
            (float(self._center_pos[0]), float(self._center_pos[1])),
            self._exemplar_size,
            s_z,
            self._channel_average,
            self._device,
        ).to(self._torch_dtype)

        if self._model is not None:
            with torch.no_grad():
                self._model.template(z_crop)

        # Cosine window (flattened) for response regularisation.
        hanning = np.hanning(self._score_size)
        self._window = np.outer(hanning, hanning).flatten()

        # Anchor-free point grid for bbox decoding.
        self._points = _generate_points(self._stride, self._score_size)

        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackState:
        """Run one tracking step and return a ``TrackState`` with telemetry."""
        if not self._initialized or self._model is None:
            # Stub fallback: emit a "lost" frame anchored at last known pose so
            # the runner can clamp / log gracefully.
            x = float(self._center_pos[0] - self._size[0] / 2.0)
            y = float(self._center_pos[1] - self._size[1] / 2.0)
            return TrackState(
                bbox=BBox(x=x, y=y, w=float(self._size[0]), h=float(self._size[1])),
                confidence=0.0,
                status="lost",
                aux={"stub": True},
            )

        if self._flops_cached is None:
            self._flops_cached = self._measure_flops()

        # Search crop (SiamBAN protocol).
        w_z = self._size[0] + self._context_amount * np.sum(self._size)
        h_z = self._size[1] + self._context_amount * np.sum(self._size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self._exemplar_size / s_z
        s_x = s_z * (self._instance_size / self._exemplar_size)

        x_crop = _get_subwindow(
            frame,
            (float(self._center_pos[0]), float(self._center_pos[1])),
            self._instance_size,
            round(s_x),
            self._channel_average,
            self._device,
        ).to(self._torch_dtype)

        with torch.no_grad():
            outputs = self._model.track(x_crop)

        score = self._convert_score(outputs["cls"])
        pred_bbox = self._convert_bbox(outputs["loc"], self._points)

        # Penalty for scale and aspect-ratio changes (SiamBAN post-proc).
        def change(r: np.ndarray) -> np.ndarray:
            return np.maximum(r, 1.0 / r)

        def sz(w: np.ndarray, h: np.ndarray) -> np.ndarray:
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :])
            / sz(self._size[0] * scale_z, self._size[1] * scale_z)
        )
        r_c = change(
            (self._size[0] / self._size[1]) / (pred_bbox[2, :] / pred_bbox[3, :])
        )
        penalty = np.exp(-(r_c * s_c - 1) * self._penalty_k)
        pscore = penalty * score
        pscore = (
            pscore * (1 - self._window_influence)
            + self._window * self._window_influence
        )

        # nan-safe argmax: nan values in pscore sort low, so replace with 0
        pscore_safe = np.where(np.isfinite(pscore), pscore, 0.0)
        best_idx = int(np.argmax(pscore_safe))
        bbox_delta = pred_bbox[:, best_idx] / scale_z
        lr = float(np.nan_to_num(penalty[best_idx] * score[best_idx] * self._lr, nan=0.0))

        cx = float(np.nan_to_num(bbox_delta[0], nan=0.0)) + self._center_pos[0]
        cy = float(np.nan_to_num(bbox_delta[1], nan=0.0)) + self._center_pos[1]
        width = self._size[0] * (1 - lr) + float(np.nan_to_num(bbox_delta[2], nan=self._size[0])) * lr
        height = self._size[1] * (1 - lr) + float(np.nan_to_num(bbox_delta[3], nan=self._size[1])) * lr

        # Clamp to frame bounds (project rule: every adapter must clip bbox to
        # frame dims). We clamp the raw cx/cy/w/h here so the centre cannot
        # drift off-screen and the box stays within the frame.
        h_img, w_img = frame.shape[:2]
        cx = float(np.clip(cx, 0, w_img))
        cy = float(np.clip(cy, 0, h_img))
        width = float(np.clip(width, 10, w_img))
        height = float(np.clip(height, 10, h_img))

        self._center_pos = np.array([cx, cy])
        self._size = np.array([width, height])

        # Build the candidate bbox in xywh and clamp to frame (defense in depth).
        new_x = cx - width / 2.0
        new_y = cy - height / 2.0
        _bx = min(max(new_x, 0.0), max(0.0, float(w_img) - 1.0))
        _by = min(max(new_y, 0.0), max(0.0, float(h_img) - 1.0))
        _bw = max(1.0, min(width, float(w_img) - _bx))
        _bh = max(1.0, min(height, float(h_img) - _by))
        new_bbox = BBox(x=_bx, y=_by, w=_bw, h=_bh)

        # ----- Score-map quality telemetry (matches SGLATrack/AVTrack convention) -----
        # We use the cls score-map (post-window) as the response map. ``score``
        # is already the foreground-class softmax probability per anchor cell;
        # reshape to (score_size, score_size) for quality stats.
        sm2d = score.reshape(self._score_size, self._score_size)
        f = sm2d.flatten().astype(np.float32)
        f_max = float(f.max()) if f.size else 0.0
        f_min = float(f.min()) if f.size else 0.0
        f_mean = float(f.mean()) if f.size else 0.0
        f_std = float(f.std()) if f.size else 0.0
        denom = float(((f - f_min) ** 2).mean()) if f.size else 0.0
        apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))

        # PSR — peak vs surrounding sidelobe stats.
        if f.size > 0:
            peak_flat = int(np.argmax(f))
            peak_r = peak_flat // self._score_size
            peak_c = peak_flat % self._score_size
            r_idx = np.arange(self._score_size).reshape(-1, 1)
            c_idx = np.arange(self._score_size).reshape(1, -1)
            peak_mask = (np.abs(r_idx - peak_r) <= 5) & (np.abs(c_idx - peak_c) <= 5)
            sidelobe = sm2d[~peak_mask]
            if sidelobe.size > 0:
                psr = float((f_max - sidelobe.mean()) / (sidelobe.std() + 1e-8))
            else:
                psr = 0.0
        else:
            psr = 0.0

        # L1-normalised entropy (matches SGLATrack convention).
        if f.size > 0:
            f_pos = f - f.min()
            total = float(f_pos.sum()) + 1e-8
            probs = f_pos / total
            response_entropy = float(-(probs * np.log(probs + 1e-8)).sum())
        else:
            response_entropy = 0.0

        # Top-1 / top-2 score-map peaks for distractor diagnostics.
        if f.size >= 2:
            sorted_desc = np.sort(f)[::-1]
            _top1 = float(sorted_desc[0])
            _top2 = float(sorted_desc[1])
            _peak_margin = _top1 - _top2
            score_map_stats = {
                "top1": _top1,
                "top2": _top2,
                "peak_margin": _peak_margin,
                "response_entropy": response_entropy,
                "n_secondary": int((sorted_desc > 0.5 * _top1).sum()) - 1,
            }
        else:
            score_map_stats = {}

        # Best-candidate confidence — the post-penalty score with cosine window
        # already mixed in is what selected the box, so report the raw cls
        # softmax probability (already in [0, 1]).
        best_score = float(np.clip(score[best_idx], 0.0, 1.0))
        if best_score > 0.6:
            status = "locked"
        elif best_score > 0.3:
            status = "uncertain"
        else:
            status = "lost"

        raw = {
            "score_max": float(f_max),
            "response_max": float(f_max),
            "response_min": float(f_min),
            "response_mean": float(f_mean),
            "response_std": float(f_std),
            "best_idx": int(best_idx),
            "pscore_max": float(pscore[best_idx]) if pscore.size else 0.0,
            "penalty_at_best": float(penalty[best_idx]) if penalty.size else 0.0,
        }

        return TrackState(
            bbox=new_bbox,
            confidence=best_score,
            status=status,
            apce=apce,
            psr=psr,
            response_entropy=response_entropy,
            score_map_stats=score_map_stats,
            aux={"raw": raw},
        )

    # ------------------------------------------------------------------
    # Control hooks (kept minimal — MobileTrack has no online template
    # update or CE pruning, so most are no-ops). They satisfy the
    # ``run_with_csc.py`` Tracker protocol so the adapter plugs in cleanly.
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset tracker state (clears template cache and model)."""
        self._model = None
        self._initialized = False
        self._flops_cached = None
        self._update_enabled = True

    def set_update_enabled(self, enabled: bool) -> None:
        """CSCAdvisor gate. MobileTrack has no online template update — no-op."""
        self._update_enabled = bool(enabled)

    def override_search_center(self, cx: float, cy: float, w: float, h: float) -> None:
        """Relocate the search region for the next ``update()`` (control hook).

        MobileTrack crops the search region around ``self._center_pos`` each
        frame, so updating it relocates where the tracker looks next. Mirrors
        ``SGLATracker.override_search_center`` and ``AVTrackAdapter.override_search_center``.
        """
        self._center_pos = np.array([float(cx), float(cy)], dtype=np.float64)
        self._size = np.array([float(w), float(h)], dtype=np.float64)

    @property
    def capabilities(self):
        from uav_tracker.trackers.capabilities import TrackerCapabilities
        # MobileTrack does not learn an online template (templates are encoded
        # once at init), and it does not expose a search-factor knob (the
        # SiamBAN crop sizes are hard-coded at 127/255). It does support
        # mid-sequence re-init via init() and bbox rejection at the runner
        # level, both of which TrackerCapabilities defaults to True.
        return TrackerCapabilities(
            can_freeze_template=False,
            can_widen_search=False,
            can_force_reinit=True,
            can_reject_bbox=True,
            can_reduce_pruning=False,
        )

    @property
    def is_stub_mode(self) -> bool:
        return self._is_stub

    def flops_per_update(self) -> float:
        """Return measured FLOPs (thop) or static fallback ~3.5 GFLOPs."""
        return self._flops_cached if self._flops_cached is not None else _FLOPS_FALLBACK

    def on_tier_enter(self, ctx: FrameContext) -> None:
        """No-op — template refresh is the runner's responsibility (re-init)."""
        return None

    def on_tier_exit(self, ctx: FrameContext) -> None:
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_score(self, score: torch.Tensor) -> np.ndarray:
        if self._cls_out_channels == 1:
            s = score.permute(1, 2, 3, 0).contiguous().view(-1)
            return s.sigmoid().detach().cpu().float().numpy()
        s = (
            score.permute(1, 2, 3, 0)
            .contiguous()
            .view(self._cls_out_channels, -1)
            .permute(1, 0)
        )
        return s.softmax(dim=1).detach()[:, 1].cpu().float().numpy()

    def _convert_bbox(
        self, delta: torch.Tensor, points: np.ndarray
    ) -> np.ndarray:
        d = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        d = d.detach().cpu().float().numpy()
        d[0, :] = points[:, 0] - d[0, :]
        d[1, :] = points[:, 1] - d[1, :]
        d[2, :] = points[:, 0] + d[2, :]
        d[3, :] = points[:, 1] + d[3, :]
        # corner2center conversion
        x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return np.stack([cx, cy, w, h], axis=0)

    def _measure_flops(self) -> float:
        """Measure GFLOPs via thop: backbone_search + neck (+ head if available).

        The per-update cost is dominated by the search-pass backbone forward
        on the 255×255 crop. Template forward runs once at init and is not
        amortised here. Falls back to ``_FLOPS_FALLBACK`` if thop fails.
        """
        try:
            from thop import profile as thop_profile  # type: ignore[import]
            assert self._model is not None

            x_dummy = torch.zeros(
                1, 3, self._instance_size, self._instance_size,
                device=self._device, dtype=self._torch_dtype,
            )
            macs_bb_search, _ = thop_profile(
                self._model.backbone, inputs=(x_dummy,), verbose=False
            )

            backbone_out = self._model.backbone(x_dummy)
            macs_neck, _ = thop_profile(
                self._model.neck, inputs=(backbone_out,), verbose=False
            )

            neck_out = self._model.neck(backbone_out)
            try:
                macs_head, _ = thop_profile(
                    self._model.head, inputs=(neck_out, neck_out), verbose=False
                )
            except Exception:
                macs_head = 0  # head is small; skip if profile fails

            total = float(macs_bb_search + macs_neck + macs_head)
            logger.debug(
                "MobileTrack FLOPs: backbone=%.2fG neck=%.3fG head=%.3fG total=%.2fG",
                macs_bb_search / 1e9, macs_neck / 1e9,
                macs_head / 1e9, total / 1e9,
            )
            return total
        except Exception as exc:
            logger.debug("MobileTrack: thop FLOPs measurement failed: %s", exc)
            return _FLOPS_FALLBACK
