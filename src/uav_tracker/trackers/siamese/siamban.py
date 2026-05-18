"""SiamBAN R50 tracker bridge (Chen et al., CVPR 2020).

Imports SiamBAN directly from the local MobileTrack repo at
``~/projects/MobileTrack`` and wraps it in the UAV Tracker Protocol.

Architecture
------------
* ResNet-50 atrous backbone (layers 2,3,4 → dilations 1,2,4)
* AdjustAllLayer neck: 3×256-ch adjust layers (512+1024+2048 → 256)
* MultiBAN head: 3 depthwise-xcorr branches, weighted avg, cls+loc outputs

Weights
-------
``$UAV_WEIGHTS_ROOT/mobiletrack/siamban_r50_l234.pth`` (epoch 20,
SHA-verified during download).  Missing weights → runtime warning + random
init so the tracker is still importable and usable in unit tests.

Device compat
-------------
The upstream SiamBAN codebase calls ``.cuda()`` and hard-codes
``cfg.CUDA = True``.  This bridge patches that: all tensors are moved via
``.to(self._device)`` and the CUDA flag in the cfg is set from the ``device``
argument, not from environment detection.

Reference: Chen et al., "Siamese Box Adaptive Network for Visual Tracking",
CVPR 2020.  Local repo: ~/projects/MobileTrack.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, TrackState

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SiamBAN repo path injection (lazy — only when tracker is instantiated)
# ---------------------------------------------------------------------------

_SIAMBAN_REPO = Path(os.environ.get("SIAMBAN_REPO", "")).expanduser()
if not _SIAMBAN_REPO or not _SIAMBAN_REPO.exists():
    _SIAMBAN_REPO = Path("~/projects/MobileTrack").expanduser()

# ---------------------------------------------------------------------------
# Default weights path
# ---------------------------------------------------------------------------

_WEIGHTS_SUBDIR = "mobiletrack"
_WEIGHTS_FILENAME = "siamban_r50_l234.pth"

# Default config values (mirrors experiments/siamban_mobilev2_l234/config.yaml
# but using the R50 settings from the upstream SiamBAN config defaults).
_DEFAULT_CFG = {
    "EXEMPLAR_SIZE": 127,
    "INSTANCE_SIZE": 255,
    "BASE_SIZE": 8,
    "CONTEXT_AMOUNT": 0.5,
    "PENALTY_K": 0.14,
    "WINDOW_INFLUENCE": 0.45,
    "LR": 0.30,
    "STRIDE": 8,
}

# BAN head / neck params that match siamban_r50_l234 checkpoint
_R50_BACKBONE_KWARGS = {"used_layers": [2, 3, 4]}
_R50_NECK_KWARGS = {
    "in_channels": [512, 1024, 2048],
    "out_channels": [256, 256, 256],
}
_R50_HEAD_KWARGS = {
    "in_channels": [256, 256, 256],
    "cls_out_channels": 2,
    "weighted": True,
}


# ---------------------------------------------------------------------------
# Internal model builder (does NOT depend on cfg global at import time)
# ---------------------------------------------------------------------------


def _build_siamban_model(repo: Path) -> "torch.nn.Module":
    """Import siamban internals and construct a ModelBuilder-equivalent.

    Avoids importing ``from siamban.core.config import cfg`` at module level so
    the bridge never crashes on import even if the repo is missing.
    """
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # yacs may not be installed — provide a thin shim so config imports succeed.
    try:
        import yacs  # noqa: F401
    except ImportError:
        # Inject a minimal CfgNode shim so siamban.core.config doesn't blow up.
        _inject_yacs_shim()

    from siamban.models.backbone.resnet_atrous import resnet50  # type: ignore[import]
    from siamban.models.neck.neck import AdjustAllLayer  # type: ignore[import]
    from siamban.models.head.ban import MultiBAN  # type: ignore[import]

    import torch.nn as nn

    class _SiamBANModel(nn.Module):
        """Minimal ModelBuilder-equivalent without cfg dependency."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = resnet50(**_R50_BACKBONE_KWARGS)
            self.neck = AdjustAllLayer(**_R50_NECK_KWARGS)
            self.head = MultiBAN(**_R50_HEAD_KWARGS)
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

    return _SiamBANModel()


def _inject_yacs_shim() -> None:
    """Register a minimal yacs.config.CfgNode shim so siamban imports don't fail."""
    import types

    class _CfgNode(dict):
        """Minimal stand-in for yacs.config.CfgNode."""

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
# Crop helper (matches SiamBAN's get_subwindow but device-agnostic)
# ---------------------------------------------------------------------------


def _get_subwindow(
    im: np.ndarray,
    pos: tuple[float, float],
    model_sz: int,
    original_sz: int,
    avg_chans: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Crop and resize a sub-window, pad OOB with avg_chans.

    Returns a (1, 3, model_sz, model_sz) float32 tensor on *device*.
    The upstream SiamBAN implementation placed this tensor on CUDA
    unconditionally; here we use ``device`` throughout.
    """
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

    # HWC uint8 → (1, 3, H, W) float32 tensor on device
    patch = torch.from_numpy(im_patch.transpose(2, 0, 1).astype(np.float32))
    patch = patch.unsqueeze(0).to(device)
    return patch


# ---------------------------------------------------------------------------
# SiamBAN point-grid helper
# ---------------------------------------------------------------------------


def _generate_points(stride: int, size: int) -> np.ndarray:
    """Generate the (size*size, 2) anchor-free point grid."""
    ori = -(size // 2) * stride
    x, y = np.meshgrid(
        [ori + stride * dx for dx in range(size)],
        [ori + stride * dy for dy in range(size)],
    )
    points = np.zeros((size * size, 2), dtype=np.float32)
    points[:, 0] = x.flatten()
    points[:, 1] = y.flatten()
    return points


# ---------------------------------------------------------------------------
# SiamBANTracker
# ---------------------------------------------------------------------------


@TRACKERS.register("siamban")
class SiamBANTracker:
    """SiamBAN R50 Siamese tracker (Chen et al., CVPR 2020).

    Parameters
    ----------
    device:
        ``"cpu"`` or ``"cuda"``.  Falls back to cpu if CUDA unavailable.
    dtype:
        ``"float32"`` or ``"float16"``.  ``"float16"`` requires CUDA.
    weights_path:
        Optional explicit path to ``.pth`` checkpoint.  ``None`` → resolves
        via ``$UAV_WEIGHTS_ROOT/mobiletrack/siamban_r50_l234.pth``.
    siamban_repo:
        Optional override for the MobileTrack repo path (default
        ``~/projects/MobileTrack``).  Can also be set via ``$SIAMBAN_REPO``.
    """

    name: str = "siamban"
    tier_hint: int = 1
    _FLOPS_FALLBACK: float = 8.5e9  # R50 backbone ~4 GFLOPs × 2 passes + head

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float32",
        weights_path: str | None = None,
        siamban_repo: str | None = None,
    ) -> None:
        # Device resolution
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "SiamBANTracker: CUDA requested but not available — falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            device = "cpu"
        self.device = device
        self._device = torch.device(device)

        # dtype
        if dtype == "float16" and device == "cpu":
            warnings.warn(
                "SiamBANTracker: float16 not supported on CPU — using float32.",
                RuntimeWarning,
                stacklevel=2,
            )
            dtype = "float32"
        self.dtype = dtype
        self._torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        self.weights_path = weights_path
        self.weights_loaded: bool = False

        # Repo override
        self._repo = (
            Path(siamban_repo).expanduser()
            if siamban_repo
            else _SIAMBAN_REPO
        )

        # Tracker hyper-parameters (mirroring siamban_r50_l234 defaults)
        self._exemplar_size: int = _DEFAULT_CFG["EXEMPLAR_SIZE"]
        self._instance_size: int = _DEFAULT_CFG["INSTANCE_SIZE"]
        self._base_size: int = _DEFAULT_CFG["BASE_SIZE"]
        self._context_amount: float = _DEFAULT_CFG["CONTEXT_AMOUNT"]
        self._penalty_k: float = _DEFAULT_CFG["PENALTY_K"]
        self._window_influence: float = _DEFAULT_CFG["WINDOW_INFLUENCE"]
        self._lr: float = _DEFAULT_CFG["LR"]
        self._stride: int = _DEFAULT_CFG["STRIDE"]

        # Derived from instance/exemplar sizes
        self._score_size: int = (
            (self._instance_size - self._exemplar_size) // self._stride
            + 1
            + self._base_size
        )

        # Runtime state (populated by init())
        self._model: Any | None = None
        self._center_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._size: np.ndarray = np.zeros(2, dtype=np.float32)
        self._channel_average: np.ndarray = np.zeros(3, dtype=np.float32)
        self._window: np.ndarray = np.zeros(1, dtype=np.float32)
        self._points: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self._cls_out_channels: int = 2
        self._initialized: bool = False
        self._flops_cached: float | None = None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _resolve_weights_path(self) -> Path | None:
        if self.weights_path is not None:
            return Path(self.weights_path).expanduser()
        from uav_tracker.paths import weights_root

        candidate = weights_root() / _WEIGHTS_SUBDIR / _WEIGHTS_FILENAME
        return candidate

    def _try_load_weights(self, model: Any) -> bool:
        path = self._resolve_weights_path()
        if path is None or not path.exists():
            warnings.warn(
                f"SiamBANTracker: weights not found at {path} — using random init.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        try:
            raw = torch.load(
                str(path), map_location=self._device, weights_only=False
            )
            state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                _log.debug("SiamBAN: missing keys (first 5): %s", missing[:5])
            if unexpected:
                _log.debug("SiamBAN: unexpected keys (first 5): %s", unexpected[:5])
            _log.info("SiamBANTracker: loaded weights from %s", path)
            return True
        except Exception as exc:
            warnings.warn(
                f"SiamBANTracker: weight loading failed ({exc}) — using random init.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False

    # ------------------------------------------------------------------
    # Tracker Protocol
    # ------------------------------------------------------------------

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise tracker: build model, load weights, encode template."""
        # Build model lazily (avoids torch overhead at package import)
        try:
            self._model = _build_siamban_model(self._repo)
        except Exception as exc:
            warnings.warn(
                f"SiamBANTracker: failed to build model from {self._repo}: {exc}. "
                "Check that ~/projects/MobileTrack exists or set $SIAMBAN_REPO.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model = None

        if self._model is not None:
            self._model = self._model.to(self._device).to(self._torch_dtype)
            self._model.eval()
            self.weights_loaded = self._try_load_weights(self._model)

        # Target centre and size
        cx = bbox.x + bbox.w / 2.0
        cy = bbox.y + bbox.h / 2.0
        self._center_pos = np.array([cx, cy], dtype=np.float64)
        self._size = np.array([bbox.w, bbox.h], dtype=np.float64)

        # Per-channel mean for OOB padding
        self._channel_average = np.mean(frame, axis=(0, 1))

        # Template crop
        w_z = self._size[0] + self._context_amount * np.sum(self._size)
        h_z = self._size[1] + self._context_amount * np.sum(self._size)
        s_z = round(np.sqrt(w_z * h_z))

        z_crop = _get_subwindow(
            frame,
            (self._center_pos[0], self._center_pos[1]),
            self._exemplar_size,
            s_z,
            self._channel_average,
            self._device,
        ).to(self._torch_dtype)

        if self._model is not None:
            with torch.no_grad():
                self._model.template(z_crop)

        # Cosine window (flattened, shape = score_size*score_size)
        hanning = np.hanning(self._score_size)
        window = np.outer(hanning, hanning)
        self._window = window.flatten()

        # Anchor-free point grid
        self._points = _generate_points(self._stride, self._score_size)

        self._cls_out_channels = _R50_HEAD_KWARGS["cls_out_channels"]
        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackState:
        """Run one tracking step and return a TrackState."""
        if not self._initialized or self._model is None:
            return TrackState(
                bbox=BBox(
                    x=self._center_pos[0] - self._size[0] / 2.0,
                    y=self._center_pos[1] - self._size[1] / 2.0,
                    w=float(self._size[0]),
                    h=float(self._size[1]),
                ),
                confidence=0.0,
                status="lost",
            )

        if self._flops_cached is None:
            self._flops_cached = self._measure_flops()

        # Search crop
        w_z = self._size[0] + self._context_amount * np.sum(self._size)
        h_z = self._size[1] + self._context_amount * np.sum(self._size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self._exemplar_size / s_z
        s_x = s_z * (self._instance_size / self._exemplar_size)

        x_crop = _get_subwindow(
            frame,
            (self._center_pos[0], self._center_pos[1]),
            self._instance_size,
            round(s_x),
            self._channel_average,
            self._device,
        ).to(self._torch_dtype)

        with torch.no_grad():
            outputs = self._model.track(x_crop)

        score = self._convert_score(outputs["cls"])
        pred_bbox = self._convert_bbox(outputs["loc"], self._points)

        # Penalty for size/aspect changes
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

        best_idx = int(np.argmax(pscore))
        bbox_delta = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self._lr

        cx = bbox_delta[0] + self._center_pos[0]
        cy = bbox_delta[1] + self._center_pos[1]
        width = self._size[0] * (1 - lr) + bbox_delta[2] * lr
        height = self._size[1] * (1 - lr) + bbox_delta[3] * lr

        # Clip to frame boundary
        h_img, w_img = frame.shape[:2]
        cx = float(np.clip(cx, 0, w_img))
        cy = float(np.clip(cy, 0, h_img))
        width = float(np.clip(width, 10, w_img))
        height = float(np.clip(height, 10, h_img))

        self._center_pos = np.array([cx, cy])
        self._size = np.array([width, height])

        best_score = float(np.clip(score[best_idx], 0.0, 1.0))
        if best_score > 0.6:
            status = "locked"
        elif best_score > 0.3:
            status = "uncertain"
        else:
            status = "lost"

        return TrackState(
            bbox=BBox(
                x=cx - width / 2.0,
                y=cy - height / 2.0,
                w=width,
                h=height,
            ),
            confidence=best_score,
            status=status,
            aux={"best_idx": best_idx, "pscore_max": float(pscore[best_idx])},
        )

    def reset(self) -> None:
        """Reset tracker state (clears template cache)."""
        self._model = None
        self._initialized = False
        self._flops_cached = None

    def flops_per_update(self) -> float:
        """Return measured FLOPs (thop) or static fallback 8.5 GFLOPs."""
        return self._flops_cached if self._flops_cached is not None else self._FLOPS_FALLBACK

    def on_tier_enter(self, ctx: Any) -> None:
        """No-op; template refresh is caller's responsibility via re-init."""
        return None

    def on_tier_exit(self, ctx: Any) -> None:
        return None

    # ------------------------------------------------------------------
    # Helpers
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
        # corner2center
        x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        result = np.stack([cx, cy, w, h], axis=0)
        return result

    def _measure_flops(self) -> float:
        try:
            from thop import profile as thop_profile  # type: ignore[import]

            assert self._model is not None
            dummy = torch.zeros(
                1, 3, self._instance_size, self._instance_size,
                device=self._device, dtype=self._torch_dtype,
            )
            macs, _ = thop_profile(self._model.backbone, inputs=(dummy,), verbose=False)
            return float(macs) * 2.0
        except Exception as exc:
            _log.debug("SiamBAN: thop FLOPs measurement failed: %s", exc)
            return self._FLOPS_FALLBACK
