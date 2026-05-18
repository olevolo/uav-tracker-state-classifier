"""SiamFC tracker (Bertinetto 2016) — Phase 2 deep tracker.

Paper §3.2 / engineer.md §3.2B — AlexNet-style siamese backbone cross-
correlates a 127×127 template against a 255×255 search crop to produce a
17×17 response map.  Three-scale pyramid + cosine-window dampening give
sub-pixel localisation from a single forward pass.

Tier hint: 1.  Target FPS on T4 (FP16): ~80.

Reference: Bertinetto et al., "Fully-Convolutional Siamese Networks for
Object Tracking", ECCVW 2016.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, TrackState

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AlexNet-style backbone (Bertinetto 2016, Table 1)
# ---------------------------------------------------------------------------


class _SiamFCBackbone(nn.Module):
    """Five-block AlexNet-style feature extractor.

    No padding → spatial size strictly determined by kernel / stride.

    Input 127×127 → 6×6×256 (template)
    Input 255×255 → 22×22×256 (search)
    """

    def __init__(self) -> None:
        super().__init__()
        # Block 1: conv(11,4) + BN + ReLU + MaxPool(3,2) = 127→29
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Block 2: conv(5,1) + BN + ReLU + MaxPool(3,2) = 29→13
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Block 3: conv(3,1) + BN + ReLU = 13→11
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        # Block 4: conv(3,1) + BN + ReLU = 11→9
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        # Block 5: conv(3,1) (no BN per Bertinetto) = 9→7
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0, bias=False),
        )
        # Weight init: MSRA
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) → (B,256,h,w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


# ---------------------------------------------------------------------------
# SiamFC model wrapper (template caching + cross-correlation)
# ---------------------------------------------------------------------------


class _SiamFCModel(nn.Module):
    """Wraps the shared backbone and exposes template/search forward passes."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _SiamFCBackbone()

    def encode_template(self, z: torch.Tensor) -> torch.Tensor:
        return self.backbone(z)

    def encode_search(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def correlate(
        self, z_feat: torch.Tensor, x_feat: torch.Tensor
    ) -> torch.Tensor:
        """Cross-correlate template features over search features.

        z_feat: (1, C, Hz, Wz) — template
        x_feat: (1, C, Hx, Wx) — search
        returns: (1, 1, Hx-Hz+1, Wx-Wz+1) score map
        """
        # Reshape z into conv kernel: (1, C, Hz, Wz) → (C, 1, Hz, Wz) per group
        # Then grouped conv: one filter per output channel
        b, c, hz, wz = z_feat.shape
        z_kernel = z_feat.reshape(1, c, hz, wz)
        # Use groups=1 depthwise-style cross-correlation
        score = F.conv2d(x_feat, z_kernel)
        return score


# ---------------------------------------------------------------------------
# Helpers: cropping + cosine window
# ---------------------------------------------------------------------------


def _get_subwindow(
    frame: np.ndarray,
    cx: float,
    cy: float,
    size: int,
    out_size: int,
    avg_chans: np.ndarray,
) -> np.ndarray:
    """Crop and resize a sub-window centred at (cx,cy) from frame.

    Regions outside the image are filled with the per-channel mean.
    Returns a uint8 RGB HxWx3 array of size out_size×out_size.
    """
    h, w = frame.shape[:2]
    half = size / 2.0
    x0 = round(cx - half)
    y0 = round(cy - half)
    x1 = x0 + size
    y1 = y0 + size

    # Padding amounts
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)

    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(w, x1)
    y1c = min(h, y1)

    crop = frame[y0c:y1c, x0c:x1c].copy()
    # BGR → RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    if any((pad_l, pad_t, pad_r, pad_b)):
        crop = np.pad(
            crop,
            ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        # Fill padded region with per-channel means
        for c in range(3):
            mask = np.zeros(crop.shape[:2], dtype=bool)
            if pad_t:
                mask[:pad_t, :] = True
            if pad_b:
                mask[crop.shape[0] - pad_b :, :] = True
            if pad_l:
                mask[:, :pad_l] = True
            if pad_r:
                mask[:, crop.shape[1] - pad_r :] = True
            crop[:, :, c][mask] = int(avg_chans[c])

    if crop.shape[0] != out_size or crop.shape[1] != out_size:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    return crop


def _to_tensor(img: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """HxWx3 uint8 RGB ndarray → (1,3,H,W) normalised tensor."""
    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    t = t / 255.0
    # ImageNet-style mean subtract (approximate for goturn/siamfc pre-trained)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).reshape(1, 3, 1, 1)
    t = t.to(device=device, dtype=dtype) - mean
    return t


def _cosine_window(size: int) -> np.ndarray:
    """2-D cosine (Hanning) window of shape (size, size)."""
    hann_1d = np.hanning(size)
    return np.outer(hann_1d, hann_1d)


# ---------------------------------------------------------------------------
# SiamFCTracker
# ---------------------------------------------------------------------------


@TRACKERS.register("siamfc")
class SiamFCTracker:
    """SiamFC Siamese tracker (Bertinetto 2016).

    Parameters
    ----------
    device:
        ``"cpu"`` or ``"cuda"``. Falls back to cpu if CUDA unavailable.
    dtype:
        ``"float32"`` or ``"float16"``. ``"float16"`` is only allowed on CUDA.
    weights_path:
        Optional path to a ``.pth`` checkpoint. ``None`` → resolve via
        ``$UAV_WEIGHTS_ROOT/siamfc/siamfc_alexnet_e50.pth``.
    """

    name: str = "siamfc"
    tier_hint: int = 1
    _FLOPS_FALLBACK: float = 1.2e9

    # SiamFC hyper-parameters (Bertinetto 2016 defaults / configs/trackers/siamfc.yaml)
    _exemplar_size: int = 127
    _instance_size: int = 255
    _scale_num: int = 3
    _scale_step: float = 1.0375
    _scale_penalty: float = 0.9745
    _window_influence: float = 0.176
    _response_up: int = 16
    _scale_lr: float = 0.35
    _score_size: int = 17  # response map side before upsampling

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float32",
        weights_path: str | None = None,
    ) -> None:
        # Device resolution
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "SiamFCTracker: CUDA requested but not available — falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

        # dtype
        if dtype == "float16" and device == "cpu":
            warnings.warn(
                "SiamFCTracker: float16 not supported on CPU — using float32.",
                RuntimeWarning,
                stacklevel=2,
            )
            dtype = "float32"
        self.dtype = dtype
        self._torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        self.weights_path = weights_path
        self.weights_loaded: bool = False

        # State (populated in init())
        self._model: _SiamFCModel | None = None
        self._template_feat: torch.Tensor | None = None
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._target_w: float = 0.0
        self._target_h: float = 0.0
        self._scale_z: float = 1.0
        self._avg_chans: np.ndarray = np.zeros(3, dtype=np.float32)
        self._cosine_win: np.ndarray | None = None
        self._initialized: bool = False

        # FLOPs (measured lazily on first update)
        self._flops_cached: float | None = None

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------

    def _resolve_weights_path(self) -> Path | None:
        if self.weights_path is not None:
            return Path(self.weights_path)
        from uav_tracker.paths import weights_root
        candidate = weights_root() / "siamfc" / "siamfc_alexnet_e50.pth"
        return candidate if candidate.exists() or os.environ.get("UAV_WEIGHTS_ROOT") else None

    def _try_load_weights(self, model: _SiamFCModel) -> bool:
        path = self._resolve_weights_path()
        if path is None:
            return False
        if not path.exists():
            warnings.warn(
                f"SiamFCTracker: weights not found at {path} — using random init.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        try:
            state = torch.load(str(path), map_location=self._torch_device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                _log.debug("SiamFC: missing keys %s", missing[:5])
            if unexpected:
                _log.debug("SiamFC: unexpected keys %s", unexpected[:5])
            _log.info("SiamFCTracker: loaded weights from %s", path)
            return True
        except Exception as exc:
            warnings.warn(
                f"SiamFCTracker: weight loading failed ({exc}) — using random init.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False

    # ------------------------------------------------------------------
    # Tracker Protocol
    # ------------------------------------------------------------------

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        """Crop 127×127 template, cache backbone features."""
        # Build model on first init (lazy — avoids torch import cost at package load)
        self._model = _SiamFCModel().to(self._torch_device).to(self._torch_dtype)
        self._model.eval()
        self.weights_loaded = self._try_load_weights(self._model)

        # Centre of target
        cx = bbox.x + bbox.w / 2.0
        cy = bbox.y + bbox.h / 2.0
        self._cx = cx
        self._cy = cy
        self._target_w = bbox.w
        self._target_h = bbox.h

        # Per-channel mean of the frame (used for padded crops)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._avg_chans = rgb.mean(axis=(0, 1)).astype(np.float32)

        # Template crop size with context (Bertinetto 2016 §2.1)
        wc = bbox.w + (bbox.w + bbox.h) / 4.0
        hc = bbox.h + (bbox.w + bbox.h) / 4.0
        s_z = math.sqrt(wc * hc)
        self._scale_z = self._exemplar_size / s_z

        # Crop + encode template
        template_img = _get_subwindow(
            frame, cx, cy,
            round(s_z),
            self._exemplar_size,
            self._avg_chans,
        )
        z_tensor = _to_tensor(template_img, self._torch_device, self._torch_dtype)
        with torch.no_grad():
            self._template_feat = self._model.encode_template(z_tensor)

        # Cosine window for upsampled response map
        ups = self._score_size * self._response_up
        self._cosine_win = _cosine_window(ups)

        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackState:
        """Three-scale search + argmax → new bbox."""
        if not self._initialized or self._model is None or self._template_feat is None:
            # Fallback: return current bbox with low confidence
            return TrackState(
                bbox=BBox(self._cx - self._target_w / 2, self._cy - self._target_h / 2,
                          self._target_w, self._target_h),
                confidence=0.0,
                status="lost",
            )

        # Lazy FLOPs measurement (first update)
        if self._flops_cached is None:
            self._flops_cached = self._measure_flops()

        # Compute instance crop size at current scale
        wc = self._target_w + (self._target_w + self._target_h) / 4.0
        hc = self._target_h + (self._target_w + self._target_h) / 4.0
        s_z = math.sqrt(wc * hc)
        scale_z = self._exemplar_size / s_z
        # Instance size in original image pixels
        s_x = self._instance_size / scale_z

        # Scales
        scale_factors = [
            self._scale_step ** (i - self._scale_num // 2)
            for i in range(self._scale_num)
        ]

        # One response map per scale
        response_maps: list[np.ndarray] = []
        with torch.no_grad():
            for sf in scale_factors:
                size_i = round(s_x * sf)
                search_img = _get_subwindow(
                    frame,
                    self._cx, self._cy,
                    size_i,
                    self._instance_size,
                    self._avg_chans,
                )
                x_tensor = _to_tensor(search_img, self._torch_device, self._torch_dtype)
                x_feat = self._model.encode_search(x_tensor)
                score = self._model.correlate(self._template_feat, x_feat)  # (1,1,H,W)
                score_np = score.squeeze().cpu().float().numpy()
                # Bicubic upsample to score_size * response_up
                ups = self._score_size * self._response_up
                score_up = cv2.resize(
                    score_np, (ups, ups), interpolation=cv2.INTER_CUBIC
                )
                response_maps.append(score_up)

        ups = self._score_size * self._response_up

        # Apply scale penalty to off-centre scales
        penalised = []
        center_idx = self._scale_num // 2
        for i, rm in enumerate(response_maps):
            if i != center_idx:
                rm = rm * self._scale_penalty
            penalised.append(rm)

        # Normalise each map to [0,1] then apply cosine window
        windowed = []
        for rm in penalised:
            rm_min, rm_max = rm.min(), rm.max()
            if rm_max > rm_min:
                rm_norm = (rm - rm_min) / (rm_max - rm_min)
            else:
                rm_norm = rm - rm_min
            rm_win = (1 - self._window_influence) * rm_norm + self._window_influence * self._cosine_win
            windowed.append(rm_win)

        # Stack and pick best scale + position
        stack = np.stack(windowed, axis=0)  # (S, ups, ups)
        best_scale_idx, best_r, best_c = np.unravel_index(np.argmax(stack), stack.shape)
        best_response = float(stack[best_scale_idx, best_r, best_c])

        # Map argmax → displacement in original image coords
        disp_r = best_r - (ups - 1) / 2.0
        disp_c = best_c - (ups - 1) / 2.0
        # stride in upsampled map vs instance image
        total_stride = 8  # net stride of backbone
        disp_search = (disp_r, disp_c)  # in instance image pixels
        # Convert to original image displacement
        scale_at_best = scale_factors[best_scale_idx]
        disp_orig_y = disp_search[0] * (total_stride / self._response_up) / (scale_z * scale_at_best)
        disp_orig_x = disp_search[1] * (total_stride / self._response_up) / (scale_z * scale_at_best)

        new_cx = self._cx + disp_orig_x
        new_cy = self._cy + disp_orig_y

        # Smooth scale update
        lr = self._scale_lr * best_response
        new_w = self._target_w * (scale_factors[best_scale_idx] ** lr)
        new_h = self._target_h * (scale_factors[best_scale_idx] ** lr)

        # Update state
        self._cx = new_cx
        self._cy = new_cy
        self._target_w = new_w
        self._target_h = new_h

        # Clip confidence to [0, 1]
        confidence = float(np.clip(best_response, 0.0, 1.0))
        status: str
        if confidence > 0.5:
            status = "locked"
        elif confidence > 0.2:
            status = "uncertain"
        else:
            status = "lost"

        return TrackState(
            bbox=BBox(
                x=new_cx - new_w / 2.0,
                y=new_cy - new_h / 2.0,
                w=new_w,
                h=new_h,
            ),
            confidence=confidence,
            status=status,
            aux={"best_scale_idx": int(best_scale_idx)},
        )

    def flops_per_update(self) -> float:
        """Return measured FLOPs (thop) or static fallback 1.2 GFLOPs."""
        if self._flops_cached is not None:
            return self._flops_cached
        return self._FLOPS_FALLBACK

    def _measure_flops(self) -> float:
        """Try thop.profile; return static fallback on failure."""
        try:
            from thop import profile as thop_profile  # type: ignore[import]

            assert self._model is not None
            assert self._template_feat is not None
            dummy_search = torch.zeros(
                1, 3, self._instance_size, self._instance_size,
                device=self._torch_device, dtype=self._torch_dtype,
            )
            macs, _ = thop_profile(self._model.backbone, inputs=(dummy_search,), verbose=False)
            # Multiply by 2 for template + search passes + correlation overhead
            return float(macs) * 2.0
        except Exception as exc:  # pragma: no cover
            _log.debug("SiamFC: thop FLOPs measurement failed: %s", exc)
            return self._FLOPS_FALLBACK

    def on_tier_enter(self, ctx: Any) -> None:
        """LIGHT→DEEP: no state reset needed here (template already cached)."""
        return None

    def on_tier_exit(self, ctx: Any) -> None:
        return None
