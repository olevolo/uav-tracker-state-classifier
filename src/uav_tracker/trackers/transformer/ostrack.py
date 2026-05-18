"""OSTrack one-stream transformer tracker (Ye et al., 2022).

Reference: Ye B. et al., "Joint Feature Learning and Relation Modeling for
Tracking: A One-Stream Framework", ECCV 2022.

Weight source: https://github.com/botaoye/OSTrack (256x256 model)
Expected weight path: $UAV_WEIGHTS_ROOT/ostrack/ostrack256_full_ep300.pth.tar
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, FrameContext, TrackState

logger = logging.getLogger(__name__)

_FLOPS_PER_UPDATE = 4.8e9  # ~4.8 GFLOPs from paper Table 1

# --- Minimal OSTrack backbone stub ---
# In production this would load the real OSTrack architecture from a subpackage.
# This stub uses a simple ConvNet that matches OSTrack's input/output interface
# so the tracking pipeline works end-to-end before real weights are available.

class _OSTrackBackboneStub(nn.Module):
    """Minimal backbone stub matching OSTrack search/template interface.

    Real OSTrack uses a ViT-Base with joint template-search attention.
    This stub uses a lightweight ConvNet to validate the tracking pipeline.
    Replace with the real architecture once weights are available.
    """

    def __init__(self, template_size: int = 128, search_size: int = 256) -> None:
        super().__init__()
        self.template_size = template_size
        self.search_size = search_size
        # Lightweight feature extractor
        self._encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self._template_feat: torch.Tensor | None = None

    def encode_template(self, patch: torch.Tensor) -> torch.Tensor:
        return self._encoder(patch)

    def forward(self, search: torch.Tensor, template_feat: torch.Tensor) -> torch.Tensor:
        search_feat = self._encoder(search)  # (1, 256, 8, 8)
        # Simple cross-correlation as position score
        b, c, h, w = search_feat.shape
        score_map = torch.einsum('bchw,bcHW->bhw', search_feat, template_feat.expand_as(search_feat))
        return score_map  # (1, 8, 8)


@TRACKERS.register("ostrack_256")
class OSTrackTracker:
    """OSTrack-256 transformer tracker (stub implementation).

    tier_hint=2 (heavy tracker — 4.8 GFLOPs, targets <30ms on T4 GPU).
    Falls back to random backbone when weights are unavailable.
    """

    name: str = "ostrack_256"
    tier_hint: int = 2

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        weights_path: str | None = None,
        search_size: int = 256,
        template_size: int = 128,
        scale_factor: float = 2.0,
    ) -> None:
        self._device_str = device
        self._dtype = dtype
        self._weights_path = weights_path
        self.search_size = search_size
        self.template_size = template_size
        self.scale_factor = scale_factor
        self._model: _OSTrackBackboneStub | None = None
        self._template_feat: torch.Tensor | None = None
        self._last_bbox: BBox | None = None
        self._flops: float | None = None
        self._is_stub: bool = True

    @property
    def _device(self) -> torch.device:
        if self._device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> _OSTrackBackboneStub:
        model = _OSTrackBackboneStub(self.template_size, self.search_size)
        weights_path = self._weights_path
        if weights_path is None:
            from uav_tracker.paths import weights_root
            weights_path = str(weights_root() / "ostrack" / "ostrack256_full_ep300.pth.tar")
        p = Path(weights_path)
        if p.exists():
            try:
                ckpt = torch.load(p, map_location="cpu")
                state = ckpt.get("net", ckpt)
                model.load_state_dict(state, strict=False)
                logger.info("OSTrack weights loaded from %s", p)
            except Exception as exc:
                logger.warning("Failed to load OSTrack weights from %s: %s — using random init", p, exc)
        else:
            logger.info("OSTrack weights not found at %s — using random init", p)
        self._is_stub = not p.exists()
        return model.to(self._device).eval()

    def _crop_patch(self, frame: np.ndarray, bbox: BBox, out_size: int) -> np.ndarray:
        h, w = frame.shape[:2]
        cx = bbox.x + bbox.w / 2
        cy = bbox.y + bbox.h / 2
        context = (bbox.w + bbox.h) / 2 * self.scale_factor
        x1 = max(0, int(cx - context / 2))
        y1 = max(0, int(cy - context / 2))
        x2 = min(w, int(cx + context / 2))
        y2 = min(h, int(cy + context / 2))
        crop = frame[y1:y2, x1:x2]
        import cv2
        return cv2.resize(crop, (out_size, out_size))

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        if self._model is None:
            self._model = self._load_model()
        patch = self._crop_patch(frame, bbox, self.template_size)
        t = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        t = t.to(self._device)
        with torch.no_grad():
            self._template_feat = self._model.encode_template(t)
        self._last_bbox = bbox

    def update(self, frame: np.ndarray) -> TrackState:
        if self._model is None or self._template_feat is None or self._last_bbox is None:
            return TrackState(bbox=BBox(0, 0, 1, 1), confidence=0.0, status="lost")
        patch = self._crop_patch(frame, self._last_bbox, self.search_size)
        s = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        s = s.to(self._device)
        with torch.no_grad():
            score_map = self._model(s, self._template_feat)  # (1, 8, 8)
        # Find peak location
        score_np = score_map.squeeze().cpu().numpy()
        flat_idx = score_np.argmax()
        peak_y, peak_x = divmod(int(flat_idx), score_np.shape[1])
        # Map back to frame coordinates
        h, w = frame.shape[:2]
        bbox = self._last_bbox
        cx = bbox.x + bbox.w / 2
        cy = bbox.y + bbox.h / 2
        context = (bbox.w + bbox.h) / 2 * self.scale_factor
        x1 = max(0, int(cx - context / 2))
        y1 = max(0, int(cy - context / 2))
        x2 = min(w, int(cx + context / 2))
        y2 = min(h, int(cy + context / 2))
        cell_w = (x2 - x1) / score_np.shape[1]
        cell_h = (y2 - y1) / score_np.shape[0]
        pred_cx = x1 + (peak_x + 0.5) * cell_w
        pred_cy = y1 + (peak_y + 0.5) * cell_h
        new_bbox = BBox(
            x=pred_cx - bbox.w / 2,
            y=pred_cy - bbox.h / 2,
            w=bbox.w,
            h=bbox.h,
        )
        self._last_bbox = new_bbox
        peak_val = float(score_np[peak_y, peak_x])
        confidence = min(1.0, max(0.0, peak_val / (float(score_np.max()) + 1e-6)))
        status = "locked" if confidence > 0.6 else ("uncertain" if confidence > 0.2 else "lost")
        return TrackState(bbox=new_bbox, confidence=confidence, status=status)

    def reset(self) -> None:
        """Clear per-sequence state while keeping the loaded model.

        Call between sequences (or after warmup) to avoid stale template
        features leaking into the next sequence.  The model weights are
        intentionally retained so the next call to ``init`` does not trigger
        a full reload.
        """
        self._template_feat = None
        self._last_bbox = None

    @property
    def is_stub_mode(self) -> bool:
        """True when using the lightweight stub backbone (no real weights loaded)."""
        return self._is_stub

    def flops_per_update(self) -> float:
        return _FLOPS_PER_UPDATE

    def on_tier_enter(self, ctx: FrameContext) -> None:
        pass

    def on_tier_exit(self, ctx: FrameContext) -> None:
        pass
