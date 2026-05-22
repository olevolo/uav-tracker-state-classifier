"""OSTrack one-stream transformer tracker (Ye et al., ECCV 2022).

Reference: Ye B. et al., "Joint Feature Learning and Relation Modeling for
Tracking: A One-Stream Framework", ECCV 2022.

Weight source: https://github.com/botaoye/OSTrack
Expected weight path: $UAV_WEIGHTS_ROOT/ostrack/ostrack256_full_ep300.pth.tar

Checkpoint inspection reveals ViT-Base backbone with:
  template: 192x192 (144 patches = 12x12), search: 384x384 (576 patches = 24x24)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, FrameContext, TrackState

logger = logging.getLogger(__name__)

_FLOPS_PER_UPDATE = 4.8e9

# Architecture constants derived from checkpoint
_EMBED_DIM = 768
_NUM_HEADS = 12
_DEPTH = 12
_MLP_RATIO = 4
_PATCH_SIZE = 16
_TEMPLATE_SIZE = 192   # 12x12 = 144 patches
_SEARCH_SIZE = 384     # 24x24 = 576 patches
_N_Z = 144
_N_X = 576
_FEAT_SZ = 24          # sqrt(_N_X)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, _EMBED_DIM, kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, C)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = (_EMBED_DIM // _NUM_HEADS) ** -0.5
        self.qkv = nn.Linear(_EMBED_DIM, _EMBED_DIM * 3)
        self.proj = nn.Linear(_EMBED_DIM, _EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, _NUM_HEADS, C // _NUM_HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = _EMBED_DIM * _MLP_RATIO
        self.fc1 = nn.Linear(_EMBED_DIM, hidden)
        self.fc2 = nn.Linear(hidden, _EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(_EMBED_DIM)
        self.attn = _Attention()
        self.norm2 = nn.LayerNorm(_EMBED_DIM)
        self.mlp = _Mlp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _OSTrackBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, _EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, _EMBED_DIM))
        self.pos_embed_z = nn.Parameter(torch.zeros(1, _N_Z, _EMBED_DIM))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, _N_X, _EMBED_DIM))
        self.patch_embed = _PatchEmbed()
        self.blocks = nn.ModuleList([_Block() for _ in range(_DEPTH)])
        self.norm = nn.LayerNorm(_EMBED_DIM)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_tok = self.patch_embed(z) + self.pos_embed_z   # (B, 144, 768)
        x_tok = self.patch_embed(x) + self.pos_embed_x   # (B, 576, 768)
        tokens = torch.cat([z_tok, x_tok], dim=1)         # (B, 720, 768)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens[:, _N_Z:]                            # search tokens: (B, 576, 768)


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------

def _conv_bn(in_c: int, out_c: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=k // 2),
        nn.BatchNorm2d(out_c),
    )


class _BoxHead(nn.Module):
    """Three-branch head: center score, sub-pixel offset, normalized size."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1_ctr = _conv_bn(_EMBED_DIM, 256)
        self.conv2_ctr = _conv_bn(256, 128)
        self.conv3_ctr = _conv_bn(128, 64)
        self.conv4_ctr = _conv_bn(64, 32)
        self.conv5_ctr = nn.Conv2d(32, 1, 1)

        self.conv1_offset = _conv_bn(_EMBED_DIM, 256)
        self.conv2_offset = _conv_bn(256, 128)
        self.conv3_offset = _conv_bn(128, 64)
        self.conv4_offset = _conv_bn(64, 32)
        self.conv5_offset = nn.Conv2d(32, 2, 1)

        self.conv1_size = _conv_bn(_EMBED_DIM, 256)
        self.conv2_size = _conv_bn(256, 128)
        self.conv3_size = _conv_bn(128, 64)
        self.conv4_size = _conv_bn(64, 32)
        self.conv5_size = nn.Conv2d(32, 2, 1)

    def _run(self, x, c1, c2, c3, c4, c5):
        return c5(F.relu(c4(F.relu(c3(F.relu(c2(F.relu(c1(x)))))))))

    def forward(self, x: torch.Tensor):
        ctr = self._run(x, self.conv1_ctr, self.conv2_ctr, self.conv3_ctr,
                        self.conv4_ctr, self.conv5_ctr).sigmoid()
        offset = self._run(x, self.conv1_offset, self.conv2_offset, self.conv3_offset,
                           self.conv4_offset, self.conv5_offset).sigmoid()
        size = self._run(x, self.conv1_size, self.conv2_size, self.conv3_size,
                         self.conv4_size, self.conv5_size).sigmoid()
        return ctr, offset, size


class _OSTrackNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _OSTrackBackbone()
        self.box_head = _BoxHead()

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        feat = self.backbone(z, x)                                          # (B, N_x, C)
        feat = feat.transpose(1, 2).reshape(-1, _EMBED_DIM, _FEAT_SZ, _FEAT_SZ)
        return self.box_head(feat)                                          # ctr, offset, size


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@TRACKERS.register("ostrack_256")
class OSTrackTracker:
    """OSTrack-256 transformer tracker.

    tier_hint=2 (heavy tracker — 4.8 GFLOPs, targets <30ms on T4 GPU).
    """

    name: str = "ostrack_256"
    tier_hint: int = 2

    # Area factors matching the original OSTrack training config
    _TEMPLATE_FACTOR = 2.0
    _SEARCH_FACTOR = 4.0

    def __init__(
        self,
        device: str = "auto",
        weights_path: str | None = None,
    ) -> None:
        self._device_str = device
        self._weights_path = weights_path
        self._model: _OSTrackNet | None = None
        self._template: torch.Tensor | None = None    # cached template patch tensor
        self._last_bbox: BBox | None = None
        self._is_stub: bool = True

    @property
    def _device(self) -> torch.device:
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    def _load_model(self) -> _OSTrackNet:
        model = _OSTrackNet()
        weights_path = self._weights_path
        if weights_path is None:
            from uav_tracker.paths import weights_root
            weights_path = str(weights_root() / "ostrack" / "ostrack256_full_ep300.pth.tar")
        p = Path(weights_path)
        if p.exists():
            try:
                ckpt = torch.load(p, map_location="cpu")
                state = ckpt.get("net", ckpt)
                missing, unexpected = model.load_state_dict(state, strict=True)
                if missing:
                    logger.warning("OSTrack: missing keys: %s", missing[:5])
                self._is_stub = bool(missing)
                logger.info("OSTrack weights loaded from %s", p)
            except Exception as exc:
                logger.warning("Failed to load OSTrack weights: %s — using random init", exc)
                self._is_stub = True
        else:
            logger.info("OSTrack weights not found at %s — using random init", p)
            self._is_stub = True
        return model.to(self._device).eval()

    def _crop(self, frame: np.ndarray, bbox: BBox, out_size: int, area_factor: float) -> tuple[np.ndarray, float]:
        """Square crop centred on bbox, padded with image mean. Returns (patch, resize_factor)."""
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        crop_sz = max(1, round(math.sqrt(w * h) * area_factor))
        cx, cy = x + w / 2, y + h / 2
        x1 = round(cx - crop_sz / 2)
        y1 = round(cy - crop_sz / 2)
        x2 = x1 + crop_sz
        y2 = y1 + crop_sz

        H, W = frame.shape[:2]
        x1_pad = max(0, -x1)
        y1_pad = max(0, -y1)
        x2_pad = max(x2 - W, 0)
        y2_pad = max(y2 - H, 0)

        crop = frame[y1 + y1_pad: y2 - y2_pad or None, x1 + x1_pad: x2 - x2_pad or None]
        if x1_pad or x2_pad or y1_pad or y2_pad:
            mean_val = frame.mean(axis=(0, 1)).tolist()
            crop = cv2.copyMakeBorder(crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                      cv2.BORDER_CONSTANT, value=mean_val)
        patch = cv2.resize(crop, (out_size, out_size))
        return patch, out_size / crop_sz

    def _to_tensor(self, patch: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        return t.to(self._device)

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        if self._model is None:
            self._model = self._load_model()
        patch, _ = self._crop(frame, bbox, _TEMPLATE_SIZE, self._TEMPLATE_FACTOR)
        self._template = self._to_tensor(patch)
        self._last_bbox = bbox

    def update(self, frame: np.ndarray) -> TrackState:
        if self._model is None or self._template is None or self._last_bbox is None:
            return TrackState(bbox=BBox(0, 0, 1, 1), confidence=0.0, status="lost")

        patch, resize_factor = self._crop(frame, self._last_bbox, _SEARCH_SIZE, self._SEARCH_FACTOR)
        x = self._to_tensor(patch)

        with torch.no_grad():
            ctr, offset, size = self._model(self._template, x)

        # ctr: (1,1,24,24), offset: (1,2,24,24), size: (1,2,24,24) — all sigmoid'd
        ctr_np = ctr[0, 0].cpu().numpy()
        flat_idx = ctr_np.argmax()
        py, px = divmod(int(flat_idx), _FEAT_SZ)
        confidence = float(ctr_np[py, px])

        off_x = float(offset[0, 0, py, px].cpu())
        off_y = float(offset[0, 1, py, px].cpu())
        w_norm = float(size[0, 0, py, px].cpu())
        h_norm = float(size[0, 1, py, px].cpu())

        # Predicted centre in search crop pixels (384x384)
        cx_crop = (px + off_x) / _FEAT_SZ * _SEARCH_SIZE
        cy_crop = (py + off_y) / _FEAT_SZ * _SEARCH_SIZE
        w_crop = w_norm * _SEARCH_SIZE
        h_crop = h_norm * _SEARCH_SIZE

        # Map back to frame via resize_factor (crop_px = frame_px * resize_factor)
        bbox = self._last_bbox
        cx_frame = (bbox.x + bbox.w / 2) + (cx_crop - _SEARCH_SIZE / 2) / resize_factor
        cy_frame = (bbox.y + bbox.h / 2) + (cy_crop - _SEARCH_SIZE / 2) / resize_factor
        w_frame = w_crop / resize_factor
        h_frame = h_crop / resize_factor

        new_bbox = BBox(
            x=cx_frame - w_frame / 2,
            y=cy_frame - h_frame / 2,
            w=max(1.0, w_frame),
            h=max(1.0, h_frame),
        )
        self._last_bbox = new_bbox

        if confidence > 0.6:
            status = "locked"
        elif confidence > 0.2:
            status = "uncertain"
        else:
            status = "lost"

        return TrackState(bbox=new_bbox, confidence=confidence, status=status)

    def update_with_action(self, frame: "np.ndarray", action: "Any") -> "TrackState":
        """Action routing stub — OSTrack does not support CE/search overrides."""
        return self.update(frame)

    def reset(self) -> None:
        self._template = None
        self._last_bbox = None

    @property
    def is_stub_mode(self) -> bool:
        """True when real weights are not loaded."""
        return self._is_stub

    def flops_per_update(self) -> float:
        return _FLOPS_PER_UPDATE

    def on_tier_enter(self, ctx: FrameContext) -> None:
        pass

    def on_tier_exit(self, ctx: FrameContext) -> None:
        pass
