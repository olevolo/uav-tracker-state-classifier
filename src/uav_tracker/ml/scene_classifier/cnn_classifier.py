"""MobileNetV3-Small-based scene classifier.

Architecture:
  - Backbone: MobileNetV3-Small (timm, ImageNet pre-trained if available)
             OR a lightweight 4-layer ConvNet if timm not available
  - Input: 128×128 patch (3 channels, normalized) + 32-d flow features
  - Fusion: backbone_features (576-d) concatenated with flow projection (64-d) → 640-d
  - Head: Linear(640, 6) → 6-class logits
  - Uncertain gate: max softmax probability < confidence_threshold → scene_class=CLEAR

FLOPs: ~0.06 GFLOPs (timm MobileNetV3-Small backbone)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from uav_tracker.ml.scene_classifier.feature_extractor import FlowFeatureExtractor
from uav_tracker.registry import SCENE_CLASSIFIERS
from uav_tracker.types import (
    BBox,
    FrameContext,
    SceneClass,
    SceneClassification,
    TrackState,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Backbone helpers                                                             #
# --------------------------------------------------------------------------- #

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_PATCH_SIZE = 128
_FLOW_DIM = 32
_FLOW_PROJ_DIM = 64


class _FallbackBackbone(nn.Module):
    """Lightweight 4-layer ConvNet used when timm is unavailable."""

    out_features: int = 2048  # 128 * 4 * 4

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, 128, 128) → (B, 2048)
        return self.net(x).flatten(1)


def _build_backbone() -> tuple[nn.Module, int]:
    """Return (backbone_module, out_features).

    Tries timm MobileNetV3-Small first; falls back to _FallbackBackbone.
    """
    try:
        import timm  # type: ignore[import]
        backbone = timm.create_model(
            "mobilenetv3_small_050",
            pretrained=False,
            features_only=False,
            num_classes=0,   # remove classifier head → returns penultimate features
        )
        # Get output dimension via a forward pass on a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, _PATCH_SIZE, _PATCH_SIZE)
            out_features = int(backbone(dummy).shape[1])
        logger.debug("Using timm MobileNetV3-Small backbone (out=%d)", out_features)
        return backbone, out_features
    except Exception as exc:
        logger.debug("timm not available (%s); using fallback ConvNet backbone", exc)
        fb = _FallbackBackbone()
        return fb, fb.out_features


# --------------------------------------------------------------------------- #
# Full model                                                                   #
# --------------------------------------------------------------------------- #


class _SceneClassifierNet(nn.Module):
    """Full forward-pass module: image patch + flow features → 6-class logits."""

    def __init__(self, backbone: nn.Module, backbone_out: int, n_classes: int = 6) -> None:
        super().__init__()
        self.backbone = backbone
        fusion_dim = backbone_out + _FLOW_PROJ_DIM
        self.flow_proj = nn.Sequential(
            nn.Linear(_FLOW_DIM, _FLOW_PROJ_DIM),
            nn.ReLU(),
        )
        self.head = nn.Linear(fusion_dim, n_classes)

    def forward(
        self,
        patch: torch.Tensor,         # (B, 3, 128, 128)
        flow_feat: torch.Tensor,     # (B, 32)
    ) -> torch.Tensor:               # (B, n_classes)  raw logits
        img_feat = self.backbone(patch)           # (B, backbone_out)
        flow_projected = self.flow_proj(flow_feat)  # (B, 64)
        fused = torch.cat([img_feat, flow_projected], dim=1)  # (B, fusion_dim)
        return self.head(fused)


# --------------------------------------------------------------------------- #
# Registered classifier                                                        #
# --------------------------------------------------------------------------- #


@SCENE_CLASSIFIERS.register("mobilenetv3_tiny")
class MobileNetV3TinyClassifier:
    """Lightweight CNN scene classifier satisfying the SceneClassifier Protocol.

    Uses a MobileNetV3-Small backbone (timm) or a 4-layer ConvNet fallback.
    Model weights are loaded lazily on the first call to ``classify``.

    Parameters
    ----------
    weights_path :
        Path to a saved ``state_dict`` (``.pth``).  ``None`` → random weights
        (useful for testing and integration without trained weights).
    device :
        ``"cpu"`` or ``"cuda"``.  Defaults to ``"cpu"`` for the 3 ms budget.
    dtype :
        ``"float32"`` or ``"float16"``.
    classify_interval :
        Run the CNN every *N* frames; return the cached result between calls.
    confidence_threshold :
        If ``max_softmax_prob < threshold``, the result is gated to
        ``SceneClass.CLEAR`` with the raw ``max_softmax_prob`` as confidence.
    """

    name: str = "mobilenetv3_tiny"
    n_classes: int = 6
    classify_interval: int = 5

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        classify_interval: int = 5,
        confidence_threshold: float = 0.60,
    ) -> None:
        self.classify_interval = classify_interval
        self._weights_path = weights_path
        self._device_str = device
        self._dtype = torch.float16 if dtype == "float16" else torch.float32
        self._confidence_threshold = confidence_threshold

        # Lazy-loaded model
        self._model: Optional[_SceneClassifierNet] = None
        self._device: Optional[torch.device] = None

        # Feature extractor (stateful: carries EMA/deque across frames)
        self._feature_extractor = FlowFeatureExtractor()

        # Frame counter & cached result
        self._frame_counter: int = 0
        self._cached_result: Optional[SceneClassification] = None

    # ---------------------------------------------------------------------- #
    # Protocol methods                                                        #
    # ---------------------------------------------------------------------- #

    def classify(self, ctx: FrameContext, state: TrackState) -> SceneClassification:
        """Run classification every ``classify_interval`` frames; cache between calls.

        Parameters
        ----------
        ctx :
            Current-frame context.
        state :
            Current tracker output.

        Returns
        -------
        SceneClassification
            A fresh result every ``classify_interval`` frames; the previous
            result (with updated ``frame_idx``) on intervening frames.
        """
        self._frame_counter += 1

        # Return cached result on non-classification frames
        if (
            self._cached_result is not None
            and self._frame_counter % self.classify_interval != 0
        ):
            # Update frame_idx so callers always get the current index
            cached = self._cached_result
            return SceneClassification(
                scene_class=cached.scene_class,
                probabilities=cached.probabilities,
                confidence=cached.confidence,
                frame_idx=ctx.frame_idx,
                aux=cached.aux,
            )

        # Lazy model load
        if self._model is None:
            self._load_model()

        assert self._model is not None
        assert self._device is not None

        # Extract flow features
        flow_feat = self._feature_extractor.extract(ctx, state)  # (32,) float32

        # Prepare image patch
        patch_tensor = self._prepare_patch(ctx)  # (1, 3, 128, 128)
        flow_tensor = torch.from_numpy(flow_feat).unsqueeze(0).to(
            self._device, dtype=self._dtype
        )  # (1, 32)

        # Inference
        self._model.eval()
        with torch.no_grad():
            logits = self._model(patch_tensor, flow_tensor)  # (1, 6)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # (6,)

        probs_np = probs.cpu().float().numpy()  # always float32 for output
        max_prob = float(probs_np.max())
        predicted_idx = int(probs_np.argmax())

        # Uncertain gate
        if max_prob < self._confidence_threshold:
            scene_class = SceneClass.CLEAR
        else:
            scene_class = SceneClass(predicted_idx)

        result = SceneClassification(
            scene_class=scene_class,
            probabilities=probs_np,
            confidence=max_prob,
            frame_idx=ctx.frame_idx,
        )
        self._cached_result = result
        return result

    def update_online(
        self,
        ctx: FrameContext,
        state: TrackState,
        feedback: SceneClass,
    ) -> None:
        """No-op for Phase 12; implemented in Phase 14 online adaptation."""
        pass

    def reset(self) -> None:
        """Reset per-sequence state (frame counter, cache, feature extractor)."""
        self._frame_counter = 0
        self._cached_result = None
        self._feature_extractor.reset()

    # ---------------------------------------------------------------------- #
    # Public extras (matching plan extract_features signature)                #
    # ---------------------------------------------------------------------- #

    def extract_features(
        self, ctx: FrameContext, state: TrackState
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Return ``(patch_tensor, flow_feature_vector)`` for external use.

        Parameters
        ----------
        ctx :
            Current-frame context.
        state :
            Current tracker output.

        Returns
        -------
        tuple[torch.Tensor, np.ndarray]
            ``patch_tensor`` has shape ``(1, 3, 128, 128)`` on ``self._device``;
            ``flow_feature_vector`` is a float32 ndarray of shape ``(32,)``.
        """
        if self._model is None:
            self._load_model()
        patch = self._prepare_patch(ctx)
        flow = self._feature_extractor.extract(ctx, state)
        return patch, flow

    def _ensure_model_loaded(self) -> None:
        """Eagerly trigger model construction (backbone + head) without classifying.

        Calling this before any classify() call pre-warms the model so the first
        call to classify() is latency-free.  Safe to call multiple times.
        """
        if self._model is None:
            self._load_model()

    # ---------------------------------------------------------------------- #
    # Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _load_model(self) -> None:
        """Build the model, optionally load weights, move to device."""
        self._device = torch.device(self._device_str)
        backbone, backbone_out = _build_backbone()
        self._model = _SceneClassifierNet(backbone, backbone_out, self.n_classes)
        if self._weights_path is not None:
            try:
                state_dict = torch.load(
                    self._weights_path,
                    map_location=self._device,
                    weights_only=True,
                )
                self._model.load_state_dict(state_dict)
                logger.info("Loaded scene classifier weights from %s", self._weights_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load weights from %s: %s — using random weights",
                    self._weights_path,
                    exc,
                )
        self._model.to(self._device, dtype=self._dtype)
        self._model.eval()
        logger.debug(
            "Scene classifier loaded (device=%s, dtype=%s)",
            self._device,
            self._dtype,
        )

    def _prepare_patch(self, ctx: FrameContext) -> torch.Tensor:
        """Crop a 128×128 patch around the tracker ROI and normalise to ImageNet stats.

        Returns a ``(1, 3, 128, 128)`` float tensor on ``self._device``.
        """
        assert self._device is not None

        frame = ctx.frame  # H × W × 3  BGR uint8 (OpenCV convention)
        fh, fw = frame.shape[:2]

        if ctx.bbox is not None:
            bx, by, bw, bh = ctx.bbox.x, ctx.bbox.y, ctx.bbox.w, ctx.bbox.h
            # 2× context crop
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            half = max(bw, bh)
            x1 = int(max(0, cx - half))
            y1 = int(max(0, cy - half))
            x2 = int(min(fw, cx + half))
            y2 = int(min(fh, cy + half))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
            else:
                crop = frame
        else:
            crop = frame

        # Resize to 128×128
        patch = cv2.resize(crop, (_PATCH_SIZE, _PATCH_SIZE), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB → float32 in [0, 1]
        patch_rgb = patch[:, :, ::-1].astype(np.float32) / 255.0

        # ImageNet normalisation
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)
        patch_norm = (patch_rgb - mean) / std  # (128, 128, 3)

        # HWC → CHW
        patch_chw = np.ascontiguousarray(patch_norm.transpose(2, 0, 1))  # (3, 128, 128)
        tensor = (
            torch.from_numpy(patch_chw)
            .unsqueeze(0)
            .to(self._device, dtype=self._dtype)
        )  # (1, 3, 128, 128)
        return tensor


__all__ = ["MobileNetV3TinyClassifier"]
