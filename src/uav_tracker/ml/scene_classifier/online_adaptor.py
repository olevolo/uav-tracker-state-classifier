"""Online adaptation for scene classifier — updates per-target without full retraining.

Uses a small replay buffer of recent (features, label) pairs and performs
periodic SGD micro-updates on the classifier head only (backbone frozen).

Self-learning component: improves scene class predictions for the current
target as tracking proceeds without any offline training.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import numpy as np

from uav_tracker.types import FrameContext, SceneClass, TrackState

logger = logging.getLogger(__name__)


class SceneClassifierOnlineAdaptor:
    """Adapts the scene classifier head online during tracking.

    Strategy
    --------
    - Buffer last ``buffer_size`` (feature_vector, scene_class) pairs using a
      FIFO deque.
    - Every ``adapt_interval`` frames, when the tracker is confident and the
      buffer has accumulated at least ``min_buffer_size`` samples:

        * Sample a random mini-batch from the buffer.
        * Freeze the backbone; update only the classification head.
        * Run ``n_adapt_steps`` SGD steps with learning rate ``adapt_lr``.

    - Disabled (no update) when ``TrackState.status == 'lost'``.
    - Disabled when buffer size < ``min_buffer_size``.
    - Gracefully degrades to buffer-only mode when the classifier has no
      ``_model`` loaded yet or no optimizer can be created.

    Parameters
    ----------
    classifier:
        A ``MobileNetV3TinyClassifier`` instance (or any object that exposes
        ``_model``, ``_feature_extractor``, ``_device``, and ``_dtype``).
    buffer_size:
        Maximum number of (feature, label) pairs to retain (FIFO eviction).
    adapt_interval:
        Trigger an SGD micro-update every N frames (when conditions are met).
    adapt_lr:
        Learning rate for the online classification-head SGD update.
    min_buffer_size:
        Do not adapt until at least this many samples are in the buffer.
    n_adapt_steps:
        Number of SGD steps to perform per adaptation event (1–5 recommended).
    mini_batch_size:
        Number of samples drawn from the buffer per SGD step.
    confidence_gate:
        Minimum ``TrackState.confidence`` required to store a sample and
        trigger adaptation.
    """

    def __init__(
        self,
        classifier: Any,
        buffer_size: int = 100,
        adapt_interval: int = 20,
        adapt_lr: float = 1e-4,
        min_buffer_size: int = 20,
        n_adapt_steps: int = 3,
        mini_batch_size: int = 8,
        confidence_gate: float = 0.5,
    ) -> None:
        self._classifier = classifier
        self._buffer_size = int(buffer_size)
        self._adapt_interval = int(adapt_interval)
        self._adapt_lr = float(adapt_lr)
        self._min_buffer_size = int(min_buffer_size)
        self._n_adapt_steps = int(n_adapt_steps)
        self._mini_batch_size = int(mini_batch_size)
        self._confidence_gate = float(confidence_gate)

        # (flow_feature_vector, scene_class_int) pairs
        self._buffer: deque[tuple[np.ndarray, int]] = deque(maxlen=self._buffer_size)
        self._frame_count: int = 0
        self._adapt_count: int = 0  # total adaptation events completed

        # Lazily-created optimizer (one per session; recreated on reset)
        self._optimizer: Any = None

    # ------------------------------------------------------------------
    # Public API

    def step(self, ctx: FrameContext, state: TrackState) -> None:
        """Called every frame.

        Accumulates the replay buffer from high-confidence frames and
        triggers a classification-head SGD micro-update every
        ``adapt_interval`` frames when conditions are met.

        Parameters
        ----------
        ctx:
            Current-frame context.
        state:
            Tracker output for this frame.
        """
        self._frame_count += 1

        # Never adapt when the tracker has lost the target.
        if state.status == "lost":
            return

        # Only store high-confidence frames as supervision signal.
        if state.confidence < self._confidence_gate:
            return

        # Derive pseudo-label from current scene classification (if available).
        scene_class_int = self._derive_label(ctx, state)
        if scene_class_int is None:
            return

        # Extract flow features for the replay buffer.
        feature_vec = self._extract_flow_features(ctx, state)
        if feature_vec is None:
            return

        self._buffer.append((feature_vec, scene_class_int))

        # Check adaptation trigger.
        buffer_ready = len(self._buffer) >= self._min_buffer_size
        interval_hit = (self._frame_count % self._adapt_interval) == 0
        if buffer_ready and interval_hit:
            self._adapt()

    def reset(self) -> None:
        """Discard the replay buffer and per-sequence counters.

        Does NOT restore base model weights (head-only updates in this
        implementation are small enough that resetting the buffer is
        sufficient; a hard weight reload is reserved for RECOVERY
        transitions per ADR-0015).
        """
        self._buffer.clear()
        self._frame_count = 0
        self._adapt_count = 0
        self._optimizer = None
        logger.debug("SceneClassifierOnlineAdaptor: buffer cleared")

    # ------------------------------------------------------------------
    # Properties / diagnostics

    @property
    def buffer_len(self) -> int:
        """Current number of samples in the replay buffer."""
        return len(self._buffer)

    @property
    def adapt_count(self) -> int:
        """Total number of adaptation events completed this session."""
        return self._adapt_count

    # ------------------------------------------------------------------
    # Private helpers

    def _derive_label(self, ctx: FrameContext, state: TrackState) -> Optional[int]:
        """Return a scene-class integer label for the current frame.

        Uses the classifier's cached ``_cached_result`` when available;
        otherwise defaults to ``SceneClass.CLEAR`` as a conservative
        fallback so the buffer still accumulates.
        """
        cached = getattr(self._classifier, "_cached_result", None)
        if cached is not None:
            return int(cached.scene_class)

        # Fallback: check FrameContext for v2 scene_classification field.
        sc = getattr(ctx, "scene_classification", None)
        if sc is not None:
            return int(sc.scene_class)

        # Conservative default.
        return int(SceneClass.CLEAR)

    def _extract_flow_features(
        self, ctx: FrameContext, state: TrackState
    ) -> Optional[np.ndarray]:
        """Extract a 32-d flow feature vector via the classifier's extractor."""
        extractor = getattr(self._classifier, "_feature_extractor", None)
        if extractor is None:
            return None
        try:
            return extractor.extract(ctx, state)  # (32,) float32
        except Exception as exc:
            logger.debug("OnlineAdaptor: feature extraction failed: %s", exc)
            return None

    def _adapt(self) -> None:
        """Run one adaptation event: N SGD steps on the classifier head.

        Backbone parameters are frozen (``requires_grad=False`` during the
        update) so only the final ``Linear`` layer is modified. Skips
        gracefully if PyTorch is unavailable or the model isn't loaded.
        """
        model = getattr(self._classifier, "_model", None)
        if model is None:
            # Model hasn't been loaded yet — buffer-only mode.
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return

        device = getattr(self._classifier, "_device", None)
        dtype = getattr(self._classifier, "_dtype", torch.float32)
        if device is None:
            return

        # Build or reuse the head-only optimizer.
        if self._optimizer is None:
            head = getattr(model, "head", None)
            if head is None or not isinstance(head, nn.Module):
                logger.debug("OnlineAdaptor: no 'head' layer found — skipping adapt")
                return
            self._optimizer = torch.optim.SGD(
                head.parameters(),
                lr=self._adapt_lr,
                momentum=0.9,
            )

        criterion = nn.CrossEntropyLoss()

        # Freeze backbone (flow_proj + backbone weights).
        backbone = getattr(model, "backbone", None)
        flow_proj = getattr(model, "flow_proj", None)
        for module in filter(None, [backbone, flow_proj]):
            for param in module.parameters():
                param.requires_grad_(False)

        model.train()
        self._optimizer.zero_grad()

        for _step in range(self._n_adapt_steps):
            batch = self._sample_batch()
            if not batch:
                break

            flow_vecs, labels = zip(*batch)
            flow_arr = np.stack(flow_vecs, axis=0)  # (B, 32)
            label_arr = np.array(labels, dtype=np.int64)  # (B,)

            flow_t = torch.from_numpy(flow_arr).to(device, dtype=dtype)
            label_t = torch.from_numpy(label_arr).to(device)

            # We only have flow features here; use zeros for the image path
            # so only the flow→head path is updated. The backbone contribution
            # is zeroed through the projection before the head.
            B = flow_t.shape[0]
            flow_proj_layer = getattr(model, "flow_proj", None)
            head_layer = getattr(model, "head", None)
            if flow_proj_layer is None or head_layer is None:
                break

            with torch.set_grad_enabled(True):
                # Use dummy backbone features (zeros) — only flow path updated.
                backbone_out = getattr(model.backbone, "out_features", None)
                if backbone_out is None:
                    # Try to infer from a forward pass.
                    try:
                        with torch.no_grad():
                            dummy_patch = torch.zeros(
                                1, 3, 128, 128, device=device, dtype=dtype
                            )
                            backbone_out = int(model.backbone(dummy_patch).shape[1])
                    except Exception:
                        break

                backbone_feat = torch.zeros(B, backbone_out, device=device, dtype=dtype)
                flow_feat_proj = flow_proj_layer(flow_t)  # (B, 64)
                fused = torch.cat([backbone_feat, flow_feat_proj], dim=1)
                logits = head_layer(fused)  # (B, n_classes)

                loss = criterion(logits, label_t)

            loss.backward()
            # Gradient clipping (ADR-0015: max norm 0.5).
            torch.nn.utils.clip_grad_norm_(head_layer.parameters(), max_norm=0.5)
            self._optimizer.step()
            self._optimizer.zero_grad()

        # Restore eval mode; unfreeze backbone (leave it as-is for inference).
        model.eval()
        if backbone is not None:
            for param in backbone.parameters():
                param.requires_grad_(True)
        if flow_proj is not None:
            for param in flow_proj.parameters():
                param.requires_grad_(True)

        self._adapt_count += 1
        logger.debug(
            "OnlineAdaptor: adaptation #%d complete (buffer=%d)",
            self._adapt_count,
            len(self._buffer),
        )

    def _sample_batch(self) -> list[tuple[np.ndarray, int]]:
        """Return a random mini-batch from the buffer (up to mini_batch_size)."""
        buf = list(self._buffer)
        if not buf:
            return []
        size = min(self._mini_batch_size, len(buf))
        indices = np.random.choice(len(buf), size=size, replace=False)
        return [buf[i] for i in indices]


__all__ = ["SceneClassifierOnlineAdaptor"]
