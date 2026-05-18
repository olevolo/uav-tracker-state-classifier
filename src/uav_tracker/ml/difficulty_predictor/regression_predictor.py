"""MLP regression predictor for expected IoU drop over next K frames.

Reference: Phase 12, docs/v2-plan.md §5 Phase 12.
Registration: DIFFICULTY_PREDICTORS["mlp_regressor"]

Architecture:
  Input: flow_features(32) + scene_class_probs(6) + confidence_history(10) = 48-d
  Layers: Linear(48,128) -> ReLU -> Dropout(0.2) -> Linear(128,64) -> ReLU -> Linear(64,1) -> Sigmoid
  Output: scalar [0, 1] -- expected max IoU drop in next horizon_frames frames
  Training: supervised on (frame_features) -> (actual IoU drop from label_generator traces)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from uav_tracker.registry import DIFFICULTY_PREDICTORS
from uav_tracker.types import DifficultyPrediction, FrameContext, TrackState

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

_INPUT_DIM = 48
_FLOW_DIM = 32
_SCENE_DIM = 6
_HIST_DIM = 10

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as _nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _nn = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# MLP definition (only constructed when torch is available)
# ---------------------------------------------------------------------------


def _build_mlp() -> "torch.nn.Module":
    """Return the 48 -> 1 Sigmoid MLP."""
    return _nn.Sequential(
        _nn.Linear(_INPUT_DIM, 128),
        _nn.ReLU(),
        _nn.Dropout(0.2),
        _nn.Linear(128, 64),
        _nn.ReLU(),
        _nn.Linear(64, 1),
        _nn.Sigmoid(),
    )


# ---------------------------------------------------------------------------
# Flow feature extraction
# ---------------------------------------------------------------------------


def _extract_flow_features(ctx: FrameContext) -> np.ndarray:
    """Extract a 32-d flow feature vector from the FrameContext.

    Uses optical flow cache when available (avoids recomputing LK flow).
    Falls back to a zero vector if no flow data is present.

    Features (32-d):
      [0:8]   — 8-bin histogram of flow magnitudes inside the ROI
      [8:16]  — 8-bin histogram of flow angles inside the ROI (0-360 deg, mapped to [0,1])
      [16]    — mean magnitude
      [17]    — std magnitude
      [18]    — mean angle (normalised 0-1)
      [19]    — fraction of points with magnitude > 2 px (high-motion indicator)
      [20:24] — normalised bbox: cx/W, cy/H, w/W, h/H  (or zeros if no bbox)
      [24:32] — padding zeros (reserved for future features)
    """
    feats = np.zeros(_FLOW_DIM, dtype=np.float32)

    frame = ctx.frame
    fH, fW = frame.shape[:2]

    # --- bbox normalised coords ---
    if ctx.bbox is not None:
        bb = ctx.bbox
        cx = (bb.x + bb.w / 2.0) / max(fW, 1)
        cy = (bb.y + bb.h / 2.0) / max(fH, 1)
        nw = bb.w / max(fW, 1)
        nh = bb.h / max(fH, 1)
        feats[20] = float(np.clip(cx, 0.0, 1.0))
        feats[21] = float(np.clip(cy, 0.0, 1.0))
        feats[22] = float(np.clip(nw, 0.0, 1.0))
        feats[23] = float(np.clip(nh, 0.0, 1.0))

    # --- try to get flow vectors from cache ---
    cache = ctx.optical_flow_cache
    flow_pts: np.ndarray | None = None
    flow_vecs: np.ndarray | None = None

    if cache is not None:
        # Common cache keys used by signals in this project.
        for key in ("residual_flow", "local_flow", "flow_vectors"):
            val = cache.get(key)
            if val is not None and isinstance(val, np.ndarray) and val.ndim >= 2:
                flow_vecs = val.reshape(-1, 2)
                break
        for key in ("flow_points", "points", "prev_pts"):
            val = cache.get(key)
            if val is not None and isinstance(val, np.ndarray):
                flow_pts = val.reshape(-1, 2)
                break

    if flow_vecs is None or len(flow_vecs) == 0:
        # No flow data available — return the bbox-populated vector with zeros elsewhere.
        return feats

    magnitudes = np.linalg.norm(flow_vecs, axis=1).astype(np.float32)
    angles_rad = np.arctan2(flow_vecs[:, 1], flow_vecs[:, 0])  # [-pi, pi]
    angles_deg = np.degrees(angles_rad) % 360.0  # [0, 360)

    # 8-bin magnitude histogram (0-8+ px range, clipped to [0, 8])
    mag_clipped = np.clip(magnitudes, 0.0, 8.0)
    mag_hist, _ = np.histogram(mag_clipped, bins=8, range=(0.0, 8.0))
    n = max(len(magnitudes), 1)
    feats[0:8] = (mag_hist / n).astype(np.float32)

    # 8-bin angle histogram (0-360 deg)
    ang_hist, _ = np.histogram(angles_deg, bins=8, range=(0.0, 360.0))
    feats[8:16] = (ang_hist / n).astype(np.float32)

    # Scalar statistics
    feats[16] = float(np.clip(magnitudes.mean() / 10.0, 0.0, 1.0))  # normalised by 10 px
    feats[17] = float(np.clip(magnitudes.std() / 10.0, 0.0, 1.0))
    feats[18] = float(angles_deg.mean() / 360.0)
    feats[19] = float((magnitudes > 2.0).mean())

    return feats


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------


@DIFFICULTY_PREDICTORS.register("mlp_regressor")
class MLPDifficultyPredictor:
    """MLP regression predictor for expected IoU drop over next horizon_frames.

    When weights are not loaded all predictions return 0.5 (maximum uncertainty).

    Parameters
    ----------
    weights_path:
        Path to a ``torch.save``-d state-dict file.  If ``None`` the MLP is
        constructed but left at random init; predictions return 0.5 until
        weights are loaded.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    horizon_frames:
        How many frames ahead the prediction covers.
    """

    name: str = "mlp_regressor"

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cpu",
        horizon_frames: int = 10,
    ) -> None:
        self.horizon_frames = horizon_frames
        self.device = device
        self._weights_loaded: bool = False
        self._confidence_history: deque[float] = deque(maxlen=_HIST_DIM)
        self._net = None

        if _TORCH_AVAILABLE:
            self._net = _build_mlp()
            if weights_path is not None:
                self._load_weights(weights_path)
            self._net.eval()
            try:
                self._net.to(device)
            except Exception as exc:  # pragma: no cover
                _log.warning("MLPDifficultyPredictor: cannot move model to %s: %s", device, exc)
        else:  # pragma: no cover
            _log.warning("MLPDifficultyPredictor: torch not available; predictions will be 0.5")

    # -----------------------------------------------------------------------
    # DifficultyPredictor Protocol
    # -----------------------------------------------------------------------

    def predict(self, ctx: FrameContext, history: list[TrackState]) -> DifficultyPrediction:
        """Extract 48-d features from ctx + history, run MLP, return DifficultyPrediction.

        Parameters
        ----------
        ctx:
            Current frame context.  May be ``FrameContext`` or ``FrameContextV2``.
        history:
            Recent track states (most recent last).  Used for rolling confidence
            history.  May be empty.

        Returns
        -------
        DifficultyPrediction
            ``expected_iou_drop`` in [0, 1].  0.5 when no weights loaded.
        """
        # --- update rolling confidence history ---
        if history:
            for state in history[-_HIST_DIM:]:
                self._confidence_history.append(float(state.confidence))
        elif ctx.bbox is not None:
            # No history provided; append a neutral 1.0 confidence.
            self._confidence_history.append(1.0)

        # Build feature vector.
        feature_vec = self._build_feature_vector(ctx)

        # Fallback: no weights or no torch.
        if not self._weights_loaded or self._net is None or not _TORCH_AVAILABLE:
            return DifficultyPrediction(
                expected_iou_drop=0.5,
                horizon_frames=self.horizon_frames,
                feature_vector=feature_vec,
                aux={"weights_loaded": False},
            )

        # Forward pass.
        try:
            with torch.no_grad():
                x = torch.from_numpy(feature_vec).unsqueeze(0).to(self.device)
                out = self._net(x)
                score = float(out.squeeze().item())
        except Exception as exc:  # pragma: no cover
            _log.warning("MLPDifficultyPredictor.predict: forward pass failed: %s", exc)
            score = 0.5

        return DifficultyPrediction(
            expected_iou_drop=float(np.clip(score, 0.0, 1.0)),
            horizon_frames=self.horizon_frames,
            feature_vector=feature_vec,
            aux={"weights_loaded": True},
        )

    def reset(self) -> None:
        """Clear rolling confidence history. Idempotent."""
        self._confidence_history.clear()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_feature_vector(self, ctx: FrameContext) -> np.ndarray:
        """Construct the 48-d input feature vector.

        Layout:
          [0:32]  — flow features from ctx.optical_flow_cache + ctx.bbox
          [32:38] — scene class probabilities (uniform if not available)
          [38:48] — rolling confidence history (last 10, zero-padded at start)
        """
        feats = np.zeros(_INPUT_DIM, dtype=np.float32)

        # --- 32-d flow features ---
        feats[0:_FLOW_DIM] = _extract_flow_features(ctx)

        # --- 6-d scene class probabilities ---
        # FrameContextV2 may carry a scene_classification; FrameContext does not.
        sc_probs = self._get_scene_probs(ctx)
        feats[_FLOW_DIM : _FLOW_DIM + _SCENE_DIM] = sc_probs

        # --- 10-d confidence history ---
        hist = list(self._confidence_history)
        n = len(hist)
        if n > 0:
            conf_arr = np.zeros(_HIST_DIM, dtype=np.float32)
            # Right-align: most recent value at index 9.
            conf_arr[_HIST_DIM - n :] = np.array(hist[-_HIST_DIM:], dtype=np.float32)
            feats[_FLOW_DIM + _SCENE_DIM :] = conf_arr

        return feats

    @staticmethod
    def _get_scene_probs(ctx: FrameContext) -> np.ndarray:
        """Extract the 6-d scene-class probability vector from ctx.

        Returns a uniform [1/6, ...] vector when no classification is present
        (i.e., when ctx is plain FrameContext or scene_classification is None).
        """
        uniform = np.full(_SCENE_DIM, 1.0 / _SCENE_DIM, dtype=np.float32)
        # FrameContextV2 has scene_classification; FrameContext does not.
        sc = getattr(ctx, "scene_classification", None)
        if sc is None:
            return uniform
        probs = getattr(sc, "probabilities", None)
        if probs is None:
            return uniform
        if not isinstance(probs, np.ndarray):
            try:
                probs = np.asarray(probs, dtype=np.float32)
            except Exception:
                return uniform
        if probs.shape != (_SCENE_DIM,):
            if probs.size == _SCENE_DIM:
                probs = probs.reshape(_SCENE_DIM).astype(np.float32)
            else:
                return uniform
        return probs.astype(np.float32)

    def _load_weights(self, weights_path: str) -> None:
        """Load a torch state-dict from *weights_path* into self._net."""
        if not _TORCH_AVAILABLE or self._net is None:
            _log.warning(
                "MLPDifficultyPredictor: torch not available; cannot load weights from %s",
                weights_path,
            )
            return
        try:
            import pathlib

            path = pathlib.Path(weights_path)
            if not path.exists():
                _log.warning(
                    "MLPDifficultyPredictor: weights file not found: %s", weights_path
                )
                return
            state = torch.load(path, map_location=self.device)
            # Support plain state-dict or {"state_dict": ...} wrappers.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._net.load_state_dict(state)
            self._weights_loaded = True
            _log.info("MLPDifficultyPredictor: loaded weights from %s", weights_path)
        except Exception as exc:
            _log.warning(
                "MLPDifficultyPredictor: failed to load weights from %s: %s",
                weights_path,
                exc,
            )


__all__ = ["MLPDifficultyPredictor"]
