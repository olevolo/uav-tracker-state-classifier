"""Online LSTM motion predictor — self-learning via online SGD.

Predicts next-frame bbox center from a rolling window of past bbox centers.
Updates its weights online after each frame (tiny SGD step on MSE prediction error).

Registration: MOTION_PREDICTORS["lstm_online"]
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from uav_tracker.registry import MOTION_PREDICTORS
from uav_tracker.types import BBox

# ---------------------------------------------------------------------------
# Tiny LSTM model (no external deps beyond torch)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


class _LSTMNet(nn.Module):
    """Single-layer LSTM + linear head for (cx, cy) prediction.

    Input per step:  4-d normalised (cx, cy, w, h).
    Output:          predicted (cx, cy) — 2-d.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)

    def forward(
        self,
        x: torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hc_out = self.lstm(x, hc)
        pred = self.head(out[:, -1, :])  # last timestep
        return pred, hc_out


# ---------------------------------------------------------------------------
# Registry implementation
# ---------------------------------------------------------------------------


@MOTION_PREDICTORS.register("lstm_online")
class OnlineLSTMMotionPredictor:
    """Online LSTM motion predictor with per-frame SGD updates.

    The model is initialised with random weights and improves continuously
    as tracking progresses — no offline training data is required.

    Protocol compliance: implements ``MotionPredictor`` from
    ``uav_tracker.ml.motion_predictor.base``.
    """

    name: str = "lstm_online"
    hidden_size: int = 32
    seq_len: int = 10

    def __init__(
        self,
        hidden_size: int = 32,
        seq_len: int = 10,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for OnlineLSTMMotionPredictor. "
                "Install it with: pip install torch"
            )

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lr = lr
        self.device = torch.device(device)

        # Model + optimiser
        self._net = _LSTMNet(hidden_size).to(self.device)
        self._optim = torch.optim.SGD(self._net.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()

        # State
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._history: Deque[BBox] = deque(maxlen=seq_len + 1)
        self._last_pred: BBox | None = None

        # Running normalisation statistics (updated from history)
        self._mean: np.ndarray = np.zeros(4, dtype=np.float64)
        self._std: np.ndarray = np.ones(4, dtype=np.float64)

    # ---------------------------------------------------------------------- #
    # Protocol methods                                                         #
    # ---------------------------------------------------------------------- #

    def predict_next(self, history: list[BBox], timestamps: list[int]) -> BBox:
        """Predict the next bbox from *history*.

        If ``len(history) < seq_len`` the sequence is padded by repeating the
        first available bbox.  The prediction is cached so ``update()`` can
        compute the error against the actual bbox.

        Returns a ``BBox`` whose (w, h) are carried over from the last known box.
        """
        if not history:
            self._last_pred = None
            return BBox(0.0, 0.0, 1.0, 1.0)

        # Update normalisation stats from all available history
        self._update_stats(history)

        # Build input sequence (padded to seq_len)
        seq = self._build_sequence(history)

        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, 4)

        self._net.eval()
        with torch.no_grad():
            pred_tensor, _ = self._net(x, None)

        pred_np = pred_tensor.squeeze(0).cpu().numpy()  # (2,) = (cx_norm, cy_norm)

        # Denormalise back to pixel space
        cx = float(pred_np[0]) * self._std[0] + self._mean[0]
        cy = float(pred_np[1]) * self._std[1] + self._mean[1]

        last = history[-1]
        # Reconstruct BBox from predicted centre + last known size
        pred_bbox = BBox(
            x=cx - last.w / 2.0,
            y=cy - last.h / 2.0,
            w=last.w,
            h=last.h,
        )
        self._last_pred = pred_bbox
        # Keep history for update step
        self._history.clear()
        for b in history:
            self._history.append(b)
        return pred_bbox

    def update(self, actual_bbox: BBox) -> None:
        """Perform one online SGD step using the error between last prediction and *actual_bbox*.

        If ``predict_next`` was never called this is a no-op.
        """
        if self._last_pred is None or len(self._history) == 0:
            return

        history = list(self._history)
        self._update_stats(history)

        # Build input from stored history (excluding the just-revealed frame)
        seq = self._build_sequence(history)
        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Target: normalised (cx, cy) of the actual bbox
        actual_cx = actual_bbox.x + actual_bbox.w / 2.0
        actual_cy = actual_bbox.y + actual_bbox.h / 2.0
        target_cx_norm = (actual_cx - self._mean[0]) / max(self._std[0], 1e-6)
        target_cy_norm = (actual_cy - self._mean[1]) / max(self._std[1], 1e-6)
        target = torch.tensor(
            [[target_cx_norm, target_cy_norm]], dtype=torch.float32, device=self.device
        )

        self._net.train()
        self._optim.zero_grad()
        pred, _ = self._net(x, None)
        loss = self._loss_fn(pred, target)
        loss.backward()
        self._optim.step()

        # Append the actual bbox to history for next prediction
        self._history.append(actual_bbox)
        # Invalidate last prediction so a missing predict_next before update won't double-update
        self._last_pred = None

    def reset(self) -> None:
        """Reset LSTM hidden state, history buffer, and discard the cached prediction."""
        self._hidden = None
        self._history.clear()
        self._last_pred = None
        self._mean[:] = 0.0
        self._std[:] = 1.0

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _bbox_to_array(self, bbox: BBox) -> np.ndarray:
        """Convert BBox to (cx, cy, w, h) float64 array."""
        return np.array(
            [bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0, bbox.w, bbox.h],
            dtype=np.float64,
        )

    def _update_stats(self, history: list[BBox]) -> None:
        """Recompute running mean/std from *history* (for normalisation)."""
        arr = np.array([self._bbox_to_array(b) for b in history], dtype=np.float64)
        self._mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        self._std = np.where(std < 1e-6, 1.0, std)

    def _build_sequence(self, history: list[BBox]) -> np.ndarray:
        """Return a (seq_len, 4) float32 array suitable as LSTM input.

        Pads by repeating the first element if the history is shorter than seq_len.
        Values are normalised by subtracting mean and dividing by std.
        """
        arr = np.array([self._bbox_to_array(b) for b in history], dtype=np.float64)
        # Normalise
        arr = (arr - self._mean) / self._std

        if len(arr) >= self.seq_len:
            seq = arr[-self.seq_len :]
        else:
            pad_count = self.seq_len - len(arr)
            first = arr[:1].repeat(pad_count, axis=0)
            seq = np.concatenate([first, arr], axis=0)

        return seq.astype(np.float32)
