from __future__ import annotations

from collections import deque

import numpy as np

from uav_tracker.types import BBox

_POS_WINDOW = 20
_PSR_WINDOW = 15
_MIN_FRAMES = 20

# freeze: velocity variance < this * diagonal² for _PSR_WINDOW consecutive frames
_FREEZE_VAR_RATIO = 0.02

# psr decay: slope < this (PSR units/frame) with current PSR below _PSR_DECAY_LEVEL
_PSR_SLOPE_HARD = -10.0
_PSR_SLOPE_STRICT = -15.0
_PSR_DECAY_LEVEL = 500.0

# is_drifted: only flag when tracker "thinks" it's fine
_APCE_CONFIDENCE_FLOOR = 100.0


class VelocityDriftMonitor:
    def __init__(self) -> None:
        self._centers: deque[tuple[float, float]] = deque(maxlen=_POS_WINDOW)
        self._diagonals: deque[float] = deque(maxlen=_POS_WINDOW)
        self._psrs: deque[float] = deque(maxlen=_PSR_WINDOW)
        self._frame_count: int = 0

    def update(self, bbox: BBox) -> None:
        cx = bbox.x + bbox.w / 2.0
        cy = bbox.y + bbox.h / 2.0
        diagonal = (bbox.w ** 2 + bbox.h ** 2) ** 0.5
        self._centers.append((cx, cy))
        self._diagonals.append(diagonal)
        self._frame_count += 1

    def update_psr(self, psr: float) -> None:
        self._psrs.append(psr)

    def drift_score(self) -> float:
        return 0.5 * self._freeze_score() + 0.5 * self._psr_decay_score()

    def is_drifted(self, psr: float, consistency_score: float, apce: float, threshold: float = 0.35) -> bool:
        if self._frame_count < _MIN_FRAMES:
            return False
        if apce < _APCE_CONFIDENCE_FLOOR:
            return False
        score = self.drift_score()
        if score <= threshold:
            return False
        # Guard: require that flow also disagrees OR PSR is decaying very steeply.
        # This blocks false-positives on genuinely static targets (e.g. building1)
        # where zero velocity is correct.
        flow_disagrees = consistency_score < 0.5
        psr_steep = self._psr_slope() < _PSR_SLOPE_STRICT
        return flow_disagrees or psr_steep

    def reset(self) -> None:
        self._centers.clear()
        self._diagonals.clear()
        self._psrs.clear()
        self._frame_count = 0

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _freeze_score(self) -> float:
        centers = list(self._centers)
        if len(centers) < _PSR_WINDOW + 1:
            return 0.0

        # Use the most recent _PSR_WINDOW+1 centers to match the required window
        recent = centers[-((_PSR_WINDOW + 1)):]
        pts = np.array(recent, dtype=np.float64)
        diffs = np.diff(pts, axis=0)           # (_PSR_WINDOW, 2)
        vel_var = float(np.var(diffs))

        diags = list(self._diagonals)
        mean_diag = float(np.mean(diags[-((_PSR_WINDOW + 1)):]))
        diag_sq = mean_diag ** 2 if mean_diag > 0 else 1.0

        return 1.0 if vel_var < _FREEZE_VAR_RATIO * diag_sq else 0.0

    def _psr_slope(self) -> float:
        psrs = list(self._psrs)
        n = len(psrs)
        if n < 5:
            return 0.0
        t = np.arange(n, dtype=np.float64)
        A = np.column_stack([t, np.ones(n)])
        coeffs, *_ = np.linalg.lstsq(A, np.array(psrs, dtype=np.float64), rcond=None)
        return float(coeffs[0])

    def _psr_decay_score(self) -> float:
        psrs = list(self._psrs)
        if len(psrs) < _PSR_WINDOW:
            return 0.0
        current_psr = psrs[-1]
        slope = self._psr_slope()
        return 1.0 if (slope < _PSR_SLOPE_HARD and current_psr < _PSR_DECAY_LEVEL) else 0.0


__all__ = ["VelocityDriftMonitor"]
