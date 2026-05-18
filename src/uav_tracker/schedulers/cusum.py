"""CUSUMScheduler — change-point detection scheduler (Phase 5).

Uses the ``ruptures`` library (Pelt segmentation with L2 cost) to detect
change-points in the recent signal history. If a change-point is detected
within the last ``lookback`` frames the scheduler escalates to tier 1;
otherwise it stays at tier 0.

**Dependency note:** ``ruptures`` is an optional dependency. If it is not
installed, the scheduler registers successfully but ``decide()`` raises a
``RuntimeError`` with instructions to install it:

    uv pip install ruptures

or:

    pip install ruptures

Registration key: ``"cusum"``
"""

from __future__ import annotations

from collections import deque

from uav_tracker.registry import SCHEDULERS
from uav_tracker.types import SchedulerDecision, SignalReport

try:
    import ruptures as _ruptures  # type: ignore[import]
    _RUPTURES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ruptures = None  # type: ignore[assignment]
    _RUPTURES_AVAILABLE = False

_LIGHT = 0
_DEEP = 1


@SCHEDULERS.register("cusum")
class CUSUMScheduler:
    """Change-point detection scheduler via ``ruptures.Pelt``.

    Maintains a rolling window of the last ``history_len`` reliable signal
    values. On each frame, runs ``ruptures.Pelt(model="l2").fit_predict``
    with the configured ``penalty``. If the last change-point (if any) falls
    within the last ``lookback`` frames, escalates to tier 1.

    Parameters
    ----------
    penalty:
        Pelt penalty parameter (default 3.0). Higher values produce fewer
        change-points (less sensitive to small shifts).
    min_size:
        Minimum segment length passed to Pelt (default 5 frames).
    lookback:
        How many recent frames to search for a fresh change-point. If a
        change-point is detected within this window, tier 1 is selected
        (default 10 frames).
    history_len:
        Length of the rolling history window passed to Pelt (default 50).
    cooldown_frames:
        Frames to hold tier 1 after a change-point is detected before
        allowing a return to tier 0 (default 5).
    signal_name:
        Which signal key to consume (default ``"motion_entropy"``).
    """

    name: str = "cusum"
    tiers: int = 2

    def __init__(
        self,
        penalty: float = 3.0,
        min_size: int = 5,
        lookback: int = 10,
        history_len: int = 50,
        cooldown_frames: int = 5,
        signal_name: str = "motion_entropy",
    ) -> None:
        self.penalty = penalty
        self.min_size = min_size
        self.lookback = lookback
        self.history_len = history_len
        self.cooldown_frames = cooldown_frames
        self.signal_name = signal_name

        self._history: deque[float] = deque(maxlen=history_len)
        self._tier: int = _LIGHT
        self._cooldown_left: int = 0

    # ------------------------------------------------------------------

    def decide(
        self,
        signals: dict[str, SignalReport],
        current_tier: int,
        frame_idx: int,
    ) -> SchedulerDecision:
        """Advance state machine and return tier decision.

        Raises
        ------
        RuntimeError
            If ``ruptures`` is not installed.
        """
        if not _RUPTURES_AVAILABLE:
            raise RuntimeError(
                "CUSUMScheduler requires the 'ruptures' package. "
                "Install it with:  uv pip install ruptures"
            )

        prev_tier = self._tier
        value, reliable = self._pick_signal(signals)

        if not reliable:
            return SchedulerDecision(
                tier=self._tier,
                reason=f"unreliable signal — holding tier {self._tier}",
                switched=False,
            )

        self._history.append(value)

        # Cooldown: hold current tier, tick down.
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return SchedulerDecision(
                tier=self._tier,
                reason=f"cooldown ({self._cooldown_left + 1} frames left)",
                switched=False,
            )

        hist = list(self._history)
        n = len(hist)

        # Need at least 2 * min_size samples for Pelt to be meaningful.
        if n < 2 * self.min_size:
            return SchedulerDecision(
                tier=_LIGHT,
                reason=f"history too short ({n} < {2 * self.min_size}); staying tier 0",
                switched=self._tier != _LIGHT,
            )

        import numpy as np

        signal_arr = np.array(hist).reshape(-1, 1)
        algo = _ruptures.Pelt(model="l2", min_size=self.min_size).fit(signal_arr)
        breakpoints = algo.predict(pen=self.penalty)
        # breakpoints includes the last index (== n) as a sentinel.
        internal_bps = [bp for bp in breakpoints if bp < n]

        change_detected = any(n - bp <= self.lookback for bp in internal_bps)

        if change_detected:
            if self._tier == _LIGHT:
                self._tier = _DEEP
                self._cooldown_left = self.cooldown_frames
                switched = True
                reason = (
                    f"CUSUM: change-point detected within last {self.lookback} frames "
                    f"(breakpoints={internal_bps})"
                )
            else:
                switched = False
                reason = f"CUSUM: change-point detected; already at tier 1"
        else:
            if self._tier == _DEEP:
                self._tier = _LIGHT
                self._cooldown_left = self.cooldown_frames
                switched = True
                reason = "CUSUM: no recent change-point → return to tier 0"
            else:
                switched = False
                reason = "CUSUM: no recent change-point → staying tier 0"

        return SchedulerDecision(tier=self._tier, reason=reason, switched=switched)

    def reset(self) -> None:
        """Restore to construction state. Idempotent."""
        self._history.clear()
        self._tier = _LIGHT
        self._cooldown_left = 0

    # ------------------------------------------------------------------

    def _pick_signal(
        self, signals: dict[str, SignalReport]
    ) -> tuple[float, bool]:
        if self.signal_name in signals:
            r = signals[self.signal_name]
            return r.value, r.reliable
        for r in signals.values():
            if r.reliable:
                return r.value, True
        if signals:
            first = next(iter(signals.values()))
            return first.value, False
        return 0.0, False


__all__ = ["CUSUMScheduler"]
