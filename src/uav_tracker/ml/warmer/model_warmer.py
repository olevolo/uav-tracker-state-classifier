"""DefaultModelWarmer — pre-loads and JIT-warms all tracker models before tracking starts.

Registration key: ``"default"``
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from uav_tracker.registry import ML_WARMERS
from uav_tracker.types import BBox

logger = logging.getLogger(__name__)


@ML_WARMERS.register("default")
class DefaultModelWarmer:
    """Pre-loads and JIT-warms all tracker models before the tracking loop starts.

    For each tracker tier, runs ``n_warmup_passes`` of ``init`` + ``update``
    on a synthetic dummy frame to force any lazy model-loading (OpenCV model
    file reads, PyTorch weight loads, CUDA kernel JIT compilation, etc.) to
    happen *before* frame 0.

    Parameters
    ----------
    lazy:
        Reserved for future incremental loading; currently unused.
    n_warmup_passes:
        Number of ``update`` calls to execute per tracker during warmup.
        More passes amortise JIT variance; 3 is usually enough for KCF/SiamFC.
    target_latency_ms:
        Per-tracker warmup budget in milliseconds.  Exceeding it generates a
        WARNING but does not abort.
    dummy_frame_shape:
        ``(H, W, C)`` tuple for the synthetic random frame used during warmup.
        Should match the expected runtime frame size to expose shape-dependent
        kernel compilation.
    """

    name: str = "default"

    def __init__(
        self,
        lazy: bool = True,
        n_warmup_passes: int = 3,
        target_latency_ms: float = 50.0,
        dummy_frame_shape: tuple = (360, 640, 3),
    ) -> None:
        self._lazy = lazy
        self._n_warmup_passes = n_warmup_passes
        self._target_latency_ms = target_latency_ms
        self._dummy_frame_shape = tuple(dummy_frame_shape)
        self._status: dict[str, Any] = {}
        self._warmed: bool = False

    # ------------------------------------------------------------------
    # Public API

    def warmup(self, trackers: dict[int, Any]) -> None:
        """Pre-load and warm-start all tracker models.

        For each tracker in ``trackers``:

        1. Calls ``tracker.init(dummy_frame, dummy_bbox)``.
        2. Runs ``n_warmup_passes`` of ``tracker.update(dummy_frame)``.
        3. Records total warmup latency.
        4. Warns if latency > ``target_latency_ms``.

        Tracker-typed plugins are detected by duck-typing (presence of
        ``update``).  Detector-only tiers (no ``update``) are skipped
        silently.

        Parameters
        ----------
        trackers:
            Mapping from tier index to tracker/plugin instance.  This is
            ``HybridRunner._tier_plugins`` in normal usage.
        """
        dummy_frame = np.random.randint(
            0, 255, self._dummy_frame_shape, dtype=np.uint8
        )
        dummy_bbox = BBox(x=100, y=100, w=50, h=50)

        for tier, tracker in trackers.items():
            # Skip detector-only tiers (no update method).
            if not callable(getattr(tracker, "update", None)):
                logger.debug(
                    "ModelWarmer: tier %d has no .update(); skipping warmup", tier
                )
                continue

            tracker_name: str = getattr(tracker, "name", str(tier))
            t_start = time.perf_counter()
            try:
                if callable(getattr(tracker, "init", None)):
                    tracker.init(dummy_frame, dummy_bbox)
                for _ in range(self._n_warmup_passes):
                    tracker.update(dummy_frame)
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                self._status[tracker_name] = {
                    "latency_ms": latency_ms,
                    "status": "ok",
                    "tier": tier,
                }
                if latency_ms > self._target_latency_ms:
                    logger.warning(
                        "ModelWarmer: tier %d (%s) warmup took %.1f ms > target %.1f ms",
                        tier,
                        tracker_name,
                        latency_ms,
                        self._target_latency_ms,
                    )
                else:
                    logger.debug(
                        "ModelWarmer: tier %d (%s) warmed in %.1f ms",
                        tier,
                        tracker_name,
                        latency_ms,
                    )
            except Exception as exc:
                logger.error(
                    "ModelWarmer: tier %d (%s) warmup failed: %s",
                    tier,
                    tracker_name,
                    exc,
                )
                self._status[tracker_name] = {
                    "status": "failed",
                    "error": str(exc),
                    "tier": tier,
                }

        self._warmed = True

    def warmup_single(self, tracker: Any, dummy_frame: np.ndarray) -> float:
        """Warm a single tracker and return the warmup latency in milliseconds.

        Parameters
        ----------
        tracker:
            A tracker instance with ``init`` and ``update`` methods.
        dummy_frame:
            Pre-constructed dummy frame to use for this pass.

        Returns
        -------
        float
            Elapsed time in milliseconds for ``n_warmup_passes`` update calls.
        """
        dummy_bbox = BBox(x=100, y=100, w=50, h=50)
        t_start = time.perf_counter()
        if callable(getattr(tracker, "init", None)):
            tracker.init(dummy_frame, dummy_bbox)
        for _ in range(self._n_warmup_passes):
            tracker.update(dummy_frame)
        return (time.perf_counter() - t_start) * 1000.0

    def get_status(self) -> dict[str, Any]:
        """Return a copy of the per-tracker warmup status dict.

        Keys are tracker names (or tier index strings as fallback).  Values
        are dicts with at minimum ``"status"`` (``"ok"`` or ``"failed"``) and
        ``"tier"``.
        """
        return dict(self._status)

    @property
    def is_warmed(self) -> bool:
        """``True`` after ``warmup()`` has been called at least once."""
        return self._warmed


__all__ = ["DefaultModelWarmer"]
