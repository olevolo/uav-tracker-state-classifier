"""Timing context manager.

Deliberately implemented (not stubbed) — the API is trivial and we want
telemetry timing to work from day one so Phase 0 smoke tests can verify
the scaffold.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Type


class Timer:
    """Measure elapsed wall-clock time inside a ``with`` block.

    Example
    -------
    >>> with Timer() as t:
    ...     pass
    >>> t.elapsed_ms >= 0
    True
    """

    def __init__(self) -> None:
        self._t0: float = 0.0
        self._t1: float = 0.0
        self.elapsed_s: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._t1 = time.perf_counter()
        self.elapsed_s = self._t1 - self._t0
        self.elapsed_ms = self.elapsed_s * 1000.0
