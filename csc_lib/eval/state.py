"""Cat 6 — Per-state rate and persistence metrics.

state_rates, mean_time_in_state, state_persistence, lost_rate.
Used for paper tables M3-M4.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "lost_rate",
    "mean_time_in_state",
    "state_persistence",
    "state_rates",
]


def state_rates(states: list[str] | np.ndarray) -> dict[str, float]:
    """Fraction of frames per state name.

    Returns a dict mapping each unique state to count/total.
    """
    arr = list(states)
    n = len(arr)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for s in arr:
        counts[s] = counts.get(s, 0) + 1
    return {k: v / n for k, v in counts.items()}


def mean_time_in_state(
    states: list[str] | np.ndarray, state_name: str
) -> float:
    """Average length of contiguous runs of ``state_name``.

    Interpretation: mean segment count (not P(state)*T).
    Returns 0.0 if the state never appears.
    """
    arr = list(states)
    lengths: list[int] = []
    run = 0
    for s in arr:
        if s == state_name:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
                run = 0
    if run > 0:
        lengths.append(run)
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def state_persistence(states: list[str] | np.ndarray) -> dict[str, float]:
    """P(next_state == current | current == state) for each state.

    Measures how stable each state is frame-to-frame.
    """
    arr = list(states)
    n = len(arr)
    if n < 2:
        return {}
    total: dict[str, int] = {}
    stayed: dict[str, int] = {}
    for t in range(n - 1):
        s = arr[t]
        total[s] = total.get(s, 0) + 1
        if arr[t + 1] == s:
            stayed[s] = stayed.get(s, 0) + 1
    return {
        s: stayed.get(s, 0) / total[s]
        for s in total
    }


def lost_rate(
    states: list[str] | np.ndarray, lost_state: str = "lost"
) -> float:
    """Convenience wrapper: fraction of frames in ``lost_state``."""
    rates = state_rates(states)
    return rates.get(lost_state, 0.0)
