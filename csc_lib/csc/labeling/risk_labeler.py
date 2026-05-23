"""Future-risk label generation for CSC-V3 proactive forecasting.

Given a sequence of per-frame derived states, produce per-frame binary labels
indicating whether a failure / FALSE_CONFIRMED / LOST_AWARE event will occur
in the next ``horizon`` frames.

No-leakage rule: the labels at frame ``t`` are computed from
``derived_states[t+1 : t+1+horizon]`` only.  The current frame ``t`` is NOT
included.  Input features for the model must use frames ``<= t`` only — that
constraint lives in :mod:`csc_lib.csc.dataset` and is unaffected by this
module.

Edge handling: when fewer than ``horizon`` future frames are available
(near the end of a sequence), labels are computed on the partial future
window AND ``ignore_forecast`` is set to 1 so the training loss can mask
those frames out.  The very last frame has no future at all and is
always ignored.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from csc_lib.csc.labeling.label_schema import DerivedState


_FAILURE_STATES: frozenset[int] = frozenset(
    {int(DerivedState.LOST_AWARE), int(DerivedState.FALSE_CONFIRMED)}
)
_FALSE_CONFIRMED: int = int(DerivedState.FALSE_CONFIRMED)
_LOST_AWARE: int = int(DerivedState.LOST_AWARE)


def build_future_risk_labels(
    derived_states: Sequence[int] | Iterable[int],
    horizon: int = 10,
) -> list[dict[str, int]]:
    """Compute per-frame future-risk labels.

    Parameters
    ----------
    derived_states:
        Per-frame derived state IDs (values from
        :class:`~csc_lib.csc.labeling.label_schema.DerivedState`).
    horizon:
        Number of future frames to look ahead.  Defaults to 10.

    Returns
    -------
    list of dicts, one per frame, each with keys:
        - ``failure_next_10``         (int 0/1)
        - ``false_confirmed_next_10`` (int 0/1)
        - ``lost_aware_next_10``      (int 0/1)
        - ``ignore_forecast``         (int 0/1) — 1 when the full horizon was
          NOT available; training pipelines should mask those frames.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    states = list(derived_states)
    n = len(states)
    out: list[dict[str, int]] = []

    for t in range(n):
        end = t + 1 + horizon
        full_horizon_available = end <= n
        future = states[t + 1 : min(end, n)]

        if not future:
            out.append(
                {
                    "failure_next_10": 0,
                    "false_confirmed_next_10": 0,
                    "lost_aware_next_10": 0,
                    "ignore_forecast": 1,
                }
            )
            continue

        has_fc = any(s == _FALSE_CONFIRMED for s in future)
        has_lost = any(s == _LOST_AWARE for s in future)
        has_failure = has_fc or has_lost

        out.append(
            {
                "failure_next_10": int(has_failure),
                "false_confirmed_next_10": int(has_fc),
                "lost_aware_next_10": int(has_lost),
                "ignore_forecast": 0 if full_horizon_available else 1,
            }
        )

    return out


def summarize_future_risk(labels: list[dict[str, int]]) -> dict[str, float]:
    """Distribution stats over a list of forecast-label dicts."""
    n = len(labels)
    if n == 0:
        return {
            "n": 0,
            "n_valid": 0,
            "failure_rate": 0.0,
            "fc_rate": 0.0,
            "lost_rate": 0.0,
            "ignore_rate": 0.0,
        }
    n_ignore = sum(int(r["ignore_forecast"]) for r in labels)
    n_valid = max(1, n - n_ignore)
    valid = [r for r in labels if not r["ignore_forecast"]]
    return {
        "n": n,
        "n_valid": n - n_ignore,
        "failure_rate": sum(r["failure_next_10"] for r in valid) / n_valid,
        "fc_rate": sum(r["false_confirmed_next_10"] for r in valid) / n_valid,
        "lost_rate": sum(r["lost_aware_next_10"] for r in valid) / n_valid,
        "ignore_rate": n_ignore / n,
    }


__all__ = ["build_future_risk_labels", "summarize_future_risk"]
