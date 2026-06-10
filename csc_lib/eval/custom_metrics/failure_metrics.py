"""Failure-event detection, false-confirmation analysis, and recovery metrics.

Standard failure: IoU < threshold (default 0.2 → "LOST" band per CSC.md).
False-confirmed failure: same IoU criterion + tracker confidence is high.
Severe failure: IoU < severe_threshold (default 0.1).

The false-confirmation block follows the paper's notation
(``FCFR``, ``FC Failure Share``, ``False Lock Duration``,
``False Confirmation Delay``, ``Recovery@K_FC``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FailureEpisode:
    start_frame: int
    end_frame: int
    length: int
    min_iou: float
    severe: bool
    is_false_confirmed: bool = False        # any frame in episode with high conf?
    fc_fraction: float = 0.0                # share of frames with high conf
    mean_conf_in_episode: float = 0.0


# ---------------------------------------------------------------------------
# Episode detection
# ---------------------------------------------------------------------------


def detect_failure_episodes(
    ious: np.ndarray,
    *,
    threshold: float = 0.2,
    severe_threshold: float = 0.1,
    min_length: int = 1,
    confidence: Optional[np.ndarray] = None,
    confidence_high_threshold: Optional[float] = None,
) -> list[FailureEpisode]:
    """Return contiguous failure intervals.

    If ``confidence`` and ``confidence_high_threshold`` are provided,
    each episode is also tagged with its false-confirmed status:

        FALSE_CONFIRMED episode  ←→  any frame had ``confidence >=
        confidence_high_threshold`` while ``iou < threshold``.

    The threshold is intentionally caller-supplied (callers should
    use a per-tracker percentile calibration, not a global magic
    number — see CSC.md).
    """
    ious = np.asarray(ious, dtype=np.float64)
    in_fail = np.where(np.isfinite(ious) & (ious < threshold), 1, 0).astype(np.int8)
    if in_fail.sum() == 0:
        return []

    if confidence is not None:
        conf_arr = np.asarray(confidence, dtype=np.float64)
        if conf_arr.shape != ious.shape:
            raise ValueError(
                f"confidence shape {conf_arr.shape} != ious shape {ious.shape}"
            )
    else:
        conf_arr = None

    episodes: list[FailureEpisode] = []
    i = 0
    n = len(in_fail)
    while i < n:
        if in_fail[i] == 0:
            i += 1
            continue
        start = i
        while i < n and in_fail[i] == 1:
            i += 1
        end = i - 1
        length = end - start + 1
        if length < min_length:
            continue
        seg = ious[start : end + 1]
        seg_min = float(np.nanmin(seg)) if np.isfinite(seg).any() else 0.0

        is_fc = False
        fc_frac = 0.0
        mean_conf = 0.0
        if conf_arr is not None and confidence_high_threshold is not None:
            seg_conf = conf_arr[start : end + 1]
            valid = np.isfinite(seg_conf)
            if valid.any():
                mean_conf = float(seg_conf[valid].mean())
                hi_mask = seg_conf >= float(confidence_high_threshold)
                hi_count = int(hi_mask.sum())
                fc_frac = hi_count / max(1, length)
                is_fc = hi_count > 0  # any high-confidence frame inside

        episodes.append(
            FailureEpisode(
                start_frame=int(start),
                end_frame=int(end),
                length=int(length),
                min_iou=seg_min,
                severe=bool(seg_min < severe_threshold),
                is_false_confirmed=is_fc,
                fc_fraction=float(fc_frac),
                mean_conf_in_episode=float(mean_conf),
            )
        )
    return episodes


# ---------------------------------------------------------------------------
# Aggregate per-sequence summary (legacy + extensions)
# ---------------------------------------------------------------------------


def failure_summary(
    ious: np.ndarray,
    *,
    threshold: float = 0.2,
    severe_threshold: float = 0.1,
    confidence: Optional[np.ndarray] = None,
    confidence_high_threshold: Optional[float] = None,
) -> dict:
    """Aggregate failure statistics for one sequence.

    With ``confidence`` and ``confidence_high_threshold`` provided,
    additionally returns:

    - ``fcfr`` — False-Confirmed Frame Rate
    - ``fc_failure_share`` — fraction of failure frames that are FC
    - ``n_fc_episodes`` — count of FC episodes
    - ``false_lock_duration_*`` — mean / median / max / p90 length
    """
    episodes = detect_failure_episodes(
        ious,
        threshold=threshold,
        severe_threshold=severe_threshold,
        confidence=confidence,
        confidence_high_threshold=confidence_high_threshold,
    )
    n = len(np.asarray(ious))
    total_failure_frames = int(sum(e.length for e in episodes))
    severe = [e for e in episodes if e.severe]

    if episodes:
        time_to_first_failure = episodes[0].start_frame
        mean_length = float(np.mean([e.length for e in episodes]))
        max_length = int(max(e.length for e in episodes))
    else:
        time_to_first_failure = -1
        mean_length = 0.0
        max_length = 0

    if len(episodes) >= 2:
        gaps = [
            episodes[i + 1].start_frame - episodes[i].end_frame - 1
            for i in range(len(episodes) - 1)
        ]
        mean_recovery_gap = float(np.mean(gaps))
    else:
        mean_recovery_gap = 0.0

    out = {
        "n_frames": n,
        "n_failures": len(episodes),
        "n_severe_failures": len(severe),
        "total_failure_frames": total_failure_frames,
        "failure_frame_rate": (total_failure_frames / n) if n > 0 else 0.0,
        "time_to_first_failure": time_to_first_failure,
        "mean_failure_length": mean_length,
        "max_failure_length": max_length,
        "mean_recovery_gap": mean_recovery_gap,
        "episodes": [e.__dict__ for e in episodes],
    }

    # ---- false-confirmation block (only if confidence provided) ----
    if confidence is not None and confidence_high_threshold is not None:
        conf = np.asarray(confidence, dtype=np.float64)
        ious_arr = np.asarray(ious, dtype=np.float64)
        valid = np.isfinite(ious_arr) & np.isfinite(conf)

        # FCFR — any frame with low IoU + high confidence
        fc_frame_mask = valid & (ious_arr < threshold) & (conf >= confidence_high_threshold)
        n_fc_frames = int(fc_frame_mask.sum())
        out["fcfr"] = (n_fc_frames / n) if n > 0 else 0.0
        out["fc_frame_count"] = n_fc_frames

        # FC failure share of all failure frames
        out["fc_failure_share"] = (
            n_fc_frames / total_failure_frames if total_failure_frames > 0 else 0.0
        )

        fc_episodes = [e for e in episodes if e.is_false_confirmed]
        out["n_fc_episodes"] = len(fc_episodes)
        if fc_episodes:
            lengths = np.asarray([e.length for e in fc_episodes], dtype=np.float64)
            out["false_lock_duration_mean"] = float(lengths.mean())
            out["false_lock_duration_median"] = float(np.median(lengths))
            out["false_lock_duration_max"] = int(lengths.max())
            out["false_lock_duration_p90"] = float(np.percentile(lengths, 90))
        else:
            out["false_lock_duration_mean"] = 0.0
            out["false_lock_duration_median"] = 0.0
            out["false_lock_duration_max"] = 0
            out["false_lock_duration_p90"] = 0.0

    return out


# ---------------------------------------------------------------------------
# Recovery metrics
# ---------------------------------------------------------------------------


def recovery_at_k(
    ious: np.ndarray,
    *,
    threshold: float = 0.2,
    k: int = 30,  # paper standard (CLAUDE.md §Metrics)
    recovery_threshold: float = 0.5,
    only_false_confirmed: bool = False,
    confidence: Optional[np.ndarray] = None,
    confidence_high_threshold: Optional[float] = None,
) -> float:
    """Recovery@K: fraction of failure episodes where the tracker
    returns to ``IoU >= recovery_threshold`` within K frames after the
    episode ends.

    With ``only_false_confirmed=True``, restricts attention to
    false-confirmed episodes (Recovery@K_FC).
    Default K=30 per paper specification.
    """
    episodes = detect_failure_episodes(
        ious,
        threshold=threshold,
        confidence=confidence,
        confidence_high_threshold=confidence_high_threshold,
    )
    if only_false_confirmed:
        episodes = [e for e in episodes if e.is_false_confirmed]
    if not episodes:
        return 0.0

    ious_arr = np.asarray(ious, dtype=np.float64)
    n = len(ious_arr)
    recovered = 0
    for ep in episodes:
        end = ep.end_frame
        lo = end + 1
        hi = min(n, end + 1 + k)
        if lo >= hi:
            continue
        seg = ious_arr[lo:hi]
        if np.any(np.isfinite(seg) & (seg >= recovery_threshold)):
            recovered += 1
    return recovered / len(episodes)


def unrecovered_episode_rate(
    ious: np.ndarray,
    *,
    threshold: float = 0.2,
    k: int = 30,
    recovery_threshold: float = 0.5,
) -> float:
    return 1.0 - recovery_at_k(
        ious, threshold=threshold, k=k, recovery_threshold=recovery_threshold
    )


# ---------------------------------------------------------------------------
# False-confirmation detection delay (FCD)
# ---------------------------------------------------------------------------


def false_confirmation_detection_delay(
    ious: np.ndarray,
    risk_score: np.ndarray,
    *,
    risk_threshold: float = 0.5,
    iou_threshold: float = 0.2,
    confidence: Optional[np.ndarray] = None,
    confidence_high_threshold: Optional[float] = None,
) -> dict:
    """For each false-confirmed episode, measure the gap between
    episode start and the first frame where ``risk_score >= risk_threshold``.

    This is DETECTION DELAY (TTFC-like): how many frames after the FC episode
    starts does the risk score first exceed the threshold.
    NOT Duration. For Duration (mean contiguous FC segment length) see
    ``compute_fcd`` in tools/compute_paper_metrics.py.

    Negative delay = early warning, positive delay = late detection,
    None = never detected.

    Returns aggregate stats: count, mean, median, fraction-detected,
    fraction-early-warned (negative delay).
    """
    episodes = detect_failure_episodes(
        ious,
        threshold=iou_threshold,
        confidence=confidence,
        confidence_high_threshold=confidence_high_threshold,
    )
    fc_eps = [e for e in episodes if e.is_false_confirmed]
    if not fc_eps:
        return {
            "n_fc_episodes": 0,
            "n_detected": 0,
            "n_early_warned": 0,
            "ttfc_mean": None,
            "ttfc_median": None,
            "ttfc_min": None,
            "detection_rate": 0.0,
            "early_warning_rate": 0.0,
            "delays": [],
        }

    risk = np.asarray(risk_score, dtype=np.float64)
    above = risk >= risk_threshold

    delays: list[Optional[int]] = []
    for ep in fc_eps:
        start = ep.start_frame
        # First crossing strictly before start (max 60 frames lookback).
        warn_t: Optional[int] = None
        for tt in range(start - 1, max(-1, start - 61), -1):
            if tt < 0:
                break
            if above[tt]:
                warn_t = tt
                break
        if warn_t is not None:
            delays.append(int(warn_t - start))  # negative
            continue
        # Else first crossing inside or after the episode.
        seg = above[start:]
        hits = np.where(seg)[0]
        if hits.size:
            delays.append(int(hits[0]))
        else:
            delays.append(None)

    detected = [d for d in delays if d is not None]
    early = [d for d in detected if d < 0]
    return {
        "n_fc_episodes": len(fc_eps),
        "n_detected": len(detected),
        "n_early_warned": len(early),
        "ttfc_mean": float(np.mean(detected)) if detected else None,
        "ttfc_median": float(np.median(detected)) if detected else None,
        "ttfc_min": int(min(detected)) if detected else None,
        "detection_rate": len(detected) / len(fc_eps),
        "early_warning_rate": len(early) / len(fc_eps),
        "delays": delays,
    }


import warnings as _warnings


def false_confirmation_delay(
    ious: np.ndarray,
    risk_score: np.ndarray,
    *,
    risk_threshold: float = 0.5,
    iou_threshold: float = 0.2,
    confidence: Optional[np.ndarray] = None,
    confidence_high_threshold: Optional[float] = None,
) -> dict:
    """Deprecated alias for ``false_confirmation_detection_delay``.

    Use ``false_confirmation_detection_delay`` instead.
    Note: returned keys are now ``ttfc_mean`` / ``ttfc_median`` (not fcd_*).
    """
    _warnings.warn(
        "false_confirmation_delay is deprecated; use false_confirmation_detection_delay. "
        "Returned keys have changed from fcd_mean/fcd_median to ttfc_mean/ttfc_median.",
        DeprecationWarning,
        stacklevel=2,
    )
    return false_confirmation_detection_delay(
        ious,
        risk_score,
        risk_threshold=risk_threshold,
        iou_threshold=iou_threshold,
        confidence=confidence,
        confidence_high_threshold=confidence_high_threshold,
    )


# ---------------------------------------------------------------------------
# Hard / post-failure AUC (kept from the original module)
# ---------------------------------------------------------------------------


def hard_frame_auc(
    ious: np.ndarray,
    *,
    threshold: float = 0.2,
    n_thresholds: int = 21,
) -> float:
    from csc_lib.eval.custom_metrics.tracking_metrics import success_auc

    ious = np.asarray(ious, dtype=np.float64)
    valid = np.isfinite(ious) & (ious >= 0)
    if not valid.any():
        return 0.0
    mask = valid & (ious < threshold)
    if not mask.any():
        return 0.0
    return success_auc(ious[mask], n_thresholds=n_thresholds)


def post_first_failure_auc(
    ious: np.ndarray,
    *,
    threshold: float = 0.2,
) -> float:
    from csc_lib.eval.custom_metrics.tracking_metrics import success_auc

    ious = np.asarray(ious, dtype=np.float64)
    episodes = detect_failure_episodes(ious, threshold=threshold)
    if not episodes:
        return success_auc(ious)
    start = episodes[0].start_frame
    return success_auc(ious[start:])


# ---------------------------------------------------------------------------
# Episode-level detection statistics
# ---------------------------------------------------------------------------


def episode_detection_stats(
    y_true_binary: np.ndarray,   # 1 = failure frame (LOST or FC), 0 = ok
    risk_score: np.ndarray,      # CSC risk score (P(LOST))
    tau_risk: float = 0.5,       # threshold to call a "detection"
    k_values: tuple = (5, 10, 20),
    min_episode_len: int = 3,
) -> dict:
    """Compute episode-level detection statistics.

    Parameters
    ----------
    y_true_binary : ndarray of shape (T,)
        Binary label: 1 = failure frame (LOST or FALSE_CONFIRMED), 0 = ok.
    risk_score : ndarray of shape (T,)
        CSC risk score — e.g. P(LOST).
    tau_risk : float
        Detection threshold: a frame is "detected" if risk_score >= tau_risk.
    k_values : tuple of int
        Episode recall is evaluated at these lead-times (frames).
    min_episode_len : int
        Minimum contiguous failure run to count as an episode (filters noise).

    Returns
    -------
    dict with keys:
        episode_recall_at_k : dict[int, float]
            Fraction of failure episodes detected within k frames of episode start.
        mean_detection_delay_frames : float
            Mean frames between episode_start and first detection inside that episode.
            NaN if no episodes were ever detected.
        false_alarm_episodes_per_1000 : float
            Number of false-alarm episodes (contiguous detected runs outside any true
            episode) per 1000 non-failure frames.
    """
    y = np.asarray(y_true_binary, dtype=np.int8)
    r = np.asarray(risk_score, dtype=np.float64)
    T = len(y)

    if T == 0:
        return {
            "episode_recall_at_k": {k: 0.0 for k in k_values},
            "mean_detection_delay_frames": float("nan"),
            "false_alarm_episodes_per_1000": 0.0,
        }

    # ---- Step 1: Find contiguous failure episodes (run-length encoding) ----
    episodes: list[tuple[int, int]] = []   # (t_start, t_end) inclusive
    i = 0
    while i < T:
        if y[i] == 1:
            start = i
            while i < T and y[i] == 1:
                i += 1
            end = i - 1
            if (end - start + 1) >= min_episode_len:
                episodes.append((start, end))
        else:
            i += 1

    # Detected mask
    detected = r >= tau_risk  # (T,) bool

    # ---- Step 2: Episode recall@k and detection delays ----
    detection_delays: list[int] = []   # only for detected episodes

    recall_at_k: dict[int, int] = {k: 0 for k in k_values}

    for t_start, t_end in episodes:
        # Check within [t_start, t_start + k - 1] for each k
        for k in k_values:
            look_end = min(T, t_start + k)
            if detected[t_start:look_end].any():
                recall_at_k[k] += 1

        # Detection delay: first detected frame at or after t_start
        after_start = np.where(detected[t_start:])[0]
        if after_start.size > 0:
            delay = int(after_start[0])   # frames from t_start
            detection_delays.append(delay)

    n_episodes = len(episodes)
    episode_recall_at_k = {
        k: (recall_at_k[k] / n_episodes if n_episodes > 0 else 0.0)
        for k in k_values
    }
    mean_delay = (
        float(np.mean(detection_delays)) if detection_delays else float("nan")
    )

    # ---- Step 3: False alarm episodes ----
    # A false-alarm episode is a contiguous run of detected[t]=True that does
    # not overlap any true failure episode.

    # Build a failure mask for overlap checking
    failure_mask = y.astype(bool)

    fa_count = 0
    j = 0
    while j < T:
        if detected[j]:
            fa_start = j
            while j < T and detected[j]:
                j += 1
            fa_end = j - 1
            # Check if this run overlaps any true failure frame
            if not failure_mask[fa_start: fa_end + 1].any():
                fa_count += 1
        else:
            j += 1

    n_non_failure = int((y == 0).sum())
    false_alarm_rate = (
        float(fa_count) / (n_non_failure / 1000.0)
        if n_non_failure > 0
        else 0.0
    )

    return {
        "episode_recall_at_k": episode_recall_at_k,
        "mean_detection_delay_frames": mean_delay,
        "false_alarm_episodes_per_1000": round(false_alarm_rate, 4),
    }


def compute_all_episode_metrics(
    labels_df,
    risk_col: str = "risk_score",
    state_col: str = "derived_state_name",
) -> dict:
    """Convenience wrapper: compute episode-level metrics from a states/*.jsonl DataFrame.

    The ``labels_df`` can be either a pandas DataFrame or a list of dicts
    (one per frame) in the format written by ``run_with_csc.py``.

    Parameters
    ----------
    labels_df : list[dict] or pandas.DataFrame
        Per-frame rows from a states/*.jsonl file.  Must contain columns/keys
        ``risk_score`` (float) and ``derived_state_name`` (str).
    risk_col : str
        Column name for the CSC risk score.
    state_col : str
        Column name for the derived state name string.

    Returns
    -------
    dict with episode_detection_stats output merged with additional summary:
        n_frames, n_failure_frames, n_episodes, failure_frame_rate
    """
    # Accept both list-of-dicts and DataFrame
    try:
        import pandas as pd
        if isinstance(labels_df, pd.DataFrame):
            rows = labels_df.to_dict(orient="records")
        else:
            rows = list(labels_df)
    except ImportError:
        rows = list(labels_df)

    if not rows:
        return {
            "n_frames": 0,
            "n_failure_frames": 0,
            "n_episodes": 0,
            "failure_frame_rate": 0.0,
            "episode_recall_at_k": {5: 0.0, 10: 0.0, 20: 0.0},
            "mean_detection_delay_frames": float("nan"),
            "false_alarm_episodes_per_1000": 0.0,
        }

    # Build binary failure label: LOST_AWARE (2) or FALSE_CONFIRMED (3)
    _FAILURE_STATES = {"LOST_AWARE", "FALSE_CONFIRMED", "LOST"}

    y_true: list[int] = []
    risk: list[float] = []

    for row in rows:
        state_name = row.get(state_col, "")
        y_true.append(1 if state_name in _FAILURE_STATES else 0)
        # Handle missing / NaN risk score
        rv = row.get(risk_col)
        try:
            rv_f = float(rv) if rv is not None else 0.0
        except (TypeError, ValueError):
            rv_f = 0.0
        risk.append(rv_f)

    y_arr = np.array(y_true, dtype=np.int8)
    r_arr = np.array(risk, dtype=np.float64)

    stats = episode_detection_stats(y_arr, r_arr)

    n_failure = int(y_arr.sum())
    # Count episodes for summary
    from itertools import groupby
    n_episodes = sum(
        1
        for k, g in groupby(y_arr)
        if k == 1 and sum(1 for _ in g) >= 3
    )

    return {
        "n_frames": len(y_arr),
        "n_failure_frames": n_failure,
        "n_episodes": n_episodes,
        "failure_frame_rate": round(n_failure / max(1, len(y_arr)), 6),
        **stats,
    }
