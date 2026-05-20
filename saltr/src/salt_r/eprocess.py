"""Phase 2A — Sequential e-process alerts over SALT-RD v2 calibrated predictions.

Two modes:
  formal     — multiplicative martingale, no decay, anytime-valid under H0.
  engineering — same martingale with per-frame floor and decay for deployment usability.

Usage:
    python -m salt_r.eprocess \
        --preds   saltr/results/preds_val_v2.json \
        --labels  saltr/data/salt_rd_v2_labels.npz \
        --out-json saltr/results/eprocess_val_v2.json \
        --out-sweep saltr/results/eprocess_sweep_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIAGNOSTIC_SEQS = {
    "visdrone_sot/uav0000164",
    "uav123/bike2",
    "dtb70/Gull2",
    "dtb70/Sheep1",
    "dtb70/StreetBasketball1",
}

# Calibration uses ~40 % of val sequences (deterministic by lexicographic split)
_CAL_FRACTION = 0.40

_DEFAULT_EPSILONS = (0.25, 0.50, 0.75)
_DEFAULT_ALPHAS   = (0.20, 0.10, 0.05)
_DEFAULT_DECAYS   = (0.95, 0.98, 1.00)


# ---------------------------------------------------------------------------
# Risk score
# ---------------------------------------------------------------------------

def compute_risk_score(frame_probs: dict[str, float], mode: str = "all_risk") -> float:
    """Composite risk score from calibrated head probabilities.

    Parameters
    ----------
    frame_probs:
        Per-head probability dict from the SALT-RD model.
    mode:
        One of "all_risk" (default), "ifd5", "ifd10", "ifd20", "fc_ifd20".
        - "all_risk"  — weighted combination of all risk heads (legacy default).
        - "ifd5"      — P(imminent_failure_dynamic) only.
        - "ifd10"     — P(imminent_failure_dynamic_10) only.
        - "ifd20"     — P(imminent_failure_dynamic_20) only.
        - "fc_ifd20"  — 0.60 * P(false_confirmed) + 0.40 * P(ifd20).
    """
    if mode == "ifd5":
        return frame_probs.get("imminent_failure_dynamic", 0.0)
    elif mode == "ifd10":
        return frame_probs.get("imminent_failure_dynamic_10", 0.0)
    elif mode == "ifd20":
        return frame_probs.get("imminent_failure_dynamic_20", 0.0)
    elif mode == "fc_ifd20":
        return (0.60 * frame_probs.get("false_confirmed", 0.0)
              + 0.40 * frame_probs.get("imminent_failure_dynamic_20", 0.0))
    else:  # "all_risk" (default)
        p_fc    = frame_probs.get("false_confirmed", 0.0)
        p_ifd   = frame_probs.get("imminent_failure_dynamic", 0.0)
        p_fi5   = frame_probs.get("failure_in_5", 0.0)
        p_ifd20 = frame_probs.get("imminent_failure_dynamic_20", 0.0)
        return 0.45 * p_fc + 0.35 * p_ifd + 0.15 * p_fi5 + 0.05 * p_ifd20


# ---------------------------------------------------------------------------
# Conformal null calibration
# ---------------------------------------------------------------------------

def build_null_distribution(
    preds: dict[str, list[dict[str, float]]],
    labels: dict[str, np.ndarray],
    iou_traces: dict[str, np.ndarray],
    label_names: list[str],
    cal_seq_keys: list[str],
    risk_mode: str = "all_risk",
) -> np.ndarray:
    """Return risk scores from null frames in calibration sequences.

    Null frame: IoU >= 0.5  AND  not false_confirmed  AND  not failure_in_5/10/20
                AND  not imminent_failure_dynamic (any horizon).
    """
    idx = {n: i for i, n in enumerate(label_names)}
    failure_cols = [
        idx.get("false_confirmed"),
        idx.get("failure_in_5"),
        idx.get("failure_in_10"),
        idx.get("failure_in_20"),
        idx.get("imminent_failure_dynamic"),
        idx.get("imminent_failure_dynamic_10"),
        idx.get("imminent_failure_dynamic_20"),
    ]
    failure_cols = [c for c in failure_cols if c is not None]

    null_scores: list[float] = []
    for seq in cal_seq_keys:
        if seq not in preds or seq not in labels or seq not in iou_traces:
            continue
        frames = preds[seq]
        lab = labels[seq].astype(int)
        iou = iou_traces[seq]
        n = min(len(frames), len(lab), len(iou))
        for t in range(n):
            if iou[t] < 0.5:
                continue
            if any(c < lab.shape[1] and lab[t, c] == 1 for c in failure_cols):
                continue
            null_scores.append(compute_risk_score(frames[t], mode=risk_mode))

    return np.array(null_scores, dtype=np.float32)


def conformal_pvalue(score: float, null_scores: np.ndarray) -> float:
    """Empirical p-value: fraction of null scores >= query score."""
    return float(1.0 + np.sum(null_scores >= score)) / float(1.0 + len(null_scores))


# ---------------------------------------------------------------------------
# E-value betting function
# ---------------------------------------------------------------------------

def power_evalue(p_t: float, epsilon: float) -> float:
    """Power betting function: e = epsilon * p^(epsilon-1).

    Defined for p in (0, 1]; clipped to [1e-8, 1] to avoid div-by-zero.
    """
    p_t = float(np.clip(p_t, 1e-8, 1.0))
    return epsilon * (p_t ** (epsilon - 1.0))


# ---------------------------------------------------------------------------
# Sequential accumulation
# ---------------------------------------------------------------------------

def run_eprocess_sequence(
    risk_scores: np.ndarray,
    null_scores: np.ndarray,
    alpha: float = 0.10,
    epsilon: float = 0.50,
    decay: float = 1.00,
    mode: str = "formal",
) -> tuple[np.ndarray, list[int]]:
    """Accumulate e-process over one sequence; return (E_trace, alert_frames).

    mode="formal"      — E_t = E_{t-1} * e_t, alert when max_so_far >= 1/alpha.
    mode="engineering" — E_t = max(1, decay * E_{t-1} * e_t).
    """
    threshold = 1.0 / alpha
    n = len(risk_scores)
    E = np.ones(n, dtype=np.float64)
    alerts: list[int] = []
    e_prev = 1.0
    e_max = 1.0
    alerted = False  # formal mode: once alerted, stays alerted (running-max)

    for t in range(n):
        p_t = conformal_pvalue(float(risk_scores[t]), null_scores)
        e_t = power_evalue(p_t, epsilon)

        if mode == "formal":
            e_curr = e_prev * e_t
            e_prev = e_curr
            e_max = max(e_max, e_curr)
            E[t] = e_curr
            if not alerted and e_max >= threshold:
                alerts.append(t)
                alerted = True
        else:  # engineering
            e_curr = max(1.0, decay * e_prev * e_t)
            e_prev = e_curr
            E[t] = e_curr
            if e_curr >= threshold:
                alerts.append(t)

    return E, alerts


# ---------------------------------------------------------------------------
# aGRAPAbetting — paper Eq. 12 (no calibration set required)
# ---------------------------------------------------------------------------

def compute_quality_score(frame_probs: dict[str, float], risk_mode: str = "all_risk") -> float:
    """Quality score = 1 - risk_score.  High value means low failure risk."""
    return 1.0 - compute_risk_score(frame_probs, mode=risk_mode)


def run_eprocess_agrapa(
    quality_scores: np.ndarray,
    epsilon: float = 0.50,
    alpha: float = 0.10,
    window: int = 20,
) -> tuple[np.ndarray, list[int]]:
    """aGRAPAbetting e-process (Eq. 12 from WACV2026 paper).

    Estimates the betting rate λ_t *online* from a rolling window of recent
    quality scores — no calibration set is needed.

    Accumulation rule::

        λ_t = clip((ε - μ_w) / (σ²_w + (ε - μ_w)²), 0, 1/(2ε))
        X_t = X_{t-1} * (1 + λ_t * (ε - M_t))

    where M_t = quality_scores[t] ∈ [0, 1] (high = good),
    and μ_w / σ²_w are the mean/variance of the last ``window`` scores.

    Formal mode: at most one alert (once running maximum ≥ 1/alpha).

    Parameters
    ----------
    quality_scores:
        Per-frame quality, ``1 - risk_score``.  High → tracker healthy.
    epsilon:
        Tolerance threshold.  Alerts fire when quality is consistently below ε.
    alpha:
        Significance level; alert threshold = 1/alpha.
    window:
        Recency window w_E for online λ estimation (paper default: 20).

    Returns
    -------
    (E_trace, alert_frames)
    """
    n = len(quality_scores)
    E = np.ones(n, dtype=np.float64)
    alerts: list[int] = []
    X_prev = 1.0
    X_max = 1.0
    threshold = 1.0 / alpha
    alerted = False

    lam_max = 1.0 / (2.0 * epsilon) if epsilon > 0 else 1.0

    for t in range(n):
        # Rolling window of recent quality scores
        start = max(0, t - window)
        win = quality_scores[start:t + 1]
        mu = float(np.mean(win))
        var = float(np.var(win)) + 1e-8

        # Adaptive betting rate (Eq. 12)
        num = epsilon - mu
        lam = num / (var + num ** 2)
        lam = float(np.clip(lam, 0.0, lam_max))

        M_t = float(quality_scores[t])
        X_curr = X_prev * (1.0 + lam * (epsilon - M_t))
        X_curr = max(X_curr, 0.0)  # martingale is non-negative
        X_prev = X_curr
        X_max = max(X_max, X_curr)
        E[t] = X_curr

        if not alerted and X_max >= threshold:
            alerts.append(t)
            alerted = True

    return E, alerts


def reset_at_boundary(
    reset_frames: list[int],
    risk_scores: np.ndarray,
    null_scores: np.ndarray,
    alpha: float,
    epsilon: float,
    decay: float,
    mode: str,
) -> tuple[np.ndarray, list[int]]:
    """Run e-process with resets at specified frames (re-init / episode boundaries)."""
    n = len(risk_scores)
    boundaries = sorted(set([0] + reset_frames + [n]))
    E_full = np.ones(n, dtype=np.float64)
    all_alerts: list[int] = []

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg = risk_scores[start:end]
        if len(seg) == 0:
            continue
        E_seg, seg_alerts = run_eprocess_sequence(
            seg, null_scores, alpha=alpha, epsilon=epsilon, decay=decay, mode=mode
        )
        E_full[start:end] = E_seg
        all_alerts.extend(a + start for a in seg_alerts)

    return E_full, all_alerts


# ---------------------------------------------------------------------------
# Alert → event metrics
# ---------------------------------------------------------------------------

def _failure_events(iou: np.ndarray, iou_threshold: float = 0.30) -> list[int]:
    """Return frames where IoU first drops below threshold after being >= threshold.

    Sequences starting below the threshold are not counted as failure events —
    the tracker must have had at least one good frame (IoU >= threshold) first.
    """
    events = []
    above = False  # require at least one good frame before counting failures
    for t, v in enumerate(iou):
        if above and v < iou_threshold:
            events.append(t)
            above = False
        elif v >= iou_threshold:
            above = True
    return events


def compute_alert_metrics(
    alerts: list[int],
    iou: np.ndarray,
    n_frames: int,
    iou_failure_threshold: float = 0.30,
    lead_horizon: int = 20,
) -> dict[str, Any]:
    """Compute lead-time and false-alert metrics for one sequence."""
    failure_events = _failure_events(iou, iou_failure_threshold)
    if not failure_events:
        return {
            "n_alerts": len(alerts),
            "n_failure_events": 0,
            "false_alerts": len(alerts),
            "tp_alerts": 0,
            "lead_times": [],
        }

    alert_set = set(alerts)
    # For each failure event, check if there's an alert in [event-lead_horizon, event)
    tp_events: list[int] = []
    lead_times: list[int] = []
    for ev in failure_events:
        window_start = max(0, ev - lead_horizon)
        prior_alerts = [a for a in alerts if window_start <= a < ev]
        if prior_alerts:
            tp_events.append(ev)
            lead_times.append(ev - min(prior_alerts))

    # False alerts: alerts NOT within lead_horizon before any failure event
    fail_neighborhoods: set[int] = set()
    for ev in failure_events:
        for a in range(max(0, ev - lead_horizon), ev):
            fail_neighborhoods.add(a)
    false_alerts = [a for a in alerts if a not in fail_neighborhoods]

    return {
        "n_alerts": len(alerts),
        "n_failure_events": len(failure_events),
        "tp_alerts": len(tp_events),
        "false_alerts": len(false_alerts),
        "lead_times": lead_times,
        "failure_event_recall": len(tp_events) / max(len(failure_events), 1),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    preds: dict[str, list[dict[str, float]]],
    labels: dict[str, np.ndarray],
    iou_traces: dict[str, np.ndarray],
    label_names: list[str],
    alpha: float = 0.10,
    epsilon: float = 0.50,
    decay: float = 1.00,
    mode: str = "formal",
    cal_fraction: float = _CAL_FRACTION,
    iou_failure_threshold: float = 0.30,
    risk_mode: str = "all_risk",
    diagnostic_seqs: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate e-process alerts on all non-diagnostic val sequences.

    Splits sequences lexicographically: first ``cal_fraction`` → calibration,
    rest → alert evaluation.  Diagnostic sequences are excluded from both.

    Parameters
    ----------
    risk_mode:
        Risk score aggregation mode passed to ``compute_risk_score()``.
        One of "all_risk", "ifd5", "ifd10", "ifd20", "fc_ifd20".
    mode:
        E-process accumulation mode.  "formal" | "engineering" | "agrapa".
        "agrapa" uses online aGRAPAbetting (no calibration set needed).
    diagnostic_seqs:
        Set of sequence keys to exclude from calibration and evaluation.
        If None, falls back to the module-level ``_DIAGNOSTIC_SEQS`` constant.
    """
    _diag = diagnostic_seqs if diagnostic_seqs is not None else _DIAGNOSTIC_SEQS
    all_seqs = sorted(
        s for s in preds if s not in _diag and s in labels and s in iou_traces
    )
    n_cal = max(1, int(len(all_seqs) * cal_fraction))
    cal_seqs = all_seqs[:n_cal]
    eval_seqs = all_seqs[n_cal:]

    # aGRAPAmode does not need a calibration distribution
    if mode == "agrapa":
        null_scores = np.array([], dtype=np.float32)
    else:
        null_scores = build_null_distribution(preds, labels, iou_traces, label_names, cal_seqs, risk_mode=risk_mode)
        if len(null_scores) == 0:
            return {"error": "null distribution is empty — check calibration sequences"}

    total_alerts = 0
    total_false_alerts = 0
    total_tp_alerts = 0
    total_failure_events = 0
    total_frames = 0
    all_lead_times: list[int] = []
    event_recalls: list[float] = []

    # Baselines: raw P(head) > 0.5 for four probe heads
    _baseline_heads = [
        ("ifd5",  "imminent_failure_dynamic"),
        ("ifd10", "imminent_failure_dynamic_10"),
        ("ifd20", "imminent_failure_dynamic_20"),
        ("fc",    "false_confirmed"),
    ]
    baseline_totals: dict[str, dict[str, int]] = {
        name: {"alerts": 0, "tp": 0, "false": 0}
        for name, _ in _baseline_heads
    }

    per_seq: dict[str, dict] = {}
    for seq in eval_seqs:
        frames = preds[seq]
        iou = iou_traces[seq]
        n = min(len(frames), len(iou))

        risk_scores = np.array(
            [compute_risk_score(frames[t], mode=risk_mode) for t in range(n)], dtype=np.float32
        )

        if mode == "agrapa":
            quality_scores = 1.0 - risk_scores
            E_trace, alerts = run_eprocess_agrapa(
                quality_scores, epsilon=epsilon, alpha=alpha
            )
        else:
            E_trace, alerts = run_eprocess_sequence(
                risk_scores, null_scores, alpha=alpha, epsilon=epsilon, decay=decay, mode=mode
            )

        m = compute_alert_metrics(alerts, iou[:n], n, iou_failure_threshold)
        total_alerts += m["n_alerts"]
        total_false_alerts += m["false_alerts"]
        total_tp_alerts += m["tp_alerts"]
        total_failure_events += m["n_failure_events"]
        total_frames += n
        all_lead_times.extend(m["lead_times"])
        if m["n_failure_events"] > 0:
            event_recalls.append(m["failure_event_recall"])

        # Baselines: P(head) > 0.5
        for bl_name, head_key in _baseline_heads:
            bl_alerts = [t for t in range(n) if frames[t].get(head_key, 0.0) > 0.5]
            bl_m = compute_alert_metrics(bl_alerts, iou[:n], n, iou_failure_threshold)
            baseline_totals[bl_name]["alerts"] += bl_m["n_alerts"]
            baseline_totals[bl_name]["tp"]     += bl_m["tp_alerts"]
            baseline_totals[bl_name]["false"]  += bl_m["false_alerts"]

        per_seq[seq] = {
            "n_frames": n,
            "n_alerts": m["n_alerts"],
            "n_failure_events": m["n_failure_events"],
            "tp_alerts": m["tp_alerts"],
            "false_alerts": m["false_alerts"],
            "lead_times": m["lead_times"],
            "e_trace": E_trace.tolist(),
        }

    fa_per_1000 = 1000.0 * total_false_alerts / max(total_frames, 1)

    # Build baseline result entries
    baselines: dict[str, dict] = {}
    for bl_name, _ in _baseline_heads:
        bt = baseline_totals[bl_name]
        bl_fa = 1000.0 * bt["false"] / max(total_frames, 1)
        baselines[f"baseline_raw_{bl_name}_0.5"] = {
            "total_alerts": bt["alerts"],
            "total_tp_alerts": bt["tp"],
            "total_false_alerts": bt["false"],
            "false_alerts_per_1000_frames": round(bl_fa, 3),
            "failure_event_recall": round(bt["tp"] / max(total_failure_events, 1), 4),
        }

    return {
        "config": {
            "alpha": alpha,
            "epsilon": epsilon,
            "decay": decay,
            "mode": mode,
            "risk_mode": risk_mode,
            "n_cal_seqs": len(cal_seqs),
            "n_eval_seqs": len(eval_seqs),
            "n_null_frames": int(len(null_scores)),
            "iou_failure_threshold": iou_failure_threshold,
        },
        "summary": {
            "total_frames": total_frames,
            "total_failure_events": total_failure_events,
            "total_alerts": total_alerts,
            "total_tp_alerts": total_tp_alerts,
            "total_false_alerts": total_false_alerts,
            "false_alerts_per_1000_frames": round(fa_per_1000, 3),
            "failure_event_recall": round(total_tp_alerts / max(total_failure_events, 1), 4),
            "median_lead_time": float(np.median(all_lead_times)) if all_lead_times else float("nan"),
            "mean_lead_time": float(np.mean(all_lead_times)) if all_lead_times else float("nan"),
            "p25_lead_time": float(np.percentile(all_lead_times, 25)) if all_lead_times else float("nan"),
            "p75_lead_time": float(np.percentile(all_lead_times, 75)) if all_lead_times else float("nan"),
            "seq_level_far": round(
                sum(1 for s in per_seq if per_seq[s]["false_alerts"] > 0) / max(len(per_seq), 1),
                4,
            ),
        },
        **baselines,
        "per_sequence": per_seq,
    }


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

_DEFAULT_RISK_MODES = ("ifd5", "ifd10", "ifd20", "fc_ifd20", "all_risk")


def sweep(
    preds: dict[str, list[dict[str, float]]],
    labels: dict[str, np.ndarray],
    iou_traces: dict[str, np.ndarray],
    label_names: list[str],
    epsilons: tuple[float, ...] = _DEFAULT_EPSILONS,
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS,
    decays: tuple[float, ...] = _DEFAULT_DECAYS,
    risk_modes: tuple[str, ...] = _DEFAULT_RISK_MODES,
    diagnostic_seqs: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run evaluate() over all combinations of (epsilon, alpha, decay, mode, risk_mode)."""
    rows: list[dict[str, Any]] = []
    combos = list(product(epsilons, alphas, decays, ("formal", "engineering", "agrapa"), risk_modes))
    for eps, alp, dec, mode, rmode in combos:
        if dec != 1.00 and mode == "formal":
            continue  # decay != 1 only meaningful in engineering mode
        if dec != 1.00 and mode == "agrapa":
            continue  # agrapa doesn't use decay
        result = evaluate(
            preds, labels, iou_traces, label_names,
            alpha=alp, epsilon=eps, decay=dec, mode=mode, risk_mode=rmode,
            diagnostic_seqs=diagnostic_seqs,
        )
        s = result.get("summary", {})
        row = {
            "epsilon": eps,
            "alpha": alp,
            "decay": dec,
            "mode": mode,
            "risk_mode": rmode,
            "median_lead_time": s.get("median_lead_time"),
            "failure_event_recall": s.get("failure_event_recall"),
            "false_alerts_per_1000": s.get("false_alerts_per_1000_frames"),
            "seq_level_far": s.get("seq_level_far"),
            "total_alerts": s.get("total_alerts"),
        }
        rows.append(row)
        print(
            f"  eps={eps} alpha={alp} decay={dec} mode={mode:<12} risk={rmode:<12} "
            f"lead={row['median_lead_time']:.1f}f "
            f"recall={row['failure_event_recall']:.3f} "
            f"fa/1k={row['false_alerts_per_1000']:.1f}",
            flush=True,
        )
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_npz(npz_path: str) -> tuple[dict, dict, list[str], set[str]]:
    """Return (labels_dict, iou_dict, label_names, diagnostic_seqs) from a v2 NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    label_names = list(data["label_names"])
    labels: dict[str, np.ndarray] = {}
    ious: dict[str, np.ndarray] = {}
    diagnostic_seqs: set[str] = set()
    for k in data.files:
        if k.startswith("labels/"):
            seq = k[len("labels/"):]
            labels[seq] = data[k]
        elif k.startswith("iou_trace/"):
            seq = k[len("iou_trace/"):]
            ious[seq] = data[k]
        elif k.startswith("split/"):
            seq = k[len("split/"):]
            try:
                split_val = str(data[k])
            except Exception:
                split_val = ""
            if split_val == "diagnostic":
                diagnostic_seqs.add(seq)
    # Fallback: if NPZ has no split metadata, use hardcoded set
    if not diagnostic_seqs:
        diagnostic_seqs = _DIAGNOSTIC_SEQS.copy()
    return labels, ious, label_names, diagnostic_seqs


def _load_preds(preds_path: str) -> dict[str, list[dict[str, float]]]:
    with open(preds_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2A: e-process sequential alerts")
    p.add_argument("--preds", required=True, help="saltr/results/preds_val_v2.json")
    p.add_argument("--labels", required=True, help="saltr/data/salt_rd_v2_labels.npz")
    p.add_argument("--out-json", default="saltr/results/eprocess_val_v2.json")
    p.add_argument("--out-sweep", default="saltr/results/eprocess_sweep_v2.csv")
    p.add_argument("--alpha",   type=float, default=0.10)
    p.add_argument("--epsilon", type=float, default=0.50)
    p.add_argument("--decay",   type=float, default=1.00)
    p.add_argument("--mode",    default="formal", choices=["formal", "engineering", "agrapa"])
    p.add_argument("--risk-mode", default="all_risk",
                   choices=["all_risk", "ifd5", "ifd10", "ifd20", "fc_ifd20"])
    p.add_argument("--sweep",   action="store_true", help="Run full parameter sweep")
    p.add_argument(
        "--fail-on-nogo",
        action="store_true",
        default=False,
        help="Exit with code 1 when GO verdict fails. Default: exit 0 always (NO-GO is an analysis result).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[eprocess] Loading predictions from {args.preds}", flush=True)
    preds = _load_preds(args.preds)
    print(f"[eprocess] Loading labels from {args.labels}", flush=True)
    labels, iou_traces, label_names, diagnostic_seqs = _load_npz(args.labels)
    print(f"[eprocess] {len(preds)} pred seqs, {len(labels)} label seqs, "
          f"labels={label_names}, diagnostic_seqs={len(diagnostic_seqs)}")

    if args.sweep:
        print("[eprocess] Running threshold sweep ...", flush=True)
        rows = sweep(preds, labels, iou_traces, label_names,
                     diagnostic_seqs=diagnostic_seqs)
        Path(args.out_sweep).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_sweep, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[eprocess] Sweep written to {args.out_sweep}", flush=True)

    print(
        f"[eprocess] Evaluating: alpha={args.alpha} epsilon={args.epsilon} "
        f"decay={args.decay} mode={args.mode} risk_mode={args.risk_mode}",
        flush=True,
    )
    result = evaluate(
        preds, labels, iou_traces, label_names,
        alpha=args.alpha, epsilon=args.epsilon, decay=args.decay, mode=args.mode,
        risk_mode=args.risk_mode, diagnostic_seqs=diagnostic_seqs,
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eprocess] Results written to {args.out_json}", flush=True)

    s = result["summary"]
    print("\n--- Summary ---")
    print(f"  median lead time : {s['median_lead_time']:.1f} frames  (target >= 3)")
    print(f"  event recall     : {s['failure_event_recall']:.3f}  (target >= 0.60)")
    print(f"  FA per 1000f     : {s['false_alerts_per_1000_frames']:.1f}  (target <= 100)")
    print(f"  seq-level FAR    : {s['seq_level_far']:.3f}  (target <= 0.10)")
    bl = result["baseline_raw_ifd5_0.5"]
    print(f"\n--- Baseline P(ifd5)>0.5 ---")
    print(f"  event recall : {bl['failure_event_recall']:.3f}")
    print(f"  FA per 1000f : {bl['false_alerts_per_1000_frames']:.1f}")

    go = (
        s["median_lead_time"] >= 3
        and s["failure_event_recall"] >= 0.60
        and s["false_alerts_per_1000_frames"] <= 100
    )
    print(f"\n[eprocess] GO verdict: {'✅ GO' if go else '❌ NO-GO'}")
    if not go and args.fail_on_nogo:
        sys.exit(1)


if __name__ == "__main__":
    main()
