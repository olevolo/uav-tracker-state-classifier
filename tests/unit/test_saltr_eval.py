"""Unit tests for saltr/src/salt_r/eval.py — metrics correctness.

All tests use only numpy + the eval module; no model weights or NPZ files needed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: wrap a flat metric dict into the nested structure check_go_nogo expects.
# check_go_nogo looks up results["head_metrics"][head][metric], where the flat
# key is "<metric>_<head>" (split on the FIRST underscore only).
# ---------------------------------------------------------------------------

def _make_go_nogo_input(flat: dict[str, float]) -> dict:
    """Convert {'auprc_false_confirmed': 0.35, ...} into
    {'head_metrics': {'false_confirmed': {'auprc': 0.35}, ...}}."""
    head_metrics: dict[str, dict[str, float]] = {}
    for key, val in flat.items():
        # split on first underscore only: "auprc_false_confirmed" -> ("auprc", "false_confirmed")
        parts = key.split("_", 1)
        if len(parts) != 2:
            continue
        metric, head = parts
        head_metrics.setdefault(head, {})[metric] = val
    return {"head_metrics": head_metrics}


# ---------------------------------------------------------------------------
# Test 1: ECE known values
# ---------------------------------------------------------------------------

def test_ece_known_values():
    from salt_r.eval import _ece

    # Perfect calibration: predicted prob equals actual frequency -> ECE ~0
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    labels = np.array([0,   0,   0,   0,   1,   1,   1,   1,   1,   1 ], dtype=float)
    ece_perfect = _ece(probs, labels, n_bins=5)

    # Terrible calibration: always predict 0.9 when half correct -> ECE ~0.4
    probs_bad = np.full(10, 0.9)
    labels_bad = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    ece_bad = _ece(probs_bad, labels_bad, n_bins=5)

    assert ece_perfect < 0.25, f"Expected low ECE for reasonable calibration, got {ece_perfect}"
    assert ece_bad > ece_perfect, f"Bad calibration should have higher ECE: {ece_bad} vs {ece_perfect}"


# ---------------------------------------------------------------------------
# Test 2: NT2F known trace
# ---------------------------------------------------------------------------

def test_nt2f_known_trace():
    from salt_r.eval import compute_nt2f

    # IoU [1, .8, .6, .4, .3] at threshold 0.5 -> failure at frame 3 (0-indexed)
    # NT2F = 3/5 = 0.6
    iou = {"seq1": np.array([1.0, 0.8, 0.6, 0.4, 0.3], dtype=np.float32)}
    result = compute_nt2f(iou, iou_threshold=0.5)
    assert abs(result["nt2f_mean"] - 3/5) < 0.01, f"Expected NT2F=0.6, got {result['nt2f_mean']}"
    assert result["n_never_failed"] == 0
    assert result["n_sequences"] == 1

    # Sequence that never fails -> NT2F = 1.0
    iou_good = {"seq2": np.array([1.0, 0.9, 0.8, 0.7, 0.6], dtype=np.float32)}
    result_good = compute_nt2f(iou_good, iou_threshold=0.5)
    assert result_good["nt2f_mean"] == 1.0
    assert result_good["n_never_failed"] == 1


# ---------------------------------------------------------------------------
# Test 3: Bootstrap CI — sequence level
# ---------------------------------------------------------------------------

def test_bootstrap_ci_sequence_level():
    from salt_r.eval import bootstrap_ci

    rng = np.random.default_rng(42)
    scores = rng.uniform(0.2, 0.8, size=30).tolist()
    lo, hi = bootstrap_ci(scores, n_bootstrap=500, seed=42)
    mean = sum(scores) / len(scores)
    assert lo < mean < hi, f"Mean {mean:.3f} not in CI [{lo:.3f}, {hi:.3f}]"
    assert hi - lo < 0.3, f"CI too wide: {hi-lo:.3f}"

    # Single-element degenerate case: CI = (val, val)
    lo1, hi1 = bootstrap_ci([0.5], n_bootstrap=100, seed=0)
    assert lo1 == hi1 == 0.5


# ---------------------------------------------------------------------------
# Test 4: GO/NO-GO thresholds
# ---------------------------------------------------------------------------

def test_go_nogo_thresholds():
    from salt_r.eval import check_go_nogo

    # Clearly GO
    go_results = _make_go_nogo_input({
        "auprc_false_confirmed": 0.35,
        "auroc_false_confirmed": 0.80,
        "auroc_failure_in_5":    0.80,
        "auroc_hard_dynamic_scene": 0.80,
        "auroc_needs_full_compute": 0.75,
        "ece_false_confirmed":   0.08,
    })
    assert check_go_nogo(go_results) == "GO", \
        f"Expected GO, got {check_go_nogo(go_results)}"

    # Clearly STOP
    stop_results = _make_go_nogo_input({
        "auprc_false_confirmed": 0.10,
        "auroc_false_confirmed": 0.50,
        "auroc_failure_in_5":    0.60,
        "auroc_hard_dynamic_scene": 0.55,
        "auroc_needs_full_compute": 0.55,
        "ece_false_confirmed":   0.25,
    })
    assert check_go_nogo(stop_results) == "STOP", \
        f"Expected STOP, got {check_go_nogo(stop_results)}"

    # BORDERLINE: some pass, some fail
    border_results = _make_go_nogo_input({
        "auprc_false_confirmed": 0.32,        # pass
        "auroc_false_confirmed": 0.88,        # pass
        "auroc_failure_in_5":    0.86,        # pass
        "auroc_hard_dynamic_scene": 0.64,     # fail (< 0.75)
        "auroc_needs_full_compute": 0.64,     # fail (< 0.70)
        "ece_false_confirmed":   0.32,        # fail (> 0.12)
    })
    verdict = check_go_nogo(border_results)
    assert verdict in ("BORDERLINE", "GO"), \
        f"Expected BORDERLINE/GO for mixed results, got {verdict}"


# ---------------------------------------------------------------------------
# Test 5: Predictions JSON export schema
# ---------------------------------------------------------------------------

def test_predictions_json_export_schema(tmp_path):
    from salt_r.eval import _save_predictions_json
    from salt_r.model import HEAD_NAMES

    # Fake predictions: 2 sequences, 5 frames each, 7 heads
    preds = {
        "uav123/bike1": np.random.rand(5, 7).astype(np.float32),
        "dtb70/Gull2":  np.random.rand(5, 7).astype(np.float32),
    }
    out_path = str(tmp_path / "preds.json")
    _save_predictions_json(preds, out_path)

    with open(out_path) as f:
        data = json.load(f)

    assert set(data.keys()) == {"uav123/bike1", "dtb70/Gull2"}
    for seq_key, frames in data.items():
        assert len(frames) == 5, f"{seq_key}: expected 5 frames"
        for frame in frames:
            assert set(frame.keys()) == set(HEAD_NAMES), \
                f"{seq_key}: unexpected keys {set(frame.keys())}"
            for h, p in frame.items():
                assert isinstance(p, float)
                assert 0.0 <= p <= 1.0, f"{seq_key}/{h}: {p} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 6: Head metrics base rate sanity
# ---------------------------------------------------------------------------

def test_head_metrics_base_rate_sanity():
    from salt_r.eval import compute_head_metrics

    rng = np.random.default_rng(0)
    n = 200
    y_true = (rng.random(n) < 0.05).astype(float)  # 5% base rate
    y_pred = rng.random(n).astype(float)             # random predictions
    m = compute_head_metrics(y_true, y_pred, "false_confirmed")
    assert abs(m["base_rate"] - y_true.mean()) < 0.01
    # Random predictions -> AUROC ≈ 0.5 (within tolerance)
    assert 0.3 < m["auroc"] < 0.7, f"Random AUROC should be ~0.5, got {m['auroc']:.3f}"
    # ECE/Brier/NLL must be finite
    for key in ("ece", "brier", "nll"):
        assert np.isfinite(m[key]), f"{key} is not finite: {m[key]}"
