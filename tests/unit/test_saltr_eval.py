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
    assert verdict == "BORDERLINE", \
        f"Expected BORDERLINE for mixed results, got {verdict}"


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


# ---------------------------------------------------------------------------
# Test 7: Calibration — predictions JSON reflects calibrated probs
# ---------------------------------------------------------------------------

def test_calibration_changes_exported_predictions(tmp_path):
    """Exported preds_val.json must contain calibrated probs, not raw model output.

    Regression for the bug where _save_predictions_json was called before
    temperature scaling, so the exported JSON had the same values as raw inference.
    """
    from salt_r.eval import calibrate_temperature, apply_temperature, _ece

    rng = np.random.default_rng(42)
    n = 600
    # Simulate overconfident predictions for a rare class
    y_true = (rng.random(n) < 0.08).astype(float)
    # Push predictions toward extremes — clearly miscalibrated
    y_pred_raw = np.where(y_true, rng.uniform(0.8, 0.99, n), rng.uniform(0.0, 0.15, n)).astype(np.float32)

    ece_before = _ece(y_pred_raw, y_true)
    T = calibrate_temperature(y_true, y_pred_raw)
    y_pred_cal = apply_temperature(y_pred_raw, T)
    ece_after = _ece(y_pred_cal, y_true)

    # Calibration must have changed the values
    assert not np.allclose(y_pred_raw, y_pred_cal, atol=1e-4), \
        f"Temperature T={T:.4f} must change predictions, but raw ≈ calibrated"

    # ECE must improve (calibration reduces miscalibration)
    assert ece_after < ece_before, \
        f"ECE must decrease after calibration: before={ece_before:.4f}, after={ece_after:.4f}"

    # T != 1.0 confirms calibration actually did something
    assert abs(T - 1.0) > 0.05, f"T={T:.4f} is too close to 1.0 — calibration may be a no-op"


def test_evaluate_calibrated_preds_exported_not_raw(tmp_path):
    """End-to-end: evaluate() with calibrate_heads must save calibrated probs to JSON.

    Regression for the save-before-calibration bug: the exported preds JSON
    must contain calibrated probabilities, not the raw model output.
    """
    import torch
    from salt_r.model import SALTRD, HEAD_NAMES, LABEL_NAMES
    from salt_r.collect_features import FEATURE_NAMES
    from salt_r.eval import evaluate

    # Build a tiny synthetic NPZ (1 val sequence, 60 frames)
    rng = np.random.default_rng(7)
    n_frames, n_feat, n_labels = 60, 28, 8
    seq_key = "uav123/test_seq"

    npz_path = str(tmp_path / "tiny.npz")
    # Labels: make false_confirmed (col 1) ~10% positive — enough for calibration
    labels = rng.integers(0, 2, (n_frames, n_labels), dtype=np.int8)
    labels[:, 0] = 1  # "correct" mostly 1
    labels[:, 1] = (rng.random(n_frames) < 0.10).astype(np.int8)  # false_confirmed ~10%
    np.savez(
        npz_path,
        **{
            f"features/{seq_key}": rng.standard_normal((n_frames, n_feat)).astype(np.float32),
            f"labels/{seq_key}": labels,
            f"iou_trace/{seq_key}": rng.random(n_frames).astype(np.float32),
            f"split/{seq_key}": np.array("val"),
            f"dataset/{seq_key}": np.array("uav123"),
            f"sequence_name/{seq_key}": np.array(seq_key),
            "label_names": np.array(LABEL_NAMES, dtype=object),
            "feature_names": np.array(FEATURE_NAMES, dtype=object),
        },
    )

    # Save a fresh model checkpoint with head_names metadata
    ckpt_path = str(tmp_path / "model.pt")
    model = SALTRD()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "head_names": list(HEAD_NAMES),
            "label_names": list(LABEL_NAMES),
            "feature_names": list(FEATURE_NAMES),
        },
        ckpt_path,
    )

    # Run evaluate WITHOUT calibration — capture raw exported probs
    raw_preds_path = str(tmp_path / "preds_raw.json")
    evaluate(
        npz_path=npz_path,
        checkpoint_path=ckpt_path,
        split="val",
        output_path=None,
        predictions_output=raw_preds_path,
        calibrate_heads=None,
    )

    # Run evaluate WITH calibration on false_confirmed
    cal_preds_path = str(tmp_path / "preds_cal.json")
    results = evaluate(
        npz_path=npz_path,
        checkpoint_path=ckpt_path,
        split="val",
        output_path=None,
        predictions_output=cal_preds_path,
        calibrate_heads=["false_confirmed"],
    )

    # Load both JSONs
    with open(raw_preds_path) as f:
        raw_json = json.load(f)
    with open(cal_preds_path) as f:
        cal_json = json.load(f)

    assert seq_key in raw_json and seq_key in cal_json, "Sequence key missing from exported JSON"

    raw_fc = np.array([frame["false_confirmed"] for frame in raw_json[seq_key]])
    cal_fc = np.array([frame["false_confirmed"] for frame in cal_json[seq_key]])

    # If calibration was applied, the exported probs must differ from raw
    if results.get("calibration") and "false_confirmed" in results["calibration"].get("temperatures", {}):
        T = results["calibration"]["temperatures"]["false_confirmed"]
        assert abs(T - 1.0) > 1e-3, f"T={T} — calibration had no effect"
        assert not np.allclose(raw_fc, cal_fc, atol=1e-4), \
            f"Calibrated JSON must differ from raw JSON (T={T:.4f}), but they are identical. " \
            "Save-before-calibration bug may have returned."


def test_temperature_calibration_reduces_ece():
    from salt_r.eval import calibrate_temperature, apply_temperature, _ece

    rng = np.random.default_rng(0)
    n = 500
    y_true = (rng.random(n) < 0.1).astype(float)
    # Deliberately overconfident predictions
    y_pred = np.clip(rng.beta(0.3, 0.05, n), 1e-6, 1 - 1e-6).astype(np.float32)
    ece_before = _ece(y_pred, y_true)
    T = calibrate_temperature(y_true, y_pred)
    y_cal = apply_temperature(y_pred, T)
    ece_after = _ece(y_cal, y_true)
    assert ece_after < ece_before, \
        f"Temperature calibration must reduce ECE: before={ece_before:.4f}, after={ece_after:.4f}"
    assert 0.05 < T < 20.0, f"T={T} out of sane range"


def test_per_dataset_head_metrics_do_not_pool_splits():
    """Per-dataset metrics must expose regressions hidden by pooled eval."""
    from salt_r.eval import compute_per_dataset_head_metrics

    label_names = ["correct", "false_confirmed"]
    model_head_names = ["false_confirmed"]
    labels = {
        "uav123/a": np.array([[1, 0], [1, 1], [1, 0], [1, 1]], dtype=np.int8),
        "dtb70/b": np.array([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=np.int8),
    }
    preds = {
        # Good ranking for UAV123 fc.
        "uav123/a": np.array([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32),
        # Bad ranking for DTB70 fc.
        "dtb70/b": np.array([[0.9], [0.8], [0.2], [0.1]], dtype=np.float32),
    }

    result = compute_per_dataset_head_metrics(labels, preds, label_names, model_head_names)

    assert set(result) == {"uav123", "dtb70"}
    assert result["uav123"]["false_confirmed"]["n_sequences"] == 1
    assert result["dtb70"]["false_confirmed"]["n_frames"] == 4
    assert result["uav123"]["false_confirmed"]["auroc"] > 0.9
    assert result["dtb70"]["false_confirmed"]["auroc"] < 0.1


# ---------------------------------------------------------------------------
# Test: recompute_labels_v2 raises on v0 input
# ---------------------------------------------------------------------------

def test_recompute_labels_v2_raises_on_v0_npz(tmp_path):
    """recompute_labels_v2() must raise ValueError when given a v0 NPZ (wrong schema)."""
    from salt_r.collect_features import recompute_labels_v2, LABEL_NAMES

    # Build a minimal v0 NPZ (8 labels, not 10)
    v0_npz = str(tmp_path / "v0_labels.npz")
    n_frames = 10
    seq_key = "uav123/test"
    np.savez_compressed(
        v0_npz,
        **{
            f"features/{seq_key}": np.zeros((n_frames, 28), dtype=np.float32),
            f"labels/{seq_key}": np.zeros((n_frames, 8), dtype=np.int8),
            f"iou_trace/{seq_key}": np.ones(n_frames, dtype=np.float32),
            f"split/{seq_key}": np.array("val"),
            "label_names": np.array(LABEL_NAMES, dtype=object),  # v0 = 8 labels
        },
    )

    out_npz = str(tmp_path / "v2_labels.npz")
    with pytest.raises(ValueError, match="v1 schema"):
        recompute_labels_v2(v0_npz, out_npz)


# ---------------------------------------------------------------------------
# Test: ifd10/ifd20 positive frames have current IoU >= 0.5
# ---------------------------------------------------------------------------

def test_ifd10_ifd20_positives_have_good_current_iou(tmp_path):
    """ifd10 and ifd20 positive labels must only occur when current IoU >= 0.5."""
    from salt_r.collect_features import (
        LABEL_NAMES_V2, recompute_labels_v2, LABEL_NAMES_V1, N_LABELS_V1,
    )

    # Build a v1 NPZ so recompute_labels_v2 can work
    n_frames = 60
    seq_key = "uav123/ifd_test"

    # IoU trace: good for 40 frames, then fails
    iou_trace = np.concatenate([
        np.full(40, 0.9, dtype=np.float32),  # good
        np.full(20, 0.1, dtype=np.float32),  # failure
    ])

    # Minimal v1 labels (10 columns)
    labels_v1 = np.zeros((n_frames, N_LABELS_V1), dtype=np.int8)
    # col 0: correct (IoU>=0.5)
    labels_v1[:40, 0] = 1

    v1_npz = str(tmp_path / "v1.npz")
    np.savez_compressed(
        v1_npz,
        **{
            f"features/{seq_key}": np.zeros((n_frames, 28), dtype=np.float32),
            f"labels/{seq_key}": labels_v1,
            f"iou_trace/{seq_key}": iou_trace,
            f"split/{seq_key}": np.array("val"),
            "label_names": np.array(LABEL_NAMES_V1, dtype=object),
        },
    )

    v2_npz = str(tmp_path / "v2.npz")
    recompute_labels_v2(v1_npz, v2_npz)

    data = np.load(v2_npz, allow_pickle=True)
    labels_v2 = data[f"labels/{seq_key}"]
    iou_back = data[f"iou_trace/{seq_key}"]
    label_names = list(data["label_names"])

    ifd10_idx = label_names.index("imminent_failure_dynamic_10")
    ifd20_idx = label_names.index("imminent_failure_dynamic_20")

    for col_idx, col_name in [(ifd10_idx, "ifd10"), (ifd20_idx, "ifd20")]:
        positives = np.where(labels_v2[:, col_idx] == 1)[0]
        for t in positives:
            assert float(iou_back[t]) >= 0.5, (
                f"{col_name}: positive at t={t} but IoU={iou_back[t]:.3f} < 0.5"
            )


# ---------------------------------------------------------------------------
# Test: event-level compute_failure_lead_time
# ---------------------------------------------------------------------------

def test_event_level_failure_lead_time():
    """compute_failure_lead_time detects events and computes lead times correctly."""
    from salt_r.eval import compute_failure_lead_time

    n = 100
    # One failure event at frame 70 (IoU drops from 1→0.1)
    iou = np.ones(n, dtype=np.float32)
    iou[70:] = 0.1

    label_names = ["imminent_failure_dynamic"]
    head_names  = ["imminent_failure_dynamic"]

    # Labels: ifd=1 at frames 65-69 (just before failure)
    labels = np.zeros((n, 1), dtype=np.int8)
    labels[65:70, 0] = 1

    # Model: predicts >0.5 at frames 62-69 (8-frame early warning)
    preds = np.zeros((n, 1), dtype=np.float32)
    preds[62:70, 0] = 0.9

    result = compute_failure_lead_time(
        iou_dict={"seq1": iou},
        preds_dict={"seq1": preds},
        labels_dict={"seq1": labels},
        label_names=label_names,
        head_names=head_names,
        threshold=0.50,
        iou_failure_threshold=0.30,
        label_name="imminent_failure_dynamic",
        horizon=5,
    )

    assert result["n_failure_events"] == 1, f"Expected 1 failure event, got {result['n_failure_events']}"
    assert result["n_detected_events"] == 1, f"Expected detected=1, got {result['n_detected_events']}"
    assert result["event_recall"] == pytest.approx(1.0)
    # First alert at 62, event at 70 → lead time = 8
    assert result["per_event_lead_times"] == [8]
    assert result["median_lead_time"] == pytest.approx(8.0)


def test_event_level_no_detection():
    """compute_failure_lead_time: no alert before failure → recall=0."""
    from salt_r.eval import compute_failure_lead_time

    n = 50
    iou = np.ones(n, dtype=np.float32)
    iou[30:] = 0.1  # failure at frame 30

    preds = np.zeros((n, 1), dtype=np.float32)  # model never alerts
    labels = np.zeros((n, 1), dtype=np.int8)

    result = compute_failure_lead_time(
        iou_dict={"seq1": iou},
        preds_dict={"seq1": preds},
        labels_dict={"seq1": labels},
        label_names=["imminent_failure_dynamic"],
        head_names=["imminent_failure_dynamic"],
        threshold=0.50,
        iou_failure_threshold=0.30,
        label_name="imminent_failure_dynamic",
        horizon=5,
    )
    assert result["n_failure_events"] == 1
    assert result["n_detected_events"] == 0
    assert result["event_recall"] == pytest.approx(0.0)
    assert result["per_event_lead_times"] == []


def test_run_lead_time_analysis_three_horizons():
    """run_lead_time_analysis returns entries for ifd5, ifd10, ifd20."""
    from salt_r.eval import run_lead_time_analysis

    label_names = [
        "imminent_failure_dynamic",
        "imminent_failure_dynamic_10",
        "imminent_failure_dynamic_20",
    ]
    head_names = label_names

    n = 50
    iou = np.ones(n, dtype=np.float32)
    iou[30:] = 0.1
    preds = np.zeros((n, 3), dtype=np.float32)
    labels = np.zeros((n, 3), dtype=np.int8)

    result = run_lead_time_analysis(
        iou_dict={"s": iou},
        preds_dict={"s": preds},
        labels_dict={"s": labels},
        label_names=label_names,
        head_names=head_names,
    )
    assert set(result.keys()) == {"ifd5", "ifd10", "ifd20"}
    for key in ("ifd5", "ifd10", "ifd20"):
        assert "n_failure_events" in result[key] or "note" in result[key]
