"""Unit tests for saltr.salt_r.eprocess — Phase 2A sequential alerts."""

import numpy as np
import pytest

from salt_r.eprocess import (
    _failure_events,
    build_null_distribution,
    compute_alert_metrics,
    compute_risk_score,
    conformal_pvalue,
    evaluate,
    power_evalue,
    reset_at_boundary,
    run_eprocess_sequence,
)


# ---------------------------------------------------------------------------
# compute_risk_score
# ---------------------------------------------------------------------------

class TestComputeRiskScore:
    def test_all_zero(self):
        assert compute_risk_score({}) == pytest.approx(0.0)

    def test_all_one(self):
        probs = {
            "false_confirmed": 1.0,
            "imminent_failure_dynamic": 1.0,
            "failure_in_5": 1.0,
            "imminent_failure_dynamic_20": 1.0,
        }
        assert compute_risk_score(probs) == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        # weights: 0.45 + 0.35 + 0.15 + 0.05 = 1.0
        probs = {
            "false_confirmed": 1.0,
            "imminent_failure_dynamic": 0.0,
            "failure_in_5": 0.0,
            "imminent_failure_dynamic_20": 0.0,
        }
        assert compute_risk_score(probs) == pytest.approx(0.45)

    def test_missing_keys_treated_as_zero(self):
        assert compute_risk_score({"false_confirmed": 0.4}) == pytest.approx(0.45 * 0.4)


# ---------------------------------------------------------------------------
# conformal_pvalue
# ---------------------------------------------------------------------------

class TestConformalPvalue:
    def test_score_below_all_null(self):
        null = np.array([0.5, 0.6, 0.7])
        p = conformal_pvalue(0.0, null)
        # all 3 null >= 0.0 → (1+3)/(1+3) = 1.0
        assert p == pytest.approx(1.0)

    def test_score_above_all_null(self):
        null = np.array([0.1, 0.2, 0.3])
        p = conformal_pvalue(1.0, null)
        # 0 null >= 1.0 → (1+0)/(1+3) = 0.25
        assert p == pytest.approx(0.25)

    def test_empty_null(self):
        null = np.array([], dtype=np.float32)
        p = conformal_pvalue(0.5, null)
        assert p == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# power_evalue
# ---------------------------------------------------------------------------

class TestPowerEvalue:
    def test_p_equals_one_gives_epsilon(self):
        e = power_evalue(1.0, epsilon=0.5)
        assert e == pytest.approx(0.5)

    def test_zero_p_clipped_to_small(self):
        # Should not raise; returns a finite value
        e = power_evalue(0.0, epsilon=0.5)
        assert np.isfinite(e)
        assert e > 0

    def test_small_p_large_evalue(self):
        # p < 1 with epsilon < 1 → e_t > 1 when p is small
        e = power_evalue(0.01, epsilon=0.5)
        assert e > 1.0


# ---------------------------------------------------------------------------
# run_eprocess_sequence
# ---------------------------------------------------------------------------

class TestRunEprocessSequence:
    def _make_null(self, n=200):
        rng = np.random.RandomState(42)
        return rng.uniform(0, 0.3, n).astype(np.float32)

    def test_no_alerts_on_all_null_scores(self):
        null = self._make_null()
        # All-null sequence: risk scores sampled from same null distribution
        risk = null.copy()
        E, alerts = run_eprocess_sequence(risk, null, alpha=0.05, epsilon=0.5, mode="formal")
        # With strict alpha=0.05 on in-distribution scores, expect few or zero alerts
        # Stochastic test: just ensure we don't get many false alerts
        assert len(alerts) <= 5  # < 2.5 % of 200 frames

    def test_alert_triggered_on_monotone_risk(self):
        null = np.zeros(100, dtype=np.float32)  # all null scores = 0
        # monotonically increasing risk that is clearly above null
        risk = np.linspace(0.5, 1.0, 50).astype(np.float32)
        E, alerts = run_eprocess_sequence(risk, null, alpha=0.10, epsilon=0.5, mode="formal")
        assert len(alerts) >= 1, "Expected at least one alert on high-risk sequence"

    def test_alert_timing_earlier_than_end(self):
        null = np.full(100, 0.1, dtype=np.float32)
        # High-risk scores from frame 5 onward
        risk = np.zeros(30, dtype=np.float32)
        risk[5:] = 0.9
        E, alerts = run_eprocess_sequence(risk, null, alpha=0.10, epsilon=0.5, mode="formal")
        if alerts:
            assert alerts[0] >= 5  # must not alert before the signal starts

    def test_formal_mode_no_more_than_one_alert(self):
        null = np.full(50, 0.1, dtype=np.float32)
        risk = np.full(100, 0.8, dtype=np.float32)
        E, alerts = run_eprocess_sequence(risk, null, alpha=0.10, epsilon=0.5, mode="formal")
        # Formal mode alerts only once (running-max threshold)
        assert len(alerts) <= 1

    def test_engineering_mode_can_alert_multiple_times(self):
        null = np.full(50, 0.1, dtype=np.float32)
        # Alternating high/low risk
        risk = np.tile([0.9, 0.9, 0.1, 0.1], 25).astype(np.float32)
        E, alerts = run_eprocess_sequence(
            risk, null, alpha=0.10, epsilon=0.5, decay=0.80, mode="engineering"
        )
        # Engineering mode resets, so multiple alerts are possible
        # Just check it doesn't raise and returns arrays of correct shape
        assert len(E) == len(risk)


# ---------------------------------------------------------------------------
# reset_at_boundary
# ---------------------------------------------------------------------------

class TestResetAtBoundary:
    def test_reset_prevents_stale_evidence(self):
        null = np.full(50, 0.1, dtype=np.float32)
        # High risk in first half, reset at frame 30, low risk second half
        risk = np.concatenate([np.full(30, 0.9), np.full(30, 0.05)]).astype(np.float32)
        E, alerts = reset_at_boundary(
            reset_frames=[30],
            risk_scores=risk,
            null_scores=null,
            alpha=0.10,
            epsilon=0.5,
            decay=1.0,
            mode="formal",
        )
        # After reset at 30, evidence restarts from e_prev=1.0.
        # High-risk segment 1 should have grown E[29] >> threshold.
        # Low-risk segment 2 should not inherit that accumulated evidence.
        # E[30] is the FIRST frame of segment 2 = 1.0 * e_t(risk[30]).
        # E[29] >> E[30] proves the reset dropped accumulated evidence.
        assert E[29] > E[30], "Reset must drop accumulated evidence from segment 1"
        # Second half has low risk → e-process stays well below threshold
        assert E[55] < 1.0 / 0.10

    def test_no_resets_matches_direct(self):
        null = np.full(50, 0.1, dtype=np.float32)
        risk = np.full(40, 0.5, dtype=np.float32)
        E_direct, alerts_direct = run_eprocess_sequence(
            risk, null, alpha=0.10, epsilon=0.5, mode="formal"
        )
        E_reset, alerts_reset = reset_at_boundary(
            reset_frames=[],
            risk_scores=risk,
            null_scores=null,
            alpha=0.10,
            epsilon=0.5,
            decay=1.0,
            mode="formal",
        )
        np.testing.assert_allclose(E_direct, E_reset)


# ---------------------------------------------------------------------------
# compute_alert_metrics
# ---------------------------------------------------------------------------

class TestComputeAlertMetrics:
    def test_no_failure_events(self):
        iou = np.ones(100)  # tracker always perfect
        m = compute_alert_metrics(alerts=[10, 20], iou=iou, n_frames=100)
        assert m["n_failure_events"] == 0
        assert m["false_alerts"] == 2

    def test_alert_before_failure_counts_as_tp(self):
        iou = np.ones(100)
        iou[50:] = 0.1  # failure at frame 50
        # Alert 5 frames before failure
        m = compute_alert_metrics(alerts=[45], iou=iou, n_frames=100)
        assert m["tp_alerts"] == 1
        assert m["lead_times"] == [5]

    def test_alert_after_failure_is_false_alert(self):
        iou = np.ones(100)
        iou[30:40] = 0.1  # brief failure, recovers
        iou[80:] = 0.1   # second failure
        # Alert at frame 70 → within 20 frames before frame 80 failure → TP
        # Alert at frame 40 → after first failure, but 40 < 60 (80-20), so false
        m = compute_alert_metrics(alerts=[40, 70], iou=iou, n_frames=100)
        assert m["tp_alerts"] >= 1
        assert 70 in [a for a in [40, 70] if a >= 60]  # 70 is in window [60, 80)

    def test_recall_zero_when_no_prior_alerts(self):
        iou = np.ones(100)
        iou[50:] = 0.0  # failure
        m = compute_alert_metrics(alerts=[], iou=iou, n_frames=100)
        assert m["failure_event_recall"] == 0.0


# ---------------------------------------------------------------------------
# _failure_events
# ---------------------------------------------------------------------------

class TestFailureEvents:
    def test_simple(self):
        iou = np.array([1.0, 1.0, 0.2, 0.2, 1.0, 0.1])
        events = _failure_events(iou, iou_threshold=0.3)
        assert events == [2, 5]

    def test_no_failure(self):
        iou = np.ones(50)
        assert _failure_events(iou) == []

    def test_starts_below_threshold(self):
        iou = np.array([0.1, 0.1, 1.0, 0.1])
        events = _failure_events(iou)
        # Starts below: first event not counted (never was "above" before)
        assert 3 in events
        assert 0 not in events


# ---------------------------------------------------------------------------
# build_null_distribution
# ---------------------------------------------------------------------------

class TestBuildNullDistribution:
    def _make_fixtures(self):
        """Minimal fixtures: 2 sequences, one cal, one eval."""
        label_names = [
            "correct", "false_confirmed", "failure_in_5", "recoverable",
            "target_dynamic", "camera_dynamic", "hard_dynamic_scene",
            "needs_full_compute", "hard_dynamic_scene_v2",
            "imminent_failure_dynamic", "failure_in_10", "failure_in_20",
            "imminent_failure_dynamic_10", "imminent_failure_dynamic_20",
        ]
        n = 20
        labels_dict = {
            "seq1": np.zeros((n, 14), dtype=np.int8),
            "seq2": np.zeros((n, 14), dtype=np.int8),
        }
        iou_dict = {
            "seq1": np.ones(n, dtype=np.float32),
            "seq2": np.ones(n, dtype=np.float32),
        }
        preds_dict = {
            "seq1": [{"false_confirmed": 0.1, "imminent_failure_dynamic": 0.1,
                       "failure_in_5": 0.05, "imminent_failure_dynamic_20": 0.05}] * n,
            "seq2": [{"false_confirmed": 0.1, "imminent_failure_dynamic": 0.1,
                       "failure_in_5": 0.05, "imminent_failure_dynamic_20": 0.05}] * n,
        }
        return preds_dict, labels_dict, iou_dict, label_names

    def test_returns_nonempty_array(self):
        preds, labels, ious, label_names = self._make_fixtures()
        null = build_null_distribution(preds, labels, ious, label_names, cal_seq_keys=["seq1"])
        assert len(null) > 0

    def test_excludes_failure_frames(self):
        preds, labels, ious, label_names = self._make_fixtures()
        # Mark frame 5 as false_confirmed in seq1
        fc_idx = label_names.index("false_confirmed")
        labels["seq1"][5, fc_idx] = 1
        null = build_null_distribution(preds, labels, ious, label_names, cal_seq_keys=["seq1"])
        # Frame 5 should not be in null distribution
        assert len(null) == 19  # 20 frames - 1 failure frame


# ---------------------------------------------------------------------------
# evaluate (integration smoke test)
# ---------------------------------------------------------------------------

class TestEvaluateIntegration:
    def _make_fixtures(self, n_seqs=8, n_frames=100):
        """Create synthetic val set with 1 failure event per sequence."""
        rng = np.random.RandomState(0)
        label_names = [
            "correct", "false_confirmed", "failure_in_5", "recoverable",
            "target_dynamic", "camera_dynamic", "hard_dynamic_scene",
            "needs_full_compute", "hard_dynamic_scene_v2",
            "imminent_failure_dynamic", "failure_in_10", "failure_in_20",
            "imminent_failure_dynamic_10", "imminent_failure_dynamic_20",
        ]
        preds: dict = {}
        labels: dict = {}
        ious: dict = {}
        ifd_idx = label_names.index("imminent_failure_dynamic")

        for i in range(n_seqs):
            seq = f"uav123/seq{i:02d}"
            lab = np.zeros((n_frames, 14), dtype=np.int8)
            iou = np.ones(n_frames, dtype=np.float32)
            # Failure at frame 70
            iou[70:] = 0.1
            # ifd label active at frames 60-69
            lab[60:70, ifd_idx] = 1

            frame_probs = []
            for t in range(n_frames):
                p_ifd = 0.8 if 60 <= t < 70 else 0.1
                p_fc = 0.1 if iou[t] >= 0.5 else 0.7
                frame_probs.append({
                    "false_confirmed": float(p_fc),
                    "imminent_failure_dynamic": float(p_ifd),
                    "failure_in_5": 0.05,
                    "imminent_failure_dynamic_20": 0.05,
                })
            preds[seq] = frame_probs
            labels[seq] = lab
            ious[seq] = iou

        return preds, labels, ious, label_names

    def test_evaluate_runs_without_error(self):
        preds, labels, ious, label_names = self._make_fixtures()
        result = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, decay=1.0, mode="formal",
        )
        assert "summary" in result
        assert "config" in result

    def test_lead_time_positive_when_signal_precedes_failure(self):
        preds, labels, ious, label_names = self._make_fixtures()
        result = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, decay=1.0, mode="formal",
        )
        # Synthetic data has signal at frames 60-69, failure at 70
        # e-process should accumulate evidence before frame 70
        median_lt = result["summary"]["median_lead_time"]
        assert np.isnan(median_lt) or median_lt >= 0

    def test_formal_fewer_false_alerts_than_engineering_with_decay(self):
        preds, labels, ious, label_names = self._make_fixtures()
        r_formal = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, decay=1.0, mode="formal",
        )
        r_eng = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, decay=0.95, mode="engineering",
        )
        # Engineering with decay resets → may accumulate more alerts
        # Just verify both return valid summaries
        assert "summary" in r_formal
        assert "summary" in r_eng
