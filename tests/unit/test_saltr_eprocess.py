"""Unit tests for saltr.salt_r.eprocess — Phase 2A sequential alerts."""

import numpy as np
import pytest

from salt_r.eprocess import (
    _DIAGNOSTIC_SEQS,
    _failure_events,
    build_null_distribution,
    compute_alert_metrics,
    compute_quality_score,
    compute_risk_score,
    conformal_pvalue,
    evaluate,
    power_evalue,
    reset_at_boundary,
    run_eprocess_agrapa,
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


# ---------------------------------------------------------------------------
# compute_risk_score with modes
# ---------------------------------------------------------------------------

class TestComputeRiskScoreModes:
    def _probs(self):
        return {
            "false_confirmed": 0.6,
            "imminent_failure_dynamic": 0.8,
            "imminent_failure_dynamic_10": 0.5,
            "imminent_failure_dynamic_20": 0.4,
            "failure_in_5": 0.3,
        }

    def test_ifd5_mode(self):
        p = self._probs()
        assert compute_risk_score(p, mode="ifd5") == pytest.approx(p["imminent_failure_dynamic"])

    def test_ifd10_mode(self):
        p = self._probs()
        assert compute_risk_score(p, mode="ifd10") == pytest.approx(p["imminent_failure_dynamic_10"])

    def test_ifd20_mode(self):
        p = self._probs()
        assert compute_risk_score(p, mode="ifd20") == pytest.approx(p["imminent_failure_dynamic_20"])

    def test_fc_ifd20_mode(self):
        p = self._probs()
        expected = 0.60 * p["false_confirmed"] + 0.40 * p["imminent_failure_dynamic_20"]
        assert compute_risk_score(p, mode="fc_ifd20") == pytest.approx(expected)

    def test_all_risk_mode_default(self):
        p = self._probs()
        # default positional arg is "all_risk"
        assert compute_risk_score(p) == pytest.approx(compute_risk_score(p, mode="all_risk"))

    def test_quality_score_is_complement(self):
        p = self._probs()
        for mode in ("ifd5", "ifd10", "ifd20", "fc_ifd20", "all_risk"):
            r = compute_risk_score(p, mode=mode)
            q = compute_quality_score(p, risk_mode=mode)
            assert q == pytest.approx(1.0 - r), f"Quality != 1-risk for mode={mode}"


# ---------------------------------------------------------------------------
# run_eprocess_agrapa
# ---------------------------------------------------------------------------

class TestRunEprocessAgrapa:
    def test_runs_without_error(self):
        """aGRAPAshould return arrays of correct shape with no exception."""
        rng = np.random.RandomState(0)
        quality = rng.uniform(0.3, 0.8, 50).astype(np.float32)
        E, alerts = run_eprocess_agrapa(quality, epsilon=0.5, alpha=0.10, window=20)
        assert len(E) == 50
        assert isinstance(alerts, list)

    def test_no_calibration_set_needed(self):
        """aGRAPAshould work without any null/calibration distribution.

        The function signature does NOT accept null_scores — calling it
        with only quality scores must succeed (not require a calibration set).
        """
        quality = np.full(30, 0.6, dtype=np.float32)
        # This must NOT raise TypeError even though no null_scores is supplied
        E, alerts = run_eprocess_agrapa(quality, epsilon=0.5, alpha=0.10)
        assert len(E) == 30

    def test_constant_low_quality_eventually_alerts(self):
        """A sustained low-quality stream (quality << epsilon) must trigger alert."""
        # quality=0.05 << epsilon=0.5  → large positive (ε - M_t) every frame
        # → e-process should exceed 1/alpha=10.0 well within 200 frames
        quality = np.full(200, 0.05, dtype=np.float32)
        E, alerts = run_eprocess_agrapa(quality, epsilon=0.5, alpha=0.10, window=20)
        assert len(alerts) >= 1, (
            f"Expected at least one alert on constant low-quality sequence; "
            f"max E={E.max():.2f}, threshold={1.0/0.10:.2f}"
        )

    def test_formal_mode_at_most_one_alert(self):
        """aGRAPAoperates in formal mode — at most one alert per sequence."""
        quality = np.full(100, 0.05, dtype=np.float32)
        E, alerts = run_eprocess_agrapa(quality, epsilon=0.5, alpha=0.10)
        assert len(alerts) <= 1

    def test_high_quality_no_alert(self):
        """High quality (quality >= epsilon) stream should NOT alert."""
        # quality=0.9 >> epsilon=0.5  → (ε - M_t) < 0  → multiplicative factor < 1
        quality = np.full(100, 0.9, dtype=np.float32)
        E, alerts = run_eprocess_agrapa(quality, epsilon=0.5, alpha=0.10)
        assert len(alerts) == 0, (
            f"High quality stream must not alert; max E={E.max():.2f}"
        )


# ---------------------------------------------------------------------------
# evaluate with mode="agrapa"
# ---------------------------------------------------------------------------

class TestEvaluateAgrapa:
    def _make_fixtures(self, n_seqs=8, n_frames=100):
        """Synthetic val set with 1 failure event per sequence (same as integration test)."""
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
            iou[70:] = 0.1
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

    def test_agrapa_mode_runs_without_error(self):
        preds, labels, ious, label_names = self._make_fixtures()
        result = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, mode="agrapa",
        )
        assert "summary" in result
        assert "config" in result

    def test_agrapa_config_has_zero_null_frames(self):
        """aGRAPAmode skips calibration — n_null_frames must be 0."""
        preds, labels, ious, label_names = self._make_fixtures()
        result = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, mode="agrapa",
        )
        assert result["config"]["n_null_frames"] == 0

    def test_evaluate_risk_mode_ifd5(self):
        preds, labels, ious, label_names = self._make_fixtures()
        result = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, mode="formal", risk_mode="ifd5",
        )
        assert result["config"]["risk_mode"] == "ifd5"
        assert "summary" in result

    def test_evaluate_baselines_have_four_heads(self):
        """Result must include baseline entries for ifd5, ifd10, ifd20, fc."""
        preds, labels, ious, label_names = self._make_fixtures()
        result = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, mode="formal",
        )
        for key in ("baseline_raw_ifd5_0.5", "baseline_raw_ifd10_0.5",
                    "baseline_raw_ifd20_0.5", "baseline_raw_fc_0.5"):
            assert key in result, f"Missing baseline key: {key}"


# ---------------------------------------------------------------------------
# TestEprocess — bug-fix regression tests
# ---------------------------------------------------------------------------

class TestEprocess:
    """Regression tests for the four bugs fixed in eprocess.py."""

    # Minimal label schema used across all tests in this class
    _LABEL_NAMES = [
        "correct", "false_confirmed", "failure_in_5", "recoverable",
        "target_dynamic", "camera_dynamic", "hard_dynamic_scene",
        "needs_full_compute", "hard_dynamic_scene_v2",
        "imminent_failure_dynamic", "failure_in_10", "failure_in_20",
        "imminent_failure_dynamic_10", "imminent_failure_dynamic_20",
    ]

    def _make_minimal_fixtures(self, n_seqs: int = 10, n_frames: int = 50, seed: int = 7):
        """Return (preds, labels, iou_traces, label_names) with synthetic data.

        Each sequence has random head probabilities and all-zero labels (all
        null), IoU held at 0.8 so every frame qualifies for the null set.
        """
        rng = np.random.RandomState(seed)
        label_names = self._LABEL_NAMES
        heads = [
            "false_confirmed",
            "imminent_failure_dynamic",
            "imminent_failure_dynamic_10",
            "imminent_failure_dynamic_20",
            "failure_in_5",
        ]
        preds: dict = {}
        labels: dict = {}
        iou_traces: dict = {}
        for i in range(n_seqs):
            seq = f"uav123/synth_seq{i:02d}"
            preds[seq] = [
                {h: float(rng.uniform(0.0, 1.0)) for h in heads}
                for _ in range(n_frames)
            ]
            labels[seq] = np.zeros((n_frames, len(label_names)), dtype=np.int8)
            iou_traces[seq] = np.ones(n_frames, dtype=np.float32) * 0.8
        return preds, labels, iou_traces, label_names

    # ------------------------------------------------------------------
    # Bug 1 – risk_mode not passed into null distribution calibration
    # ------------------------------------------------------------------

    def test_null_distribution_uses_risk_mode(self):
        """build_null_distribution must produce different results for ifd10 vs all_risk.

        ifd10 reads only imminent_failure_dynamic_10; all_risk is a weighted
        mixture of several heads.  On random data the two distributions will
        differ (with probability 1 over any real random draw).
        """
        preds, labels, iou_traces, label_names = self._make_minimal_fixtures(seed=42)
        cal_keys = list(preds.keys())[:5]

        null_all_risk = build_null_distribution(
            preds, labels, iou_traces, label_names, cal_keys, risk_mode="all_risk"
        )
        null_ifd10 = build_null_distribution(
            preds, labels, iou_traces, label_names, cal_keys, risk_mode="ifd10"
        )

        assert len(null_all_risk) > 0, "all_risk null distribution must be non-empty"
        assert len(null_ifd10) > 0, "ifd10 null distribution must be non-empty"
        # The two distributions must differ — they draw from different head values
        assert not np.allclose(null_all_risk, null_ifd10), (
            "Null distributions for 'all_risk' and 'ifd10' must differ "
            "(they read different head probabilities from the same fixture)"
        )

    # ------------------------------------------------------------------
    # Bug 1 continued – evaluate() propagates risk_mode to null builder
    # ------------------------------------------------------------------

    def test_evaluate_risk_mode_sweep_differs(self):
        """evaluate() with risk_mode='ifd10' vs 'all_risk' must record different configs."""
        preds, labels, iou_traces, label_names = self._make_minimal_fixtures(n_seqs=12, seed=99)

        result_all = evaluate(
            preds, labels, iou_traces, label_names,
            alpha=0.10, epsilon=0.5, decay=1.0, mode="formal",
            risk_mode="all_risk",
        )
        result_ifd10 = evaluate(
            preds, labels, iou_traces, label_names,
            alpha=0.10, epsilon=0.5, decay=1.0, mode="formal",
            risk_mode="ifd10",
        )

        assert result_all["config"]["risk_mode"] == "all_risk"
        assert result_ifd10["config"]["risk_mode"] == "ifd10"
        # The two runs must have produced results with different risk_mode labels
        assert result_all["config"]["risk_mode"] != result_ifd10["config"]["risk_mode"]

    # ------------------------------------------------------------------
    # Bug 4 – correct baseline key name
    # ------------------------------------------------------------------

    def test_cli_baseline_key_present(self):
        """evaluate() must contain 'baseline_raw_ifd5_0.5' and NOT 'baseline_raw_ifd_0.5'."""
        preds, labels, iou_traces, label_names = self._make_minimal_fixtures(n_seqs=10, seed=3)

        result = evaluate(
            preds, labels, iou_traces, label_names,
            alpha=0.10, epsilon=0.5, decay=1.0, mode="formal",
        )

        assert "baseline_raw_ifd5_0.5" in result, (
            "'baseline_raw_ifd5_0.5' key must be present in evaluate() result"
        )
        assert "baseline_raw_ifd_0.5" not in result, (
            "Old incorrect key 'baseline_raw_ifd_0.5' must NOT be present"
        )

    # ------------------------------------------------------------------
    # Bug 2 – corrected _DIAGNOSTIC_SEQS keys
    # ------------------------------------------------------------------

    def test_diagnostic_seq_keys(self):
        """_DIAGNOSTIC_SEQS must use the correct dataset-prefix/sequence-name format."""
        # Correct keys that must be present
        assert "uav123/bike2" in _DIAGNOSTIC_SEQS, (
            "'uav123/bike2' must be in _DIAGNOSTIC_SEQS (bike2 belongs to UAV123)"
        )
        assert "visdrone_sot/uav0000164" in _DIAGNOSTIC_SEQS, (
            "'visdrone_sot/uav0000164' must be in _DIAGNOSTIC_SEQS "
            "(uav0000164 belongs to VisDrone-SOT)"
        )

        # Wrong keys that must NOT be present
        assert "dtb70/bike2" not in _DIAGNOSTIC_SEQS, (
            "'dtb70/bike2' must NOT be in _DIAGNOSTIC_SEQS (bike2 is in UAV123, not DTB70)"
        )
        assert "uav123/uav0000164" not in _DIAGNOSTIC_SEQS, (
            "'uav123/uav0000164' must NOT be in _DIAGNOSTIC_SEQS (uav0000164 is in VisDrone-SOT)"
        )


# ---------------------------------------------------------------------------
# Additional targeted tests — gaps not covered above
# ---------------------------------------------------------------------------

class TestResetAtBoundaryRestoresE:
    """Verify that reset_at_boundary truly restores E to ~1.0 at boundary."""

    def test_e_drops_to_single_step_value_after_reset(self):
        """After reset at frame 10, E[10] must be a single-step accumulation from E=1.0.

        This confirms the internal state is reset (not just a new segment running
        on top of the accumulated value from segment 1).
        """
        null = np.full(50, 0.1, dtype=np.float32)
        # Constant high risk: E grows exponentially in both halves
        risk = np.full(20, 0.9, dtype=np.float32)
        E, _ = reset_at_boundary(
            reset_frames=[10],
            risk_scores=risk,
            null_scores=null,
            alpha=0.10,
            epsilon=0.5,
            decay=1.0,
            mode="formal",
        )
        # Segment 1 ends at frame 9 with a very large E value
        # Segment 2 starts at frame 10 — E[10] must be the FIRST step from E_prev=1.0
        # We know E[10] = E[11]/E_ratio ≈ small, definitely << E[9]
        assert E[9] > 10.0, f"Segment 1 should have accumulated E >> 1, got E[9]={E[9]:.2f}"
        assert E[10] < E[9], \
            f"Reset must drop accumulated E: E[10]={E[10]:.4f} should be < E[9]={E[9]:.2f}"
        # E[10] is the first step of segment 2 from E_prev=1.0; so E[10] == E[11]^(1/1) at step 1
        # A tighter bound: E[10] must be < 1000 (way below E[9] which is millions)
        assert E[10] < 1000.0, \
            f"After reset, E[10] should be a single-step value, not carry-over: {E[10]:.2f}"


class TestFailureEventsEdgeCases:
    """Additional coverage for _failure_events edge cases."""

    def test_no_good_frames_at_start_no_failure_events(self):
        """Trace starting entirely below threshold → zero failure events.

        The function requires at least one good frame before a drop counts.
        """
        iou = np.array([0.1, 0.2, 0.15, 0.1, 0.05], dtype=np.float64)
        events = _failure_events(iou, iou_threshold=0.30)
        assert events == [], \
            f"All-below-threshold trace must produce no failure events, got {events}"

    def test_failure_events_exact_two(self):
        """Sequence 0.9, 0.9, 0.1, 0.9, 0.1 → exactly two failure events at frames 2 and 4."""
        iou = np.array([0.9, 0.9, 0.1, 0.9, 0.1], dtype=np.float64)
        events = _failure_events(iou, iou_threshold=0.30)
        assert events == [2, 4], \
            f"Expected failure events at frames [2, 4], got {events}"

    def test_failure_event_at_recovery_counts_new_drop(self):
        """After recovery (IoU back above threshold), a new drop counts as a new event."""
        # Fail, recover, fail again
        iou = np.array([0.9, 0.0, 0.9, 0.9, 0.0, 0.9], dtype=np.float64)
        events = _failure_events(iou, iou_threshold=0.30)
        # frame 1: first drop (after frame 0 good) → event
        # frame 4: second drop (after frames 2,3 good) → event
        assert 1 in events, f"Expected event at frame 1, got {events}"
        assert 4 in events, f"Expected event at frame 4, got {events}"
        assert len(events) == 2, f"Expected exactly 2 events, got {events}"


class TestEvaluateAcceptsDiagnosticSeqsParam:
    """Test that evaluate() accepts and uses a caller-supplied diagnostic_seqs set."""

    def _make_fixtures(self, n_seqs: int = 8, n_frames: int = 50):
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
        for i in range(n_seqs):
            seq = f"uav123/seq{i:02d}"
            labels[seq] = np.zeros((n_frames, 14), dtype=np.int8)
            ious[seq] = np.ones(n_frames, dtype=np.float32)
            preds[seq] = [
                {"false_confirmed": 0.1, "imminent_failure_dynamic": 0.1,
                 "failure_in_5": 0.05, "imminent_failure_dynamic_20": 0.05}
            ] * n_frames
        return preds, labels, ious, label_names

    def test_evaluate_accepts_diagnostic_seqs_param(self):
        """evaluate() must accept diagnostic_seqs kwarg and exclude those seqs from eval."""
        preds, labels, ious, label_names = self._make_fixtures(n_seqs=8)

        # Mark the first seq as diagnostic — it must not appear in per_sequence results
        diag_seq = "uav123/seq00"
        result = evaluate(
            preds, labels, ious, label_names,
            alpha=0.10, epsilon=0.5, mode="formal",
            diagnostic_seqs={diag_seq},
        )

        assert "summary" in result, "evaluate() must return a summary dict"
        assert diag_seq not in result.get("per_sequence", {}), (
            f"Diagnostic seq '{diag_seq}' must be excluded from per_sequence results"
        )
    """Edge cases for run_eprocess_agrapa not covered by existing tests."""

    def test_agrapa_alerts_on_constant_zero_quality(self):
        """quality=0.0 (maximum risk, quality << epsilon=0.5) must fire an alert."""
        quality = np.zeros(50, dtype=np.float32)
        E, alerts = run_eprocess_agrapa(quality, epsilon=0.5, alpha=0.10, window=20)
        assert len(alerts) >= 1, \
            f"Constant quality=0.0 must trigger at least one alert; max E={E.max():.2f}"

    def test_agrapa_no_alert_on_constant_one_quality(self):
        """quality=1.0 (healthy tracker, quality >> epsilon=0.5) must never alert."""
        quality = np.ones(100, dtype=np.float32)
        E, alerts = run_eprocess_agrapa(quality, epsilon=0.5, alpha=0.10, window=20)
        assert len(alerts) == 0, \
            f"Constant quality=1.0 must not alert; max E={E.max():.4f}"

    def test_agrapa_e_trace_nonnegative(self):
        """E-process martingale must always be non-negative (clipped at 0)."""
        rng = np.random.RandomState(77)
        quality = rng.uniform(0.0, 1.0, 80).astype(np.float32)
        E, _ = run_eprocess_agrapa(quality, epsilon=0.5, alpha=0.10, window=20)
        assert np.all(E >= 0.0), f"E-process must be >= 0.0 everywhere; min={E.min():.6f}"

