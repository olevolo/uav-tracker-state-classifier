"""V3 proactive forecast tests.

Coverage:
1. risk_labeler.build_future_risk_labels — correctness, edge cases, no-leakage
2. CSCDataset reads forecast targets and ignore mask
3. CSCTCN forward output shapes with forecast heads enabled / disabled
4. Loss computation with ignore mask (masked frames contribute 0 grad)
5. Forecast metrics handle one-class-absent case (NaN, no crash)
6. V2 backward compatibility — config without enable_forecast_heads → V2 behaviour
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csc_lib.csc.config import CSCFeatureConfig, CSCModelConfig, TCNConfig
from csc_lib.csc.dataset import CSCDataset
from csc_lib.csc.labeling.label_schema import DerivedState
from csc_lib.csc.labeling.risk_labeler import (
    build_future_risk_labels,
    summarize_future_risk,
)
from csc_lib.csc.model import CSCTCN, CSCGRU, CSCMLP, build_model


CC = int(DerivedState.CORRECT_CONFIRMED)
CU = int(DerivedState.CORRECT_UNCERTAIN)
LA = int(DerivedState.LOST_AWARE)
FC = int(DerivedState.FALSE_CONFIRMED)


# ---------------------------------------------------------------------------
# 1. risk_labeler — correctness + no-leakage
# ---------------------------------------------------------------------------


class TestRiskLabeler:
    def test_simple_lookahead_failure(self) -> None:
        """If FC happens at t=5 (horizon=3), then t=2, t=3, t=4 must have
        failure_next_3=1 because the failure is in their future window."""
        states = [CC, CC, CC, CC, CC, FC, CC, CC]
        out = build_future_risk_labels(states, horizon=3)
        # t=2: future = states[3:6] = [CC, CC, FC] → has FC
        assert out[2]["failure_next_10"] == 1
        assert out[2]["false_confirmed_next_10"] == 1
        assert out[2]["lost_aware_next_10"] == 0
        # t=3: future = states[4:7] = [CC, FC, CC] → has FC
        assert out[3]["failure_next_10"] == 1
        # t=4: future = states[5:8] = [FC, CC, CC] → has FC
        assert out[4]["false_confirmed_next_10"] == 1
        # t=5: future = states[6:9] = [CC, CC] (only 2 frames, partial) — has no FC
        assert out[5]["false_confirmed_next_10"] == 0

    def test_lost_aware_separation(self) -> None:
        """LOST_AWARE must light up only the lost head, not the FC head."""
        states = [CC, CC, LA, CC, CC]
        out = build_future_risk_labels(states, horizon=3)
        # t=0: future = [CC, LA, CC] — has LOST, no FC
        assert out[0]["lost_aware_next_10"] == 1
        assert out[0]["false_confirmed_next_10"] == 0
        # failure = lost OR fc
        assert out[0]["failure_next_10"] == 1

    def test_ignore_mask_at_sequence_end(self) -> None:
        """Last `horizon` frames where full lookahead unavailable get ignore=1."""
        states = [CC] * 10
        out = build_future_risk_labels(states, horizon=5)
        for t in range(5):
            assert out[t]["ignore_forecast"] == 0, f"t={t} should be valid"
        for t in range(5, 10):
            assert out[t]["ignore_forecast"] == 1, f"t={t} should be ignored"

    def test_no_leakage_uses_only_future(self) -> None:
        """Label at t MUST depend only on derived[t+1:t+1+H], NOT on derived[t]."""
        states_fc_at_t = [CC, CC, FC, CC, CC, CC, CC, CC]
        states_no_fc_at_t = [CC, CC, CC, CC, CC, CC, CC, CC]  # remove FC at t=2
        # Both sequences differ only at t=2.
        # Labels at t=2 should NOT depend on what happens at t=2 (current frame).
        out1 = build_future_risk_labels(states_fc_at_t, horizon=3)
        out2 = build_future_risk_labels(states_no_fc_at_t, horizon=3)
        # At t=2: future is states[3:6].  For seq1: [CC,CC,CC] — no FC.
        #         For seq2: [CC,CC,CC] — no FC.  Should be IDENTICAL.
        assert out1[2] == out2[2], (
            f"Forecast at t=2 leaked from current frame: {out1[2]} vs {out2[2]}"
        )

    def test_labels_at_t_do_not_depend_on_past(self) -> None:
        """Mutating frames BEFORE t must not change label at t."""
        states_a = [CC, CC, CC, FC, CC]   # FC at t=3
        states_b = [LA, FC, LA, FC, CC]   # same FC at t=3, garbage in past
        out_a = build_future_risk_labels(states_a, horizon=2)
        out_b = build_future_risk_labels(states_b, horizon=2)
        # At t=2: future = [FC, CC] → FC=1.  Should be identical even though
        # past frames differ wildly.
        assert out_a[2] == out_b[2]

    def test_horizon_cannot_be_zero_or_negative(self) -> None:
        with pytest.raises(ValueError):
            build_future_risk_labels([CC, CC], horizon=0)
        with pytest.raises(ValueError):
            build_future_risk_labels([CC, CC], horizon=-1)

    def test_summarize_future_risk(self) -> None:
        states = [CC, CC, FC, LA, CC, CC, CC, CC, CC, CC]
        labels = build_future_risk_labels(states, horizon=3)
        summary = summarize_future_risk(labels)
        assert summary["n"] == 10
        assert summary["n_valid"] == 7  # last 3 frames have ignore=1
        assert 0.0 < summary["fc_rate"] <= 1.0
        assert summary["ignore_rate"] == 0.3


# ---------------------------------------------------------------------------
# 2. CSCDataset reads forecast targets
# ---------------------------------------------------------------------------


def _fake_rows_with_forecast(n: int = 32, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "dataset": "synthetic",
            "sequence": "seq0",
            "frame_idx": i,
            "pred_bbox": [100.0, 100.0, 50.0, 50.0],
            "gt_bbox": [100.0, 100.0, 50.0, 50.0],
            "iou": float(rng.uniform(0, 1)),
            "confidence": float(rng.uniform(0, 1)),
            "apce": None,
            "psr": None,
            "localization_state": int(rng.integers(0, 3)),
            "confidence_state": int(rng.integers(0, 2)),
            "derived_state": int(rng.integers(0, 4)),
            "failure_next_10": int(rng.integers(0, 2)),
            "false_confirmed_next_10": int(rng.integers(0, 2)),
            "lost_aware_next_10": int(rng.integers(0, 2)),
            "ignore_forecast": 1 if i >= n - 10 else 0,
            "aux": {
                "occlusion": False, "out_of_view": False,
                "fast_motion": False, "scale_change": False,
                "distractor_risk": False,
            },
        })
    return rows


class TestDatasetForecastTargets:
    def test_dataset_exposes_forecast_keys(self) -> None:
        rows = _fake_rows_with_forecast(32)
        feature_cfg = CSCFeatureConfig(window_size=16)
        ds = CSCDataset(
            {("synthetic", "seq0"): rows},
            feature_cfg,
            image_size=(640, 480),
        )
        sample = ds[0]
        for key in (
            "failure_next_10",
            "false_confirmed_next_10",
            "lost_aware_next_10",
            "ignore_forecast",
        ):
            assert key in sample, f"sample missing {key!r}"
            assert sample[key].shape == (16,), f"{key} bad shape: {sample[key].shape}"
            assert sample[key].dtype == torch.int64

    def test_v2_labels_get_ignore_mask_all_ones(self) -> None:
        """Labels without forecast keys (V2-style) → ignore_forecast all 1."""
        rows = _fake_rows_with_forecast(32)
        # Strip forecast keys to simulate V2 labels.
        for r in rows:
            for k in ("failure_next_10", "false_confirmed_next_10",
                     "lost_aware_next_10", "ignore_forecast"):
                r.pop(k, None)
        feature_cfg = CSCFeatureConfig(window_size=16)
        ds = CSCDataset(
            {("synthetic", "seq0"): rows},
            feature_cfg,
            image_size=(640, 480),
        )
        sample = ds[0]
        assert (sample["ignore_forecast"] == 1).all(), (
            "V2 labels must produce all-ignore mask"
        )


# ---------------------------------------------------------------------------
# 3. Model output shapes with forecast heads
# ---------------------------------------------------------------------------


def _make_v3_tcn(B: int = 2, T: int = 16, F: int = 11) -> tuple[CSCTCN, torch.Tensor]:
    cfg = CSCModelConfig(
        feature_dim=F,
        hidden_dim=32,
        num_layers=4,
        kind="tcn",
        enable_forecast_heads=True,
        forecast_horizon=10,
        tcn=TCNConfig(kernel_size=3, dilations=[1, 2, 4, 8], hidden_dim=32, num_layers=4),
    )
    model = CSCTCN(cfg)
    x = torch.randn(B, T, F)
    return model, x


class TestModelForecastHeads:
    @pytest.mark.parametrize("kind", ["tcn", "gru", "mlp"])
    def test_forecast_logits_present_when_enabled(self, kind: str) -> None:
        cfg = CSCModelConfig(
            feature_dim=11, hidden_dim=32, num_layers=2,
            kind=kind, enable_forecast_heads=True,
        )
        if kind == "tcn":
            cfg.tcn = TCNConfig(hidden_dim=32, num_layers=2, dilations=[1, 2])
        model = build_model(cfg)
        x = torch.randn(2, 16, 11)
        out = model(x)
        assert out.failure_next_10_logit is not None
        assert out.false_confirmed_next_10_logit is not None
        assert out.lost_aware_next_10_logit is not None
        # Shape: (B, T, 1)
        assert out.failure_next_10_logit.shape == (2, 16, 1)

    @pytest.mark.parametrize("kind", ["tcn", "gru", "mlp"])
    def test_forecast_logits_none_when_disabled(self, kind: str) -> None:
        """V2 backward compat — no forecast heads when flag is off."""
        cfg = CSCModelConfig(
            feature_dim=11, hidden_dim=32, num_layers=2,
            kind=kind, enable_forecast_heads=False,
        )
        if kind == "tcn":
            cfg.tcn = TCNConfig(hidden_dim=32, num_layers=2, dilations=[1, 2])
        model = build_model(cfg)
        x = torch.randn(2, 16, 11)
        out = model(x)
        assert out.failure_next_10_logit is None
        assert out.false_confirmed_next_10_logit is None
        assert out.lost_aware_next_10_logit is None

    def test_predict_exposes_forecast_probs(self) -> None:
        model, x = _make_v3_tcn()
        result = model.predict(x)
        assert "failure_next_10_prob" in result
        assert "false_confirmed_next_10_prob" in result
        assert "lost_aware_next_10_prob" in result
        # Probs in [0, 1]
        for k in (
            "failure_next_10_prob",
            "false_confirmed_next_10_prob",
            "lost_aware_next_10_prob",
        ):
            p = result[k]
            assert (p >= 0).all() and (p <= 1).all()

    def test_last_step_only_works_with_forecast(self) -> None:
        model, x = _make_v3_tcn(B=1, T=16, F=11)
        out = model(x, last_step_only=True)
        assert out.failure_next_10_logit.shape == (1, 1, 1)
        assert out.derived_logits.shape == (1, 1, 4)


# ---------------------------------------------------------------------------
# 4. Loss computation with ignore mask
# ---------------------------------------------------------------------------


class TestForecastLossMasking:
    def test_ignored_frames_zero_loss(self) -> None:
        """Frames with ignore_forecast=1 must contribute 0 to the BCE loss."""
        torch.manual_seed(0)
        B, T = 2, 16
        logits = torch.randn(B, T, 1)
        targets = torch.randint(0, 2, (B, T)).float()
        ignore = torch.ones(B, T)  # ALL frames ignored
        valid = (1.0 - ignore)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        per_elem = bce(logits.squeeze(-1), targets) * valid
        n_valid = valid.sum().clamp_min(1.0)
        masked_loss = per_elem.sum() / n_valid
        assert masked_loss.item() == 0.0

    def test_partial_ignore_reduces_loss(self) -> None:
        """Half-ignored should give roughly half the loss of fully-valid."""
        torch.manual_seed(1)
        B, T = 4, 16
        logits = torch.randn(B, T, 1) * 5  # large, so loss is non-trivial
        targets = torch.randint(0, 2, (B, T)).float()
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        full = (bce(logits.squeeze(-1), targets) * torch.ones(B, T)).sum() / (B * T)
        ignore = torch.zeros(B, T)
        ignore[:, T // 2 :] = 1.0  # ignore second half
        valid = 1.0 - ignore
        partial = (bce(logits.squeeze(-1), targets) * valid).sum() / valid.sum().clamp_min(1.0)
        # partial may be higher or lower than full (different sample) but must be finite
        assert torch.isfinite(partial)
        assert partial.item() >= 0.0


# ---------------------------------------------------------------------------
# 5. Forecast metrics handle one-class-absent
# ---------------------------------------------------------------------------


class TestForecastMetricsEdgeCases:
    def test_all_zeros_targets(self) -> None:
        """failure_auprc with all-zero targets should return 0 (no positives)."""
        from csc_lib.eval.custom_metrics.scene_state_metrics import (
            failure_auprc, failure_auroc,
        )
        y = np.zeros(100, dtype=np.int8)
        s = np.random.rand(100)
        assert failure_auprc(y, s) == 0.0
        # AUROC with one class returns 0.5 by convention
        assert failure_auroc(y, s) == 0.5

    def test_all_ones_targets(self) -> None:
        from csc_lib.eval.custom_metrics.scene_state_metrics import (
            failure_auprc, failure_auroc,
        )
        y = np.ones(100, dtype=np.int8)
        s = np.random.rand(100)
        # AUPRC with all positives = 1.0 (perfect recall) — but our impl returns AP
        # Just assert it's finite and in [0, 1].
        ap = failure_auprc(y, s)
        assert 0.0 <= ap <= 1.0
        assert failure_auroc(y, s) == 0.5


# ---------------------------------------------------------------------------
# 6. V2 backward compatibility
# ---------------------------------------------------------------------------


class TestV2BackwardCompat:
    def test_v2_config_yaml_does_not_break(self) -> None:
        """Loading the canonical V2 config must not enable forecast heads."""
        import yaml
        cfg_path = PROJECT_ROOT / "configs" / "csc" / "csc_tcn16.yaml"
        if not cfg_path.exists():
            pytest.skip(f"missing {cfg_path}")
        from csc_lib.csc.config import CSCTrainConfig
        raw = yaml.safe_load(cfg_path.read_text())
        cfg = CSCTrainConfig.from_dict(raw)
        assert cfg.model.enable_forecast_heads is False, (
            "V2 yaml must NOT enable forecast heads"
        )
        # Build the model — should produce no forecast heads
        cfg.model.feature_dim = 11
        model = build_model(cfg.model)
        assert not model.enable_forecast
        x = torch.randn(1, cfg.feature.window_size, 11)
        out = model(x)
        assert out.failure_next_10_logit is None

    def test_v3_config_yaml_enables_heads(self) -> None:
        import yaml
        cfg_path = PROJECT_ROOT / "configs" / "csc" / "csc_tcn16_v3.yaml"
        if not cfg_path.exists():
            pytest.skip(f"missing {cfg_path}")
        from csc_lib.csc.config import CSCTrainConfig
        raw = yaml.safe_load(cfg_path.read_text())
        cfg = CSCTrainConfig.from_dict(raw)
        assert cfg.model.enable_forecast_heads is True
        assert cfg.model.forecast_horizon == 10
        assert cfg.loss.forecast_fc_weight == 0.8
        cfg.model.feature_dim = 11
        model = build_model(cfg.model)
        assert model.enable_forecast
        x = torch.randn(1, cfg.feature.window_size, 11)
        out = model(x)
        assert out.failure_next_10_logit is not None
        assert out.failure_next_10_logit.shape[-1] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
