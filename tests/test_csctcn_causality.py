"""Causality unit tests for CSCTCN.

Proves that perturbing x[:, t+1:, :] does NOT change out[:, :t+1, :]
for all tested time positions t.  If any future frame leaks into a
past output, the test fails.

Run with:
    pytest tests/test_csctcn_causality.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make sure the project root is on sys.path so csc_lib can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csc_lib.csc.config import CSCModelConfig, TCNConfig
from csc_lib.csc.model import CSCTCN, _CausalConv1d


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_tcn(
    window_size: int = 16,
    hidden_dim: int = 32,
    feature_dim: int = 11,
    dilations: list[int] | None = None,
    kernel_size: int = 3,
) -> CSCTCN:
    if dilations is None:
        dilations = [1, 2, 4, 8]
    tcn_cfg = TCNConfig(
        kernel_size=kernel_size,
        num_layers=len(dilations),
        dilations=dilations,
        hidden_dim=hidden_dim,
        dropout=0.0,  # deterministic
    )
    cfg = CSCModelConfig(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=len(dilations),
        dropout=0.0,
        kind="tcn",
        tcn=tcn_cfg,
    )
    model = CSCTCN(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CausalConv1d — basic left-pad property
# ---------------------------------------------------------------------------


class TestCausalConv1d:
    def test_output_length_preserved(self) -> None:
        """Output length must equal input length (strict causal padding)."""
        conv = _CausalConv1d(in_channels=8, out_channels=8, kernel_size=3, dilation=2)
        x = torch.randn(2, 8, 20)
        y = conv(x)
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    def test_no_future_leakage_single_conv(self) -> None:
        """A single causal conv at position t must not use x[:, :, t+1:]."""
        for dilation in [1, 2, 4]:
            conv = _CausalConv1d(in_channels=4, out_channels=4, kernel_size=3, dilation=dilation)
            conv.eval()
            T = 20
            x_ref = torch.randn(1, 4, T)
            x_perturbed = x_ref.clone()
            t_probe = 8
            # Perturb all frames strictly after t_probe
            x_perturbed[:, :, t_probe + 1:] = torch.randn_like(x_perturbed[:, :, t_probe + 1:])
            with torch.no_grad():
                y_ref = conv(x_ref)
                y_perturbed = conv(x_perturbed)
            # Output at positions [0..t_probe] must be identical
            assert torch.allclose(
                y_ref[:, :, : t_probe + 1],
                y_perturbed[:, :, : t_probe + 1],
                atol=1e-6,
            ), (
                f"CausalConv1d leaks future frames! "
                f"dilation={dilation}, t_probe={t_probe}"
            )


# ---------------------------------------------------------------------------
# CSCTCN — end-to-end causality
# ---------------------------------------------------------------------------


class TestCSCTCNCausality:
    """Core causality guarantee: output at frame t depends only on frames 0..t."""

    @pytest.mark.parametrize("t_probe", [0, 4, 8, 15])
    def test_no_future_leakage_localization(self, t_probe: int) -> None:
        """Perturbing x[:, t+1:, :] must not change localization_logits[:, :t+1, :]."""
        model = _make_tcn(window_size=16, dilations=[1, 2, 4, 8])
        T = 16
        if t_probe >= T:
            pytest.skip(f"t_probe={t_probe} >= T={T}")

        x_ref = torch.randn(2, T, 11)
        x_perturbed = x_ref.clone()
        # Corrupt all frames strictly AFTER t_probe
        if t_probe + 1 < T:
            x_perturbed[:, t_probe + 1:, :] = torch.randn_like(
                x_perturbed[:, t_probe + 1:, :]
            ) * 10.0

        with torch.no_grad():
            out_ref = model(x_ref)
            out_perturbed = model(x_perturbed)

        max_diff = (
            out_ref.localization_logits[:, : t_probe + 1, :]
            - out_perturbed.localization_logits[:, : t_probe + 1, :]
        ).abs().max().item()
        assert torch.allclose(
            out_ref.localization_logits[:, : t_probe + 1, :],
            out_perturbed.localization_logits[:, : t_probe + 1, :],
            atol=1e-5,
        ), (
            f"CSCTCN localization_logits leaks future frames! "
            f"t_probe={t_probe}, max_diff={max_diff:.3e}"
        )

    @pytest.mark.parametrize("t_probe", [0, 4, 8, 15])
    def test_no_future_leakage_confidence(self, t_probe: int) -> None:
        """Perturbing x[:, t+1:, :] must not change confidence_logits[:, :t+1, :]."""
        model = _make_tcn(window_size=16, dilations=[1, 2, 4, 8])
        T = 16
        if t_probe >= T:
            pytest.skip(f"t_probe={t_probe} >= T={T}")

        x_ref = torch.randn(2, T, 11)
        x_perturbed = x_ref.clone()
        if t_probe + 1 < T:
            x_perturbed[:, t_probe + 1:, :] = torch.randn_like(
                x_perturbed[:, t_probe + 1:, :]
            ) * 10.0

        with torch.no_grad():
            out_ref = model(x_ref)
            out_perturbed = model(x_perturbed)

        assert torch.allclose(
            out_ref.confidence_logits[:, : t_probe + 1, :],
            out_perturbed.confidence_logits[:, : t_probe + 1, :],
            atol=1e-5,
        ), (
            f"CSCTCN confidence_logits leaks future frames! t_probe={t_probe}"
        )

    def test_no_future_leakage_all_frames(self) -> None:
        """Exhaustive: for every t in [0, T-1], check output[:, :t+1, :] is causal."""
        model = _make_tcn(window_size=16, dilations=[1, 2, 4, 8])
        T = 16
        x_ref = torch.randn(1, T, 11)

        with torch.no_grad():
            out_ref = model(x_ref)

        for t in range(T - 1):
            x_p = x_ref.clone()
            x_p[:, t + 1:, :] = torch.randn_like(x_p[:, t + 1:, :]) * 5.0
            with torch.no_grad():
                out_p = model(x_p)
            diff = (
                out_ref.localization_logits[:, : t + 1, :]
                - out_p.localization_logits[:, : t + 1, :]
            ).abs().max().item()
            assert diff < 1e-5, (
                f"Future-frame leakage detected at t={t}: max_diff={diff:.3e}"
            )

    def test_different_architectures_causal(self) -> None:
        """TCN-32 with 5-layer dilations [1,2,4,8,16] must also be causal."""
        model = _make_tcn(
            window_size=32,
            hidden_dim=32,
            dilations=[1, 2, 4, 8, 16],
        )
        T = 32
        x_ref = torch.randn(1, T, 11)
        x_p = x_ref.clone()
        t_probe = 15
        x_p[:, t_probe + 1:, :] = torch.randn_like(x_p[:, t_probe + 1:, :]) * 10.0

        with torch.no_grad():
            out_ref = model(x_ref)
            out_p = model(x_p)

        diff = (
            out_ref.localization_logits[:, : t_probe + 1, :]
            - out_p.localization_logits[:, : t_probe + 1, :]
        ).abs().max().item()
        assert diff < 1e-5, (
            f"TCN-32 has future leakage at t={t_probe}: max_diff={diff:.3e}"
        )


# ---------------------------------------------------------------------------
# CSCMLP — should also be causal (per-frame, no temporal dependency)
# ---------------------------------------------------------------------------


class TestCSCMLPCausality:
    """MLP has no temporal context so is trivially causal; just sanity-check."""

    def test_mlp_causal(self) -> None:
        from csc_lib.csc.model import CSCMLP
        cfg = CSCModelConfig(
            feature_dim=11, hidden_dim=32, num_layers=2, dropout=0.0, kind="mlp"
        )
        model = CSCMLP(cfg)
        model.eval()
        T = 16
        x_ref = torch.randn(1, T, 11)
        x_p = x_ref.clone()
        x_p[:, 8:, :] = torch.randn_like(x_p[:, 8:, :]) * 10.0
        with torch.no_grad():
            out_ref = model(x_ref)
            out_p = model(x_p)
        diff = (
            out_ref.localization_logits[:, :8, :]
            - out_p.localization_logits[:, :8, :]
        ).abs().max().item()
        assert diff < 1e-6, f"MLP unexpectedly depends on future frames: diff={diff:.3e}"
