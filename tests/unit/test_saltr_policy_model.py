"""Unit tests for saltr/src/salt_r/policy_model.py."""
from __future__ import annotations

import ast
import inspect
import tempfile
import os

import pytest
import torch

from salt_r.policy_model import SALTRDPolicyNet, compute_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model() -> SALTRDPolicyNet:
    return SALTRDPolicyNet(
        n_features=28,
        hidden_size=64,
        n_layers=2,
        window_size=20,
        dropout=0.1,
    )


def _windowed_input(batch: int = 2, seq_len: int = 20, n_feat: int = 28) -> torch.Tensor:
    return torch.randn(batch, seq_len, n_feat)


def _single_frame_input(batch: int = 2, n_feat: int = 28) -> torch.Tensor:
    return torch.randn(batch, n_feat)


# ---------------------------------------------------------------------------
# 1. Forward pass returns all expected keys
# ---------------------------------------------------------------------------

def test_forward_returns_expected_keys():
    model = _make_model()
    model.eval()
    out = model(_windowed_input())

    assert "risk_probs" in out, "Missing key: risk_probs"
    assert "action_logits" in out, "Missing key: action_logits"
    assert "candidate_score" in out, "Missing key: candidate_score"

    risk = out["risk_probs"]
    assert "false_confirmed" in risk
    assert "imminent_failure_dynamic" in risk
    assert "recoverable" in risk

    actions = out["action_logits"]
    assert "compute" in actions
    assert "recovery" in actions


# ---------------------------------------------------------------------------
# 2. Risk probs are in [0, 1]
# ---------------------------------------------------------------------------

def test_risk_probs_in_unit_interval():
    model = _make_model()
    model.eval()
    out = model(_windowed_input())

    for head_name, prob in out["risk_probs"].items():
        assert prob.min().item() >= 0.0, f"{head_name} prob below 0"
        assert prob.max().item() <= 1.0, f"{head_name} prob above 1"


# ---------------------------------------------------------------------------
# 3. Action logits have correct shapes
# ---------------------------------------------------------------------------

def test_action_logit_shapes():
    batch = 3
    model = _make_model()
    model.eval()
    out = model(_windowed_input(batch=batch))

    compute_logits = out["action_logits"]["compute"]
    assert compute_logits.shape == (batch, 3), (
        f"compute logits shape {compute_logits.shape} != ({batch}, 3)"
    )

    recovery_logits = out["action_logits"]["recovery"]
    assert recovery_logits.shape == (batch, 4), (
        f"recovery logits shape {recovery_logits.shape} != ({batch}, 4)"
    )


# ---------------------------------------------------------------------------
# 4. Handles single-frame input (batch, 28)
# ---------------------------------------------------------------------------

def test_single_frame_input():
    batch = 2
    model = _make_model()
    model.eval()
    x = _single_frame_input(batch=batch)  # (2, 28)
    out = model(x)

    # Risk probs should be (B,)
    for head_name, prob in out["risk_probs"].items():
        assert prob.shape == (batch,), (
            f"{head_name}: expected shape ({batch},), got {prob.shape}"
        )

    # Action logits should be (B, n_classes)
    assert out["action_logits"]["compute"].shape == (batch, 3)
    assert out["action_logits"]["recovery"].shape == (batch, 4)


# ---------------------------------------------------------------------------
# 5. Handles windowed input (batch, 20, 28)
# ---------------------------------------------------------------------------

def test_windowed_input():
    batch, seq_len = 4, 20
    model = _make_model()
    model.eval()
    x = _windowed_input(batch=batch, seq_len=seq_len)  # (4, 20, 28)
    out = model(x)

    for head_name, prob in out["risk_probs"].items():
        assert prob.shape == (batch,), (
            f"{head_name}: expected ({batch},), got {prob.shape}"
        )
    assert out["action_logits"]["compute"].shape == (batch, 3)
    assert out["action_logits"]["recovery"].shape == (batch, 4)


# ---------------------------------------------------------------------------
# 6. save/load round-trip preserves weights
# ---------------------------------------------------------------------------

def test_save_load_round_trip():
    model = _make_model()
    model.eval()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        model.save(tmp_path)
        loaded = SALTRDPolicyNet.load(tmp_path)
        loaded.eval()

        # Compare parameters
        for (name_a, p_a), (name_b, p_b) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert name_a == name_b
            assert torch.allclose(p_a, p_b), f"Parameter mismatch: {name_a}"

        # Functional equivalence
        x = _windowed_input(batch=2)
        with torch.no_grad():
            out_orig = model(x)
            out_loaded = loaded(x)

        for head in ("false_confirmed", "imminent_failure_dynamic", "recoverable"):
            assert torch.allclose(
                out_orig["risk_probs"][head], out_loaded["risk_probs"][head]
            ), f"Risk head mismatch after load: {head}"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 7. load() raises ValueError on wrong model_family
# ---------------------------------------------------------------------------

def test_load_raises_on_wrong_model_family():
    """Simulate a risk-only SALTRD checkpoint and verify load() rejects it."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        # Write a fake checkpoint with model_family = 'saltrd_risk_only'
        fake_ckpt = {
            "model_family": "saltrd_risk_only",
            "model_state_dict": {},
            "init_kwargs": {},
        }
        torch.save(fake_ckpt, tmp_path)

        with pytest.raises(ValueError, match="saltrd_policy"):
            SALTRDPolicyNet.load(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 8. No TSA imports in policy_model module
# ---------------------------------------------------------------------------

def test_no_tsa_imports():
    import salt_r.policy_model as m

    src = inspect.getsource(m)
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module or ""]
            for name in names:
                assert "tsa" not in (name or "").lower(), (
                    f"TSA import found in policy_model.py: {name}"
                )
                assert "target_state" not in (name or "").lower(), (
                    f"TargetState import found in policy_model.py: {name}"
                )
