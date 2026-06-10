"""CSC-v4 multi-head model (module A6).

V4 turns CSC from "state-classifier + hand-policy" into **diagnosis +
action-utility**. The model is a single causal-TCN encoder feeding seven heads
sized by :data:`csc_lib.csc.v4.v4types.HEAD_DIMS`:

  - ``derived``               (4)  softmax  — CC/CU/LA/FC                 (PRIMARY)
  - ``fc_subtype``            (3)  softmax  — NONE/DISTRACTOR/BACKGROUND  (collapses to FC)
  - ``la_subtype``            (6)  softmax  — NONE/FALSE/SMOOTH/ABRUPT/OCCLUDED/CANDIDATE
  - ``hazard``                (3)  sigmoid  — P(failure within next 1 / 3 / 10 frames)
  - ``action_utility``        (7)  linear   — predicted ΔIoU per Action (regression)
  - ``do_not_act``            (1)  sigmoid  — all actions ≤ 0
  - ``template_update_safe``  (1)  sigmoid  — safe to update the template now?

The encoder mirrors V3's :class:`csc_lib.csc.model.CSCTCN`: causal 1-D conv
blocks, kernel 3, dilations ``[1, 2, 4, 8]``, hidden 64, residual + LayerNorm,
strictly causal (left-only padding -> ``output[b, t]`` depends only on
``input[b, :t+1]``). Not bidirectional.

Additive, V3-frozen: this file does not import or touch any V3 model / training
code. It depends only on torch and :mod:`csc_lib.csc.v4.v4types`.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from csc_lib.csc.v4.v4types import (
    ACTION_NAMES,
    HEAD_DIMS,
    N_ACTIONS,
    V4Prediction,
)

__all__ = ["CSCv4"]


# ---------------------------------------------------------------------------
# Causal TCN encoder (mirrors csc_lib/csc/model.py CSCTCN: kernel 3,
# dilations [1,2,4,8], hidden 64, residual + LayerNorm, not bidirectional)
# ---------------------------------------------------------------------------


class _CausalConv1d(nn.Module):
    """Causal 1-D convolution — left-only padding, no future leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    """Residual causal block: two causal convs + LayerNorm + ReLU + dropout."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T). LayerNorm operates over channels, so permute around it.
        h = self.act(self.norm1(self.conv1(x).permute(0, 2, 1)).permute(0, 2, 1))
        h = self.drop(h)
        h = self.norm2(self.conv2(h).permute(0, 2, 1)).permute(0, 2, 1)
        return self.drop(self.act(h + x))


# ---------------------------------------------------------------------------
# CSCv4 — causal-TCN encoder + 7 heads
# ---------------------------------------------------------------------------


class CSCv4(nn.Module):
    """V4 multi-head diagnosis + action-utility model.

    Args:
        feature_dim: per-frame input feature width (from
            :mod:`csc_lib.csc.v4.features_v4` ``FEATURE_NAMES_V4``).
        hidden_dim: TCN channel width (default 64, matching V3 CSCTCN).
        kernel_size: causal-conv kernel (default 3).
        dilations: dilation schedule, one residual block per entry
            (default ``[1, 2, 4, 8]``).
        dropout: dropout inside each TCN block.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        dilations = [1, 2, 4, 8] if dilations is None else list(dilations)
        self.dilations = dilations

        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.tcn_blocks = nn.ModuleList(
            [_TCNBlock(hidden_dim, kernel_size, d, dropout) for d in dilations]
        )

        # Heads — sizes pinned to the shared HEAD_DIMS contract.
        self.head_derived = nn.Linear(hidden_dim, HEAD_DIMS["derived"])                      # softmax
        self.head_fc_subtype = nn.Linear(hidden_dim, HEAD_DIMS["fc_subtype"])                # softmax
        self.head_la_subtype = nn.Linear(hidden_dim, HEAD_DIMS["la_subtype"])                # softmax
        self.head_hazard = nn.Linear(hidden_dim, HEAD_DIMS["hazard"])                        # sigmoid (next_1/3/10)
        self.head_action_utility = nn.Linear(hidden_dim, HEAD_DIMS["action_utility"])        # linear regression (ΔIoU)
        self.head_do_not_act = nn.Linear(hidden_dim, HEAD_DIMS["do_not_act"])                # sigmoid
        self.head_template_update_safe = nn.Linear(hidden_dim, HEAD_DIMS["template_update_safe"])  # sigmoid

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- encoder ----------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, T, hidden) causal temporal features."""
        h = self.proj(x)          # (B, T, hidden)
        h = h.permute(0, 2, 1)    # (B, hidden, T) for Conv1d
        for block in self.tcn_blocks:
            h = block(h)
        return h.permute(0, 2, 1)  # (B, T, hidden)

    # ---- forward (logits) -------------------------------------------------
    def forward(self, x: torch.Tensor, last_step_only: bool = False) -> dict[str, torch.Tensor]:
        """Return a dict of raw logits (no activations applied).

        Shapes (with ``T_out = 1`` if ``last_step_only`` else ``T``):
          derived (B,T_out,4), fc_subtype (B,T_out,3), la_subtype (B,T_out,6),
          hazard (B,T_out,3), action_utility (B,T_out,7),
          do_not_act (B,T_out,1), template_update_safe (B,T_out,1).
        """
        h = self._encode(x)
        h_out = h[:, -1:, :] if last_step_only else h
        return {
            "derived": self.head_derived(h_out),
            "fc_subtype": self.head_fc_subtype(h_out),
            "la_subtype": self.head_la_subtype(h_out),
            "hazard": self.head_hazard(h_out),                       # raw — sigmoid at predict()
            "action_utility": self.head_action_utility(h_out),       # regression — raw ΔIoU
            "do_not_act": self.head_do_not_act(h_out),               # raw — sigmoid at predict()
            "template_update_safe": self.head_template_update_safe(h_out),  # raw — sigmoid at predict()
        }

    # ---- predict (V4Prediction, Student / causal / no GT) -----------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, last_step_only: bool = True) -> V4Prediction:
        """Run the model and pack a :class:`V4Prediction` for the *last* frame.

        Softmax for the categorical heads, sigmoid for hazard / do_not_act /
        template_update_safe, identity (raw) for the action_utility regression.
        ``action_utility`` is returned as a dict keyed by
        :data:`ACTION_NAMES`; ``hazard`` as ``{'next_1','next_3','next_10'}``.

        Operates on the first batch element (single-sequence Student inference).
        """
        t0 = time.perf_counter()
        out = self.forward(x, last_step_only=last_step_only)

        # Reduce to the single decision step of the first batch element: (dim,)
        def _last(name: str) -> torch.Tensor:
            return out[name][0, -1]

        derived_logits = _last("derived")
        derived_probs = F.softmax(derived_logits, dim=-1)
        fc_probs = F.softmax(_last("fc_subtype"), dim=-1)
        la_probs = F.softmax(_last("la_subtype"), dim=-1)

        hazard_p = torch.sigmoid(_last("hazard"))                    # (3,)
        action_util = _last("action_utility")                        # (N_ACTIONS,) raw ΔIoU
        do_not_act_p = torch.sigmoid(_last("do_not_act")).reshape(-1)[0]
        template_safe_p = torch.sigmoid(_last("template_update_safe")).reshape(-1)[0]

        derived_probs_np = derived_probs.detach().cpu().numpy()
        derived_state = int(derived_probs_np.argmax())
        # Risk = P(LA) + P(FC), mirroring V3's derived-head risk definition.
        risk_score = float(derived_probs_np[2] + derived_probs_np[3])

        hazard_np = hazard_p.detach().cpu().numpy()
        hazard = {
            "next_1": float(hazard_np[0]),
            "next_3": float(hazard_np[1]),
            "next_10": float(hazard_np[2]),
        }

        util_np = action_util.detach().cpu().numpy()
        action_utility = {name: float(util_np[i]) for i, name in enumerate(ACTION_NAMES)}

        return V4Prediction(
            derived_probs=derived_probs_np,
            derived_state=derived_state,
            fc_subtype_probs=fc_probs.detach().cpu().numpy(),
            la_subtype_probs=la_probs.detach().cpu().numpy(),
            hazard=hazard,
            action_utility=action_utility,
            do_not_act_prob=float(do_not_act_p),
            template_update_safe_prob=float(template_safe_p),
            risk_score=risk_score,
            latency_ms=(time.perf_counter() - t0) * 1e3,
        )


# ---------------------------------------------------------------------------
# Standalone smoke (CPU-only, no datasets)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, Fdim = 2, 32, 20
    model = CSCv4(feature_dim=Fdim)
    print(f"[smoke] CSCv4(feature_dim={Fdim}) params={model.num_params:,}")

    x = torch.randn(B, T, Fdim)

    # --- forward: assert every head's logit shape ---
    model.eval()
    out = model.forward(x)
    expected = {
        "derived": 4,
        "fc_subtype": 3,
        "la_subtype": 6,
        "hazard": 3,
        "action_utility": N_ACTIONS,  # 7
        "do_not_act": 1,
        "template_update_safe": 1,
    }
    assert set(out.keys()) == set(expected.keys()), f"head set mismatch: {set(out.keys())}"
    for name, dim in expected.items():
        assert out[name].shape == (B, T, dim), f"{name}: {tuple(out[name].shape)} != {(B, T, dim)}"
        assert torch.isfinite(out[name]).all(), f"{name} has non-finite logits"
        print(f"[smoke] forward head {name:<22} {tuple(out[name].shape)}")

    # --- forward last_step_only ---
    out_last = model.forward(x, last_step_only=True)
    for name, dim in expected.items():
        assert out_last[name].shape == (B, 1, dim), f"{name} last_step: {tuple(out_last[name].shape)}"

    # --- predict: assert it returns a V4Prediction with N_ACTIONS utility keys ---
    pred = model.predict(x, last_step_only=True)
    assert isinstance(pred, V4Prediction), type(pred)
    assert pred.derived_probs.shape == (4,), pred.derived_probs.shape
    assert pred.fc_subtype_probs.shape == (3,), pred.fc_subtype_probs.shape
    assert pred.la_subtype_probs.shape == (6,), pred.la_subtype_probs.shape
    assert len(pred.action_utility) == N_ACTIONS, len(pred.action_utility)
    assert set(pred.action_utility.keys()) == set(ACTION_NAMES), pred.action_utility.keys()
    assert set(pred.hazard.keys()) == {"next_1", "next_3", "next_10"}, pred.hazard.keys()
    assert 0.0 <= pred.do_not_act_prob <= 1.0, pred.do_not_act_prob
    assert 0.0 <= pred.template_update_safe_prob <= 1.0, pred.template_update_safe_prob
    assert abs(float(pred.derived_probs.sum()) - 1.0) < 1e-4, pred.derived_probs.sum()
    for k, v in pred.hazard.items():
        assert 0.0 <= v <= 1.0, f"hazard[{k}]={v} out of [0,1]"

    print(f"[smoke] predict -> V4Prediction: derived_state={pred.derived_state} "
          f"risk={pred.risk_score:.3f} latency={pred.latency_ms:.2f}ms")
    print(f"[smoke] action_utility keys = {list(pred.action_utility.keys())}")
    print(f"[smoke] hazard = {pred.hazard}")
    print("[smoke] OK")
