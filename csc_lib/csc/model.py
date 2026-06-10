"""CSC model implementations — CSCGRU, CSCMLP, CSCTCN with composite heads.

Architecture (V2 — 3-head composite, locked baseline):
  - head_loc:     localization_state  (STABLE/UNCERTAIN/LOST)    — auxiliary
  - head_conf:    confidence_state    (LOW/HIGH)                  — auxiliary
  - head_derived: derived_state       (CORRECT_CONFIRMED / CORRECT_UNCERTAIN /
                                       LOST_AWARE / FALSE_CONFIRMED)  — PRIMARY

Architecture (V3 — adds proactive forecast heads, gated by ``cfg.enable_forecast_heads``):
  - head_failure_next_10         (binary)  — any failure in t+1..t+horizon?
  - head_false_confirmed_next_10 (binary)  — FALSE_CONFIRMED in t+1..t+horizon?
  - head_lost_aware_next_10      (binary)  — LOST_AWARE in t+1..t+horizon?

V3 input features are unchanged from V2 — strictly causal, no GT/IoU/center_error
leakage.  The forecast targets are derived from future ``derived_state`` labels at
training time only (see :mod:`csc_lib.csc.labeling.risk_labeler`).

The derived head is supervised directly on the 4-class DerivedState labels so
FALSE_CONFIRMED gets an explicit gradient, instead of the model having to discover
the LOST+HIGH_CONFIDENCE conjunction by itself.

Risk score: P(LOST_AWARE) + P(FALSE_CONFIRMED)  =  P(derived ∈ failure_states)
           (or equivalently: 1 - P(CORRECT_CONFIRMED) - P(CORRECT_UNCERTAIN))

Legacy compatibility (V0 — 2-head):
  Some checkpoints were trained with the old 2-head architecture:
  - head_state:  6-class (CONFIRMED/UNCERTAIN/OCCLUDED/LOST/DISTRACTOR/FALSE_CONFIRMED)
  - head_risk:   1-class (binary failure risk)
  These checkpoints are detected automatically in load_runtime() and loaded via
  LegacyCSCGRU which adapts the old head outputs to the new CSCOutput format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from csc_lib.csc.config import CSCModelConfig
from csc_lib.csc.labeling.label_schema import (
    AUX_FLAGS,
    NUM_CONFIDENCE_STATES,
    NUM_DERIVED_STATES,
    NUM_LOCALIZATION_STATES,
)


@dataclass
class CSCOutput:
    """Per-frame model output."""

    localization_logits: torch.Tensor   # (B, T, 3)  — auxiliary
    confidence_logits: torch.Tensor     # (B, T, 2)  — auxiliary
    aux_logits: torch.Tensor            # (B, T, n_aux)
    derived_logits: torch.Tensor        # (B, T, 4)  — PRIMARY
    # ----- V3 forecast logits (None when forecast heads disabled) ---
    failure_next_10_logit: Optional[torch.Tensor] = None          # (B, T, 1)
    false_confirmed_next_10_logit: Optional[torch.Tensor] = None  # (B, T, 1)
    lost_aware_next_10_logit: Optional[torch.Tensor] = None       # (B, T, 1)


# ---------------------------------------------------------------------------
# Shared predict() mixin
# ---------------------------------------------------------------------------


class _CSCPredictMixin:
    """Mixin that adds a predict() method for any model that implements forward()."""

    @torch.no_grad()
    def predict(self, x: torch.Tensor, last_step_only: bool = False) -> dict[str, torch.Tensor]:
        out = self.forward(x, last_step_only=last_step_only)  # type: ignore[call-arg]
        loc_probs  = F.softmax(out.localization_logits, dim=-1)
        conf_probs = F.softmax(out.confidence_logits,   dim=-1)
        der_probs  = F.softmax(out.derived_logits,      dim=-1)

        loc_pred  = loc_probs.argmax(dim=-1)
        conf_pred = conf_probs.argmax(dim=-1)
        der_pred  = der_probs.argmax(dim=-1)

        risk = der_probs[..., 2] + der_probs[..., 3]
        risk_from_loc = loc_probs[..., 2]

        result = {
            "derived_probs":            der_probs,
            "predicted_derived":        der_pred,
            "localization_probs":       loc_probs,
            "confidence_probs":         conf_probs,
            "predicted_localization":   loc_pred,
            "predicted_confidence":     conf_pred,
            "risk_score":               risk,
            "risk_score_loc":           risk_from_loc,
            "aux_probs":                torch.sigmoid(out.aux_logits),
        }

        # V3 forecast probabilities (only when forecast heads are enabled)
        if out.failure_next_10_logit is not None:
            result["failure_next_10_prob"] = torch.sigmoid(
                out.failure_next_10_logit.squeeze(-1)
            )
        if out.false_confirmed_next_10_logit is not None:
            result["false_confirmed_next_10_prob"] = torch.sigmoid(
                out.false_confirmed_next_10_logit.squeeze(-1)
            )
        if out.lost_aware_next_10_logit is not None:
            result["lost_aware_next_10_prob"] = torch.sigmoid(
                out.lost_aware_next_10_logit.squeeze(-1)
            )

        return result


def _build_heads(out_dim: int, n_aux: int) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
    """Return (head_loc, head_conf, head_aux, head_derived)."""
    return (
        nn.Linear(out_dim, NUM_LOCALIZATION_STATES),
        nn.Linear(out_dim, NUM_CONFIDENCE_STATES),
        nn.Linear(out_dim, n_aux),
        nn.Linear(out_dim, NUM_DERIVED_STATES),
    )


def _build_forecast_heads(out_dim: int) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    """Return (failure_next_10, false_confirmed_next_10, lost_aware_next_10) heads."""
    return (
        nn.Linear(out_dim, 1),
        nn.Linear(out_dim, 1),
        nn.Linear(out_dim, 1),
    )


# ---------------------------------------------------------------------------
# CSCGRU
# ---------------------------------------------------------------------------


class CSCGRU(_CSCPredictMixin, nn.Module):
    """Tiny GRU — 3 heads: loc (aux), conf (aux), derived (primary).

    With ``cfg.enable_forecast_heads=True`` (V3) adds 3 binary forecast heads:
    failure_next_10, false_confirmed_next_10, lost_aware_next_10.
    """

    def __init__(self, cfg: CSCModelConfig, n_aux: int = len(AUX_FLAGS)) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_aux = n_aux
        self.enable_forecast = bool(getattr(cfg, "enable_forecast_heads", False))

        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.head_loc, self.head_conf, self.head_aux, self.head_derived = _build_heads(out_dim, n_aux)
        if self.enable_forecast:
            (
                self.failure_next_10_head,
                self.false_confirmed_next_10_head,
                self.lost_aware_next_10_head,
            ) = _build_forecast_heads(out_dim)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, last_step_only: bool = False) -> CSCOutput:
        h = self.proj(x)
        h, _ = self.gru(h)
        h_out = h[:, -1:, :] if last_step_only else h
        fail_logit = fc_logit = lost_logit = None
        if self.enable_forecast:
            fail_logit = self.failure_next_10_head(h_out)
            fc_logit = self.false_confirmed_next_10_head(h_out)
            lost_logit = self.lost_aware_next_10_head(h_out)
        return CSCOutput(
            localization_logits=self.head_loc(h_out),
            confidence_logits=self.head_conf(h_out),
            aux_logits=self.head_aux(h_out),
            derived_logits=self.head_derived(h_out),
            failure_next_10_logit=fail_logit,
            false_confirmed_next_10_logit=fc_logit,
            lost_aware_next_10_logit=lost_logit,
        )


# ---------------------------------------------------------------------------
# CSCMLP  (per-frame baseline, no temporal context)
# ---------------------------------------------------------------------------


class CSCMLP(_CSCPredictMixin, nn.Module):
    """Per-frame MLP — 3 heads: loc (aux), conf (aux), derived (primary).

    With ``cfg.enable_forecast_heads=True`` (V3) adds 3 binary forecast heads.
    """

    def __init__(self, cfg: CSCModelConfig, n_aux: int = len(AUX_FLAGS)) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_aux = n_aux
        self.enable_forecast = bool(getattr(cfg, "enable_forecast_heads", False))

        layers: list[nn.Module] = [
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(0, cfg.num_layers - 1)):
            layers += [
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.mlp = nn.Sequential(*layers)
        self.head_loc, self.head_conf, self.head_aux, self.head_derived = _build_heads(cfg.hidden_dim, n_aux)
        if self.enable_forecast:
            (
                self.failure_next_10_head,
                self.false_confirmed_next_10_head,
                self.lost_aware_next_10_head,
            ) = _build_forecast_heads(cfg.hidden_dim)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, last_step_only: bool = False) -> CSCOutput:
        B, T, F = x.shape
        h = self.mlp(x.reshape(B * T, F)).reshape(B, T, -1)
        h_out = h[:, -1:, :] if last_step_only else h
        fail_logit = fc_logit = lost_logit = None
        if self.enable_forecast:
            fail_logit = self.failure_next_10_head(h_out)
            fc_logit = self.false_confirmed_next_10_head(h_out)
            lost_logit = self.lost_aware_next_10_head(h_out)
        return CSCOutput(
            localization_logits=self.head_loc(h_out),
            confidence_logits=self.head_conf(h_out),
            aux_logits=self.head_aux(h_out),
            derived_logits=self.head_derived(h_out),
            failure_next_10_logit=fail_logit,
            false_confirmed_next_10_logit=fc_logit,
            lost_aware_next_10_logit=lost_logit,
        )


# ---------------------------------------------------------------------------
# CSCTCN  (causal TCN — primary model for paper)
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
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x).permute(0, 2, 1)).permute(0, 2, 1))
        h = self.drop(h)
        h = self.norm2(self.conv2(h).permute(0, 2, 1)).permute(0, 2, 1)
        return self.drop(self.act(h + x))


class CSCTCN(_CSCPredictMixin, nn.Module):
    """Causal TCN — 3 heads: loc (aux), conf (aux), derived (primary).

    Strict causal guarantee: output[b, t, :] depends only on input[b, :t+1, :].
    Tested in tests/test_csctcn_causality.py.

    With ``cfg.enable_forecast_heads=True`` (V3) adds 3 binary forecast heads.
    Forecast heads share the same TCN encoder — causality is preserved.
    """

    def __init__(self, cfg: CSCModelConfig, n_aux: int = len(AUX_FLAGS)) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_aux = n_aux
        self.enable_forecast = bool(getattr(cfg, "enable_forecast_heads", False))

        tcn_cfg    = getattr(cfg, "tcn", None)
        kernel_size: int       = getattr(tcn_cfg, "kernel_size", 3)   if tcn_cfg else 3
        dilations: list[int]   = list(getattr(tcn_cfg, "dilations", [1, 2, 4, 8]))  if tcn_cfg else [1, 2, 4, 8]
        tcn_hidden: int        = getattr(tcn_cfg, "hidden_dim", cfg.hidden_dim) if tcn_cfg else cfg.hidden_dim
        tcn_dropout: float     = getattr(tcn_cfg, "dropout", cfg.dropout) if tcn_cfg else cfg.dropout

        n_layers = max(cfg.num_layers, len(dilations))
        while len(dilations) < n_layers:
            dilations.append(dilations[-1] * 2)
        dilations = dilations[:n_layers]

        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, tcn_hidden),
            nn.LayerNorm(tcn_hidden),
            nn.ReLU(inplace=True),
        )
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(tcn_hidden, kernel_size, d, tcn_dropout) for d in dilations
        ])
        self.head_loc, self.head_conf, self.head_aux, self.head_derived = _build_heads(tcn_hidden, n_aux)
        if self.enable_forecast:
            (
                self.failure_next_10_head,
                self.false_confirmed_next_10_head,
                self.lost_aware_next_10_head,
            ) = _build_forecast_heads(tcn_hidden)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, last_step_only: bool = False) -> CSCOutput:
        h = self.proj(x)
        h = h.permute(0, 2, 1)
        for block in self.tcn_blocks:
            h = block(h)
        h = h.permute(0, 2, 1)
        h_out = h[:, -1:, :] if last_step_only else h
        fail_logit = fc_logit = lost_logit = None
        if self.enable_forecast:
            fail_logit = self.failure_next_10_head(h_out)
            fc_logit = self.false_confirmed_next_10_head(h_out)
            lost_logit = self.lost_aware_next_10_head(h_out)
        return CSCOutput(
            localization_logits=self.head_loc(h_out),
            confidence_logits=self.head_conf(h_out),
            aux_logits=self.head_aux(h_out),
            derived_logits=self.head_derived(h_out),
            failure_next_10_logit=fail_logit,
            false_confirmed_next_10_logit=fc_logit,
            lost_aware_next_10_logit=lost_logit,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

CSCModel = Union[CSCGRU, CSCMLP, CSCTCN]


def build_model(cfg: CSCModelConfig) -> CSCModel:
    kind: str = getattr(cfg, "kind", "gru")
    if kind == "gru":
        return CSCGRU(cfg)
    if kind == "mlp":
        return CSCMLP(cfg)
    if kind == "tcn":
        return CSCTCN(cfg)
    raise ValueError(f"Unknown model kind: {kind!r}. Expected 'gru', 'mlp', or 'tcn'.")


# ---------------------------------------------------------------------------
# Legacy V0 — 2-head GRU (head_state 6-class + head_risk 1-class)
# Loaded automatically from old checkpoints for backward compatibility.
# ---------------------------------------------------------------------------

# Mapping from old 6-class head_state to new 3-class LocalizationState:
#   CONFIRMED(0)    -> STABLE(0)
#   UNCERTAIN(1)    -> UNCERTAIN(1)
#   OCCLUDED(2)     -> LOST(2)   (occluded → geometric loss of lock)
#   LOST(3)         -> LOST(2)
#   DISTRACTOR(4)   -> LOST(2)   (distractor → localization failure)
#   FALSE_CONFIRMED(5) -> LOST(2) (wrong bbox → localization failure)
_LEGACY_STATE_TO_LOC: list[int] = [0, 1, 2, 2, 2, 2]

# Mapping from old 6-class head_state to new 4-class DerivedState:
#   CONFIRMED(0)       -> CORRECT_CONFIRMED(0)
#   UNCERTAIN(1)       -> CORRECT_UNCERTAIN(1)
#   OCCLUDED(2)        -> LOST_AWARE(2)
#   LOST(3)            -> LOST_AWARE(2)
#   DISTRACTOR(4)      -> LOST_AWARE(2)
#   FALSE_CONFIRMED(5) -> FALSE_CONFIRMED(3)
_LEGACY_STATE_TO_DERIVED: list[int] = [0, 1, 2, 2, 2, 3]


class LegacyCSCGRU(_CSCPredictMixin, nn.Module):
    """Backward-compat wrapper for V0 checkpoints with head_state + head_risk.

    Exposes the same predict() → CSCOutput API as the current 3-head model.
    head_state (6-class) is remapped to loc (3-class) and derived (4-class)
    via static index permutation matrices.  head_risk (1-class sigmoid) is
    used as the single-node confidence head (HIGH if sigmoid > 0.5).
    head_aux is kept as-is but zero-padded to 5 entries if needed.
    """

    _N_LEGACY_STATES: int = 6
    _N_AUX_LEGACY: int = 4

    def __init__(self, cfg: "CSCModelConfig", n_aux_out: int = 5) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_aux_out = n_aux_out

        # Backbone (identical to current CSCGRU)
        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.gru = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        # Legacy heads (names match old checkpoint keys)
        self.head_state = nn.Linear(out_dim, self._N_LEGACY_STATES)
        self.head_risk   = nn.Linear(out_dim, 1)
        self.head_aux    = nn.Linear(out_dim, self._N_AUX_LEGACY)

        # Static permutation matrices for remapping old→new classes
        loc_perm = torch.zeros(self._N_LEGACY_STATES, 3)   # (6, 3)
        for old_idx, new_idx in enumerate(_LEGACY_STATE_TO_LOC):
            loc_perm[old_idx, new_idx] = 1.0
        self.register_buffer("_loc_perm", loc_perm)

        der_perm = torch.zeros(self._N_LEGACY_STATES, 4)   # (6, 4)
        for old_idx, new_idx in enumerate(_LEGACY_STATE_TO_DERIVED):
            der_perm[old_idx, new_idx] = 1.0
        self.register_buffer("_der_perm", der_perm)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> "CSCOutput":
        import torch.nn.functional as F_

        h = self.proj(x)
        h, _ = self.gru(h)

        state_logits = self.head_state(h)        # (B, T, 6)
        risk_logit   = self.head_risk(h)         # (B, T, 1) — binary risk
        aux_logits   = self.head_aux(h)          # (B, T, 4)

        # Softmax over old 6 classes, then remap to loc (3) and derived (4).
        state_probs = F_.softmax(state_logits, dim=-1)           # (B, T, 6)
        loc_probs_remap   = state_probs @ self._loc_perm         # (B, T, 3)
        der_probs_remap   = state_probs @ self._der_perm         # (B, T, 4)

        # Convert probs back to "logits" (log-probs) so that _CSCPredictMixin
        # can apply softmax again.  We add a small epsilon to prevent log(0).
        eps = 1e-7
        loc_logits = torch.log(loc_probs_remap.clamp(min=eps))
        der_logits = torch.log(der_probs_remap.clamp(min=eps))

        # Confidence head: broadcast risk sigmoid to 2-class logits.
        # P(HIGH_CONF) = sigmoid(risk_logit)  →  conf_logits = [0, risk_logit]
        conf_logits = torch.cat(
            [torch.zeros_like(risk_logit), risk_logit], dim=-1
        )  # (B, T, 2)

        # Pad aux from 4 → 5 zeros (new AUX_FLAGS has distractor_risk added)
        aux_pad = torch.zeros(
            *aux_logits.shape[:-1], 1, device=aux_logits.device, dtype=aux_logits.dtype
        )
        aux_logits_padded = torch.cat([aux_logits, aux_pad], dim=-1)  # (B, T, 5)

        return CSCOutput(
            localization_logits=loc_logits,
            confidence_logits=conf_logits,
            aux_logits=aux_logits_padded,
            derived_logits=der_logits,
        )
