"""SALT-RD Policy Model: GRU trunk with risk heads and action heads.

Input:  (B, T, N_FEATURES=28) float32 scalar telemetry window
Output: dict with 'risk_probs' and 'action_logits' (and optionally 'candidate_score')

Risk heads   (sigmoid probability, same semantics as model.py):
  false_confirmed          — tracker is tracking wrong target
  imminent_failure_dynamic — failure in next 10 frames
  recoverable              — target is recoverable via detector

Action heads (raw logits, no softmax — argmax at inference):
  compute  — 3-class: FULL / PRUNE_LIGHT / PRUNE_MEDIUM
  recovery — 4-class: NONE / SCORE_CANDIDATES / REINIT / REJECT_REINIT

Optional:
  candidate_score — scalar utility for a single candidate (only when
                    candidate_features provided to forward())

No TSA imports. No thresholds. All decisions learned.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from salt_r.feature_schema import FEATURE_SCHEMA_VERSION
from salt_r.actions import ComputeAction, RecoveryAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES: int = 28
HIDDEN_SIZE: int = 64
N_LAYERS: int = 2
WINDOW_SIZE: int = 20

# Ordered so that index == enum value — used by the loss function.
COMPUTE_ACTION_ORDER: list[str] = [
    ComputeAction.FULL.value,
    ComputeAction.PRUNE_LIGHT.value,
    ComputeAction.PRUNE_MEDIUM.value,
]  # 3 classes

RECOVERY_ACTION_ORDER: list[str] = [
    RecoveryAction.NONE.value,
    RecoveryAction.SCORE_CANDIDATES.value,
    RecoveryAction.REINIT.value,
    RecoveryAction.REJECT_REINIT.value,
]  # 4 classes

MODEL_FAMILY: str = "saltrd_policy"


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _RiskHead(nn.Module):
    """Single binary risk head: linear → sigmoid → scalar (B,)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, h: Tensor) -> Tensor:  # (B, in_dim) → (B,)
        return torch.sigmoid(self.fc(h)).squeeze(-1)


class _ActionHead(nn.Module):
    """Multi-class action head: linear projection, raw logits (B, n_classes)."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, h: Tensor) -> Tensor:  # (B, in_dim) → (B, n_classes)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------


class SALTRDPolicyNet(nn.Module):
    """GRU-based policy model with risk heads and action heads.

    No TSA. No thresholds. All decisions learned.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        n_layers: int = N_LAYERS,
        window_size: int = WINDOW_SIZE,
        dropout: float = 0.1,
        feature_schema: str = FEATURE_SCHEMA_VERSION,
        zero_feature_indices: tuple[int, ...] = (22, 23, 24, 25, 26, 27),
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.window_size = window_size
        self.feature_schema = feature_schema
        self.zero_feature_indices = tuple(zero_feature_indices)

        # Shared trunk
        self.input_norm = nn.LayerNorm(n_features)
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Risk heads — binary, sigmoid output
        self.risk_heads = nn.ModuleDict({
            "false_confirmed": _RiskHead(hidden_size),
            "imminent_failure_dynamic": _RiskHead(hidden_size),
            "recoverable": _RiskHead(hidden_size),
        })

        # Action heads — raw logits
        self.action_heads = nn.ModuleDict({
            "compute": _ActionHead(hidden_size, len(COMPUTE_ACTION_ORDER)),    # 3
            "recovery": _ActionHead(hidden_size, len(RECOVERY_ACTION_ORDER)),  # 4
        })

        # Candidate scorer — projects candidate features to scalar utility
        # The input dimension is flexible; we use a lazy approach: build the
        # linear lazily on first call so callers can pass any feature width.
        self._candidate_scorer: nn.Linear | None = None
        self._candidate_feat_dim: int | None = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "gru" in name:
                if p.dim() > 1:
                    nn.init.orthogonal_(p)
                else:
                    nn.init.zeros_(p)
            elif "fc.weight" in name:
                nn.init.xavier_uniform_(p)
            elif "fc.bias" in name:
                nn.init.constant_(p, -2.0)  # low prior — rare events

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        candidate_features: Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            x: ``(B, T, n_features)`` or ``(B, n_features)`` float32.
               If 2-D, treated as single-frame (seq_len=1).
            candidate_features: ``(B, n_candidate_features)`` optional float32
               per-candidate feature vector.  When provided the model also
               returns a scalar utility score per sample.

        Returns:
            dict with keys:
              ``risk_probs``     — dict: 'false_confirmed', 'imminent_failure_dynamic',
                                   'recoverable'  (sigmoid probs, shape ``(B,)``)
              ``action_logits``  — dict: 'compute' (B, 3), 'recovery' (B, 4)
              ``candidate_score`` — ``(B,)`` scalar or ``None``
        """
        # Handle 2-D input: (B, n_features) → (B, 1, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Trunk
        x = self.input_norm(x)
        gru_out, _ = self.gru(x)          # (B, T, hidden_size)
        h = gru_out[:, -1, :]             # (B, hidden_size) — last timestep

        # Risk heads
        risk_probs: dict[str, Tensor] = {
            name: head(h) for name, head in self.risk_heads.items()
        }

        # Action heads
        action_logits: dict[str, Tensor] = {
            name: head(h) for name, head in self.action_heads.items()
        }

        # Candidate scorer (lazy init)
        candidate_score: Tensor | None = None
        if candidate_features is not None:
            feat_dim = candidate_features.shape[-1]
            if self._candidate_scorer is None or self._candidate_feat_dim != feat_dim:
                self._candidate_scorer = nn.Linear(feat_dim, 1).to(x.device)
                nn.init.xavier_uniform_(self._candidate_scorer.weight)
                nn.init.zeros_(self._candidate_scorer.bias)
                self._candidate_feat_dim = feat_dim
            candidate_score = self._candidate_scorer(candidate_features).squeeze(-1)

        return {
            "risk_probs": risk_probs,
            "action_logits": action_logits,
            "candidate_score": candidate_score,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_family": MODEL_FAMILY,
            "feature_schema": self.feature_schema,
            "n_base_features": self.n_features,
            "zero_feature_indices": list(self.zero_feature_indices),
            "action_schema": "v1_reinit_compute",
            "trained_heads": [],   # filled by train_policy.py
            "created_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, extra_metadata: dict[str, Any] | None = None) -> None:
        """Save model checkpoint with metadata.

        Args:
            path: Destination file path (.pt).
            extra_metadata: Optional extra keys merged into the saved dict.

        Raises:
            ValueError: If ``extra_metadata`` contains a conflicting
                        ``feature_schema`` that does not match this model's
                        ``feature_schema``.
        """
        if extra_metadata and "feature_schema" in extra_metadata:
            if extra_metadata["feature_schema"] != self.feature_schema:
                raise ValueError(
                    f"feature_schema mismatch: model has '{self.feature_schema}' "
                    f"but extra_metadata specifies '{extra_metadata['feature_schema']}'"
                )
        ckpt: dict[str, Any] = {
            **self.get_metadata(),
            "model_state_dict": self.state_dict(),
            "init_kwargs": {
                "n_features": self.n_features,
                "hidden_size": self.hidden_size,
                "n_layers": self.n_layers,
                "window_size": self.window_size,
                "feature_schema": self.feature_schema,
                "zero_feature_indices": self.zero_feature_indices,
            },
        }
        if extra_metadata:
            ckpt.update(extra_metadata)
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path: str) -> "SALTRDPolicyNet":
        """Load and reconstruct model from checkpoint.

        Args:
            path: Path to a ``.pt`` checkpoint saved by :meth:`save`.

        Returns:
            Reconstructed :class:`SALTRDPolicyNet` with weights loaded.

        Raises:
            ValueError: If the checkpoint ``model_family`` != ``'saltrd_policy'``
                        (e.g. a risk-only SALTRD checkpoint was passed).
        """
        ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        family = ckpt.get("model_family", "")
        if family != MODEL_FAMILY:
            raise ValueError(
                f"Cannot load checkpoint: expected model_family='{MODEL_FAMILY}', "
                f"got '{family}'. Pass a policy model checkpoint, not a risk-only one."
            )
        init_kwargs: dict[str, Any] = ckpt.get("init_kwargs", {})
        model = cls(**init_kwargs)
        state = ckpt["model_state_dict"]
        # Pre-initialize candidate scorer head if checkpoint includes trained scorer
        # weights (added by train_candidate_scorer.py). Without this, the lazily-created
        # Linear would be ignored by load_state_dict even with strict=False.
        scorer_w_key = "_candidate_scorer.weight"
        if scorer_w_key in state:
            import torch as _torch
            feat_dim = state[scorer_w_key].shape[1]
            model._candidate_scorer = _torch.nn.Linear(feat_dim, 1)
            model._candidate_feat_dim = feat_dim
        model.load_state_dict(state, strict=False)
        return model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss function (module-level, not a method — keeps the class clean)
# ---------------------------------------------------------------------------

# Map RecoveryAction enum values to their integer class indices in
# RECOVERY_ACTION_ORDER so we can build cross-entropy targets.
_RECOVERY_NONE_IDX: int = RECOVERY_ACTION_ORDER.index(RecoveryAction.NONE.value)
_RECOVERY_REINIT_IDX: int = RECOVERY_ACTION_ORDER.index(RecoveryAction.REINIT.value)
_RECOVERY_REJECT_IDX: int = RECOVERY_ACTION_ORDER.index(RecoveryAction.REJECT_REINIT.value)


def compute_loss(
    outputs: dict[str, Any],
    targets: dict[str, Tensor],
    lambda_recovery: float = 1.0,
    lambda_candidate: float = 0.5,
    lambda_compute: float = 0.3,
    lambda_risk: float = 1.0,
) -> dict[str, Tensor]:
    """Compute joint policy loss.

    Args:
        outputs:  Return value of :meth:`SALTRDPolicyNet.forward`.
        targets:  Dict of ground-truth tensors.  Expected keys:

          * ``label_reinit``           (B,) float/bool — 1 if REINIT is correct action
          * ``label_reject``           (B,) float/bool — 1 if REJECT_REINIT is correct
          * ``risk_false_confirmed``   (B,) float — risk label
          * ``risk_imminent_failure_dynamic`` (B,) float — risk label
          * ``risk_recoverable``       (B,) float — risk label

          Optional keys for compute/candidate supervision:
          * ``label_compute``          (B,) long  — target compute class index
          * ``candidate_score_target`` (B,) float — target candidate utility

        lambda_recovery:  Weight for recovery action CE loss.
        lambda_candidate: Weight for candidate scoring MSE loss.
        lambda_compute:   Weight for compute action CE loss.
        lambda_risk:      Weight for combined risk BCE loss.

    Returns:
        dict with keys ``'total'``, ``'risk'``, ``'recovery'``,
        ``'candidate'``, ``'compute'``.

    Notes:
        Recovery action label logic::

            label_reinit=1 → class REINIT
            label_reject=1 → class REJECT_REINIT
            else           → class NONE

        (If both flags are set, REINIT takes priority.)
    """
    risk_probs = outputs["risk_probs"]
    action_logits = outputs["action_logits"]
    candidate_score = outputs["candidate_score"]

    device = next(iter(risk_probs.values())).device

    # ------------------------------------------------------------------
    # Risk loss (binary cross-entropy per head)
    # ------------------------------------------------------------------
    risk_loss = torch.tensor(0.0, device=device)
    risk_pairs = [
        ("false_confirmed", "risk_false_confirmed"),
        ("imminent_failure_dynamic", "risk_imminent_failure_dynamic"),
        ("recoverable", "risk_recoverable"),
    ]
    for head_name, target_key in risk_pairs:
        if target_key in targets:
            t = targets[target_key].float().to(device)
            p = risk_probs[head_name]
            risk_loss = risk_loss + nn.functional.binary_cross_entropy(p, t)

    # ------------------------------------------------------------------
    # Recovery action loss (4-class CE)
    # ------------------------------------------------------------------
    label_reinit = targets.get("label_reinit")
    label_reject = targets.get("label_reject")

    recovery_loss = torch.tensor(0.0, device=device)
    if label_reinit is not None and label_reject is not None:
        B = label_reinit.shape[0]
        recovery_target = torch.full((B,), _RECOVERY_NONE_IDX, dtype=torch.long, device=device)
        # REINIT takes priority over REJECT if both are set
        recovery_target = torch.where(
            label_reject.bool().to(device), torch.full_like(recovery_target, _RECOVERY_REJECT_IDX), recovery_target
        )
        recovery_target = torch.where(
            label_reinit.bool().to(device), torch.full_like(recovery_target, _RECOVERY_REINIT_IDX), recovery_target
        )
        recovery_logits = action_logits["recovery"]
        recovery_loss = nn.functional.cross_entropy(recovery_logits, recovery_target)

    # ------------------------------------------------------------------
    # Compute action loss (3-class CE)
    # ------------------------------------------------------------------
    compute_loss_val = torch.tensor(0.0, device=device)
    if "label_compute" in targets:
        compute_target = targets["label_compute"].long().to(device)
        compute_loss_val = nn.functional.cross_entropy(action_logits["compute"], compute_target)

    # ------------------------------------------------------------------
    # Candidate score loss (MSE)
    # ------------------------------------------------------------------
    candidate_loss = torch.tensor(0.0, device=device)
    if candidate_score is not None and "candidate_score_target" in targets:
        cs_target = targets["candidate_score_target"].float().to(device)
        candidate_loss = nn.functional.mse_loss(candidate_score, cs_target)

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------
    total = (
        lambda_risk * risk_loss
        + lambda_recovery * recovery_loss
        + lambda_compute * compute_loss_val
        + lambda_candidate * candidate_loss
    )

    return {
        "total": total,
        "risk": risk_loss,
        "recovery": recovery_loss,
        "candidate": candidate_loss,
        "compute": compute_loss_val,
    }
