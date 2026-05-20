"""SALT-RD model: multi-head GRU tracking-risk controller.

Input:  (B, T, N_FEATURES=28) float32 scalar telemetry window
Output: dict[str, Tensor] of shape (B,) sigmoid probabilities per head
Heads:  false_confirmed, failure_in_5, recoverable,
        target_dynamic, camera_dynamic, hard_dynamic_scene, needs_full_compute
Params: ~7k (GRU hidden=64, layers=2, shared trunk → 7 linear heads)
"""
import torch
import torch.nn as nn
from torch import Tensor

LABEL_NAMES = [
    "correct", "false_confirmed", "failure_in_5", "recoverable",
    "target_dynamic", "camera_dynamic", "hard_dynamic_scene", "needs_full_compute",
]
HEAD_NAMES = LABEL_NAMES[1:]  # all except "correct" (correct = 1 - P(failure) implicitly)

# v1 schema: adds two semantically distinct dynamic-label heads.
LABEL_NAMES_V1 = LABEL_NAMES + ["hard_dynamic_scene_v2", "imminent_failure_dynamic"]
HEAD_NAMES_V1 = LABEL_NAMES_V1[1:]  # 9 heads

# v2 schema: longer-horizon failure labels for proactive recovery research.
LABEL_NAMES_V2 = LABEL_NAMES_V1 + [
    "failure_in_10",
    "failure_in_20",
    "imminent_failure_dynamic_10",
    "imminent_failure_dynamic_20",
]
HEAD_NAMES_V2 = LABEL_NAMES_V2[1:]  # 13 heads
N_FEATURES = 28
HIDDEN_DIM = 64
N_LAYERS = 2


class SALTRDHead(nn.Module):
    """Single binary prediction head."""
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, h: Tensor) -> Tensor:
        return torch.sigmoid(self.fc(h)).squeeze(-1)  # (B,)


class SALTRD(nn.Module):
    """SALT-RD multi-head GRU temporal reliability/dynamicity controller."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        dropout: float = 0.2,
        window: int = 20,
        head_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.window = window
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        _heads = head_names if head_names is not None else HEAD_NAMES

        self.input_norm = nn.LayerNorm(n_features)
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # Separate head per predicted label (excluding "correct")
        self.heads = nn.ModuleDict({
            name: SALTRDHead(hidden_dim)
            for name in _heads
        })
        self._init_weights()

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
                nn.init.constant_(p, -2.0)  # start with low prior (rare events)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: (B, T, n_features) float32 — telemetry window
        Returns:
            dict of head_name → (B,) sigmoid probability
        """
        x = self.input_norm(x)
        out, _ = self.gru(x)          # (B, T, hidden_dim)
        last = out[:, -1, :]          # (B, hidden_dim) — last timestep
        return {name: head(last) for name, head in self.heads.items()}

    @torch.no_grad()
    def predict_single(self, features: "np.ndarray", device: str = "cpu") -> dict[str, float]:
        """Online inference for one frame window.

        Args:
            features: (T, n_features) float32 array — last T frames of telemetry
        Returns:
            dict of head_name → float probability
        """
        import numpy as np
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
        probs = self.forward(x)
        return {k: float(v.item()) for k, v in probs.items()}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(checkpoint: str | None = None, device: str = "cpu") -> SALTRD:
    """Build SALTRD model, optionally loading a checkpoint."""
    model = SALTRD()
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    model.to(device)
    return model
