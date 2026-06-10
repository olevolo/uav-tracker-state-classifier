"""SceneStateClassifier — multi-head GRU scene state model for SALT-RD.

Input:  (B, T, 28) telemetry window  (same as PolicyNet)
Output: 5 calibrated probabilities → SceneState enum

Heads included (per PROJECT_X.md metrics):
  false_confirmed          AUROC 0.885  AUPRC 0.361  base 5.5%  ✅
  imminent_failure_dynamic AUROC 0.889  AUPRC 0.263  base 4.2%  ✅
  recoverable              AUROC 0.894  AUPRC 0.042  base 0.6%  ✅ (low AUPRC, monitor)
  target_dynamic           AUROC 0.730  AUPRC 0.123  base 5.6%  ✅
  hard_dynamic_scene       AUROC 0.604  AUPRC 0.214  base 11.9% ✅

Heads explicitly excluded:
  failure_in_10    AUPRC 0.013  — base rate too low, gate always open
  camera_dynamic   AUROC 0.457  — worse than random, do not use

SceneState ≠ SceneClass.  SceneState is new, lives in SALT-RD controller.
SceneClass is the old scheduler concept (disabled).

Training note (PROJECT_X.md): train first, calibrate thresholds on val set,
then wire into controller. DEFAULT_THRESHOLDS are placeholders — do not use
in production before calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from csc_uav_tracking.telemetry.schema import FEATURE_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES: int = 28
HIDDEN_SIZE: int = 64
N_LAYERS: int = 2
WINDOW_SIZE: int = 20

HEAD_NAMES: list[str] = [
    "false_confirmed",
    "imminent_failure_dynamic",
    "recoverable",
    "target_dynamic",
    "hard_dynamic_scene",
]

# Empirical base rates — used to compute pos_weight for BCEWithLogitsLoss.
HEAD_BASE_RATES: dict[str, float] = {
    "false_confirmed":          0.055,
    "imminent_failure_dynamic": 0.042,
    "recoverable":              0.006,
    "target_dynamic":           0.056,
    "hard_dynamic_scene":       0.119,
}

# Label column names in the training NPZ (same as model.py LABEL_NAMES_V2).
HEAD_LABEL_KEYS: dict[str, str] = {
    "false_confirmed":          "fc",
    "imminent_failure_dynamic": "ifd10",
    "recoverable":              "recoverable",
    "target_dynamic":           "target_dynamic",
    "hard_dynamic_scene":       "hard_dynamic_scene_v2",
}

MODEL_FAMILY: str = "scene_state_classifier"


# ---------------------------------------------------------------------------
# SceneState enum
# ---------------------------------------------------------------------------

class SceneState(str, Enum):
    """Ordered scene states from worst to best (priority order in combiner)."""
    FALSE_CONFIRMED = "FALSE_CONFIRMED"   # tracker on wrong object — highest priority
    AT_RISK         = "AT_RISK"           # imminent failure, dynamic scene
    RECOVERING      = "RECOVERING"        # target is recoverable
    DYNAMIC         = "DYNAMIC"           # hard or dynamic scene, not yet failing
    STABLE          = "STABLE"            # no risk head fired — default cheapest state


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _Head(nn.Module):
    """Single binary classification head: Linear → scalar logit."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, h: Tensor) -> Tensor:
        return self.fc(h).squeeze(-1)  # (B,)


class SceneStateNet(nn.Module):
    """Multi-head GRU scene state classifier.

    Architecture mirrors SALTRDPolicyNet for consistency:
      GRU(28 → hidden_size, n_layers) → last hidden → 5 binary heads

    Each head outputs a scalar logit. Calibration is applied via per-head
    temperature scaling (log_temperature learnable parameter, init 0 → T=1).
    """

    def __init__(
        self,
        input_dim:   int = N_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        n_layers:    int = N_LAYERS,
        window_size: int = WINDOW_SIZE,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.input_dim   = input_dim
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.heads = nn.ModuleDict({
            name: _Head(hidden_size) for name in HEAD_NAMES
        })

        # Per-head temperature scaling (calibration). log_T initialised to 0 → T=1.
        # During calibration, only these parameters are trained (backbone frozen).
        self.log_temperatures = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in HEAD_NAMES
        })

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return raw logits (B,) per head. Use predict_probs for calibrated output.

        Args:
            x: (B, T, input_dim) float32 telemetry window.

        Returns:
            Dict head_name → (B,) logit tensor.
        """
        out, _ = self.gru(x)          # (B, T, hidden)
        h = out[:, -1, :]             # (B, hidden) — last timestep
        return {name: self.heads[name](h) for name in HEAD_NAMES}

    def predict_probs(self, x: Tensor) -> dict[str, Tensor]:
        """Calibrated sigmoid probabilities via temperature scaling.

        Args:
            x: (B, T, input_dim) telemetry window.

        Returns:
            Dict head_name → (B,) probability in [0, 1].
        """
        logits = self.forward(x)
        return {
            name: torch.sigmoid(logits[name] / self.log_temperatures[name].exp())
            for name in HEAD_NAMES
        }

    def predict_probs_numpy(self, x: Tensor) -> dict[str, float]:
        """Convenience: single-sample inference → plain float dict.

        Args:
            x: (1, T, input_dim) or (T, input_dim) — auto-unsqueezed.

        Returns:
            Dict head_name → float probability.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            probs = self.predict_probs(x)
        return {name: float(p[0].detach()) for name, p in probs.items()}

    # ------------------------------------------------------------------

    def pos_weights(self, device: torch.device | None = None) -> dict[str, Tensor]:
        """Per-head pos_weight = (1 - base_rate) / base_rate for BCEWithLogitsLoss."""
        return {
            name: torch.tensor(
                [(1.0 - r) / max(r, 1e-6)],
                dtype=torch.float32,
                device=device,
            )
            for name, r in HEAD_BASE_RATES.items()
        }

    # ------------------------------------------------------------------

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        ckpt: dict[str, Any] = {
            "model_family": MODEL_FAMILY,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "config": {
                "input_dim":   self.input_dim,
                "hidden_size": self.hidden_size,
                "n_layers":    self.n_layers,
                "window_size": self.window_size,
            },
            "head_names": HEAD_NAMES,
            "model": self.state_dict(),
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SceneStateNet":
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = cls(
            input_dim=cfg.get("input_dim",   N_FEATURES),
            hidden_size=cfg.get("hidden_size", HIDDEN_SIZE),
            n_layers=cfg.get("n_layers",     N_LAYERS),
            window_size=cfg.get("window_size", WINDOW_SIZE),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model.to(device)


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------

@dataclass
class SceneStateThresholds:
    """Per-head decision thresholds.

    IMPORTANT: DEFAULT values are placeholders. Run calibrate_scene_thresholds()
    on the val set after retraining before deploying to production.
    """
    false_confirmed:          float = 0.50
    imminent_failure_dynamic: float = 0.50
    recoverable:              float = 0.50
    target_dynamic:           float = 0.40
    hard_dynamic_scene:       float = 0.40

    def as_dict(self) -> dict[str, float]:
        return {
            "false_confirmed":          self.false_confirmed,
            "imminent_failure_dynamic": self.imminent_failure_dynamic,
            "recoverable":              self.recoverable,
            "target_dynamic":           self.target_dynamic,
            "hard_dynamic_scene":       self.hard_dynamic_scene,
        }


_DEFAULT_THRESHOLDS = SceneStateThresholds()


def classify_scene(
    probs: dict[str, float],
    thresholds: SceneStateThresholds | None = None,
) -> SceneState:
    """Priority combiner: worst risk state wins.

    Evaluation order follows clinical priority — FALSE_CONFIRMED is the most
    actionable state (tracker is already wrong), AT_RISK means imminent failure.

    Args:
        probs:      Dict from SceneStateNet.predict_probs_numpy().
        thresholds: Calibrated thresholds. Uses DEFAULT_THRESHOLDS if None
                    (placeholder values, not calibrated).

    Returns:
        SceneState enum value.
    """
    θ = (thresholds or _DEFAULT_THRESHOLDS).as_dict()

    if probs.get("false_confirmed", 0.0) > θ["false_confirmed"]:
        return SceneState.FALSE_CONFIRMED

    if probs.get("imminent_failure_dynamic", 0.0) > θ["imminent_failure_dynamic"]:
        return SceneState.AT_RISK

    if probs.get("recoverable", 0.0) > θ["recoverable"]:
        return SceneState.RECOVERING

    if (probs.get("hard_dynamic_scene", 0.0) > θ["hard_dynamic_scene"] or
            probs.get("target_dynamic", 0.0) > θ["target_dynamic"]):
        return SceneState.DYNAMIC

    return SceneState.STABLE


# ---------------------------------------------------------------------------
# Runtime inference helper
# ---------------------------------------------------------------------------

class SceneStateClassifier:
    """Stateful runtime wrapper: maintains telemetry window, classifies each frame.

    Usage:
        classifier = SceneStateClassifier.from_checkpoint("path/to/scene_state.pt")
        classifier.reset()
        for frame in sequence:
            features = evidence_extractor.extract(frame)   # (28,) np.ndarray
            state = classifier.step(features)
            # state: SceneState
    """

    def __init__(
        self,
        model: SceneStateNet,
        thresholds: SceneStateThresholds | None = None,
        device: str = "cpu",
    ) -> None:
        self._model      = model.to(device).eval()
        self._thresholds = thresholds or SceneStateThresholds()
        self._device     = device
        self._window:  list = []
        self._window_size = model.window_size

    def reset(self) -> None:
        self._window.clear()

    def step(self, features: "np.ndarray") -> SceneState:
        """Process one frame of telemetry and return current SceneState.

        Args:
            features: (28,) float32 telemetry vector from EvidenceExtractor.

        Returns:
            SceneState for the current frame.
        """
        import numpy as np

        self._window.append(features.astype(np.float32))
        if len(self._window) > self._window_size:
            self._window.pop(0)

        # Left-pad with zeros if window not full yet (same as controller.py)
        pad = self._window_size - len(self._window)
        hist = [np.zeros(self._model.input_dim, dtype=np.float32)] * pad + self._window
        x = torch.tensor(np.stack(hist), dtype=torch.float32).unsqueeze(0).to(self._device)

        probs = self._model.predict_probs_numpy(x)
        return classify_scene(probs, self._thresholds)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        thresholds: SceneStateThresholds | None = None,
        device: str = "cpu",
    ) -> "SceneStateClassifier":
        model = SceneStateNet.load(path, device=device)
        return cls(model, thresholds=thresholds, device=device)

    @property
    def thresholds(self) -> SceneStateThresholds:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value: SceneStateThresholds) -> None:
        self._thresholds = value


__all__ = [
    "SceneState",
    "SceneStateNet",
    "SceneStateClassifier",
    "SceneStateThresholds",
    "classify_scene",
    "HEAD_NAMES",
    "HEAD_BASE_RATES",
    "HEAD_LABEL_KEYS",
]
