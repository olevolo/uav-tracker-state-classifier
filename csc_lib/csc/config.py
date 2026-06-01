"""CSC model + training configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional


@dataclass
class CSCFeatureConfig:
    """Selection + normalisation of causal features."""

    use_telemetry: bool = True       # confidence, APCE, PSR, response_max
    use_bbox_dynamics: bool = True   # cx, cy, w, h, area, aspect, vel, acc
    use_geometry_normalised: bool = True  # divide by image w/h
    window_size: int = 20            # temporal window of past frames

    # Normalisation: clip extreme outliers to keep gradients stable.
    clip_value: float = 8.0

    # Feature builder version: "v1" (default) or "v2" (scale-context features).
    # When "v2", training pipeline uses csc_lib.csc.features.build_sequence_features_v2.
    feature_version: str = "v1"


@dataclass
class TCNConfig:
    """Hyper-parameters for CSCTCN."""

    kernel_size: int = 3
    num_layers: int = 4
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    hidden_dim: int = 64
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TCNConfig":
        return cls(
            kernel_size=d.get("kernel_size", 3),
            num_layers=d.get("num_layers", 4),
            dilations=list(d.get("dilations", [1, 2, 4, 8])),
            hidden_dim=d.get("hidden_dim", 64),
            dropout=d.get("dropout", 0.1),
        )


@dataclass
class CSCModelConfig:
    feature_dim: int = 11          # auto-overridden by builder
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False    # must be False for causal inference
    n_states: int = 4              # CORRECT_CONFIRMED / CORRECT_UNCERTAIN / LOST_AWARE / FALSE_CONFIRMED
    kind: str = "gru"              # "gru" | "mlp" | "tcn"
    tcn: Optional[TCNConfig] = None  # TCN-specific config; used only when kind="tcn"
    # ----- V3 proactive forecast heads ------------------------------
    enable_forecast_heads: bool = False
    forecast_horizon: int = 10


@dataclass
class CSCLossConfig:
    state_weights: list[float] = field(
        default_factory=lambda: [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
    )  # CONFIRMED, UNCERTAIN, OCCLUDED, LOST, DISTRACTOR, FALSE_CONFIRMED
    risk_weight: float = 1.0
    aux_weight: float = 0.3
    use_focal: bool = False
    focal_gamma: float = 2.0
    # ----- V3 forecast head loss weights ----------------------------
    forecast_failure_weight: float = 0.5
    forecast_fc_weight: float = 0.8
    forecast_lost_weight: float = 0.6


@dataclass
class CSCOptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    epochs: int = 30
    batch_size: int = 64
    early_stopping_patience: int = 6
    grad_clip: float = 1.0
    use_balanced_sampler: bool = True
    scheduler: str = "none"        # "none" | "cosine" — per-epoch LR decay
    min_lr_ratio: float = 0.02     # cosine eta_min = lr * min_lr_ratio


@dataclass
class CSCTrainConfig:
    seed: int = 42
    device: str = "cuda"
    feature: CSCFeatureConfig = field(default_factory=CSCFeatureConfig)
    model: CSCModelConfig = field(default_factory=CSCModelConfig)
    loss: CSCLossConfig = field(default_factory=CSCLossConfig)
    optim: CSCOptimConfig = field(default_factory=CSCOptimConfig)

    labels_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    val_fraction: float = 0.15
    stratified_split: bool = False

    # Sampler config (train2_v2+)
    sampler_type: str = "derived_wrs"   # "derived_wrs" | "fc_source_balanced"
    fc_per_batch: int = 20
    lasot_fc_cap: float = 0.60
    aerial_fc_floor: float = 0.30
    # v3fix Run 1 hard-negative pools (drone CC with high scale = anti-shortcut)
    aerial_non_fc_floor: int = 16        # batch slots for drone non-FC (low scale)
    hard_scale_cc_floor: int = 8         # batch slots for drone non-FC with high log_area_ratio
    scale_pct_threshold: float = 75.0    # percentile cutoff for "hard scale" CC

    # Two-stage training (Stage 2 = forecast-only fine-tune on frozen Stage-1 encoder)
    training_stage: int = 1              # 1 = joint, 2 = forecast heads only
    stage1_checkpoint: Optional[Path] = None  # path to Stage-1 .pth; required when stage=2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CSCTrainConfig":
        feature = CSCFeatureConfig(**d.get("feature", {}))

        # Parse model config — handle TCNConfig sub-block separately
        model_d = dict(d.get("model", {}))
        tcn_d = model_d.pop("tcn", None)
        model = CSCModelConfig(**model_d)
        if tcn_d is not None:
            model.tcn = TCNConfig.from_dict(tcn_d)

        # Parse loss config — allow extra keys gracefully
        loss_d = d.get("loss", {})
        loss = CSCLossConfig(
            state_weights=loss_d.get("state_weights", [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]),
            risk_weight=loss_d.get("risk_weight", 1.0),
            aux_weight=loss_d.get("aux_weight", 0.3),
            use_focal=loss_d.get("use_focal", False),
            focal_gamma=loss_d.get("focal_gamma", 2.0),
            forecast_failure_weight=loss_d.get("forecast_failure_weight", 0.5),
            forecast_fc_weight=loss_d.get("forecast_fc_weight", 0.8),
            forecast_lost_weight=loss_d.get("forecast_lost_weight", 0.6),
        )

        # Parse optim config — allow extra keys gracefully
        optim_d = d.get("optim", {})
        optim = CSCOptimConfig(
            lr=optim_d.get("lr", 1e-3),
            weight_decay=optim_d.get("weight_decay", 1e-4),
            optimizer=optim_d.get("optimizer", "adamw"),
            epochs=optim_d.get("epochs", 30),
            batch_size=optim_d.get("batch_size", 64),
            early_stopping_patience=optim_d.get("early_stopping_patience", 6),
            grad_clip=optim_d.get("grad_clip", 1.0),
            use_balanced_sampler=optim_d.get("use_balanced_sampler", True),
            scheduler=optim_d.get("scheduler", "none"),
            min_lr_ratio=optim_d.get("min_lr_ratio", 0.02),
        )

        return cls(
            seed=d.get("seed", 42),
            device=d.get("device", "cuda"),
            feature=feature,
            model=model,
            loss=loss,
            optim=optim,
            labels_dir=Path(d["labels_dir"]) if d.get("labels_dir") else None,
            output_dir=Path(d["output_dir"]) if d.get("output_dir") else None,
            val_fraction=d.get("val_fraction", 0.15),
            stratified_split=d.get("stratified_split", False),
            sampler_type=d.get("sampler_type", "derived_wrs"),
            fc_per_batch=d.get("fc_per_batch", 20),
            lasot_fc_cap=d.get("lasot_fc_cap", 0.60),
            aerial_fc_floor=d.get("aerial_fc_floor", 0.30),
            aerial_non_fc_floor=d.get("aerial_non_fc_floor", 16),
            hard_scale_cc_floor=d.get("hard_scale_cc_floor", 8),
            scale_pct_threshold=d.get("scale_pct_threshold", 75.0),
            training_stage=d.get("training_stage", 1),
            stage1_checkpoint=Path(d["stage1_checkpoint"]) if d.get("stage1_checkpoint") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        out = asdict(self)
        out["labels_dir"] = str(self.labels_dir) if self.labels_dir else None
        out["output_dir"] = str(self.output_dir) if self.output_dir else None
        out["stage1_checkpoint"] = str(self.stage1_checkpoint) if self.stage1_checkpoint else None
        return out
