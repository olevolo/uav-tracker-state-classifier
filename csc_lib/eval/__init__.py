"""CSC evaluation public API — re-exports from all eval sub-modules."""

from csc_lib.eval.calibration import (
    brier_score,
    calibrated_distribution_stats,
    confidence_entropy,
    expected_calibration_error,
    pct_calibrated_high,
    pct_calibrated_low,
)
from csc_lib.eval.classification import (
    balanced_accuracy,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    per_state_prf,
    weighted_f1,
)
from csc_lib.eval.feature_quality import (
    feature_auprc,
    feature_auroc,
    feature_cohens_d,
    feature_group_ablation,
    feature_missing_rate,
    feature_per_state_stats,
)
from csc_lib.eval.state import (
    lost_rate,
    mean_time_in_state,
    state_persistence,
    state_rates,
)

__all__ = [
    "balanced_accuracy",
    "brier_score",
    "calibrated_distribution_stats",
    "confidence_entropy",
    "confusion_matrix",
    "expected_calibration_error",
    "feature_auprc",
    "feature_auroc",
    "feature_cohens_d",
    "feature_group_ablation",
    "feature_missing_rate",
    "feature_per_state_stats",
    "lost_rate",
    "macro_f1",
    "mean_time_in_state",
    "pct_calibrated_high",
    "pct_calibrated_low",
    "per_class_metrics",
    "per_state_prf",
    "state_persistence",
    "state_rates",
    "weighted_f1",
]
