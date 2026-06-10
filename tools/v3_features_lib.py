"""V3-parity feature/label/split builder for the CondFactorized rematch.

This module is a THIN, reusable wrapper around V3's *exact* training pipeline.
It does NOT reimplement features — it calls the same ``CSCDataset`` (which
dispatches to ``build_sequence_features_v2``) and the same
``split_sequences_stratified`` that produced the frozen V3-prod model
(``outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2``).  An integrity
assert below confirms the reproduced split equals that run's
``split_info.json``, exactly as ``tools/v3_fc_heldout_repro.py`` does.

-------------------------------------------------------------------------------
PARITY NOTE — the active 16-dim ``FEATURE_NAMES_V2`` vs. the run's stale
``label_mapping.json`` feature_names
-------------------------------------------------------------------------------
The V3 run dir's ``label_mapping.json`` lists feature_names ending in
``... aspect_ratio, ..., log_w_ratio_to_init, ..., log_h_ratio_to_init,
conf_ema_trend`` (positions 8/11/14/15).  The ACTIVE builder
``csc_lib.csc.features.build_sequence_features_v2`` instead emits
``... log_aspect_ratio, ..., edge_pressure_score, ..., scale_smoothness_8,
aspect_instability_8`` at those positions.  The ``label_mapping.json`` list is
STALE — it predates a feature refactor.  Proof: running V3-prod over its own
held-out val with the ACTIVE builder reproduces its reported metrics EXACTLY
(FC-F1 0.7850 == reported 0.7850, derived macro-F1 0.6889 == reported 0.6889;
see ``tools/v3_fc_heldout_repro.py``).  Therefore the ACTIVE 16 features ARE
what V3-prod was trained on, and using them IS V3-feature parity.

GEOM / RESP partition (the task's stated semantic rule: geometry/localization
-> off-target tower; response/confidence -> confirmed towers), mapped onto the
ACTIVE 16 features:

    RESP  (confirmed towers): confidence, apce, psr                       -> resp_dim=3
    GEOM  (off-target tower): cx_norm, cy_norm, w_norm, h_norm, area_norm,
        log_aspect_ratio, velocity_norm, edge_contact_score,
        edge_pressure_score, log_area_ratio_to_init, motion_angle_change,
        scale_smoothness_8, aspect_instability_8                          -> geom_dim=13

This is GEOM=13 / RESP=3, NOT the task's nominal GEOM=12 / RESP=4.  The single
difference is position 15: the task expected ``conf_ema_trend`` (a confidence
feature -> RESP) but the active builder has ``aspect_instability_8`` (a
geometry feature -> GEOM).  Every other feature lands in the tower the task
intended.  We honour the task's SEMANTIC partition rule on V3's REAL features
rather than fabricate a ``conf_ema_trend`` channel the builder does not emit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from csc_lib.csc.config import CSCTrainConfig  # noqa: E402
from csc_lib.csc.dataset import (  # noqa: E402
    CSCDataset,
    _group_by_sequence,
    load_labels_dir,
    split_sequences_stratified,
)
from csc_lib.csc.features import FEATURE_NAMES_V2  # noqa: E402
from csc_lib.csc.labeling.label_schema import DerivedState  # noqa: E402

# --- canonical V3-prod run (frozen) -----------------------------------------
V3_RUN_DIR = PROJECT_ROOT / "outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2"
V3_CONFIG = V3_RUN_DIR / "config_resolved.yaml"
V3_SPLIT_INFO = V3_RUN_DIR / "split_info.json"
LABELS_DIR = PROJECT_ROOT / "outputs/csc_labels/sglatrack/v3fix_combined"
IMAGE_SIZE = (1280, 720)  # CSCDataset training default

# --- derived-state ids ------------------------------------------------------
CC = int(DerivedState.CORRECT_CONFIRMED)  # 0
CU = int(DerivedState.CORRECT_UNCERTAIN)  # 1
LA = int(DerivedState.LOST_AWARE)         # 2
FC = int(DerivedState.FALSE_CONFIRMED)    # 3

# --- GEOM / RESP partition by feature NAME (robust to index drift) ----------
RESP_FEATURES: tuple[str, ...] = ("confidence", "apce", "psr")
# GEOM = everything else, in FEATURE_NAMES_V2 order.
GEOM_FEATURES: tuple[str, ...] = tuple(
    n for n in FEATURE_NAMES_V2 if n not in RESP_FEATURES
)
_NAME_TO_IDX = {n: i for i, n in enumerate(FEATURE_NAMES_V2)}
GEOM_IDX: list[int] = [_NAME_TO_IDX[n] for n in GEOM_FEATURES]
RESP_IDX: list[int] = [_NAME_TO_IDX[n] for n in RESP_FEATURES]
GEOM_DIM = len(GEOM_IDX)   # 13
RESP_DIM = len(RESP_IDX)   # 3


def load_v3_config() -> CSCTrainConfig:
    import yaml

    with open(V3_CONFIG) as fh:
        d = yaml.safe_load(fh)
    cfg = CSCTrainConfig.from_dict(d)
    assert cfg.feature.feature_version == "v2", cfg.feature.feature_version
    assert cfg.feature.window_size == 32, cfg.feature.window_size
    return cfg


def load_groups() -> dict[tuple[str, str], list[dict]]:
    rows = load_labels_dir(LABELS_DIR)
    if not rows:
        raise FileNotFoundError(f"no labels under {LABELS_DIR}")
    return _group_by_sequence(rows)


def reproduce_v3_split(
    groups: dict[tuple[str, str], list[dict]],
    val_fraction: float,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Reproduce V3's EXACT stratified split and ASSERT identity w/ split_info.json."""
    train_keys, val_keys = split_sequences_stratified(
        list(groups.keys()), groups, val_fraction
    )
    si = json.loads(V3_SPLIT_INFO.read_text())
    si_val = {tuple(x) for x in si["val_sequences"]}
    si_train = {tuple(x) for x in si["train_sequences"]}
    assert set(val_keys) == si_val, "VAL split != split_info.json"
    assert set(train_keys) == si_train, "TRAIN split != split_info.json"
    return train_keys, val_keys


def build_windows(
    seq_rows: dict[tuple[str, str], list[dict]],
    feature_cfg,
) -> list:
    """Return the EXACT CSCDataset windows (16-dim v2 features, window 32).

    Each window is a full ``window_size``-frame causal slice; the last step is
    the supervised target (matches runtime ``last_step_only`` behaviour).  This
    is stricter parity than left-padding: it is the identical windowing V3 used.
    """
    ds = CSCDataset(seq_rows, feature_cfg, image_size=IMAGE_SIZE)
    return ds.windows


def split_features(feats16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a (T,16) v2 feature block into (geom (T,13), resp (T,3))."""
    return feats16[:, GEOM_IDX], feats16[:, RESP_IDX]


def derived_to_targets(derived: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame conditional targets from derived 4-state.

    y_off  = 1 if derived in {LA, FC}        (off-target)
    y_conf = 1 if derived in {CC, FC}        (high-confidence / "confirmed")
    """
    d = np.asarray(derived, dtype=np.int64)
    y_off = np.isin(d, (LA, FC)).astype(np.int64)
    y_conf = np.isin(d, (CC, FC)).astype(np.int64)
    return y_off, y_conf


__all__ = [
    "PROJECT_ROOT", "V3_RUN_DIR", "LABELS_DIR", "IMAGE_SIZE",
    "CC", "CU", "LA", "FC",
    "RESP_FEATURES", "GEOM_FEATURES", "GEOM_IDX", "RESP_IDX",
    "GEOM_DIM", "RESP_DIM",
    "load_v3_config", "load_groups", "reproduce_v3_split",
    "build_windows", "split_features", "derived_to_targets",
]
