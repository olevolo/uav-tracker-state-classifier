"""Torch Dataset — composite labels (localization + confidence + aux)."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from csc_lib.csc.config import CSCFeatureConfig
from csc_lib.csc.features import FEATURE_DIM, build_sequence_features
from csc_lib.csc.labeling.label_schema import (
    AUX_FLAGS,
    ConfidenceState,
    DerivedState,
    LocalizationState,
)


@dataclass
class CSCWindow:
    features: np.ndarray
    localization: np.ndarray         # (T,) int
    confidence: np.ndarray           # (T,) int
    aux: np.ndarray                  # (T, n_aux)
    derived: np.ndarray              # (T,) int (for evaluation)
    # ---- V3 proactive forecast targets (zero when not present in labels) ----
    failure_next_10: np.ndarray      # (T,) int 0/1
    false_confirmed_next_10: np.ndarray  # (T,) int 0/1
    lost_aware_next_10: np.ndarray   # (T,) int 0/1
    ignore_forecast: np.ndarray      # (T,) int 0/1 (1 = exclude from forecast loss)
    sequence: str
    dataset: str


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_sequence(rows: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["sequence"])
        groups[key].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r["frame_idx"])
    return groups


def split_sequences_train_val(
    sequence_keys: list[tuple[str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    keys = sorted(sequence_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(val_fraction * len(keys))))
    return keys[n_val:], keys[:n_val]


class CSCDataset(Dataset):
    def __init__(
        self,
        sequence_rows: dict[tuple[str, str], list[dict]],
        feature_cfg: CSCFeatureConfig,
        *,
        image_size: tuple[int, int] = (1280, 720),
        stride: int = 1,
    ) -> None:
        self.feature_cfg = feature_cfg
        self.window_size = feature_cfg.window_size
        self.stride = stride
        self.image_size = image_size

        self.windows: list[CSCWindow] = []
        for (dataset, sequence), rows in sequence_rows.items():
            feats = build_sequence_features(rows, image_size, cfg=feature_cfg)
            loc = np.array([r.get("localization_state", 0) for r in rows], dtype=np.int64)
            conf = np.array([r.get("confidence_state", 0) for r in rows], dtype=np.int64)
            derived = np.array([r.get("derived_state", 0) for r in rows], dtype=np.int64)
            aux = np.zeros((len(rows), len(AUX_FLAGS)), dtype=np.float32)
            for j, name in enumerate(AUX_FLAGS):
                aux[:, j] = np.array(
                    [bool((r.get("aux") or {}).get(name, False)) for r in rows],
                    dtype=np.float32,
                )

            # V3 forecast targets — default to 0 / ignore=1 when absent (V2 labels).
            fail_n10 = np.array(
                [int(r.get("failure_next_10", 0)) for r in rows], dtype=np.int64
            )
            fc_n10 = np.array(
                [int(r.get("false_confirmed_next_10", 0)) for r in rows], dtype=np.int64
            )
            lost_n10 = np.array(
                [int(r.get("lost_aware_next_10", 0)) for r in rows], dtype=np.int64
            )
            # If forecast keys are missing entirely, mark all frames as ignore=1
            # so the training loss does not penalise V2 labels for forecasts.
            has_forecast = any("failure_next_10" in r for r in rows)
            if has_forecast:
                ignore_fc = np.array(
                    [int(r.get("ignore_forecast", 0)) for r in rows], dtype=np.int64
                )
            else:
                ignore_fc = np.ones(len(rows), dtype=np.int64)

            T = len(rows)
            for end in range(self.window_size, T + 1, stride):
                start = end - self.window_size
                self.windows.append(
                    CSCWindow(
                        features=feats[start:end],
                        localization=loc[start:end],
                        confidence=conf[start:end],
                        aux=aux[start:end],
                        derived=derived[start:end],
                        failure_next_10=fail_n10[start:end],
                        false_confirmed_next_10=fc_n10[start:end],
                        lost_aware_next_10=lost_n10[start:end],
                        ignore_forecast=ignore_fc[start:end],
                        sequence=sequence,
                        dataset=dataset,
                    )
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        w = self.windows[idx]
        return {
            "features": torch.from_numpy(w.features),
            "localization": torch.from_numpy(w.localization),
            "confidence": torch.from_numpy(w.confidence),
            "aux": torch.from_numpy(w.aux),
            "derived": torch.from_numpy(w.derived),
            "failure_next_10": torch.from_numpy(w.failure_next_10),
            "false_confirmed_next_10": torch.from_numpy(w.false_confirmed_next_10),
            "lost_aware_next_10": torch.from_numpy(w.lost_aware_next_10),
            "ignore_forecast": torch.from_numpy(w.ignore_forecast),
        }

    def make_sampler(self) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler that up-samples rare localization classes.

        The "dominant class" of a window is the most frequent localization
        label across all T frames in that window.  Windows ending in
        STABLE-after-LOST are thus still treated as LOST-dominant if the
        majority of the window frames are LOST.

        Weights are ``1 / class_count[dominant_class]`` (inverse frequency),
        normalised so the expected number of draws per epoch equals
        ``len(self.windows)``.
        """
        n_loc_classes = len(LocalizationState)
        class_counts: list[int] = [0] * n_loc_classes
        dominant: list[int] = []

        for w in self.windows:
            # Most-frequent localization label in the window
            counts = np.bincount(w.localization, minlength=n_loc_classes)
            dom = int(counts.argmax())
            dominant.append(dom)
            class_counts[dom] += 1

        weights: list[float] = []
        for dom in dominant:
            cnt = class_counts[dom]
            weights.append(1.0 / cnt if cnt > 0 else 0.0)

        weight_tensor = torch.tensor(weights, dtype=torch.float64)
        return WeightedRandomSampler(
            weights=weight_tensor,
            num_samples=len(self.windows),
            replacement=True,
        )


def load_labels_dir(labels_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for jsonl in sorted(Path(labels_dir).rglob("labels.jsonl")):
        rows.extend(_load_jsonl(jsonl))
    return rows


def build_train_val_datasets(
    labels_dir: Path,
    feature_cfg: CSCFeatureConfig,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
    image_size: tuple[int, int] = (1280, 720),
    return_sampler: bool = False,
) -> tuple:
    """Build train/val datasets from a ``labels_dir``.

    Returns
    -------
    If ``return_sampler=False`` (default, backward-compatible):
        ``(train_ds, val_ds, info)``
    If ``return_sampler=True``:
        ``(train_ds, val_ds, info, sampler)``
        where ``sampler`` is a :class:`WeightedRandomSampler` for
        the training dataset, suitable for use in a DataLoader.
    """
    rows = load_labels_dir(labels_dir)
    if not rows:
        raise FileNotFoundError(f"no labels.jsonl found under {labels_dir}")
    groups = _group_by_sequence(rows)
    train_keys, val_keys = split_sequences_train_val(
        list(groups.keys()), val_fraction, seed
    )
    train_rows = {k: groups[k] for k in train_keys}
    val_rows = {k: groups[k] for k in val_keys}

    train_ds = CSCDataset(train_rows, feature_cfg, image_size=image_size)
    val_ds = CSCDataset(val_rows, feature_cfg, image_size=image_size)
    info = {
        "n_train_sequences": len(train_keys),
        "n_val_sequences": len(val_keys),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "train_sequences": [list(k) for k in train_keys],
        "val_sequences": [list(k) for k in val_keys],
    }
    if return_sampler:
        sampler = train_ds.make_sampler()
        return train_ds, val_ds, info, sampler
    return train_ds, val_ds, info


__all__ = [
    "CSCDataset",
    "CSCWindow",
    "_group_by_sequence",
    "build_train_val_datasets",
    "load_labels_dir",
    "split_sequences_train_val",
]
