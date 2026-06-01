"""Torch Dataset — composite labels (localization + confidence + aux)."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from csc_lib.csc.config import CSCFeatureConfig
from csc_lib.csc.features import (
    FEATURE_DIM,
    FEATURE_DIM_V2,
    FEATURE_DIM_V3,
    build_sequence_features,
    build_sequence_features_v2,
    build_sequence_features_v3,
)
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


def split_sequences_stratified(
    sequence_keys: list[tuple[str, str]],
    sequence_rows: dict[tuple[str, str], list[dict]],
    val_fraction: float,
    min_fc_val_seqs: int = 5,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministic hash-based stratified split with guaranteed FC sequences in val.

    Uses MD5 of the sequence key (no seed) for reproducibility across runs.
    Ensures at least *min_fc_val_seqs* FALSE_CONFIRMED sequences in the val split.
    """
    fc_state = int(DerivedState.FALSE_CONFIRMED)

    fc_keys: list[tuple[str, str]] = []
    non_fc_keys: list[tuple[str, str]] = []
    for k in sorted(sequence_keys):
        rows = sequence_rows.get(k, [])
        if any(r.get("derived_state", 0) == fc_state for r in rows):
            fc_keys.append(k)
        else:
            non_fc_keys.append(k)

    def _in_val(k: tuple[str, str]) -> bool:
        h = int(hashlib.md5(str(k).encode()).hexdigest(), 16)
        return (h % 1000) < int(val_fraction * 1000)

    fc_val   = [k for k in fc_keys if _in_val(k)]
    fc_train = [k for k in fc_keys if not _in_val(k)]

    # Guarantee minimum FC coverage in val by moving from train if needed
    while len(fc_val) < min_fc_val_seqs and fc_train:
        fc_val.append(fc_train.pop())

    non_fc_val   = [k for k in non_fc_keys if _in_val(k)]
    non_fc_train = [k for k in non_fc_keys if not _in_val(k)]

    return fc_train + non_fc_train, fc_val + non_fc_val


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
        # Dispatch by feature_version: v1 → build_sequence_features, v2 → build_sequence_features_v2.
        # Was a silent no-op for months (CRITICAL BUG fixed 2026-05-30): training always used V1
        # regardless of `feature_version: v2` in YAML.
        feat_ver = (getattr(feature_cfg, "feature_version", "v1") or "v1").lower()
        if feat_ver == "v2":
            _build_features = build_sequence_features_v2
        elif feat_ver == "v3":
            # V3 = V2 (16) + 7 response-structure passthroughs (response_entropy,
            # sm_*). build_sequence_features_v3 reads them from each row via extra=r,
            # so the labels must carry those fields (see tools/join_v3_features_to_labels.py).
            _build_features = build_sequence_features_v3
        elif feat_ver in ("v1", ""):
            _build_features = build_sequence_features
        else:
            raise ValueError(f"unknown feature_version={feat_ver!r}; expected 'v1', 'v2', or 'v3'")
        for (dataset, sequence), rows in sequence_rows.items():
            feats = _build_features(rows, image_size, cfg=feature_cfg)
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
        """WeightedRandomSampler by dominant DerivedState (CC/CU/LA/FC).

        FC-dominant windows get their own weight bucket, separate from
        LA-dominant windows.  Previously this used LocalizationState which
        could not distinguish FC from LA (both are LOST-dominant).
        """
        n_der = len(DerivedState)
        class_counts: list[int] = [0] * n_der
        dominant: list[int] = []

        for w in self.windows:
            counts = np.bincount(w.derived, minlength=n_der)
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


class FCSourceBalancedSampler(Sampler):
    """Batch sampler that balances FC sources to prevent LaSOT FC dominance.

    Problem: LaSOT contributes ~93% of FC windows; model learns
    "LaSOT-style FC" (low-conf failure) and ignores aerial FC patterns
    (high-APCE confident wrong localization).

    Solution: per-batch FC quota:
        - LaSOT FC:  ≤ lasot_fc_cap  (default 60%)
        - Aerial FC: ≥ aerial_fc_floor (default 30%)
        - Non-FC:    remaining batch slots (any source)

    This gives aerial FC ~5× over-representation vs. their natural 6-7%
    frequency, matching the effect of UAVDT×5 oversampling but without
    duplicating labels.

    Parameters
    ----------
    dataset:        The CSCDataset whose windows are indexed.
    batch_size:     Total windows per batch.
    n_steps:        Steps per epoch (default: len/batch_size).
    fc_per_batch:   FC windows per batch (default 20, ≈31% of 64).
    lasot_fc_cap:   Max fraction of fc_per_batch from LaSOT  (default 0.60).
    aerial_fc_floor: Min fraction of fc_per_batch from aerial (default 0.30).
    aerial_datasets: Dataset names considered "aerial" for the quota.
    """

    AERIAL_DEFAULT = frozenset({"uavdt_sot", "visdrone_sot", "uavtrack112", "dtb70"})

    # Index of log_area_ratio_to_init in FEATURE_NAMES (see csc_lib/csc/features.py).
    LOG_AREA_RATIO_IDX = 12

    def __init__(
        self,
        dataset: "CSCDataset",
        batch_size: int = 64,
        n_steps: Optional[int] = None,
        fc_per_batch: int = 20,
        lasot_fc_cap: float = 0.60,
        aerial_fc_floor: float = 0.30,
        aerial_non_fc_floor: int = 16,
        hard_scale_cc_floor: int = 8,
        scale_pct_threshold: float = 75.0,
        aerial_datasets: frozenset = AERIAL_DEFAULT,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.fc_per_batch = fc_per_batch
        self.aerial_non_fc_floor = aerial_non_fc_floor
        self.hard_scale_cc_floor = hard_scale_cc_floor
        self.n_steps = n_steps or max(1, len(dataset.windows) // batch_size)

        fc_state = int(DerivedState.FALSE_CONFIRMED)

        # Pre-compute FC labels and log_area_ratio per window (mean over second half).
        is_fc_list: list[bool] = []
        log_area_ratios: list[float] = []
        for w in dataset.windows:
            last_half = w.derived[len(w.derived) // 2:]
            is_fc_list.append(any(int(d) == fc_state for d in last_half))
            half = len(w.features) // 2
            ratio = float(np.mean(w.features[half:, self.LOG_AREA_RATIO_IDX]))
            log_area_ratios.append(ratio)

        # Threshold from aerial non-FC distribution: windows above the p75
        # cutoff are "hard scale CC" — natural bbox growth that shouldn't be
        # mistaken for FC at runtime.
        aerial_non_fc_ratios = [
            log_area_ratios[i]
            for i, w in enumerate(dataset.windows)
            if (not is_fc_list[i]) and w.dataset in aerial_datasets
        ]
        if aerial_non_fc_ratios:
            scale_threshold = float(
                np.percentile(aerial_non_fc_ratios, scale_pct_threshold)
            )
        else:
            scale_threshold = float("inf")
        self._scale_threshold = scale_threshold

        # Cache feature[12] (log_area_ratio_to_init) per window for per-batch
        # logging.  Features live in memory on each CSCWindow, so this is
        # essentially free; we use NaN whenever a window is degenerate.
        cached_log_area = np.full(len(dataset.windows), np.nan, dtype=np.float64)
        for i, w in enumerate(dataset.windows):
            try:
                if w.features.size and w.features.shape[1] > self.LOG_AREA_RATIO_IDX:
                    cached_log_area[i] = float(
                        np.mean(w.features[:, self.LOG_AREA_RATIO_IDX])
                    )
            except Exception:
                cached_log_area[i] = np.nan
        self._cached_log_area = cached_log_area

        # Track per-window dataset name (used for non-FC source distribution).
        self._window_dataset = [w.dataset for w in dataset.windows]

        # Partition windows into 6 pools
        self._lasot_fc: list[int] = []
        self._aerial_fc: list[int] = []
        self._other_fc: list[int] = []    # FC from non-LaSOT non-aerial (rare)
        self._hard_scale_cc: list[int] = []
        self._aerial_non_fc_normal: list[int] = []
        self._non_fc:   list[int] = []

        for i, w in enumerate(dataset.windows):
            ds = w.dataset
            if is_fc_list[i]:
                if ds == "lasot":
                    self._lasot_fc.append(i)
                elif ds in aerial_datasets:
                    self._aerial_fc.append(i)
                else:
                    self._other_fc.append(i)
            else:
                if ds in aerial_datasets:
                    if log_area_ratios[i] > scale_threshold:
                        self._hard_scale_cc.append(i)
                    else:
                        self._aerial_non_fc_normal.append(i)
                else:
                    self._non_fc.append(i)

        import logging as _log
        _log.getLogger(__name__).info(
            "FCSourceBalancedSampler: LaSOT_FC=%d  Aerial_FC=%d  Other_FC=%d  "
            "HardScaleCC=%d  AerialNonFCNormal=%d  Non_FC=%d  scale_threshold=%.4f",
            len(self._lasot_fc), len(self._aerial_fc), len(self._other_fc),
            len(self._hard_scale_cc), len(self._aerial_non_fc_normal),
            len(self._non_fc), scale_threshold,
        )
        self._lasot_fc_cap = int(fc_per_batch * lasot_fc_cap)
        self._aerial_fc_floor = int(fc_per_batch * aerial_fc_floor)

        # Overall non-FC mean(log_area_ratio_to_init) — reference for the
        # "hard pool sanity check" logged at the end of every epoch.
        non_fc_all_idx = (
            self._hard_scale_cc + self._aerial_non_fc_normal + self._non_fc
        )
        if non_fc_all_idx:
            vals = self._cached_log_area[non_fc_all_idx]
            vals = vals[~np.isnan(vals)]
            self._overall_non_fc_log_area = float(vals.mean()) if vals.size else float("nan")
        else:
            self._overall_non_fc_log_area = float("nan")

        # Per-epoch logging counters — reset at the start of every __iter__.
        self._epoch_stats: dict = self._reset_epoch_stats()

    def _reset_epoch_stats(self) -> dict:
        return {
            "n_batches": 0,
            "pool_counts": {
                "lasot_fc": 0,
                "aerial_fc": 0,
                "other_fc": 0,
                "hard_scale_cc": 0,
                "aerial_non_fc_normal": 0,
                "non_fc": 0,
            },
            "hard_scale_cc_log_area": [],
            "non_fc_log_area": [],
            "non_fc_dataset_counts": defaultdict(int),
        }

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self):
        rng = np.random.default_rng()

        # Reset per-epoch logging counters at the start of every epoch.
        self._epoch_stats = self._reset_epoch_stats()

        def _pool(lst):
            return rng.permutation(lst).tolist()

        pools = {
            "lasot_fc":              _pool(self._lasot_fc),
            "aerial_fc":             _pool(self._aerial_fc),
            "other_fc":              _pool(self._other_fc),
            "hard_scale_cc":         _pool(self._hard_scale_cc),
            "aerial_non_fc_normal":  _pool(self._aerial_non_fc_normal),
            "non_fc":                _pool(self._non_fc),
        }
        ptrs = {k: 0 for k in pools}

        def _draw(key: str, n: int) -> list[int]:
            if n <= 0 or not pools[key]:
                return []
            pool = pools[key]
            ptr = ptrs[key]
            out: list[int] = []
            while len(out) < n:
                chunk = pool[ptr: ptr + (n - len(out))]
                out.extend(chunk)
                ptr += len(chunk)
                if ptr >= len(pool):
                    pools[key] = rng.permutation(pool).tolist()
                    pool = pools[key]
                    ptr = 0
            ptrs[key] = ptr
            return out[:n]

        for _ in range(self.n_steps):
            # Step 1: aerial FC quota (floor) — fall back to other FC
            aerial_fc = _draw("aerial_fc", self._aerial_fc_floor)
            n_aerial_fc_real = len(aerial_fc)
            if len(aerial_fc) < self._aerial_fc_floor:
                aerial_fc += _draw(
                    "other_fc", self._aerial_fc_floor - len(aerial_fc)
                )
            n_other_fc_from_aerial = len(aerial_fc) - n_aerial_fc_real

            # Step 2: remaining FC (LaSOT cap)
            n_remaining_fc = self.fc_per_batch - len(aerial_fc)
            n_lasot = min(n_remaining_fc, self._lasot_fc_cap)
            lasot = _draw("lasot_fc", n_lasot)
            extra_fc = _draw("other_fc", max(0, n_remaining_fc - len(lasot)))

            # Step 3: hard_scale_cc — natural bbox growth on aerial sequences
            hard_cc = _draw("hard_scale_cc", self.hard_scale_cc_floor)
            n_hard_cc_real = len(hard_cc)
            if len(hard_cc) < self.hard_scale_cc_floor:
                hard_cc += _draw(
                    "aerial_non_fc_normal",
                    self.hard_scale_cc_floor - len(hard_cc),
                )
            n_aerial_normal_from_hard = len(hard_cc) - n_hard_cc_real

            # Step 4: aerial_non_fc_normal floor
            aerial_normal = _draw(
                "aerial_non_fc_normal", self.aerial_non_fc_floor
            )
            n_aerial_normal_real = len(aerial_normal)
            if len(aerial_normal) < self.aerial_non_fc_floor:
                aerial_normal += _draw(
                    "non_fc",
                    self.aerial_non_fc_floor - len(aerial_normal),
                )
            n_non_fc_from_aerial = len(aerial_normal) - n_aerial_normal_real

            # Step 5: fill remainder with non_fc; fall back to aerial_non_fc_normal
            n_used = (
                len(aerial_fc) + len(lasot) + len(extra_fc)
                + len(hard_cc) + len(aerial_normal)
            )
            n_non_fc = self.batch_size - n_used
            non_fc = _draw("non_fc", n_non_fc)
            n_non_fc_real = len(non_fc)
            if len(non_fc) < n_non_fc:
                non_fc += _draw(
                    "aerial_non_fc_normal", n_non_fc - len(non_fc)
                )
            n_aerial_normal_from_non_fc = len(non_fc) - n_non_fc_real

            # ---- Update per-epoch statistics ----
            stats = self._epoch_stats
            pc = stats["pool_counts"]
            pc["aerial_fc"]            += n_aerial_fc_real
            pc["other_fc"]             += n_other_fc_from_aerial + len(extra_fc)
            pc["lasot_fc"]             += len(lasot)
            pc["hard_scale_cc"]        += n_hard_cc_real
            pc["aerial_non_fc_normal"] += (
                n_aerial_normal_from_hard
                + n_aerial_normal_real
                + n_aerial_normal_from_non_fc
            )
            pc["non_fc"]               += n_non_fc_real + n_non_fc_from_aerial
            stats["n_batches"]         += 1

            # Collect log_area_ratio values for hard_scale_cc + non_fc draws.
            for idx in hard_cc[:n_hard_cc_real]:
                v = self._cached_log_area[idx]
                if not np.isnan(v):
                    stats["hard_scale_cc_log_area"].append(float(v))
            for idx in non_fc[:n_non_fc_real]:
                v = self._cached_log_area[idx]
                if not np.isnan(v):
                    stats["non_fc_log_area"].append(float(v))

            # Source distribution for non-FC samples actually drawn this batch.
            # "non-FC samples" = hard_scale_cc + aerial_non_fc_normal + non_fc.
            non_fc_sample_indices = hard_cc + aerial_normal + non_fc
            for idx in non_fc_sample_indices:
                ds = self._window_dataset[idx]
                stats["non_fc_dataset_counts"][ds] += 1

            batch = aerial_fc + lasot + extra_fc + hard_cc + aerial_normal + non_fc
            rng.shuffle(batch)
            yield batch

        # ---- End-of-epoch summary ----
        self._log_epoch_summary()

    def _log_epoch_summary(self) -> None:
        """Emit a concise epoch-level batch-composition summary at INFO level."""
        import logging as _log
        logger = _log.getLogger(__name__)

        stats = self._epoch_stats
        n_batches = max(1, stats["n_batches"])
        pc = stats["pool_counts"]

        hard_vals = stats["hard_scale_cc_log_area"]
        non_fc_vals = stats["non_fc_log_area"]
        hard_mean = float(np.mean(hard_vals)) if hard_vals else float("nan")
        non_fc_mean = float(np.mean(non_fc_vals)) if non_fc_vals else float("nan")
        overall = self._overall_non_fc_log_area

        # PASS hard>overall: hard_scale_cc should be on the high-log_area tail.
        if np.isnan(hard_mean) or np.isnan(overall):
            pass_str = "N/A"
        else:
            pass_str = "YES" if hard_mean > overall else "NO"

        # Build non-FC source distribution as fractions over all non-FC draws.
        ds_counts = stats["non_fc_dataset_counts"]
        total_non_fc = sum(ds_counts.values())
        if total_non_fc > 0:
            ds_frac = {
                k: ds_counts[k] / total_non_fc for k in sorted(ds_counts)
            }
            ds_str = ", ".join(f"{k}: {v:.2f}" for k, v in ds_frac.items())
        else:
            ds_str = ""

        logger.info(
            "[FCSourceBalancedSampler] epoch composition:\n"
            "  pool sizes (mean per batch over %d batches):\n"
            "    lasot_fc:               %.1f\n"
            "    aerial_fc:              %.1f\n"
            "    other_fc:               %.1f\n"
            "    hard_scale_cc:          %.1f\n"
            "    aerial_non_fc_normal:   %.1f\n"
            "    non_fc:                 %.1f\n"
            "  hard pool sanity check:\n"
            "    hard_scale_cc mean(log_area_ratio_to_init): %.3f\n"
            "    non_fc        mean(log_area_ratio_to_init): %.3f\n"
            "    overall non-FC mean:                        %.3f\n"
            "    PASS hard>overall: %s\n"
            "  source distribution in non-FC samples drawn this epoch:\n"
            "    {%s}",
            n_batches,
            pc["lasot_fc"]             / n_batches,
            pc["aerial_fc"]            / n_batches,
            pc["other_fc"]             / n_batches,
            pc["hard_scale_cc"]        / n_batches,
            pc["aerial_non_fc_normal"] / n_batches,
            pc["non_fc"]               / n_batches,
            hard_mean,
            non_fc_mean,
            overall,
            pass_str,
            ds_str,
        )


class DatasetAwareSampler(Sampler):
    """Batch sampler with fixed dataset composition and FC/LA quota per batch.

    Every batch contains a deterministic fraction from each source dataset,
    with a fixed fraction of FC/LA-dominant windows within each dataset pool.
    This prevents epoch-level oscillation caused by random over-representation
    of one dataset's FC signature (e.g. LaSOT low-conf FC vs UAVDT high-APCE FC).

    Parameters
    ----------
    dataset:
        The CSCDataset whose windows are indexed.
    batch_size:
        Total windows per batch.
    dataset_fractions:
        Mapping ``dataset_name → fraction_of_batch`` (must sum to ~1.0).
        Datasets not present in the mapping are excluded from batches.
    fc_la_fraction:
        Fraction of each dataset's batch contribution that should be
        FC/LA-dominant windows (dominant = any FC or LA frame in the window).
    n_steps:
        Steps per epoch.  Defaults to ``len(dataset) // batch_size``.
    """

    def __init__(
        self,
        dataset: "CSCDataset",
        batch_size: int,
        dataset_fractions: dict[str, float],
        fc_la_fraction: float = 0.18,
        n_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.fractions = dataset_fractions
        self.fc_la_frac = fc_la_fraction
        self.n_steps = n_steps or max(1, len(dataset.windows) // batch_size)

        fc_states = {int(DerivedState.FALSE_CONFIRMED), int(DerivedState.LOST_AWARE)}
        self._fc_idx:    dict[str, list[int]] = defaultdict(list)
        self._other_idx: dict[str, list[int]] = defaultdict(list)

        for i, w in enumerate(dataset.windows):
            last_half = w.derived[len(w.derived) // 2:]
            is_fc_la = any(int(d) in fc_states for d in last_half)
            if is_fc_la:
                self._fc_idx[w.dataset].append(i)
            else:
                self._other_idx[w.dataset].append(i)

        import logging as _log
        for ds in dataset_fractions:
            if not self._fc_idx.get(ds):
                _log.getLogger(__name__).warning(
                    "DatasetAwareSampler: no FC/LA windows for dataset %r — "
                    "FC quota will be filled from regular windows", ds
                )

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self):
        rng = np.random.default_rng()  # new seed each epoch for diversity

        def _pool(idx_list: list[int]) -> np.ndarray:
            return rng.permutation(idx_list) if idx_list else np.array([], dtype=np.int64)

        fc_pools    = {ds: _pool(self._fc_idx.get(ds, [])).tolist()    for ds in self.fractions}
        other_pools = {ds: _pool(self._other_idx.get(ds, [])).tolist() for ds in self.fractions}
        fc_ptrs     = {ds: 0 for ds in self.fractions}
        other_ptrs  = {ds: 0 for ds in self.fractions}

        def _draw(pools: dict, ptrs: dict, ds: str, n: int) -> list[int]:
            if n <= 0:
                return []
            pool = pools.get(ds, [])
            if not pool:
                return []
            ptr = ptrs[ds]
            out: list[int] = []
            while len(out) < n:
                chunk = pool[ptr: ptr + (n - len(out))]
                out.extend(chunk)
                ptr += len(chunk)
                if ptr >= len(pool):
                    pools[ds] = rng.permutation(pool).tolist()
                    pool = pools[ds]
                    ptr = 0
            ptrs[ds] = ptr
            return out[:n]

        for _ in range(self.n_steps):
            batch: list[int] = []
            for ds, frac in self.fractions.items():
                n_total = max(1, round(self.batch_size * frac))
                n_fc    = round(n_total * self.fc_la_frac)
                n_other = n_total - n_fc
                # FC/LA windows; fall back to other pool if insufficient
                fc_drawn = _draw(fc_pools, fc_ptrs, ds, n_fc)
                if len(fc_drawn) < n_fc:
                    n_other += n_fc - len(fc_drawn)
                batch.extend(fc_drawn)
                batch.extend(_draw(other_pools, other_ptrs, ds, n_other))
            rng.shuffle(batch)
            yield batch


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
    stratified_split: bool = False,
    dataset_fractions: Optional[dict[str, float]] = None,
    fc_la_fraction: float = 0.18,
    sampler_type: str = "derived_wrs",
    batch_size: int = 64,
    fc_per_batch: int = 20,
    lasot_fc_cap: float = 0.60,
    aerial_fc_floor: float = 0.30,
    aerial_non_fc_floor: int = 16,
    hard_scale_cc_floor: int = 8,
    scale_pct_threshold: float = 75.0,
    **kwargs,
) -> tuple:
    """Build train/val datasets from a ``labels_dir``.

    Parameters
    ----------
    stratified_split:
        If True, use hash-based deterministic split with guaranteed FC sequences
        in val (via :func:`split_sequences_stratified`).
    dataset_fractions:
        If provided, returns a :class:`DatasetAwareSampler` instead of a
        :class:`WeightedRandomSampler`.  Keys are dataset names; values are
        batch fractions (must sum to ~1.0).
    fc_la_fraction:
        FC/LA window fraction within each dataset pool (used by DatasetAwareSampler).

    Returns
    -------
    If ``return_sampler=False`` (default, backward-compatible):
        ``(train_ds, val_ds, info)``
    If ``return_sampler=True``:
        ``(train_ds, val_ds, info, sampler)``
    """
    rows = load_labels_dir(labels_dir)
    if not rows:
        raise FileNotFoundError(f"no labels.jsonl found under {labels_dir}")
    groups = _group_by_sequence(rows)

    if stratified_split:
        train_keys, val_keys = split_sequences_stratified(
            list(groups.keys()), groups, val_fraction
        )
    else:
        train_keys, val_keys = split_sequences_train_val(
            list(groups.keys()), val_fraction, seed
        )

    train_rows = {k: groups[k] for k in train_keys}
    val_rows   = {k: groups[k] for k in val_keys}

    train_ds = CSCDataset(train_rows, feature_cfg, image_size=image_size)
    val_ds   = CSCDataset(val_rows,   feature_cfg, image_size=image_size)
    info = {
        "n_train_sequences": len(train_keys),
        "n_val_sequences":   len(val_keys),
        "n_train_windows":   len(train_ds),
        "n_val_windows":     len(val_ds),
        "train_sequences":   [list(k) for k in train_keys],
        "val_sequences":     [list(k) for k in val_keys],
    }
    if return_sampler:
        if sampler_type == "fc_source_balanced":
            sampler: Sampler = FCSourceBalancedSampler(
                train_ds,
                batch_size=batch_size,
                fc_per_batch=fc_per_batch,
                lasot_fc_cap=lasot_fc_cap,
                aerial_fc_floor=aerial_fc_floor,
                aerial_non_fc_floor=aerial_non_fc_floor,
                hard_scale_cc_floor=hard_scale_cc_floor,
                scale_pct_threshold=scale_pct_threshold,
            )
        elif dataset_fractions:
            sampler = DatasetAwareSampler(
                train_ds, batch_size=64,
                dataset_fractions=dataset_fractions,
                fc_la_fraction=fc_la_fraction,
            )
        else:
            sampler = train_ds.make_sampler()
        return train_ds, val_ds, info, sampler
    return train_ds, val_ds, info


__all__ = [
    "CSCDataset",
    "CSCWindow",
    "DatasetAwareSampler",
    "FCSourceBalancedSampler",
    "_group_by_sequence",
    "build_train_val_datasets",
    "load_labels_dir",
    "split_sequences_train_val",
    "split_sequences_stratified",
]
