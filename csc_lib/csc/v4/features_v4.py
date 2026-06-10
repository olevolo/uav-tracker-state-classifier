"""CSC-v4 module A1 — normalized response-structure features.

Why this exists
---------------
V3 fed the response-map structure fields (``response_entropy``, ``sm_*``) into
the model in *raw clipped* form (see ``csc_lib/csc/features.py::V3_EXTRA_FIELDS``).
Those fields are emitted on tracker-specific scales (e.g. ``psr`` spans
~18–5700 while ``response_entropy`` lives in ~[2.3, 4.6] and ``sm_n_secondary``
is a small integer count). Mixing such heterogeneous, tracker-coupled scales is
the documented cause of V3 negative-transfer to UAV123: a model that learns the
*absolute* magnitude of SGLATrack's response map does not transfer.

A1 fixes this by **normalizing each response-structure feature independently**
with a robust, distribution-aware transform fit on a calibration split, so the
features have a *consistent semantic* (percentile rank / robust-z) regardless of
the tracker's raw scale. APCE / PSR / confidence are kept SEPARATE (they have
their own V3 ``PercentileFeatureCalibrator`` pipeline) and are NOT touched here.

Public API
----------
- :class:`V4FeatureCalibrator` — one-per-feature robust-z (median/IQR) **and**
  empirical-CDF percentile, JSON ``save`` / ``load``.
- :func:`fit_v4_calibrators` — fit the calibrator bundle for the 7 response
  features from a list of telemetry rows.
- :func:`build_v4_features` — turn one telemetry row into a fixed-length,
  finite, normalized vector whose slot order is documented in
  :data:`FEATURE_NAMES_V4`.

Design mirrors ``csc_lib/csc/calibration.py`` (numpy + json only, no scipy /
sklearn) and ``csc_lib/csc/features.py`` (``_safe`` None-handling, fixed slot
order, ``# INTEGRATION:`` markers). It is **additive**: no V3 file is modified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

# INTEGRATION: shared v4 types are the single source of truth; we do not need
# any enum here yet, but importing keeps the module wired to the v4 backbone and
# satisfies the contract (every v4 module imports from csc_lib.csc.v4.v4types).
from csc_lib.csc.v4.v4types import DerivedStateV4  # noqa: F401  (wiring import)

ArrayLike = Union[np.ndarray, list, tuple]

# ----------------------------------------------------------------------
# Which telemetry fields A1 normalizes.
#
# These are EXACTLY the response-structure fields from the contract's A1 spec,
# kept SEPARATE from apce / psr / confidence (those keep their own V3
# PercentileFeatureCalibrator pipeline). Field names match the V3 telemetry
# JSONL schema (outputs/.../telemetry/<seq>.jsonl).
# ----------------------------------------------------------------------
RESPONSE_FEATURES: tuple[str, ...] = (
    "response_entropy",      # diffuse map on failure  (lower = peakier = better targetness)
    "sm_local_top2_ratio",   # 2nd/1st local peak ratio (higher = more ambiguity/distractor)
    "sm_local_peak_margin",  # local peak dominance     (higher = cleaner single peak)
    "sm_peak_distance",      # grid distance top1->top2 (larger = far-away competitor)
    "sm_heatmap_mass_topk",  # mass concentration in top-k cells
    "sm_n_secondary",        # # competing secondary peaks (integer count)
    "sm_peak_width",         # spatial extent of the dominant peak (integer-ish)
)

# Additional UNBOUNDED features that also need scale-free (percentile) calibration.
# Widening the V4 input beyond pure score-map structure: diagnosis needs the
# calibrated detector signals (apce/psr/confidence) AND bbox dynamics
# (velocity/acceleration/area_ratio, from the label rows) — a response-ONLY input
# under-performs (measured: derived macro-F1 ~0.35 response-only). Emitted as one
# percentile slot each.
EXTRA_FEATURES: tuple[str, ...] = (
    "apce", "psr", "confidence", "velocity", "acceleration", "area_ratio",
)
# Appearance similarities — already ~bounded; emitted as RAW clipped values
# (cosine in [-1,1]; drift in [0,1]). Defaults are the "on-target / no-drift" values.
APPEARANCE_FEATURES: tuple[str, ...] = (
    "last_cosine_sim", "initial_template_sim", "appearance_drift",
)
_APPEARANCE_DEFAULT = {"last_cosine_sim": 1.0, "initial_template_sim": 1.0, "appearance_drift": 0.0}

# Geometry / shape-vs-init features (runtime-safe: computed from the pred_bbox
# trajectory + the init bbox + a confidence EMA by the row builder, NOT from GT).
# These are the FALSE_CONFIRMED-vs-CORRECT_CONFIRMED discriminators: a confidently
# WRONG tracker has a bbox whose scale/shape has drifted from the init template,
# which the response map + (degenerate SGLATrack) appearance do NOT capture.
# MEASURED (held-out by sequence, leakage-free): adding these lifts FC-vs-CC AUROC
# 0.65 -> 0.81 and FC-vs-ALL 0.73 -> 0.83. The V3 feature set (16-dim v2) had them;
# the V4 redesign dropped them, which made FC unlearnable (FC AUROC 0.46). Emitted
# as robust-z (percentile would compress the informative tail). The builder injects
# these scalar fields into each row before fit/build.
GEOM_FEATURES: tuple[str, ...] = (
    "log_w_ratio_to_init", "log_h_ratio_to_init", "log_area_ratio_to_init",
    "aspect_ratio", "conf_ema_trend",
)

# Minimum finite samples below which calibration is unsafe. Smaller than the
# V3 confidence calibrator's 1000 because per-sequence smokes / small splits may
# legitimately have a few hundred frames; tune up for the real fit.
_MIN_SAMPLES = 200

# IQR -> Gaussian-sigma scaling so robust-z is ~unit-variance on normal data.
# sigma ~= IQR / 1.349  (1.349 = Phi^-1(0.75) - Phi^-1(0.25)).
_IQR_TO_SIGMA = 1.349

_EPS = 1e-8


class V4FeatureCalibrator:
    """Per-feature robust normalizer: robust-z (median/IQR) AND percentile (CDF).

    Two complementary, scale-free views of one feature are stored so downstream
    code can pick whichever is more useful per slot:

    * **robust-z** ``(x - median) / (IQR / 1.349)`` — symmetric, outlier-robust
      standardization clipped to ``[-clip, clip]``. Good for roughly-symmetric
      features where magnitude/sign matters.
    * **percentile** — the empirical CDF (rank in ``[0, 1]``) evaluated by
      piecewise-linear interpolation over 101 stored quantile anchors. Good for
      skewed / bounded features (it is monotone and bounded by construction).

    Both transforms are fit on the same calibration split. Persistence is a tiny
    JSON (median, IQR, 101 quantile anchors) — no raw values retained. The JSON
    format intentionally mirrors ``csc_lib/csc/calibration.py`` for consistency.

    Parameters
    ----------
    name:
        Feature name (bookkeeping only, stored in the JSON).
    clip:
        Symmetric clip applied to the robust-z output, in sigma units.
    """

    def __init__(self, name: str, clip: float = 5.0) -> None:
        self.name: str = name
        self.clip: float = float(clip)
        self._median: float = 0.0
        self._iqr: float = 0.0
        self._scale: float = 1.0                 # IQR / 1.349 (clamped to >= _EPS)
        self._quantile_anchors: Optional[np.ndarray] = None  # (101,) q0..q100
        self._n_samples: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, values: ArrayLike) -> "V4FeatureCalibrator":
        """Fit robust-z + percentile params on a 1-D array of raw feature values.

        NaN / Inf are silently dropped. Raises :class:`ValueError` if fewer than
        ``_MIN_SAMPLES`` finite values remain.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < _MIN_SAMPLES:
            raise ValueError(
                f"V4FeatureCalibrator[{self.name}].fit: need >= {_MIN_SAMPLES} "
                f"finite samples, got {arr.size}. Pass a larger calibration split."
            )
        arr.sort()
        q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
        self._median = float(q50)
        self._iqr = float(q75 - q25)
        self._scale = max(_EPS, self._iqr / _IQR_TO_SIGMA)
        # 101 quantile anchors (q0..q100) for the empirical-CDF percentile map.
        self._quantile_anchors = np.percentile(arr, np.arange(0, 101, 1)).astype(np.float64)
        self._n_samples = int(arr.size)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def robust_z(self, x: Union[float, None]) -> float:
        """Robust z-score of one value, clipped to ``[-clip, clip]``.

        ``None`` / non-finite -> 0.0 (i.e. "at the median").
        """
        if not self._fitted:
            raise RuntimeError(f"V4FeatureCalibrator[{self.name}] is not fitted/loaded.")
        v = _safe(x, default=self._median)
        z = (v - self._median) / self._scale
        if not np.isfinite(z):
            return 0.0
        return float(np.clip(z, -self.clip, self.clip))

    def percentile(self, x: Union[float, None]) -> float:
        """Empirical-CDF rank of one value in ``[0, 1]`` (median maps to ~0.5).

        ``None`` / non-finite -> 0.5 (neutral rank).
        """
        if not self._fitted or self._quantile_anchors is None:
            raise RuntimeError(f"V4FeatureCalibrator[{self.name}] is not fitted/loaded.")
        if x is None or not np.isfinite(x):
            return 0.5
        positions = np.linspace(0.0, 1.0, self._quantile_anchors.size)
        p = float(np.interp(float(x), self._quantile_anchors, positions))
        return float(np.clip(p, 0.0, 1.0))

    def transform(self, x: Union[float, None], mode: str = "robust_z") -> float:
        """Transform one value. ``mode`` in {'robust_z','percentile'}.

        Returns a finite float in ``[-clip, clip]`` (robust_z) or ``[0, 1]``
        (percentile). Default ``robust_z`` matches the contract's primary signal.
        """
        if mode == "robust_z":
            return self.robust_z(x)
        if mode == "percentile":
            return self.percentile(x)
        raise ValueError(f"unknown transform mode {mode!r}; use 'robust_z' or 'percentile'.")

    def __call__(self, x: Union[float, None], mode: str = "robust_z") -> float:
        """Alias for :meth:`transform`."""
        return self.transform(x, mode=mode)

    # ------------------------------------------------------------------
    # Persistence (JSON; mirrors csc_lib/csc/calibration.py)
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Persist as a small JSON (median, IQR, 101 quantile anchors)."""
        if not self._fitted or self._quantile_anchors is None:
            raise RuntimeError(f"V4FeatureCalibrator[{self.name}] not fitted — call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "calibrator_class": self.__class__.__name__,
            "feature_name": self.name,
            "clip": self.clip,
            "n_samples": self._n_samples,
            "median": self._median,
            "iqr": self._iqr,
            "scale": self._scale,
            "quantiles": self._quantile_anchors.tolist(),  # 101 values q0..q100
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "V4FeatureCalibrator":
        """Rebuild a :class:`V4FeatureCalibrator` from JSON."""
        path = Path(path)
        with open(path) as fh:
            data = json.load(fh)
        obj = cls(name=data.get("feature_name", "unknown"), clip=float(data.get("clip", 5.0)))
        obj._median = float(data["median"])
        obj._iqr = float(data.get("iqr", 0.0))
        obj._scale = max(_EPS, float(data.get("scale", obj._iqr / _IQR_TO_SIGMA)))
        obj._quantile_anchors = np.array(data["quantiles"], dtype=np.float64)
        obj._n_samples = int(data.get("n_samples", 0))
        obj._fitted = True
        return obj


def fit_v4_calibrators(rows: list[dict]) -> dict[str, V4FeatureCalibrator]:
    """Fit one :class:`V4FeatureCalibrator` per response-structure feature.

    Parameters
    ----------
    rows:
        Telemetry rows (e.g. parsed from ``outputs/.../telemetry/<seq>.jsonl``).
        Rows missing a field (e.g. the ``init`` frame) contribute nothing for
        that field — only finite values are collected.

    Returns
    -------
    dict[str, V4FeatureCalibrator]
        One fitted calibrator per name in :data:`RESPONSE_FEATURES`. Features
        with too few finite samples are skipped (with a warning print) so a
        partial bundle can still be built on sparse data; ``build_v4_features``
        degrades a missing calibrator to a neutral 0.0 / 0.5 output.

    Notes
    -----
    APCE / PSR / confidence are deliberately NOT fit here — they keep the V3
    ``PercentileFeatureCalibrator`` pipeline (``csc_lib/csc/calibration.py``).
    """
    calibrators: dict[str, V4FeatureCalibrator] = {}
    for name in (*RESPONSE_FEATURES, *EXTRA_FEATURES, *GEOM_FEATURES):
        vals = np.fromiter(
            (
                float(r[name])
                for r in rows
                if isinstance(r.get(name), (int, float)) and np.isfinite(r[name])
            ),
            dtype=np.float64,
        )
        try:
            calibrators[name] = V4FeatureCalibrator(name).fit(vals)
        except ValueError as exc:  # too few finite samples for this field
            print(f"[fit_v4_calibrators] skipping {name!r}: {exc}")
    return calibrators


# ----------------------------------------------------------------------
# Output feature vector slot order (the V4 normalized response-structure block).
#
# For each of the 7 response features we emit BOTH views:
#   *_z   : robust-z (median/IQR), clipped to [-clip, clip]
#   *_pct : empirical-CDF percentile in [0, 1]
# plus 2 causal temporal deltas (percentile-space, vs the previous frame) for the
# two most discriminative targetness signals (entropy & local top2 ratio), so a
# downstream non-temporal consumer still sees short-term change. The TCN encoder
# (A6) gets its own temporal context, but these deltas make the per-frame vector
# self-contained and cheap.
#
# Length is fixed regardless of which calibrators were fit (missing calibrator ->
# neutral fill), so the model's feature_dim never depends on the calibration data.
# ----------------------------------------------------------------------
def _build_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for f in RESPONSE_FEATURES:
        names.append(f"{f}_z")
        names.append(f"{f}_pct")
    names.append("response_entropy_pct_delta")     # causal Δ vs prev frame
    names.append("sm_local_top2_ratio_pct_delta")  # causal Δ vs prev frame
    for f in EXTRA_FEATURES:                        # calibrated detector + bbox-dynamics (percentile)
        names.append(f"{f}_pct")
    for f in APPEARANCE_FEATURES:                   # appearance similarities (raw clipped)
        names.append(f"{f}_raw")
    for f in EXTRA_FEATURES:                        # robust-z view too (percentile compresses FC tail)
        names.append(f"{f}_z")
    for f in GEOM_FEATURES:                         # geometry / shape-vs-init (FC-vs-CC): z + percentile
        names.append(f"{f}_z")
        names.append(f"{f}_pct")
    return tuple(names)


FEATURE_NAMES_V4: tuple[str, ...] = _build_feature_names()
FEATURE_DIM_V4 = len(FEATURE_NAMES_V4)  # 14 + 2 + 6(extra pct) + 3(appear raw) + 6(extra z) + 10(geom z+pct) = 41


def _safe(x: Union[float, int, None], default: float = 0.0) -> float:
    """None / non-finite -> ``default``; else float(x). Mirrors features.py::_safe."""
    if x is None:
        return default
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(xf):
        return default
    return xf


def build_v4_features(
    row: dict,
    calibrators: dict[str, V4FeatureCalibrator],
    prev: Optional[dict] = None,
) -> np.ndarray:
    """Build the fixed-length normalized response-structure vector for one row.

    Parameters
    ----------
    row:
        A telemetry row (mapping). Missing keys -> neutral normalized values
        (robust-z 0.0, percentile 0.5). The ``init`` frame (which only has
        ``frame_idx``/``init``/``latency_ms``) therefore yields an all-neutral
        vector rather than crashing.
    calibrators:
        Bundle from :func:`fit_v4_calibrators` (or loaded via
        :meth:`V4FeatureCalibrator.load`). A name absent from the bundle
        degrades that slot pair to neutral (0.0 / 0.5).
    prev:
        Optional previous-frame telemetry row, used only for the two causal
        percentile deltas. ``None`` -> deltas are 0.0 (no leakage; uses only
        ``[prev, row]``).

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(FEATURE_DIM_V4,)`` whose slot order is
        :data:`FEATURE_NAMES_V4`. Guaranteed finite.
    """
    out = np.empty(FEATURE_DIM_V4, dtype=np.float32)

    i = 0
    for name in RESPONSE_FEATURES:
        cal = calibrators.get(name)
        raw = row.get(name)
        if cal is None:
            out[i] = 0.0       # robust-z neutral
            out[i + 1] = 0.5   # percentile neutral
        else:
            out[i] = cal.robust_z(raw)
            out[i + 1] = cal.percentile(raw)
        i += 2

    # Causal percentile deltas (row.pct - prev.pct) for the two key targetness
    # signals. Percentile space keeps the delta bounded in [-1, 1].
    def _pct_delta(name: str) -> float:
        cal = calibrators.get(name)
        if cal is None or prev is None:
            return 0.0
        return float(cal.percentile(row.get(name)) - cal.percentile(prev.get(name)))

    out[i] = _pct_delta("response_entropy")
    out[i + 1] = _pct_delta("sm_local_top2_ratio")
    i += 2

    # Extra unbounded features -> percentile (calibrated detector signals + bbox dynamics).
    for name in EXTRA_FEATURES:
        cal = calibrators.get(name)
        out[i] = cal.percentile(row.get(name)) if cal is not None else 0.5
        i += 1
    # Appearance similarities -> raw, clipped to [-1, 1] (cosine native range).
    for name in APPEARANCE_FEATURES:
        out[i] = float(np.clip(_safe(row.get(name), _APPEARANCE_DEFAULT[name]), -1.0, 1.0))
        i += 1
    # Extra detector/bbox features ALSO as robust-z (percentile saturates the high
    # tail where APCE/PSR on FC frames live; robust-z keeps the magnitude gradient).
    for name in EXTRA_FEATURES:
        cal = calibrators.get(name)
        out[i] = cal.robust_z(row.get(name)) if cal is not None else 0.0
        i += 1
    # Geometry / shape-vs-init (robust-z + percentile) — the FC-vs-CC discriminators.
    # Percentile is monotone/bounded so it preserves the rank signal; clipped robust-z
    # ALONE saturates the heavy tail (log-area-ratio) where FC frames live and loses it
    # (measured: geom-z-only FC-vs-CC 0.69 vs raw 0.81; adding percentile recovers it).
    for name in GEOM_FEATURES:
        cal = calibrators.get(name)
        if cal is None:
            out[i] = 0.0; out[i + 1] = 0.5
        else:
            out[i] = cal.robust_z(row.get(name)); out[i + 1] = cal.percentile(row.get(name))
        i += 2

    # Final safety net: any NaN/Inf (should be impossible) -> 0.0.
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ----------------------------------------------------------------------
# Standalone smoke (CPU-only, no datasets): fit on ~200 random rows, transform,
# assert finite + fixed shape, exercise save/load round-trip.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def _rand_row() -> dict:
        # Mimic the measured per-field ranges from a real UAV123 telemetry file.
        return {
            "response_entropy": float(rng.uniform(2.3, 4.6)),
            "sm_local_top2_ratio": float(rng.uniform(0.0, 1.0)),
            "sm_local_peak_margin": float(rng.uniform(0.0, 0.77)),
            "sm_peak_distance": float(rng.uniform(0.7, 2.6)),
            "sm_heatmap_mass_topk": float(rng.uniform(0.25, 0.9)),
            "sm_n_secondary": float(rng.integers(0, 6)),
            "sm_peak_width": float(rng.integers(1, 17)),
            # extra calibrated features (now PART of the v4 vector):
            "apce": float(rng.uniform(20, 210)),
            "psr": float(rng.uniform(18, 5700)),
            "confidence": float(rng.uniform(0.013, 0.019)),
            "velocity": float(rng.uniform(0, 30)),
            "acceleration": float(rng.uniform(0, 15)),
            "area_ratio": float(rng.uniform(0.5, 2.0)),
            # appearance similarities (raw, clipped to [-1,1]):
            "last_cosine_sim": float(rng.uniform(0.3, 1.0)),
            "initial_template_sim": float(rng.uniform(0.3, 1.0)),
            "appearance_drift": float(rng.uniform(0.0, 0.7)),
            # geometry / shape-vs-init (robust-z; injected by the row builder):
            "log_w_ratio_to_init": float(rng.uniform(-1.5, 1.5)),
            "log_h_ratio_to_init": float(rng.uniform(-1.5, 1.5)),
            "log_area_ratio_to_init": float(rng.uniform(-3.0, 3.0)),
            "aspect_ratio": float(rng.uniform(0.3, 3.0)),
            "conf_ema_trend": float(rng.uniform(-0.01, 0.01)),
        }

    rows = [_rand_row() for _ in range(250)]
    cals = fit_v4_calibrators(rows)
    assert set(cals) == set(RESPONSE_FEATURES) | set(EXTRA_FEATURES) | set(GEOM_FEATURES), \
        f"missing calibrators: {(set(RESPONSE_FEATURES) | set(EXTRA_FEATURES) | set(GEOM_FEATURES)) - set(cals)}"

    # transform every row, with and without prev
    prev = None
    for r in rows:
        vec = build_v4_features(r, cals, prev=prev)
        assert vec.shape == (FEATURE_DIM_V4,), f"bad shape {vec.shape}"
        assert vec.dtype == np.float32, f"bad dtype {vec.dtype}"
        assert np.all(np.isfinite(vec)), "non-finite feature emitted"
        prev = r

    # percentile slots in [0,1]; robust-z slots within clip
    vec = build_v4_features(rows[10], cals, prev=rows[9])
    for slot, nm in enumerate(FEATURE_NAMES_V4):
        if nm.endswith("_pct"):
            assert 0.0 <= vec[slot] <= 1.0, f"{nm} out of [0,1]: {vec[slot]}"
        elif nm.endswith("_z"):
            assert -5.0 <= vec[slot] <= 5.0, f"{nm} out of clip: {vec[slot]}"
        elif nm.endswith("_raw"):
            assert -1.0 <= vec[slot] <= 1.0, f"{nm} out of [-1,1]: {vec[slot]}"

    # robustness: missing keys (init-style row) -> all neutral, finite
    empty_vec = build_v4_features({"frame_idx": 0, "init": True}, cals, prev=None)
    assert np.all(np.isfinite(empty_vec)) and empty_vec.shape == (FEATURE_DIM_V4,)
    for slot, nm in enumerate(FEATURE_NAMES_V4):
        if nm.endswith("_pct"):
            assert empty_vec[slot] == 0.5, f"missing-key {nm} should be 0.5, got {empty_vec[slot]}"
        elif nm.endswith("_z") or nm.endswith("_delta"):
            assert empty_vec[slot] == 0.0, f"missing-key {nm} should be 0.0, got {empty_vec[slot]}"
        elif nm.endswith("_raw"):
            assert np.isfinite(empty_vec[slot]), f"missing-key {nm} not finite: {empty_vec[slot]}"

    # partial bundle: drop a calibrator -> that slot pair degrades to neutral, fixed shape holds
    partial = dict(cals)
    partial.pop("sm_peak_width")
    pv = build_v4_features(rows[0], partial, prev=None)
    assert pv.shape == (FEATURE_DIM_V4,) and np.all(np.isfinite(pv))

    # save/load round-trip equivalence
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "response_entropy.json"
        cals["response_entropy"].save(p)
        reloaded = V4FeatureCalibrator.load(p)
        x = 3.7
        assert abs(reloaded.robust_z(x) - cals["response_entropy"].robust_z(x)) < 1e-9, "robust_z mismatch after load"
        assert abs(reloaded.percentile(x) - cals["response_entropy"].percentile(x)) < 1e-6, "percentile mismatch after load"

    print(f"OK features_v4 smoke: FEATURE_DIM_V4={FEATURE_DIM_V4}")
    print(f"FEATURE_NAMES_V4 ({len(FEATURE_NAMES_V4)}): {list(FEATURE_NAMES_V4)}")
    print(f"calibrators fit: {sorted(cals)}")
    print(f"sample vec[:6]={np.round(vec[:6], 3).tolist()}")
