"""Per-tracker percentile-based confidence / feature calibrators.

Each tracker exposes raw scores on a scale that differs from the paper-default
thresholds (e.g. SGLATrack-DeiT confidence is in ~[0.01, 0.02] while
tau_fc_conf=0.65).  This module maps raw scores to rank-percentiles in [0, 1]
so that label-generation thresholds have a consistent semantic across trackers.

Usage
-----
::

    from csc_lib.csc.calibration import PercentileConfidenceCalibrator

    cal = PercentileConfidenceCalibrator()
    cal.fit(raw_conf_array)          # needs >= 1000 finite samples
    norm = cal.transform(raw_score)  # float or np.ndarray -> np.ndarray

    cal.save(Path("outputs/calibration/sglatrack_got10k_confidence.json"))
    cal2 = PercentileConfidenceCalibrator.load(path)

Pure numpy / json — no scipy, no sklearn.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

_MIN_SAMPLES = 1_000  # below this, calibration is unsafe


class PercentileConfidenceCalibrator:
    """Maps raw tracker confidence scores to rank-percentiles in [0, 1].

    The transform is a searchsorted-based rank computation (i.e. the exact
    empirical CDF on the calibration split), so it is monotone non-decreasing
    and bijective on the calibration support.

    When loaded from a saved JSON (which stores only 101 quantile anchors at
    1% steps), a piecewise-linear interpolation replaces the exact searchsorted.
    The interpolation error is < 0.5 pp for well-behaved distributions.
    """

    def __init__(self) -> None:
        self._sorted_scores: np.ndarray | None = None  # full array when fitted directly
        self._quantile_anchors: np.ndarray | None = None  # 101 values when loaded from JSON
        self._n_samples: int = 0
        self._from_file: bool = False  # True iff loaded from saved JSON

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, scores: Union[np.ndarray, list]) -> "PercentileConfidenceCalibrator":
        """Fit on a 1-D array of raw scores.

        Parameters
        ----------
        scores:
            Raw tracker output values (any finite float).  NaN / Inf are
            silently dropped.

        Raises
        ------
        ValueError
            If fewer than 1000 finite samples remain after filtering.
        """
        arr = np.asarray(scores, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < _MIN_SAMPLES:
            raise ValueError(
                f"PercentileConfidenceCalibrator.fit: need >= {_MIN_SAMPLES} finite "
                f"samples, got {arr.size}.  Pass a larger calibration split."
            )
        self._sorted_scores = np.sort(arr)
        self._n_samples = int(arr.size)
        self._quantile_anchors = None
        self._from_file = False
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self, score: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Map raw score(s) to rank-percentile in [0, 1].

        Vectorised: scalar input → 0-d array; array input → same shape.
        """
        scalar_in = np.ndim(score) == 0
        x = np.atleast_1d(np.asarray(score, dtype=np.float64))

        if self._sorted_scores is not None:
            # Exact empirical CDF via searchsorted
            out = np.searchsorted(self._sorted_scores, x, side="right").astype(
                np.float64
            )
            out /= float(len(self._sorted_scores))
        elif self._quantile_anchors is not None:
            # Piecewise-linear interpolation over 101 quantile anchors
            anchors = self._quantile_anchors  # shape (101,)
            # percentile positions: 0, 1, 2, ..., 100 → map to [0, 1]
            positions = np.linspace(0.0, 1.0, 101)
            out = np.interp(x, anchors, positions)
        else:
            raise RuntimeError(
                "PercentileConfidenceCalibrator is neither fitted nor loaded."
            )

        if scalar_in:
            return out[0]
        return out

    def __call__(
        self, score: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Alias for :py:meth:`transform`."""
        return self.transform(score)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist as a small JSON (< 5 KB for 101 quantile anchors).

        The JSON stores quantiles at every 1% step so the calibrator can
        be rebuilt without keeping all raw scores.

        Parameters
        ----------
        path:
            Destination file.  Parent directories are created automatically.
        """
        if self._sorted_scores is None:
            raise RuntimeError("Calibrator not fitted — call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        quantiles = np.percentile(
            self._sorted_scores, np.arange(0, 101, 1)
        ).tolist()

        payload = {
            "calibrator_class": self.__class__.__name__,
            "n_samples": self._n_samples,
            "min": float(self._sorted_scores[0]),
            "max": float(self._sorted_scores[-1]),
            "quantiles": quantiles,  # 101 values, q0..q100
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PercentileConfidenceCalibrator":
        """Rebuild from a saved JSON.

        Uses piecewise-linear interpolation over the 101 stored quantile
        anchors — no raw scores needed.
        """
        path = Path(path)
        with open(path) as fh:
            data = json.load(fh)
        obj = cls()
        obj._quantile_anchors = np.array(data["quantiles"], dtype=np.float64)
        obj._n_samples = int(data.get("n_samples", 0))
        obj._from_file = True
        return obj


class PercentileFeatureCalibrator(PercentileConfidenceCalibrator):
    """Percentile calibrator for a named auxiliary feature (APCE, PSR, …).

    Identical API to :class:`PercentileConfidenceCalibrator`; the ``name``
    argument is stored for bookkeeping only.

    Parameters
    ----------
    name:
        Feature name, e.g. ``"apce"`` or ``"psr"``.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: str = name

    def save(self, path: Path) -> None:
        """Persist with ``feature_name`` added to the JSON payload."""
        if self._sorted_scores is None:
            raise RuntimeError("Calibrator not fitted — call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        quantiles = np.percentile(
            self._sorted_scores, np.arange(0, 101, 1)
        ).tolist()

        payload = {
            "calibrator_class": self.__class__.__name__,
            "feature_name": self.name,
            "n_samples": self._n_samples,
            "min": float(self._sorted_scores[0]),
            "max": float(self._sorted_scores[-1]),
            "quantiles": quantiles,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PercentileFeatureCalibrator":
        """Rebuild a :class:`PercentileFeatureCalibrator` from JSON."""
        path = Path(path)
        with open(path) as fh:
            data = json.load(fh)
        obj = cls(name=data.get("feature_name", "unknown"))
        obj._quantile_anchors = np.array(data["quantiles"], dtype=np.float64)
        obj._n_samples = int(data.get("n_samples", 0))
        obj._from_file = True
        return obj
