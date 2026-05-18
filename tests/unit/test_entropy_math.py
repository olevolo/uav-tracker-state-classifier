"""Property tests for entropy math invariants (Phase 4).

Paper §3.2 defines ``H̃ = H / log₂(N) ∈ [0, 1]`` with closed-form endpoints:

    * Delta distribution (all mass in one bin) → ``H̃ = 0``.
    * Uniform distribution (mass split evenly across N bins) → ``H̃ = 1``.

The tests below target a helper function pair that the Engineer will expose in
``uav_tracker.signals.motion_entropy``:

    shannon_entropy(histogram: np.ndarray) -> float
        Raw Shannon entropy in bits (log₂ base).

    normalize_entropy(h: float, n_bins: int) -> float
        ``H / log₂(N)``, mapping the raw value to ``[0, 1]``.

If those names do not exist yet, the import guard skips the import-dependent
tests while keeping the reference-implementation tests runnable at all times.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

# --------------------------------------------------------------------------- #
# Try to import the real helpers from motion_entropy.                          #
# --------------------------------------------------------------------------- #

try:
    from uav_tracker.signals.motion_entropy import normalize_entropy, shannon_entropy

    _HAVE_REAL_IMPL = True
except ImportError:
    _HAVE_REAL_IMPL = False
    shannon_entropy = None  # type: ignore[assignment]
    normalize_entropy = None  # type: ignore[assignment]

_skip_real = pytest.mark.skipif(
    not _HAVE_REAL_IMPL,
    reason="motion_entropy.shannon_entropy / normalize_entropy not yet exposed (Phase 4)",
)


# --------------------------------------------------------------------------- #
# Reference implementation (test-side truth table, always available)          #
# --------------------------------------------------------------------------- #


def _ref_shannon_entropy(histogram: np.ndarray) -> float:
    """Reference Shannon entropy (bits) — test-side truth table."""
    p = histogram / histogram.sum()
    nonzero = p[p > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _ref_normalize(h: float, n_bins: int) -> float:
    return h / math.log2(n_bins)


# --------------------------------------------------------------------------- #
# Deterministic invariant tests (reference implementation)                    #
# --------------------------------------------------------------------------- #


def test_delta_distribution_gives_zero_entropy() -> None:
    """All mass in one bin → H = 0."""
    hist = np.zeros(16, dtype=np.float64)
    hist[7] = 1.0
    assert _ref_shannon_entropy(hist) == pytest.approx(0.0, abs=1e-9)


def test_uniform_distribution_gives_max_entropy() -> None:
    """Equal mass in all N bins → H = log₂(N)."""
    hist = np.ones(16, dtype=np.float64)
    assert _ref_shannon_entropy(hist) == pytest.approx(math.log2(16), abs=1e-9)


def test_normalized_delta_is_zero() -> None:
    """Delta distribution → H̃ = 0."""
    hist = np.zeros(32, dtype=np.float64)
    hist[0] = 100.0
    h = _ref_shannon_entropy(hist)
    assert _ref_normalize(h, 32) == pytest.approx(0.0, abs=1e-9)


def test_normalized_uniform_is_one() -> None:
    """Uniform distribution → H̃ = 1."""
    n = 32
    hist = np.ones(n, dtype=np.float64)
    h = _ref_shannon_entropy(hist)
    assert _ref_normalize(h, n) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Hypothesis-based property tests (reference implementation)                  #
# --------------------------------------------------------------------------- #


@given(
    masses=st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=64,
    )
)
def test_normalized_entropy_in_unit_interval(masses: list[float]) -> None:
    """For any non-degenerate histogram H̃ ∈ [0, 1]."""
    hist = np.asarray(masses, dtype=np.float64)
    assume(hist.sum() > 0)
    n = len(hist)
    assume(n >= 2)
    h = _ref_shannon_entropy(hist)
    h_norm = _ref_normalize(h, n)
    # Allow tiny floating-point overshoot.
    assert -1e-9 <= h_norm <= 1.0 + 1e-9


@given(
    n_bins=st.integers(min_value=2, max_value=256),
    alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_entropy_monotone_toward_uniform(n_bins: int, alpha: float) -> None:
    """Mixing a spike toward uniform → H̃ is non-decreasing.

    Construct delta = mass all in bin 0.
    Uniform = mass 1/N in every bin.
    Mixture(alpha) = (1 - alpha) * delta + alpha * uniform.

    H̃(alpha=0) == 0, H̃(alpha=1) == 1, and H̃ is monotonically non-decreasing
    in alpha for all intermediate values.
    """
    delta = np.zeros(n_bins, dtype=np.float64)
    delta[0] = 1.0
    uniform = np.ones(n_bins, dtype=np.float64) / n_bins

    # Three mixture points: alpha=0, alpha/2, alpha.
    def h_tilde(a: float) -> float:
        mix = (1.0 - a) * delta + a * uniform
        if mix.sum() <= 0:
            return 0.0
        return _ref_normalize(_ref_shannon_entropy(mix), n_bins)

    h0 = h_tilde(0.0)
    h_mid = h_tilde(alpha / 2.0)
    h_full = h_tilde(alpha)

    # Monotonicity: h0 <= h_mid <= h_full (with floating-point tolerance).
    assert h0 <= h_mid + 1e-9
    assert h_mid <= h_full + 1e-9


# --------------------------------------------------------------------------- #
# Tests against the real signal helpers (skipped until Phase 4 exposes them)  #
# --------------------------------------------------------------------------- #


@_skip_real
def test_real_shannon_entropy_delta() -> None:
    """shannon_entropy(delta probability vector) == 0.0.

    The real helper takes a *probability vector* (sums to 1), not raw counts.
    """
    p = np.zeros(16, dtype=np.float64)
    p[3] = 1.0  # already a valid probability vector
    result = shannon_entropy(p)  # type: ignore[misc]
    assert result == pytest.approx(0.0, abs=1e-9)


@_skip_real
def test_real_shannon_entropy_uniform() -> None:
    """shannon_entropy(uniform probability vector of length N) == log₂(N)."""
    n = 16
    p = np.full(n, 1.0 / n, dtype=np.float64)
    result = shannon_entropy(p)  # type: ignore[misc]
    assert result == pytest.approx(math.log2(n), abs=1e-9)


@_skip_real
def test_real_normalize_entropy_roundtrip() -> None:
    """normalize_entropy(shannon_entropy(uniform), N) == 1.0."""
    n = 16
    p = np.full(n, 1.0 / n, dtype=np.float64)
    h_raw = shannon_entropy(p)  # type: ignore[misc]
    h_norm = normalize_entropy(h_raw, n)  # type: ignore[misc]
    assert h_norm == pytest.approx(1.0, abs=1e-9)


@_skip_real
@given(
    masses=st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=64,
    )
)
def test_real_normalized_entropy_in_unit_interval(masses: list[float]) -> None:
    """Real normalize_entropy output is always in [0, 1].

    Converts raw counts to a probability vector before calling shannon_entropy,
    matching the real API contract (probability vector, not raw histogram).
    """
    hist = np.asarray(masses, dtype=np.float64)
    assume(hist.sum() > 0)
    n = len(hist)
    assume(n >= 2)
    p = hist / hist.sum()
    h_raw = shannon_entropy(p)  # type: ignore[misc]
    h_norm = normalize_entropy(h_raw, n)  # type: ignore[misc]
    assert -1e-9 <= h_norm <= 1.0 + 1e-9
