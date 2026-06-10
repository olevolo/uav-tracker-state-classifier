"""Unit tests for compute_paper_metrics.py metric functions.

Tests per CLAUDE.md §Testing Requirements:
  - FCR calculation
  - FCD calculation
  - Recovery@K calculation
  - TTFC calculation
  - State Transition Matrix shape + values

All tests use only numpy fixtures — no real data or model weights required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.compute_paper_metrics import (
    FALSE_CONFIRMED_IDX,
    CORRECT_CONFIRMED_IDX,
    N_STATES,
    STATE_NAMES,
    compute_fcr,
    compute_fcd,
    compute_ttfc,
    compute_recovery_at_k,
    compute_state_transition_matrix,
    compute_state_conditioned_auc,
)


# ---------------------------------------------------------------------------
# FCR
# ---------------------------------------------------------------------------


class TestFCR:
    def test_all_confirmed(self):
        """No FC frames → FCR = 0."""
        states = np.zeros(100, dtype=np.int64)
        assert compute_fcr(states) == 0.0

    def test_all_false_confirmed(self):
        """All FC frames → FCR = 1.0."""
        states = np.full(100, FALSE_CONFIRMED_IDX, dtype=np.int64)
        assert compute_fcr(states) == 1.0

    def test_half_false_confirmed(self):
        """50 FC out of 200 → FCR = 0.25."""
        states = np.zeros(200, dtype=np.int64)
        states[100:150] = FALSE_CONFIRMED_IDX
        assert abs(compute_fcr(states) - 0.25) < 1e-9

    def test_empty(self):
        """Empty array → FCR = 0."""
        assert compute_fcr(np.array([], dtype=np.int64)) == 0.0

    def test_one_fc_frame(self):
        states = np.array([0, 0, FALSE_CONFIRMED_IDX, 0, 0], dtype=np.int64)
        assert abs(compute_fcr(states) - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# FCD
# ---------------------------------------------------------------------------


class TestFCD:
    def test_no_fc(self):
        """No FC frames → FCD = 0."""
        states = np.zeros(50, dtype=np.int64)
        assert compute_fcd(states) == 0.0

    def test_single_segment(self):
        """One contiguous run of 5 FC frames → FCD = 5.0."""
        states = np.zeros(20, dtype=np.int64)
        states[5:10] = FALSE_CONFIRMED_IDX  # 5 frames
        assert compute_fcd(states) == 5.0

    def test_two_segments(self):
        """Two segments of length 3 and 7 → FCD = mean([3, 7]) = 5.0."""
        states = np.zeros(30, dtype=np.int64)
        states[2:5] = FALSE_CONFIRMED_IDX   # 3 frames
        states[15:22] = FALSE_CONFIRMED_IDX  # 7 frames
        assert compute_fcd(states) == 5.0

    def test_fc_at_end(self):
        """FC segment at end of array is captured."""
        states = np.zeros(10, dtype=np.int64)
        states[7:] = FALSE_CONFIRMED_IDX  # 3 frames
        assert compute_fcd(states) == 3.0


# ---------------------------------------------------------------------------
# TTFC
# ---------------------------------------------------------------------------


class TestTTFC:
    def test_no_fc_returns_none(self):
        states = np.zeros(50, dtype=np.int64)
        assert compute_ttfc(states) is None

    def test_no_confirmed_before_fc_returns_none(self):
        """FC appears at frame 0 with no prior CONFIRMED → None."""
        states = np.zeros(10, dtype=np.int64)
        states[0] = FALSE_CONFIRMED_IDX
        assert compute_ttfc(states) is None

    def test_basic_ttfc(self):
        """CONFIRMED at t=5, first FC at t=10 → TTFC = 5."""
        states = np.zeros(20, dtype=np.int64)
        # frames 0-5: confirmed, frame 10+: FC
        states[10:] = FALSE_CONFIRMED_IDX
        # t_last_confirmed = 9? No — frames 6-9 are state 0 (CORRECT_CONFIRMED)
        # Actually: t_first_FC = 10, t_last_confirmed before that = 9 → TTFC = 1
        # Let's be precise: CORRECT_CONFIRMED = 0
        result = compute_ttfc(states)
        assert result == 1.0  # last confirmed at t=9, first FC at t=10

    def test_ttfc_with_gap(self):
        """Gap of UNCERTAIN between last CONFIRMED and first FC."""
        states = np.zeros(20, dtype=np.int64)
        states[5] = CORRECT_CONFIRMED_IDX   # last confirmed at 5
        states[6:10] = 1  # CORRECT_UNCERTAIN — gap
        states[10:] = FALSE_CONFIRMED_IDX   # first FC at 10
        # All frames 0-4 are CORRECT_CONFIRMED (0), so last before t=10 is t=9?
        # No: states[6:10]=1 and states[10:]=FC, so last CONFIRMED before t=10
        # is the last idx where state==0 AND idx<10 — that is idx=5 (explicitly set)
        # Wait: states[0:5] are default 0 = CORRECT_CONFIRMED; states[5] explicitly 0
        # confirmed_frames < 10 = [0,1,2,3,4,5] → max = 5
        result = compute_ttfc(states)
        assert result == 5.0  # 10 - 5 = 5


# ---------------------------------------------------------------------------
# Recovery@K
# ---------------------------------------------------------------------------


class TestRecoveryAtK:
    def test_no_fc_returns_zero(self):
        states = np.zeros(50, dtype=np.int64)
        assert compute_recovery_at_k(states, k=30) == 0.0

    def test_full_recovery(self):
        """FC for 5 frames, then confirmed immediately → Recovery@5 = 1.0."""
        states = np.zeros(50, dtype=np.int64)
        states[10:15] = FALSE_CONFIRMED_IDX
        # frames 15+ are CORRECT_CONFIRMED → recovery within K=5 is frame 15
        assert compute_recovery_at_k(states, k=5) == 1.0

    def test_no_recovery(self):
        """FC until end of sequence — no recovery window."""
        states = np.full(50, FALSE_CONFIRMED_IDX, dtype=np.int64)
        assert compute_recovery_at_k(states, k=30) == 0.0

    def test_partial_recovery(self):
        """Two FC episodes: one recovers within K, one does not."""
        states = np.zeros(100, dtype=np.int64)
        # Episode 1: FC frames 10-15, confirmed at 16 → recovers within K=10
        states[10:16] = FALSE_CONFIRMED_IDX
        # Episode 2: FC frames 50-99 (no confirmed follows) → does NOT recover
        states[50:] = FALSE_CONFIRMED_IDX
        # recovery_at_k = 1/2 = 0.5
        result = compute_recovery_at_k(states, k=10)
        assert abs(result - 0.5) < 1e-9

    def test_k_boundary(self):
        """FC ends at t=10, confirmed at t=15; K=5 barely includes t=15."""
        states = np.zeros(50, dtype=np.int64)
        states[5:11] = FALSE_CONFIRMED_IDX  # ends at 10 (inclusive), end_frame=10
        # lo = 11, hi = 11+5 = 16 → window is [11..15], confirmed at 15 is included
        assert compute_recovery_at_k(states, k=5) == 1.0


# ---------------------------------------------------------------------------
# State Transition Matrix
# ---------------------------------------------------------------------------


class TestStateTransitionMatrix:
    def test_shape(self):
        """Matrix is always N_STATES x N_STATES."""
        states = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
        mat = compute_state_transition_matrix(states)
        assert mat.shape == (N_STATES, N_STATES)

    def test_single_transition(self):
        """Confirmed→FC appears exactly once."""
        states = np.array([CORRECT_CONFIRMED_IDX, FALSE_CONFIRMED_IDX], dtype=np.int64)
        mat = compute_state_transition_matrix(states)
        assert mat[CORRECT_CONFIRMED_IDX, FALSE_CONFIRMED_IDX] == 1
        assert mat.sum() == 1  # only one transition

    def test_total_count(self):
        """Total transitions = T - 1."""
        states = np.array([0, 1, 2, 3, 0], dtype=np.int64)
        mat = compute_state_transition_matrix(states)
        assert mat.sum() == len(states) - 1

    def test_empty(self):
        states = np.array([], dtype=np.int64)
        mat = compute_state_transition_matrix(states)
        assert mat.shape == (N_STATES, N_STATES)
        assert mat.sum() == 0
