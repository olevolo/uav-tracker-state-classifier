"""Tests for csc_lib/eval/state.py"""
import pytest

from csc_lib.eval.state import (
    lost_rate,
    mean_time_in_state,
    state_persistence,
    state_rates,
)


def test_state_rates_simple():
    states = ["confirmed", "confirmed", "lost", "uncertain"]
    rates = state_rates(states)
    assert rates["confirmed"] == pytest.approx(0.5)
    assert rates["lost"] == pytest.approx(0.25)
    assert rates["uncertain"] == pytest.approx(0.25)


def test_state_rates_empty():
    assert state_rates([]) == {}


def test_mean_time_single_segment():
    states = ["lost", "lost", "lost"]
    assert mean_time_in_state(states, "lost") == pytest.approx(3.0)


def test_mean_time_multiple_segments():
    states = ["lost", "lost", "confirmed", "lost"]
    assert mean_time_in_state(states, "lost") == pytest.approx(1.5)


def test_mean_time_absent_state():
    states = ["confirmed", "confirmed"]
    assert mean_time_in_state(states, "lost") == pytest.approx(0.0)


def test_state_persistence_stable():
    states = ["a", "a", "a", "b", "b"]
    p = state_persistence(states)
    assert p["a"] == pytest.approx(2 / 3)
    # "b" at index 3 → next is "b" → stayed; at index 4 it is last (not counted)
    assert p["b"] == pytest.approx(1.0)


def test_lost_rate_convenience():
    states = ["confirmed", "lost", "lost", "confirmed"]
    assert lost_rate(states) == pytest.approx(0.5)
