"""Unit tests for viz.entropy_plot.plot_entropy_timeline (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


def _make_synthetic_data(n: int = 30) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(0)
    H_bar = 0.5 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, n)) + 0.05 * rng.standard_normal(n)
    tier_seq = np.zeros(n, dtype=int)
    tier_seq[10:20] = 1
    tier_seq[20:] = 2
    switch_events = [10, 20]
    return H_bar, tier_seq, switch_events


def test_png_is_created(tmp_path: Path) -> None:
    """plot_entropy_timeline writes a PNG file."""
    from uav_tracker.viz.entropy_plot import plot_entropy_timeline

    H_bar, tier_seq, switch_events = _make_synthetic_data(30)
    out = tmp_path / "entropy_timeline.png"

    plot_entropy_timeline(
        H_bar=H_bar,
        tier_sequence=tier_seq,
        E_hi=0.65,
        E_lo=0.50,
        switch_events=switch_events,
        out_path=out,
    )

    assert out.exists(), "PNG file was not created"
    assert out.stat().st_size > 1024, f"PNG too small ({out.stat().st_size} bytes)"


def test_nested_out_dir_created(tmp_path: Path) -> None:
    """plot_entropy_timeline creates parent directories as needed."""
    from uav_tracker.viz.entropy_plot import plot_entropy_timeline

    H_bar, tier_seq, switch_events = _make_synthetic_data(30)
    out = tmp_path / "nested" / "deep" / "timeline.png"

    plot_entropy_timeline(
        H_bar=H_bar,
        tier_sequence=tier_seq,
        E_hi=0.65,
        E_lo=0.50,
        switch_events=switch_events,
        out_path=out,
    )

    assert out.exists()


def test_no_switch_events(tmp_path: Path) -> None:
    """Function handles an empty switch_events list without error."""
    from uav_tracker.viz.entropy_plot import plot_entropy_timeline

    H_bar = np.linspace(0.3, 0.7, 30)
    tier_seq = np.zeros(30, dtype=int)
    out = tmp_path / "no_switch.png"

    plot_entropy_timeline(
        H_bar=H_bar,
        tier_sequence=tier_seq,
        E_hi=0.65,
        E_lo=0.50,
        switch_events=[],
        out_path=out,
    )

    assert out.exists()
    assert out.stat().st_size > 1024


def test_single_frame(tmp_path: Path) -> None:
    """Function handles a degenerate single-frame input without crashing."""
    from uav_tracker.viz.entropy_plot import plot_entropy_timeline

    out = tmp_path / "single_frame.png"
    plot_entropy_timeline(
        H_bar=np.array([0.55]),
        tier_sequence=np.array([0]),
        E_hi=0.65,
        E_lo=0.50,
        switch_events=[],
        out_path=out,
    )

    assert out.exists()
