"""Integration test for the ablate CLI command (Phase 5).

Calls the ablate command via typer's CliRunner and asserts:
  1. The command exits with code 0.
  2. A per-variant CSV is written for each variant in the sweep.
  3. The output contains the variant names.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from typer.testing import CliRunner

from uav_tracker.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def signals_sweep(tmp_path: Path) -> Path:
    """Write a minimal ablation_signals.yaml using only motion_entropy + apce."""
    config = """\
sweep_type: signals

base:
  dataset: {name: synthetic}
  seed: 42
  trackers:
    0: {name: kcf_kalman, args: {}}
    1: {name: mobiletrack, args: {device: cpu, dtype: float32}}
  scheduler:
    name: hysteresis_binary
    args: {E_hi: 0.65, E_lo: 0.50, confirm_frames: 5, cooldown_frames: 5}

variants:
  - name: motion_entropy
    signal: {name: motion_entropy, args: {n_bins: 16, alpha: 0.8, mag_threshold: 1.0}}
    scheduler_signal_name: motion_entropy

  - name: apce
    signal: {name: apce, args: {}}
    scheduler_signal_name: apce
"""
    p = tmp_path / "ablation_signals_test.yaml"
    p.write_text(config)
    return p


@pytest.fixture()
def schedulers_sweep(tmp_path: Path) -> Path:
    """Write a minimal ablation_schedulers.yaml with hysteresis_binary + trajectory_aware."""
    config = """\
sweep_type: schedulers

base:
  dataset: {name: synthetic}
  seed: 42
  trackers:
    0: {name: kcf_kalman, args: {}}
    1: {name: mobiletrack, args: {device: cpu, dtype: float32}}
  signal:
    name: motion_entropy
    args: {n_bins: 16, alpha: 0.8, mag_threshold: 1.0}

variants:
  - name: hysteresis_binary
    scheduler:
      name: hysteresis_binary
      args: {E_hi: 0.65, E_lo: 0.50, confirm_frames: 5, cooldown_frames: 5, signal_name: motion_entropy}

  - name: trajectory_aware
    scheduler:
      name: trajectory_aware
      args: {E_hi: 0.65, E_lo: 0.50, confirm_frames: 5, cooldown_frames: 5, tau: 0.1, signal_name: motion_entropy}
"""
    p = tmp_path / "ablation_schedulers_test.yaml"
    p.write_text(config)
    return p


def _invoke_ablate(runner: CliRunner, sweep: Path, out_dir: Path, limit: int = 1) -> any:
    env = {**os.environ, "UAV_RESULTS_ROOT": str(out_dir)}
    return runner.invoke(
        app,
        ["ablate", "--sweep", str(sweep), "--limit", str(limit)],
        env=env,
    )


class TestAblateIntegration:

    def test_signals_sweep_exits_zero(self, runner, signals_sweep, tmp_path) -> None:
        """ablate with signals sweep must exit 0."""
        out_dir = tmp_path / "results"
        result = _invoke_ablate(runner, signals_sweep, out_dir, limit=1)
        assert result.exit_code == 0, (
            f"ablate exited {result.exit_code}.\nOutput:\n{result.output}"
        )

    def test_signals_sweep_csv_written(self, runner, signals_sweep, tmp_path) -> None:
        """A CSV file must be written for each variant."""
        out_dir = tmp_path / "results"
        _invoke_ablate(runner, signals_sweep, out_dir, limit=1)

        # Find the timestamped ablation subdirectory.
        subdirs = list(out_dir.glob("ablation_*"))
        assert len(subdirs) == 1, f"Expected 1 ablation subdir; got {subdirs}"
        ablation_dir = subdirs[0]

        csvs = list(ablation_dir.glob("*.csv"))
        csv_names = {p.stem for p in csvs}
        assert "motion_entropy" in csv_names, f"Missing motion_entropy.csv; found: {csv_names}"
        assert "apce" in csv_names, f"Missing apce.csv; found: {csv_names}"

    def test_signals_sweep_csv_has_required_columns(self, runner, signals_sweep, tmp_path) -> None:
        """Each CSV must have at least the columns: variant, auc, pr20, fps, n_seq."""
        out_dir = tmp_path / "results"
        _invoke_ablate(runner, signals_sweep, out_dir, limit=1)

        subdirs = list(out_dir.glob("ablation_*"))
        ablation_dir = subdirs[0]
        required = {"variant", "auc", "pr20", "fps", "n_seq"}

        for csv_path in ablation_dir.glob("*.csv"):
            with open(csv_path) as fh:
                reader = csv.DictReader(fh)
                headers = set(reader.fieldnames or [])
                missing = required - headers
                assert not missing, (
                    f"{csv_path.name}: missing columns {missing}; got {headers}"
                )

    def test_schedulers_sweep_exits_zero(self, runner, schedulers_sweep, tmp_path) -> None:
        """ablate with schedulers sweep must exit 0."""
        out_dir = tmp_path / "results"
        result = _invoke_ablate(runner, schedulers_sweep, out_dir, limit=1)
        assert result.exit_code == 0, (
            f"ablate exited {result.exit_code}.\nOutput:\n{result.output}"
        )

    def test_schedulers_sweep_csv_written(self, runner, schedulers_sweep, tmp_path) -> None:
        """CSVs must be written for all scheduler variants."""
        out_dir = tmp_path / "results"
        _invoke_ablate(runner, schedulers_sweep, out_dir, limit=1)

        subdirs = list(out_dir.glob("ablation_*"))
        assert len(subdirs) == 1
        ablation_dir = subdirs[0]

        csv_names = {p.stem for p in ablation_dir.glob("*.csv")}
        assert "hysteresis_binary" in csv_names, f"Missing hysteresis_binary.csv; found: {csv_names}"
        assert "trajectory_aware" in csv_names, f"Missing trajectory_aware.csv; found: {csv_names}"

    def test_output_contains_variant_names(self, runner, signals_sweep, tmp_path) -> None:
        """Rich table output must mention at least one variant name."""
        out_dir = tmp_path / "results"
        result = _invoke_ablate(runner, signals_sweep, out_dir, limit=1)
        output = result.output
        assert "motion_entropy" in output or "apce" in output, (
            f"Expected variant names in output; got:\n{output}"
        )
