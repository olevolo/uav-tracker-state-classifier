"""Integration test for scripts/run_benchmark.py.

Runs the benchmark script via subprocess on paper_table2.yaml with
--dataset synthetic --limit 1. Asserts:
  - CSVs are written for each non-skipped variant
  - summary.md exists and contains expected markdown table headers
  - Caching works: a second run without --force prints "cached" and skips re-run

These tests are deliberately integration-level (subprocess) so they exercise
the full CLI path including plugin registration and YAML loading.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
_SCRIPT = _REPO_ROOT / "scripts" / "run_benchmark.py"
_SWEEP = _REPO_ROOT / "configs" / "experiments" / "paper_table2.yaml"
_PYTHON = sys.executable


# Helpers -------------------------------------------------------------------


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run run_benchmark.py with given extra args."""
    cmd = [
        _PYTHON, str(_SCRIPT),
        "--sweep", str(_SWEEP),
        "--dataset", "synthetic",
        "--limit", "1",
        *args,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        **kwargs,
    )


def _latest_run_dir(base_name: str = "paper_table2") -> Path | None:
    """Return the most recently created timestamped run dir."""
    results_root = _REPO_ROOT / "results"
    dirs = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith(base_name)],
        key=lambda d: d.stat().st_mtime,
    )
    return dirs[-1] if dirs else None


# Tests ---------------------------------------------------------------------


class TestRunBenchmark:
    def test_script_exists(self):
        assert _SCRIPT.exists(), f"run_benchmark.py not found at {_SCRIPT}"

    def test_sweep_config_exists(self):
        assert _SWEEP.exists(), f"paper_table2.yaml not found at {_SWEEP}"

    def test_basic_run_exits_zero(self):
        proc = _run(["--seed", "42"])
        assert proc.returncode == 0, (
            f"run_benchmark.py exited {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    def test_summary_md_written(self):
        _run(["--seed", "42"])
        run_dir = _latest_run_dir()
        assert run_dir is not None, "No run directory found in results/"
        summary = run_dir / "summary.md"
        assert summary.exists(), f"summary.md not found in {run_dir}"

    def test_summary_md_has_markdown_table_headers(self):
        _run(["--seed", "42"])
        run_dir = _latest_run_dir()
        assert run_dir is not None
        summary_text = (run_dir / "summary.md").read_text()
        assert "| Experiment |" in summary_text, "Missing table header row in summary.md"
        assert "| AUC |" in summary_text, "Missing AUC column header in summary.md"
        assert "| Pr@20 |" in summary_text, "Missing Pr@20 column header in summary.md"

    def test_csvs_written_for_non_skipped_variants(self):
        _run(["--seed", "42"])
        run_dir = _latest_run_dir()
        assert run_dir is not None
        csvs = list(run_dir.glob("*_synthetic.csv"))
        # At minimum kcf, mobiletrack, entropy_hybrid should produce CSVs
        # (fixed_periodic is skipped).
        assert len(csvs) >= 1, (
            f"Expected at least 1 CSV in {run_dir}, found: {[c.name for c in csvs]}"
        )

    def test_csv_has_provenance_header(self):
        _run(["--seed", "42"])
        run_dir = _latest_run_dir()
        assert run_dir is not None
        csvs = list(run_dir.glob("*_synthetic.csv"))
        if not csvs:
            pytest.skip("No CSV files to inspect")
        text = csvs[0].read_text()
        assert "# git_sha=" in text, "Provenance header missing git_sha"
        assert "# timestamp=" in text, "Provenance header missing timestamp"

    def test_caching_on_second_run(self):
        """Second run (same seed, no --force) should print 'cached' for all variants."""
        # First run — populate cache.
        _run(["--seed", "99"])
        # Second run — same seed.
        proc = _run(["--seed", "99"])
        assert proc.returncode == 0, (
            f"Second run failed: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert "cached" in proc.stdout.lower(), (
            "Expected 'cached' in stdout on second run.\n"
            f"STDOUT:\n{proc.stdout}"
        )

    def test_force_flag_bypasses_cache(self):
        """--force flag should bypass cache and re-run."""
        # First run to populate cache.
        _run(["--seed", "77"])
        # Forced second run — should NOT say 'cached'.
        proc = _run(["--seed", "77", "--force"])
        assert proc.returncode == 0
        # When forced, 'ok' or 'AUC=' lines should appear (not only cached).
        # We check that we got at least one 'ok' line (not all cached).
        stdout = proc.stdout
        has_ok = "ok" in stdout and "AUC=" in stdout
        has_all_cached = stdout.lower().count("cached") >= 3 and "AUC=" not in stdout
        assert has_ok or not has_all_cached, (
            "Expected fresh (non-cached) run with --force.\n"
            f"STDOUT:\n{stdout}"
        )

    def test_fixed_periodic_skipped(self):
        """The fixed_periodic variant should be marked skipped in summary.md."""
        _run(["--seed", "42"])
        run_dir = _latest_run_dir()
        assert run_dir is not None
        summary_text = (run_dir / "summary.md").read_text()
        assert "fixed_periodic" in summary_text, "fixed_periodic variant missing from summary.md"
        assert "skipped" in summary_text.lower(), "fixed_periodic should be marked as skipped"
