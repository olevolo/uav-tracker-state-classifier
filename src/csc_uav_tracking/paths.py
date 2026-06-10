"""Canonical path resolution for project-local data / weights / outputs.

The project anchors its runtime artefacts under the repo root by default::

    <repo_root>/data/       ← datasets (symlinkable to external storage)
    <repo_root>/weights/    ← tracker weights
    <repo_root>/outputs/    ← CSVs, figures, ablation sweeps
    <repo_root>/demos/      ← rendered MP4 demo videos

Environment variables ``UAV_DATA_ROOT`` / ``UAV_WEIGHTS_ROOT`` /
``UAV_RESULTS_ROOT`` override the corresponding defaults when set.
``demos`` has no env var — always project-local.

These helpers are used by the CLI (`evaluate`, `ablate`, `demo`, `figures`)
and by scripts (`run_benchmark.py`, `run_ablation.py`).  Callers should
prefer ``data_root()`` / ``weights_root()`` / ``outputs_root()`` /
``demos_root()`` over inlining ``os.environ.get`` + default fallbacks.
"""

from __future__ import annotations

import os
from pathlib import Path

# This file lives at ``src/uav_tracker/paths.py`` — three parents up is the
# repo root (``.../uav-entropy-tracker``).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _resolve(env_var: str | None, default_subdir: str) -> Path:
    if env_var is not None:
        value = os.environ.get(env_var, "").strip()
        if value:
            return Path(value).expanduser().resolve()
    return REPO_ROOT / default_subdir


def data_root() -> Path:
    """Dataset root. Defaults to ``<repo>/data/``; ``$UAV_DATA_ROOT`` overrides."""
    return _resolve("UAV_DATA_ROOT", "data")


def weights_root() -> Path:
    """Tracker-weights root. Defaults to ``<repo>/weights/``; ``$UAV_WEIGHTS_ROOT`` overrides."""
    return _resolve("UAV_WEIGHTS_ROOT", "weights")


def outputs_root() -> Path:
    """Results / CSV / figures root. Defaults to ``<repo>/outputs/``; ``$UAV_RESULTS_ROOT`` overrides."""
    return _resolve("UAV_RESULTS_ROOT", "outputs")


def demos_root() -> Path:
    """Rendered demo-MP4 root. Always ``<repo>/demos/`` (no env override)."""
    return REPO_ROOT / "demos"


__all__ = ["REPO_ROOT", "data_root", "weights_root", "outputs_root", "demos_root"]
