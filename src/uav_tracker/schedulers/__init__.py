# Engineer-owned package namespace.
"""Scheduler plugins. Architect owns ``base.py``."""

from . import multi_tier  # noqa: F401 — triggers @SCHEDULERS.register("multi_tier")
