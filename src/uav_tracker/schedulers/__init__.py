# Engineer-owned package namespace.
"""Scheduler plugins. Architect owns ``base.py``."""

from . import hysteresis_binary  # noqa: F401 — triggers @SCHEDULERS.register
from . import cusum  # noqa: F401 — triggers @SCHEDULERS.register("cusum")
from . import adaptive_threshold  # noqa: F401 — triggers @SCHEDULERS.register("adaptive_threshold")
from . import trajectory_aware  # noqa: F401 — triggers @SCHEDULERS.register("trajectory_aware")
from . import multi_tier  # noqa: F401 — triggers @SCHEDULERS.register("multi_tier")
