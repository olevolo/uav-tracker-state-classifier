# Engineer-owned package namespace.
"""Switch-signal plugins. Architect owns ``base.py``."""

from . import tracker_confidence  # noqa: F401 — triggers @SIGNALS.register
from . import motion_entropy  # noqa: F401 — triggers @SIGNALS.register("motion_entropy")
