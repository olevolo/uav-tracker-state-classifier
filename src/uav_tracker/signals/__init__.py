# Engineer-owned package namespace.
"""Switch-signal plugins. Architect owns ``base.py``."""

from . import tracker_confidence  # noqa: F401 — triggers @SIGNALS.register
from . import motion_entropy  # noqa: F401 — triggers @SIGNALS.register("motion_entropy")
from . import circular_resultant  # noqa: F401 — triggers @SIGNALS.register("circular_resultant")
from . import apce  # noqa: F401 — triggers @SIGNALS.register("apce")
from . import flow_divergence  # noqa: F401 — triggers @SIGNALS.register("flow_divergence")
