"""Re-export shim for ORTrack at the legacy flat path.

The canonical implementation lives under ``trackers.transformer.ortrack``;
this module re-exports it so existing callers using
``from uav_tracker.trackers import ortrack`` keep working.
"""
from uav_tracker.trackers.transformer.ortrack import *  # noqa: F401,F403
from uav_tracker.trackers.transformer import ortrack as _impl

# Re-export module attributes so ``uav_tracker.trackers.ortrack.X``
# resolves to ``uav_tracker.trackers.transformer.ortrack.X``.
import sys as _sys
_sys.modules[__name__] = _impl
