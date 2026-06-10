# Engineer-owned package namespace.
"""Tracker plugins. Architect owns ``base.py`` (Protocol).

Adapter modules self-register with ``uav_tracker.registry.TRACKERS`` at
import time. Callers (e.g. ``tools/profile_pipeline.py``) import the
module they need by name (``from uav_tracker.trackers import sglatrack``)
to trigger registration; this package does not eagerly import every
adapter, since some require optional dependencies (CLIP, AVTrack repo,
SiamBAN repo, etc.) that may not be present.

Registry names:
    sglatrack, avtrack, ortrack, ostrack, evptrack, fartrack,
    uetrack, kcf_henriques, mobiletrack
"""

from __future__ import annotations


def _register_mobiletrack() -> None:
    """Side-effect import for ``from uav_tracker.trackers import mobiletrack``.

    The module lives under ``trackers/siamese/`` for organisation, but the
    canonical registry name is ``mobiletrack`` (no ``siamese.`` prefix).
    Importing the module registers the adapter in ``TRACKERS``. We re-export
    it under ``trackers.mobiletrack`` so callers can use either path.
    """
    import sys
    from uav_tracker.trackers.siamese import mobiletrack as _mt
    sys.modules.setdefault(f"{__name__}.mobiletrack", _mt)


try:
    _register_mobiletrack()
except Exception:
    # Don't crash the trackers package if optional deps for one adapter are
    # missing. The adapter self-reports stub mode at construction time.
    pass
