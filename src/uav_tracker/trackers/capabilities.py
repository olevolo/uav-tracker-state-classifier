"""Tracker capability declarations for CSC-guided control.

Each adapter exposes a ``capabilities`` property returning a
``TrackerCapabilities`` instance.  ``run_with_csc.py`` gates every
control action on the relevant flag so the same script works across all
trackers without silent no-ops.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackerCapabilities:
    can_freeze_template: bool = False  # set_update_enabled() is wired and effective
    can_widen_search: bool = False     # search_factor is an instance var with a setter
    can_force_reinit: bool = True      # init() mid-sequence (all trackers support this)
    can_reject_bbox: bool = True       # output-level guard (all trackers)
    can_reduce_pruning: bool = False   # token / CE pruning rate can be lowered


__all__ = ["TrackerCapabilities"]
