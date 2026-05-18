# Engineer-owned package namespace.
"""Visualization utilities (signals, success curves, overlays)."""

from __future__ import annotations

from uav_tracker.viz.entropy_plot import plot_entropy_timeline
from uav_tracker.viz.ope_curves import plot_success_curve, plot_precision_curve

# Engineer B's overlay/video modules — imported defensively so that the
# package can still be imported even if those files aren't written yet.
try:
    from uav_tracker.viz.overlay import draw_frame_overlay  # type: ignore[import]
except ImportError:
    draw_frame_overlay = None  # type: ignore[assignment]

try:
    from uav_tracker.viz.video import write_mp4  # type: ignore[import]
except ImportError:
    write_mp4 = None  # type: ignore[assignment]

__all__ = [
    "plot_entropy_timeline",
    "plot_success_curve",
    "plot_precision_curve",
    "draw_frame_overlay",
    "write_mp4",
]
