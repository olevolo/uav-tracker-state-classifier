"""Per-frame overlay renderer (Phase 8).

Draws bbox rectangles (tier-colored), a top-left text badge showing
tier + FPS on a semi-transparent background, and bottom-left signal
gauges with horizontal bar indicators.

Public API
----------
draw_frame_overlay(frame, bbox, tier, signals, fps, gt_bbox=None) -> np.ndarray
    Returns an annotated copy of ``frame`` (BGR uint8); input is never mutated.

Ground-truth rectangle
----------------------
When ``gt_bbox`` is supplied a thin solid white rectangle is drawn with
line thickness 1.  Using a solid thin rect is preferable to a dashed
simulation because cv2 has no native dash support and pixel-level
simulation would be brittle across resolution scales.
"""

from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np

from uav_tracker.types import BBox, SignalReport

# Tier -> BGR colour (same indices used throughout PLAN § 8).
_TIER_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),    # green
    1: (0, 165, 255),  # orange
    2: (0, 0, 255),    # red
}
_DEFAULT_COLOUR: tuple[int, int, int] = (128, 128, 128)  # fallback for unknown tiers

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_LINE_TYPE = cv2.LINE_AA

# Badge geometry
_BADGE_PAD_X = 6
_BADGE_PAD_Y = 4

# Gauge geometry
_GAUGE_BAR_W = 120    # maximum bar width in pixels
_GAUGE_BAR_H = 6      # bar height in pixels
_GAUGE_TEXT_H = 16    # text row height including small gap
_GAUGE_LEFT_PAD = 6
_GAUGE_BOTTOM_PAD = 8


def draw_frame_overlay(
    frame: np.ndarray,
    bbox: BBox | None,
    tier: int,
    signals: Mapping[str, SignalReport],
    fps: float,
    gt_bbox: BBox | None = None,
) -> np.ndarray:
    """Return an annotated copy of ``frame`` with overlaid tracking info.

    Parameters
    ----------
    frame:
        Source BGR uint8 image.  Never mutated.
    bbox:
        Tracker estimate.  When ``None`` the bbox rectangle is skipped but
        the badge and gauges are still rendered.
    tier:
        Active tier index.  0 = fast/light, 1 = medium, 2 = deep.
    signals:
        Mapping of signal name to ``SignalReport`` for the current frame.
        Values are expected in ``[0, 1]``; out-of-range values are clamped
        for the bar indicator only.
    fps:
        Current tracking frame-rate (shown in the badge).
    gt_bbox:
        Optional ground-truth bbox drawn as a thin solid white rectangle.

    Returns
    -------
    np.ndarray
        Annotated frame (same shape and dtype as input).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    tier_colour = _TIER_COLOURS.get(tier, _DEFAULT_COLOUR)

    # ------------------------------------------------------------------ #
    # Ground-truth rectangle (thin white, drawn first so bbox appears on
    # top when the two overlap).
    # ------------------------------------------------------------------ #
    if gt_bbox is not None:
        x0 = int(round(gt_bbox.x))
        y0 = int(round(gt_bbox.y))
        x1 = int(round(gt_bbox.x + gt_bbox.w))
        y1 = int(round(gt_bbox.y + gt_bbox.h))
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), thickness=1,
                      lineType=cv2.LINE_4)

    # ------------------------------------------------------------------ #
    # Tracker bbox rectangle
    # ------------------------------------------------------------------ #
    if bbox is not None:
        x0 = int(round(bbox.x))
        y0 = int(round(bbox.y))
        x1 = int(round(bbox.x + bbox.w))
        y1 = int(round(bbox.y + bbox.h))
        cv2.rectangle(out, (x0, y0), (x1, y1), tier_colour, thickness=2,
                      lineType=_LINE_TYPE)

    # ------------------------------------------------------------------ #
    # Top-left badge: "tier N  XX.X FPS"
    # ------------------------------------------------------------------ #
    badge_text = f"tier {tier}  {fps:.1f} FPS"
    (text_w, text_h), baseline = cv2.getTextSize(
        badge_text, _FONT, _FONT_SCALE, _FONT_THICKNESS
    )
    badge_x1 = _BADGE_PAD_X + text_w + _BADGE_PAD_X
    badge_y1 = _BADGE_PAD_Y + text_h + baseline + _BADGE_PAD_Y

    # Draw semi-transparent black background via weighted blend.
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (badge_x1, badge_y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # Draw badge text.
    text_y = _BADGE_PAD_Y + text_h
    cv2.putText(
        out, badge_text,
        (_BADGE_PAD_X, text_y),
        _FONT, _FONT_SCALE, (255, 255, 255),
        _FONT_THICKNESS, _LINE_TYPE,
    )

    # ------------------------------------------------------------------ #
    # Bottom-left signal gauges
    # ------------------------------------------------------------------ #
    if signals:
        n_signals = len(signals)
        # Each gauge occupies: text row + bar row + small gap
        gauge_row_h = _GAUGE_TEXT_H + _GAUGE_BAR_H + 4
        total_gauge_h = n_signals * gauge_row_h + _GAUGE_BOTTOM_PAD

        # Determine background width based on text + bar.
        sample_label = "x" * 16 + "  0.00"
        (label_w, _), _ = cv2.getTextSize(sample_label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        gauge_bg_w = _GAUGE_LEFT_PAD + max(label_w, _GAUGE_BAR_W) + _GAUGE_LEFT_PAD

        bg_y0 = h - total_gauge_h
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, bg_y0), (gauge_bg_w, h), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay2, 0.55, out, 0.45, 0, out)

        cursor_y = bg_y0 + 2
        for name, report in signals.items():
            value = report.value

            # Text row.
            label = f"{name[:16]:16s} {value:.2f}"
            cursor_y += _GAUGE_TEXT_H
            cv2.putText(
                out, label,
                (_GAUGE_LEFT_PAD, cursor_y),
                _FONT, _FONT_SCALE, (200, 200, 200),
                _FONT_THICKNESS, _LINE_TYPE,
            )

            # Bar row.
            bar_y0 = cursor_y + 2
            bar_y1 = bar_y0 + _GAUGE_BAR_H
            # Background bar (dark).
            cv2.rectangle(
                out,
                (_GAUGE_LEFT_PAD, bar_y0),
                (_GAUGE_LEFT_PAD + _GAUGE_BAR_W, bar_y1),
                (60, 60, 60), thickness=-1,
            )
            # Filled portion.
            fill_frac = float(np.clip(value, 0.0, 1.0))
            fill_w = int(round(fill_frac * _GAUGE_BAR_W))
            if fill_w > 0:
                cv2.rectangle(
                    out,
                    (_GAUGE_LEFT_PAD, bar_y0),
                    (_GAUGE_LEFT_PAD + fill_w, bar_y1),
                    tier_colour, thickness=-1,
                )

            cursor_y += _GAUGE_BAR_H + 4  # gap between gauges

    return out


__all__ = ["draw_frame_overlay"]
