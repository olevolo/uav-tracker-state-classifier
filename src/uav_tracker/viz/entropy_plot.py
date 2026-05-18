"""Entropy-timeline visualization (Phase 8).

Renders H̄(t) with tier background bands, E_hi / E_lo threshold lines,
and switch-event markers.  Uses the Agg backend so tests run without X.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless — must come before any other matplotlib import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Tier colour palette (green / orange / red)
_TIER_COLORS = {
    0: "#d4edda",  # light green
    1: "#ffeeba",  # light orange/amber
    2: "#f8d7da",  # light red
}
_TIER_LABELS = {0: "LIGHT", 1: "MEDIUM", 2: "DEEP"}


def plot_entropy_timeline(
    H_bar: np.ndarray,
    tier_sequence: np.ndarray,
    E_hi: float,
    E_lo: float,
    switch_events: list[int],
    out_path: Path,
) -> None:
    """Render entropy-vs-time plot and save to ``out_path`` as PNG.

    Parameters
    ----------
    H_bar:
        1-D array of per-frame entropy values (length N).
    tier_sequence:
        1-D integer array of active tier per frame (length N).
    E_hi:
        Upper threshold; plotted as a horizontal dashed line.
    E_lo:
        Lower threshold; plotted as a horizontal dashed line.
    switch_events:
        Frame indices at which a tier switch occurred.  Annotated with
        a vertical dashed line.
    out_path:
        Destination PNG file path.
    """
    H_bar = np.asarray(H_bar, dtype=float)
    tier_sequence = np.asarray(tier_sequence, dtype=int)
    N = len(H_bar)
    frames = np.arange(N)

    fig, ax = plt.subplots(figsize=(10, 4))

    # ------------------------------------------------------------------
    # Coloured background bands by tier
    # ------------------------------------------------------------------
    if N > 0:
        # Walk tier_sequence and paint contiguous bands.
        run_start = 0
        current_tier = int(tier_sequence[0])
        for i in range(1, N + 1):
            next_tier = int(tier_sequence[i]) if i < N else -1
            if next_tier != current_tier or i == N:
                color = _TIER_COLORS.get(current_tier, "#ffffff")
                ax.axvspan(run_start - 0.5, i - 0.5, color=color, alpha=0.6, zorder=0)
                run_start = i
                current_tier = next_tier

    # ------------------------------------------------------------------
    # Entropy line
    # ------------------------------------------------------------------
    ax.plot(frames, H_bar, color="#1f77b4", linewidth=1.5, zorder=3, label=r"$\bar{H}(t)$")

    # ------------------------------------------------------------------
    # Threshold lines
    # ------------------------------------------------------------------
    ax.axhline(E_hi, color="red", linestyle="--", linewidth=1.0, zorder=2, label=f"$E_{{hi}}={E_hi:.2f}$")
    ax.axhline(E_lo, color="darkorange", linestyle="--", linewidth=1.0, zorder=2, label=f"$E_{{lo}}={E_lo:.2f}$")

    # ------------------------------------------------------------------
    # Vertical switch markers
    # ------------------------------------------------------------------
    for idx, frame_idx in enumerate(switch_events):
        if 0 <= frame_idx < N:
            ax.axvline(frame_idx, color="purple", linestyle=":", linewidth=1.0, zorder=4)
            # Determine transition label from tier_sequence if possible.
            if frame_idx > 0 and frame_idx < len(tier_sequence):
                prev_tier = int(tier_sequence[frame_idx - 1])
                curr_tier = int(tier_sequence[frame_idx])
                prev_label = _TIER_LABELS.get(prev_tier, f"T{prev_tier}")
                curr_label = _TIER_LABELS.get(curr_tier, f"T{curr_tier}")
                ann_label = f"{prev_label}→{curr_label}"
            else:
                ann_label = "SWITCH"
            y_ann = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(H_bar) * 0.95
            ax.annotate(
                ann_label,
                xy=(frame_idx, H_bar[frame_idx] if frame_idx < N else 0),
                xytext=(frame_idx + max(1, N // 50), y_ann),
                fontsize=6,
                color="purple",
                rotation=90,
                va="top",
                ha="left",
                zorder=5,
            )

    # ------------------------------------------------------------------
    # Legend patches for tier bands
    # ------------------------------------------------------------------
    tier_patches = [
        mpatches.Patch(color=_TIER_COLORS[t], alpha=0.6, label=f"Tier {t} ({_TIER_LABELS[t]})")
        for t in sorted(_TIER_COLORS)
    ]
    handles, labels_existing = ax.get_legend_handles_labels()
    ax.legend(handles=handles + tier_patches, loc="upper right", fontsize=7, framealpha=0.8)

    # ------------------------------------------------------------------
    # Labels and layout
    # ------------------------------------------------------------------
    ax.set_xlabel("Frame index")
    ax.set_ylabel(r"$\bar{H}(t)$ (entropy)")
    ax.set_title("Entropy Timeline with Tier Scheduling")
    ax.set_xlim(-0.5, max(N - 0.5, 0.5))
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
