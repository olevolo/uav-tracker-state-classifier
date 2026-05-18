"""MobileTrack — the paper's deep tracker (Oleksiuk & Velhosh 2026 §3.2).

**Reference RESOLVED (2026-05-17):**
  Xue, F. et al. (2022). "MobileTrack: Lightweight mobile network for UAV
  single object tracking." *IET Image Processing* 16, 3300–3313.
  DOI: 10.1049/ipr2.12553

  Architecture: MobileNetV2-based Siamese backbone + BAN head.
  Trained on: GOT-10k + LaSOT + COCO (standard SOT training split).
  UAV123 reported: AUC=0.690, Pr@20=77.3% (Oleksiuk & Velhosh 2026, Table 2).

**Obtaining weights:**
  1. Find the Xue et al. 2022 repository (IET Image Processing supplemental,
     or search "MobileTrack IET 2022 UAV tracking" on GitHub).
  2. Download the pretrained checkpoint (.pth) to
     ``$UAV_WEIGHTS_ROOT/mobiletrack/mobiletrack.pth``.
  3. Swap the backbone: replace ``SiamFCTracker`` inheritance with
     ``SiamBANTracker`` (siamban.py) once weights are confirmed compatible,
     or implement the MobileNetV2 backbone in this file.

**Current status:** this class registers as ``mobiletrack`` and inherits
the SiamFC AlexNet architecture as a placeholder until the actual
MobileTrack checkpoint is obtained and the MobileNetV2 backbone is wired in.

**Alternative now available:** ``SiamBANTracker`` (registered as "siamban")
provides the SiamBAN R50 bridge using weights already on disk at
``$UAV_WEIGHTS_ROOT/mobiletrack/siamban_r50_l234.pth``. Use "siamban"
as a stronger tier-1 baseline while sourcing the exact MobileTrack weights.

Weights slot: ``$UAV_WEIGHTS_ROOT/mobiletrack/mobiletrack.pth``.
Tier hint: 1 (same as SiamFC).
"""

from __future__ import annotations

import os

from uav_tracker.registry import TRACKERS
from uav_tracker.trackers.siamese.siamfc import SiamFCTracker


@TRACKERS.register("mobiletrack")
class MobileTrackTracker(SiamFCTracker):
    """MobileTrack Siamese tracker (paper's deep tier, placeholder architecture).

    See module docstring for the unresolved-reference caveat. This class
    inherits SiamFC's AlexNet-style backbone; the name + weights path are
    the only things distinguishing it from ``SiamFCTracker``.
    """

    name: str = "mobiletrack"
    tier_hint: int = 1

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float32",
        weights_path: str | None = None,
    ) -> None:
        if weights_path is None:
            from uav_tracker.paths import weights_root
            weights_path = str(weights_root() / "mobiletrack" / "mobiletrack.pth")
        super().__init__(device=device, dtype=dtype, weights_path=weights_path)
