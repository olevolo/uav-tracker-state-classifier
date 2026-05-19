"""integrate.py — SALT-RD integration with frozen SALTRunner.

Thin adapter layer that:
  1. Wraps the frozen SALTRunner (read-only, no mutations to src/uav_tracker/).
  2. Extracts 28 FEATURE_NAMES scalar features per frame from TelemetryEntry.aux.
  3. Maintains a rolling FeatureBuffer for GRU input.
  4. Calls the SALTRD model to produce per-head risk probabilities.
  5. Translates probabilities into TrackerAction policy recommendations.

IMPORTANT: src/uav_tracker/ is FROZEN — this module must never modify files
under that path.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from salt_r.collect_features import FEATURE_NAMES, N_FEATURES
from salt_r.policy import TrackerAction, RiskThresholds, DEFAULT_THRESHOLDS, apply_policy


# ---------------------------------------------------------------------------
# FeatureBuffer — rolling window of per-frame scalar features
# ---------------------------------------------------------------------------


class FeatureBuffer:
    """Rolling window of per-frame scalar features for GRU input."""

    def __init__(self, window_size: int = 20, n_features: int = 28) -> None:
        self.window_size = window_size
        self.n_features = n_features
        self._buf: deque = deque(maxlen=window_size)

    def push(self, features: np.ndarray) -> None:
        """Add one frame's features (n_features,) to the buffer."""
        self._buf.append(features.astype(np.float32))

    def get_window(self) -> np.ndarray | None:
        """Return (window_size, n_features) float32 array, or None if not full."""
        if len(self._buf) < self.window_size:
            return None
        return np.stack(list(self._buf), axis=0)

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Per-frame feature extraction — mirrors collect_features.py collect_sequence()
# ---------------------------------------------------------------------------


def extract_features_from_entry(
    entry: Any,
    prev_entry: Any | None,
    prev_frame: np.ndarray | None,
    curr_frame: np.ndarray,
    buffer: FeatureBuffer,
) -> np.ndarray:
    """Extract 28 scalar features from a TelemetryEntry for SALTRD input.

    Mirrors the logic in collect_features.py but operates on a single live
    frame rather than a batch. Feature order must match FEATURE_NAMES.

    Features 0-8:  score map stats  (from entry.aux)
    Features 9-14: temporal rolling windows (from buffer history)
    Features 15-21: target dynamics (from entry.bbox vs prev_entry.bbox)
    Features 22-27: camera/flow     (optical flow between prev_frame and curr_frame)

    Parameters
    ----------
    entry:
        Current frame TelemetryEntry (from SALTRunner.run).
    prev_entry:
        Previous frame TelemetryEntry, or None on frame 0.
    prev_frame:
        Previous raw BGR frame array, or None on frame 0.
    curr_frame:
        Current raw BGR frame array.
    buffer:
        FeatureBuffer containing pushed features for frames prior to this one.
        Used for temporal rolling-window features (indices 9-14).

    Returns
    -------
    np.ndarray of shape (N_FEATURES,) = (28,), dtype float32.
    """
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    # ------------------------------------------------------------------
    # Features 0-8: score map stats
    # ------------------------------------------------------------------
    sms = entry.aux.get("score_map_stats", {})
    apce_raw = float(entry.aux.get("apce_raw", 0.0))
    feat[0] = apce_raw
    feat[1] = apce_raw / 256.0                            # apce_norm
    feat[2] = float(entry.aux.get("psr_raw", 0.0))       # psr
    feat[3] = float(entry.aux.get("entropy_raw", 0.0))   # entropy
    feat[4] = float(sms.get("peak_margin", 0.0))
    feat[5] = float(sms.get("peak_width", 0))
    feat[6] = float(sms.get("n_secondary", 0))
    feat[7] = float(sms.get("peak_distance", 0.0))
    feat[8] = float(sms.get("heatmap_mass_topk", 0.0))

    # ------------------------------------------------------------------
    # Features 9-14: temporal rolling windows from buffer history
    # ------------------------------------------------------------------
    # buffer contains frames *before* this one (current frame not yet pushed).
    # Build a view of historical apce/entropy/peak_margin from the deque.
    buf_list = list(buffer._buf)   # list of (N_FEATURES,) arrays, oldest first
    n_hist = len(buf_list)

    if n_hist > 0:
        hist_apce = np.array([f[0] for f in buf_list], dtype=np.float32)
        hist_ent  = np.array([f[3] for f in buf_list], dtype=np.float32)
        hist_pm   = np.array([f[4] for f in buf_list], dtype=np.float32)

        # apce_ratio_5: current / mean of last 5 historical frames
        w5 = hist_apce[-5:]
        feat[9] = apce_raw / (w5.mean() + 1e-8) if len(w5) > 0 else 1.0

        # apce_ratio_20: current / mean of last 20 historical frames
        w20 = hist_apce[-20:]
        feat[10] = apce_raw / (w20.mean() + 1e-8) if len(w20) > 0 else 1.0

        # entropy_delta_5: current - mean of last 5 historical entropy
        e5 = hist_ent[-5:]
        feat[11] = feat[3] - (e5.mean() if len(e5) > 0 else feat[3])

        # peak_margin_delta_5: current - mean of last 5 historical peak_margin
        pm5 = hist_pm[-5:]
        feat[12] = feat[4] - (pm5.mean() if len(pm5) > 0 else feat[4])

        # confirmed_streak: consecutive historical frames with APCE > 100,
        # counting from most recent backward, then add current if APCE > 100
        streak = 0
        for k in range(n_hist - 1, -1, -1):
            if buf_list[k][0] > 100.0:
                streak += 1
            else:
                break
        if apce_raw > 100.0:
            streak += 1
        feat[13] = float(streak)

        # low_conf_streak: consecutive frames (history + current) with APCE < 50
        low_s = 0
        for k in range(n_hist - 1, -1, -1):
            if buf_list[k][0] < 50.0:
                low_s += 1
            else:
                break
        if apce_raw < 50.0:
            low_s += 1
        feat[14] = float(low_s)
    else:
        # No history yet — defaults
        feat[9]  = 1.0
        feat[10] = 1.0
        feat[11] = 0.0
        feat[12] = 0.0
        feat[13] = 1.0 if apce_raw > 100.0 else 0.0
        feat[14] = 1.0 if apce_raw < 50.0 else 0.0

    # ------------------------------------------------------------------
    # Features 15-21: target dynamics
    # ------------------------------------------------------------------
    if prev_entry is None:
        feat[15:22] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5]
    else:
        cur = entry.bbox
        prv = prev_entry.bbox
        diag = max((cur.w ** 2 + cur.h ** 2) ** 0.5, 1.0)
        cx_v = ((cur.x + cur.w / 2) - (prv.x + prv.w / 2)) / diag
        cy_v = ((cur.y + cur.h / 2) - (prv.y + prv.h / 2)) / diag
        speed = (cx_v ** 2 + cy_v ** 2) ** 0.5

        # acceleration: compare current speed against previous bbox pair
        accel = 0.0
        if len(buf_list) >= 2:
            # second-to-last frame bbox is unavailable directly, so fall back
            # to estimating from the buffer's second-most-recent entry
            # (buf_list[-2] was the t-2 frame's features — but bbox is not stored).
            # We approximate accel from speed comparison: prev entry had its own
            # speed stored in feat[17] of the last buffer entry.
            prev_speed = buf_list[-1][17] if len(buf_list) >= 1 else 0.0
            accel = abs(speed - prev_speed)

        scale_r = (cur.w * cur.h) / max(prv.w * prv.h, 1.0)
        asp_d   = (cur.w / max(cur.h, 1e-3)) - (prv.w / max(prv.h, 1e-3))

        h_img, w_img = curr_frame.shape[:2]
        cx = cur.x + cur.w / 2
        cy = cur.y + cur.h / 2
        search_sz = max(cur.w, cur.h) * 4.0
        dist_border = min(cx, cy, w_img - cx, h_img - cy) / max(search_sz, 1.0)

        feat[15] = float(cx_v)
        feat[16] = float(cy_v)
        feat[17] = float(speed)
        feat[18] = float(accel)
        feat[19] = float(scale_r)
        feat[20] = float(asp_d)
        feat[21] = float(np.clip(dist_border, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Features 22-27: camera / optical flow
    # ------------------------------------------------------------------
    if prev_frame is None or curr_frame is None:
        feat[22] = 0.0
        feat[23] = 0.0
        feat[24] = 0.0
        feat[25] = 0.5
        feat[26] = 0.0
        feat[27] = 0.5
    else:
        gray_cur  = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_prev = cv2.cvtColor(prev_frame,  cv2.COLOR_BGR2GRAY).astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_cur, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.hypot(flow[..., 0], flow[..., 1])
        global_flow_mag = float(mag.mean())

        bbox = entry.bbox
        h_img, w_img = curr_frame.shape[:2]
        x1, y1 = max(0, int(bbox.x)),         max(0, int(bbox.y))
        x2, y2 = min(w_img, int(bbox.x + bbox.w)), min(h_img, int(bbox.y + bbox.h))
        if x2 > x1 and y2 > y1:
            target_mag = float(mag[y1:y2, x1:x2].mean())
            tflow      = flow[y1:y2, x1:x2]
        else:
            target_mag, tflow = global_flow_mag, flow

        ego_residual = abs(target_mag - global_flow_mag)

        gf_mean = flow.mean(axis=(0, 1))
        tf_mean = tflow.mean(axis=(0, 1))
        denom   = np.linalg.norm(gf_mean) * np.linalg.norm(tf_mean) + 1e-8
        flow_cos = float(np.dot(gf_mean, tf_mean) / denom)
        flow_iou = float(np.clip((flow_cos + 1.0) / 2.0, 0.0, 1.0))

        # flow_consistency: compare global mag with previous frame's
        if n_hist >= 1:
            prev_gmag = buf_list[-1][22]
            flow_consistency = 1.0 / (1.0 + abs(global_flow_mag - float(prev_gmag)))
        else:
            flow_consistency = 0.5

        feat[22] = global_flow_mag
        feat[23] = target_mag
        feat[24] = ego_residual
        feat[25] = flow_iou
        feat[26] = ego_residual        # flow_residual = ego_residual (v0, mirrors collect_features)
        feat[27] = flow_consistency

    return feat


# ---------------------------------------------------------------------------
# RiskEntry — per-frame output dataclass
# ---------------------------------------------------------------------------


@dataclass
class RiskEntry:
    """Per-frame output: base tracker entry + SALT-RD risk assessment."""
    entry: Any                        # TelemetryEntry
    probs: dict[str, float]           # raw head probabilities (empty if window not full)
    action: TrackerAction             # policy recommendation
    window_full: bool                 # False for first window_size-1 frames


# ---------------------------------------------------------------------------
# Helper — read enable_salt_rd flag from YAML config
# ---------------------------------------------------------------------------


def _read_enable_salt_rd(config_path: str) -> bool:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    return bool(cfg.get("enable_salt_rd", False))


# ---------------------------------------------------------------------------
# SALTRDRunner — wraps frozen SALTRunner with SALTRD risk head
# ---------------------------------------------------------------------------


class SALTRDRunner:
    """SALT-RD runtime wrapper around frozen SALTRunner.

    Adds per-frame risk prediction and policy recommendations on top of
    the frozen SGLATrack/SALTRunner pipeline. The frozen runner is never
    modified — SALT-RD observes its outputs and produces risk signals.

    Usage::

        runner = SALTRDRunner.from_config(
            salt_config="configs/prod/salt.yaml",
            saltrd_checkpoint="saltr/checkpoints/saltrd_best.pt",
        )
        for risk in runner.run_with_risk(sequence):
            if risk.action.compute_mode == "cheap":
                pass  # skip expensive operations
            bbox = risk.entry.bbox
    """

    def __init__(
        self,
        base_runner: Any,
        saltrd_model: Any | None = None,
        window_size: int = 20,
        device: str = "cpu",
        enable_salt_rd: bool = True,
    ) -> None:
        self.base_runner = base_runner
        self.saltrd_model = saltrd_model
        self.window_size = window_size
        self.device = device
        self.enable_salt_rd = enable_salt_rd
        self._buffer = FeatureBuffer(window_size=window_size)
        self._prev_entry: Any | None = None
        self._prev_frame: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        salt_config: str,
        saltrd_checkpoint: str | None = None,
        device: str = "cpu",
    ) -> "SALTRDRunner":
        """Build SALTRDRunner from config paths.

        Parameters
        ----------
        salt_config:
            Path to the frozen SALT/SGLATrack YAML config.
        saltrd_checkpoint:
            Optional path to a SALTRD checkpoint (.pt).  If None or the
            file does not exist, the runner operates in observation-only
            mode (probs always empty, action always safe defaults).
        device:
            Torch device for SALTRD model inference ("cpu" or "cuda").
        """
        sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
        from uav_tracker.salt_runner import SALTRunner  # frozen
        base_runner = SALTRunner.from_config(salt_config)

        saltrd_model = None
        if saltrd_checkpoint and Path(saltrd_checkpoint).exists():
            from salt_r.model import build_model
            saltrd_model = build_model(saltrd_checkpoint, device=device)
            saltrd_model.eval()

        cfg_enabled = _read_enable_salt_rd(salt_config)
        return cls(
            base_runner=base_runner,
            saltrd_model=saltrd_model,
            device=device,
            enable_salt_rd=cfg_enabled,
        )

    def run_with_risk(
        self,
        sequence: Any,
    ) -> Iterator[RiskEntry]:
        """Run sequence through base tracker + SALT-RD risk head.

        Yields one RiskEntry per frame. For the first window_size-1 frames,
        probs is empty and action defaults are safe (full compute, allow update).

        Parameters
        ----------
        sequence:
            Dataset sequence object with .frames (iterable of BGR arrays),
            .ground_truth, and .name — compatible with SALTRunner.run().
        """
        self._buffer.reset()
        self._prev_entry = None
        self._prev_frame = None

        frames_list = list(sequence.frames)

        for idx, (entry, frame) in enumerate(zip(
            self.base_runner.run(sequence), frames_list
        )):
            features = extract_features_from_entry(
                entry, self._prev_entry, self._prev_frame, frame, self._buffer
            )
            self._buffer.push(features)

            probs: dict[str, float] = {}
            window_full = False

            if self.enable_salt_rd and self.saltrd_model is not None:
                window = self._buffer.get_window()
                if window is not None:
                    window_full = True
                    probs = self.saltrd_model.predict_single(
                        window, device=self.device
                    )

            action = apply_policy(probs)

            self._prev_entry = entry
            self._prev_frame = frame

            yield RiskEntry(
                entry=entry,
                probs=probs,
                action=action,
                window_full=window_full,
            )
