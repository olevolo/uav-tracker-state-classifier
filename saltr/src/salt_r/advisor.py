"""advisor.py — SALT-RD Stage 2 Advisory/Veto controller.

Computes the 32-dim feature vector (28 base telemetry + 4 pos-only RAM memory)
online from live SGLATrack outputs and gates template updates by predicted
p_false_confirmed.

Usage
-----
    from salt_r.advisor import SALTRDAdvisor

    advisor = SALTRDAdvisor(
        checkpoint="saltr/checkpoints/v2_1_memory/saltrd_best.pt",
        device="cpu",
    )
    tracker.set_salt_rd_advisor(advisor)

    # Each tracking step:
    track_state = tracker.update_with_state(frame, tsa_state)
    # advisor.step() is called automatically inside update_with_state()

    # Before template update:
    if not advisor.should_block_template_update():
        tracker.try_update_template(frame, bbox, apce, psr, frame_idx, cosine_sim)
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

_FC_BLOCK_DEFAULT = 0.60   # mirrors shadow_mode._FC_BLOCK

# RAM update gate thresholds (mirrors PositiveMemory.should_update())
_APCE_NORM_GATE  = 0.4    # apce_norm > 0.4  (apce > ~100)
_P_FC_GATE       = 0.20   # p_fc_prev < 0.20 (tracker not in risk)
_RAM_UPDATE_INTERVAL = 5  # min frames between RAM updates


# ---------------------------------------------------------------------------
# Online Positive RAM
# ---------------------------------------------------------------------------

class _OnlinePositiveRAM:
    """Minimal online positive-only RAM matching PositiveMemory from memory.py."""

    def __init__(self, max_slots: int = 6, update_interval: int = _RAM_UPDATE_INTERVAL) -> None:
        self._entries: deque[np.ndarray] = deque()  # stores normalised (D,) embeddings
        self._max_slots = max_slots
        self._update_interval = update_interval
        self._last_update_frame: int = -999
        self._current_frame: int = 0

    def reset(self) -> None:
        self._entries.clear()
        self._last_update_frame = -999
        self._current_frame = 0

    def should_update(self, apce_norm: float, p_fc_prev: float) -> bool:
        """Gate: apce high + not in false-confirmed risk + enough time elapsed."""
        if apce_norm <= _APCE_NORM_GATE:
            return False
        if p_fc_prev >= _P_FC_GATE:
            return False
        if (self._current_frame - self._last_update_frame) < self._update_interval:
            return False
        return True

    def add(self, embedding: np.ndarray, frame_idx: int) -> None:
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        if len(self._entries) >= self._max_slots:
            self._entries.popleft()
        self._entries.append(emb)
        self._last_update_frame = frame_idx

    def compute_features(self, query_emb: np.ndarray) -> np.ndarray:
        """Return (4,) array: [max_sim, mean_sim, recency_sim, update_age]."""
        update_age = float(
            9999 if self._last_update_frame < 0
            else self._current_frame - self._last_update_frame
        )
        if not self._entries or query_emb is None:
            return np.array([0.0, 0.0, 0.0, update_age], dtype=np.float32)

        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        entries = list(self._entries)
        sims = np.array([float(np.dot(q, e)) for e in entries], dtype=np.float32)

        max_sim  = float(sims.max())
        mean_sim = float(sims.mean())

        # recency-weighted: 0.9^age, age=0 is most recent
        weights = np.array([0.9 ** i for i in range(len(entries) - 1, -1, -1)],
                           dtype=np.float32)
        recency_sim = float((weights * sims).sum() / weights.sum())

        return np.array([max_sim, mean_sim, recency_sim, update_age], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main advisor
# ---------------------------------------------------------------------------

class SALTRDAdvisor:
    """Online SALT-RD Stage 2 advisory controller.

    Drop-in for SGLATrack: tracks the last ``window_size`` frames of 32-dim
    telemetry (28 base + 4 RAM memory) and emits p_fc after each frame.
    Template updates are vetoed when p_fc >= fc_block.

    Notes
    -----
    Flow features (indices 22–27) are zeroed: Farneback dense optical flow adds
    ~20ms/frame and is not needed for the primary false-confirmed signal. Pass
    ``prev_frame`` and ``curr_frame`` to ``step()`` to compute them via
    ``cv2.calcOpticalFlowFarneback``.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        fc_block: float = _FC_BLOCK_DEFAULT,
    ) -> None:
        self.device = device
        self.fc_block = fc_block

        # Ensure salt_r on path when called from outside saltr/src
        _src = str(Path(__file__).parents[1])
        if _src not in sys.path:
            sys.path.insert(0, _src)

        from salt_r.model import build_model
        import torch as _torch

        self._model = build_model(checkpoint, device=device)
        self._model.eval()

        ck = _torch.load(checkpoint, map_location="cpu")
        self._window_size: int = int(ck.get("window_size", 20))
        # v2.1 checkpoint: memory_dim=4, point_dim=0
        self._extra_dim: int = (
            int(ck.get("memory_dim", 0)) + int(ck.get("point_dim", 0))
        )
        self._n_features: int = int(ck.get("n_features", 28))
        self._total_dim: int = self._n_features + self._extra_dim

        # Rolling buffers for base features (max 25 so apce_ratio_20 works)
        self._apce_buf:   deque[float] = deque(maxlen=25)
        self._ent_buf:    deque[float] = deque(maxlen=10)
        self._pmarg_buf:  deque[float] = deque(maxlen=10)

        # Streak counters
        self._conf_streak: int = 0
        self._low_streak:  int = 0

        # Motion history
        self._prev_bbox: Optional[tuple[float, float, float, float]] = None  # cx,cy,w,h
        self._prev_speed: float = 0.0

        # Online RAM
        self._ram = _OnlinePositiveRAM()
        self._current_frame: int = 0

        # GRU window: deque of (total_dim,) float32 vectors
        self._window: deque[np.ndarray] = deque(maxlen=self._window_size)

        # Outputs
        self._last_p_fc: float = 0.0

        # Monitoring
        self.n_blocked: int = 0
        self.n_allowed: int = 0
        self.n_steps:   int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call on tracker re-init (new target / new sequence)."""
        self._apce_buf.clear()
        self._ent_buf.clear()
        self._pmarg_buf.clear()
        self._conf_streak = 0
        self._low_streak  = 0
        self._prev_bbox   = None
        self._prev_speed  = 0.0
        self._ram.reset()
        self._current_frame = 0
        self._window.clear()
        self._last_p_fc = 0.0

    def step(
        self,
        track_state: Any,
        frame_h: int,
        frame_w: int,
        search_embedding: Optional[np.ndarray] = None,
        prev_frame: Optional[np.ndarray] = None,
        curr_frame: Optional[np.ndarray] = None,
    ) -> float:
        """Ingest one tracker frame and return current p_fc.

        Parameters
        ----------
        track_state:
            TrackState returned by SGLATrack.update_with_state().
            Required attrs: apce, psr, response_entropy, score_map_stats, bbox.
        frame_h, frame_w:
            Raw frame dimensions for dist_to_search_border.
        search_embedding:
            Optional (192,) _last_search_score_weighted for RAM updates.
        prev_frame, curr_frame:
            Optional BGR frames for Farneback flow features (if None, flow=0).
        """
        t = self._current_frame
        self._current_frame += 1
        self._ram._current_frame = t
        self.n_steps += 1

        # ------------------------------------------------------------------
        # 1. Extract raw scalars from track_state
        # ------------------------------------------------------------------
        apce = float(getattr(track_state, "apce", 0.0))
        psr  = float(getattr(track_state, "psr",  0.0))
        ent  = float(getattr(track_state, "response_entropy", 0.0))
        sms  = getattr(track_state, "score_map_stats", {}) or {}
        peak_margin  = float(sms.get("peak_margin",       0.0))
        peak_width   = float(sms.get("peak_width",        0.0))
        n_sec        = float(sms.get("n_secondary",       0.0))
        peak_dist    = float(sms.get("peak_distance",     0.0))
        hm_mass      = float(sms.get("heatmap_mass_topk", 0.0))
        apce_norm = apce / 256.0

        # ------------------------------------------------------------------
        # 2. Rolling ratio / delta features
        # ------------------------------------------------------------------
        self._apce_buf.append(apce)
        self._ent_buf.append(ent)
        self._pmarg_buf.append(peak_margin)

        def _ratio(val: float, buf: deque, n: int) -> float:
            hist = list(buf)[-n:] if len(buf) >= n else list(buf)
            m = float(np.mean(hist)) if hist else 1.0
            return val / (m + 1e-8) if m > 1e-8 else 1.0

        def _delta(val: float, buf: deque, n: int) -> float:
            hist = list(buf)[-n:] if len(buf) >= n else list(buf)
            return val - float(np.mean(hist)) if hist else 0.0

        apce_r5  = _ratio(apce, self._apce_buf, 5)
        apce_r20 = _ratio(apce, self._apce_buf, 20)
        ent_d5   = _delta(ent, self._ent_buf, 5)
        pm_d5    = _delta(peak_margin, self._pmarg_buf, 5)

        # Streak counters
        if apce > 100.0:
            self._conf_streak += 1
            self._low_streak   = 0
        elif apce < 50.0:
            self._low_streak  += 1
            self._conf_streak  = 0
        else:
            self._conf_streak = 0
            self._low_streak  = 0

        # ------------------------------------------------------------------
        # 3. Dynamics from bbox
        # ------------------------------------------------------------------
        bbox = getattr(track_state, "bbox", None)
        if bbox is not None:
            cx = float(getattr(bbox, "x", 0.0) + getattr(bbox, "w", 0.0) / 2)
            cy = float(getattr(bbox, "y", 0.0) + getattr(bbox, "h", 0.0) / 2)
            bw = float(getattr(bbox, "w", 1.0))
            bh = float(getattr(bbox, "h", 1.0))
        else:
            cx = cy = 0.0
            bw = bh = 1.0

        if self._prev_bbox is not None:
            pcx, pcy, pw, ph = self._prev_bbox
            diag = math.sqrt(max(bw * bh, 1.0))
            vx   = (cx - pcx) / diag
            vy   = (cy - pcy) / diag
            speed  = math.sqrt(vx * vx + vy * vy)
            accel  = abs(speed - self._prev_speed)
            scale_r   = (bw * bh) / max(pw * ph, 1.0)
            ar_delta  = (bw / max(bh, 1e-4)) - (pw / max(ph, 1e-4))
        else:
            vx = vy = speed = accel = 0.0
            scale_r  = 1.0
            ar_delta = 0.0

        self._prev_bbox  = (cx, cy, bw, bh)
        self._prev_speed = speed

        border_d   = min(cx, cy, frame_w - cx, frame_h - cy)
        bbox_ref   = max(bw, bh, 1.0)
        dist_border = float(np.clip(border_d / (4.0 * bbox_ref), 0.0, 1.0))

        # ------------------------------------------------------------------
        # 4. Flow features (Farneback if frames provided, else zero)
        # ------------------------------------------------------------------
        if prev_frame is not None and curr_frame is not None:
            global_flow_mag, target_flow_mag = _compute_flow_features(
                prev_frame, curr_frame, cx, cy, bw, bh
            )
            ego_residual    = abs(target_flow_mag - global_flow_mag)
            flow_iou        = 0.5   # full vector similarity skipped for speed
            flow_residual   = ego_residual
            flow_consistency = 1.0
        else:
            global_flow_mag = target_flow_mag = 0.0
            ego_residual = flow_residual = 0.0
            flow_iou = 0.5
            flow_consistency = 1.0

        # ------------------------------------------------------------------
        # 5. Assemble 28-dim base feature vector
        # ------------------------------------------------------------------
        base = np.array([
            apce, apce_norm, psr, ent,
            peak_margin, peak_width, n_sec, peak_dist, hm_mass,
            apce_r5, apce_r20, ent_d5, pm_d5,
            float(self._conf_streak), float(self._low_streak),
            vx, vy, speed, accel, scale_r, ar_delta, dist_border,
            global_flow_mag, target_flow_mag, ego_residual,
            flow_iou, flow_residual, flow_consistency,
        ], dtype=np.float32)  # (28,)

        # ------------------------------------------------------------------
        # 6. RAM memory features (4) — compute BEFORE updating RAM
        # ------------------------------------------------------------------
        mem_feat = np.zeros(self._extra_dim, dtype=np.float32)
        if self._extra_dim > 0:
            ram_feat = self._ram.compute_features(search_embedding)
            mem_feat[:4] = ram_feat

        # Update RAM if gate passes (uses previous p_fc for p_fc gate)
        if search_embedding is not None:
            if self._ram.should_update(apce_norm, self._last_p_fc):
                self._ram.add(search_embedding, frame_idx=t)

        # ------------------------------------------------------------------
        # 7. Append to GRU window and run model
        # ------------------------------------------------------------------
        feat = np.concatenate([base, mem_feat])  # (total_dim,)
        self._window.append(feat)

        if len(self._window) < self._window_size:
            self._last_p_fc = 0.0
            return 0.0

        window = np.stack(list(self._window), axis=0)  # (window_size, total_dim)
        probs = self._model.predict_single(window, device=self.device)
        self._last_p_fc = float(probs.get("false_confirmed", 0.0))
        return self._last_p_fc

    def should_block_template_update(self) -> bool:
        """Return True if template update should be vetoed."""
        if self._last_p_fc >= self.fc_block:
            self.n_blocked += 1
            return True
        self.n_allowed += 1
        return False

    @property
    def last_p_fc(self) -> float:
        return self._last_p_fc

    @property
    def stats(self) -> dict[str, Any]:
        total = self.n_blocked + self.n_allowed
        return {
            "n_steps":   self.n_steps,
            "n_blocked": self.n_blocked,
            "n_allowed": self.n_allowed,
            "block_rate": self.n_blocked / total if total > 0 else 0.0,
            "last_p_fc": self._last_p_fc,
        }


# ---------------------------------------------------------------------------
# Optional Farneback flow helper
# ---------------------------------------------------------------------------

def _compute_flow_features(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    cx: float, cy: float, bw: float, bh: float,
) -> tuple[float, float]:
    """Compute (global_flow_mag, target_flow_mag) via Farneback on small frame."""
    import cv2

    _MAX_SIDE = 64
    h, w = prev_frame.shape[:2]
    scale = min(_MAX_SIDE / max(h, w, 1), 1.0)
    if scale < 1.0:
        dsize = (max(1, int(w * scale)), max(1, int(h * scale)))
        p = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), dsize)
        c = cv2.resize(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), dsize)
    else:
        p = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        c = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        scale = 1.0

    flow = cv2.calcOpticalFlowFarneback(
        p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    global_mag = float(mag.mean())

    # Target region in scaled coords
    x0 = int(max(0, (cx - bw / 2) * scale))
    y0 = int(max(0, (cy - bh / 2) * scale))
    x1 = int(min(mag.shape[1], (cx + bw / 2) * scale))
    y1 = int(min(mag.shape[0], (cy + bh / 2) * scale))
    if x1 > x0 and y1 > y0:
        target_mag = float(mag[y0:y1, x0:x1].mean())
    else:
        target_mag = global_mag

    return global_mag, target_mag
