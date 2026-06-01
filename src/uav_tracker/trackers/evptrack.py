"""EVPTrack — Spatio-temporal Visual Prompt Tracker (HiViT-Base, 224 search).

Reference: Wang et al., "Tracking with Spatio-temporal Visual Prompt", AAAI 2024.
Source repo: ``papers/code/EVPTrack/`` (vendored, do NOT modify).

Backbone: HiViT-Base (embed_dim=512, depths=[2, 2, 20], heads=8)
  Template 112×112 (7×7 patches),  Search 224×224 (14×14 patches), stride 16
  CE_LOC = []   — no candidate elimination in EVPTrack-full-224.

Expected weights:
  $UAV_WEIGHTS_ROOT/evptrack/EVPTrack-full-224.pth.tar
  or ~/uav-tracker-weights/evptrack/EVPTrack-full-224.pth.tar

At inference we set ``cfg.MODEL.PRETRAIN_FILE = ''`` so the backbone is built
without trying to load the MAE pretrain — model weights come from the checkpoint.
If the checkpoint is missing the model runs with random init (DRY-RUN mode).

This adapter emits a TrackState compatible with the rest of the tracker baseline
contract: bbox, confidence, apce, psr, response_entropy, plus an ``aux`` dict
with response statistics for downstream telemetry.
"""
from __future__ import annotations

import logging
import math
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, FrameContext, TrackState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and architecture constants (EVPTrack-full-224)
# ---------------------------------------------------------------------------

_EVPTRACK_ROOT = Path(__file__).resolve().parents[3] / "papers" / "code" / "EVPTrack"
_YAML_REL      = "experiments/evptrack/EVPTrack-full-224.yaml"
_WEIGHTS_NAME  = "EVPTrack-full-224.pth.tar"
_WEIGHTS_SUBDIR = "evptrack"

# Architecture constants (EVPTrack-full-224 YAML):
#   DATA.TEMPLATE.SIZE=112, DATA.TEMPLATE.FACTOR=2.0
#   DATA.SEARCH.SIZE=224,   DATA.SEARCH.FACTOR=4.0
#   MODEL.BACKBONE.STRIDE=16  → feat_sz = 224/16 = 14
_TEMPLATE_SIZE   = 112
_TEMPLATE_FACTOR = 2.0
_SEARCH_SIZE     = 224
_SEARCH_FACTOR   = 4.0
_FEAT_SZ         = 14             # 224 / 16
_FEAT_LEN        = _FEAT_SZ * _FEAT_SZ  # 196

# HiViT-Base ~32 GFLOPs/forward (EVPTrack paper Table 4; approximate).
_FLOPS_PER_UPDATE = 32e9

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# easydict shim — EVPTrack's config loader requires it. Inject a minimal
# subclass-of-dict shim before any import to avoid installing the package.
# ---------------------------------------------------------------------------

def _ensure_easydict_shim() -> None:
    if "easydict" in sys.modules:
        return

    class _EasyDict(dict):
        def __init__(self, d: dict | None = None, **kw: Any) -> None:
            super().__init__()
            if d:
                for k, v in d.items():
                    self[k] = _EasyDict(v) if isinstance(v, dict) else v
            for k, v in kw.items():
                self[k] = _EasyDict(v) if isinstance(v, dict) else v

        def __getattr__(self, k: str) -> Any:
            try:
                return self[k]
            except KeyError as exc:
                raise AttributeError(k) from exc

        def __setattr__(self, k: str, v: Any) -> None:
            self[k] = v

        def __delattr__(self, k: str) -> None:
            del self[k]

    shim = types.ModuleType("easydict")
    shim.EasyDict = _EasyDict  # type: ignore[attr-defined]
    sys.modules["easydict"] = shim


def _ensure_evptrack_on_path() -> None:
    root = str(_EVPTRACK_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _ensure_easydict_shim()


# ---------------------------------------------------------------------------
# Model build (lazy — called once in _load)
# ---------------------------------------------------------------------------

def _build_model(device: torch.device):
    """Build EVPTrack-full-224 from the vendored YAML, no MAE pretrain at inference."""
    _ensure_evptrack_on_path()

    from lib.config.evptrack.config import cfg, update_config_from_file  # type: ignore
    from lib.models.evptrack import build_evptrack                        # type: ignore

    yaml_path = _EVPTRACK_ROOT / _YAML_REL
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"EVPTrack YAML not found: {yaml_path}. "
            f"Expected vendored repo at papers/code/EVPTrack."
        )
    update_config_from_file(str(yaml_path))

    # Disable MAE backbone pretrain — at inference the full-model checkpoint
    # supplies all weights. build_evptrack() only attempts the MAE load when
    # PRETRAIN_FILE is non-empty AND training=True; passing training=False is
    # sufficient, but emptying the path is belt-and-braces.
    cfg.MODEL.PRETRAIN_FILE = ""

    model = build_evptrack(cfg, training=False)
    model = model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _sample_target(
    frame: np.ndarray, bbox: BBox, factor: float, out_size: int
) -> tuple[np.ndarray, float]:
    """Mean-padded square crop centred on bbox. Returns (patch_rgb, resize_factor).

    Mirrors EVPTrack's ``sample_target`` semantics (BORDER_CONSTANT padding) but
    fills with the image mean rather than zeros — matches the rest of our adapters
    and avoids boundary artefacts on border-adjacent objects.
    """
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    crop_sz = max(1, int(math.ceil(math.sqrt(w * h) * factor)))
    cx, cy = x + w / 2, y + h / 2
    x1 = round(cx - crop_sz / 2)
    y1 = round(cy - crop_sz / 2)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    H, W = frame.shape[:2]
    x1p = max(0, -x1)
    y1p = max(0, -y1)
    x2p = max(x2 - W, 0)
    y2p = max(y2 - H, 0)

    crop = frame[y1 + y1p : y2 - y2p or None, x1 + x1p : x2 - x2p or None]
    if x1p or x2p or y1p or y2p:
        mean_val = frame.mean(axis=(0, 1)).tolist()
        crop = cv2.copyMakeBorder(
            crop, y1p, y2p, x1p, x2p, cv2.BORDER_CONSTANT, value=mean_val
        )
    patch = cv2.resize(crop, (out_size, out_size))
    return patch, out_size / crop_sz


def _to_tensor(patch_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize RGB patch to ImageNet-normalised (1,3,H,W) tensor on device."""
    t = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0)
    t = (t - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _make_hann(device: torch.device) -> torch.Tensor:
    """2-D centred Hann window (1,1,_FEAT_SZ,_FEAT_SZ) on target device."""
    h1d = 0.5 * (
        1 - torch.cos(
            2 * math.pi / (_FEAT_SZ + 1) * torch.arange(1, _FEAT_SZ + 1).float()
        )
    )
    return (h1d.unsqueeze(1) * h1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@TRACKERS.register("evptrack")
class EVPTracker:
    """EVPTrack HiViT-Base 224 — spatio-temporal visual prompt tracker.

    tier_hint=2 (heavy ViT tracker, ~32 GFLOPs).

    Template size: 112×112.  Search size: 224×224.  Feature grid: 14×14.
    EVPTrack-full-224 has CE_LOC=[] (no candidate elimination), so all 196
    search tokens are always processed — there is no token_keep_ratio telemetry.

    The model call signature is::

        out = model(template=z, search=x, frame_id=t, ce_template_mask=None)

    where ``frame_id`` drives the spatio-temporal prompt scheduling inside the
    HiViT backbone.
    """

    name: str = "evptrack"
    tier_hint: int = 2

    def __init__(
        self,
        device: str = "auto",
        weights_path: str | None = None,
    ) -> None:
        self._device_str = device
        self._weights_path = weights_path
        self._model = None
        self._cfg = None
        self._hann: torch.Tensor | None = None
        self._z_tensor: torch.Tensor | None = None
        self._state: BBox | None = None
        self._frame_id: int = 0
        self._is_stub: bool = True
        self._update_enabled: bool = True  # CSCAdvisor gate (EVPTrack: no internal template update)

    @property
    def _device(self) -> torch.device:
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    def _resolve_ckpt_path(self) -> Path:
        """Return the checkpoint path, preferring explicit override then env var then home."""
        if self._weights_path is not None:
            return Path(self._weights_path).expanduser()

        # 1. $UAV_WEIGHTS_ROOT/evptrack/
        import os
        env_root = os.environ.get("UAV_WEIGHTS_ROOT")
        if env_root:
            candidate = Path(env_root) / _WEIGHTS_SUBDIR / _WEIGHTS_NAME
            if candidate.exists():
                return candidate

        # 2. ~/uav-tracker-weights/evptrack/
        home_default = Path.home() / "uav-tracker-weights" / _WEIGHTS_SUBDIR / _WEIGHTS_NAME
        if home_default.exists():
            return home_default

        # 3. Project weights_root() (config-aware, best effort)
        try:
            from uav_tracker.paths import weights_root  # type: ignore
            return weights_root() / _WEIGHTS_SUBDIR / _WEIGHTS_NAME
        except Exception:
            return home_default

    def _load(self) -> None:
        dev = self._device

        try:
            model, cfg = _build_model(dev)
        except Exception as exc:
            raise NotImplementedError(
                f"Failed to build EVPTrack model from {_EVPTRACK_ROOT}. "
                f"Check that papers/code/EVPTrack exists and lib/ is importable. "
                f"Original error: {exc}"
            ) from exc

        ckpt_path = self._resolve_ckpt_path()
        if ckpt_path.exists():
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                state = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
                missing, unexpected = model.load_state_dict(state, strict=False)
                self._is_stub = bool(missing)
                if missing:
                    logger.warning(
                        "EVPTrack: %d missing keys (showing 5): %s",
                        len(missing), missing[:5],
                    )
                if unexpected:
                    logger.warning(
                        "EVPTrack: %d unexpected keys (showing 5): %s",
                        len(unexpected), unexpected[:5],
                    )
                if not missing:
                    logger.info("EVPTrack weights loaded from %s", ckpt_path)
            except Exception as exc:
                logger.warning(
                    "EVPTrack weight load failed (%s) — DRY-RUN with random init", exc
                )
                self._is_stub = True
        else:
            logger.warning(
                "EVPTrack weights not found at %s — MISSING_WEIGHTS — dry-run mode",
                ckpt_path,
            )
            self._is_stub = True

        self._model = model
        self._cfg = cfg
        self._hann = _make_hann(dev)

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        if self._model is None:
            self._load()

        # Reset per-sequence state (allows reuse without reconstruction).
        self._frame_id = 0

        # OpenCV delivers BGR — EVPTrack training pipeline used RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, _ = _sample_target(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        self._z_tensor = _to_tensor(patch, self._device)
        self._state = bbox

    def update(self, frame: np.ndarray) -> TrackState:
        if self._model is None or self._z_tensor is None or self._state is None:
            return TrackState(bbox=BBox(0, 0, 1, 1), confidence=0.0, status="lost")

        self._frame_id += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, resize_factor = _sample_target(
            rgb, self._state, _SEARCH_FACTOR, _SEARCH_SIZE
        )
        x_tensor = _to_tensor(patch, self._device)

        with torch.no_grad():
            out = self._model(
                template=self._z_tensor,
                search=x_tensor,
                frame_id=self._frame_id,
                ce_template_mask=None,
            )

        # score_map: (1, 1, 14, 14) — sigmoid-clamped centres from CenterPredictor
        score_map = out["score_map"]
        size_map   = out["size_map"]
        offset_map = out["offset_map"]

        # Hann-window the score map for smoother localisation
        response = self._hann * score_map

        with torch.no_grad():
            # Decode bbox via CenterPredictor.cal_bbox
            pred_boxes = self._model.box_head.cal_bbox(
                response, size_map, offset_map
            ).view(-1, 4)
            pred = (
                pred_boxes.mean(dim=0) * _SEARCH_SIZE / resize_factor
            ).tolist()  # cx, cy, w, h in search-crop pixels

        cx_pred, cy_pred, w_pred, h_pred = pred

        # Map back to frame coords (search crop is centred on previous bbox).
        cx_prev = self._state.x + self._state.w / 2
        cy_prev = self._state.y + self._state.h / 2
        half = _SEARCH_SIZE / (2 * resize_factor)

        new_bbox = BBox(
            x=cx_prev + cx_pred - half - w_pred / 2,
            y=cy_prev + cy_pred - half - h_pred / 2,
            w=max(1.0, w_pred),
            h=max(1.0, h_pred),
        )
        self._state = new_bbox

        # ------------------------------------------------------------------
        # Telemetry
        # ------------------------------------------------------------------
        with torch.no_grad():
            f = response.view(-1)   # 196-element flattened response

            f_max = float(f.max())
            f_min = float(f.min())
            response_max  = float(score_map.max())
            response_mean = float(score_map.mean())
            response_std  = float(score_map.std(unbiased=False))

            # APCE — Average Peak-to-Correlation Energy
            denom = float(((f - f_min) ** 2).mean())
            apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))

            # PSR — Peak-to-Sidelobe Ratio (5-cell neighbourhood around peak)
            sm2d     = response.squeeze()   # (14, 14)
            flat_idx = int(sm2d.view(-1).argmax())
            peak_r, peak_c = flat_idx // _FEAT_SZ, flat_idx % _FEAT_SZ
            r_idx = (
                torch.arange(_FEAT_SZ, device=sm2d.device)
                .view(_FEAT_SZ, 1)
                .expand(_FEAT_SZ, _FEAT_SZ)
            )
            c_idx = (
                torch.arange(_FEAT_SZ, device=sm2d.device)
                .view(1, _FEAT_SZ)
                .expand(_FEAT_SZ, _FEAT_SZ)
            )
            peak_mask = (
                ((r_idx - peak_r).abs() <= 5) & ((c_idx - peak_c).abs() <= 5)
            )
            sidelobe = sm2d[~peak_mask]
            if sidelobe.numel() > 0:
                psr = float((f_max - sidelobe.mean()) / (sidelobe.std() + 1e-8))
            else:
                psr = 0.0

            # Response entropy via L1-normalised score map
            f_pos = f - f.min()
            probs = f_pos / (f_pos.sum() + 1e-8)
            response_entropy = float(-(probs * (probs + 1e-8).log()).sum())

            # Top-3 softmax mass — better-calibrated confidence than raw peak value
            sm     = torch.softmax(response.view(-1), dim=0)
            top3   = sm.topk(min(3, sm.numel())).values.sum()
            sm_top1 = float(sm.max().cpu())
            confidence = float(top3.clamp(0.0, 1.0).cpu())

        if confidence > 0.25:
            status = "locked"
        elif confidence > 0.10:
            status = "uncertain"
        else:
            status = "lost"

        raw = {
            "score_max":     float(f_max),
            "response_max":  response_max,
            "response_mean": response_mean,
            "response_std":  response_std,
            "sm_top1":       sm_top1,
            # EVPTrack-full-224 has CE_LOC=[] — no removed-token bookkeeping.
            # token_keep_ratio is always 1.0 (all search tokens retained).
            "token_keep_ratio": 1.0,
            "num_prompts":   None,
        }

        return TrackState(
            bbox=new_bbox,
            confidence=confidence,
            status=status,
            apce=apce,
            psr=psr,
            response_entropy=response_entropy,
            aux={"raw": raw},
        )

    def update_with_action(self, frame: np.ndarray, action: Any) -> TrackState:
        """Action routing stub — EVPTrack adapter does not support CE/search overrides."""
        return self.update(frame)

    def reset(self) -> None:
        self._z_tensor = None
        self._state = None
        self._frame_id = 0
        self._update_enabled = True

    def set_update_enabled(self, enabled: bool) -> None:
        """CSCAdvisor gate.  EVPTrack has no internal template update — this is a
        no-op but satisfies the _try_set_update_enabled() protocol in run_with_csc.py."""
        self._update_enabled = bool(enabled)

    @property
    def capabilities(self):
        from uav_tracker.trackers.capabilities import TrackerCapabilities
        return TrackerCapabilities()  # all defaults: only can_reject_bbox + can_force_reinit

    @property
    def is_stub_mode(self) -> bool:
        return self._is_stub

    def flops_per_update(self) -> float:
        return _FLOPS_PER_UPDATE

    def on_tier_enter(self, ctx: FrameContext) -> None:  # pragma: no cover
        pass

    def on_tier_exit(self, ctx: FrameContext) -> None:  # pragma: no cover
        pass
