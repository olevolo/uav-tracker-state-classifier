"""UETrack — Unified Event/RGB/Language Tracker (SULIGHT / tiny variant).

Reference: UETrack, source repo: papers/code/UETrack/
Weights: $UAV_WEIGHTS_ROOT/uetrack_tiny.tar  (net_type=SULIGHT, ep496)

Architecture (uetrack_tiny.yaml):
  Encoder: fastitpnt_layer2 (tiny, stride=16)
  Decoder: CENTER head
  Template 112×112, search 224×224
  MULTI_MODAL_LANGUAGE=True, MULTI_MODAL_VISION=True in yaml
  TEXT_ENCODER: ViT-L/14 (CLIP)

Dependency notes:
  * Needs ``clip`` (pip install git+https://github.com/openai/CLIP).
    If missing, adapter falls back to stub mode with a clear warning.
  * UETrack's Preprocessor and tracker code calls ``.cuda()`` unconditionally.
    We build a device-portable preprocessor instead.
  * ``lib.train.admin.tensorboard`` wraps tensorboard gracefully (already done).
  * update_intervals=999999 → NO active template update → can_freeze_template=False.

Stub fallback:
  If model load fails for any reason (missing clip, missing timm layers, CUDA not
  available) the adapter logs a warning and returns
  ``TrackState(bbox=init_bbox, confidence=0.5, status="uncertain")`` every frame.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
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
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UETRACK_ROOT = _REPO_ROOT / "papers" / "code" / "UETrack"
_WEIGHTS_NAME = "uetrack_tiny.tar"
_YAML_REL = "experiments/uetrack/uetrack_tiny.yaml"

# Architecture defaults from uetrack_tiny.yaml (overridden by cfg after load)
_TEMPLATE_SIZE = 112
_TEMPLATE_FACTOR = 2.0
_SEARCH_SIZE = 224
_SEARCH_FACTOR = 4.0
_FEAT_SZ = 14   # 224 / 16

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_FLOPS_PER_UPDATE = 0.0   # fastitpnt_layer2: TODO measure; placeholder


def _default_weights_path() -> Path:
    env = os.environ.get("UAV_WEIGHTS_ROOT")
    if env:
        return Path(env).expanduser() / _WEIGHTS_NAME
    return Path("~/uav-tracker-weights").expanduser() / _WEIGHTS_NAME


# ---------------------------------------------------------------------------
# easydict shim (same as other adapters)
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

        def __getattr__(self, k: str):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k: str, v: Any) -> None:
            self[k] = v

        def __delattr__(self, k: str) -> None:
            del self[k]

    shim = types.ModuleType("easydict")
    shim.EasyDict = _EasyDict
    sys.modules["easydict"] = shim


# ---------------------------------------------------------------------------
# Safe pickle for checkpoints that may contain training objects
# ---------------------------------------------------------------------------

def _make_safe_pickle_module() -> types.ModuleType:
    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("lib."):
                return type(name, (), {
                    "__new__": classmethod(lambda cls, *a, **kw: object.__new__(cls)),
                    "__init__": lambda self, *a, **kw: None,
                    "__setstate__": lambda self, s: None,
                })
            return super().find_class(module, name)

    mod = types.ModuleType("_uetrack_safe_pickle")
    mod.Unpickler = _SafeUnpickler
    for attr in ("UnpicklingError", "PicklingError", "HIGHEST_PROTOCOL",
                 "DEFAULT_PROTOCOL", "dumps", "loads"):
        setattr(mod, attr, getattr(pickle, attr))
    return mod


_SAFE_PICKLE = _make_safe_pickle_module()


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def _ensure_uetrack_on_path() -> None:
    root = str(_UETRACK_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    _ensure_easydict_shim()


# ---------------------------------------------------------------------------
# Device-portable preprocessing (replaces the CUDA-only Preprocessor)
# ---------------------------------------------------------------------------

def _sample_target(frame: np.ndarray, bbox: BBox, factor: float,
                   out_size: int) -> tuple[np.ndarray, float]:
    """Mean-padded square crop centered on bbox. Returns (patch_rgb, resize_factor)."""
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    crop_sz = max(1, math.ceil(math.sqrt(w * h) * factor))
    cx, cy = x + w / 2, y + h / 2
    x1 = round(cx - crop_sz / 2)
    y1 = round(cy - crop_sz / 2)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    H, W = frame.shape[:2]
    x1p = max(0, -x1);  y1p = max(0, -y1)
    x2p = max(x2 - W, 0); y2p = max(y2 - H, 0)

    crop = frame[y1 + y1p: y2 - y2p or None, x1 + x1p: x2 - x2p or None]
    if crop.size == 0:
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
    if x1p or x2p or y1p or y2p:
        crop = cv2.copyMakeBorder(crop, y1p, y2p, x1p, x2p,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    patch = cv2.resize(crop, (out_size, out_size))
    return patch, out_size / max(1, crop_sz)


def _to_tensor(patch_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize RGB patch → (1,3,H,W) ImageNet-normed tensor on device."""
    t = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0)
    t = (t - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _map_box_back(pred_box: list, state: BBox, search_size: int,
                  resize_factor: float) -> BBox:
    """Map search-crop-centred [cx,cy,w,h] prediction back to frame coords."""
    cx_prev = state.x + 0.5 * state.w
    cy_prev = state.y + 0.5 * state.h
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return BBox(
        x=cx_real - 0.5 * w,
        y=cy_real - 0.5 * h,
        w=max(1.0, w),
        h=max(1.0, h),
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(device: torch.device):
    """Build UETrack-tiny (SULIGHT) inference model on device.

    Raises RuntimeError on any failure so the adapter can fall back to stub.
    The main failure modes are:
      * ``clip`` not installed (text encoder depends on it)
      * ``timm`` fastitpnt_layer2 not recognised
      * CUDA unavailable but model tries .cuda()

    We patch .cuda() / .to('cuda') calls on nn.Module to redirect to device.
    """
    _ensure_uetrack_on_path()

    # Inject device-redirect so .cuda() calls don't crash on CPU/MPS.
    # We only patch if device is not CUDA — on CUDA everything works normally.
    if device.type != "cuda":
        import torch.nn as _nn
        _orig_cuda = _nn.Module.cuda
        _orig_to = _nn.Module.to

        def _patched_cuda(self, *a, **kw):
            return _orig_to(self, device)

        # Only patch for the duration of model construction
        _nn.Module.cuda = _patched_cuda

    try:
        from lib.config.uetrack.config import cfg, update_config_from_file

        yaml_path = _UETRACK_ROOT / _YAML_REL
        update_config_from_file(str(yaml_path))

        # Disable MULTI_MODAL_LANGUAGE to avoid clip download at build time.
        # We set text_src=None at inference (UAV tracking task index 0 = RGB only).
        cfg.DATA.MULTI_MODAL_LANGUAGE = False
        cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False

        from lib.models.uetrack import build_uetrack_inference
        model = build_uetrack_inference(cfg)
        model = model.to(device).eval()

        return model, cfg

    finally:
        if device.type != "cuda":
            import torch.nn as _nn
            _nn.Module.cuda = _orig_cuda  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# Tracker adapter
# ---------------------------------------------------------------------------

@TRACKERS.register("uetrack")
class UETrackAdapter:
    """UETrack-tiny (SULIGHT) adapter for passive CSC diagnosis.

    UETrack has update_intervals=999999 → no internal template update by
    default. The adapter freezes the template at init. ``can_freeze_template``
    is therefore False (nothing to freeze additionally).

    Confidence: derived from the spatial score-map peak (post-Hann window)
    via the CENTER decoder output.

    If model construction fails (missing clip, CUDA-only layers, etc.) the
    adapter runs in stub mode and logs "UETrack: stub mode" every frame.

    tier_hint=1 (fastitpn-tiny, real-time capable).
    """

    name: str = "uetrack"
    tier_hint: int = 1

    def __init__(
        self,
        device: str = "auto",
        weights_path: str | None = None,
    ) -> None:
        self._device_str = device
        self._weights_path = weights_path
        self._model = None
        self._cfg = None
        self._state: BBox | None = None
        self._is_stub: bool = True
        self._z_tensor: torch.Tensor | None = None
        self._template_anno: torch.Tensor | None = None
        # Hann window for score-map smoothing (built on first use)
        self._hann: torch.Tensor | None = None
        # Resolved sizes (updated from cfg after load)
        self._template_size: int = _TEMPLATE_SIZE
        self._template_factor: float = _TEMPLATE_FACTOR
        self._search_size: int = _SEARCH_SIZE
        self._search_factor: float = _SEARCH_FACTOR
        self._feat_sz: int = _FEAT_SZ
        # Task index for UAV/RGB tracking (index 0 from uetrack_tiny.yaml)
        self._task_index: torch.Tensor | None = None

    # --------------------------------------------------------------------- #
    # Device                                                                  #
    # --------------------------------------------------------------------- #

    @property
    def _device(self) -> torch.device:
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    # --------------------------------------------------------------------- #
    # Hann window                                                             #
    # --------------------------------------------------------------------- #

    def _make_hann(self, feat_sz: int, device: torch.device) -> torch.Tensor:
        h1d = 0.5 * (1 - torch.cos(
            2 * math.pi / (feat_sz + 1) * torch.arange(1, feat_sz + 1).float()
        ))
        return (h1d.unsqueeze(1) * h1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(device)

    # --------------------------------------------------------------------- #
    # Model loading                                                           #
    # --------------------------------------------------------------------- #

    def _load(self) -> None:
        dev = self._device
        try:
            model, cfg = _build_model(dev)
        except Exception as exc:
            logger.warning(
                "UETrackAdapter: model build failed (%s) — running in stub mode. "
                "Install clip: pip install git+https://github.com/openai/CLIP",
                exc,
            )
            self._is_stub = True
            return

        weights_path = self._weights_path
        if weights_path is None:
            weights_path = str(_default_weights_path())
        p = Path(weights_path).expanduser()

        if p.exists():
            try:
                ckpt = torch.load(str(p), map_location="cpu",
                                  weights_only=False, pickle_module=_SAFE_PICKLE)
                # checkpoint layout: {'net': state_dict, 'net_type': 'SULIGHT', ...}
                if isinstance(ckpt, dict) and "net" in ckpt:
                    state_dict = ckpt["net"]
                    net_type = ckpt.get("net_type", "unknown")
                    logger.info(
                        "UETrackAdapter: checkpoint net_type=%s", net_type
                    )
                else:
                    state_dict = ckpt
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                core_missing = [k for k in missing
                                if k.startswith(("encoder.", "decoder."))]
                self._is_stub = bool(core_missing)
                if not core_missing:
                    logger.info(
                        "UETrackAdapter weights loaded from %s "
                        "(missing=%d unexpected=%d)",
                        p, len(missing), len(unexpected),
                    )
                else:
                    logger.warning(
                        "UETrackAdapter: %d core keys missing (e.g. %s) — "
                        "stub mode",
                        len(core_missing), core_missing[:3],
                    )
            except Exception as exc:
                logger.warning(
                    "UETrackAdapter weight load failed: %s — random init (stub mode)",
                    exc,
                )
                self._is_stub = True
        else:
            logger.warning(
                "UETrackAdapter: weights not found at %s — random init (stub mode). "
                "Place %s at $UAV_WEIGHTS_ROOT/ or set --weights_path.",
                p, _WEIGHTS_NAME,
            )
            self._is_stub = True

        self._model = model
        self._cfg = cfg

        # Update sizes from loaded config
        ts = getattr(cfg.TEST, "TEMPLATE_SIZE", _TEMPLATE_SIZE)
        tf = getattr(cfg.TEST, "TEMPLATE_FACTOR", _TEMPLATE_FACTOR)
        ss = getattr(cfg.TEST, "SEARCH_SIZE", _SEARCH_SIZE)
        sf = getattr(cfg.TEST, "SEARCH_FACTOR", _SEARCH_FACTOR)
        stride = getattr(cfg.MODEL.ENCODER, "STRIDE", 16)
        self._template_size = int(ts)
        self._template_factor = float(tf)
        self._search_size = int(ss)
        self._search_factor = float(sf)
        self._feat_sz = int(ss) // int(stride)

        use_window = getattr(cfg.TEST, "WINDOW", True)
        if use_window:
            self._hann = self._make_hann(self._feat_sz, dev)
        else:
            self._hann = None

        # UAV/RGB = task index 0 (GOT10K/LaSOT/UAV group per uetrack_tiny.yaml)
        self._task_index = torch.tensor([0], device=dev)

    # --------------------------------------------------------------------- #
    # Template annotation helper (UETrack requires bbox in crop coords)      #
    # --------------------------------------------------------------------- #

    def _make_template_anno(self, template_size: int, device: torch.device) -> torch.Tensor:
        """Template annotation = center bbox in normalized crop coords."""
        # Center crop bbox: x=0.25, y=0.25, w=0.5, h=0.5 in [0,1]
        # Matches transform_image_to_crop with the same bbox used for cropping.
        half = 0.5
        anno = torch.tensor(
            [half - 0.25, half - 0.25, half, half],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)  # (1, 4)
        return anno

    # --------------------------------------------------------------------- #
    # Lifecycle                                                               #
    # --------------------------------------------------------------------- #

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        if self._model is None:
            self._load()

        self._state = bbox

        if self._model is None:
            # stub mode: keep state, z_tensor stays None
            return

        dev = self._device
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, _ = _sample_target(rgb, bbox, self._template_factor, self._template_size)
        self._z_tensor = _to_tensor(patch, dev)
        self._template_anno = self._make_template_anno(self._template_size, dev)

    def update(self, frame: np.ndarray) -> TrackState:
        # Stub path: model failed to load or init was not called
        if self._model is None or self._state is None:
            logger.debug("UETrackAdapter: stub mode — returning last bbox")
            return TrackState(
                bbox=self._state or BBox(0, 0, 1, 1),
                confidence=0.5,
                status="uncertain",
                aux={"stub": True},
            )

        if self._z_tensor is None:
            logger.debug("UETrackAdapter: not initialized — returning last bbox")
            return TrackState(
                bbox=self._state,
                confidence=0.5,
                status="uncertain",
                aux={"stub": True, "reason": "not_initialized"},
            )

        dev = self._device
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        search_patch, resize_factor = _sample_target(
            rgb, self._state, self._search_factor, self._search_size
        )
        search_tensor = _to_tensor(search_patch, dev)  # (1,3,H,W)

        template_list = [self._z_tensor]
        search_list = [search_tensor]
        template_anno_list = [self._template_anno]

        try:
            with torch.no_grad():
                enc_opt, feature_list, _ = self._model.forward_encoder(
                    template_list,
                    search_list,
                    template_anno_list,
                    text_src=None,        # language disabled
                    task_index=self._task_index,
                )
                out_dict = self._model.forward_decoder(feature=enc_opt)

        except Exception as exc:
            logger.warning(
                "UETrackAdapter: forward pass failed (%s) — returning last bbox", exc
            )
            return TrackState(
                bbox=self._state,
                confidence=0.3,
                status="uncertain",
                aux={"stub": True, "forward_error": str(exc)},
            )

        # ---- Score map + Hann window ----
        pred_score_map = out_dict.get("score_map")
        if pred_score_map is None:
            logger.warning("UETrackAdapter: no score_map in output — stub bbox")
            return TrackState(
                bbox=self._state,
                confidence=0.3,
                status="uncertain",
                aux={"stub": True, "missing_score_map": True},
            )

        if self._hann is not None:
            response = self._hann * pred_score_map
        else:
            response = pred_score_map

        # ---- Decode bbox ----
        with torch.no_grad():
            f = response.view(-1)
            f_max = float(f.max())
            f_min = float(f.min())
            denom = float(((f - f_min) ** 2).mean())
            apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))

            feat_sz = self._feat_sz
            sm2d = response.squeeze()
            flat_idx = int(sm2d.view(-1).argmax())
            peak_r = flat_idx // feat_sz
            peak_c = flat_idx % feat_sz
            r_idx = torch.arange(feat_sz, device=sm2d.device).view(feat_sz, 1).expand(feat_sz, feat_sz)
            c_idx = torch.arange(feat_sz, device=sm2d.device).view(1, feat_sz).expand(feat_sz, feat_sz)
            peak_mask = ((r_idx - peak_r).abs() <= 4) & ((c_idx - peak_c).abs() <= 4)
            sidelobe = sm2d[~peak_mask]
            if len(sidelobe) > 0:
                psr = float((f_max - sidelobe.mean()) / (sidelobe.std() + 1e-8))
            else:
                psr = 0.0

            # L1-normalised entropy
            f_pos = f - f.min()
            probs_norm = f_pos / (f_pos.sum() + 1e-8)
            response_entropy = float(-(probs_norm * (probs_norm + 1e-8).log()).sum())

        # Decode bbox through CENTER decoder
        try:
            if "size_map" in out_dict:
                pred_boxes, conf_score = self._model.decoder.cal_bbox(
                    response, out_dict["size_map"], out_dict["offset_map"],
                    return_score=True,
                )
            else:
                pred_boxes, conf_score = self._model.decoder.cal_bbox(
                    response, out_dict["offset_map"], return_score=True,
                )
            pred_boxes = pred_boxes.view(-1, 4)
            pred_box = (pred_boxes.mean(dim=0) * self._search_size / resize_factor).tolist()
        except Exception as exc:
            logger.warning("UETrackAdapter: bbox decode failed (%s) — stub", exc)
            return TrackState(
                bbox=self._state,
                confidence=0.3,
                status="uncertain",
                apce=apce,
                psr=psr,
                response_entropy=response_entropy,
                aux={"stub": True, "decode_error": str(exc)},
            )

        new_bbox = _map_box_back(pred_box, self._state, self._search_size, resize_factor)
        self._state = new_bbox

        confidence = float(max(0.0, min(1.0, f_max)))

        if confidence > 0.5:
            status = "locked"
        elif confidence > 0.2:
            status = "uncertain"
        else:
            status = "lost"

        return TrackState(
            bbox=new_bbox,
            confidence=confidence,
            status=status,
            apce=apce,
            psr=psr,
            response_entropy=response_entropy,
            aux={
                "response_max": float(f_max),
                "response_min": float(f_min),
            },
        )

    # --------------------------------------------------------------------- #
    # Control interface                                                       #
    # --------------------------------------------------------------------- #

    def set_update_enabled(self, enabled: bool) -> None:
        """CSCAdvisor gate. UETrack has no active template update (interval=999999).
        This is a no-op but satisfies the protocol used by run_with_csc.py."""
        pass

    def reset(self) -> None:
        self._z_tensor = None
        self._template_anno = None
        self._state = None

    # --------------------------------------------------------------------- #
    # Capabilities & metadata                                                 #
    # --------------------------------------------------------------------- #

    @property
    def capabilities(self):
        from uav_tracker.trackers.capabilities import TrackerCapabilities
        return TrackerCapabilities(
            can_freeze_template=False,  # no active update to freeze
            can_widen_search=False,
            can_force_reinit=True,
            can_reject_bbox=True,
            can_reduce_pruning=False,
        )

    @property
    def is_stub_mode(self) -> bool:
        return self._is_stub

    def flops_per_update(self) -> float:
        return _FLOPS_PER_UPDATE

    def on_tier_enter(self, ctx: FrameContext) -> None:
        pass

    def on_tier_exit(self, ctx: FrameContext) -> None:
        pass
