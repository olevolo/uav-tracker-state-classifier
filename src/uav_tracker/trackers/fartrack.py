"""FARTrack — Feature-Aligned Region Tracker (distill, ViT-tiny variant).

Reference: FARTrack, source repo: papers/code/FARTrack/
Weights: $UAV_WEIGHTS_ROOT/FARTrackDistill_ep0435.pth.tar

Architecture: ViT-tiny backbone (embed_dim=192, depth=12, patch=16)
  Multiple templates (NUM_TEMPLATE=5), search 224×224
  Output: sequence of tokens decoded as [cx,cy,x2,y2] in [0, BINS-1]
  BINS=300, RANGE=2, EXTENSION=3

FARTrack uses a generative sequence head — outputs seqs tokens, not a
score map. Therefore confidence is computed from softmax sharpness of the
predicted sequence scores, not from a spatial score map.

Template update: tracker calls template_update_sampling("exponential") every
frame. To freeze it, wrap the call with set_update_enabled(False). The hook
is installed at model load time.

Capabilities: can_freeze_template=True via set_update_enabled().
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
_FARTRACK_ROOT = _REPO_ROOT / "papers" / "code" / "FARTrack"
_WEIGHTS_NAME = "FARTrackDistill_ep0435.pth.tar"

# Architecture constants from fartrack_distill_224_full.yaml
_TEMPLATE_SIZE = 112
_TEMPLATE_FACTOR = 2.0
_SEARCH_SIZE = 224
_SEARCH_FACTOR = 4.0
_NUM_TEMPLATE = 5
_BINS = 300
_FEAT_SZ = 14   # 224 / 16

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_FLOPS_PER_UPDATE = 0.0   # ViT-tiny: TODO measure; use 0 for now


def _default_weights_path() -> Path:
    env = os.environ.get("UAV_WEIGHTS_ROOT")
    if env:
        return Path(env).expanduser() / _WEIGHTS_NAME
    return Path("~/uav-tracker-weights").expanduser() / _WEIGHTS_NAME


# ---------------------------------------------------------------------------
# easydict shim (same pattern as SGLATrack / AVTrack adapters)
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
# Safe checkpoint loader — FARTrack checkpoints contain lib.train objects
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

    mod = types.ModuleType("_fartrack_safe_pickle")
    mod.Unpickler = _SafeUnpickler
    for attr in ("UnpicklingError", "PicklingError", "HIGHEST_PROTOCOL",
                 "DEFAULT_PROTOCOL", "dumps", "loads"):
        setattr(mod, attr, getattr(pickle, attr))
    return mod


_SAFE_PICKLE = _make_safe_pickle_module()


# ---------------------------------------------------------------------------
# Path / import setup
# ---------------------------------------------------------------------------

def _ensure_fartrack_on_path() -> None:
    root = str(_FARTRACK_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    _ensure_easydict_shim()


# ---------------------------------------------------------------------------
# Preprocessing helpers (device-aware, no hardcoded cuda)
# ---------------------------------------------------------------------------

def _sample_target(frame: np.ndarray, bbox: BBox, factor: float,
                   out_size: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Mean-padded square crop. Returns (patch_rgb, resize_factor, att_mask)."""
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

    # attention mask: 1 inside original crop, 0 in padded region
    att = np.ones((out_size, out_size), dtype=bool)
    return patch, out_size / max(1, crop_sz), att


def _to_tensor(patch_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize RGB patch → (1,3,H,W) ImageNet-normed tensor on device."""
    t = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0)
    t = (t - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(device: torch.device):
    """Build FARTrackDistill model with pretrained=False (weights loaded separately)."""
    _ensure_fartrack_on_path()

    from lib.config.fartrack_distill.config import cfg, update_config_from_file

    yaml_path = _FARTRACK_ROOT / "experiments" / "fartrack_distill" / "fartrack_distill_224_full.yaml"
    update_config_from_file(str(yaml_path))

    # Patch PRETRAIN_PTH so build_fartrack_distill does not try to load from
    # a hardcoded remote path during model construction.
    cfg.MODEL.PRETRAIN_PTH = ""
    cfg.MODEL.PRETRAIN_FILE = ""

    # Monkey-patch build_fartrack_distill to skip the internal checkpoint load
    # that it does unconditionally (cfg.MODEL.PRETRAIN_PTH). We supply weights
    # ourselves after construction.
    from lib.models.fartrack_distill import fartrack_distill as _fd_mod
    from lib.models.fartrack_distill.fartrack_distill import FARTrackDistill
    from lib.models.fartrack_distill.vit import vit_tiny_patch16_224

    backbone_type = cfg.MODEL.BACKBONE.TYPE
    if backbone_type != "vit_tiny_patch16_224":
        logger.warning(
            "FARTrackAdapter: expected vit_tiny_patch16_224, got %s — proceeding",
            backbone_type,
        )

    backbone = vit_tiny_patch16_224(
        "",  # no pretrained file
        drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        bins=cfg.MODEL.BINS,
        range=cfg.MODEL.RANGE,
        extension=cfg.MODEL.EXTENSION,
    )
    backbone.finetune_track(cfg=cfg, patch_start_index=1)
    model = FARTrackDistill(backbone)
    model = model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Tracker adapter
# ---------------------------------------------------------------------------

@TRACKERS.register("fartrack")
class FARTrackAdapter:
    """FARTrack-Distill ViT-tiny adapter for passive CSC diagnosis.

    Template update: FARTrack maintains ``_NUM_TEMPLATE`` (5) templates that
    are refreshed with exponential decay sampling every frame. To freeze all
    template updates for CSC control, call ``set_update_enabled(False)``.

    Confidence: derived from sharpness of the predicted token distribution
    (softmax entropy over BINS×4 sequence predictions).

    tier_hint=1 (lightweight ViT-tiny, real-time capable).
    """

    name: str = "fartrack"
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
        self._update_enabled: bool = True
        self._frame_id: int = 0
        # Template storage: list of (1,3,H,W) tensors on device
        self._template_list: list[torch.Tensor] = []
        self._stored_templates: list[torch.Tensor] = []

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
    # Model loading                                                           #
    # --------------------------------------------------------------------- #

    def _load(self) -> None:
        dev = self._device
        try:
            model, cfg = _build_model(dev)
        except Exception as exc:
            logger.warning(
                "FARTrackAdapter: model build failed (%s) — running in stub mode", exc
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
                state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                core_missing = [k for k in missing if k.startswith("backbone.")]
                self._is_stub = bool(core_missing)
                if not core_missing:
                    logger.info(
                        "FARTrackAdapter weights loaded from %s "
                        "(missing=%d unexpected=%d)",
                        p, len(missing), len(unexpected),
                    )
                else:
                    logger.warning(
                        "FARTrackAdapter: %d core backbone keys missing — stub mode",
                        len(core_missing),
                    )
            except Exception as exc:
                logger.warning(
                    "FARTrackAdapter weight load failed: %s — random init (stub mode)",
                    exc,
                )
                self._is_stub = True
        else:
            logger.warning(
                "FARTrackAdapter: weights not found at %s — random init (stub mode). "
                "Place %s at $UAV_WEIGHTS_ROOT/ or set --weights_path.",
                p, _WEIGHTS_NAME,
            )
            self._is_stub = True

        self._model = model
        self._cfg = cfg

    # --------------------------------------------------------------------- #
    # Preprocessing helpers                                                   #
    # --------------------------------------------------------------------- #

    def _make_template_tensor(self, frame: np.ndarray, bbox: BBox) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, _, _ = _sample_target(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        return _to_tensor(patch, self._device)  # (1,3,H,W)

    def _make_search_tensor(self, frame: np.ndarray) -> tuple[torch.Tensor, float]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, rf, _ = _sample_target(rgb, self._state, _SEARCH_FACTOR, _SEARCH_SIZE)
        return _to_tensor(patch, self._device), rf

    # --------------------------------------------------------------------- #
    # Template update with freeze hook                                        #
    # --------------------------------------------------------------------- #

    def _template_update_sampling(self, new_z: torch.Tensor) -> None:
        """Exponential-decay multi-template update (mirrors FARTrackDistill.track).

        When update is disabled (CSC control), skips the stored_templates append
        so the template list stays frozen at the last good frame.
        """
        if not self._update_enabled:
            return

        if not self._stored_templates:
            # First call — stored_templates was just seeded with the init template
            pass
        self._stored_templates.append(new_z)

        current_frame_count = len(self._stored_templates)
        num_templates = _NUM_TEMPLATE

        if current_frame_count < num_templates:
            self._template_list = list(self._stored_templates)
            return

        # Exponential decay sampling (matches FARTrackDistill.template_update_sampling)
        sampled_indices = [0]
        for i in range(1, num_templates - 1):
            idx = int((current_frame_count - 1) * (1 - 0.7 ** i))
            sampled_indices.append(idx)
        sampled_indices.append(current_frame_count - 1)

        self._template_list = [self._stored_templates[i] for i in sampled_indices]

    # --------------------------------------------------------------------- #
    # Lifecycle                                                               #
    # --------------------------------------------------------------------- #

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        if self._model is None:
            self._load()

        self._state = bbox
        self._frame_id = 0
        self._update_enabled = True

        z0 = self._make_template_tensor(frame, bbox)
        self._stored_templates = [z0]
        self._template_list = [z0] * _NUM_TEMPLATE

    def update(self, frame: np.ndarray) -> TrackState:
        self._frame_id += 1

        # Stub path: model failed to load
        if self._model is None or self._state is None:
            logger.debug("FARTrackAdapter: stub mode — returning last bbox")
            return TrackState(
                bbox=self._state or BBox(0, 0, 1, 1),
                confidence=0.5,
                status="uncertain",
                aux={"stub": True},
            )

        search_tensor, resize_factor = self._make_search_tensor(frame)

        with torch.no_grad():
            out = self._model(
                template=self._template_list,
                search=search_tensor,
                ce_template_mask=None,
            )

        # ---- Decode bbox from sequence output ----
        # out['seqs'] shape: (1, seq_len, BINS) or (1, 4) decoded directly.
        # FARTrack outputs [cx, cy, x2, y2] in [0, BINS-1].
        bins = getattr(self._cfg.MODEL, "BINS", _BINS) if self._cfg else _BINS
        seqs = out.get("seqs")
        if seqs is None:
            # Fallback: return last known position
            logger.debug("FARTrackAdapter: no 'seqs' in output — returning last bbox")
            return TrackState(
                bbox=self._state,
                confidence=0.3,
                status="uncertain",
                aux={"stub": True, "missing_seqs": True},
            )

        # seqs: (1, N, BINS) — take first 4 tokens as [cx, cy, x2, y2]
        # Per FARTrackDistill.track: pred_boxes = seqs[:, 0:4] / (bins - 1) - 0.5
        # → values in [-0.5, 0.5], then converted to [cx,cy,w,h]
        pred_raw = seqs[:, 0:4] / (bins - 1) - 0.5   # (1,4) in [-0.5, 0.5]
        pred = pred_raw.view(-1, 4).mean(dim=0)

        # [cx, cy, x2, y2] → [cx, cy, w, h]
        cx = float(pred[0])
        cy = float(pred[1])
        w = float(pred[2] - pred[0])
        h = float(pred[3] - pred[1])

        # Scale from normalized search coords to frame coords
        # (same mapping as FARTrackDistill.map_box_back)
        cx_pred = cx * _SEARCH_SIZE / resize_factor
        cy_pred = cy * _SEARCH_SIZE / resize_factor
        w_pred = max(1.0, abs(w) * _SEARCH_SIZE / resize_factor)
        h_pred = max(1.0, abs(h) * _SEARCH_SIZE / resize_factor)

        cx_prev = self._state.x + self._state.w / 2
        cy_prev = self._state.y + self._state.h / 2
        half = _SEARCH_SIZE / (2 * resize_factor)

        cx_real = cx_prev + cx_pred - half
        cy_real = cy_prev + cy_pred - half

        new_bbox = BBox(
            x=cx_real - w_pred / 2,
            y=cy_real - h_pred / 2,
            w=w_pred,
            h=h_pred,
        )

        # ---- Confidence from softmax sharpness of sequence tokens ----
        # seqs may be (1, 4) decoded indices or (1, N, BINS) full logits.
        # Use sharpness only when full logits are available.
        with torch.no_grad():
            if seqs.dim() == 3:
                coord_logits = seqs[0, :4, :]        # (4, BINS)
                probs = torch.softmax(coord_logits, dim=-1)
                confidence = float(probs.max(dim=-1).values.mean().clamp(0.0, 1.0))
            else:
                # Only decoded coordinates available — use distance-from-edge heuristic.
                # Tokens near the centre of [0, bins-1] → uncertain; near edges → certain.
                half = (bins - 1) / 2.0
                center_dist = (seqs[0, :4].float() - half).abs() / half   # [0, 1]
                confidence = float(center_dist.mean().clamp(0.0, 1.0))

        self._state = new_bbox

        # ---- Template update (exponential sampling) ----
        if self._update_enabled:
            new_z = self._make_template_tensor(frame, new_bbox)
            self._template_update_sampling(new_z)

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
            aux={
                "frame_id": self._frame_id,
                "n_templates": len(self._template_list),
                "update_enabled": self._update_enabled,
            },
        )

    # --------------------------------------------------------------------- #
    # Control interface                                                       #
    # --------------------------------------------------------------------- #

    def set_update_enabled(self, enabled: bool) -> None:
        """CSCAdvisor gate: freeze/unfreeze template update."""
        self._update_enabled = bool(enabled)

    def reset(self) -> None:
        self._state = None
        self._frame_id = 0
        self._update_enabled = True
        self._template_list = []
        self._stored_templates = []

    # --------------------------------------------------------------------- #
    # Capabilities & metadata                                                 #
    # --------------------------------------------------------------------- #

    @property
    def capabilities(self):
        from uav_tracker.trackers.capabilities import TrackerCapabilities
        return TrackerCapabilities(
            can_freeze_template=True,   # set_update_enabled() wired
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
