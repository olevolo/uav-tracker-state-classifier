"""ORTrack-DeiT tracker wrapper for UAV benchmarking baseline.

Reference: "ORTrack: One-stream Robust Tracker", AAAI 2024.
Source repo: papers/code/ORTrack (DeiT-tiny variant)

Architecture: DeiT-tiny backbone, CENTER head (16×16 feat grid)
  Template 128×128, Search 256×256, stride 16
  IS_DISTILL=False for deit_tiny_patch16_224 config

Expected weights: /Users/voleksiuk/uav-tracker-weights/ortrack/ORTrack-D-DeiT.pth.tar
"""
from __future__ import annotations

import logging
import math
import pickle
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, FrameContext, TrackState
from uav_tracker.trackers._redetect_common import run_redetect as _run_redetect

logger = logging.getLogger(__name__)

_ORTRACK_ROOT   = Path("papers/code/ORTrack")
_ORTRACK_CONFIG = "experiments/ortrack/deit_tiny_distilled_patch16_224.yaml"  # ORTrack-D: depth=6, IS_DISTILL=True
_WEIGHTS_NAME   = "ORTrack-D-DeiT.pth.tar"
_WEIGHTS_DIR    = Path("/Users/voleksiuk/uav-tracker-weights/ortrack")

_FLOPS_PER_UPDATE = 0.9e9   # DeiT-tiny distilled: ~0.5 GFLOPs (depth=6 vs depth=12)

# Architecture constants matching deit_tiny_patch16_224.yaml
_TEMPLATE_SIZE   = 128
_TEMPLATE_FACTOR = 2.0
_SEARCH_SIZE     = 256
_SEARCH_FACTOR   = 4.0
_FEAT_SZ         = 16   # 256 / 16 (patch stride)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Checkpoint loader — checkpoint may contain training objects from lib.train.
# Stub those classes to avoid importing the full training harness.
# ---------------------------------------------------------------------------

def _make_safe_pickle_module():
    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("lib."):
                return type(name, (), {
                    "__new__": classmethod(lambda cls, *a, **kw: object.__new__(cls)),
                    "__init__": lambda self, *a, **kw: None,
                    "__setstate__": lambda self, s: None,
                })
            return super().find_class(module, name)

    mod = types.ModuleType("_ortrack_safe_pickle")
    mod.Unpickler = _SafeUnpickler
    for attr in ("UnpicklingError", "PicklingError", "HIGHEST_PROTOCOL",
                 "DEFAULT_PROTOCOL", "dumps", "loads"):
        setattr(mod, attr, getattr(pickle, attr))
    return mod


_SAFE_PICKLE = _make_safe_pickle_module()


def _load_ortrack_state(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu",
                      weights_only=False, pickle_module=_SAFE_PICKLE)
    if isinstance(ckpt, dict) and "net" in ckpt:
        return ckpt["net"]
    return ckpt


# ---------------------------------------------------------------------------
# Import ORTrack model builder (adds repo to sys.path once)
# ---------------------------------------------------------------------------

def _ensure_ortrack_on_path() -> None:
    root = str(_ORTRACK_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    # easydict shim — ORTrack config requires it but it may not be installed
    if "easydict" not in sys.modules:
        class _EasyDict(dict):
            def __init__(self, d=None, **kw):
                super().__init__()
                if d:
                    for k, v in d.items():
                        self[k] = _EasyDict(v) if isinstance(v, dict) else v
                for k, v in kw.items():
                    self[k] = _EasyDict(v) if isinstance(v, dict) else v
            def __getattr__(self, k):
                try: return self[k]
                except KeyError: raise AttributeError(k)
            def __setattr__(self, k, v): self[k] = v
            def __delattr__(self, k): del self[k]
        shim = types.ModuleType("easydict")
        shim.EasyDict = _EasyDict
        sys.modules["easydict"] = shim


def _build_model(device: torch.device):
    _ensure_ortrack_on_path()
    from lib.config.ortrack.config import cfg, update_config_from_file
    from lib.models.ortrack import build_ortrack
    import lib.models.ortrack.ortrack as _ortrack_mod
    import lib.models.ortrack.deit as _deit_mod

    yaml_path = _ORTRACK_ROOT.resolve() / _ORTRACK_CONFIG
    update_config_from_file(str(yaml_path))

    # build_ortrack uses `from lib.models.ortrack.deit import deit_tiny_patch16_224`
    # which creates a local binding — must patch in the ortrack module namespace,
    # not the deit module, otherwise pretrained=True still fires a download.
    _builders = [
        "deit_tiny_patch16_224", "deit_tiny_patch16_224_distill",
        "deit_small_patch16_224", "deit_base_patch16_224",
        "vit_tiny_patch16_224", "vit_tiny_distilled_patch16_224",
        "eva02_tiny_patch14_224", "eva02_tiny_patch14_224_distill",
    ]
    _saved = {}
    for _name in _builders:
        for _mod in (_ortrack_mod, _deit_mod):
            if hasattr(_mod, _name):
                _orig = getattr(_mod, _name)
                _saved[(_mod, _name)] = _orig
                setattr(_mod, _name,
                        lambda *a, _f=_orig, **kw: _f(*a, **{**kw, "pretrained": False}))

    try:
        model = build_ortrack(cfg, training=False)
    finally:
        for (_mod, _name), _orig in _saved.items():
            setattr(_mod, _name, _orig)

    model = model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _sample_target(frame: np.ndarray, bbox: BBox, factor: float,
                   out_size: int) -> tuple[np.ndarray, float]:
    """Mean-padded square crop centred on bbox. Returns (patch_rgb, resize_factor)."""
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    crop_sz = max(1, round(math.sqrt(w * h) * factor))
    cx, cy = x + w / 2, y + h / 2
    x1 = round(cx - crop_sz / 2)
    y1 = round(cy - crop_sz / 2)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    H, W = frame.shape[:2]
    x1p = max(0, -x1);  y1p = max(0, -y1)
    x2p = max(x2 - W, 0); y2p = max(y2 - H, 0)

    crop = frame[y1 + y1p: y2 - y2p or None, x1 + x1p: x2 - x2p or None]
    if x1p or x2p or y1p or y2p:
        mean_val = frame.mean(axis=(0, 1)).tolist()
        crop = cv2.copyMakeBorder(crop, y1p, y2p, x1p, x2p,
                                  cv2.BORDER_CONSTANT, value=mean_val)
    patch = cv2.resize(crop, (out_size, out_size))
    return patch, out_size / crop_sz


def _to_tensor(patch_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize RGB patch to ImageNet-normalised (1,3,H,W) tensor."""
    t = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0)
    t = (t - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _make_hann(device: torch.device) -> torch.Tensor:
    """2-D Hann window (1,1,feat_sz,feat_sz) on target device."""
    h1d = 0.5 * (1 - torch.cos(
        2 * math.pi / (_FEAT_SZ + 1) * torch.arange(1, _FEAT_SZ + 1).float()
    ))
    return (h1d.unsqueeze(1) * h1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@TRACKERS.register("ortrack_deit")
@TRACKERS.register("ortrack")  # alias — run_with_csc.py imports this path
class ORTrackDeiT:
    """ORTrack DeiT-tiny — one-stream robust transformer tracker for UAV.

    tier_hint=1 (lightweight learned tracker, ~0.9 GFLOPs, targets real-time).
    Uses deit_tiny_patch16_224 config with CENTER head.
    """

    name: str = "ortrack_deit"
    tier_hint: int = 1

    def __init__(
        self,
        device: str = "auto",
        weights_path: str | None = None,
    ) -> None:
        self._device_str = device
        self._weights_path = weights_path
        self._model = None
        self._hann: torch.Tensor | None = None
        self._z_tensor: torch.Tensor | None = None
        self._state: BBox | None = None
        self._is_stub: bool = True
        self._update_enabled: bool = True  # CSCAdvisor gate (no-op: ORTrack has no internal template update)
        self._search_factor: float = _SEARCH_FACTOR

    @property
    def _device(self) -> torch.device:
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    def _load(self) -> None:
        dev = self._device

        try:
            model, _ = _build_model(dev)
        except Exception as exc:
            raise NotImplementedError(
                f"Failed to build ORTrack model from {_ORTRACK_ROOT}. "
                f"Check that papers/code/ORTrack exists and lib/ is importable. "
                f"Original error: {exc}"
            ) from exc

        weights_path = self._weights_path
        if weights_path is None:
            weights_path = str(_WEIGHTS_DIR / _WEIGHTS_NAME)
        p = Path(weights_path)

        if p.exists():
            try:
                state = _load_ortrack_state(p)
                missing, unexpected = model.load_state_dict(state, strict=True)
                self._is_stub = bool(missing)
                if not missing:
                    logger.info("ORTrackDeiT weights loaded from %s", p)
                else:
                    logger.warning("ORTrackDeiT: missing keys: %s", missing[:3])
            except Exception as exc:
                logger.warning("ORTrackDeiT weight load failed: %s — random init", exc)
                self._is_stub = True
        else:
            logger.info("ORTrackDeiT weights not found at %s — random init", p)
            self._is_stub = True

        self._model = model
        self._hann = _make_hann(dev)

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        if self._model is None:
            self._load()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, _ = _sample_target(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        self._z_tensor = _to_tensor(patch, self._device)
        self._state = bbox

    def update(self, frame: np.ndarray) -> TrackState:
        if self._model is None or self._z_tensor is None or self._state is None:
            return TrackState(bbox=BBox(0, 0, 1, 1), confidence=0.0, status="lost")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, resize_factor = _sample_target(rgb, self._state, self._search_factor, _SEARCH_SIZE)
        x_tensor = _to_tensor(patch, self._device)

        with torch.no_grad():
            out = self._model(
                template=self._z_tensor,
                search=x_tensor,
                is_distill=False,
            )

        score_map = out["score_map"]              # (1,1,16,16)
        score_map = self._hann * score_map

        with torch.no_grad():
            f = score_map.view(-1)
            f_max = float(f.max())
            f_min = float(f.min())
            denom = float(((f - f_min) ** 2).mean())
            apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))

            sm2d = score_map.squeeze()           # (16, 16)
            flat_idx = int(sm2d.view(-1).argmax())
            peak_r, peak_c = flat_idx // _FEAT_SZ, flat_idx % _FEAT_SZ
            r_idx = torch.arange(_FEAT_SZ, device=sm2d.device).view(_FEAT_SZ, 1).expand(_FEAT_SZ, _FEAT_SZ)
            c_idx = torch.arange(_FEAT_SZ, device=sm2d.device).view(1, _FEAT_SZ).expand(_FEAT_SZ, _FEAT_SZ)
            peak_mask = ((r_idx - peak_r).abs() <= 5) & ((c_idx - peak_c).abs() <= 5)
            sidelobe = sm2d[~peak_mask]
            if len(sidelobe) > 0:
                psr = float((f_max - sidelobe.mean()) / (sidelobe.std() + 1e-8))
            else:
                psr = 0.0

            f_pos = f - f.min()
            probs = f_pos / (f_pos.sum() + 1e-8)
            response_entropy = float(-(probs * (probs + 1e-8).log()).sum())

        pred_boxes = self._model.box_head.cal_bbox(
            score_map, out["size_map"], out["offset_map"]
        ).view(-1, 4)

        # Confidence = post-Hann score-map peak (f_max), identical to the legacy
        # ORTrack adapter's `score_max` (ortrack.py:340) and to the statistic the
        # ortrack_*_v2 calibrator was fit on. The previous top-3-softmax over the
        # 16x16 map collapsed to a ~3/256=0.012 floor (softmax of a near-uniform
        # [0,1] map), which the calibrator mapped to ~0 -> confidence head locked
        # LOW -> FALSE_CONFIRMED impossible -> FCR identically 0 on UAV123.
        confidence = float(max(0.0, min(1.0, f_max)))

        pred = (pred_boxes.mean(dim=0) * _SEARCH_SIZE / resize_factor).tolist()
        cx_pred, cy_pred, w_pred, h_pred = pred

        cx_prev = self._state.x + self._state.w / 2
        cy_prev = self._state.y + self._state.h / 2
        half = _SEARCH_SIZE / (2 * resize_factor)

        new_bbox = BBox(
            x=cx_prev + cx_pred - half - w_pred / 2,
            y=cy_prev + cy_pred - half - h_pred / 2,
            w=max(1.0, w_pred),
            h=max(1.0, h_pred),
        )
        # Clamp to frame bounds (project rule: every adapter must clip bbox to
        # frame; the OSTrack/ORTrack family clips natively via lib.utils.box_ops,
        # this adapter had dropped it -> off-frame drift scored IoU~0, understating
        # baseline AUC/Pr@20 and inflating FCR). Restores faithful behaviour.
        _H_img, _W_img = frame.shape[:2]
        _bx = min(max(new_bbox.x, 0.0), max(0.0, float(_W_img) - 1.0))
        _by = min(max(new_bbox.y, 0.0), max(0.0, float(_H_img) - 1.0))
        _bw = max(1.0, min(new_bbox.w, float(_W_img) - _bx))
        _bh = max(1.0, min(new_bbox.h, float(_H_img) - _by))
        new_bbox = BBox(x=_bx, y=_by, w=_bw, h=_bh)
        self._state = new_bbox

        if confidence > 0.25:
            status = "locked"
        elif confidence > 0.10:
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
        )

    def update_with_action(self, frame: np.ndarray, action: "Any") -> TrackState:
        """Action routing stub — ORTrackDeiT does not support CE/search overrides."""
        return self.update(frame)

    def reset(self) -> None:
        self._z_tensor = None
        self._state = None
        self._update_enabled = True
        self._search_factor = _SEARCH_FACTOR

    def set_update_enabled(self, enabled: bool) -> None:
        """CSCAdvisor gate. ORTrack has no internal template update — this is a no-op
        but satisfies the _try_set_update_enabled() protocol in run_with_csc.py."""
        self._update_enabled = bool(enabled)

    def override_search_center(self, cx: float, cy: float, w: float, h: float) -> None:
        """Relocate the search region for the next update() (control hook).

        ORTrack crops the search region around ``self._state`` each frame, so
        setting it relocates where the tracker looks next — the lever behind
        motion_bridge LA recovery and FC hold_lastgood. Mirrors SGLATracker."""
        self._state = BBox(x=cx - w / 2, y=cy - h / 2, w=w, h=h)

    def set_search_factor(self, factor: float) -> None:
        """Widen/narrow the search region for the next update() (LA recovery hook).

        Search crop side = sqrt(w*h) * factor. Clamped to [base, 4*base]."""
        base = _SEARCH_FACTOR
        self._search_factor = float(min(max(float(factor), base), 4.0 * base))

    def reset_search_factor(self) -> None:
        """Restore the default search-region factor (recovery complete)."""
        self._search_factor = _SEARCH_FACTOR

    def redetect(
        self,
        frame: np.ndarray,
        *,
        factors: tuple[float, ...] | list[float] | None = None,
        anchor_bboxes: list[BBox] | None = None,
        include_current: bool = True,
        grid_size: int = 0,
        max_candidates: int = 3,
        min_apce: float = 0.0,
        rank_by: str = "quality",
        top_k: int = 1,
        frame_idx: int = -1,
    ) -> "dict | list[dict] | None":
        """Event-driven wide-crop re-detection (shared loop, frozen template).

        Same interface as SGLATracker/AVTrack.redetect() so the CSC runner's
        ``_try_sgla_redetect`` works for ORTrack unchanged. ORTrack exposes no
        backbone embedding, so candidates carry sim_to_init=NaN — the FC challenge
        association (proximity-based) and quality-ranked LA sgla_redetect both
        work; only the identity switch_mode is unavailable."""
        if self._model is None or self._z_tensor is None or self._state is None:
            return None
        if self._hann is None:
            self._hann = _make_hann(self._device)

        def _fwd(x_tensor):
            return self._model(template=self._z_tensor, search=x_tensor, is_distill=False)

        return _run_redetect(
            model_forward=_fwd,
            sample_target=_sample_target,
            to_tensor=_to_tensor,
            hann=self._hann,
            device=self._device,
            state_bbox=self._state,
            frame=frame,
            factors=factors or (8.0, 12.0, 16.0),
            anchor_bboxes=anchor_bboxes,
            include_current=include_current,
            grid_size=grid_size,
            max_candidates=max_candidates,
            min_apce=min_apce,
            rank_by=rank_by,
            top_k=top_k,
            initial_template_embedding=None,
        )

    @property
    def capabilities(self):
        from uav_tracker.trackers.capabilities import TrackerCapabilities
        # override_search_center() + set_search_factor() are now wired (ORTrack
        # crops around self._state like SGLATrack), so search-region control
        # levers (widen / relocate / motion_bridge / FC hold_lastgood) apply.
        return TrackerCapabilities(can_widen_search=True)

    @property
    def is_stub_mode(self) -> bool:
        return self._is_stub

    def flops_per_update(self) -> float:
        return _FLOPS_PER_UPDATE

    def on_tier_enter(self, ctx: FrameContext) -> None:
        pass

    def on_tier_exit(self, ctx: FrameContext) -> None:
        pass
