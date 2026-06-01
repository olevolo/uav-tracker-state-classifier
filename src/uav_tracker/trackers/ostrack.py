"""OSTrack — One-Stream Tracker (ECCV 2022) adapter.

Reference: Ye et al., "Joint Feature Learning and Relation Modeling for Tracking:
A One-Stream Framework" (ECCV 2022).
Source repo: papers/code/OSTrack/  (do NOT modify; we add it to sys.path lazily).

Default variant: ViT-B/16 384-search (template 192, search 384,
``vitb_384_mae_ce_32x4_ep300``). That YAML matches the bundled checkpoint at
``$UAV_WEIGHTS_ROOT/ostrack/ostrack256_full_ep300.pth.tar`` (the "256"
filename is misleading — pos_embed_z has 144 tokens = 12×12 → 192 px template,
pos_embed_x has 576 tokens = 24×24 → 384 px search).

Pattern mirrors ``src/uav_tracker/trackers/transformer/ortrack.py``:
  sys.path injection, easydict shim, safe-pickle checkpoint loader,
  TrackState telemetry (apce, psr, response_entropy, confidence), and the
  ``set_update_enabled`` / ``_update_enabled`` CSCAdvisor gate (no-op:
  OSTrack does not perform internal template updates at run time).
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, FrameContext, TrackState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo / weights paths
# ---------------------------------------------------------------------------

_OSTRACK_ROOT = Path("papers/code/OSTrack")
_DEFAULT_YAML = (
    _OSTRACK_ROOT
    / "experiments"
    / "ostrack"
    / "vitb_384_mae_ce_32x4_ep300.yaml"
)
_DEFAULT_WEIGHTS_NAME = "ostrack256_full_ep300.pth.tar"

_FLOPS_PER_UPDATE = 48.0e9  # ViT-B/16 at 384 search ≈ 48 GFLOPs (paper Table 6)

# Architecture fallback constants (overridden at load time from the YAML).
_TEMPLATE_SIZE_DEFAULT   = 192
_TEMPLATE_FACTOR_DEFAULT = 2.0
_SEARCH_SIZE_DEFAULT     = 384
_SEARCH_FACTOR_DEFAULT   = 5.0

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Safe pickle module — checkpoints stash training objects from lib.train;
# we stub them out to avoid importing the full training harness.
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

    mod = types.ModuleType("_ostrack_safe_pickle")
    mod.Unpickler = _SafeUnpickler
    for attr in ("UnpicklingError", "PicklingError", "HIGHEST_PROTOCOL",
                 "DEFAULT_PROTOCOL", "dumps", "loads"):
        setattr(mod, attr, getattr(pickle, attr))
    return mod


_SAFE_PICKLE = _make_safe_pickle_module()


def _load_ostrack_state(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu",
                      weights_only=False, pickle_module=_SAFE_PICKLE)
    if isinstance(ckpt, dict) and "net" in ckpt:
        return ckpt["net"]
    return ckpt


# ---------------------------------------------------------------------------
# sys.path injection + easydict shim
# ---------------------------------------------------------------------------

def _ensure_ostrack_on_path() -> None:
    root = str(_OSTRACK_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

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


def _build_model(device: torch.device, yaml_path: Path):
    _ensure_ostrack_on_path()
    from lib.config.ostrack.config import cfg, update_config_from_file
    from lib.models.ostrack import build_ostrack

    update_config_from_file(str(yaml_path))
    model = build_ostrack(cfg, training=False)
    model = model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Preprocessing helpers — local ports; do not import OSTrack training utils.
# ---------------------------------------------------------------------------

def _sample_target(
    frame: np.ndarray,
    bbox: BBox,
    factor: float,
    out_size: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Mean-padded square crop centred on bbox. Returns (patch_rgb, resize_factor, att_mask)."""
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    crop_sz = math.ceil(math.sqrt(max(w, 1.0) * max(h, 1.0)) * factor)
    if crop_sz < 1:
        crop_sz = 1

    cx, cy = x + 0.5 * w, y + 0.5 * h
    x1 = round(cx - 0.5 * crop_sz)
    y1 = round(cy - 0.5 * crop_sz)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    H, W = frame.shape[:2]
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - W + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - H + 1, 0)

    crop = frame[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    crop_padded = cv2.copyMakeBorder(crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                     cv2.BORDER_CONSTANT)

    Hc, Wc = crop_padded.shape[:2]
    att_mask = np.ones((Hc, Wc), dtype=np.float32)
    end_x = -x2_pad if x2_pad else None
    end_y = -y2_pad if y2_pad else None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    resize_factor = out_size / crop_sz
    crop_resized = cv2.resize(crop_padded, (out_size, out_size))
    att_mask = cv2.resize(att_mask, (out_size, out_size)).astype(np.bool_)
    return crop_resized, resize_factor, att_mask


def _to_nested_tensor(
    patch_rgb: np.ndarray,
    att_mask: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
):
    """Pack normalised (H,W,3) uint8 RGB patch into a NestedTensor."""
    from lib.utils.misc import NestedTensor  # needs sys.path

    img = torch.from_numpy(patch_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
    img = (img / 255.0 - mean) / std
    mask = torch.from_numpy(att_mask).to(torch.bool).unsqueeze(0).to(device)
    return NestedTensor(img, mask)


def _make_hann2d(feat_sz: int, device: torch.device) -> torch.Tensor:
    """2-D Hann window (1,1,feat_sz,feat_sz) on target device."""
    h1d = 0.5 * (1 - torch.cos(
        2 * math.pi / (feat_sz + 1) * torch.arange(1, feat_sz + 1, dtype=torch.float32)
    ))
    return (h1d.view(-1, 1) * h1d.view(1, -1)).view(1, 1, feat_sz, feat_sz).to(device)


def _transform_init_bbox_to_crop(
    bbox: BBox,
    resize_factor: float,
    template_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalised [x,y,w,h] tensor (1,1,4) for the CE template mask."""
    box_in = torch.tensor([bbox.x, bbox.y, bbox.w, bbox.h], dtype=torch.float32)
    crop_sz = torch.tensor([template_size, template_size], dtype=torch.float32)
    box_out_center = (crop_sz - 1) / 2
    box_out_wh = box_in[2:4] * resize_factor
    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    box_out = box_out / crop_sz[0]
    return box_out.view(1, 1, 4).to(device)


def _clip_box(box: list, H: int, W: int, margin: int = 10) -> list:
    """Clip predicted xywh box to frame bounds (mirrors OSTrack lib.utils.box_ops)."""
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@TRACKERS.register("ostrack_384")
@TRACKERS.register("ostrack")  # primary alias — used by configs and CSC pipeline
class OSTrack:
    """OSTrack ViT-B/16 384 — one-stream transformer tracker for UAV benchmarking.

    tier_hint=2 (heavy tracker, ~48 GFLOPs, targets <50 ms on a T4 GPU).
    OSTrack has no online template-update mechanism; set_update_enabled is a
    no-op that satisfies the CSCAdvisor _try_set_update_enabled protocol.

    Weights path resolution order:
      1. ``weights_path`` constructor argument
      2. ``$UAV_WEIGHTS_ROOT/ostrack/<_DEFAULT_WEIGHTS_NAME>``
      3. ``~/uav-tracker-weights/ostrack/<_DEFAULT_WEIGHTS_NAME>``

    OSTrack repo must be present at ``papers/code/OSTrack/`` (relative to cwd)
    or the build step will raise NotImplementedError with a clear message.
    """

    name: str = "ostrack"
    tier_hint: int = 2

    def __init__(
        self,
        device: str = "auto",
        weights_path: str | None = None,
        yaml_path: str | None = None,
    ) -> None:
        self._device_str = device
        self._weights_path = weights_path
        self._yaml_path = Path(yaml_path) if yaml_path else _DEFAULT_YAML
        self._model = None
        self._cfg = None
        # Geometry — filled by _load() from the YAML
        self._template_size: int = _TEMPLATE_SIZE_DEFAULT
        self._template_factor: float = _TEMPLATE_FACTOR_DEFAULT
        self._search_size: int = _SEARCH_SIZE_DEFAULT
        self._search_factor: float = _SEARCH_FACTOR_DEFAULT
        self._feat_sz: int = 0
        # Per-sequence state
        self._hann: torch.Tensor | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._z_dict = None          # NestedTensor
        self._box_mask_z: torch.Tensor | None = None
        self._state: BBox | None = None
        self._is_stub: bool = True
        self._template_age: int = 0
        self._update_enabled: bool = True  # CSCAdvisor gate (no-op for OSTrack)

    # ------------------------------------------------------------------
    # Device property
    # ------------------------------------------------------------------

    @property
    def _device(self) -> torch.device:
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if self._device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("OSTrack: cuda requested but not available — falling back to CPU")
            return torch.device("cpu")
        return torch.device(self._device_str)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _resolve_weights_path(self) -> Path:
        if self._weights_path:
            return Path(self._weights_path).expanduser()
        env = os.environ.get("UAV_WEIGHTS_ROOT", "").strip()
        base = Path(env).expanduser() if env else Path.home() / "uav-tracker-weights"
        return base / "ostrack" / _DEFAULT_WEIGHTS_NAME

    def _load(self) -> None:
        dev = self._device

        try:
            model, cfg = _build_model(dev, self._yaml_path)
        except Exception as exc:
            raise NotImplementedError(
                f"Failed to build OSTrack model from {_OSTRACK_ROOT}. "
                f"Ensure papers/code/OSTrack exists and lib/ is importable. "
                f"Original error: {exc}"
            ) from exc

        self._cfg = cfg

        # Pull inference geometry from config so both 256 and 384 YAMLs work.
        self._template_size   = int(cfg.TEST.TEMPLATE_SIZE)
        self._template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
        self._search_size     = int(cfg.TEST.SEARCH_SIZE)
        self._search_factor   = float(cfg.TEST.SEARCH_FACTOR)
        stride = int(cfg.MODEL.BACKBONE.STRIDE)
        self._feat_sz = self._search_size // stride

        weights_path = self._resolve_weights_path()
        if weights_path.exists():
            try:
                state = _load_ostrack_state(weights_path)
                missing, unexpected = model.load_state_dict(state, strict=False)
                self._is_stub = bool(missing) and len(missing) > 5
                if not missing:
                    logger.info("OSTrack weights loaded from %s", weights_path)
                else:
                    logger.warning(
                        "OSTrack: %d missing keys (e.g. %s)", len(missing), missing[:3]
                    )
                if unexpected:
                    logger.debug("OSTrack: %d unexpected keys", len(unexpected))
            except Exception as exc:
                logger.warning("OSTrack weight load failed: %s — random init", exc)
                self._is_stub = True
        else:
            logger.info("OSTrack weights not found at %s — random init", weights_path)
            self._is_stub = True

        self._model = model
        self._hann = _make_hann2d(self._feat_sz, dev)
        self._mean = torch.tensor(_IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
        self._std  = torch.tensor(_IMAGENET_STD,  device=dev).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    # Tracker API
    # ------------------------------------------------------------------

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        if self._model is None:
            self._load()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        z_patch, resize_factor, z_amask = _sample_target(
            rgb, bbox, self._template_factor, self._template_size
        )
        self._z_dict = _to_nested_tensor(
            z_patch, z_amask, self._mean, self._std, self._device
        )

        # Candidate-elimination template mask (enabled when CE_LOC is set in YAML).
        cfg = self._cfg
        if getattr(cfg.MODEL.BACKBONE, "CE_LOC", None):
            from lib.utils.ce_utils import generate_mask_cond
            tmpl_bbox = _transform_init_bbox_to_crop(
                bbox, resize_factor, self._template_size, self._device
            ).squeeze(1)
            self._box_mask_z = generate_mask_cond(cfg, 1, self._device, tmpl_bbox)
        else:
            self._box_mask_z = None

        self._state = bbox
        self._template_age = 0

    def update(self, frame: np.ndarray) -> TrackState:
        if self._model is None or self._z_dict is None or self._state is None:
            return TrackState(bbox=BBox(0, 0, 1, 1), confidence=0.0, status="lost")

        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_patch, resize_factor, x_amask = _sample_target(
            rgb, self._state, self._search_factor, self._search_size
        )
        x_dict = _to_nested_tensor(
            x_patch, x_amask, self._mean, self._std, self._device
        )

        with torch.no_grad():
            out = self._model.forward(
                template=self._z_dict.tensors,
                search=x_dict.tensors,
                ce_template_mask=self._box_mask_z,
            )

        score_map_raw = out["score_map"]          # (1,1,feat_sz,feat_sz)
        response = self._hann * score_map_raw      # Hann-weighted

        feat_sz = self._feat_sz

        # Telemetry derived from the Hann-weighted response map.
        with torch.no_grad():
            f = response.view(-1)
            f_max = float(f.max())
            f_min = float(f.min())
            f_mean = float(f.mean())
            f_std  = float(f.std(unbiased=False))

            # APCE — Average Peak-to-Correlation Energy
            denom = float(((f - f_min) ** 2).mean())
            apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))

            # PSR — Peak-to-Sidelobe Ratio (5-cell radius)
            sm2d = response.squeeze()             # (feat_sz, feat_sz)
            flat_idx = int(sm2d.view(-1).argmax())
            peak_r, peak_c = flat_idx // feat_sz, flat_idx % feat_sz
            r_idx = torch.arange(feat_sz, device=sm2d.device).view(feat_sz, 1).expand(feat_sz, feat_sz)
            c_idx = torch.arange(feat_sz, device=sm2d.device).view(1, feat_sz).expand(feat_sz, feat_sz)
            peak_mask = ((r_idx - peak_r).abs() <= 5) & ((c_idx - peak_c).abs() <= 5)
            sidelobe = sm2d[~peak_mask]
            if sidelobe.numel() > 0:
                psr = float((f_max - float(sidelobe.mean())) / (float(sidelobe.std(unbiased=False)) + 1e-8))
            else:
                psr = 0.0

            # Response entropy — softmax over flattened Hann-weighted response.
            probs = torch.softmax(f, dim=0)
            response_entropy = float(-(probs * (probs + 1e-12).log()).sum())

            # Auxiliary raw map stats.
            response_max  = float(score_map_raw.max())
            response_mean = float(score_map_raw.mean())
            response_std  = float(score_map_raw.std(unbiased=False))

        # Decode bbox: cal_bbox returns normalised [cx,cy,w,h].
        pred_boxes = self._model.box_head.cal_bbox(
            response, out["size_map"], out["offset_map"]
        ).view(-1, 4)
        pred = (pred_boxes.mean(dim=0) * self._search_size / resize_factor).tolist()
        cx, cy, w_pred, h_pred = pred

        cx_prev = self._state.x + 0.5 * self._state.w
        cy_prev = self._state.y + 0.5 * self._state.h
        half_side = 0.5 * self._search_size / resize_factor
        x = cx + (cx_prev - half_side) - 0.5 * w_pred
        y = cy + (cy_prev - half_side) - 0.5 * h_pred
        clipped = _clip_box([x, y, w_pred, h_pred], H, W, margin=10)
        new_bbox = BBox(
            x=float(clipped[0]),
            y=float(clipped[1]),
            w=float(max(1.0, clipped[2])),
            h=float(max(1.0, clipped[3])),
        )
        self._state = new_bbox
        self._template_age += 1

        confidence = float(f_max)
        if confidence > 0.5:
            status = "locked"
        elif confidence > 0.25:
            status = "uncertain"
        else:
            status = "lost"

        raw = {
            "score_max":      float(f_max),
            "response_max":   float(response_max),
            "response_mean":  float(response_mean),
            "response_std":   float(response_std),
            "f_min":          float(f_min),
            "f_mean":         float(f_mean),
            "f_std":          float(f_std),
            "template_age":   int(self._template_age),
            "search_factor":  float(self._search_factor),
            "feat_sz":        int(feat_sz),
        }

        return TrackState(
            bbox=new_bbox,
            confidence=confidence,
            status=status,
            apce=apce,
            psr=psr,
            response_entropy=response_entropy,
            aux={
                "raw": raw,
                "score_map_stats": {
                    "top1":   float(f_max),
                    "peak_r": int(peak_r),
                    "peak_c": int(peak_c),
                },
            },
        )

    def update_with_action(self, frame: np.ndarray, action: "Any") -> TrackState:
        """Action routing stub — OSTrack does not support CE/search overrides."""
        return self.update(frame)

    def reset(self) -> None:
        self._z_dict = None
        self._box_mask_z = None
        self._state = None
        self._template_age = 0
        self._update_enabled = True

    def set_update_enabled(self, enabled: bool) -> None:
        """CSCAdvisor gate. OSTrack has no internal template update — this is a no-op
        but satisfies the _try_set_update_enabled() protocol in run_with_csc.py."""
        self._update_enabled = bool(enabled)

    @property
    def capabilities(self):
        from uav_tracker.trackers.capabilities import TrackerCapabilities
        return TrackerCapabilities()  # all defaults: only can_reject_bbox + can_force_reinit

    # ------------------------------------------------------------------
    # Misc.
    # ------------------------------------------------------------------

    @property
    def is_stub_mode(self) -> bool:
        return self._is_stub

    def flops_per_update(self) -> float:
        return _FLOPS_PER_UPDATE

    def on_tier_enter(self, ctx: FrameContext) -> None:
        pass

    def on_tier_exit(self, ctx: FrameContext) -> None:
        pass
