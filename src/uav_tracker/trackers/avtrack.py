"""AVTrack — Adaptive ViT Tracker (DeiT-tiny variant) baseline + telemetry adapter.

Reference: "Learning Adaptive and View-Invariant Vision Transformer for
Real-Time UAV Tracking", ECCV 2024.
Source repo: papers/code/AVTrack/  (do NOT modify upstream files)

Architecture: DeiT-tiny backbone (embed_dim=192, depth=12, num_heads=3, patch=16)
  Template 128×128 (8×8 = 64 tokens), Search 256×256 (16×16 = 256 tokens), stride 16
  CENTER head, IS_DISTILL=False for the standard ``deit_tiny_patch16_224`` variant.

Adaptive layer skipping
-----------------------
Per ``papers/code/AVTrack/lib/models/avtrack/vision_transformer.py::forward_features``,
the gating is::

    for i, blk in enumerate(self.blocks):           # depth=12
        if i > 1 and not is_distill:
            prob_active = sigmoid(blk.active_score_module(x[:, :, 0]))   # (B, 1)
            idx, _ = torch.where(prob_active > 0.5)
            if len(idx) > 0 and not training:
                x = blk(x)
            probs_active.append(prob_active)
        else:
            x = blk(x)                              # blocks 0,1 always run

So at inference (B=1), block ``i`` (i >= 2) is *executed* iff
``prob_active.item() > 0.5``. We expose ``active_layers`` per frame and
``token_keep_ratio`` (=1.0 here — AVTrack skips *layers*, not tokens).

Expected weights: ``$UAV_WEIGHTS_ROOT/avtrack/AVTrack-DeiT.pth.tar``
  or ``~/uav-tracker-weights/avtrack/AVTrack-DeiT.pth.tar``.
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
from uav_tracker.trackers._redetect_common import (
    select_candidate_peak_indices as _select_candidate_peak_indices,
    candidate_bbox_from_peak as _candidate_bbox_from_peak,
    redetect_rank_key as _redetect_rank_key,
    redetect_topk as _redetect_topk,
    sim_init_at_cell as _sim_init_at_cell,
)

logger = logging.getLogger(__name__)

# Project root: src/uav_tracker/trackers/avtrack.py → parents[3] = project root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AVTRACK_ROOT = _REPO_ROOT / "papers" / "code" / "AVTrack"
_AVTRACK_CONFIG_REL = "experiments/avtrack/deit_tiny_patch16_224.yaml"
_WEIGHTS_NAME = "AVTrack-DeiT.pth.tar"

# Auto-detect weights dir: $UAV_WEIGHTS_ROOT/avtrack or ~/uav-tracker-weights/avtrack
def _default_weights_dir() -> Path:
    env = os.environ.get("UAV_WEIGHTS_ROOT")
    if env:
        return Path(env).expanduser() / "avtrack"
    return Path("~/uav-tracker-weights/avtrack").expanduser()


# DeiT-tiny ViT-tiny FLOPs ~0.6-0.9 GFLOPs/forward. AVTrack skips ~30% of layers
# adaptively at inference, so realised compute is lower. Use 0.7e9 as a coarse
# estimate; the real cost depends on per-frame active_layers (telemetered).
_FLOPS_PER_UPDATE = 0.7e9

# Architecture constants (deit_tiny_patch16_224.yaml + AVTrack search/template).
_TEMPLATE_SIZE = 128
_TEMPLATE_FACTOR = 2.0
_SEARCH_SIZE = 256
_SEARCH_FACTOR = 4.0
_FEAT_SZ = 16            # 256 / 16 (patch stride)
_N_Z = 64                # 8 × 8 template tokens
_N_X = 256               # 16 × 16 search tokens
_DEPTH = 12              # DeiT-tiny depth
_ALWAYS_ON_LAYERS = 2    # Blocks 0, 1 always run regardless of gate.

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Checkpoint loader — checkpoint may contain training objects from lib.train,
# so we stub those classes to avoid importing the full training harness.
# (Same recipe as SGLATrack / ORTrack adapters.)
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

    mod = types.ModuleType("_avtrack_safe_pickle")
    mod.Unpickler = _SafeUnpickler
    for attr in ("UnpicklingError", "PicklingError", "HIGHEST_PROTOCOL",
                 "DEFAULT_PROTOCOL", "dumps", "loads"):
        setattr(mod, attr, getattr(pickle, attr))
    return mod


_SAFE_PICKLE = _make_safe_pickle_module()


def _load_avtrack_state(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu",
                      weights_only=False, pickle_module=_SAFE_PICKLE)
    if isinstance(ckpt, dict) and "net" in ckpt:
        return ckpt["net"]
    return ckpt


# ---------------------------------------------------------------------------
# AVTrack source on sys.path + easydict shim (lazy / idempotent).
# ---------------------------------------------------------------------------

def _ensure_avtrack_on_path() -> None:
    root = str(_AVTRACK_ROOT.resolve())
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
                try:
                    return self[k]
                except KeyError:
                    raise AttributeError(k)

            def __setattr__(self, k, v):
                self[k] = v

            def __delattr__(self, k):
                del self[k]

        shim = types.ModuleType("easydict")
        shim.EasyDict = _EasyDict
        sys.modules["easydict"] = shim


def _build_model(device: torch.device):
    """Build AVTrack-DeiT model with no online pretrained download.

    AVTrack's ``build_avtrack`` calls ``deit_tiny_patch16_224(pretrained=True)``
    which would attempt to fetch weights from facebookresearch/deit. We monkey-
    patch the timm backbone constructors to ``pretrained=False`` while AVTrack
    builds, then restore them — the AVTrack checkpoint we load afterwards
    already contains all backbone weights.
    """
    _ensure_avtrack_on_path()

    from lib.config.avtrack.config import cfg, update_config_from_file
    from lib.models.avtrack import build_avtrack
    import lib.models.avtrack.avtrack as _avtrack_mod
    import lib.models.avtrack.deit as _deit_mod
    import lib.models.avtrack.vision_transformer as _vt_mod
    import lib.models.avtrack.eva as _eva_mod

    yaml_path = _AVTRACK_ROOT.resolve() / _AVTRACK_CONFIG_REL
    update_config_from_file(str(yaml_path))

    # Patch any timm-style backbone factory imported into the avtrack namespace
    # so ``pretrained=True`` is forced to False. We patch in *all* relevant
    # modules because Python rebinds names at import time.
    _builders = [
        "deit_tiny_patch16_224", "deit_tiny_patch16_224_distill",
        "deit_small_patch16_224", "deit_base_patch16_224",
        "vit_tiny_patch16_224", "vit_tiny_distilled_patch16_224",
        "eva02_tiny_patch14_224", "eva02_tiny_patch14_224_distill",
    ]
    _saved: dict[tuple[object, str], object] = {}
    for _name in _builders:
        for _mod in (_avtrack_mod, _deit_mod, _vt_mod, _eva_mod):
            if hasattr(_mod, _name):
                _orig = getattr(_mod, _name)
                _saved[(_mod, _name)] = _orig
                setattr(_mod, _name,
                        lambda *a, _f=_orig, **kw: _f(*a, **{**kw, "pretrained": False}))
    try:
        model = build_avtrack(cfg, training=False)
    finally:
        for (_mod, _name), _orig in _saved.items():
            setattr(_mod, _name, _orig)

    model = model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Preprocessing helpers — match AVTrack's lib/train/data/processing_utils.py
# (mean-padded square crop, ImageNet normalisation).
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

@TRACKERS.register("avtrack")
class AVTrackAdapter:
    """AVTrack DeiT-tiny — adaptive-layer ViT tracker for UAV.

    Telemetry hooks (CSC pipeline):
      * ``confidence`` = score_max (post-Hann), clamped to [0, 1]
      * ``apce``, ``psr``, ``response_entropy``: standard score-map quality
      * ``raw.active_layers``: integer count of executed transformer blocks
      * ``raw.token_keep_ratio``: 1.0 (AVTrack skips layers, not tokens)
      * ``raw.score_max``, ``response_max/mean/std``: score-map summary stats

    tier_hint=1 (lightweight learned tracker, ~0.7 GFLOPs effective).
    """

    name: str = "avtrack"
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
        self._search_factor: float = _SEARCH_FACTOR
        self._is_stub: bool = True
        self._update_enabled: bool = True  # CSCAdvisor gate (no-op: AVTrack has no internal template update)
        self._initial_template_embedding: torch.Tensor | None = None  # frozen at frame 0 for redetect identity
        self._last_cosine_sim: float = 1.0  # cosine(mean search tokens, template); matches SGLATrack signal

    # --------------------------------------------------------------------- #
    # Lifecycle                                                               #
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

    def _load(self) -> None:
        dev = self._device

        try:
            model, _ = _build_model(dev)
        except Exception as exc:
            raise NotImplementedError(
                f"Failed to build AVTrack model from {_AVTRACK_ROOT}. "
                f"Check that papers/code/AVTrack exists and lib/ is importable. "
                f"Original error: {exc}"
            ) from exc

        weights_path = self._weights_path
        if weights_path is None:
            weights_path = str(_default_weights_dir() / _WEIGHTS_NAME)
        p = Path(weights_path).expanduser()

        if p.exists():
            try:
                state = _load_avtrack_state(p)
                missing, unexpected = model.load_state_dict(state, strict=False)
                # AVTrack training checkpoints include head-only and statistics-
                # network weights that aren't all required at inference; tolerate
                # them but flag if the *backbone / box_head* are missing.
                core_missing = [k for k in missing
                                if k.startswith(("backbone.", "box_head."))]
                self._is_stub = bool(core_missing)
                if not core_missing:
                    logger.info(
                        "AVTrackAdapter weights loaded from %s "
                        "(missing=%d, unexpected=%d)",
                        p, len(missing), len(unexpected),
                    )
                else:
                    logger.warning(
                        "AVTrackAdapter: %d core keys missing (e.g. %s) — "
                        "treating as stub mode",
                        len(core_missing), core_missing[:3],
                    )
            except Exception as exc:
                logger.warning(
                    "AVTrackAdapter weight load failed: %s — random init "
                    "(MISSING_WEIGHTS — dry-run mode)", exc)
                self._is_stub = True
        else:
            logger.warning(
                "AVTrackAdapter weights not found at %s — random init "
                "(MISSING_WEIGHTS — dry-run mode). "
                "Place AVTrack-DeiT.pth.tar at %s or set $UAV_WEIGHTS_ROOT.",
                p, p,
            )
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
        self._initial_template_embedding = None
        self._last_cosine_sim = 1.0

    def update(self, frame: np.ndarray) -> TrackState:
        if self._model is None or self._z_tensor is None or self._state is None:
            return TrackState(bbox=BBox(0, 0, 1, 1), confidence=0.0, status="lost")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, resize_factor = _sample_target(
            rgb, self._state, self._search_factor, _SEARCH_SIZE
        )
        x_tensor = _to_tensor(patch, self._device)

        # AVTrack expects template/search anno args. Empty lists are accepted
        # by the inference path because ``self.training is False``.
        with torch.no_grad():
            out = self._model(
                template=self._z_tensor,
                search=x_tensor,
                template_anno=[],
                search_anno=[],
                is_distill=False,
            )

        # Freeze initial template embedding on first frame for redetect identity.
        if self._initial_template_embedding is None and "backbone_feat" in out:
            _bfeat = out["backbone_feat"]  # (1, N_Z+N_X, C) = (1, 320, 192)
            self._initial_template_embedding = _bfeat[:, :_N_Z, :].mean(dim=1).squeeze(0).detach()

        # Compute cosine similarity between mean search tokens and template.
        # Matches SGLATrack's _last_cosine_sim so the 'cosine' LA gate works cross-tracker.
        if self._initial_template_embedding is not None and "backbone_feat" in out:
            with torch.no_grad():
                _bfeat_cur = out["backbone_feat"]
                _search_mean = _bfeat_cur[:, _N_Z:, :].mean(dim=1).squeeze(0)
                self._last_cosine_sim = float(
                    torch.nn.functional.cosine_similarity(
                        _search_mean.unsqueeze(0),
                        self._initial_template_embedding.unsqueeze(0),
                    ).item()
                )
        else:
            self._last_cosine_sim = 1.0

        # ----- Adaptive-layer telemetry (the key AVTrack signal) -----
        # ``probs_active`` is a list of (B, 1) sigmoid tensors, one per gated
        # block (depth - 2 = 10 entries). At inference, block i is *executed*
        # iff ``(prob_active > 0.5).any()``. With B=1 this collapses to a
        # single scalar comparison per block.
        probs_active = out.get("probs_active", []) or []
        active_per_block: list[int] = []
        prob_per_block: list[float] = []
        for p_gate in probs_active:
            try:
                p_val = float(p_gate.detach().max().item())
            except Exception:
                p_val = 0.0
            prob_per_block.append(p_val)
            active_per_block.append(1 if p_val > 0.5 else 0)
        active_layers = _ALWAYS_ON_LAYERS + sum(active_per_block)
        token_keep_ratio = 1.0  # AVTrack skips layers, not tokens.

        # ----- Score-map quality (Hann-windowed) -----
        score_map = out["score_map"]                          # (1, 1, 16, 16)
        score_map = self._hann * score_map

        with torch.no_grad():
            f = score_map.view(-1)
            f_max = float(f.max())
            f_min = float(f.min())
            f_mean = float(f.mean())
            f_std = float(f.std())
            denom = float(((f - f_min) ** 2).mean())
            apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))

            sm2d = score_map.squeeze()                        # (16, 16)
            flat_idx = int(sm2d.view(-1).argmax())
            peak_r = flat_idx // _FEAT_SZ
            peak_c = flat_idx % _FEAT_SZ
            r_idx = torch.arange(_FEAT_SZ, device=sm2d.device).view(_FEAT_SZ, 1).expand(_FEAT_SZ, _FEAT_SZ)
            c_idx = torch.arange(_FEAT_SZ, device=sm2d.device).view(1, _FEAT_SZ).expand(_FEAT_SZ, _FEAT_SZ)
            peak_mask = ((r_idx - peak_r).abs() <= 5) & ((c_idx - peak_c).abs() <= 5)
            sidelobe = sm2d[~peak_mask]
            if len(sidelobe) > 0:
                psr = float((f_max - sidelobe.mean()) / (sidelobe.std() + 1e-8))
            else:
                psr = 0.0

            # L1-normalised entropy (matches SGLATrack convention).
            f_pos = f - f.min()
            probs_norm = f_pos / (f_pos.sum() + 1e-8)
            response_entropy = float(-(probs_norm * (probs_norm + 1e-8).log()).sum())

            score_max = f_max  # primary "raw confidence" reading

        # ----- Decode bbox from CENTER head -----
        pred_boxes = self._model.box_head.cal_bbox(
            score_map, out["size_map"], out["offset_map"]
        ).view(-1, 4)
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
        # frame). Off-frame drift was scoring IoU~0 even when the on-screen part
        # overlapped the target -> baseline AUC/Pr@20 understated, FCR inflated.
        _H_img, _W_img = frame.shape[:2]
        _bx = min(max(new_bbox.x, 0.0), max(0.0, float(_W_img) - 1.0))
        _by = min(max(new_bbox.y, 0.0), max(0.0, float(_H_img) - 1.0))
        _bw = max(1.0, min(new_bbox.w, float(_W_img) - _bx))
        _bh = max(1.0, min(new_bbox.h, float(_H_img) - _by))
        new_bbox = BBox(x=_bx, y=_by, w=_bw, h=_bh)
        self._state = new_bbox

        # Confidence = score_max (per task spec). The CENTER head emits a
        # sigmoid-bounded score map, so f_max is already in [0, 1].
        confidence = float(max(0.0, min(1.0, score_max)))

        if confidence > 0.5:
            status = "locked"
        elif confidence > 0.2:
            status = "uncertain"
        else:
            status = "lost"

        raw = {
            "score_max": float(score_max),
            "response_max": float(f_max),
            "response_mean": float(f_mean),
            "response_std": float(f_std),
            "active_layers": int(active_layers),
            "token_keep_ratio": float(token_keep_ratio),
            "active_per_block": list(active_per_block),
            "prob_per_block": [float(v) for v in prob_per_block],
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

    def update_with_action(self, frame: np.ndarray, action: object) -> TrackState:
        """Action routing stub — AVTrack does not support CE/search overrides."""
        return self.update(frame)

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
    ) -> dict | list[dict] | None:
        """Event-driven self re-detection using the frozen AVTrack template.

        Wide-crop forward passes with the frozen template; returns the best
        frame-space candidate without mutating ``self._state``. Same interface
        as SGLATracker.redetect() so the CSC runner's ``_try_sgla_redetect``
        works for AVTrack unchanged (it calls ``tracker.redetect()`` generically).
        """
        if self._model is None or self._z_tensor is None or self._state is None:
            return None
        if self._hann is None:
            self._hann = _make_hann(self._device)

        _factors = tuple(float(f) for f in (factors or (8.0, 12.0, 16.0)) if float(f) > 0.0)
        if not _factors:
            return None

        H, W = frame.shape[:2]
        anchors: list[tuple[str, BBox]] = []
        if include_current:
            anchors.append(("current", self._state))
        if anchor_bboxes:
            for i, b in enumerate(anchor_bboxes):
                if b is not None and b.w > 0 and b.h > 0:
                    anchors.append((f"hint{i}", b))

        if grid_size and grid_size > 1:
            ref = anchor_bboxes[0] if anchor_bboxes else self._state
            gw = max(1.0, float(ref.w))
            gh = max(1.0, float(ref.h))
            for gy in range(int(grid_size)):
                cy = (gy + 0.5) * float(H) / float(grid_size)
                for gx in range(int(grid_size)):
                    cx = (gx + 0.5) * float(W) / float(grid_size)
                    anchors.append((
                        f"grid{gy}_{gx}",
                        BBox(x=cx - gw / 2.0, y=cy - gh / 2.0, w=gw, h=gh),
                    ))

        # De-duplicate near-identical anchors.
        uniq: list[tuple[str, BBox]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for name, b in anchors:
            key = (
                int(round(b.x + b.w / 2.0)),
                int(round(b.y + b.h / 2.0)),
                int(round(b.w)),
                int(round(b.h)),
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append((name, b))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        best: dict | None = None
        collected: list[dict] = []

        for anchor_name, anchor in uniq:
            for factor in _factors:
                patch, resize_factor = _sample_target(rgb, anchor, factor, _SEARCH_SIZE)
                x_tensor = _to_tensor(patch, self._device)
                with torch.no_grad():
                    out = self._model(
                        template=self._z_tensor,
                        search=x_tensor,
                        template_anno=[],
                        search_anno=[],
                        is_distill=False,
                    )
                    score_map = self._hann * out["score_map"]

                    # Identity verification via backbone_feat (same N_Z+N_X layout as SGLATrack).
                    _bf = None
                    _crop_sim_init = float("nan")
                    if self._initial_template_embedding is not None and "backbone_feat" in out:
                        _bf = out["backbone_feat"][:, -_N_X:, :].squeeze(0)  # (256, 192) search tokens
                        _scf = score_map.reshape(_N_X)
                        _pk = int(_scf.argmax().item())
                        _crop_sim_init = _sim_init_at_cell(
                            _bf, _pk // _FEAT_SZ, _pk % _FEAT_SZ, self._initial_template_embedding
                        )

                    flat = score_map.reshape(-1)
                    f_max = float(flat.max())
                    f_min = float(flat.min())
                    denom = float(((flat - f_min) ** 2).mean())
                    apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))
                    if apce < min_apce:
                        continue

                    f_pos = flat - flat.min()
                    probs = f_pos / (f_pos.sum() + 1e-8)
                    entropy = float(-(probs * (probs + 1e-8).log()).sum())

                    peak_indices = _select_candidate_peak_indices(
                        score_map, max_candidates=max(1, int(max_candidates))
                    )
                    top_score = float(flat[peak_indices[0]]) if peak_indices else 0.0

                for cand_idx in peak_indices:
                    cand = _candidate_bbox_from_peak(
                        idx=cand_idx,
                        score=float(flat[cand_idx]),
                        top_score=top_score,
                        size_map=out["size_map"],
                        offset_map=out["offset_map"],
                        prev_bbox=anchor,
                        resize_factor=resize_factor,
                        rank=peak_indices.index(cand_idx),
                    )
                    x, y, bw, bh = cand["bbox"]
                    bx = min(max(float(x), 0.0), max(0.0, float(W) - 1.0))
                    by = min(max(float(y), 0.0), max(0.0, float(H) - 1.0))
                    bw = max(1.0, min(float(bw), float(W) - bx))
                    bh = max(1.0, min(float(bh), float(H) - by))
                    cx = bx + bw / 2.0
                    cy = by + bh / 2.0
                    cand["bbox"] = [float(bx), float(by), float(bw), float(bh)]
                    cand["center"] = [float(cx), float(cy)]

                    score_ratio = float(cand.get("score_ratio", 0.0))
                    quality = float(apce) * max(0.05, score_ratio)

                    _c_row = cand.get("row")
                    _c_col = cand.get("col")
                    if _bf is not None and _c_row is not None and _c_col is not None:
                        _cand_sim_init = _sim_init_at_cell(
                            _bf, int(_c_row), int(_c_col), self._initial_template_embedding
                        )
                    else:
                        _cand_sim_init = float(_crop_sim_init)

                    out_cand = {
                        "bbox": cand["bbox"],
                        "center": cand["center"],
                        "score": float(cand.get("score", 0.0)),
                        "score_ratio": score_ratio,
                        "rank": int(cand.get("rank", 0)),
                        "factor": float(factor),
                        "anchor": anchor_name,
                        "apce": float(apce),
                        "response_entropy": float(entropy),
                        "quality": float(quality),
                        "sim_to_init": float(_cand_sim_init),
                    }
                    if best is None or _redetect_rank_key(out_cand, rank_by) > _redetect_rank_key(best, rank_by):
                        best = out_cand
                    if top_k and top_k > 1:
                        collected.append(out_cand)

        if top_k and top_k > 1:
            return _redetect_topk(collected, rank_by, int(top_k))
        return best

    def reset(self) -> None:
        self._z_tensor = None
        self._state = None
        self._update_enabled = True
        self._search_factor = _SEARCH_FACTOR
        self._initial_template_embedding = None
        self._last_cosine_sim = 1.0

    def set_update_enabled(self, enabled: bool) -> None:
        """CSCAdvisor gate. AVTrack has no internal template update — this is a no-op
        but satisfies the _try_set_update_enabled() protocol in run_with_csc.py."""
        self._update_enabled = bool(enabled)

    def override_search_center(self, cx: float, cy: float, w: float, h: float) -> None:
        """Relocate the search region for the next update() (control hook).

        AVTrack crops the search region around ``self._state`` each frame, so
        setting it relocates where the tracker looks next — the lever behind
        motion_bridge LA recovery, FC hold_lastgood, and the FC challenge switch.
        Mirrors SGLATracker.override_search_center."""
        self._state = BBox(x=cx - w / 2, y=cy - h / 2, w=w, h=h)

    def set_search_factor(self, factor: float) -> None:
        """Widen/narrow the search region for the next update() (LA recovery hook).

        Search crop side = sqrt(w*h) * factor. Clamped to [base, 4*base] to avoid
        degenerate low-resolution crops. Mirrors SGLATracker.set_search_factor."""
        base = _SEARCH_FACTOR
        self._search_factor = float(min(max(float(factor), base), 4.0 * base))

    def reset_search_factor(self) -> None:
        """Restore the default search-region factor (recovery complete)."""
        self._search_factor = _SEARCH_FACTOR

    @property
    def capabilities(self):
        from uav_tracker.trackers.capabilities import TrackerCapabilities
        # can_widen_search: set_search_factor() + override_search_center() are now
        # wired (AVTrack crops around self._state like SGLATrack), so search-region
        # control levers (widen / relocate / motion_bridge / FC hold_lastgood) apply.
        # can_freeze_template stays False: AVTrack has no online template update.
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
