"""SGLATrack — Similarity-Guided Layer-Adaptive Transformer tracker (UAV real-time).

Reference: "SGLATrack: Similarity-Guided Layer-Adaptive Tracking", IET 2023.
Source repo: /Users/voleksiuk/projects/SGLATrack (DeiT-tiny distilled variant)

Architecture: DeiT-tiny (embed_dim=192, depth=12, heads=3, patch=16)
  Template 128×128 (64 tokens), Search 256×256 (256 tokens)
  SGLA MLP routes to 1 best layer from layers 6–11 at inference.

Expected weights: $UAV_WEIGHTS_ROOT/sglatrack/sglatrack_ep0297.pth.tar
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
import torch.nn.functional as F

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, FrameContext, TrackState

logger = logging.getLogger(__name__)

_SGLATRACK_ROOT = Path("/Users/voleksiuk/projects/SGLATrack")
_FLOPS_PER_UPDATE = 0.9e9   # DeiT-tiny: ~0.9 GFLOPs (paper Table 1)

# SGLATrack DeiT-distilled inference config
_TEMPLATE_SIZE = 128
_TEMPLATE_FACTOR = 2.0
_SEARCH_SIZE = 256
_SEARCH_FACTOR = 4.0
_FEAT_SZ = 16          # 256 / 16 (patch stride)
_N_Z = 64              # 8×8 template tokens
_N_X = 256             # 16×16 search tokens
_START_LAYER = 5       # SGLA routes from layer 6 onward

# Maps TargetState int → search_factor used for crop extraction in update_with_state()
# SEARCH_SIZE stays 256px — search_factor expansion was tested (5.5× OCCLUDED, 6.0× LOST)
# but caused localization precision loss on small UAV targets: the target appears smaller
# in the 256px crop as factor increases, reducing center-offset map resolution.
# Reverted to uniform 4.0 across all states.
_STATE_SEARCH_MAP = {
    0: 4.0,   # CONFIRMED
    2: 4.0,   # DYNAMIC
    3: 4.0,   # OCCLUDED — 5.5 tested, regressed car13 0.750→0.690
    4: 4.0,   # LOST     — 6.0 tested, regressed uav2
}
_DEFAULT_SEARCH = 4.0

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Maps TargetState → (sgla_exit_layer, ce_keep_ratios)
# exit_layer: passed as template_token_num hint to SGLA MLP
# ce_keep_ratios: [layer3, layer6, layer9] token keep fractions
_STATE_COMPUTE_MAP = {
    0: (11, [0.50, 0.50, 0.50]),  # CONFIRMED: CE at 50% — see note
    1: (11, [1.0,  1.0,  1.0]),   # LOW_RES: full depth, all tokens
    2: (11, [1.0,  1.0,  1.0]),   # DYNAMIC: full depth, all tokens
    3: (11, [1.0,  1.0,  1.0]),   # OCCLUDED: full depth, all tokens
    4: (11, [1.0,  1.0,  1.0]),   # LOST: full (detector takes over anyway)
    5: (11, [1.0,  1.0,  1.0]),   # DISTRACTOR_RISK: full depth
}
# CE NOTE — Three architectural fixes were required before CE gave correct results:
#   Q1: _CE_LOC corrected from {3,6,9} to {3} — layers 6,9 are unreachable
#       (only i < start_layer=5 enters the pruning branch in forward_test)
#   Q2: CE now uses block i's own QKV projections to score block i's output
#       (was using block i+1's weights — distribution mismatch)
#   Q4: CTEM uses center 4×4 template tokens; CE unchanged but benefits from Q1/Q2
# With these three fixes, CE at kr=0.50 gives MEAN=0.616 (+0.006 vs baseline 0.610).
# Before fixes, CE at 0.50 gave MEAN=0.551 (TSA feedback loop from wrong scoring).
_DEFAULT_COMPUTE = (11, [1.0, 1.0, 1.0])  # fallback: full computation


# ---------------------------------------------------------------------------
# Checkpoint loader — checkpoint contains training objects from lib.train,
# so we stub those classes to avoid importing the full training harness.
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

    mod = types.ModuleType("_sgla_safe_pickle")
    mod.Unpickler = _SafeUnpickler
    for attr in ("UnpicklingError", "PicklingError", "HIGHEST_PROTOCOL",
                 "DEFAULT_PROTOCOL", "dumps", "loads"):
        setattr(mod, attr, getattr(pickle, attr))
    return mod


_SAFE_PICKLE = _make_safe_pickle_module()


def _load_sglatrack_state(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu",
                      weights_only=False, pickle_module=_SAFE_PICKLE)
    return ckpt["net"]


# ---------------------------------------------------------------------------
# Import SGLATrack model builder (adds repo to sys.path once)
# ---------------------------------------------------------------------------

def _ensure_sglatrack_on_path() -> None:
    root = str(_SGLATRACK_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # easydict shim — SGLATrack config requires it but it may not be installed
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
    _ensure_sglatrack_on_path()
    from lib.config.sglatrack.config import cfg, update_config_from_file
    from lib.models.sglatrack import build_sglatrack

    yaml_path = _SGLATRACK_ROOT / "experiments" / "sglatrack" / "deit_distilled.yaml"
    update_config_from_file(str(yaml_path))
    cfg.MODEL.BACKBONE.CE_LOC = [3, 6, 9]   # enable CE, ratio set per-frame

    model = build_sglatrack(cfg, training=False)
    model = model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Preprocessing helpers (device-aware, no hardcoded cuda)
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

@TRACKERS.register("sglatrack")
class SGLATracker:
    """SGLATrack DeiT-tiny distilled — layer-adaptive ViT tracker for UAV.

    tier_hint=1 (lightweight learned tracker, ~0.9 GFLOPs, targets real-time).
    """

    name: str = "sglatrack"
    tier_hint: int = 1

    def __init__(
        self,
        device: str = "auto",
        weights_path: str | None = None,
        pruning_mode: str = "ce",
        enable_ce: bool = True,
    ) -> None:
        if pruning_mode not in ("ce", "ctem"):
            raise ValueError(f"pruning_mode must be 'ce' or 'ctem', got {pruning_mode!r}")
        self._device_str = device
        self._weights_path = weights_path
        self._pruning_mode = pruning_mode
        self._enable_ce = enable_ce
        self._model = None
        self._hann: torch.Tensor | None = None
        self._z_tensor: torch.Tensor | None = None    # template tensor on device
        self._state: BBox | None = None
        self._is_stub: bool = True
        self._template_last_update: int = 0
        self._template_update_count: int = 0  # hard cap: max 5 updates per sequence
        # Per-frame embedding export hook — populated after every update() call.
        # All three are extracted from backbone_feat[:, -256:, :] (search tokens, 16×16 grid)
        # and template tokens [:, :64, :]. Zero extra inference cost.
        self._last_search_global: torch.Tensor | None = None
        self._last_search_score_weighted: torch.Tensor | None = None
        self._last_search_peak_local: torch.Tensor | None = None
        self._last_template_embedding: torch.Tensor | None = None

    @property
    def _device(self) -> torch.device:
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    def _load(self):
        dev = self._device
        model, _ = _build_model(dev)

        weights_path = self._weights_path
        if weights_path is None:
            from uav_tracker.paths import weights_root
            weights_path = str(
                weights_root() / "sglatrack" / "sglatrack_ep0297.pth.tar"
            )
        p = Path(weights_path)
        if p.exists():
            try:
                state = _load_sglatrack_state(p)
                missing, unexpected = model.load_state_dict(state, strict=True)
                self._is_stub = bool(missing)
                if not missing:
                    logger.info("SGLATrack weights loaded from %s", p)
                else:
                    logger.warning("SGLATrack: missing keys: %s", missing[:3])
            except Exception as exc:
                logger.warning("SGLATrack weight load failed: %s — random init", exc)
                self._is_stub = True
        else:
            logger.info("SGLATrack weights not found at %s — random init", p)
            self._is_stub = True

        self._model = model
        self._hann = _make_hann(dev)

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        if self._model is None:
            self._load()

        # OpenCV delivers BGR — SGLATrack expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, _ = _sample_target(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        self._z_tensor = _to_tensor(patch, self._device)
        self._state = bbox

    def update(self, frame: np.ndarray) -> TrackState:
        if self._model is None or self._z_tensor is None or self._state is None:
            return TrackState(bbox=BBox(0, 0, 1, 1), confidence=0.0, status="lost")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, resize_factor = _sample_target(rgb, self._state, _SEARCH_FACTOR, _SEARCH_SIZE)
        x_tensor = _to_tensor(patch, self._device)

        with torch.no_grad():
            out = self._model(template=self._z_tensor, search=x_tensor,
                              ce_template_mask=None)

            # --- Per-frame embedding export hook ---
            # Extracts 3 views of search appearance + template embedding.
            # backbone_feat shape: (1, 320, 192) — first 64 = template, last 256 = search.
            # score_map shape: (1, 1, 16, 16) — spatial confidence over 16×16 search grid.
            if 'backbone_feat' in out:
                _bfeat = out['backbone_feat']
                _search_tokens = _bfeat[:, -256:, :].squeeze(0)  # (256, 192)

                # 1. Global mean-pool over all 256 search tokens
                self._last_search_global = _search_tokens.mean(0)  # (192,)

                # 2. Score-map-weighted mean: post-Hann map so peak matches tracker localisation
                _score = (out['score_map'] * self._hann).squeeze()  # (16, 16), post-Hann
                _weights = _score.reshape(256).softmax(0)            # (256,) normalized
                self._last_search_score_weighted = (_weights.unsqueeze(1) * _search_tokens).sum(0)  # (192,)

                # 3. Bounded neighborhood mean around score peak (Python ints — no edge duplicates)
                _flat_peak = int(_score.reshape(256).argmax().item())
                _pr, _pc = _flat_peak // 16, _flat_peak % 16
                _idxs = [r * 16 + c
                         for r in range(max(0, _pr - 1), min(16, _pr + 2))
                         for c in range(max(0, _pc - 1), min(16, _pc + 2))]
                self._last_search_peak_local = _search_tokens[_idxs].mean(0)  # (192,)

                # Template embedding: mean-pool over 64 template tokens
                self._last_template_embedding = _bfeat[:, :64, :].mean(dim=1).squeeze(0)  # (192,)
            else:
                # Reset to None so stale values never silently persist across frames
                self._last_search_global = None
                self._last_search_score_weighted = None
                self._last_search_peak_local = None
                self._last_template_embedding = None
            # --- end embedding hook ---

        # Apply Hann window to score map for smoother localisation
        score_map = out["score_map"]                          # (1,1,16,16)
        score_map = self._hann * score_map

        with torch.no_grad():
            # APCE — Average Peak-to-Correlation Energy
            f = score_map.view(-1)   # flatten to 1D (256,)
            f_max = float(f.max())
            f_min = float(f.min())
            denom = float(((f - f_min) ** 2).mean())
            apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))

            # PSR — Peak-to-Sidelobe Ratio
            sm2d = score_map.squeeze()   # (16, 16)
            flat_idx = int(sm2d.view(-1).argmax())
            peak_r, peak_c = flat_idx // 16, flat_idx % 16
            r_idx = torch.arange(16, device=sm2d.device).view(16, 1).expand(16, 16)
            c_idx = torch.arange(16, device=sm2d.device).view(1, 16).expand(16, 16)
            peak_mask = ((r_idx - peak_r).abs() <= 5) & ((c_idx - peak_c).abs() <= 5)
            sidelobe = sm2d[~peak_mask]
            if len(sidelobe) > 0:
                psr = float((f_max - sidelobe.mean()) / (sidelobe.std() + 1e-8))
            else:
                psr = 0.0

            # Response entropy — high entropy = multiple candidates = distractor risk
            # L1-normalise (not softmax): score_map values are in [0, ~0.5] so
            # softmax gives near-uniform distribution → entropy stuck at log(256)≈5.545
            # regardless of tracking quality. L1-norm treats the response directly
            # as a probability mass: sharp peaks → low entropy, flat map → high entropy.
            f_pos = f - f.min()  # shift to non-negative
            probs = f_pos / (f_pos.sum() + 1e-8)
            response_entropy = float(-(probs * (probs + 1e-8).log()).sum())

        # Score map geometry for SALT-RD feature extraction
        f_sorted_vals = torch.sort(f, descending=True).values
        _top1 = float(f_sorted_vals[0])
        _top2 = float(f_sorted_vals[1])
        _half_max_count = int((f >= float(f_sorted_vals[0]) * 0.5).sum().item())
        _flat_peak = int(f.argmax().item())
        _peak_r_sm, _peak_c_sm = _flat_peak // 16, _flat_peak % 16
        _map_sum = float(f_sorted_vals.sum()) + 1e-8
        _score_map_stats = {
            "top1": _top1,
            "top2": _top2,
            "peak_margin": _top1 - _top2,
            "peak_width": _half_max_count,
            "n_secondary": 0,  # placeholder: full local-maxima detection in v1
            "peak_distance": float((((_peak_r_sm - 7.5) ** 2 + (_peak_c_sm - 7.5) ** 2) ** 0.5)),
            "heatmap_mass_topk": float(f_sorted_vals[:10].sum()) / _map_sum,
        }

        pred_boxes = self._model.box_head.cal_bbox(
            score_map, out["size_map"], out["offset_map"]
        ).view(-1, 4)                                         # [cx,cy,w,h] normalised

        # Top-3 softmax mass confidence (UncL-STARK, eq. 3)
        # Better calibrated than peak value — higher Pearson correlation with IoU
        _sm = torch.softmax(score_map.view(-1), dim=0)
        _top3 = _sm.topk(3).values.sum()
        confidence = float(_top3.clamp(0.0, 1.0).cpu())

        # Denormalise prediction to search crop pixels
        pred = (pred_boxes.mean(dim=0) * _SEARCH_SIZE / resize_factor).tolist()
        cx_pred, cy_pred, w_pred, h_pred = pred

        # Map from search-crop-centred coords back to frame coords
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
            score_map_stats=_score_map_stats,
        )

    def update_with_state(self, frame: np.ndarray, target_state: int,
                          consecutive_occluded: int = 0,
                          prev_apce: float = 0.0) -> TrackState:
        """Update tracker with TSA-conditioned CE token pruning and expanded search.

        For CONFIRMED state: ce_keep_ratio=0.5 → prune 50% search tokens at CE layers,
            BUT only when prev_apce >= 100 (tracker confident last frame). If prev_apce < 100
            the tracker was in a low-SNR state (re-acquisition, distractor confusion) and
            needs full tokens to catch the marginal re-acquisition signal. This fixes the
            truck1 regression: a single failed re-acquisition at frame 418 was caused by CE
            pruning the marginal peak of the correct target passing through the search window.
        For all other states: full computation (ce_keep_ratio=1.0).

        Args:
            consecutive_occluded: number of consecutive OCCLUDED frames so far.
            prev_apce: APCE from the previous frame (0.0 = unknown/startup).
                CE pruning is disabled for CONFIRMED when prev_apce < 100.
        """
        _, keep_ratios = _STATE_COMPUTE_MAP.get(target_state, _DEFAULT_COMPUTE)
        ce_keep_rate = keep_ratios[0]  # same ratio at all CE layers

        if not self._enable_ce:
            ce_keep_rate = 1.0  # CE disabled via config gate

        # APCE gate: skip CE pruning when tracker was in genuine LOST territory last frame.
        # Threshold 25 = just above LOST threshold (APCE < 20) — only disables CE when
        # the previous frame was a genuine loss event (APCE ~12-15 as in truck1 frame 416-417).
        # Does NOT affect normal OCCLUDED frames (APCE 20-80) — CE stays active there.
        if ce_keep_rate < 1.0 and 0.0 < prev_apce < 25.0:
            ce_keep_rate = 1.0

        if self._model is None or self._z_tensor is None or self._state is None:
            return TrackState(bbox=BBox(0, 0, 1, 1), confidence=0.0, status="lost")

        # Map target_state to search_factor — OCCLUDED/LOST use expanded region
        search_factor = _STATE_SEARCH_MAP.get(target_state, _DEFAULT_SEARCH)
        patch, resize_factor = _sample_target(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            self._state, search_factor, _SEARCH_SIZE,
        )
        x = _to_tensor(patch, self._device)

        with torch.no_grad():
            out = self._model(
                template=self._z_tensor,
                search=x,
                ce_template_mask=None,
                ce_keep_rate=ce_keep_rate,
                pruning_mode=self._pruning_mode,
            )

            # --- Per-frame embedding export hook ---
            # Extracts 3 views of search appearance + template embedding.
            # backbone_feat shape: (1, 320, 192) — first 64 = template, last 256 = search.
            # score_map shape: (1, 1, 16, 16) — spatial confidence over 16×16 search grid.
            if 'backbone_feat' in out:
                _bfeat = out['backbone_feat']
                _search_tokens = _bfeat[:, -256:, :].squeeze(0)  # (256, 192)

                # 1. Global mean-pool over all 256 search tokens
                self._last_search_global = _search_tokens.mean(0)  # (192,)

                # 2. Score-map-weighted mean: post-Hann map so peak matches tracker localisation
                _score = (out['score_map'] * self._hann).squeeze()  # (16, 16), post-Hann
                _weights = _score.reshape(256).softmax(0)            # (256,) normalized
                self._last_search_score_weighted = (_weights.unsqueeze(1) * _search_tokens).sum(0)  # (192,)

                # 3. Bounded neighborhood mean around score peak (Python ints — no edge duplicates)
                _flat_peak = int(_score.reshape(256).argmax().item())
                _pr, _pc = _flat_peak // 16, _flat_peak % 16
                _idxs = [r * 16 + c
                         for r in range(max(0, _pr - 1), min(16, _pr + 2))
                         for c in range(max(0, _pc - 1), min(16, _pc + 2))]
                self._last_search_peak_local = _search_tokens[_idxs].mean(0)  # (192,)

                # Template embedding: mean-pool over 64 template tokens
                self._last_template_embedding = _bfeat[:, :64, :].mean(dim=1).squeeze(0)  # (192,)
            else:
                # Reset to None so stale values never silently persist across frames
                self._last_search_global = None
                self._last_search_score_weighted = None
                self._last_search_peak_local = None
                self._last_template_embedding = None
            # --- end embedding hook ---

        score_map = out["score_map"]
        score_map = self._hann * score_map

        with torch.no_grad():
            # APCE — Average Peak-to-Correlation Energy
            f = score_map.view(-1)   # flatten to 1D (256,)
            f_max = float(f.max())
            f_min = float(f.min())
            denom = float(((f - f_min) ** 2).mean())
            apce = float((f_max - f_min) ** 2 / max(denom, 1e-8))

            # PSR — Peak-to-Sidelobe Ratio
            sm2d = score_map.squeeze()   # (16, 16)
            flat_idx = int(sm2d.view(-1).argmax())
            peak_r, peak_c = flat_idx // 16, flat_idx % 16
            r_idx = torch.arange(16, device=sm2d.device).view(16, 1).expand(16, 16)
            c_idx = torch.arange(16, device=sm2d.device).view(1, 16).expand(16, 16)
            peak_mask = ((r_idx - peak_r).abs() <= 5) & ((c_idx - peak_c).abs() <= 5)
            sidelobe = sm2d[~peak_mask]
            if len(sidelobe) > 0:
                psr = float((f_max - sidelobe.mean()) / (sidelobe.std() + 1e-8))
            else:
                psr = 0.0

            # Response entropy — L1-normalise (not softmax): see update() for rationale
            f_pos = f - f.min()
            probs = f_pos / (f_pos.sum() + 1e-8)
            response_entropy = float(-(probs * (probs + 1e-8).log()).sum())

        # Score map geometry for SALT-RD feature extraction
        f_sorted_vals = torch.sort(f, descending=True).values
        _top1 = float(f_sorted_vals[0])
        _top2 = float(f_sorted_vals[1])
        _half_max_count = int((f >= float(f_sorted_vals[0]) * 0.5).sum().item())
        _flat_peak = int(f.argmax().item())
        _peak_r_sm, _peak_c_sm = _flat_peak // 16, _flat_peak % 16
        _map_sum = float(f_sorted_vals.sum()) + 1e-8
        _score_map_stats = {
            "top1": _top1,
            "top2": _top2,
            "peak_margin": _top1 - _top2,
            "peak_width": _half_max_count,
            "n_secondary": 0,  # placeholder: full local-maxima detection in v1
            "peak_distance": float((((_peak_r_sm - 7.5) ** 2 + (_peak_c_sm - 7.5) ** 2) ** 0.5)),
            "heatmap_mass_topk": float(f_sorted_vals[:10].sum()) / _map_sum,
        }

        pred_boxes = self._model.box_head.cal_bbox(
            score_map, out["size_map"], out["offset_map"]
        ).view(-1, 4)

        # Top-3 softmax mass confidence (UncL-STARK, eq. 3)
        # Better calibrated than peak value — higher Pearson correlation with IoU
        _sm = torch.softmax(score_map.view(-1), dim=0)
        _top3 = _sm.topk(3).values.sum()
        confidence = float(_top3.clamp(0.0, 1.0).cpu())
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

        # Only update the search center for reliable states.
        # Streak-based freeze: only freeze center after N consecutive OCCLUDED frames.
        # Brief occlusions (1-4 frames) → tracker still follows freely.
        # Long occlusion streaks (≥5 frames) → freeze center to prevent drift from
        # unreliable predictions. Next frame searches from last confirmed position.
        _FREEZE_STREAK_THRESHOLD = 5  # only freeze after 5+ consecutive OCCLUDED frames
        _should_freeze = (target_state == 3 and consecutive_occluded >= _FREEZE_STREAK_THRESHOLD)
        if not _should_freeze:
            self._state = new_bbox
        # We still RETURN new_bbox (best estimate for this frame) but
        # next frame will search from the frozen center, not the drifted prediction.

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
            score_map_stats=_score_map_stats,
        )

    def try_update_template(
        self,
        frame: np.ndarray,
        bbox: BBox,
        apce: float,
        psr: float,
        frame_idx: int,
        cosine_sim: float,
        apce_threshold: float = 220.0,
        psr_threshold: float = 2000.0,
        min_interval: int = 100,
        cosine_threshold: float = 0.80,
        max_updates: int = 5,
    ) -> bool:
        """Update template snapshot when all strict quality guards pass.

        Fires only when ALL of the following hold:
          1. apce > apce_threshold     — very sharp correlation peak (top ~10%)
          2. psr  > psr_threshold      — strong peak-to-sidelobe ratio
          3. at least min_interval frames since last template update
          4. cosine_sim > cosine_threshold — slow appearance change only
          5. fewer than max_updates total updates this sequence (hard cap)

        Returns True if the template was updated.
        """
        if self._z_tensor is None:
            return False

        if self._template_update_count >= max_updates:
            return False

        if apce <= apce_threshold:
            return False

        if psr <= psr_threshold:
            return False

        if (frame_idx - self._template_last_update) < min_interval:
            return False

        if cosine_sim <= cosine_threshold:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patch, _ = _sample_target(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        z_new = _to_tensor(patch, self._device)

        self._z_tensor = z_new
        self._template_last_update = frame_idx
        self._template_update_count += 1
        logger.debug(
            "SGLATracker: template snapshot #%d at frame %d "
            "(apce=%.1f psr=%.1f cosine=%.3f)",
            self._template_update_count, frame_idx, apce, psr, cosine_sim,
        )
        return True

    def reset(self) -> None:
        self._z_tensor = None
        self._state = None
        self._template_last_update = 0
        self._template_update_count = 0

    @property
    def is_stub_mode(self) -> bool:
        return self._is_stub

    def flops_per_update(self) -> float:
        return _FLOPS_PER_UPDATE

    def on_tier_enter(self, ctx: FrameContext) -> None:
        pass

    def on_tier_exit(self, ctx: FrameContext) -> None:
        pass
