"""RT-DETRv2 detector plugin for SALT recovery state.

Uses RT-DETRv2-S (ResNet-18vd backbone) from ~/projects/RT-DETR.
217 FPS on GPU, 48.1% mAP COCO — replaces YOLOv8n for LOST state recovery.

Expected weights: $UAV_WEIGHTS_ROOT/rtdetr/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
Config:          ~/projects/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uav_tracker.registry import DETECTORS
from uav_tracker.types import BBox

logger = logging.getLogger(__name__)

_RTDETR_ROOT = Path("/Users/voleksiuk/projects/RT-DETR/rtdetrv2_pytorch")
_WEIGHTS_NAME = "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
_CONFIG_NAME  = "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"
_INPUT_SIZE   = 640


def _ensure_rtdetr_on_path() -> None:
    root = str(_RTDETR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import types

    _W = type("SummaryWriter", (), {"__init__": lambda s, *a, **k: None,
                                     "add_scalar": lambda s, *a, **k: None})

    def _make_stub(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__path__ = []
        m.__package__ = name
        m.__file__ = f"/stub/{name}.py"
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    for mod_name, attrs in {
        "tensorboard":             {"SummaryWriter": _W},
        "tensorboardX":            {"SummaryWriter": _W},
        "torch.utils.tensorboard": {"SummaryWriter": _W},
    }.items():
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub(mod_name, **attrs)
        else:
            for k, v in attrs.items():
                if not hasattr(sys.modules[mod_name], k):
                    setattr(sys.modules[mod_name], k, v)


@DETECTORS.register("rtdetrv2_s")
class RTDETRv2Detector:
    """RT-DETRv2-S detector — used for LOST state target recovery.

    Detects objects in the full frame and returns bboxes near the
    last known target location.
    """

    name: str = "rtdetrv2_s"

    def __init__(
        self,
        device: str = "auto",
        conf_threshold: float = 0.3,
        target_classes: list[int] | None = None,
    ) -> None:
        self._device_str = device
        self._conf_threshold = conf_threshold
        self._target_classes = target_classes  # None = all classes
        self._model: Any | None = None

    @property
    def _device(self) -> "torch.device":
        import torch
        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    def _load(self) -> None:
        import torch
        import torchvision.transforms as T
        from PIL import Image as _PILImage

        _ensure_rtdetr_on_path()

        try:
            from src.core import YAMLConfig
        except ImportError as e:
            raise ImportError(
                f"RT-DETRv2 not found at {_RTDETR_ROOT}. "
                "Clone https://github.com/lyuwenyu/RT-DETR into ~/projects/RT-DETR"
            ) from e

        from uav_tracker.paths import weights_root
        weights_path = weights_root() / "rtdetr" / _WEIGHTS_NAME
        config_path  = _RTDETR_ROOT / _CONFIG_NAME

        if not weights_path.exists():
            raise FileNotFoundError(
                f"RT-DETRv2 weights not found at {weights_path}. "
                f"Download from https://github.com/lyuwenyu/storage/releases/download/v0.2/{_WEIGHTS_NAME}"
            )

        cfg = YAMLConfig(str(config_path), resume=str(weights_path))
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        state = ckpt.get("ema", ckpt.get("model", ckpt))
        if "module" in state:
            state = state["module"]
        cfg.model.load_state_dict(state)

        class _DeployModel(torch.nn.Module):
            def __init__(self, model, postprocessor):
                super().__init__()
                self.model = model.deploy()
                self.postprocessor = postprocessor.deploy()

            def forward(self, images, orig_sizes):
                return self.postprocessor(self.model(images), orig_sizes)

        self._deploy = _DeployModel(cfg.model, cfg.postprocessor).eval().to(self._device)
        self._transform = T.Compose([
            T.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            T.ToTensor(),
        ])
        self._PIL = _PILImage
        logger.info("RTDETRv2-S loaded from %s (device=%s)", weights_path, self._device)

    def detect(
        self,
        frame: np.ndarray,
        hint_bbox: BBox | None = None,
    ) -> list[Any]:
        """Detect objects in frame. Returns list of Detection(bbox, confidence)."""
        import torch

        if self._model is None and not hasattr(self, '_deploy'):
            self._load()

        h, w = frame.shape[:2]

        # BGR → RGB → PIL
        rgb = frame[:, :, ::-1].copy()
        pil = self._PIL.fromarray(rgb.astype("uint8"))
        im_tensor = self._transform(pil).unsqueeze(0).to(self._device)
        orig_size  = torch.tensor([[w, h]], device=self._device)

        with torch.no_grad():
            labels, boxes, scores = self._deploy(im_tensor, orig_size)

        labels = labels[0].cpu().numpy()
        boxes  = boxes[0].cpu().numpy()   # [N, 4] xyxy
        scores = scores[0].cpu().numpy()

        results = []
        for lbl, box, score in zip(labels, boxes, scores):
            if float(score) < self._conf_threshold:
                continue
            if self._target_classes is not None and int(lbl) not in self._target_classes:
                continue
            x1, y1, x2, y2 = box
            bbox = BBox(x=float(x1), y=float(y1),
                        w=float(x2 - x1), h=float(y2 - y1))
            results.append(_Detection(bbox=bbox, confidence=float(score), class_id=int(lbl)))

        # Sort by proximity to hint if provided
        if hint_bbox is not None and results:
            hcx = hint_bbox.x + hint_bbox.w / 2
            hcy = hint_bbox.y + hint_bbox.h / 2
            results.sort(key=lambda d: (
                (d.bbox.x + d.bbox.w/2 - hcx)**2 +
                (d.bbox.y + d.bbox.h/2 - hcy)**2
            ))

        return results

    def reset(self) -> None:
        pass


class _Detection:
    __slots__ = ("bbox", "confidence", "class_id")

    def __init__(self, bbox: BBox, confidence: float, class_id: int) -> None:
        self.bbox       = bbox
        self.confidence = confidence
        self.class_id   = class_id
