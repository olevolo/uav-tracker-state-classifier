"""LEAF-YOLO detector — UAV small-object detection for SALT recovery.

Trained on VisDrone2019-DET (UAV aerial imagery, tiny objects).
AP50=48.3% on VisDrone; 32 FPS on Jetson AGX Xavier TensorRT fp16.

Variants:
  leaf-sizes/weights/best.pt  — standard (4.28M params, 28.2% AP, 48.3% AP50)
  leaf-sizen/weights/best.pt  — nano    (1.2M  params, 21.9% AP, 39.7% AP50)

Expected weights: /Users/voleksiuk/projects/LEAF-YOLO/cfg/LEAF-YOLO/<variant>/weights/best.pt
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

_LEAFYOLO_ROOT = Path("/Users/voleksiuk/projects/LEAF-YOLO")
_WEIGHTS = {
    "leaf_yolo":   _LEAFYOLO_ROOT / "cfg/LEAF-YOLO/leaf-sizes/weights/best.pt",
    "leaf_yolo_n": _LEAFYOLO_ROOT / "cfg/LEAF-YOLO/leaf-sizen/weights/best.pt",
}
_INPUT_SIZE = 640


def _ensure_leafyolo_on_path() -> None:
    root = str(_LEAFYOLO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import types, importlib.util
    # Stub seaborn — only used for plot utilities, not needed for inference
    if "seaborn" not in sys.modules:
        m = types.ModuleType("seaborn")
        m.__spec__ = importlib.util.spec_from_loader("seaborn", loader=None)
        m.color_palette = lambda *a, **k: [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        sys.modules["seaborn"] = m

    # Patch torch.load to use weights_only=False — LEAF-YOLO checkpoints
    # contain numpy objects not allowed by PyTorch 2.6 default
    import torch
    if not getattr(torch.load, "_leafyolo_patched", False):
        _orig = torch.load
        def _patched(*a, **k):
            k.setdefault("weights_only", False)
            return _orig(*a, **k)
        _patched._leafyolo_patched = True
        torch.load = _patched


class _Detection:
    __slots__ = ("bbox", "confidence", "class_id")

    def __init__(self, bbox: BBox, confidence: float, class_id: int) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id


@DETECTORS.register("leaf_yolo")
class LeafYOLODetector:
    """LEAF-YOLO detector — VisDrone-trained, optimised for tiny UAV targets.

    Used for LOST state recovery in the SALT pipeline.
    """

    name: str = "leaf_yolo"

    def __init__(
        self,
        variant: str = "leaf_yolo",
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = _INPUT_SIZE,
    ) -> None:
        self._variant = variant
        self._device_str = device
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._input_size = input_size
        self._model: Any | None = None

    @property
    def _device(self) -> str:
        if self._device_str == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "0"
                # LEAF-YOLO uses YOLOv7-style select_device — 'cpu' is universal
                return "cpu"
            except ImportError:
                return "cpu"
        return self._device_str if self._device_str not in ("mps", "auto") else "cpu"

    def _load(self) -> None:
        _ensure_leafyolo_on_path()

        import torch
        try:
            from models.experimental import attempt_load
            from utils.torch_utils import select_device
        except ImportError as e:
            raise ImportError(
                f"LEAF-YOLO not found at {_LEAFYOLO_ROOT}. "
                "Check /Users/voleksiuk/projects/LEAF-YOLO exists."
            ) from e

        weights_path = _WEIGHTS.get(self._variant)
        if weights_path is None or not weights_path.exists():
            raise FileNotFoundError(
                f"LEAF-YOLO weights not found: {weights_path}. "
                f"Available: {list(_WEIGHTS.keys())}"
            )

        device = select_device(self._device)
        model = attempt_load(str(weights_path), map_location=device)
        model = model.half() if device.type != "cpu" else model.float()
        model.eval()

        self._model = model
        self._torch_device = device
        self._use_half = device.type != "cpu"
        logger.info("LEAF-YOLO (%s) loaded from %s (device=%s)",
                    self._variant, weights_path, device)

    def detect(
        self,
        frame: np.ndarray,
        hint_bbox: BBox | None = None,
    ) -> list[_Detection]:
        """Detect objects in frame. Returns list sorted by score."""
        if self._model is None:
            self._load()

        import torch
        from utils.general import non_max_suppression

        h, w = frame.shape[:2]
        sz = self._input_size

        # BGR → RGB, resize, normalise
        rgb = frame[:, :, ::-1].copy()
        resized = cv2.resize(rgb, (sz, sz))
        img = torch.from_numpy(
            resized.transpose(2, 0, 1).copy()
        ).float().to(self._torch_device) / 255.0
        if self._use_half:
            img = img.half()
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self._model(img)[0]

        dets = non_max_suppression(pred, conf_thres=self._conf, iou_thres=self._iou)

        results: list[_Detection] = []
        if dets[0] is not None and len(dets[0]):
            # Scale back to original frame coordinates
            sx, sy = w / sz, h / sz
            for *xyxy, conf, cls in dets[0].cpu().numpy():
                x1, y1, x2, y2 = xyxy
                bbox = BBox(
                    x=float(x1 * sx),
                    y=float(y1 * sy),
                    w=float((x2 - x1) * sx),
                    h=float((y2 - y1) * sy),
                )
                results.append(_Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls),
                ))

        results.sort(key=lambda d: d.confidence, reverse=True)
        return results

    def reset(self) -> None:
        pass


@DETECTORS.register("leaf_yolo_n")
class LeafYOLONanoDetector(LeafYOLODetector):
    """LEAF-YOLO nano — faster, smaller (1.2M params, 39.7% AP50 VisDrone)."""

    name: str = "leaf_yolo_n"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(variant="leaf_yolo_n", **kwargs)
