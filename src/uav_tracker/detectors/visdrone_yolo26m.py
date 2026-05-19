"""VisDrone YOLO26m detector — fine-tuned for UAV small-object detection.

Source: huggingface.co/kailunw/visdrone-yolo26m
mAP@0.5: 55.05% on VisDrone2019-DET
FPS: ~43 (23ms/frame)
Classes: pedestrian, people, bicycle, car, van, truck, tricycle,
         awning-tricycle, bus, motor

Expected weights: ~/projects/visdrone-yolo26m/best.pt
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from uav_tracker.registry import DETECTORS
from uav_tracker.types import BBox

logger = logging.getLogger(__name__)

_WEIGHTS = Path.home() / "projects" / "visdrone-yolo26m" / "best.pt"


class _Detection:
    __slots__ = ("bbox", "confidence", "class_id")

    def __init__(self, bbox: BBox, confidence: float, class_id: int) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id


@DETECTORS.register("yolo26m_visdrone")
class VisDroneYOLO26mDetector:
    """YOLO26m fine-tuned on VisDrone — best UAV detector for SALT recovery.

    55.05% mAP@0.5 on VisDrone2019-DET; ultralytics interface.
    """

    name: str = "yolo26m_visdrone"

    def __init__(
        self,
        weights: str | None = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        self._weights = Path(weights) if weights else _WEIGHTS
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
                    return "cuda"
                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return self._device_str

    def _load(self) -> None:
        if not self._weights.exists():
            raise FileNotFoundError(
                f"VisDrone YOLO26m weights not found at {self._weights}. "
                "Download: .venv/bin/python -c \"from huggingface_hub import "
                "snapshot_download; snapshot_download('kailunw/visdrone-yolo26m', "
                "local_dir='~/projects/visdrone-yolo26m')\""
            )
        from ultralytics import YOLO
        self._model = YOLO(str(self._weights))
        self._model.to(self._device)
        logger.info("VisDrone YOLO26m loaded from %s (device=%s)",
                    self._weights, self._device)

    def detect(
        self,
        frame: np.ndarray,
        hint_bbox: BBox | None = None,
    ) -> list[_Detection]:
        """Detect objects. Returns list sorted by confidence desc."""
        if self._model is None:
            self._load()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self._model(
                frame,
                imgsz=self._input_size,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
                device=self._device,
            )

        dets: list[_Detection] = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls = int(box.cls[0].cpu())
                dets.append(_Detection(
                    bbox=BBox(x=float(x1), y=float(y1),
                              w=float(x2 - x1), h=float(y2 - y1)),
                    confidence=conf,
                    class_id=cls,
                ))

        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets

    def warmup(self) -> None:
        """Run one dummy forward pass to pay JIT/CUDA-graph init cost at load time.

        Ultralytics YOLO triggers TorchScript compilation and (on CUDA) CUDA-graph
        capture on the very first inference call.  Calling warmup() after _load()
        moves this one-time cost to from_config() so the first real recovery call
        is not penalised.
        """
        if self._model is None:
            try:
                self._load()
            except FileNotFoundError:
                # Weights not present — nothing to warm up
                return

        import warnings
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._model(
                    dummy,
                    imgsz=self._input_size,
                    conf=self._conf,
                    iou=self._iou,
                    verbose=False,
                    device=self._device,
                )
            except Exception:
                pass  # warmup failure is non-fatal
        logger.info("VisDrone YOLO26m warmup done")

    def reset(self) -> None:
        pass
