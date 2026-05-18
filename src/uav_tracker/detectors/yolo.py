"""YOLOv8-n detector plugin for the UAV Entropy-Guided Tracker.

Phase 6 Integration
-------------------
This module registers ``YOLOv8Detector`` under the key ``"yolov8n"`` in the
``DETECTORS`` registry.  The multi-tier scheduler (Engineer B) instantiates it
as the optional tier-2 re-detection stage when the tracker reports ``"lost"``
or ``"uncertain"`` status on UAV123/OTB sequences.

Ultralytics-missing behaviour
------------------------------
``ultralytics`` is **NOT** imported at module top-level.  The class registers
cleanly so that ``uav-tracker list-plugins`` works on a bare install.  The
first call to ``.detect()`` triggers the lazy import; if the package is absent
a ``RuntimeError`` with a ``uv pip install ultralytics`` hint is raised.

Weight auto-download
---------------------
Ultralytics auto-fetches ``yolov8n.pt`` into its standard cache directory
(``~/.config/Ultralytics/`` or the path set via the ``YOLO_CONFIG_DIR``
env-var) on the first ``YOLO(model_name)`` call.  ``download_weights.py``
can optionally pre-fetch and copy the weight to
``$UAV_WEIGHTS_ROOT/yolov8n/yolov8n.pt`` for offline / manifest-verified
workflows.

AGPL-3.0 Note
-------------
The ``ultralytics`` package (and its bundled YOLOv8 weights) are distributed
under the AGPL-3.0 license.  We use weights + inference only (not training
code), but AGPL is viral: if this project is ever distributed as a network
service the AGPL terms apply.  The Architect should review license compat
(see ``docs/model_cards/yolov8n.md``).
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from ..registry import DETECTORS
from ..types import BBox, Detection

if TYPE_CHECKING:  # pragma: no cover — type-checker only, never executed at runtime
    from ultralytics import YOLO as _YOLO  # noqa: F401

_log = logging.getLogger(__name__)

# Static FLOPs estimate for YOLOv8n on a single forward pass (8.7 GFLOPs).
# Used as fallback when ``thop`` is unavailable.
_YOLOV8N_FLOPS = 8.7e9


def _lazy_import_ultralytics() -> type:
    """Import and return the ``ultralytics.YOLO`` class, raising a helpful error
    if the package is not installed.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
        return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The 'ultralytics' package is required for YOLOv8 detection but is not "
            "installed.  Install it with:\n\n    uv pip install ultralytics\n"
        ) from exc


@DETECTORS.register("yolov8n")
class YOLOv8Detector:
    """YOLOv8-n full-frame (or hint-cropped) object detector.

    Parameters
    ----------
    model_name:
        Ultralytics model identifier or path to a local ``.pt`` file.
        Defaults to ``"yolov8n.pt"`` (auto-downloaded by Ultralytics on first
        use into the Ultralytics cache directory).
    device:
        PyTorch device string: ``"cpu"``, ``"cuda"``, ``"mps"``, etc.
    conf_threshold:
        Minimum detection confidence to retain (maps to Ultralytics ``conf``).
    iou_threshold:
        NMS IoU threshold (maps to Ultralytics ``iou``).
    classes:
        COCO class IDs to restrict detections to.  ``None`` keeps all 80
        COCO classes.
    """

    #: Registry key — used by the scheduler to look up this plugin.
    name: str = "yolov8n"

    #: Phase 6 scheduler tier hint (tier-0 = fast, tier-2 = heavy detection).
    tier_hint: int = 2

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._classes = classes

        # Lazy-initialised on first detect() call.
        self._model: _YOLO | None = None
        self._cached_flops: float | None = None
        self._warned_download: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> _YOLO:
        """Initialise the YOLO model on the first call, downloading if needed."""
        if self._model is not None:
            return self._model

        YOLO = _lazy_import_ultralytics()

        if not self._warned_download:
            _log.warning(
                "YOLOv8Detector: loading model '%s' (Ultralytics will auto-download "
                "to its cache if not present locally).",
                self._model_name,
            )
            self._warned_download = True

        self._model = YOLO(self._model_name)
        return self._model

    def _run_inference(
        self,
        img: np.ndarray,
    ) -> list[Detection]:
        """Run YOLO inference on ``img`` (H×W×3 BGR or RGB uint8) and convert
        to ``Detection`` instances."""
        model = self._ensure_model()
        results = model.predict(
            source=img,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            # boxes.xyxy: Tensor[N, 4]  (x1, y1, x2, y2)
            # boxes.conf: Tensor[N]
            # boxes.cls:  Tensor[N]
            for xyxy, conf, cls in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
            ):
                x1, y1, x2, y2 = xyxy
                detections.append(
                    Detection(
                        bbox=BBox(x=float(x1), y=float(y1), w=float(x2 - x1), h=float(y2 - y1)),
                        score=float(conf),
                        class_id=int(cls),
                    )
                )
        return detections

    @staticmethod
    def _expand_bbox(bbox: BBox, frame_h: int, frame_w: int, factor: float = 3.0) -> BBox:
        """Return a ``factor``-x enlarged version of ``bbox`` clamped to the frame."""
        cx = bbox.x + bbox.w / 2.0
        cy = bbox.y + bbox.h / 2.0
        new_w = min(bbox.w * factor, float(frame_w))
        new_h = min(bbox.h * factor, float(frame_h))
        x1 = max(cx - new_w / 2.0, 0.0)
        y1 = max(cy - new_h / 2.0, 0.0)
        x2 = min(x1 + new_w, float(frame_w))
        y2 = min(y1 + new_h, float(frame_h))
        return BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        hint: BBox | None = None,
    ) -> list[Detection]:
        """Run detection on ``frame``, optionally using ``hint`` for a focused crop.

        Parameters
        ----------
        frame:
            H×W×3 image array (uint8, BGR or RGB — consistent with OpenCV input).
        hint:
            Last known tracker bounding box.  If provided, detection is first
            attempted on a 3× expanded crop around the hint region for speed.
            If the crop yields no detections the method falls through to a
            full-frame pass.  Coordinates in returned ``Detection.bbox`` are
            always in full-frame pixel space.

        Returns
        -------
        list[Detection]
            Zero or more detection hypotheses, each with a score in ``[0, 1]``.
        """
        frame_h, frame_w = frame.shape[:2]

        if hint is not None:
            crop_bbox = self._expand_bbox(hint, frame_h, frame_w, factor=3.0)
            x1 = int(crop_bbox.x)
            y1 = int(crop_bbox.y)
            x2 = int(crop_bbox.x + crop_bbox.w)
            y2 = int(crop_bbox.y + crop_bbox.h)
            crop = frame[y1:y2, x1:x2]
            crop_dets = self._run_inference(crop)
            if crop_dets:
                # Translate back to full-frame coordinates.
                full_dets = [
                    Detection(
                        bbox=BBox(
                            x=d.bbox.x + x1,
                            y=d.bbox.y + y1,
                            w=d.bbox.w,
                            h=d.bbox.h,
                        ),
                        score=d.score,
                        class_id=d.class_id,
                    )
                    for d in crop_dets
                ]
                return full_dets
            # No detections in crop — fall through to full-frame.
            _log.debug(
                "YOLOv8Detector: no detections in hint crop; falling through to full frame."
            )

        return self._run_inference(frame)

    def flops_per_call(self) -> float:
        """Return estimated FLOPs per ``detect()`` call.

        Returns the static 8.7 GFLOPs figure for YOLOv8n.  If ``thop`` is
        installed and the model has been loaded, the cached profile from the
        first call is returned instead.
        """
        if self._cached_flops is not None:
            return self._cached_flops

        # Optional: profile with thop on the loaded model.
        if self._model is not None:
            try:
                import thop  # noqa: PLC0415

                dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
                import torch  # noqa: PLC0415

                x = torch.from_numpy(dummy)
                macs, _ = thop.profile(self._model.model, inputs=(x,), verbose=False)
                self._cached_flops = float(macs) * 2  # thop returns MACs; 1 MAC ≈ 2 FLOPs
                return self._cached_flops
            except Exception:  # noqa: BLE001 — thop optional, any failure is fine
                pass

        return _YOLOV8N_FLOPS


__all__ = ["YOLOv8Detector"]
