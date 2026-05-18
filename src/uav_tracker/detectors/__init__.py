# Engineer-owned package namespace.
"""Detector plugins (Phase 6+). Architect owns ``base.py``."""

# Import yolo module so the @DETECTORS.register("yolov8n") decorator fires.
# Ultralytics is NOT imported at this level — only at first detect() call.
from . import yolo  # noqa: F401
