"""Unit tests for YOLOv8Detector.

These tests require the ``ultralytics`` package.  They are skipped automatically
when ``ultralytics`` is not installed so that the base CI (no GPU, no ultralytics)
stays green.  Run with ultralytics present to exercise the actual model path.

    pytest -v tests/unit/test_yolov8_detector.py

The slow test (real .detect() on a dummy frame) is marked with @pytest.mark.slow
so it can be excluded with ``pytest -m 'not slow'`` when the Ultralytics
auto-download is undesirable.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip this entire module cleanly if ultralytics is not installed.
# pytest.importorskip returns the module on success; on ImportError it calls
# pytest.skip() which raises Skipped (a subclass of BaseException — not caught
# by the test runner as a failure).
ultralytics = pytest.importorskip(
    "ultralytics",
    reason="ultralytics not installed — run `uv pip install ultralytics` to enable YOLOv8 tests",
)


# ---------------------------------------------------------------------------
# Imports that are safe once ultralytics is confirmed present
# ---------------------------------------------------------------------------

from uav_tracker.registry import DETECTORS  # noqa: E402
from uav_tracker.types import Detection  # noqa: E402

# Trigger plugin registration (normally done by uav_tracker.__init__ but
# import order in tests can vary).
import uav_tracker.detectors.yolo  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestYOLOv8DetectorRegistration:
    def test_yolov8n_in_registry(self) -> None:
        """'yolov8n' must appear in DETECTORS after importing the plugin module."""
        assert "yolov8n" in DETECTORS, (
            f"'yolov8n' not found in DETECTORS. Registered detectors: {DETECTORS.names()}"
        )

    def test_registry_names_contains_yolov8n(self) -> None:
        assert "yolov8n" in DETECTORS.names()


class TestYOLOv8DetectorInstantiation:
    def test_default_instantiation(self) -> None:
        """Constructing YOLOv8Detector with default args must not crash."""
        from uav_tracker.detectors.yolo import YOLOv8Detector

        det = YOLOv8Detector()
        assert det.name == "yolov8n"
        assert det.tier_hint == 2

    def test_custom_args_instantiation(self) -> None:
        from uav_tracker.detectors.yolo import YOLOv8Detector

        det = YOLOv8Detector(
            model_name="yolov8n.pt",
            device="cpu",
            conf_threshold=0.5,
            iou_threshold=0.6,
            classes=[0, 2],
        )
        assert det._conf_threshold == 0.5
        assert det._iou_threshold == 0.6
        assert det._classes == [0, 2]

    def test_build_via_registry(self) -> None:
        """DETECTORS.build('yolov8n') must return a YOLOv8Detector instance."""
        from uav_tracker.detectors.yolo import YOLOv8Detector

        det = DETECTORS.build("yolov8n")
        assert isinstance(det, YOLOv8Detector)

    def test_flops_per_call_without_model_loaded(self) -> None:
        """flops_per_call() must return the static 8.7 GFLOPs fallback before
        the model is loaded (i.e. before detect() is ever called)."""
        from uav_tracker.detectors.yolo import YOLOv8Detector, _YOLOV8N_FLOPS

        det = YOLOv8Detector()
        assert det.flops_per_call() == pytest.approx(_YOLOV8N_FLOPS)


@pytest.mark.slow
class TestYOLOv8DetectorInference:
    """Tests that actually load the YOLOv8n model (triggers ultralytics download
    on first run).  Marked slow so CI can skip with ``-m 'not slow'``."""

    @pytest.fixture(scope="class")
    def detector(self):
        from uav_tracker.detectors.yolo import YOLOv8Detector

        return YOLOv8Detector(device="cpu", conf_threshold=0.01)  # low conf to catch any box

    def test_detect_returns_list(self, detector) -> None:
        """detect() on a small noise frame must return a list (possibly empty)."""
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)

    def test_detect_elements_are_detection_instances(self, detector) -> None:
        """Every element returned by detect() must be a Detection instance."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        result = detector.detect(frame)
        for item in result:
            assert isinstance(item, Detection), (
                f"Expected Detection, got {type(item)}: {item!r}"
            )

    def test_detect_scores_in_unit_interval(self, detector) -> None:
        """All returned scores must be in [0, 1]."""
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        result = detector.detect(frame)
        for item in result:
            assert 0.0 <= item.score <= 1.0, (
                f"score {item.score} out of [0, 1] range"
            )

    def test_detect_with_hint_bbox(self, detector) -> None:
        """detect() with a hint_bbox must not crash and must return a list."""
        from uav_tracker.types import BBox

        rng = np.random.default_rng(99)
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        hint = BBox(x=50.0, y=40.0, w=80.0, h=60.0)
        result = detector.detect(frame, hint=hint)
        assert isinstance(result, list)

    def test_detect_coordinates_within_frame(self, detector) -> None:
        """Returned bbox coordinates (after full-frame translation) must be
        within frame bounds."""
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        result = detector.detect(frame)
        for item in result:
            b = item.bbox
            assert b.x >= 0.0
            assert b.y >= 0.0
            assert b.x + b.w <= 320.0 + 1.0  # +1 for float rounding tolerance
            assert b.y + b.h <= 240.0 + 1.0
