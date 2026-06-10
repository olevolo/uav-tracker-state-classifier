"""Verify that IoU is not part of the runtime feature vector.

CSC.md §2 anti-pattern: including IoU causes data leakage — the labels are
derived from IoU thresholds so the model trivially memorises the labeling rule.
Measured empirically: train macroF1 = 0.99 vs runtime macroF1 = 0.03 on the
same checkpoint when the IoU column is present at train time but zeroed at
runtime.

Three assertions:
1. 'iou' is NOT in FEATURE_NAMES.
2. build_runtime_feature() does NOT accept an 'iou' keyword argument (it has
   no such parameter in its signature).
3. build_offline_feature() with iou=0.99 and iou=0.01 (all other args fixed)
   returns IDENTICAL vectors — confirming that the 'iou' arg is silently ignored
   and never reaches the output.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csc_lib.csc.features import (
    FEATURE_NAMES,
    _State,
    build_offline_feature,
    build_runtime_feature,
)


# ---------------------------------------------------------------------------
# 1.  FEATURE_NAMES must not contain "iou" (or any variant)
# ---------------------------------------------------------------------------

class TestFeatureNamesNoIou:
    def test_iou_not_in_feature_names(self) -> None:
        """'iou' must not appear in FEATURE_NAMES."""
        assert "iou" not in FEATURE_NAMES, (
            f"Found 'iou' in FEATURE_NAMES={FEATURE_NAMES}. "
            "IoU leaks the labeling rule — see CSC.md §2 anti-patterns."
        )

    def test_no_iou_variant_in_feature_names(self) -> None:
        """No variant spelling of IoU (case-insensitive, with/without underscores)
        should appear in FEATURE_NAMES."""
        forbidden = {"iou", "intersection_over_union", "overlap"}
        lower_names = {n.lower() for n in FEATURE_NAMES}
        found = forbidden & lower_names
        assert not found, (
            f"IoU-variant feature(s) {found} found in FEATURE_NAMES. "
            "These must be excluded from the runtime feature vector."
        )


# ---------------------------------------------------------------------------
# 2.  build_runtime_feature() signature must not accept 'iou'
# ---------------------------------------------------------------------------

class TestRuntimeFeatureSignature:
    def test_runtime_feature_has_no_iou_param(self) -> None:
        """build_runtime_feature() must have no 'iou' parameter."""
        sig = inspect.signature(build_runtime_feature)
        assert "iou" not in sig.parameters, (
            "build_runtime_feature() has an 'iou' parameter. "
            "Remove it — IoU is not available at runtime."
        )

    def test_runtime_feature_raises_on_extra_iou_kwarg(self) -> None:
        """Passing iou=... to build_runtime_feature() must raise TypeError."""
        state = _State()
        with pytest.raises(TypeError):
            build_runtime_feature(
                confidence=0.5,
                apce=0.3,
                psr=0.4,
                pred_bbox=(100.0, 100.0, 50.0, 50.0),
                image_size=(1280, 720),
                state=state,
                iou=0.9,  # type: ignore[call-arg]
            )

    def test_runtime_feature_confidence_variants_match(self) -> None:
        """Two calls with same confidence but dummy extra kwarg — just checking
        normal confidence does not change output deterministically."""
        np.random.seed(0)
        state1 = _State()
        feat1 = build_runtime_feature(
            confidence=0.5,
            apce=0.3,
            psr=0.4,
            pred_bbox=(200.0, 150.0, 80.0, 60.0),
            image_size=(1280, 720),
            state=state1,
        )
        state2 = _State()
        feat2 = build_runtime_feature(
            confidence=0.5,
            apce=0.3,
            psr=0.4,
            pred_bbox=(200.0, 150.0, 80.0, 60.0),
            image_size=(1280, 720),
            state=state2,
        )
        np.testing.assert_array_equal(
            feat1, feat2,
            err_msg="build_runtime_feature() is not deterministic for identical inputs.",
        )


# ---------------------------------------------------------------------------
# 3.  build_offline_feature() ignores the iou argument
# ---------------------------------------------------------------------------

class TestOfflineFeatureIouIgnored:
    """Verify that build_offline_feature(iou=0.99, ...) == build_offline_feature(iou=0.01, ...)."""

    _COMMON = dict(
        confidence=0.42,
        apce=0.77,
        psr=1.2,
        pred_bbox=(300.0, 200.0, 120.0, 90.0),
        image_size=(1920, 1080),
    )

    def test_iou_high_vs_low_identical(self) -> None:
        state_a = _State()
        feat_high_iou = build_offline_feature(iou=0.99, state=state_a, **self._COMMON)

        state_b = _State()
        feat_low_iou = build_offline_feature(iou=0.01, state=state_b, **self._COMMON)

        np.testing.assert_array_equal(
            feat_high_iou,
            feat_low_iou,
            err_msg=(
                "build_offline_feature() produced different vectors for iou=0.99 vs iou=0.01. "
                "The 'iou' argument must be silently ignored — it is accepted for API stability "
                "but must not affect the returned feature vector (runtime has no GT)."
            ),
        )

    def test_iou_none_vs_zero_identical(self) -> None:
        state_a = _State()
        feat_none = build_offline_feature(iou=None, state=state_a, **self._COMMON)

        state_b = _State()
        feat_zero = build_offline_feature(iou=0.0, state=state_b, **self._COMMON)

        np.testing.assert_array_equal(feat_none, feat_zero)

    def test_iou_ignored_matches_runtime(self) -> None:
        """build_offline_feature(iou=X) must match build_runtime_feature() for same args."""
        state_a = _State()
        feat_offline = build_offline_feature(iou=0.5, state=state_a, **self._COMMON)

        state_b = _State()
        feat_runtime = build_runtime_feature(state=state_b, **self._COMMON)

        np.testing.assert_array_equal(
            feat_offline,
            feat_runtime,
            err_msg=(
                "build_offline_feature() and build_runtime_feature() returned different "
                "vectors for the same inputs. They must be identical when iou is ignored."
            ),
        )

    def test_feature_dim_matches_feature_names(self) -> None:
        """Feature vector length must match len(FEATURE_NAMES)."""
        from csc_lib.csc.features import FEATURE_DIM

        state = _State()
        feat = build_runtime_feature(
            confidence=0.5,
            apce=None,
            psr=None,
            pred_bbox=(0.0, 0.0, 100.0, 100.0),
            image_size=(640, 480),
            state=state,
        )
        assert len(feat) == FEATURE_DIM == len(FEATURE_NAMES), (
            f"Feature vector length {len(feat)} != FEATURE_DIM={FEATURE_DIM} "
            f"!= len(FEATURE_NAMES)={len(FEATURE_NAMES)}"
        )
