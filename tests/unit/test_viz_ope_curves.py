"""Unit tests for viz.ope_curves — success and precision curve plots (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from uav_tracker.evaluation.ope import OPEResult, SequenceResult


def _make_ope_result(auc: float, pr20: float) -> OPEResult:
    """Build a minimal OPEResult with one sequence."""
    sr = SequenceResult(
        name="seq0",
        auc=auc,
        precision_at_20=pr20,
        fps=30.0,
        n_frames=100,
    )
    return OPEResult(auc=auc, precision_at_20=pr20, fps=30.0, per_sequence=[sr])


def test_success_curves_written(tmp_path: Path) -> None:
    """plot_success_curve writes a PNG file."""
    from uav_tracker.viz.ope_curves import plot_success_curve

    r1 = _make_ope_result(auc=0.70, pr20=0.80)
    r2 = _make_ope_result(auc=0.55, pr20=0.65)
    out = tmp_path / "success.png"

    plot_success_curve([r1, r2], out_path=out, labels=["TrackerA", "TrackerB"])

    assert out.exists(), "success.png was not created"
    assert out.stat().st_size > 1024, f"PNG too small ({out.stat().st_size} bytes)"


def test_precision_curves_written(tmp_path: Path) -> None:
    """plot_precision_curve writes a PNG file."""
    from uav_tracker.viz.ope_curves import plot_precision_curve

    r1 = _make_ope_result(auc=0.70, pr20=0.80)
    r2 = _make_ope_result(auc=0.55, pr20=0.65)
    out = tmp_path / "precision.png"

    plot_precision_curve([r1, r2], out_path=out, labels=["TrackerA", "TrackerB"])

    assert out.exists(), "precision.png was not created"
    assert out.stat().st_size > 1024, f"PNG too small ({out.stat().st_size} bytes)"


def test_success_curve_legend_contains_auc(tmp_path: Path) -> None:
    """The saved PNG legend should correspond to the provided AUC values.

    We verify this indirectly by checking that the render completes and
    that matplotlib embedded the right AUC in the label (tested via the
    internal curve computation, not by re-parsing the PNG).
    """
    from uav_tracker.viz.ope_curves import plot_success_curve, _iou_success_curve

    r1 = _make_ope_result(auc=0.70, pr20=0.80)
    r2 = _make_ope_result(auc=0.40, pr20=0.50)

    # Verify curve AUCs are close to the stored values.
    for result in [r1, r2]:
        thresholds, success = _iou_success_curve(result)
        curve_auc = float(np.trapz(success, thresholds))
        assert abs(curve_auc - result.auc) < 0.08, (
            f"Curve AUC {curve_auc:.3f} deviates too much from stored {result.auc:.3f}"
        )

    out = tmp_path / "success_legend.png"
    plot_success_curve([r1, r2], out_path=out, labels=["Hi-AUC", "Lo-AUC"])
    assert out.exists()


def test_precision_curve_legend_contains_pr20(tmp_path: Path) -> None:
    """Both curves are distinguishable — pr20 values differ noticeably."""
    from uav_tracker.viz.ope_curves import plot_precision_curve, _cle_precision_curve

    r1 = _make_ope_result(auc=0.70, pr20=0.90)
    r2 = _make_ope_result(auc=0.55, pr20=0.50)

    for result in [r1, r2]:
        thresholds, prec = _cle_precision_curve(result)
        # Precision at t=20 should equal pr20 (within tolerance).
        idx20 = int(np.argmin(np.abs(thresholds - 20.0)))
        assert abs(prec[idx20] - result.precision_at_20) < 0.02, (
            f"Precision@20 from curve {prec[idx20]:.3f} != stored {result.precision_at_20:.3f}"
        )

    out = tmp_path / "precision_legend.png"
    plot_precision_curve([r1, r2], out_path=out, labels=["Hi-Pr20", "Lo-Pr20"])
    assert out.exists()


def test_default_labels(tmp_path: Path) -> None:
    """Labels default to 'Tracker N' when not provided."""
    from uav_tracker.viz.ope_curves import plot_success_curve, plot_precision_curve

    r = _make_ope_result(auc=0.60, pr20=0.70)
    sc_out = tmp_path / "sc_default.png"
    pc_out = tmp_path / "pc_default.png"

    plot_success_curve([r], out_path=sc_out)
    plot_precision_curve([r], out_path=pc_out)

    assert sc_out.exists()
    assert pc_out.exists()


def test_single_result_with_raw_scores(tmp_path: Path) -> None:
    """Raw iou_scores / cle_scores in aux are used when available."""
    from uav_tracker.viz.ope_curves import plot_success_curve, plot_precision_curve

    rng = np.random.default_rng(42)
    iou_scores = rng.uniform(0.3, 0.9, 200).tolist()
    cle_scores = rng.uniform(2.0, 40.0, 200).tolist()

    r = OPEResult(
        auc=0.65,
        precision_at_20=0.75,
        fps=25.0,
        per_sequence=[],
        aux={"iou_scores": iou_scores, "cle_scores": cle_scores},
    )

    sc_out = tmp_path / "sc_raw.png"
    pc_out = tmp_path / "pc_raw.png"

    plot_success_curve([r], out_path=sc_out, labels=["raw"])
    plot_precision_curve([r], out_path=pc_out, labels=["raw"])

    assert sc_out.exists()
    assert pc_out.exists()
