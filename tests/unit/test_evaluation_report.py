"""Unit tests for evaluation.report: write_csv, write_markdown, write_scene_breakdown.

Phase 15 — final integration tests for the report module.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from uav_tracker.evaluation.ope import OPEResult, SequenceResult
from uav_tracker.evaluation.report import write_csv, write_markdown, write_scene_breakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ope_result() -> OPEResult:
    """Build a synthetic OPEResult with two sequence results."""
    return OPEResult(
        auc=0.5432,
        precision_at_20=0.7123,
        fps=42.0,
        per_sequence=[
            SequenceResult(
                name="seq_alpha",
                auc=0.6100,
                precision_at_20=0.8200,
                fps=50.0,
                n_frames=300,
            ),
            SequenceResult(
                name="seq_beta",
                auc=0.4764,
                precision_at_20=0.6046,
                fps=34.0,
                n_frames=200,
            ),
        ],
    )


def _make_telemetry(entries: list[tuple[int, float, float]]) -> list:
    """Build synthetic TelemetryEntry-like objects.

    Each entry is (scene_class_int, confidence, scene_confidence).
    scene_class=None omits the key from aux to test filtering.
    """
    records = []
    for sc, conf, sc_conf in entries:
        aux: dict = {"reliable": True}
        if sc is not None:
            aux["scene_class"] = sc
            aux["scene_confidence"] = sc_conf
        record = SimpleNamespace(
            frame_idx=len(records),
            confidence=conf,
            tier=0,
            switched=False,
            aux=aux,
        )
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_write_csv_creates_file(tmp_path: Path) -> None:
    """write_csv must create the output file."""
    result = _make_ope_result()
    out = tmp_path / "results" / "ope.csv"
    write_csv(result, out)
    assert out.exists(), "write_csv did not create the output file"


def test_write_csv_columns(tmp_path: Path) -> None:
    """CSV output must have exactly [sequence, auc, precision_at_20, fps] columns."""
    result = _make_ope_result()
    out = tmp_path / "ope.csv"
    write_csv(result, out)

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        assert list(fieldnames) == ["sequence", "auc", "precision_at_20", "fps"]

        rows = list(reader)

    assert len(rows) == 2, "Expected two rows (one per sequence)"
    assert rows[0]["sequence"] == "seq_alpha"
    assert rows[1]["sequence"] == "seq_beta"
    # Values are rounded to 4 decimal places for auc/precision and 1 for fps.
    assert float(rows[0]["auc"]) == pytest.approx(0.6100, abs=1e-4)
    assert float(rows[0]["fps"]) == pytest.approx(50.0, abs=0.1)


def test_write_csv_creates_parent_dirs(tmp_path: Path) -> None:
    """write_csv must create missing parent directories."""
    result = _make_ope_result()
    out = tmp_path / "deep" / "nested" / "dir" / "results.csv"
    write_csv(result, out)
    assert out.exists()


def test_write_markdown_creates_file(tmp_path: Path) -> None:
    """write_markdown must create the output file."""
    result = _make_ope_result()
    out = tmp_path / "report.md"
    write_markdown(result, out)
    assert out.exists(), "write_markdown did not create the output file"


def test_write_markdown_content(tmp_path: Path) -> None:
    """Markdown report must contain expected header and table rows."""
    result = _make_ope_result()
    out = tmp_path / "report.md"
    write_markdown(result, out, title="Test OPE")

    content = out.read_text()
    assert "# Test OPE" in content
    assert "0.5432" in content  # overall AUC
    assert "seq_alpha" in content
    assert "seq_beta" in content
    # Markdown table separators
    assert "|---|" in content


def test_write_markdown_custom_title(tmp_path: Path) -> None:
    """write_markdown must honour the title parameter."""
    result = _make_ope_result()
    out = tmp_path / "report.md"
    write_markdown(result, out, title="Custom Title 123")
    content = out.read_text()
    assert "# Custom Title 123" in content


def test_scene_breakdown_groups_by_class(tmp_path: Path) -> None:
    """write_scene_breakdown must group entries by scene_class and compute mean confidence.

    SceneClass values: CLEAR=0, MODERATE=1, CHALLENGING=2.
    """
    # 3 CLEAR frames with confidence 0.9, 0.8, 0.7  → mean = 0.8
    # 2 MODERATE frames with confidence 0.5, 0.3    → mean = 0.4
    # 1 frame with no scene_class (sc=None)          → ignored
    telemetry = _make_telemetry([
        (0, 0.9, 1.0),   # CLEAR
        (0, 0.8, 0.9),   # CLEAR
        (0, 0.7, 0.8),   # CLEAR
        (1, 0.5, 0.7),   # MODERATE
        (1, 0.3, 0.6),   # MODERATE
        (None, 0.6, 0.0),  # no scene_class — must be ignored
    ])

    out = tmp_path / "scene_breakdown.md"
    breakdown = write_scene_breakdown(telemetry, out)

    assert set(breakdown.keys()) == {"CLEAR", "MODERATE"}
    assert breakdown["CLEAR"] == pytest.approx(0.8, abs=1e-6)
    assert breakdown["MODERATE"] == pytest.approx(0.4, abs=1e-6)


def test_scene_breakdown_file_created(tmp_path: Path) -> None:
    """write_scene_breakdown must write the breakdown to the given path."""
    telemetry = _make_telemetry([(2, 0.6, 0.9)])  # CHALLENGING
    out = tmp_path / "breakdown.md"
    write_scene_breakdown(telemetry, out)
    assert out.exists()
    content = out.read_text()
    assert "CHALLENGING" in content
    assert "# Scene Class Breakdown" in content


def test_scene_breakdown_empty_telemetry(tmp_path: Path) -> None:
    """write_scene_breakdown with empty telemetry returns empty dict and creates file."""
    out = tmp_path / "empty_breakdown.md"
    breakdown = write_scene_breakdown([], out)
    assert breakdown == {}
    assert out.exists()


def test_scene_breakdown_no_scene_class_entries(tmp_path: Path) -> None:
    """Entries missing scene_class key must be silently ignored."""
    telemetry = _make_telemetry([(None, 0.8, 0.0), (None, 0.7, 0.0)])
    out = tmp_path / "no_scene.md"
    breakdown = write_scene_breakdown(telemetry, out)
    assert breakdown == {}
