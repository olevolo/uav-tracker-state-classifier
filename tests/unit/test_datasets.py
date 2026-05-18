"""Tests for the Phase 1 dataset stubs.

We verify Dataset Protocol conformance at the class level (has the
right attrs + construction signature) but skip behavior tests until
the Phase 1 loaders land.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uav_tracker.datasets.otb100 import OTB100Dataset
from uav_tracker.datasets.uav123 import UAV123Dataset


def test_uav123_has_protocol_fields(tmp_path: Path) -> None:
    ds = UAV123Dataset(root=tmp_path)
    assert ds.name == "uav123"
    assert ds.root == tmp_path
    assert ds.split == "test"
    assert ds.attributes is None


def test_otb100_has_protocol_fields(tmp_path: Path) -> None:
    ds = OTB100Dataset(root=tmp_path)
    assert ds.name == "otb100"
    assert ds.root == tmp_path


@pytest.mark.skip(reason="Phase 1: dataset loader implementation")
def test_uav123_iterates_sequences(tmp_path: Path) -> None:  # pragma: no cover
    pass


@pytest.mark.skip(reason="Phase 1: dataset loader implementation")
def test_otb100_iterates_sequences(tmp_path: Path) -> None:  # pragma: no cover
    pass
