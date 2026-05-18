"""Unit tests for ``per_attribute_breakdown``.

Builds a minimal fake OPEResult + fake dataset with two synthetic sequences
having different attribute sets, and asserts the breakdown returns correct
group means.

No filesystem access, no opencv, no torch required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs that satisfy per_attribute_breakdown's duck-typed interface.
# ---------------------------------------------------------------------------


@dataclass
class _FakeSeqResult:
    name: str
    auc: float


@dataclass
class _FakeOPEResult:
    per_sequence: list[_FakeSeqResult] = field(default_factory=list)


@dataclass
class _FakeSeq:
    name: str
    attributes: set[str]


class _FakeDataset:
    def __init__(self, seqs: list[_FakeSeq]) -> None:
        self._seqs = seqs

    def __iter__(self) -> Iterator[_FakeSeq]:
        return iter(self._seqs)


class _NoAttrDataset:
    """Dataset whose sequences do NOT expose .attributes."""

    @dataclass
    class _Seq:
        name: str
        # deliberately no 'attributes' field

    def __init__(self) -> None:
        self._seqs = [self._Seq("s1"), self._Seq("s2")]

    def __iter__(self) -> Iterator:
        return iter(self._seqs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import_per_attribute_breakdown() -> None:
    """per_attribute_breakdown must be importable from the report module."""
    from uav_tracker.evaluation.report import per_attribute_breakdown  # noqa: F401

    assert callable(per_attribute_breakdown)


def test_basic_group_means() -> None:
    """Correct mean AUC computed per attribute group."""
    from uav_tracker.evaluation.report import per_attribute_breakdown

    # seq_a has FM + OCC; seq_b has OCC + IV.
    result = _FakeOPEResult(
        per_sequence=[
            _FakeSeqResult(name="seq_a", auc=0.6),
            _FakeSeqResult(name="seq_b", auc=0.4),
        ]
    )
    dataset = _FakeDataset(
        [
            _FakeSeq(name="seq_a", attributes={"FM", "OCC"}),
            _FakeSeq(name="seq_b", attributes={"OCC", "IV"}),
        ]
    )

    breakdown = per_attribute_breakdown(result, dataset)

    # FM: only seq_a → 0.6
    assert "FM" in breakdown
    assert breakdown["FM"] == pytest.approx(0.6, abs=1e-9)

    # IV: only seq_b → 0.4
    assert "IV" in breakdown
    assert breakdown["IV"] == pytest.approx(0.4, abs=1e-9)

    # OCC: seq_a (0.6) + seq_b (0.4) → mean = 0.5
    assert "OCC" in breakdown
    assert breakdown["OCC"] == pytest.approx(0.5, abs=1e-9)


def test_attribute_not_present_omitted() -> None:
    """Attributes with no matching sequences must not appear in the breakdown."""
    from uav_tracker.evaluation.report import per_attribute_breakdown

    result = _FakeOPEResult(
        per_sequence=[
            _FakeSeqResult(name="seq_a", auc=0.7),
        ]
    )
    dataset = _FakeDataset(
        [
            _FakeSeq(name="seq_a", attributes={"FM"}),  # only FM is set
        ]
    )

    breakdown = per_attribute_breakdown(result, dataset)

    # FM should be present.
    assert "FM" in breakdown

    # All other UAV123 attributes (OCC, IV, ...) must be absent.
    for attr in ["OCC", "IV", "SV", "POC", "DEF", "MB", "CM", "BC", "SOB", "LR", "ARC"]:
        assert attr not in breakdown, f"Attribute {attr} should not be in breakdown"


def test_limit_subset_of_sequences() -> None:
    """Sequences missing from result (e.g. due to --limit) are ignored."""
    from uav_tracker.evaluation.report import per_attribute_breakdown

    # result only has seq_a; seq_b is in dataset but not in result.
    result = _FakeOPEResult(
        per_sequence=[
            _FakeSeqResult(name="seq_a", auc=0.8),
        ]
    )
    dataset = _FakeDataset(
        [
            _FakeSeq(name="seq_a", attributes={"FM"}),
            _FakeSeq(name="seq_b", attributes={"FM", "OCC"}),
        ]
    )

    breakdown = per_attribute_breakdown(result, dataset)

    # FM should only reflect seq_a (0.8), not seq_b.
    assert breakdown.get("FM") == pytest.approx(0.8, abs=1e-9)
    # OCC: seq_b not in result → OCC absent.
    assert "OCC" not in breakdown


def test_no_attributes_returns_empty_dict(caplog) -> None:
    """Dataset without .attributes returns {} and emits a log warning."""
    import logging
    from uav_tracker.evaluation.report import per_attribute_breakdown

    result = _FakeOPEResult(
        per_sequence=[
            _FakeSeqResult(name="s1", auc=0.5),
        ]
    )
    dataset = _NoAttrDataset()

    with caplog.at_level(logging.WARNING, logger="uav_tracker.evaluation.report"):
        breakdown = per_attribute_breakdown(result, dataset)

    assert breakdown == {}


def test_empty_result_returns_empty_dict() -> None:
    """Empty per_sequence list returns {} without iterating the dataset."""
    from uav_tracker.evaluation.report import per_attribute_breakdown

    result = _FakeOPEResult(per_sequence=[])
    dataset = _FakeDataset([_FakeSeq("seq_a", {"FM"})])

    breakdown = per_attribute_breakdown(result, dataset)
    assert breakdown == {}


def test_all_uav123_attributes_can_be_populated() -> None:
    """All 12 UAV123 attribute codes are returned when sequences cover them all."""
    from uav_tracker.evaluation.report import per_attribute_breakdown, _UAV123_ATTRIBUTES

    # One sequence per attribute for easy verification.
    all_attrs = list(_UAV123_ATTRIBUTES)
    seqs_data = [
        _FakeSeq(name=f"seq_{i}", attributes={attr})
        for i, attr in enumerate(all_attrs)
    ]
    seq_results = [
        _FakeSeqResult(name=f"seq_{i}", auc=float(i) / len(all_attrs))
        for i in range(len(all_attrs))
    ]

    result = _FakeOPEResult(per_sequence=seq_results)
    dataset = _FakeDataset(seqs_data)

    breakdown = per_attribute_breakdown(result, dataset)

    assert set(breakdown.keys()) == set(all_attrs), (
        f"Missing attrs: {set(all_attrs) - set(breakdown.keys())}"
    )
    # Each attribute should have its own sequence's AUC.
    for i, attr in enumerate(all_attrs):
        expected = float(i) / len(all_attrs)
        assert breakdown[attr] == pytest.approx(expected, abs=1e-9), (
            f"{attr}: expected {expected:.4f}, got {breakdown[attr]:.4f}"
        )


def test_multiple_sequences_same_attribute_averaged() -> None:
    """Mean AUC is correctly averaged when multiple sequences share an attribute."""
    from uav_tracker.evaluation.report import per_attribute_breakdown

    result = _FakeOPEResult(
        per_sequence=[
            _FakeSeqResult(name="a", auc=0.2),
            _FakeSeqResult(name="b", auc=0.4),
            _FakeSeqResult(name="c", auc=0.6),
            _FakeSeqResult(name="d", auc=0.8),
        ]
    )
    dataset = _FakeDataset(
        [
            _FakeSeq(name="a", attributes={"LR"}),
            _FakeSeq(name="b", attributes={"LR"}),
            _FakeSeq(name="c", attributes={"LR"}),
            _FakeSeq(name="d", attributes={"LR"}),
        ]
    )

    breakdown = per_attribute_breakdown(result, dataset)
    # mean of 0.2, 0.4, 0.6, 0.8 = 0.5
    assert "LR" in breakdown
    assert breakdown["LR"] == pytest.approx(0.5, abs=1e-9)
