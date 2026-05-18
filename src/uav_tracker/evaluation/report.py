"""Result reporting: CSV + markdown table writers (Phase 1+ stub).

Every result CSV carries a provenance header (git SHA, dataset SHA,
weights SHA, image tag, GPU name, hostname, timestamp) per PLAN §6.8.
Real implementation lands with the evaluation pipeline in Phase 1.

Phase 7 addition: ``per_attribute_breakdown`` — compute mean AUC per
UAV123 attribute over the subset of sequences carrying that flag.

Phase 15 addition: ``write_csv``, ``write_markdown``, ``write_scene_breakdown``
fully implemented.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# UAV123 canonical 12-attribute codes (PLAN §3.6 / Mueller 2016).
_UAV123_ATTRIBUTES = [
    "FM", "OCC", "IV", "SV", "POC", "DEF", "MB", "CM", "BC", "SOB", "LR", "ARC"
]


def write_csv(result: Any, path: Path | str) -> None:
    """Write OPE result to CSV with columns: sequence, auc, precision_at_20, fps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "auc", "precision_at_20", "fps"])
        writer.writeheader()
        for sr in result.per_sequence:
            writer.writerow({
                "sequence": sr.name,
                "auc": round(sr.auc, 4),
                "precision_at_20": round(sr.precision_at_20, 4),
                "fps": round(sr.fps, 1),
            })


def write_markdown(result: Any, path: Path | str, title: str = "OPE Results") -> None:
    """Write OPE result to markdown table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Overall AUC:** {result.auc:.4f}  \n")
        f.write(f"**Precision@20:** {result.precision_at_20:.4f}  \n")
        f.write(f"**FPS:** {result.fps:.1f}  \n\n")
        f.write("| Sequence | AUC | Pr@20 | FPS |\n")
        f.write("|---|---|---|---|\n")
        for sr in result.per_sequence:
            f.write(
                f"| {sr.name} | {sr.auc:.4f} | {sr.precision_at_20:.4f} | {sr.fps:.1f} |\n"
            )


def write_scene_breakdown(telemetry: list, path: Path | str) -> dict[str, float]:
    """Compute and write per-scene-class AUC breakdown from HybridRunner telemetry.

    telemetry: list of TelemetryEntry objects with .aux["scene_class"] populated.
    Returns: dict mapping scene class name → mean tracker confidence.
    """
    from uav_tracker.types import SceneClass

    class_confidences: dict[str, list[float]] = defaultdict(list)
    for entry in telemetry:
        sc = entry.aux.get("scene_class")
        if sc is not None:
            name = SceneClass(sc).name
            class_confidences[name].append(entry.confidence)

    breakdown = {
        k: float(sum(v) / len(v)) for k, v in class_confidences.items() if v
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Scene Class Breakdown\n\n")
        f.write("| Scene Class | Mean Confidence | Frame Count |\n")
        f.write("|---|---|---|\n")
        for sc_member in SceneClass:
            sc_name = sc_member.name
            if sc_name in breakdown:
                count = len(class_confidences[sc_name])
                f.write(f"| {sc_name} | {breakdown[sc_name]:.4f} | {count} |\n")

    return breakdown


def per_attribute_breakdown(result: Any, dataset: Any) -> dict[str, float]:
    """Compute mean AUC per UAV123 attribute over matching sequences.

    For each of the 12 UAV123 attribute codes ("FM", "OCC", "IV", "SV",
    "POC", "DEF", "MB", "CM", "BC", "SOB", "LR", "ARC"), this function
    finds all sequences in *result* whose corresponding entry in *dataset*
    carries that attribute flag, then returns the mean AUC for that subset.

    Parameters
    ----------
    result:
        An ``OPEResult``-compatible object with a ``per_sequence`` list of
        objects that have ``.name`` and ``.auc`` attributes.
    dataset:
        An iterable of sequence objects that have ``.name`` and
        ``.attributes`` (a ``set[str]`` or ``frozenset[str]``).
        If a sequence object does not expose ``.attributes``, the function
        emits a log warning and returns ``{}``.

    Returns
    -------
    dict[str, float]
        Mapping from attribute code to mean AUC over the sequences that
        carry that attribute.  Attributes with zero matching sequences are
        omitted from the dict.  Returns ``{}`` if the dataset does not
        expose per-sequence attributes.

    Notes
    -----
    - This function performs a full pass over *dataset* to build the
      attribute→sequence mapping; it does **not** trigger frame loading.
    - If sequences in *result* are a strict subset of *dataset* (e.g. due
      to ``--limit``), only the intersection is used.
    """
    # Build name → AUC map from the result.
    name_to_auc: dict[str, float] = {}
    for sr in result.per_sequence:
        name_to_auc[sr.name] = sr.auc

    if not name_to_auc:
        return {}

    # Walk dataset once to collect attribute → [auc] lists.
    attr_aucs: dict[str, list[float]] = {a: [] for a in _UAV123_ATTRIBUTES}
    found_attrs = False

    for seq in dataset:
        attrs = getattr(seq, "attributes", None)
        if attrs is None:
            _log.warning(
                "per_attribute_breakdown: sequence %r has no .attributes — "
                "dataset does not expose attribute flags (e.g. synthetic). "
                "Returning {}.",
                getattr(seq, "name", "?"),
            )
            return {}
        found_attrs = True

        auc = name_to_auc.get(seq.name)
        if auc is None:
            # Sequence not in result (e.g. limit was applied) — skip.
            continue

        for attr in _UAV123_ATTRIBUTES:
            if attr in attrs:
                attr_aucs[attr].append(auc)

    if not found_attrs:
        # Dataset was empty.
        return {}

    # Compute means; skip attributes with no sequences.
    breakdown: dict[str, float] = {}
    for attr in _UAV123_ATTRIBUTES:
        values = attr_aucs[attr]
        if values:
            breakdown[attr] = float(sum(values) / len(values))

    return breakdown
