from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComputeAction(str, Enum):
    FULL = "full"
    PRUNE_LIGHT = "prune_light"
    PRUNE_MEDIUM = "prune_medium"


class SearchAction(str, Enum):
    KEEP = "keep"
    EXPAND = "expand"
    FREEZE = "freeze"
    CENTER_ON_REINIT_HINT = "center_on_reinit_hint"


class TemplateAction(str, Enum):
    KEEP_CURRENT = "keep_current"
    UPDATE = "update"
    BLOCK_UPDATE = "block_update"


class RecoveryAction(str, Enum):
    NONE = "none"
    SCORE_CANDIDATES = "score_candidates"
    REINIT = "reinit"
    REJECT_REINIT = "reject_reinit"


BBox = tuple[float, float, float, float]  # x, y, w, h


@dataclass(frozen=True)
class TrackerAction:
    compute: ComputeAction = ComputeAction.FULL
    search: SearchAction = SearchAction.KEEP
    template: TemplateAction = TemplateAction.KEEP_CURRENT
    recovery: RecoveryAction = RecoveryAction.NONE
    bbox_hint: BBox | None = None
    detector_hint: BBox | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "compute": self.compute.value,
            "search": self.search.value,
            "template": self.template.value,
            "recovery": self.recovery.value,
            "bbox_hint": list(self.bbox_hint) if self.bbox_hint is not None else None,
            "detector_hint": list(self.detector_hint) if self.detector_hint is not None else None,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "TrackerAction":
        return cls(
            compute=ComputeAction(d["compute"]),
            search=SearchAction(d["search"]),
            template=TemplateAction(d["template"]),
            recovery=RecoveryAction(d["recovery"]),
            bbox_hint=tuple(d["bbox_hint"]) if d.get("bbox_hint") is not None else None,
            detector_hint=tuple(d["detector_hint"]) if d.get("detector_hint") is not None else None,
        )
