"""Plugin registry for the UAV Entropy-Guided Tracker.

Architect-owned module. Provides a single generic ``Registry[T]`` class and five
module-level instances (``TRACKERS``, ``DETECTORS``, ``SIGNALS``,
``SCHEDULERS``, ``DATASETS``). Plugins register via the ``@registry.register("name")``
decorator at import time. The runner and CLI discover plugins by name from
Hydra configs (see ADR-0004).
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic name-to-factory registry for a single plugin kind.

    Parameters
    ----------
    kind : str
        A short label (``"tracker"``, ``"detector"``, ``"signal"``,
        ``"scheduler"``) used in error messages. Not used for lookup.

    Notes
    -----
    Keys must be unique within a registry; duplicate registration raises
    ``ValueError`` so plugin import clashes surface at import time, not later.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that registers a plugin factory under ``name``.

        Examples
        --------
        >>> @TRACKERS.register("kcf_kalman")
        ... class KCFKalmanTracker:
        ...     ...
        """

        def deco(cls: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise ValueError(
                    f"{self._kind}:{name} already registered "
                    f"(registered factory: {self._items[name]!r})"
                )
            self._items[name] = cls
            return cls

        return deco

    def build(self, name: str, **kwargs: Any) -> T:
        """Instantiate the plugin registered under ``name`` with ``**kwargs``.

        Raises
        ------
        KeyError
            If ``name`` is not registered. The error message lists all known
            names for the plugin kind, aiding config debugging.
        """

        if name not in self._items:
            raise KeyError(
                f"unknown {self._kind}: {name!r}. "
                f"Known {self._kind}s: {sorted(self._items)}"
            )
        return self._items[name](**kwargs)

    def names(self) -> list[str]:
        """Return the list of registered plugin names (sorted)."""

        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------- #
# Module-level registries (one per plugin kind).                              #
# --------------------------------------------------------------------------- #
#
# Type hints use forward references because the Protocols live in sibling
# packages; importing them here would cause a circular import.

TRACKERS: "Registry[Any]" = Registry("tracker")
DETECTORS: "Registry[Any]" = Registry("detector")
SIGNALS: "Registry[Any]" = Registry("signal")
SCHEDULERS: "Registry[Any]" = Registry("scheduler")
DATASETS: "Registry[Any]" = Registry("dataset")

# Phase 10 — ML extension registries
SCENE_CLASSIFIERS: "Registry[Any]" = Registry("scene_classifier")
DIFFICULTY_PREDICTORS: "Registry[Any]" = Registry("difficulty_predictor")
APPEARANCE_MEMORIES: "Registry[Any]" = Registry("appearance_memory")
MOTION_PREDICTORS: "Registry[Any]" = Registry("motion_predictor")
ML_WARMERS: "Registry[Any]" = Registry("ml_warmer")


__all__ = [
    "Registry",
    "TRACKERS",
    "DETECTORS",
    "SIGNALS",
    "SCHEDULERS",
    "DATASETS",
    "SCENE_CLASSIFIERS",
    "DIFFICULTY_PREDICTORS",
    "APPEARANCE_MEMORIES",
    "MOTION_PREDICTORS",
    "ML_WARMERS",
]
