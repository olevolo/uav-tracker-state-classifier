# Architect-owned: re-exports the DATASETS registry and the Dataset/Sequence
# Protocols so callers can ``from uav_tracker.datasets import DATASETS``.
"""Dataset loaders. Architect owns ``base.py`` and the DATASETS registry."""

from uav_tracker.registry import DATASETS  # noqa: F401
from uav_tracker.datasets.base import Dataset, Sequence  # noqa: F401

from uav_tracker.datasets import synthetic as _synthetic_plugin  # noqa: F401
from uav_tracker.datasets import lasot as _lasot_plugin  # noqa: F401
from uav_tracker.datasets import got10k as _got10k_plugin  # noqa: F401
# The imports above trigger @DATASETS.register(...) side-effects.

__all__ = ["DATASETS", "Dataset", "Sequence"]
