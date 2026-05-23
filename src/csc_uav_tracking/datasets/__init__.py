# Architect-owned: re-exports the DATASETS registry and the Dataset/Sequence
# Protocols so callers can ``from csc_uav_tracking.datasets import DATASETS``.
"""Dataset loaders. Architect owns ``base.py`` and the DATASETS registry."""

from csc_uav_tracking.registry import DATASETS  # noqa: F401
from csc_uav_tracking.datasets.base import Dataset, Sequence  # noqa: F401

from csc_uav_tracking.datasets import synthetic as _synthetic_plugin  # noqa: F401
from csc_uav_tracking.datasets import uav123 as _uav123_plugin  # noqa: F401
from csc_uav_tracking.datasets import dtb70 as _dtb70_plugin  # noqa: F401
from csc_uav_tracking.datasets import visdrone_sot as _visdrone_sot_plugin  # noqa: F401
from csc_uav_tracking.datasets import lasot as _lasot_plugin  # noqa: F401
from csc_uav_tracking.datasets import got10k as _got10k_plugin  # noqa: F401
# The imports above trigger @DATASETS.register(...) side-effects.

__all__ = ["DATASETS", "Dataset", "Sequence"]
