"""Appearance-memory sub-package."""

from uav_tracker.ml.appearance_memory.base import AppearanceMemory
from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory

__all__ = ["AppearanceMemory", "CosineAppearanceMemory"]
