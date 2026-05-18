"""Difficulty-predictor sub-package."""

from uav_tracker.ml.difficulty_predictor.base import DifficultyPredictor
from uav_tracker.ml.difficulty_predictor.regression_predictor import MLPDifficultyPredictor

__all__ = ["DifficultyPredictor", "MLPDifficultyPredictor"]
