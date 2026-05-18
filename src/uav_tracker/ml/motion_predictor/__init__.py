"""Motion-predictor sub-package."""

from uav_tracker.ml.motion_predictor.base import MotionPredictor
from uav_tracker.ml.motion_predictor.lstm_predictor import OnlineLSTMMotionPredictor

__all__ = ["MotionPredictor", "OnlineLSTMMotionPredictor"]
