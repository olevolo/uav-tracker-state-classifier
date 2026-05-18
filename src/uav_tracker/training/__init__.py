"""Training infrastructure for UAV Entropy-Guided Tracker (Phase 11).

Submodules
----------
label_generator : LabelGenerator
    Runs the Henriques-KCF baseline + motion-entropy signal over UAV123
    sequences to produce per-frame scene-class difficulty labels.
augmentation : UAVAugmentPipeline
    UAV-specific augmentation pipeline for 128×128 tracking patches.
"""

__all__ = ["label_generator", "augmentation"]
