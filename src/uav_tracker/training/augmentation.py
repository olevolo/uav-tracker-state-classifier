"""UAV-specific augmentation pipeline for 128×128 tracking patches (Phase 11).

Applies a sequence of stochastic augmentations designed to improve tracker
robustness under UAV-flight conditions:

  1. RandomHFlip       — horizontal mirror (p=0.5)
  2. ColorJitter       — brightness / contrast / saturation perturbation
  3. GaussianBlur      — simulates motion blur (p=0.3, sigma in [0.1, 1.0])
  4. RandomGrayscale   — simulates IR / night channels (p=0.1)

All random state is driven by seeded ``random.Random`` and
``np.random.RandomState`` instances so results are reproducible when
both are initialised with the same seed.

Usage
-----
    from uav_tracker.training.augmentation import UAVAugmentPipeline

    aug = UAVAugmentPipeline(seed=42)
    patch_np = np.zeros((128, 128, 3), dtype=np.uint8)
    tensor = aug(patch_np)   # torch.Tensor shape [3, 128, 128], float32 in [0,1]
"""

from __future__ import annotations

import random

import cv2
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


class UAVAugmentPipeline:
    """UAV-specific augmentation for 128×128 tracking patches.

    Parameters
    ----------
    seed:
        Integer seed for both ``random.Random`` and
        ``np.random.RandomState``.  Using the same seed across different
        instances produces identical augmentation sequences.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def __call__(self, patch: np.ndarray) -> "torch.Tensor":
        """Apply the augmentation pipeline and return a normalised tensor.

        Parameters
        ----------
        patch:
            Input image patch, shape ``(H, W, 3)`` — expected to be 128×128
            but the pipeline is size-agnostic.  Accepted dtypes: ``uint8``
            (values 0–255) or ``float32`` (values 0–255 or 0–1; both handled).

        Returns
        -------
        torch.Tensor
            Shape ``(3, H, W)``, dtype ``float32``, values in ``[0, 1]``.
            (Falls back to a float32 numpy CHW array when torch is absent.)
        """
        # Ensure uint8 for OpenCV ops.
        if patch.dtype != np.uint8:
            arr = np.clip(patch, 0, 255).astype(np.uint8)
        else:
            arr = patch.copy()

        arr = self._random_hflip(arr)
        arr = self._color_jitter(arr)
        arr = self._gaussian_blur(arr)
        arr = self._random_grayscale(arr)

        # Normalise to [0, 1] float32.
        out = arr.astype(np.float32) / 255.0

        # Convert HWC → CHW.
        out = out.transpose(2, 0, 1)  # (3, H, W)

        if _TORCH_AVAILABLE:
            return torch.from_numpy(out)
        return out  # type: ignore[return-value]

    # -----------------------------------------------------------------------
    # Augmentation primitives
    # -----------------------------------------------------------------------

    def _random_hflip(self, patch: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Flip the patch horizontally with probability *p*."""
        if self._rng.random() < p:
            return patch[:, ::-1, :].copy()
        return patch

    def _color_jitter(
        self,
        patch: np.ndarray,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
    ) -> np.ndarray:
        """Randomly perturb brightness, contrast, and saturation.

        Each channel-level transform is applied with a factor drawn from
        ``[1 - delta, 1 + delta]``.  Operations are applied in a random
        order (brightness → contrast → saturation, shuffled).
        """
        ops = self._rng.sample(["brightness", "contrast", "saturation"], 3)
        result = patch.astype(np.float32)

        for op in ops:
            if op == "brightness":
                factor = 1.0 + self._np_rng.uniform(-brightness, brightness)
                result = result * factor

            elif op == "contrast":
                factor = 1.0 + self._np_rng.uniform(-contrast, contrast)
                mean = result.mean(axis=(0, 1), keepdims=True)
                result = mean + (result - mean) * factor

            elif op == "saturation":
                factor = 1.0 + self._np_rng.uniform(-saturation, saturation)
                # Convert to grayscale in-place and blend.
                gray = (
                    0.299 * result[:, :, 0]
                    + 0.587 * result[:, :, 1]
                    + 0.114 * result[:, :, 2]
                )
                gray3 = gray[:, :, np.newaxis]
                result = gray3 + (result - gray3) * factor

        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)

    def _gaussian_blur(
        self,
        patch: np.ndarray,
        sigma_range: tuple[float, float] = (0.1, 1.0),
        p: float = 0.3,
    ) -> np.ndarray:
        """Apply Gaussian blur with probability *p*.

        Sigma is sampled uniformly from *sigma_range*.  Kernel size is
        chosen as the smallest odd integer >= 6*sigma + 1, capped at 11
        to avoid excessive blurring on 128-px patches.
        """
        if self._rng.random() >= p:
            return patch

        sigma = self._np_rng.uniform(sigma_range[0], sigma_range[1])
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 11)

        return cv2.GaussianBlur(patch, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    def _random_grayscale(self, patch: np.ndarray, p: float = 0.1) -> np.ndarray:
        """Convert patch to 3-channel grayscale with probability *p*.

        Simulates IR or night-vision sensors encountered in UAV footage.
        """
        if self._rng.random() >= p:
            return patch

        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)  # (H, W)
        return np.stack([gray, gray, gray], axis=2)


__all__ = ["UAVAugmentPipeline"]
