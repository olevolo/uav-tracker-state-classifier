"""Shared pytest fixtures.

Seeding policy (PLAN §12.7): every test that touches RNG must opt into
``rng_seeded`` so ``numpy`` / ``random`` / ``torch`` are pinned to 42.
"""

from __future__ import annotations

import random
from typing import Iterator

import numpy as np
import pytest
from hypothesis import settings

from tests.fixtures.synthetic_sequences import (
    translating_rectangle,
    SyntheticSequence,
)

# Relax hypothesis defaults: CI can be slow; we tune examples down.
settings.register_profile("uav_tracker", deadline=None, max_examples=50)
settings.load_profile("uav_tracker")


@pytest.fixture
def rng_seeded() -> Iterator[None]:
    """Pin numpy / random / torch RNGs to 42 for the duration of a test."""
    random.seed(42)
    np.random.seed(42)
    try:  # torch is optional at test collection time
        import torch  # type: ignore

        torch.manual_seed(42)
        if torch.cuda.is_available():  # pragma: no cover - not on CI host
            torch.cuda.manual_seed_all(42)
    except Exception:  # noqa: BLE001
        pass
    yield


@pytest.fixture
def tiny_frame() -> np.ndarray:
    """Tiny synthetic RGB frame (for import smoke tests)."""
    return np.full((32, 48, 3), 64, dtype=np.uint8)


@pytest.fixture
def smooth_sequence() -> SyntheticSequence:
    """20-frame translating rectangle — integration-test baseline."""
    return translating_rectangle()
