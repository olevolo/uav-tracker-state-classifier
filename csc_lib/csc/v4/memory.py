"""CSC-v4 module A2 — prototype + distractor appearance memory.

V4 verifies (re-)localisation candidates by *identity*, not just confidence. This
module keeps three appearance stores over the SGLATrack embedding views and exposes
cosine-similarity signals that the candidate verifier (A3) and labeling (A4) consume:

  - anchor    : the frame-0 template prototype (set once, never moves). Mirrors the
                tracker's frozen ``_initial_template_embedding``. -> ``sim_to_init``.
  - recent    : an EMA prototype of the latest CC (correct-confirmed) appearances, so
                the model tolerates legitimate slow appearance drift. -> ``sim_to_recent``.
  - distractor: a bounded ring of embeddings sampled at FC/LA events (secondary peaks /
                wrong locks), so a candidate that looks like a known distractor is
                penalised. -> ``sim_to_distractor`` (max over the store).

Embeddings are 1-D arrays (e.g. tracker ``_last_search_peak_local`` /
``_last_template_embedding`` / ``_initial_template_embedding``, each (192,)). They are
``np.asarray``-coerced (torch tensors / lists accepted) and L2-normalised defensively, so
``sims()`` always returns cosines in [-1, 1] — or ``nan`` for any store that is empty.

V3 (csc_prod) is frozen and untouched; this is additive under csc_lib/csc/v4/.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from csc_lib.csc.v4.v4types import Prototype

_EPS = 1e-8


def _to_unit_vector(emb) -> Optional[np.ndarray]:
    """Coerce an embedding to a 1-D float64 L2-normalised np array, or None if unusable.

    Accepts np arrays, lists, or torch tensors (anything with ``.detach()``/``.numpy()``
    or that ``np.asarray`` handles). Multi-dim inputs are flattened. A zero/non-finite
    norm yields None (the store is treated as if nothing was added).
    """
    if emb is None:
        return None
    # Detach torch tensors (no hard torch dependency) before np.asarray.
    detach = getattr(emb, "detach", None)
    if callable(detach):
        emb = detach()
        cpu = getattr(emb, "cpu", None)
        if callable(cpu):
            emb = cpu()
    arr = np.asarray(emb, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < _EPS:
        return None
    return arr / norm


def _cosine_unit(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-unit vectors, clamped to [-1, 1]."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


class PrototypeMemory:
    """Appearance memory holding anchor / recent (EMA) / distractor prototypes.

    Parameters
    ----------
    max_recent : int
        Capacity of the recent-CC ring buffer (kept for history/inspection; the
        ``sim_to_recent`` signal uses the single EMA-fused recent prototype).
    max_distractor : int
        Capacity of the distractor ring buffer (oldest dropped past this).
    ema : float
        EMA weight in [0, 1] for the recent prototype: ``p <- ema*p + (1-ema)*new``
        (then re-normalised). Higher ``ema`` = slower drift / longer memory.
    """

    def __init__(self, max_recent: int = 5, max_distractor: int = 8, ema: float = 0.7) -> None:
        if not (0.0 <= ema <= 1.0):
            raise ValueError(f"ema must be in [0, 1], got {ema}")
        self.max_recent = int(max_recent)
        self.max_distractor = int(max_distractor)
        self.ema = float(ema)

        self._anchor: Optional[Prototype] = None
        self._recent: deque[Prototype] = deque(maxlen=self.max_recent)
        self._distractors: deque[Prototype] = deque(maxlen=self.max_distractor)
        # EMA-fused recent prototype (unit-norm); rebuilt incrementally on update_recent.
        self._recent_ema: Optional[np.ndarray] = None

    # ---- mutators -------------------------------------------------------------
    def update_anchor(self, emb) -> None:
        """Set the frame-0 anchor prototype. Idempotent: only the first valid call sticks."""
        if self._anchor is not None:
            return
        unit = _to_unit_vector(emb)
        if unit is None:
            return
        self._anchor = Prototype(embedding=unit, frame_idx=0, kind="anchor")

    def update_recent(self, emb, frame_idx: int) -> None:
        """Add a recent CC appearance and fold it into the EMA recent prototype."""
        unit = _to_unit_vector(emb)
        if unit is None:
            return
        self._recent.append(Prototype(embedding=unit, frame_idx=int(frame_idx), kind="recent"))
        if self._recent_ema is None:
            self._recent_ema = unit
        else:
            fused = self.ema * self._recent_ema + (1.0 - self.ema) * unit
            n = float(np.linalg.norm(fused))
            # Degenerate fusion (opposite vectors): fall back to newest sample.
            self._recent_ema = fused / n if n >= _EPS else unit

    def add_distractor(self, emb, frame_idx: int) -> None:
        """Record a distractor/wrong-lock embedding (sampled at FC/LA secondary peaks)."""
        unit = _to_unit_vector(emb)
        if unit is None:
            return
        self._distractors.append(
            Prototype(embedding=unit, frame_idx=int(frame_idx), kind="distractor")
        )

    # ---- queries --------------------------------------------------------------
    def sims(self, emb) -> dict[str, float]:
        """Cosine similarities of ``emb`` to each store.

        Returns ``{'sim_to_init', 'sim_to_recent', 'sim_to_distractor'}``. A value is
        ``nan`` when the corresponding store is empty or ``emb`` is unusable.
        ``sim_to_distractor`` is the MAX cosine over the distractor store (worst case).
        """
        query = _to_unit_vector(emb)
        nan = float("nan")
        if query is None:
            return {"sim_to_init": nan, "sim_to_recent": nan, "sim_to_distractor": nan}

        sim_to_init = (
            _cosine_unit(query, self._anchor.embedding) if self._anchor is not None else nan
        )
        sim_to_recent = (
            _cosine_unit(query, self._recent_ema) if self._recent_ema is not None else nan
        )
        if self._distractors:
            sim_to_distractor = max(
                _cosine_unit(query, p.embedding) for p in self._distractors
            )
        else:
            sim_to_distractor = nan

        return {
            "sim_to_init": sim_to_init,
            "sim_to_recent": sim_to_recent,
            "sim_to_distractor": sim_to_distractor,
        }

    # ---- introspection --------------------------------------------------------
    @property
    def has_anchor(self) -> bool:
        return self._anchor is not None

    @property
    def n_recent(self) -> int:
        return len(self._recent)

    @property
    def n_distractor(self) -> int:
        return len(self._distractors)

    def reset(self) -> None:
        """Clear all stores (e.g. between sequences)."""
        self._anchor = None
        self._recent.clear()
        self._distractors.clear()
        self._recent_ema = None


if __name__ == "__main__":
    # CPU-only, dataset-free smoke: random unit embeddings, assert sims in [-1,1] / nan-when-empty.
    rng = np.random.default_rng(0)
    DIM = 192  # SGLATrack embedding view dim

    def rand_emb() -> np.ndarray:
        v = rng.standard_normal(DIM)
        return v / np.linalg.norm(v)

    mem = PrototypeMemory(max_recent=5, max_distractor=8, ema=0.7)

    # 1) All stores empty -> every sim is nan.
    s0 = mem.sims(rand_emb())
    assert all(np.isnan(v) for v in s0.values()), f"expected all-nan on empty memory, got {s0}"
    assert not mem.has_anchor and mem.n_recent == 0 and mem.n_distractor == 0

    # 2) Anchor is set-once / idempotent.
    a0 = rand_emb()
    mem.update_anchor(a0)
    mem.update_anchor(rand_emb())  # ignored
    assert mem.has_anchor
    si = mem.sims(a0)["sim_to_init"]
    assert abs(si - 1.0) < 1e-6, f"self-similarity to anchor should be ~1, got {si}"

    # 3) Recent EMA prototype.
    for i in range(7):  # > max_recent to exercise the ring buffer
        mem.update_recent(rand_emb(), frame_idx=i)
    assert mem.n_recent == 5, f"recent ring should cap at 5, got {mem.n_recent}"
    assert not np.isnan(mem.sims(rand_emb())["sim_to_recent"])

    # 4) Distractor store: fill to capacity (8) with a known member still resident,
    #    so the max-cosine self-match below is well-defined.
    d_known = rand_emb()
    for i in range(7):
        mem.add_distractor(rand_emb(), frame_idx=20 + i)
    mem.add_distractor(d_known, frame_idx=27)
    assert mem.n_distractor == 8, f"distractor ring should be full at 8, got {mem.n_distractor}"
    # Overflow past capacity evicts the oldest, not the freshly-added d_known.
    mem.add_distractor(rand_emb(), frame_idx=28)
    assert mem.n_distractor == 8, f"distractor ring should cap at 8, got {mem.n_distractor}"

    # 5) Full sims: finite + in range, distractor self-match is the max (~1.0).
    for _ in range(200):
        s = mem.sims(rand_emb())
        for k, v in s.items():
            assert np.isfinite(v), f"{k} not finite: {v}"
            assert -1.0 <= v <= 1.0, f"{k} out of [-1,1]: {v}"
    sd = mem.sims(d_known)["sim_to_distractor"]
    assert abs(sd - 1.0) < 1e-6, f"distractor self-match (max) should be ~1, got {sd}"

    # 6) Defensive coercion: list and zero-vector inputs.
    assert _to_unit_vector(list(rand_emb())) is not None
    assert _to_unit_vector(np.zeros(DIM)) is None
    mem.update_recent(np.zeros(DIM), frame_idx=99)  # ignored, no crash
    assert mem.n_recent == 5

    # 7) reset() clears everything.
    mem.reset()
    assert not mem.has_anchor and mem.n_recent == 0 and mem.n_distractor == 0
    assert all(np.isnan(v) for v in mem.sims(rand_emb()).values())

    print("PrototypeMemory smoke OK: sims in [-1,1], nan-when-empty, EMA + ring buffers, reset.")
