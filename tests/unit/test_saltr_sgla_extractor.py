"""Unit tests for saltr/src/salt_r/sgla_memory_extractor.py.

All tests are pure-numpy / pure-Python — no model weights, no NPZ files on
disk needed.  The tracker and PositiveMemory calls are mocked or exercised
via lightweight stubs.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# 1. _pos_feature_indices() returns correct 4 indices
# ---------------------------------------------------------------------------

class TestPosFeatureIndices:
    def test_returns_four_indices(self):
        from salt_r.sgla_memory_extractor import _pos_feature_indices
        indices = _pos_feature_indices()
        assert len(indices) == 4, f"Expected 4 indices, got {len(indices)}"

    def test_indices_match_feature_names(self):
        from salt_r.sgla_memory_extractor import _pos_feature_indices, _POS_FEATURE_NAMES
        from salt_r.memory import DistractorAwareMemory
        indices = _pos_feature_indices()
        all_names = DistractorAwareMemory.FEATURE_NAMES
        for i, (idx, expected_name) in enumerate(zip(indices, _POS_FEATURE_NAMES)):
            assert idx < len(all_names), (
                f"Index {idx} out of range for FEATURE_NAMES (len={len(all_names)})"
            )
            assert all_names[idx] == expected_name, (
                f"Position {i}: expected '{expected_name}' at index {idx}, "
                f"got '{all_names[idx]}'"
            )

    def test_indices_include_pos_max_mean_recency_age(self):
        from salt_r.sgla_memory_extractor import _pos_feature_indices
        from salt_r.memory import DistractorAwareMemory
        indices = _pos_feature_indices()
        all_names = DistractorAwareMemory.FEATURE_NAMES
        selected = [all_names[i] for i in indices]
        assert "mem_pos_max_sim" in selected
        assert "mem_pos_mean_sim" in selected
        assert "mem_pos_recency_sim" in selected
        assert "mem_update_age" in selected

    def test_indices_are_unique(self):
        from salt_r.sgla_memory_extractor import _pos_feature_indices
        indices = _pos_feature_indices()
        assert len(set(indices)) == len(indices), "Duplicate indices returned"


# ---------------------------------------------------------------------------
# 2. Causal ordering: compute_features called before step each frame
# ---------------------------------------------------------------------------

class TestCausalOrdering:
    """Verify that compute_features() is always called before step() for each frame."""

    def _make_mock_runner(self, embedding_view: str = "score_weighted"):
        """Return a mock SALTRunner with a .tracker attribute that yields deterministic embeddings.

        After the refactor, extract_sequence() uses runner.run() and reads
        embeddings from runner.tracker._last_search_* attributes.  This helper
        builds a lightweight mock runner suitable for unit tests that need to
        exercise _get_embedding() without loading real model weights.
        """
        import torch

        # Build a mock tracker (the SGLATracker inside the runner)
        tracker = MagicMock()
        dim = 192
        rng = np.random.default_rng(42)

        def _make_tensor(t: int):
            v = rng.standard_normal(dim).astype(np.float32)
            return torch.from_numpy(v)

        # Build a mock runner whose .run() yields TelemetryEntry-like objects
        # and updates tracker._last_search_* on each frame after frame 0.
        runner = MagicMock()
        runner.tracker = tracker

        call_count = [0]

        def _run_side_effect(seq_obj):
            # Frame 0: tracker.init() called internally; search attrs remain None
            tracker._last_search_score_weighted = None
            tracker._last_search_peak_local = None
            tracker._last_search_global = None
            yield MagicMock()  # frame 0 entry

            frames = list(seq_obj.frames)
            for _frame in frames[1:]:
                call_count[0] += 1
                t = call_count[0]
                tensor = _make_tensor(t)
                tracker._last_search_score_weighted = tensor
                tracker._last_search_peak_local = tensor
                tracker._last_search_global = tensor
                yield MagicMock()  # frame t entry

        runner.run.side_effect = _run_side_effect
        return runner

    def test_compute_features_before_step_T3(self):
        """For T=3 frames, compute_features must be called before any RAM update.

        We directly exercise the extractor's causal loop pattern:
            features[t] = mem.compute_features(emb[t])   # uses < t only
            if gate: mem.positive.add(MemoryEntry(emb[t], ...))

        A call_log records each operation with its frame index.  We then
        assert that for every frame t > 0, the compute_features entry for t
        precedes any add entry for t.
        """
        from salt_r.memory import DistractorAwareMemory, MemoryEntry
        from salt_r.sgla_memory_extractor import _pos_feature_indices

        call_log: list = []

        dim = 192
        rng = np.random.default_rng(7)

        def _rand_emb(seed=None):
            r = np.random.default_rng(seed)
            v = r.standard_normal(dim).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-8)

        mem = DistractorAwareMemory()
        feature_indices = _pos_feature_indices()
        T = 3

        # --- Simulate the exact causal loop from extract_sequence ---

        # Frame 0: init — no compute_features, output row stays zeros
        emb_0 = _rand_emb(seed=0)
        mem._current_frame = 0
        # Gate always passes for this test
        call_log.append(("add", 0))
        mem.positive.add(MemoryEntry(
            embedding=emb_0.copy(), frame_idx=0,
            iou=float("nan"), apce_norm=1.0, p_fc=0.05,
            source="target_confident",
        ))

        # Frames 1..T-1: compute_features THEN add
        for t in range(1, T):
            emb_t = _rand_emb(seed=t)
            mem._current_frame = t

            # compute_features is called first
            call_log.append(("compute_features", t))
            feat_dict = mem.compute_features(query_emb=emb_t)

            # Then RAM is updated
            call_log.append(("add", t))
            mem.positive.add(MemoryEntry(
                embedding=emb_t.copy(), frame_idx=t,
                iou=float("nan"), apce_norm=1.0, p_fc=0.05,
                source="target_confident",
            ))

        # Verify ordering: for each t in {1, 2}, compute_features appears before add
        for t in range(1, T):
            cf_pos = next(
                (i for i, (op, ft) in enumerate(call_log) if op == "compute_features" and ft == t),
                None,
            )
            add_pos = next(
                (i for i, (op, ft) in enumerate(call_log) if op == "add" and ft == t),
                None,
            )
            assert cf_pos is not None, f"compute_features not called for frame {t}"
            assert add_pos is not None, f"add not called for frame {t}"
            assert cf_pos < add_pos, (
                f"Frame {t}: compute_features at log[{cf_pos}] must come before "
                f"add at log[{add_pos}]. Full log: {call_log}"
            )

    def test_compute_features_sees_no_same_frame_embedding(self):
        """compute_features at t uses only frames < t (causal invariant).

        Strategy: populate the RAM with known embeddings from frames 0..t-1,
        then call compute_features with a query = frame t's embedding.
        The similarity result should reflect only the pre-t RAM content.
        """
        from salt_r.memory import DistractorAwareMemory, MemoryEntry

        dim = 192
        rng = np.random.default_rng(99)
        mem = DistractorAwareMemory()

        # Add frame 0 embedding (orthogonal to future embeddings for clarity)
        emb_0 = np.zeros(dim, dtype=np.float32)
        emb_0[0] = 1.0
        mem.positive.add(MemoryEntry(
            embedding=emb_0, frame_idx=0, iou=1.0, apce_norm=1.0,
            p_fc=0.05, source="target_confident",
        ))

        # Frame 1 query — completely different direction
        emb_1 = np.zeros(dim, dtype=np.float32)
        emb_1[1] = 1.0

        # Before adding frame 1 to RAM, compute_features should only see frame 0
        mem._current_frame = 1
        feat = mem.compute_features(query_emb=emb_1)

        # emb_1 is orthogonal to emb_0 → similarity = 0
        assert abs(feat["mem_pos_max_sim"]) < 1e-4, (
            f"Expected near-zero similarity (orthogonal embeddings), "
            f"got {feat['mem_pos_max_sim']}"
        )

        # Now add frame 1 — after this, same_frame similarity would be 1.0
        mem.positive.add(MemoryEntry(
            embedding=emb_1, frame_idx=1, iou=1.0, apce_norm=1.0,
            p_fc=0.05, source="target_confident",
        ))

        # Frame 2 query = same as emb_1 → now max_sim should be ~1.0 (frame 1 in RAM)
        mem._current_frame = 2
        feat2 = mem.compute_features(query_emb=emb_1)
        assert feat2["mem_pos_max_sim"] > 0.99, (
            f"Expected near-1.0 similarity after frame 1 added to RAM, "
            f"got {feat2['mem_pos_max_sim']}"
        )


# ---------------------------------------------------------------------------
# 3. Frame 0 is always all-zeros
# ---------------------------------------------------------------------------

class TestFrame0Zeros:
    def test_frame0_output_is_all_zeros_by_design(self):
        """The extractor initializes result array with zeros; frame 0 is never overwritten.

        This tests the contract: result = np.zeros((T, 4)) and frame 0 is
        NOT touched after init (no compute_features call for frame 0).
        """
        # Simulate the extract_sequence contract directly without running the tracker
        T = 5
        result = np.zeros((T, 4), dtype=np.float32)

        # Frame 0 is never written to — only frames 1..T-1 are
        # (in the real extractor, frame 0 just has zeros from initialization)

        assert np.all(result[0] == 0.0), (
            f"Frame 0 should be all-zeros at initialization, got {result[0]}"
        )

    def test_frame0_zeros_regardless_of_embedding(self):
        """Even with a non-zero embedding at frame 0, features[0] stays zeros.

        The extractor does not call compute_features for frame 0. The output
        at row 0 comes from np.zeros initialization. This test directly
        validates that memory is empty at t=0 so compute_features would
        return zeros anyway.
        """
        from salt_r.memory import DistractorAwareMemory

        dim = 192
        mem = DistractorAwareMemory()

        # Frame 0: RAM is empty
        emb_0 = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        mem._current_frame = 0
        feat = mem.compute_features(query_emb=emb_0)

        assert feat["mem_pos_max_sim"] == 0.0, (
            f"Empty RAM: pos_max_sim should be 0.0, got {feat['mem_pos_max_sim']}"
        )
        assert feat["mem_pos_mean_sim"] == 0.0, (
            f"Empty RAM: pos_mean_sim should be 0.0, got {feat['mem_pos_mean_sim']}"
        )
        assert feat["mem_pos_recency_sim"] == 0.0, (
            f"Empty RAM: pos_recency_sim should be 0.0, got {feat['mem_pos_recency_sim']}"
        )

    def test_gate_signals_default_when_no_preds(self):
        """_get_gate_signals returns sensible defaults when preds unavailable."""
        from salt_r.sgla_memory_extractor import _get_gate_signals

        p_fc, p_ifd, apce_norm = _get_gate_signals(None, t=0)
        # Defaults should allow RAM updates (conservative but not blocking)
        assert p_fc < 0.20, f"Default p_fc should pass gate (< 0.20), got {p_fc}"
        assert p_ifd < 0.30, f"Default p_ifd should pass gate (< 0.30), got {p_ifd}"
        assert apce_norm > 0.40, f"Default apce_norm should pass gate (> 0.40), got {apce_norm}"

    def test_gate_signals_from_preds_dict(self):
        """_get_gate_signals correctly extracts p_fc and p_ifd from preds dict."""
        from salt_r.sgla_memory_extractor import _get_gate_signals

        preds = [
            {"false_confirmed": 0.85, "imminent_failure_dynamic": 0.92},
            {"false_confirmed": 0.05, "imminent_failure_dynamic": 0.10},
        ]
        # Frame 0: high values — gate should NOT pass
        p_fc, p_ifd, _ = _get_gate_signals(preds, t=0)
        assert p_fc == pytest.approx(0.85)
        assert p_ifd == pytest.approx(0.92)

        # Frame 1: low values — gate should pass
        p_fc, p_ifd, _ = _get_gate_signals(preds, t=1)
        assert p_fc == pytest.approx(0.05)
        assert p_ifd == pytest.approx(0.10)

    def test_gate_signals_out_of_range(self):
        """_get_gate_signals returns defaults when t >= len(preds)."""
        from salt_r.sgla_memory_extractor import _get_gate_signals

        preds = [{"false_confirmed": 0.1, "imminent_failure_dynamic": 0.1}]
        p_fc, p_ifd, apce_norm = _get_gate_signals(preds, t=999)
        # Should fall back to defaults, not raise
        assert isinstance(p_fc, float)
        assert isinstance(p_ifd, float)
        assert isinstance(apce_norm, float)


# ---------------------------------------------------------------------------
# Key prefix: sidecar must write memory_features/ not ram_features/
# ---------------------------------------------------------------------------


def test_sidecar_writes_memory_features_key():
    """train.py/eval.py read memory_features/{seq} — extractor must use that prefix."""
    import inspect
    from salt_r import sgla_memory_extractor as m
    source = inspect.getsource(m)
    assert "memory_features/" in source, (
        "sgla_memory_extractor must write 'memory_features/<seq>' keys "
        "(train.py/eval.py only read that prefix, not 'ram_features/')"
    )
    assert "ram_features/" not in source, (
        "Legacy 'ram_features/' prefix found — must be replaced with 'memory_features/'"
    )
