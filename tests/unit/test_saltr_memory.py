"""Unit tests for saltr/src/salt_r/memory.py and memory_features.py.

All tests use only numpy — no model weights, no NPZ files on disk needed.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_emb(dim: int = 28, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _make_entry(frame_idx: int = 0, dim: int = 28, seed: int = 0,
                source: str = "target_confident"):
    from salt_r.memory import MemoryEntry
    return MemoryEntry(
        embedding=_rand_emb(dim, seed),
        frame_idx=frame_idx,
        iou=0.8,
        apce_norm=0.6,
        p_fc=0.05,
        source=source,
    )


# ---------------------------------------------------------------------------
# 1. PositiveMemory: updates when p_fc low, apce high, interval met
# ---------------------------------------------------------------------------

class TestPositiveMemory:

    def test_updates_when_conditions_met(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=6, update_interval=5)
        # First frame is always eligible (interval from -999)
        assert pm.should_update(p_fc=0.05, p_ifd=0.10, apce_norm=0.7, current_frame=0)

    def test_does_not_update_when_p_fc_high(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=6, update_interval=5)
        assert not pm.should_update(p_fc=0.25, p_ifd=0.10, apce_norm=0.7, current_frame=0)

    def test_does_not_update_when_p_ifd_high(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=6, update_interval=5)
        assert not pm.should_update(p_fc=0.05, p_ifd=0.35, apce_norm=0.7, current_frame=0)

    def test_does_not_update_when_apce_low(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=6, update_interval=5)
        assert not pm.should_update(p_fc=0.05, p_ifd=0.10, apce_norm=0.3, current_frame=0)

    def test_does_not_update_within_interval(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=6, update_interval=5)
        # First update at frame 0
        assert pm.should_update(p_fc=0.05, p_ifd=0.10, apce_norm=0.7, current_frame=0)
        entry = _make_entry(frame_idx=0)
        pm.add(entry)
        # Frame 3 — interval not yet met
        assert not pm.should_update(p_fc=0.05, p_ifd=0.10, apce_norm=0.7, current_frame=3)
        # Frame 5 — exactly at interval boundary (5 frames elapsed)
        assert pm.should_update(p_fc=0.05, p_ifd=0.10, apce_norm=0.7, current_frame=5)

    def test_fifo_eviction(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=3, update_interval=1)
        for i in range(5):
            pm.add(_make_entry(frame_idx=i, seed=i))
        assert pm.size == 3

    def test_mean_similarity_zero_when_empty(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory()
        assert pm.mean_similarity(_rand_emb()) == 0.0

    def test_max_similarity_zero_when_empty(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory()
        assert pm.max_similarity(_rand_emb()) == 0.0

    def test_recency_weighted_sim_zero_when_empty(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory()
        assert pm.recency_weighted_similarity(_rand_emb()) == 0.0

    def test_max_ge_mean(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=6, update_interval=1)
        for i in range(4):
            pm.add(_make_entry(frame_idx=i, seed=i))
        q = _rand_emb(seed=99)
        assert pm.max_similarity(q) >= pm.mean_similarity(q) - 1e-6

    def test_recency_weights_more_recent(self):
        """Most recent entry should dominate recency-weighted sim."""
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=6, update_interval=1)
        # Add an entry that's very different from query
        old_entry = _make_entry(frame_idx=0, seed=0)
        pm.add(old_entry)
        # Add a recent entry that is identical to the query
        query = _rand_emb(seed=42)
        from salt_r.memory import MemoryEntry
        recent_entry = MemoryEntry(
            embedding=query.copy(),
            frame_idx=10,
            iou=0.9,
            apce_norm=0.7,
            p_fc=0.02,
            source="target_confident",
        )
        pm.add(recent_entry)
        rec_sim = pm.recency_weighted_similarity(query)
        mean_sim = pm.mean_similarity(query)
        # Recency-weighted should be higher since we weight the identical recent entry more
        assert rec_sim >= mean_sim - 1e-6

    def test_reset_clears_state(self):
        from salt_r.memory import PositiveMemory
        pm = PositiveMemory(max_slots=6, update_interval=1)
        for i in range(4):
            pm.add(_make_entry(frame_idx=i, seed=i))
        assert pm.size == 4
        pm.reset()
        assert pm.size == 0
        assert pm.mean_similarity(_rand_emb()) == 0.0


# ---------------------------------------------------------------------------
# 2. NegativeMemory: updates when distractor detected AND tracking reliable
# ---------------------------------------------------------------------------

class TestNegativeMemory:

    def test_updates_when_conditions_met(self):
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory(max_slots=6)
        assert nm.should_update(secondary_peak_ratio=0.8, p_fc=0.1, apce_norm=0.6)

    def test_no_update_when_secondary_peak_low(self):
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory(max_slots=6)
        assert not nm.should_update(secondary_peak_ratio=0.5, p_fc=0.1, apce_norm=0.6)

    def test_no_update_when_p_fc_high(self):
        """Distractor present but tracking unreliable — don't update DRM."""
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory(max_slots=6)
        assert not nm.should_update(secondary_peak_ratio=0.9, p_fc=0.3, apce_norm=0.6)

    def test_no_update_when_apce_low(self):
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory(max_slots=6)
        assert not nm.should_update(secondary_peak_ratio=0.9, p_fc=0.1, apce_norm=0.3)

    def test_not_evicted_by_age_timeless_prior(self):
        """DRM entries must NOT be evicted based on age — timeless prior test."""
        from salt_r.memory import NegativeMemory, MemoryEntry
        nm = NegativeMemory(max_slots=3)

        # Fill with 3 identical entries (seed=0)
        for i in range(3):
            e = _make_entry(frame_idx=i, seed=0)
            nm.add(e)
        assert nm.size == 3

        # Adding a 4th entry: eviction is by least-similar (NOT oldest).
        # All existing entries have same embedding as new one → all max similarity.
        # The new entry should be added (evicts one arbitrary, but oldest NOT guaranteed).
        new_entry = _make_entry(frame_idx=100, seed=0)  # identical embedding, very old age
        nm.add(new_entry)
        assert nm.size == 3

        # The frame_idx 100 entry (the "old" one by age interpretation) should still be in
        # memory because it's most similar to the new entry.
        # Actually since all seeds=0, all are identical → eviction can be any slot.
        # The key invariant: the OLDEST by frame_idx is NOT guaranteed eviction.
        # Let's verify: if we add a VERY dissimilar entry, THAT gets evicted, not oldest.
        from salt_r.memory import NegativeMemory as NM2, MemoryEntry as ME2
        nm2 = NM2(max_slots=2)
        # Add one very distinct entry (seed=0) and one identical-to-future entry (seed=42)
        emb_future = _rand_emb(seed=42)
        e1 = ME2(embedding=_rand_emb(seed=0), frame_idx=0, iou=0.0, apce_norm=0.5,
                  p_fc=0.1, source="secondary_peak")
        e2 = ME2(embedding=emb_future.copy(), frame_idx=1, iou=0.0, apce_norm=0.5,
                  p_fc=0.1, source="secondary_peak")
        nm2.add(e1)
        nm2.add(e2)
        # New entry is very similar to e2 (seed=42)
        new_e = ME2(embedding=emb_future.copy(), frame_idx=100, iou=0.0, apce_norm=0.5,
                    p_fc=0.1, source="secondary_peak")
        nm2.add(new_e)
        # e1 (least similar to new entry) should be evicted, NOT e2 (older by frame but similar)
        assert nm2.size == 2
        sims = [float(np.dot(e.embedding, emb_future)) for e in nm2._entries]
        # All remaining entries should be similar to emb_future (e1 was evicted)
        assert all(s > 0.5 for s in sims), f"Expected high-sim entries to survive: {sims}"

    def test_max_ge_mean(self):
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory(max_slots=6)
        for i in range(4):
            nm.add(_make_entry(frame_idx=i, seed=i, source="secondary_peak"))
        q = _rand_emb(seed=77)
        assert nm.max_similarity(q) >= nm.mean_similarity(q) - 1e-6

    def test_mean_similarity_zero_when_empty(self):
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory()
        assert nm.mean_similarity(_rand_emb()) == 0.0

    def test_max_similarity_zero_when_empty(self):
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory()
        assert nm.max_similarity(_rand_emb()) == 0.0

    def test_reset_clears_state(self):
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory(max_slots=6)
        for i in range(3):
            nm.add(_make_entry(frame_idx=i, seed=i))
        nm.reset()
        assert nm.size == 0

    def test_count_nearby_no_bbox(self):
        """count_nearby returns 0 when entries have no bbox."""
        from salt_r.memory import NegativeMemory
        nm = NegativeMemory(max_slots=6)
        for i in range(3):
            nm.add(_make_entry(frame_idx=i, seed=i))
        assert nm.count_nearby(np.array([10.0, 10.0, 50.0, 50.0])) == 0


# ---------------------------------------------------------------------------
# 3. DistractorAwareMemory.compute_features: all FEATURE_NAMES present
# ---------------------------------------------------------------------------

class TestDistractorAwareMemory:

    def test_compute_features_keys_complete(self):
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory()
        emb = _rand_emb()
        feats = mem.compute_features(emb)
        assert set(feats.keys()) == set(DistractorAwareMemory.FEATURE_NAMES), (
            f"Missing keys: {set(DistractorAwareMemory.FEATURE_NAMES) - set(feats.keys())}"
        )

    def test_compute_features_count(self):
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory()
        feats = mem.compute_features(_rand_emb())
        assert len(feats) == len(DistractorAwareMemory.FEATURE_NAMES)

    def test_feature_names_class_attr_length(self):
        from salt_r.memory import DistractorAwareMemory
        assert len(DistractorAwareMemory.FEATURE_NAMES) == 9

    def test_all_zeros_when_empty(self):
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory()
        feats = mem.compute_features(_rand_emb())
        for k in ("mem_pos_max_sim", "mem_pos_mean_sim", "mem_pos_recency_sim",
                  "mem_neg_max_sim", "mem_neg_mean_sim", "mem_neg_count_nearby",
                  "mem_neg_size"):
            assert feats[k] == 0.0, f"{k} should be 0.0 when memory empty, got {feats[k]}"

    def test_update_age_large_when_never_updated(self):
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory()
        feats = mem.compute_features(_rand_emb())
        assert feats["mem_update_age"] >= 999, (
            f"Expected large sentinel when never updated, got {feats['mem_update_age']}"
        )

    def test_reset_clears_everything(self):
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory()
        # Run a few steps to populate memory
        for t in range(10):
            emb = _rand_emb(seed=t)
            mem.step(frame_idx=t, embedding=emb, p_fc=0.05, p_ifd=0.10,
                     apce_norm=0.7, secondary_peak_ratio=0.8)
        assert mem.positive.size > 0
        mem.reset()
        feats = mem.compute_features(_rand_emb())
        assert feats["mem_pos_max_sim"] == 0.0
        assert feats["mem_neg_max_sim"] == 0.0
        assert mem.positive.size == 0
        assert mem.negative.size == 0

    def test_step_populates_positive_memory(self):
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory(pos_slots=6, update_interval=5)
        # Conditions are good: p_fc low, apce high, interval met
        mem.step(frame_idx=0, embedding=_rand_emb(seed=1), p_fc=0.05,
                 p_ifd=0.10, apce_norm=0.8)
        assert mem.positive.size == 1

    def test_step_populates_negative_memory(self):
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory()
        # Distractor detected AND tracking reliable
        mem.step(frame_idx=0, embedding=_rand_emb(seed=2), p_fc=0.10,
                 p_ifd=0.10, apce_norm=0.7, secondary_peak_ratio=0.9)
        assert mem.negative.size == 1

    def test_step_no_negative_update_without_distractor(self):
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory()
        # No distractor (secondary_peak_ratio = 0.0)
        mem.step(frame_idx=0, embedding=_rand_emb(), p_fc=0.05,
                 p_ifd=0.10, apce_norm=0.8, secondary_peak_ratio=0.0)
        assert mem.negative.size == 0


# ---------------------------------------------------------------------------
# 4. Target-minus-distractor margin: sign test
# ---------------------------------------------------------------------------

class TestDistractorMargin:

    def test_positive_margin_when_tracking_correct(self):
        """When tracker is locked onto the right object (high iou, low p_fc),
        the positive memory fills with target embeddings and the margin > 0."""
        from salt_r.memory import DistractorAwareMemory

        target_emb = _rand_emb(seed=1)
        distractor_emb = _rand_emb(seed=42)  # orthogonal to target

        mem = DistractorAwareMemory(pos_slots=6, neg_slots=6, update_interval=1)

        # Populate positive memory with target embeddings
        for t in range(6):
            mem.step(frame_idx=t, embedding=target_emb.copy(), p_fc=0.02,
                     p_ifd=0.05, apce_norm=0.8, secondary_peak_ratio=0.0)
        # Add one distractor entry to negative memory
        from salt_r.memory import MemoryEntry
        dist_entry = MemoryEntry(
            embedding=distractor_emb.copy(), frame_idx=99, iou=0.0,
            apce_norm=0.5, p_fc=0.1, source="secondary_peak"
        )
        mem.negative._entries.append(dist_entry)

        feats = mem.compute_features(query_emb=target_emb.copy())
        margin = feats["mem_target_minus_distractor_margin"]
        assert margin > 0, (
            f"Expected positive margin when tracking correct target, got {margin:.4f}"
        )

    def test_negative_margin_when_drifted(self):
        """When tracker has drifted to the distractor, querying with the distractor emb
        should show negative or zero margin (distractor similar to neg memory, not pos)."""
        from salt_r.memory import DistractorAwareMemory, MemoryEntry

        target_emb = _rand_emb(seed=1)
        distractor_emb = _rand_emb(seed=42)

        mem = DistractorAwareMemory(pos_slots=6, neg_slots=6, update_interval=1)

        # Populate positive memory with target embeddings
        for t in range(4):
            mem.step(frame_idx=t, embedding=target_emb.copy(), p_fc=0.02,
                     p_ifd=0.05, apce_norm=0.8, secondary_peak_ratio=0.0)

        # Inject distractor into negative memory directly
        for t in range(3):
            e = MemoryEntry(
                embedding=distractor_emb.copy(), frame_idx=100 + t, iou=0.0,
                apce_norm=0.5, p_fc=0.1, source="secondary_peak"
            )
            mem.negative._entries.append(e)

        # Query with distractor embedding (as if tracker drifted)
        feats = mem.compute_features(query_emb=distractor_emb.copy())
        margin = feats["mem_target_minus_distractor_margin"]
        # pos memory has target embeddings (dissimilar to distractor query)
        # neg memory has distractor embeddings (similar to distractor query)
        # So pos_mean - neg_max should be negative
        assert margin < 0, (
            f"Expected negative margin when queried with distractor embedding, got {margin:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Embedding fallback: 28-dim proxy works
# ---------------------------------------------------------------------------

class TestEmbeddingFallback:

    def test_proxy_embedding_is_normalized(self):
        from salt_r.memory_features import _make_proxy_embedding
        feat = np.random.randn(28).astype(np.float32)
        emb = _make_proxy_embedding(feat)
        assert emb.shape == (28,)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Proxy embedding not unit-norm: {norm}"

    def test_proxy_embedding_zero_vector(self):
        """Zero feature vector should produce a valid (near-zero-norm) embedding."""
        from salt_r.memory_features import _make_proxy_embedding
        feat = np.zeros(28, dtype=np.float32)
        emb = _make_proxy_embedding(feat)
        assert emb.shape == (28,)
        assert np.all(np.isfinite(emb))

    def test_compute_memory_features_for_sequence_shape(self):
        """compute_memory_features_for_sequence returns (T, 9) array."""
        from salt_r.memory_features import compute_memory_features_for_sequence, MEMORY_FEATURE_NAMES
        T = 50
        rng = np.random.default_rng(0)
        features = rng.standard_normal((T, 28)).astype(np.float32)
        features[:, 1] = np.clip(rng.standard_normal(T) * 0.3 + 0.5, 0, 1)  # apce_norm
        features[:, 6] = np.clip(rng.standard_normal(T) * 1 + 1, 0, 5)      # n_secondary
        labels = np.zeros((T, 14), dtype=np.int8)
        iou_trace = np.clip(rng.standard_normal(T) * 0.3 + 0.7, 0, 1).astype(np.float32)

        from salt_r.collect_features import LABEL_NAMES_V2
        result = compute_memory_features_for_sequence(
            features=features,
            labels=labels,
            iou_trace=iou_trace,
            preds=None,
            label_names=list(LABEL_NAMES_V2),
        )
        assert result.shape == (T, len(MEMORY_FEATURE_NAMES)), (
            f"Expected ({T}, {len(MEMORY_FEATURE_NAMES)}), got {result.shape}"
        )
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_compute_memory_features_with_preds(self):
        """When predictions are provided, p_fc/p_ifd are read from preds array."""
        from salt_r.memory_features import compute_memory_features_for_sequence
        T = 30
        rng = np.random.default_rng(1)
        features = rng.standard_normal((T, 28)).astype(np.float32)
        features[:, 1] = 0.6  # apce_norm = 0.6 (above threshold)
        labels = np.zeros((T, 14), dtype=np.int8)
        iou_trace = np.ones(T, dtype=np.float32) * 0.8
        # 13 prediction heads (all low probability → memory should update freely)
        preds = np.zeros((T, 13), dtype=np.float32)

        from salt_r.model import HEAD_NAMES_V2
        from salt_r.collect_features import LABEL_NAMES_V2
        result = compute_memory_features_for_sequence(
            features=features,
            labels=labels,
            iou_trace=iou_trace,
            preds=preds,
            label_names=list(LABEL_NAMES_V2),
            head_names=list(HEAD_NAMES_V2),
        )
        assert result.shape == (T, 9)
        assert np.all(np.isfinite(result))

    def test_memory_features_accumulate_over_time(self):
        """Memory features should become non-zero after warm-up period."""
        from salt_r.memory_features import compute_memory_features_for_sequence
        T = 40
        rng = np.random.default_rng(2)
        features = rng.standard_normal((T, 28)).astype(np.float32)
        # Set apce_norm high to trigger updates
        features[:, 1] = 0.7
        labels = np.zeros((T, 14), dtype=np.int8)
        iou_trace = np.ones(T, dtype=np.float32) * 0.85

        from salt_r.collect_features import LABEL_NAMES_V2
        result = compute_memory_features_for_sequence(
            features=features,
            labels=labels,
            iou_trace=iou_trace,
            preds=None,
            label_names=list(LABEL_NAMES_V2),
        )
        # After warmup, positive memory should have been updated, so pos_max_sim > 0
        pos_max_col = 0  # "mem_pos_max_sim" is index 0 in FEATURE_NAMES
        # By frame 10 (after at least 2 updates at interval=5), should be nonzero
        assert result[10:, pos_max_col].max() > 0, (
            "Expected positive memory to have nonzero similarity after warmup"
        )

    def test_memory_neg_size_feature(self):
        """mem_neg_size should reflect actual negative memory entries."""
        from salt_r.memory import DistractorAwareMemory
        mem = DistractorAwareMemory()
        from salt_r.memory import MemoryEntry
        # Directly add to negative memory
        for i in range(3):
            e = MemoryEntry(
                embedding=_rand_emb(seed=i), frame_idx=i, iou=0.0,
                apce_norm=0.5, p_fc=0.1, source="secondary_peak"
            )
            mem.negative._entries.append(e)

        feats = mem.compute_features(_rand_emb())
        assert feats["mem_neg_size"] == 3.0, (
            f"Expected mem_neg_size=3.0, got {feats['mem_neg_size']}"
        )


# ---------------------------------------------------------------------------
# 6. MEMORY_FEATURE_NAMES — structural guarantees
# ---------------------------------------------------------------------------

class TestMemoryFeatureNames:

    def test_memory_margin_col_exists(self):
        """mem_target_minus_distractor_margin must be a named feature in MEMORY_FEATURE_NAMES."""
        from salt_r.memory_features import MEMORY_FEATURE_NAMES
        assert "mem_target_minus_distractor_margin" in MEMORY_FEATURE_NAMES, (
            "Column 'mem_target_minus_distractor_margin' must exist in MEMORY_FEATURE_NAMES"
        )

    def test_memory_margin_col_index(self):
        """mem_target_minus_distractor_margin is at index 6 in MEMORY_FEATURE_NAMES.

        collect_memory_sidecar uses a hardcoded MARGIN_COL index — this test
        guards against accidental reordering of FEATURE_NAMES.
        """
        from salt_r.memory_features import MEMORY_FEATURE_NAMES
        idx = MEMORY_FEATURE_NAMES.index("mem_target_minus_distractor_margin")
        assert idx == 6, (
            f"Expected mem_target_minus_distractor_margin at index 6, found at {idx}. "
            f"Reordering FEATURE_NAMES breaks collect_memory_sidecar MARGIN_COL."
        )

    def test_memory_feature_names_length_is_9(self):
        """MEMORY_FEATURE_NAMES must have exactly 9 entries (matches (T, 9) output shape)."""
        from salt_r.memory_features import MEMORY_FEATURE_NAMES
        assert len(MEMORY_FEATURE_NAMES) == 9, (
            f"Expected 9 memory feature names, got {len(MEMORY_FEATURE_NAMES)}: {MEMORY_FEATURE_NAMES}"
        )

    def test_memory_features_oracle_vs_preds_differ(self):
        """Using oracle labels vs model preds should produce different memory feature arrays.

        When preds (all zeros = no risk) are provided, the memory updates more
        freely than when oracle labels mark some frames as failure, causing the
        accumulated similarity features to differ between the two runs.
        """
        from salt_r.memory_features import compute_memory_features_for_sequence
        from salt_r.collect_features import LABEL_NAMES_V2
        from salt_r.model import HEAD_NAMES_V2

        T = 40
        rng = np.random.default_rng(5)
        features = rng.standard_normal((T, 28)).astype(np.float32)
        features[:, 1] = 0.65   # apce_norm above update threshold
        iou_trace = np.ones(T, dtype=np.float32) * 0.85

        # Oracle labels: mark half frames as false_confirmed
        labels = np.zeros((T, 14), dtype=np.int8)
        fc_idx = list(LABEL_NAMES_V2).index("false_confirmed")
        labels[::2, fc_idx] = 1  # every other frame is a "failure"

        # Oracle mode (preds=None): uses label values as gate signals
        result_oracle = compute_memory_features_for_sequence(
            features=features, labels=labels, iou_trace=iou_trace,
            preds=None, label_names=list(LABEL_NAMES_V2),
        )

        # Pred mode: all-zero predictions (no risk ever) → gates always open
        preds = np.zeros((T, len(HEAD_NAMES_V2)), dtype=np.float32)
        result_preds = compute_memory_features_for_sequence(
            features=features, labels=labels, iou_trace=iou_trace,
            preds=preds, label_names=list(LABEL_NAMES_V2),
            head_names=list(HEAD_NAMES_V2),
        )

        # Both must have correct shape
        assert result_oracle.shape == (T, 9)
        assert result_preds.shape == (T, 9)

        # With oracle half-failure vs all-open preds, the update histories differ
        # → at least some feature values must differ by the end of the sequence
        assert not np.allclose(result_oracle[-1], result_preds[-1], atol=1e-5), (
            "Oracle labels vs all-zero preds should produce different memory features"
        )
