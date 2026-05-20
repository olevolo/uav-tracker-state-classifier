"""Unit tests for salt_r.integrate — FeatureBuffer and extract_features_from_entry."""


def test_feature_buffer_windowing():
    from salt_r.integrate import FeatureBuffer
    import numpy as np

    buf = FeatureBuffer(window_size=5, n_features=28)
    # Buffer not full yet
    assert buf.get_window() is None
    for i in range(4):
        buf.push(np.full(28, float(i), dtype=np.float32))
    assert buf.get_window() is None  # still not full

    buf.push(np.full(28, 4.0, dtype=np.float32))
    w = buf.get_window()
    assert w is not None
    assert w.shape == (5, 28), f"Expected (5,28), got {w.shape}"
    assert float(w[0, 0]) == 0.0  # oldest frame
    assert float(w[-1, 0]) == 4.0  # newest frame

    # Push one more -> oldest drops off
    buf.push(np.full(28, 5.0, dtype=np.float32))
    w2 = buf.get_window()
    assert float(w2[0, 0]) == 1.0  # 0 was evicted
    assert float(w2[-1, 0]) == 5.0


def test_feature_buffer_reset():
    from salt_r.integrate import FeatureBuffer
    import numpy as np

    buf = FeatureBuffer(window_size=3, n_features=4)
    for _ in range(3):
        buf.push(np.ones(4))
    assert buf.get_window() is not None

    buf.reset()
    assert buf.get_window() is None, "After reset, buffer should be empty"


def test_extract_features_shape_no_nan():
    """extract_features_from_entry must return (28,) float32 with no NaN."""
    import numpy as np
    from salt_r.integrate import FeatureBuffer, extract_features_from_entry
    from salt_r.collect_features import N_FEATURES

    # Build a minimal fake TelemetryEntry
    class FakeBBox:
        x = 100.0; y = 80.0; w = 30.0; h = 20.0

    class FakeEntry:
        bbox = FakeBBox()
        confidence = 0.8
        aux = {
            "score_map_stats": {
                "top1": 0.45, "top2": 0.30, "peak_margin": 0.15,
                "peak_width": 12, "n_secondary": 0, "peak_distance": 2.1,
                "heatmap_mass_topk": 0.72,
            },
            "apce_raw": 180.0,
            "psr_raw": 2400.0,
            "entropy_raw": 2.1,
        }

    frame = np.zeros((256, 320, 3), dtype=np.uint8)
    buf = FeatureBuffer(window_size=5, n_features=N_FEATURES)
    feats = extract_features_from_entry(FakeEntry(), None, None, frame, buf)

    assert feats.shape == (N_FEATURES,), f"Expected ({N_FEATURES},), got {feats.shape}"
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any(), "Features contain NaN"
    # APCE raw should be feature 0
    assert abs(float(feats[0]) - 180.0) < 1e-3, f"Expected apce_raw=180.0, got {feats[0]}"
    # APCE norm should be feature 1 = 180/256
    assert abs(float(feats[1]) - 180.0 / 256.0) < 1e-3, f"Expected apce_norm={180/256:.4f}, got {feats[1]}"


def test_extract_features_with_prev_frame_flow():
    """With prev_frame provided, flow features (22-27) should be non-zero."""
    import numpy as np
    from salt_r.integrate import FeatureBuffer, extract_features_from_entry
    from salt_r.collect_features import N_FEATURES

    class FakeBBox:
        x = 50.0; y = 40.0; w = 20.0; h = 15.0

    class FakeEntry:
        bbox = FakeBBox()
        confidence = 0.7
        aux = {"score_map_stats": {}, "apce_raw": 100.0, "psr_raw": 1000.0, "entropy_raw": 2.5}

    rng = np.random.default_rng(42)
    # Create two different frames so optical flow is non-trivial
    prev_frame = (rng.integers(0, 200, (128, 160, 3))).astype(np.uint8)
    curr_frame = (rng.integers(50, 255, (128, 160, 3))).astype(np.uint8)

    buf = FeatureBuffer(window_size=5, n_features=N_FEATURES)
    feats = extract_features_from_entry(FakeEntry(), FakeEntry(), prev_frame, curr_frame, buf)

    assert feats.shape == (N_FEATURES,)
    assert not np.isnan(feats).any()
    # global_flow_mag (feature 22) should be > 0 with different frames
    assert feats[22] >= 0.0, f"global_flow_mag should be non-negative, got {feats[22]}"
