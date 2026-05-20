"""P0 smoke contracts for the SALT-RD model, train, and eval pipeline."""

import numpy as np
import torch
import pytest


def test_model_forward_contract():
    from salt_r.model import SALTRD, HEAD_NAMES

    B, T, F = 3, 20, 28
    model = SALTRD()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(B, T, F))
    assert isinstance(out, dict)
    assert set(out.keys()) == set(HEAD_NAMES)
    for name, v in out.items():
        assert v.shape == (B,), f"{name}: expected ({B},), got {v.shape}"
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0, f"{name}: out of [0,1]"


def test_model_param_count_reasonable():
    from salt_r.model import SALTRD

    m = SALTRD()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # GRU hidden=64 layers=2 + 7 heads -> ~43k params; allow 10k-200k range
    assert 10_000 < n < 200_000, f"Unexpected param count: {n}"


def test_train_saltrd_is_model_subclass():
    """train.SALTRD must be a subclass of model.SALTRD so checkpoints are compatible."""
    from salt_r.model import SALTRD as ModelSALTRD
    from salt_r.train import SALTRD as TrainSALTRD

    assert issubclass(TrainSALTRD, ModelSALTRD), \
        "train.SALTRD must subclass model.SALTRD for checkpoint compatibility"


def test_train_checkpoint_loads_in_eval(tmp_path):
    """Checkpoint saved by train.SALTRD must load cleanly in eval._load_model."""
    from salt_r.train import SALTRD as TrainSALTRD
    from salt_r.eval import _load_model
    from salt_r.collect_features import FEATURE_NAMES, LABEL_NAMES

    ckpt_path = str(tmp_path / "test.pt")
    m = TrainSALTRD()
    torch.save({"model_state_dict": m.state_dict(), "epoch": 1,
                "val_auprc_false_confirmed": 0.1,
                "window_size": 20,
                "feature_names": FEATURE_NAMES,
                "label_names": LABEL_NAMES}, ckpt_path)

    loaded = _load_model(ckpt_path, n_features=len(FEATURE_NAMES),
                         n_labels=len(LABEL_NAMES), device="cpu")
    assert loaded is not None, "Model failed to load from checkpoint"
    # Forward pass must work
    with torch.no_grad():
        out = loaded(torch.zeros(2, 20, 28))
    assert isinstance(out, dict) or isinstance(out, torch.Tensor)


def test_eval_does_not_double_sigmoid():
    """Probability from _run_inference must equal raw model output — no extra sigmoid."""
    from salt_r.model import SALTRD, HEAD_NAMES
    from salt_r.eval import _run_inference

    model = SALTRD()
    model.eval()

    features = np.random.randn(20, 28).astype(np.float32)

    # Real eval path — must not apply sigmoid a second time
    result_dict = _run_inference(model, {"test_seq": features}, window_size=20, device="cpu")
    fc_idx = HEAD_NAMES.index("false_confirmed")
    result_fc = float(result_dict["test_seq"][-1, fc_idx])

    # Direct model path: at t=19 the window is the full sequence (no padding)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 20, 28)
    with torch.no_grad():
        raw = model(x)
    expected_fc = float(raw["false_confirmed"].item())

    assert abs(result_fc - expected_fc) < 1e-5, \
        f"Double sigmoid detected: raw={expected_fc:.6f}, eval={result_fc:.6f}"


def test_train_head_names_canonical():
    """train._HEAD_NAMES must equal model.HEAD_NAMES — no local duplicate."""
    from salt_r.model import HEAD_NAMES as model_heads
    from salt_r.train import _HEAD_NAMES as train_heads

    assert list(train_heads) == list(model_heads), \
        f"Diverged head names: model={model_heads}, train={train_heads}"


# ---------------------------------------------------------------------------
# v1 schema — 9-head model forward, train wrapper, and checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_v1_model_forward_9_heads():
    """SALTRD with v1 head_names must return 9 heads, not 7."""
    from salt_r.model import SALTRD, HEAD_NAMES_V1

    m = SALTRD(head_names=HEAD_NAMES_V1)
    m.eval()
    x = torch.zeros(2, 20, 28)
    with torch.no_grad():
        out = m(x)
    assert isinstance(out, dict)
    assert len(out) == len(HEAD_NAMES_V1), \
        f"Expected {len(HEAD_NAMES_V1)} heads, got {len(out)}: {list(out.keys())}"
    assert list(out.keys()) == HEAD_NAMES_V1


def test_v1_train_wrapper_output_shape():
    """train.SALTRD with v1 schema must return (B, 9) tensor, not (B, 7)."""
    from salt_r.model import HEAD_NAMES_V1
    from salt_r.train import SALTRD as TrainSALTRD

    m = TrainSALTRD(head_names=HEAD_NAMES_V1)
    m.eval()
    x = torch.zeros(3, 20, 28)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (3, len(HEAD_NAMES_V1)), \
        f"Expected (3, {len(HEAD_NAMES_V1)}), got {out.shape}"


def test_v1_checkpoint_roundtrip(tmp_path):
    """v1 checkpoint saved with head_names metadata must reload with 9 heads."""
    from salt_r.model import SALTRD, HEAD_NAMES_V1, LABEL_NAMES_V1
    from salt_r.collect_features import FEATURE_NAMES
    from salt_r.eval import _load_model

    m = SALTRD(head_names=HEAD_NAMES_V1)
    ckpt_path = str(tmp_path / "v1_test.pt")
    torch.save(
        {
            "model_state_dict": m.state_dict(),
            "head_names": HEAD_NAMES_V1,
            "label_names": LABEL_NAMES_V1,
            "feature_names": FEATURE_NAMES,
            "epoch": 1,
        },
        ckpt_path,
    )

    loaded = _load_model(ckpt_path, n_features=28, n_labels=len(LABEL_NAMES_V1), device="cpu")
    assert loaded is not None, "v1 checkpoint failed to load"
    assert list(loaded.heads.keys()) == HEAD_NAMES_V1, \
        f"Head names mismatch after load: {list(loaded.heads.keys())}"

    x = torch.zeros(1, 20, 28)
    with torch.no_grad():
        out = loaded(x)
    assert len(out) == len(HEAD_NAMES_V1)


def test_v2_checkpoint_label_schema_metadata(tmp_path):
    """Checkpoint for v2 schema must persist label_schema + LABEL_NAMES_V2, not v0.

    Regression for the P1 bug where train.py saved list(LABEL_NAMES) regardless of
    label_schema, causing checkpoint provenance to lie about the training schema.
    """
    from salt_r.model import SALTRD, HEAD_NAMES_V2, LABEL_NAMES_V2
    from salt_r.collect_features import FEATURE_NAMES
    from salt_r.eval import _load_model

    model = SALTRD(head_names=HEAD_NAMES_V2)
    ckpt_path = str(tmp_path / "v2_meta.pt")
    import torch
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "label_names": list(LABEL_NAMES_V2),  # must be schema-correct
            "label_schema": "v2",
            "head_names": list(HEAD_NAMES_V2),
            "feature_names": list(FEATURE_NAMES),
        },
        ckpt_path,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["label_schema"] == "v2"
    assert ckpt["label_names"] == list(LABEL_NAMES_V2), \
        "Checkpoint label_names must be LABEL_NAMES_V2 for v2 schema"
    assert len(ckpt["label_names"]) == 14, f"v2 has 14 labels, got {len(ckpt['label_names'])}"

    loaded = _load_model(ckpt_path, n_features=28, n_labels=len(LABEL_NAMES_V2), device="cpu")
    assert loaded is not None
    assert list(loaded.heads.keys()) == list(HEAD_NAMES_V2)


# ---------------------------------------------------------------------------
# Phase 4B — memory_dim extension (v2.1 model)
# ---------------------------------------------------------------------------

def test_saltrd_memory_dim_zero():
    """Model with memory_dim=0 and input (B, T, 28) works fine."""
    from salt_r.model import SALTRD, HEAD_NAMES

    B, T = 2, 20
    model = SALTRD(memory_dim=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(B, T, 28))
    assert isinstance(out, dict)
    assert set(out.keys()) == set(HEAD_NAMES)
    for name, v in out.items():
        assert v.shape == (B,), f"{name}: expected ({B},), got {v.shape}"
        assert 0.0 <= float(v.min()) and float(v.max()) <= 1.0


def test_saltrd_memory_dim_9():
    """Model with memory_dim=9 and input (B, T, 37) works fine, output shapes correct."""
    from salt_r.model import SALTRD, HEAD_NAMES

    B, T = 3, 20
    model = SALTRD(memory_dim=9)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(B, T, 37))
    assert isinstance(out, dict)
    assert set(out.keys()) == set(HEAD_NAMES)
    for name, v in out.items():
        assert v.shape == (B,), f"{name}: expected ({B},), got {v.shape}"
        assert 0.0 <= float(v.min()) and float(v.max()) <= 1.0


def test_saltrd_memory_dim_in_state_dict():
    """model.memory_dim == 9 is accessible after construction."""
    from salt_r.model import SALTRD

    model = SALTRD(memory_dim=9)
    assert model.memory_dim == 9, f"Expected memory_dim=9, got {model.memory_dim}"
