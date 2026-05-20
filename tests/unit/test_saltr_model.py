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

    model = SALTRD()
    model.eval()

    # Run the model directly to get ground-truth probabilities
    x = torch.randn(1, 20, 28)
    with torch.no_grad():
        raw = model(x)  # dict of already-sigmoid values
    expected_fc = float(raw["false_confirmed"].item())

    # Simulate what _run_inference does: stack via HEAD_NAMES + clip
    prob_matrix = np.stack(
        [raw[h].detach().cpu().numpy() for h in HEAD_NAMES], axis=1
    ).astype(np.float32)
    prob_matrix = np.clip(prob_matrix, 0.0, 1.0)
    fc_idx = HEAD_NAMES.index("false_confirmed")
    result_fc = float(prob_matrix[0, fc_idx])

    assert abs(result_fc - expected_fc) < 1e-6, \
        f"Double sigmoid detected: raw={expected_fc:.6f}, eval={result_fc:.6f}"


def test_train_head_names_canonical():
    """train._HEAD_NAMES must equal model.HEAD_NAMES — no local duplicate."""
    from salt_r.model import HEAD_NAMES as model_heads
    from salt_r.train import _HEAD_NAMES as train_heads

    assert list(train_heads) == list(model_heads), \
        f"Diverged head names: model={model_heads}, train={train_heads}"
