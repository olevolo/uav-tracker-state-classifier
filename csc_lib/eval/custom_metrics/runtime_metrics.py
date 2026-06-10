"""Runtime / FPS / latency metrics."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn


def latency_summary(latencies_ms: np.ndarray) -> dict[str, float]:
    """Mean / median / p90/95/99 / total over an array of per-frame
    latencies in milliseconds."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        return {
            "n_frames": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "total_s": 0.0,
            "mean_fps": 0.0,
            "median_fps": 0.0,
        }
    return {
        "n_frames": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "total_s": float(arr.sum() / 1000.0),
        "mean_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        "median_fps": float(1000.0 / np.median(arr)) if np.median(arr) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# New public helpers (extend without breaking the original API)
# ---------------------------------------------------------------------------


def summarise_latencies(latencies_ms: list[float]) -> dict:
    """Return {mean_fps, median_fps, p50_ms, p95_ms, p99_ms, total_s, n}.

    Parameters
    ----------
    latencies_ms:
        Per-frame latency measurements in milliseconds.
    """
    arr = np.asarray(latencies_ms, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean_fps": 0.0,
            "median_fps": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "total_s": 0.0,
        }
    mean_ms = float(arr.mean())
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    return {
        "n": n,
        "mean_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "median_fps": 1000.0 / p50 if p50 > 0 else 0.0,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "total_s": float(arr.sum() / 1000.0),
    }


def combined_pipeline_stats(
    tracker_ms: list[float],
    csc_ms: list[float],
) -> dict:
    """Per-stage + combined pipeline latency statistics.

    Returns a dict with keys ``"tracker"``, ``"csc"``, and ``"total"``,
    each containing the output of :func:`summarise_latencies`.  The
    ``"total"`` block is computed element-wise (tracker_ms[i] + csc_ms[i]).

    Parameters
    ----------
    tracker_ms, csc_ms:
        Per-frame latency lists of the same length.
    """
    n = min(len(tracker_ms), len(csc_ms))
    t_arr = np.asarray(tracker_ms[:n], dtype=np.float64)
    c_arr = np.asarray(csc_ms[:n], dtype=np.float64)
    total_arr = t_arr + c_arr
    return {
        "tracker": summarise_latencies(t_arr.tolist()),
        "csc": summarise_latencies(c_arr.tolist()),
        "total": summarise_latencies(total_arr.tolist()),
    }


def count_csc_flops(model: "nn.Module", *, window: int, feature_dim: int) -> int:
    """Estimate total FLOPs for one forward pass of a CSC model.

    Uses a shape-trace via ``register_forward_pre_hook`` / ``register_forward_hook``
    to record input and output shapes for every leaf module, then computes
    MAC-based FLOPs using standard formulas:

    - ``nn.Linear``:   2 * in_features * out_features  per token (time-step)
    - ``nn.Conv1d``:   2 * in_C * out_C * kernel_size * out_T  (MACs×2)
    - ``nn.GRU``:      per time-step: 6 * hidden * (input + hidden);
                       total: times ``window``

    All other module types are ignored (LayerNorm, Dropout, ReLU are
    parameter-free or compute-negligible relative to linear ops).

    Parameters
    ----------
    model:
        Any of ``CSCGRU``, ``CSCMLP``, ``CSCTCN`` (or a custom nn.Module).
    window:
        Temporal window size T — the T dimension of the ``(B=1, T, F)`` input.
    feature_dim:
        Feature width F — the F dimension of the ``(B=1, T, F)`` input.

    Returns
    -------
    int
        Total estimated FLOPs for one forward pass with ``B=1``.
    """
    import torch
    import torch.nn as nn_mod

    model = model.eval()

    # We'll accumulate FLOPs via hooks recorded in this mutable container.
    flop_log: list[int] = []
    hooks: list = []

    def _make_linear_hook(m):
        def _hook(module, inp, out):
            # inp[0]: (B, T_or_BT, in_features) or (B, in_features)
            x = inp[0]
            if x.dim() == 3:
                tokens = x.shape[0] * x.shape[1]  # B * T
            elif x.dim() == 2:
                tokens = x.shape[0]                # B*T flat
            else:
                tokens = 1
            flops = 2 * module.in_features * module.out_features * tokens
            flop_log.append(flops)
        return _hook

    def _make_conv1d_hook(m):
        def _hook(module, inp, out):
            # out: (B, out_C, out_T)
            B_val = out.shape[0]
            out_C = out.shape[1]
            out_T = out.shape[2]
            in_C = module.in_channels
            k = module.kernel_size[0]
            # MACs = out_C * in_C/groups * k * out_T * B  → FLOPs = MACs * 2
            groups = module.groups
            flops = 2 * (in_C // groups) * out_C * k * out_T * B_val
            flop_log.append(flops)
        return _hook

    def _make_gru_hook(m):
        def _hook(module, inp, out):
            # inp[0]: (B, T, input_size) if batch_first
            # Estimate: per time-step per layer:
            #   3 gates × (2 * input_size * hidden_size) +
            #   3 gates × (2 * hidden_size * hidden_size)
            #   = 6 * hidden_size * (input_size + hidden_size)
            # Then × T (time-steps) × num_layers × directions
            x = inp[0]
            if x.dim() == 3:
                T_val = x.shape[1] if module.batch_first else x.shape[0]
                B_val = x.shape[0] if module.batch_first else x.shape[1]
            else:
                T_val = 1
                B_val = 1
            hs = module.hidden_size
            is_ = module.input_size
            num_directions = 2 if module.bidirectional else 1
            flops_per_step = 6 * hs * (is_ + hs) * num_directions
            flops = flops_per_step * T_val * module.num_layers * B_val
            flop_log.append(flops)
        return _hook

    # Register hooks on leaf modules only
    for name, m in model.named_modules():
        if isinstance(m, nn_mod.Linear):
            hooks.append(m.register_forward_hook(_make_linear_hook(m)))
        elif isinstance(m, nn_mod.Conv1d):
            hooks.append(m.register_forward_hook(_make_conv1d_hook(m)))
        elif isinstance(m, nn_mod.GRU):
            hooks.append(m.register_forward_hook(_make_gru_hook(m)))

    try:
        x = torch.zeros(1, window, feature_dim)
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()

    return int(sum(flop_log))
