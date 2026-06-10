"""Factorized 2-tower CSC-v4 diagnosis model (controlled CHALLENGER to V3-prod).

Motivation
----------
In the V4 4-state taxonomy ``{CC, CU, LA, FC}`` the two semantic axes are:

  * **off_target** — is the box on the wrong thing?  ``{LA, FC}`` are off-target.
  * **confirmed**  — is the tracker internally confident?  ``{CC, FC}`` are confirmed.

The hard, safety-critical ``false_confirmed`` (FC) cell is exactly the
*intersection* ``off_target × confirmed``.  A single joint head over all 41
features is free to take a shortcut: it can satisfy its loss on the easy axis
(confidence) and largely abandon the hard FC-vs-CC **geometry** signal, because
CC dominates the data 77%-to-1.6%.  A broken joint 41-feature TCN landed at
FC-vs-ALL val AUROC ~0.485 — i.e. *worse than chance* on the very state it
exists to find.

This model removes that degree of freedom by construction:

  * a **geometry tower** sees ONLY box-dynamics features and predicts the
    ``off_target`` axis;
  * a **response tower** sees ONLY score-map / confidence features and predicts
    the ``confirmed`` axis;
  * the two towers share NO layers, NO encoder, NO parameters.

The 4-state distribution is *composed* from the two independent axis
probabilities (FC = off × conf), so FC can only be predicted when the geometry
tower says "off-target" AND the response tower says "confident".  Neither tower
can paper over the other.

Towers are deliberately SMALL (hidden 32, 3 levels) — the strict feature
isolation, not capacity, is the whole point.

Contract
--------
``FactorizedCSC.__init__(geom_dim, resp_dim, hidden=32, levels=3, kernel=3,
dropout=0.1)`` builds two independent causal dilated-Conv1d TCN towers
(dilations ``1, 2, 4, ...`` for ``levels`` blocks).

``forward(geom, resp, last_step_only=True)`` -> ``{"off_logit", "conf_logit"}``;
``geom`` is ``(B, T, geom_dim)``, ``resp`` is ``(B, T, resp_dim)``; with
``last_step_only`` each logit is ``(B,)`` (the causal last-step decision).

``set_temperatures(t_off, t_conf)`` stores per-axis temperatures.

``compose(off_logit, conf_logit)`` -> calibrated axis + 4-state probabilities::

    p_off  = sigmoid(off_logit  / t_off)
    p_conf = sigmoid(conf_logit / t_conf)
    p_fc   = p_off       * p_conf
    p_cc   = (1 - p_off) * p_conf
    p_cu   = (1 - p_off) * (1 - p_conf)
    p_la   = p_off       * (1 - p_conf)

(the four state probs sum to 1 by construction).

This file mirrors V3's causal-TCN building blocks (kernel 3, dilations
``[1, 2, 4, ...]``, residual + LayerNorm, left-only padding so ``output[b, t]``
depends only on ``input[b, :t+1]``) but is otherwise fully self-contained and
does not import or touch any V3 / other-V4 model code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FactorizedCSC"]


# ---------------------------------------------------------------------------
# Causal TCN building blocks (mirror csc_lib/csc/v4/model_v4.py CSCv4 encoder:
# kernel 3, dilations [1,2,4,...], residual + LayerNorm, strictly causal)
# ---------------------------------------------------------------------------


class _CausalConv1d(nn.Module):
    """Causal 1-D convolution — left-only padding, no future leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    """Residual causal block: two causal convs + LayerNorm + ReLU + dropout."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        # LayerNorm operates over channels, so permute around it.
        h = self.act(self.norm1(self.conv1(x).permute(0, 2, 1)).permute(0, 2, 1))
        h = self.drop(h)
        h = self.norm2(self.conv2(h).permute(0, 2, 1)).permute(0, 2, 1)
        return self.drop(self.act(h + x))


class _Tower(nn.Module):
    """One independent causal-TCN tower: proj -> N residual blocks -> scalar logit.

    Produces a single binary logit per time step (last-step when requested).
    Self-contained — no parameters are shared with any other tower.
    """

    def __init__(self, in_dim: int, hidden: int, levels: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.levels = int(levels)
        dilations = [2 ** i for i in range(self.levels)]  # 1, 2, 4, ...
        self.dilations = dilations

        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [_TCNBlock(hidden, kernel, d, dropout) for d in dilations]
        )
        self.head = nn.Linear(hidden, 1)  # binary axis logit

    def forward(self, x: torch.Tensor, last_step_only: bool = True) -> torch.Tensor:
        """``x``: (B, T, in_dim) -> logit (B,) if last_step_only else (B, T)."""
        h = self.proj(x)            # (B, T, hidden)
        h = h.permute(0, 2, 1)      # (B, hidden, T) for Conv1d
        for block in self.blocks:
            h = block(h)
        h = h.permute(0, 2, 1)      # (B, T, hidden)
        logit = self.head(h).squeeze(-1)  # (B, T)
        if last_step_only:
            return logit[:, -1]     # (B,)
        return logit                # (B, T)


# ---------------------------------------------------------------------------
# FactorizedCSC — two independent towers + factorized 4-state composition
# ---------------------------------------------------------------------------


class FactorizedCSC(nn.Module):
    """Two strictly isolated causal-TCN towers feeding a factorized 4-state head.

    Args:
        geom_dim: width of the geometry (off_target) feature group.
        resp_dim: width of the response (confirmed) feature group.
        hidden:   per-tower TCN channel width (default 32 — kept small on purpose).
        levels:   residual blocks per tower; dilations ``1, 2, 4, ...`` (default 3).
        kernel:   causal-conv kernel size (default 3).
        dropout:  dropout inside each TCN block (default 0.1).

    The geometry tower NEVER sees response features and vice versa; the towers
    share no parameters. The 4-state distribution is composed, not learned by a
    joint head — see :meth:`compose`.
    """

    def __init__(
        self,
        geom_dim: int,
        resp_dim: int,
        hidden: int = 32,
        levels: int = 3,
        kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.geom_dim = int(geom_dim)
        self.resp_dim = int(resp_dim)
        self.hidden = int(hidden)
        self.levels = int(levels)
        self.kernel = int(kernel)

        # off_target axis tower (geometry features only)
        self.geom_tower = _Tower(self.geom_dim, hidden, levels, kernel, dropout)
        # confirmed axis tower (response features only)
        self.resp_tower = _Tower(self.resp_dim, hidden, levels, kernel, dropout)

        # Calibration temperatures (per axis). Stored as buffers so they
        # serialize with the checkpoint and move with .to(device). Default 1.0.
        self.register_buffer("t_off", torch.tensor(1.0))
        self.register_buffer("t_conf", torch.tensor(1.0))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- forward (raw axis logits) ----------------------------------------
    def forward(
        self,
        geom: torch.Tensor,
        resp: torch.Tensor,
        last_step_only: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return raw (uncalibrated) per-axis logits.

        Args:
            geom: (B, T, geom_dim) geometry features.
            resp: (B, T, resp_dim) response features.
            last_step_only: if True each logit is (B,); else (B, T).

        Returns:
            ``{"off_logit", "conf_logit"}``.
        """
        return {
            "off_logit": self.geom_tower(geom, last_step_only=last_step_only),
            "conf_logit": self.resp_tower(resp, last_step_only=last_step_only),
        }

    # ---- temperature handling --------------------------------------------
    def set_temperatures(self, t_off: float, t_conf: float) -> None:
        """Store post-hoc calibration temperatures (one per axis)."""
        dev = self.t_off.device
        self.t_off = torch.as_tensor(float(t_off), dtype=torch.float32, device=dev)
        self.t_conf = torch.as_tensor(float(t_conf), dtype=torch.float32, device=dev)

    # ---- factorized composition ------------------------------------------
    def compose(self, off_logit: torch.Tensor, conf_logit: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compose calibrated axis probs + the 4-state distribution.

        ``p_off`` / ``p_conf`` use the stored temperatures. The four state
        probabilities are the outer product of the two independent axes and sum
        to 1 by construction::

            p_fc = p_off*p_conf   p_cc = (1-p_off)*p_conf
            p_la = p_off*(1-p_conf)  p_cu = (1-p_off)*(1-p_conf)

        Args:
            off_logit / conf_logit: raw logits from :meth:`forward` (any shape;
                broadcasting follows torch rules).

        Returns:
            ``{"p_off","p_conf","p_cc","p_cu","p_la","p_fc"}``.
        """
        p_off = torch.sigmoid(off_logit / self.t_off)
        p_conf = torch.sigmoid(conf_logit / self.t_conf)
        return {
            "p_off": p_off,
            "p_conf": p_conf,
            "p_cc": (1.0 - p_off) * p_conf,
            "p_cu": (1.0 - p_off) * (1.0 - p_conf),
            "p_la": p_off * (1.0 - p_conf),
            "p_fc": p_off * p_conf,
        }


# ---------------------------------------------------------------------------
# Standalone smoke (CPU-only, no datasets) — run: python -m csc_lib.csc.v4.model_v4_factorized
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, GD, RD = 4, 32, 7, 10
    model = FactorizedCSC(geom_dim=GD, resp_dim=RD, hidden=32, levels=3, kernel=3, dropout=0.1)
    print(f"[smoke] FactorizedCSC(geom={GD}, resp={RD}) params={model.num_params:,}")

    geom = torch.randn(B, T, GD)
    resp = torch.randn(B, T, RD)

    # --- isolation check: towers must NOT share parameters ---
    geom_ids = {id(p) for p in model.geom_tower.parameters()}
    resp_ids = {id(p) for p in model.resp_tower.parameters()}
    assert geom_ids.isdisjoint(resp_ids), "towers share parameters — isolation broken!"
    print("[smoke] tower parameter isolation: OK (disjoint param sets)")

    # --- forward last_step_only ---
    model.eval()
    out = model.forward(geom, resp, last_step_only=True)
    assert set(out) == {"off_logit", "conf_logit"}, out.keys()
    assert out["off_logit"].shape == (B,), out["off_logit"].shape
    assert out["conf_logit"].shape == (B,), out["conf_logit"].shape
    print(f"[smoke] forward last_step_only off_logit{tuple(out['off_logit'].shape)} "
          f"conf_logit{tuple(out['conf_logit'].shape)}")

    # --- forward full sequence ---
    out_seq = model.forward(geom, resp, last_step_only=False)
    assert out_seq["off_logit"].shape == (B, T), out_seq["off_logit"].shape
    assert out_seq["conf_logit"].shape == (B, T), out_seq["conf_logit"].shape

    # --- causality check: perturbing future frames must not change last-step logit ---
    with torch.no_grad():
        base = model.forward(geom, resp, last_step_only=False)
        g2 = geom.clone(); g2[:, : T // 2] += 7.0  # perturb only the PAST half
        pert = model.forward(g2, resp, last_step_only=False)
        # last step depends on the (now-changed) past -> should differ
        assert not torch.allclose(base["off_logit"][:, -1], pert["off_logit"][:, -1]), \
            "last step should depend on the past"
        g3 = geom.clone(); g3[:, T // 2 + 1 :] += 7.0  # perturb only the strictly-FUTURE part of t=T//2
        base_mid = base["off_logit"][:, T // 2]
        pert_mid = model.forward(g3, resp, last_step_only=False)["off_logit"][:, T // 2]
        assert torch.allclose(base_mid, pert_mid, atol=1e-5), \
            "future leakage detected — non-causal!"
    print("[smoke] causality: OK (no future leakage)")

    # --- temperatures + compose ---
    model.set_temperatures(1.7, 0.8)
    assert abs(float(model.t_off) - 1.7) < 1e-6 and abs(float(model.t_conf) - 0.8) < 1e-6
    comp = model.compose(out["off_logit"], out["conf_logit"])
    assert set(comp) == {"p_off", "p_conf", "p_cc", "p_cu", "p_la", "p_fc"}, comp.keys()
    four = comp["p_cc"] + comp["p_cu"] + comp["p_la"] + comp["p_fc"]
    assert torch.allclose(four, torch.ones_like(four), atol=1e-5), f"4-state sum != 1: {four}"
    for k in ("p_off", "p_conf", "p_cc", "p_cu", "p_la", "p_fc"):
        assert ((comp[k] >= 0) & (comp[k] <= 1)).all(), f"{k} out of [0,1]"
    print(f"[smoke] compose: OK (4-state probs sum to 1; t_off={float(model.t_off)} "
          f"t_conf={float(model.t_conf)})")

    # --- temperatures survive a state_dict round-trip ---
    sd = model.state_dict()
    m2 = FactorizedCSC(geom_dim=GD, resp_dim=RD, hidden=32, levels=3, kernel=3, dropout=0.1)
    m2.load_state_dict(sd)
    assert abs(float(m2.t_off) - 1.7) < 1e-6 and abs(float(m2.t_conf) - 0.8) < 1e-6, \
        "temperatures did not survive state_dict round-trip"
    print("[smoke] state_dict round-trip preserves temperatures: OK")

    print("[smoke] OK")
