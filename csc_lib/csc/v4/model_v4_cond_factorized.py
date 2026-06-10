"""CSC-v4 CONDITIONAL factorized diagnosis model — the controlled rematch.

Fixes the architectural error of the symmetric 2-tower model
(``model_v4_factorized.FactorizedCSC``): a SINGLE ``confirmed`` head wrongly
assumes the confidence boundary is the same for on-target and off-target frames.
It is not — CC-vs-CU and FC-vs-LA need DIFFERENT confidence boundaries (that is
why a composed loss helped FC-vs-CC but wrecked LA).

CONDITIONAL hierarchical factorization (keeps anti-shortcut, drops the bad
conditional-independence assumption):

    geometry/localization tower ──> off_logit            -> p_off
    response tower (shared)     ──> conf_on_logit         -> p_conf_on  (CC vs CU | on-target)
                                └─> conf_off_logit        -> p_conf_off (FC vs LA | off-target)

    P(CC) = (1 - p_off) * p_conf_on
    P(CU) = (1 - p_off) * (1 - p_conf_on)
    P(FC) =      p_off  * p_conf_off
    P(LA) =      p_off  * (1 - p_conf_off)      (the four sum to 1 by construction)

Hard tower isolation is preserved: the geometry tower NEVER sees response
features and vice-versa. The two response heads share ONE response encoder (the
score-map/confidence signal is the same; only the decision boundary differs by
target status), so capacity stays comparable to a single-confirmed-head model.

Per-axis temperature scaling (THREE temperatures now: t_off, t_conf_on,
t_conf_off) is fit post-hoc on a SEPARATE calibration subset (not eval-val).

Additive / no V3 / other-V4 file is touched.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CondFactorizedCSC"]


# ---------------------------------------------------------------------------
# Causal TCN building blocks (mirror model_v4_factorized: kernel 3, dilations
# [1,2,4,...], residual + LayerNorm, left-only padding -> strictly causal).
# ---------------------------------------------------------------------------
class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B,C,T)
        return self.conv(F.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(ch, ch, kernel, dilation)
        self.norm1 = nn.LayerNorm(ch)
        self.conv2 = _CausalConv1d(ch, ch, kernel, dilation)
        self.norm2 = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B,C,T)
        h = self.act(self.norm1(self.conv1(x).permute(0, 2, 1)).permute(0, 2, 1))
        h = self.drop(h)
        h = self.norm2(self.conv2(h).permute(0, 2, 1)).permute(0, 2, 1)
        return self.drop(self.act(h + x))


class _TowerEncoder(nn.Module):
    """proj -> N residual causal blocks -> per-step hidden repr (B,T,hidden)."""

    def __init__(self, in_dim: int, hidden: int, levels: int, kernel: int, dropout: float) -> None:
        super().__init__()
        dilations = [2 ** i for i in range(levels)]
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([_TCNBlock(hidden, kernel, d, dropout) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B,T,in)->(B,T,hidden)
        h = self.proj(x).permute(0, 2, 1)
        for blk in self.blocks:
            h = blk(h)
        return h.permute(0, 2, 1)


class CondFactorizedCSC(nn.Module):
    """Conditional 2-tower / 3-head factorized diagnosis model.

    geom tower -> off_logit; shared resp tower -> {conf_on_logit, conf_off_logit}.
    """

    def __init__(self, geom_dim: int, resp_dim: int, hidden: int = 32,
                 levels: int = 3, kernel: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.geom_dim = int(geom_dim); self.resp_dim = int(resp_dim)
        self.hidden = int(hidden); self.levels = int(levels); self.kernel = int(kernel)

        self.geom_enc = _TowerEncoder(geom_dim, hidden, levels, kernel, dropout)
        self.resp_enc = _TowerEncoder(resp_dim, hidden, levels, kernel, dropout)
        self.head_off = nn.Linear(hidden, 1)
        self.head_conf_on = nn.Linear(hidden, 1)    # CC vs CU  (confirmed | on-target)
        self.head_conf_off = nn.Linear(hidden, 1)   # FC vs LA  (confirmed | off-target)

        # three calibration temperatures (fit post-hoc on a separate calib subset)
        self.register_buffer("t_off", torch.tensor(1.0))
        self.register_buffer("t_conf_on", torch.tensor(1.0))
        self.register_buffer("t_conf_off", torch.tensor(1.0))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, geom: torch.Tensor, resp: torch.Tensor,
                last_step_only: bool = True) -> dict[str, torch.Tensor]:
        gh = self.geom_enc(geom)        # (B,T,hidden)
        rh = self.resp_enc(resp)        # (B,T,hidden)
        off = self.head_off(gh).squeeze(-1)            # (B,T)
        conf_on = self.head_conf_on(rh).squeeze(-1)    # (B,T)
        conf_off = self.head_conf_off(rh).squeeze(-1)  # (B,T)
        if last_step_only:
            off, conf_on, conf_off = off[:, -1], conf_on[:, -1], conf_off[:, -1]
        return {"off_logit": off, "conf_on_logit": conf_on, "conf_off_logit": conf_off}

    def set_temperatures(self, t_off: float, t_conf_on: float, t_conf_off: float) -> None:
        dev = self.t_off.device
        self.t_off = torch.as_tensor(float(t_off), dtype=torch.float32, device=dev)
        self.t_conf_on = torch.as_tensor(float(t_conf_on), dtype=torch.float32, device=dev)
        self.t_conf_off = torch.as_tensor(float(t_conf_off), dtype=torch.float32, device=dev)

    def compose(self, off_logit: torch.Tensor, conf_on_logit: torch.Tensor,
                conf_off_logit: torch.Tensor) -> dict[str, torch.Tensor]:
        p_off = torch.sigmoid(off_logit / self.t_off)
        p_on = torch.sigmoid(conf_on_logit / self.t_conf_on)     # P(confirmed | on-target)
        p_off_c = torch.sigmoid(conf_off_logit / self.t_conf_off)  # P(confirmed | off-target)
        return {
            "p_off": p_off, "p_conf_on": p_on, "p_conf_off": p_off_c,
            "p_cc": (1.0 - p_off) * p_on,
            "p_cu": (1.0 - p_off) * (1.0 - p_on),
            "p_fc": p_off * p_off_c,
            "p_la": p_off * (1.0 - p_off_c),
        }


# ---------------------------------------------------------------------------
# Standalone smoke: python -m csc_lib.csc.v4.model_v4_cond_factorized
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, GD, RD = 4, 32, 12, 4
    m = CondFactorizedCSC(GD, RD, hidden=32, levels=3, kernel=3)
    print(f"[smoke] CondFactorizedCSC(geom={GD}, resp={RD}) params={m.num_params:,}")

    # tower isolation (geom_enc vs resp_enc share no params)
    gi = {id(p) for p in m.geom_enc.parameters()}
    ri = {id(p) for p in m.resp_enc.parameters()}
    assert gi.isdisjoint(ri), "towers share params!"
    print("[smoke] geom/resp tower isolation: OK")

    geom = torch.randn(B, T, GD); resp = torch.randn(B, T, RD)
    m.eval()
    out = m.forward(geom, resp, last_step_only=True)
    assert set(out) == {"off_logit", "conf_on_logit", "conf_off_logit"}
    assert all(out[k].shape == (B,) for k in out), {k: out[k].shape for k in out}

    # causality: perturbing strictly-future frames must not change a mid-step logit
    with torch.no_grad():
        base = m.forward(geom, resp, last_step_only=False)["off_logit"][:, T // 2]
        g3 = geom.clone(); g3[:, T // 2 + 1:] += 7.0
        pert = m.forward(g3, resp, last_step_only=False)["off_logit"][:, T // 2]
        assert torch.allclose(base, pert, atol=1e-5), "future leak — not causal!"
    print("[smoke] causality: OK")

    # composition sums to 1; FC uses conf_OFF, CC uses conf_ON (conditional!)
    m.set_temperatures(1.5, 0.9, 1.2)
    c = m.compose(out["off_logit"], out["conf_on_logit"], out["conf_off_logit"])
    s = c["p_cc"] + c["p_cu"] + c["p_la"] + c["p_fc"]
    assert torch.allclose(s, torch.ones(B), atol=1e-5), s
    assert torch.allclose(c["p_fc"], c["p_off"] * c["p_conf_off"], atol=1e-6)
    assert torch.allclose(c["p_cc"], (1 - c["p_off"]) * c["p_conf_on"], atol=1e-6)
    # the two confidence axes are genuinely independent heads (generally differ)
    assert not torch.allclose(c["p_conf_on"], c["p_conf_off"]), "conf_on/conf_off collapsed"
    print(f"[smoke] conditional compose: OK (sum=1; FC=off*conf_off; CC=(1-off)*conf_on)")
    print("[smoke] OK")
