"""Drift-protected test-time adaptation of SGLATrack prediction head.

Three-level drift prevention:
  Gate:     only adapt when TargetState == CONFIRMED and confidence > threshold
  Bound:    KL(head_params, ema_ref_params) < kl_bound → rollback if exceeded
  Validate: lookahead confidence check after update → rollback if drop > 20%
"""
from __future__ import annotations

import copy

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# TargetState CONFIRMED == 0  (matches tracker convention: 0=CONFIRMED)
_CONFIRMED = 0


def _make_gaussian_heatmap(peak_r: int, peak_c: int, h: int, w: int,
                            sigma: float = 2.0) -> "torch.Tensor":
    """Return a (1,1,h,w) Gaussian centred at (peak_r, peak_c), peak value 1."""
    rs = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
    cs = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
    heatmap = torch.exp(-((rs - peak_r) ** 2 + (cs - peak_c) ** 2) / (2 * sigma ** 2))
    return heatmap.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)


def _focal_loss(pred: "torch.Tensor", target: "torch.Tensor",
                alpha: float = 2.0, beta: float = 4.0) -> "torch.Tensor":
    """CornerNet-style focal loss between predicted and Gaussian pseudo-GT heatmaps.

    pred:   (1,1,H,W) — sigmoid score map already in (0,1)
    target: (1,1,H,W) — Gaussian pseudo-GT in [0,1]
    """
    pred = pred.clamp(min=1e-6, max=1.0 - 1e-6)
    pos_mask = (target >= 0.99).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask

    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


class HeadAdaptor:
    """Drift-protected test-time training adaptor for SGLATrack box_head.

    A forward hook is installed on *box_head* at construction time to capture the
    backbone feature tensor that feeds it.  On every ``step()`` call the EMA
    reference is updated; on every *adapt_interval*-th frame (when the gate
    passes) a single SGD update is applied and the three-level drift guard runs.

    Usage::

        adaptor = HeadAdaptor(tracker._model.box_head)
        # inside tracking loop, after each model forward pass:
        adaptor.step(score_map, size_map, offset_map, confidence, target_state)
    """

    def __init__(
        self,
        box_head: "nn.Module | None",
        lr: float = 1e-5,
        kl_bound: float = 0.05,
        ema_decay: float = 0.999,
        confidence_threshold: float = 0.75,
        confidence_drop_threshold: float = 0.20,
        adapt_interval: int = 10,
    ) -> None:
        self._box_head = box_head
        self._lr = lr
        self._kl_bound = kl_bound
        self._ema_decay = ema_decay
        self._confidence_threshold = confidence_threshold
        self._confidence_drop_threshold = confidence_drop_threshold
        self._adapt_interval = adapt_interval

        self._frame_count: int = 0
        self._ema_state: "dict | None" = None
        self._optimizer: "torch.optim.SGD | None" = None
        self._last_input: "torch.Tensor | None" = None  # captured by forward hook
        self._last_peak: "tuple[int, int] | None" = None  # peak from current frame (t)
        self._prev_peak: "tuple[int, int] | None" = None  # peak from previous frame (t-1)
        self._hook_handle = None

        if _TORCH_AVAILABLE and box_head is not None:
            # Snapshot the pretrained weights as the EMA reference
            self._ema_state = {
                k: v.clone() for k, v in box_head.state_dict().items()
            }
            self._optimizer = torch.optim.SGD(box_head.parameters(), lr=lr)
            # Hook to capture the feature tensor fed into box_head each forward pass
            self._hook_handle = box_head.register_forward_hook(self._capture_input)

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def step(
        self,
        pred_score_map: "torch.Tensor | None",
        size_map: "torch.Tensor | None",
        offset_map: "torch.Tensor | None",
        confidence: float,
        target_state: int,
        flow_displacement: "tuple[float, float] | None" = None,
    ) -> bool:
        """Potentially adapt box_head for one frame. Returns True if adaptation applied.

        Args:
            pred_score_map: model score map from this frame (may be None — unused when
                flow_displacement is provided).
            size_map:        model size map (may be None).
            offset_map:      model offset map (may be None).
            confidence:      tracker confidence score for this frame.
            target_state:    TargetState int (0 = CONFIRMED).
            flow_displacement: (dx_px, dy_px) optical-flow displacement of the target
                centre in search-region pixel coordinates, estimated externally.
                When provided this is used as pseudo-GT instead of the model's own
                argmax, breaking the circularity of the original implementation.
        """
        if not _TORCH_AVAILABLE or self._box_head is None or self._ema_state is None:
            return False

        self._frame_count += 1

        # Always update EMA (track long-run parameter distribution)
        self._update_ema()

        # Gate 1: only adapt at the configured interval
        if self._frame_count % self._adapt_interval != 0:
            return False

        # Gate 2: no stored input feature (hook hasn't fired yet)
        if self._last_input is None:
            return False

        # Gate 3: CONFIRMED state + high-confidence prediction only
        if target_state != _CONFIRMED or confidence <= self._confidence_threshold:
            return False

        return self._adapt(pred_score_map, size_map, offset_map, confidence,
                           flow_displacement)

    def reset(self) -> None:
        """Reset frame counter and cached input (call when tracker re-initialises)."""
        self._frame_count = 0
        self._last_input = None
        self._last_peak = None
        self._prev_peak = None

    def set_last_peak(self, row: int, col: int) -> None:
        """Cache the argmax position from the most recent tracker forward pass.

        Call this after each tracker update so that the next TTT step can use the
        previous peak as the anchor for the flow-displaced pseudo-GT.
        """
        self._last_peak = (row, col)

    # ---------------------------------------------------------------------- #
    # Forward hook                                                             #
    # ---------------------------------------------------------------------- #

    def _capture_input(self, module: "nn.Module", args: tuple, output) -> None:
        """Store the first positional argument (opt_feat) passed to box_head.

        Rolls _last_peak → _prev_peak so that _adapt() always has the t-1 peak
        available as a clean anchor for flow-displaced pseudo-GT construction.
        """
        if args:
            self._last_input = args[0].detach()
        # Roll previous peak forward before overwriting
        self._prev_peak = self._last_peak
        try:
            score = output[0] if isinstance(output, (tuple, list)) else output
            flat_idx = int(score.detach().view(-1).argmax())
            _, _, H, W = score.shape
            self._last_peak = (flat_idx // W, flat_idx % W)
        except Exception:
            pass

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _update_ema(self) -> None:
        """Blend current head params into the EMA reference state."""
        decay = self._ema_decay
        for k, v in self._box_head.state_dict().items():
            if self._ema_state[k].is_floating_point():
                self._ema_state[k].mul_(decay).add_(
                    v.to(self._ema_state[k].device), alpha=1.0 - decay
                )

    def _param_drift(self) -> float:
        """L2-squared distance between current params and EMA state (KL proxy)."""
        total = 0.0
        for k, v in self._box_head.state_dict().items():
            if self._ema_state[k].is_floating_point():
                diff = v.to(self._ema_state[k].device) - self._ema_state[k]
                total += float(diff.pow(2).sum())
        return total

    def _adapt(
        self,
        score_map: "torch.Tensor | None",
        size_map: "torch.Tensor | None",
        offset_map: "torch.Tensor | None",
        confidence: float,
        flow_displacement: "tuple[float, float] | None" = None,
    ) -> bool:
        """Core adaptation step. Returns True if the update was kept.

        Pseudo-GT construction priority:
          1. flow_displacement + _last_peak  → flow-anchored Gaussian (breaks circularity)
          2. score_map argmax               → fallback (circular, but keeps old behaviour
                                              for callers that still pass a score_map)
          Both paths abort gracefully if neither source is available.
        """
        # Determine the score-map shape from whatever source is available
        feat = self._last_input
        if feat is None:
            return False

        # Use _prev_peak (peak at t-1) + flow(t-1→t) = expected position at t.
        # This is the correct anchor: training box_head(feat_t) → predict(peak_{t-1} + flow).
        # Do NOT use _last_peak (peak at t) here — that gives pseudo_GT = peak_t + flow = t+1 estimate.
        stashed_peak = self._prev_peak

        # Run a no-grad forward pass just to get spatial dimensions H, W and device
        with torch.no_grad():
            out_probe = self._box_head(feat)
            score_probe = out_probe[0] if isinstance(out_probe, (tuple, list)) else out_probe
            device = score_probe.device
            _, _, H, W = score_probe.shape

        # ---- Pseudo-label construction (no-grad; these are targets) ---
        with torch.no_grad():
            if flow_displacement is not None and stashed_peak is not None:
                # Flow-anchored pseudo-GT: displace previous peak by optical flow.
                # stride = search_region_size / H (typically 256/16 = 16 px)
                stride = 256.0 / H
                dx_px, dy_px = flow_displacement
                pr_new = int(np.clip(stashed_peak[0] + dy_px / stride, 0, H - 1))
                pc_new = int(np.clip(stashed_peak[1] + dx_px / stride, 0, W - 1))
                pseudo_heatmap = _make_gaussian_heatmap(pr_new, pc_new, H, W,
                                                        sigma=2.0).to(device)
                # Pseudo box: approximate from displaced peak only (no offset/size maps)
                pseudo_box = torch.tensor([
                    pc_new / W,   # cx_norm
                    pr_new / H,   # cy_norm
                    0.1,          # w_norm placeholder
                    0.1,          # h_norm placeholder
                ], dtype=torch.float32, device=device)
            elif score_map is not None:
                # Fallback: circular argmax path (original behaviour)
                flat_idx = int(score_map.view(-1).argmax())
                peak_r, peak_c = flat_idx // W, flat_idx % W
                pseudo_heatmap = _make_gaussian_heatmap(peak_r, peak_c, H, W,
                                                        sigma=2.0).to(device)
                if offset_map is not None and size_map is not None:
                    off = offset_map[0, :, peak_r, peak_c]
                    sz  = size_map[0,  :, peak_r, peak_c]
                    pseudo_box = torch.tensor([
                        (peak_c + float(off[1])) / W,
                        (peak_r + float(off[0])) / H,
                        float(sz[1]),
                        float(sz[0]),
                    ], dtype=torch.float32, device=device)
                else:
                    pseudo_box = torch.tensor(
                        [peak_c / W, peak_r / H, 0.1, 0.1],
                        dtype=torch.float32, device=device,
                    )
            else:
                # No usable pseudo-GT source — skip this TTT step
                return False

        # Save pre-step state for rollback
        pre_step_state = {k: v.clone() for k, v in self._box_head.state_dict().items()}

        # --- SGD step: re-run box_head on the captured feature (gradients ON) ---
        self._box_head.train()
        self._optimizer.zero_grad()

        out = self._box_head(feat)
        # CenterPredictor.forward returns (score_map_ctr, bbox, size_map, offset_map)
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            pred_score_ctr, _, pred_size, pred_offset = out[0], out[1], out[2], out[3]
        else:
            # Fallback: single tensor output (e.g. MLP head)
            pred_score_ctr = out if isinstance(out, torch.Tensor) else out[0]
            pred_size, pred_offset = None, None

        # Predicted box at peak of new score map
        with torch.no_grad():
            flat_idx2 = int(pred_score_ctr.view(-1).argmax())
            pr2, pc2 = flat_idx2 // W, flat_idx2 % W
            if pred_offset is not None and pred_size is not None:
                off2 = pred_offset[0, :, pr2, pc2]
                sz2  = pred_size[0,  :, pr2, pc2]
                pred_box = torch.tensor([
                    (pc2 + float(off2[1])) / W,
                    (pr2 + float(off2[0])) / H,
                    float(sz2[1]),
                    float(sz2[0]),
                ], dtype=torch.float32, device=device)
            else:
                pred_box = torch.tensor(
                    [pc2 / W, pr2 / H, 0.1, 0.1],
                    dtype=torch.float32, device=device,
                )

        loss = _focal_loss(pred_score_ctr, pseudo_heatmap.to(pred_score_ctr.dtype))
        loss = loss + F.l1_loss(pred_box, pseudo_box)
        loss.backward()
        self._optimizer.step()

        self._box_head.eval()

        # --- Drift check (KL bound via L2 proxy) ---
        if self._param_drift() > self._kl_bound:
            self._box_head.load_state_dict(pre_step_state)
            return False

        # --- Lookahead validation: confidence must not drop >drop_threshold ---
        with torch.no_grad():
            out2 = self._box_head(feat)
            new_score = out2[0] if isinstance(out2, (tuple, list)) else out2
            new_conf = float(new_score.max())

        if new_conf < confidence * (1.0 - self._confidence_drop_threshold):
            self._box_head.load_state_dict(pre_step_state)
            return False

        return True
