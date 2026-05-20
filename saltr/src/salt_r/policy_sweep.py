"""Phase 2E: Policy threshold sweep for SALT-RD interventions.

Sweeps thresholds for false_confirmed blocking, wrong_reinit rejection,
e-process alert escalation, and memory margin gating.

Based on v2-aware policy (uses ifd10/20 heads, not just ifd5).
Also incorporates SAMURAI-inspired KF residual idea as a cheap
spatial discontinuity feature.
"""
from __future__ import annotations
import argparse
import json
import math as _math
import numpy as np
from itertools import product
from pathlib import Path
from typing import Any

# Default sweep ranges
FC_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
REINIT_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80)
EPROCESS_ALPHAS = (0.20, 0.10, 0.05)
MEM_MARGIN_THRESHOLDS = (-0.10, 0.00, 0.10, 0.20)


class PolicySweepConfig:
    """Configuration for one point in the policy threshold sweep."""

    def __init__(
        self,
        fc_threshold: float = 0.60,
        reinit_threshold: float = 0.60,
        eprocess_alpha: float = 0.10,
        mem_margin_threshold: float = 0.0,
        use_ifd10: bool = True,    # use imminent_failure_dynamic_10 for expand_search
        use_ifd20: bool = True,    # use imminent_failure_dynamic_20 for early warning
        kf_residual_threshold: float = 0.4,  # SAMURAI-inspired: flag large bbox jumps
    ) -> None:
        self.fc_threshold = float(fc_threshold)
        self.reinit_threshold = float(reinit_threshold)
        self.eprocess_alpha = float(eprocess_alpha)
        self.mem_margin_threshold = float(mem_margin_threshold)
        self.use_ifd10 = bool(use_ifd10)
        self.use_ifd20 = bool(use_ifd20)
        self.kf_residual_threshold = float(kf_residual_threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fc_threshold": self.fc_threshold,
            "reinit_threshold": self.reinit_threshold,
            "eprocess_alpha": self.eprocess_alpha,
            "mem_margin_threshold": self.mem_margin_threshold,
            "use_ifd10": self.use_ifd10,
            "use_ifd20": self.use_ifd20,
            "kf_residual_threshold": self.kf_residual_threshold,
        }


# ---------------------------------------------------------------------------
# Kalman Filter for bbox trajectory (SAMURAI-inspired)
# ---------------------------------------------------------------------------

class SimpleBboxKalmanFilter:
    """Minimal Kalman Filter for bbox trajectory (SAMURAI-inspired).

    State: [cx, cy, w, h, vcx, vcy, vw, vh]
    Gives kf_residual = 1 - kf_iou(KF_prediction, actual_bbox) per frame.
    Cheap feature: ~0.1ms per frame.

    Model:
      x_{t+1} = F x_t  (constant velocity)
      z_t = H x_t      (observe position/size)
    """

    # State dimension: [cx, cy, w, h, vcx, vcy, vw, vh]
    _S = 8
    # Observation dimension: [cx, cy, w, h]
    _O = 4

    def __init__(self) -> None:
        self.initialized = False
        # State vector [cx, cy, w, h, vcx, vcy, vw, vh]
        self._x = np.zeros(self._S, dtype=np.float64)
        # State covariance
        self._P = np.eye(self._S, dtype=np.float64) * 100.0

        # Constant velocity transition matrix
        self._F = np.eye(self._S, dtype=np.float64)
        self._F[0, 4] = 1.0  # cx += vcx
        self._F[1, 5] = 1.0  # cy += vcy
        self._F[2, 6] = 1.0  # w  += vw
        self._F[3, 7] = 1.0  # h  += vh

        # Observation matrix (we observe [cx, cy, w, h])
        self._H = np.zeros((self._O, self._S), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0

        # Process noise (position/size uncertainties)
        _q = 1.0
        self._Q = np.diag([_q, _q, _q * 0.5, _q * 0.5,
                           _q * 2, _q * 2, _q, _q]).astype(np.float64)

        # Measurement noise
        _r = 4.0
        self._R = np.eye(self._O, dtype=np.float64) * _r

    @staticmethod
    def _bbox_to_state(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1,y1,x2,y2] to [cx,cy,w,h]."""
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h], dtype=np.float64)

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> np.ndarray:
        """Convert [cx,cy,w,h,...] to [x1,y1,x2,y2]."""
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)

    @staticmethod
    def _bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
        """Compute IoU between two [x1,y1,x2,y2] bboxes."""
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 1e-9 else 0.0

    def update(self, bbox: np.ndarray) -> float:
        """Update with observed bbox. Returns kf_residual = 1 - kf_iou.

        kf_residual = 0.0 → bbox matches KF prediction perfectly.
        kf_residual = 1.0 → bbox completely outside KF prediction.
        """
        z = self._bbox_to_state(bbox)

        if not self.initialized:
            # Initialize state with first observation
            self._x[:4] = z
            self._x[4:] = 0.0  # zero velocity
            self.initialized = True
            return 0.0  # no residual on first frame

        # Predict step
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # Compute predicted bbox for residual before update
        pred_bbox = self._state_to_bbox(x_pred)
        actual_bbox = np.array([
            z[0] - z[2] / 2, z[1] - z[3] / 2,
            z[0] + z[2] / 2, z[1] + z[3] / 2,
        ])
        kf_iou = self._bbox_iou(pred_bbox, actual_bbox)
        kf_residual = 1.0 - kf_iou

        # Update step (Kalman gain)
        S = self._H @ P_pred @ self._H.T + self._R
        try:
            K = P_pred @ self._H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((self._S, self._O))

        innovation = z - self._H @ x_pred
        self._x = x_pred + K @ innovation
        self._P = (np.eye(self._S) - K @ self._H) @ P_pred

        return float(kf_residual)

    def predict(self) -> np.ndarray:
        """Predict next bbox [x1,y1,x2,y2] without updating state."""
        if not self.initialized:
            return np.zeros(4, dtype=np.float64)
        x_pred = self._F @ self._x
        return self._state_to_bbox(x_pred)


# ---------------------------------------------------------------------------
# V2 policy step simulation
# ---------------------------------------------------------------------------

def simulate_policy_step(
    probs: dict[str, float],
    prev_bbox: np.ndarray | None,
    curr_bbox: np.ndarray | None,
    eprocess_value: float,
    memory_margin: float,
    config: PolicySweepConfig,
) -> dict[str, Any]:
    """Simulate one policy step. Returns dict with triggered interventions.

    V2-aware policy (unlike old policy.py which only uses v0 heads):
    1. false_confirmed → block template update, abstain recovery
    2. ifd10/ifd20 → expand search region (not just hard_dynamic_scene)
    3. e-process alert → verify before re-init
    4. memory margin < threshold → treat as distractor present → same as fc
    5. kf_residual high → spatial discontinuity → flag as potential false confirmed

    KF residual (SAMURAI idea): if bbox jumped more than expected from
    velocity estimate → potential identity switch.
    """
    p_fc = float(probs.get("false_confirmed", 0.0))
    p_ifd10 = float(probs.get("imminent_failure_dynamic_10", 0.0))
    p_ifd20 = float(probs.get("imminent_failure_dynamic_20", 0.0))
    p_rec = float(probs.get("recoverable", 0.0))
    p_fi5 = float(probs.get("failure_in_5", 0.0))

    interventions: list[str] = []
    template_update = "allow"
    recovery_action = "none"
    search_mode = "normal"
    alert_tier = "none"

    # Compute KF residual if bboxes provided
    # NOTE: this uses a simple displacement approximation, NOT the SimpleBboxKalmanFilter.
    # Wire SimpleBboxKalmanFilter into _evaluate_config (pass kf instance per sequence)
    # before reporting KF residual results in papers.
    kf_residual = 0.0
    if prev_bbox is not None and curr_bbox is not None:
        # Simple displacement-based residual approximation
        prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2.0
        prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2.0
        curr_cx = (curr_bbox[0] + curr_bbox[2]) / 2.0
        curr_cy = (curr_bbox[1] + curr_bbox[3]) / 2.0
        prev_diag = np.sqrt(
            max(prev_bbox[2] - prev_bbox[0], 1.0) ** 2 +
            max(prev_bbox[3] - prev_bbox[1], 1.0) ** 2
        )
        dist = np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)
        kf_residual = float(min(1.0, dist / max(prev_diag, 1.0)))

    # 1. false_confirmed dominates
    fc_triggered = (
        p_fc >= config.fc_threshold or
        memory_margin < config.mem_margin_threshold
    )
    if fc_triggered:
        template_update = "block"
        recovery_action = "abstain"
        interventions.append(f"fc={p_fc:.2f}")
        if memory_margin < config.mem_margin_threshold:
            interventions.append(f"mem_margin={memory_margin:.3f}")

    # 2. KF residual → flag potential identity switch
    if kf_residual > config.kf_residual_threshold:
        if template_update == "allow":
            template_update = "verify"
        interventions.append(f"kf_residual={kf_residual:.2f}")

    # 3. e-process alert
    eprocess_threshold = 1.0 / max(config.eprocess_alpha, 1e-9)
    if eprocess_value >= eprocess_threshold * 5:
        alert_tier = "critical"
        if template_update == "allow":
            template_update = "verify"
    elif eprocess_value >= eprocess_threshold:
        alert_tier = "intervene"
        interventions.append(f"eprocess={eprocess_value:.1f}")

    # 4. ifd10 → expand search
    if config.use_ifd10 and p_ifd10 >= 0.60:
        search_mode = "expand"
        if template_update == "allow":
            template_update = "verify"
        interventions.append(f"ifd10={p_ifd10:.2f}")

    # 5. ifd20 → early warning
    if config.use_ifd20 and p_ifd20 >= 0.50:
        if alert_tier == "none":
            alert_tier = "observe"
        interventions.append(f"ifd20={p_ifd20:.2f}")

    # 6. Recovery decision
    if not fc_triggered and recovery_action == "none":
        if p_rec >= config.reinit_threshold and p_fc < 0.40:
            recovery_action = "run"
            interventions.append(f"recoverable={p_rec:.2f}")

    return {
        "template_update": template_update,
        "recovery_action": recovery_action,
        "search_mode": search_mode,
        "alert_tier": alert_tier,
        "kf_residual": kf_residual,
        "triggered_by": interventions,
        "p_fc": p_fc,
        "p_ifd10": p_ifd10,
        "p_ifd20": p_ifd20,
    }


# ---------------------------------------------------------------------------
# Full sweep over policy configs
# ---------------------------------------------------------------------------

def run_policy_sweep(
    preds_json_path: str,
    labels_npz_path: str,
    eprocess_json_path: str | None = None,
    memory_sidecar_path: str | None = None,
    output_path: str = "saltr/results/policy_sweep_v2.json",
) -> dict:
    """Run full threshold sweep over policy configs.

    Metrics:
      template_corruption_rate: allowed updates when IoU < 0.5
      wrong_reinit_rate: accepted recovery when IoU < 0.3
      missed_safe_update_rate: blocked update when IoU >= 0.7
      intervention_density: interventions per 1000 frames
      failure_event_recall: events with alert before failure
      lead_time_median: median frames between first alert and failure
    """
    # Load predictions
    with open(preds_json_path) as f:
        all_preds: dict[str, list[dict[str, float]]] = json.load(f)

    # Load labels and IoU traces
    data = np.load(labels_npz_path, allow_pickle=True)
    iou_traces: dict[str, np.ndarray] = {}
    for k in data.files:
        if k.startswith("iou_trace/"):
            seq = k[len("iou_trace/"):]
            iou_traces[seq] = data[k]

    bbox_pred_data: dict[str, np.ndarray] = {}
    for k in data.files:
        if k.startswith("bbox_pred/"):
            seq = k[len("bbox_pred/"):]
            bbox_pred_data[seq] = data[k]

    # Load e-process values if provided
    eprocess_data: dict[str, list[float]] = {}
    if eprocess_json_path is not None:
        with open(eprocess_json_path) as f:
            ep_raw = json.load(f)
        if isinstance(ep_raw, dict):
            # New format: per_sequence[seq]["e_trace"]
            per_seq_ep = ep_raw.get("per_sequence", {})
            for seq, seq_data in per_seq_ep.items():
                if isinstance(seq_data, dict) and "e_trace" in seq_data:
                    eprocess_data[seq] = seq_data["e_trace"]
                elif isinstance(seq_data, list):
                    # Legacy flat format: {seq: [e_t0, ...]}
                    eprocess_data[seq] = seq_data

    # Load memory sidecar if provided
    memory_data: dict[str, np.ndarray] = {}
    if memory_sidecar_path is not None:
        mem_npz = np.load(memory_sidecar_path, allow_pickle=True)
        for k in mem_npz.files:
            if k.startswith("memory_margin/"):
                seq = k[len("memory_margin/"):]
                memory_data[seq] = mem_npz[k]

    # Generate sweep configs
    configs = []
    for fc_t, reinit_t, ep_a, mem_t in product(
        FC_THRESHOLDS, REINIT_THRESHOLDS, EPROCESS_ALPHAS, MEM_MARGIN_THRESHOLDS
    ):
        configs.append(PolicySweepConfig(
            fc_threshold=fc_t,
            reinit_threshold=reinit_t,
            eprocess_alpha=ep_a,
            mem_margin_threshold=mem_t,
        ))

    results: list[dict[str, Any]] = []

    for config in configs:
        metrics = _evaluate_config(
            config, all_preds, iou_traces, eprocess_data, memory_data, bbox_pred_data
        )
        row = config.to_dict()
        row.update(metrics)
        results.append(row)

    # Find Pareto-optimal configs
    summary = {
        "n_configs": len(results),
        "configs": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _nanmean(vals: list[float]) -> float:
    """Return mean of non-NaN values, or NaN if all are NaN."""
    valid = [v for v in vals if not _math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def _evaluate_config(
    config: PolicySweepConfig,
    all_preds: dict[str, list[dict[str, float]]],
    iou_traces: dict[str, np.ndarray],
    eprocess_data: dict[str, list[float]],
    memory_data: dict[str, np.ndarray],
    bbox_pred_data: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate one policy config across all sequences."""
    if bbox_pred_data is None:
        bbox_pred_data = {}
    total_frames = 0
    total_allowed_low_iou = 0
    total_allowed = 0
    total_wrong_reinit = 0
    total_reinit_attempts = 0
    total_missed_safe = 0
    total_blocked = 0
    total_interventions = 0
    total_failure_events = 0
    total_alerted_failures = 0
    lead_times: list[int] = []

    # Per-sequence rate lists for macro (sequence-averaged) metrics
    seq_tcr_list: list[float] = []
    seq_wrir_list: list[float] = []

    for seq, probs_seq in all_preds.items():
        if seq not in iou_traces:
            continue

        iou = iou_traces[seq]
        n = min(len(probs_seq), len(iou))
        ep_vals = eprocess_data.get(seq, [1.0] * n)
        mem_vals = memory_data.get(seq, np.zeros(n, dtype=float))
        bbox_seq = bbox_pred_data.get(seq, None)

        # Per-sequence counters for macro metrics
        seq_allowed_low_iou = 0
        seq_allowed = 0
        seq_wrong_reinit = 0
        seq_reinit_attempts = 0

        # Track failure events for lead-time calculation
        prev_above = False
        failure_frames: list[int] = []
        for t in range(n):
            if prev_above and float(iou[t]) < 0.3:
                failure_frames.append(t)
                prev_above = False
            elif float(iou[t]) >= 0.3:
                prev_above = True

        # Track first alert per failure event
        first_alerts: dict[int, int] = {}  # failure_frame → first alert within 20 frames before

        prev_bbox: np.ndarray | None = None
        for t in range(n):
            ep_val = float(ep_vals[t]) if t < len(ep_vals) else 1.0
            mem_val = float(mem_vals[t]) if t < len(mem_vals) else 0.0

            if bbox_seq is not None and t < len(bbox_seq):
                xywh = bbox_seq[t]
                curr_bbox = np.array([xywh[0], xywh[1],
                                       xywh[0] + xywh[2], xywh[1] + xywh[3]], dtype=np.float64)
            else:
                curr_bbox = None

            step = simulate_policy_step(
                probs=probs_seq[t],
                prev_bbox=prev_bbox,
                curr_bbox=curr_bbox,
                eprocess_value=ep_val,
                memory_margin=mem_val,
                config=config,
            )  # kf_residual from simple displacement; SimpleBboxKalmanFilter not yet wired here
            prev_bbox = curr_bbox

            frame_iou = float(iou[t])

            # Template update metrics
            if step["template_update"] == "allow":
                total_allowed += 1
                seq_allowed += 1
                if frame_iou < 0.5:
                    total_allowed_low_iou += 1
                    seq_allowed_low_iou += 1
            elif step["template_update"] == "block":
                total_blocked += 1
                if frame_iou >= 0.7:
                    total_missed_safe += 1

            # Recovery metrics
            if step["recovery_action"] == "run":
                total_reinit_attempts += 1
                seq_reinit_attempts += 1
                if frame_iou < 0.3:
                    total_wrong_reinit += 1
                    seq_wrong_reinit += 1

            # Intervention density
            if len(step["triggered_by"]) > 0:
                total_interventions += 1

            # Lead time: check if this frame is an alert before a failure
            if len(step["triggered_by"]) > 0:
                for fail_t in failure_frames:
                    if 0 < fail_t - t <= 20:
                        if fail_t not in first_alerts:
                            first_alerts[fail_t] = t

            total_frames += 1

        # Count alerted failures and compute lead times
        for fail_t in failure_frames:
            total_failure_events += 1
            if fail_t in first_alerts:
                total_alerted_failures += 1
                lead_times.append(fail_t - first_alerts[fail_t])

        # Accumulate per-sequence rates for macro metrics
        seq_tcr = seq_allowed_low_iou / max(seq_allowed, 1)
        seq_tcr_list.append(seq_tcr)
        if seq_reinit_attempts > 0:
            seq_wrir = seq_wrong_reinit / seq_reinit_attempts
            seq_wrir_list.append(seq_wrir)

    template_corruption_rate = (
        total_allowed_low_iou / max(total_allowed, 1)
    )
    wrong_reinit_rate = total_wrong_reinit / max(total_reinit_attempts, 1)
    missed_safe_update_rate = total_missed_safe / max(total_blocked, 1)
    intervention_density = 1000.0 * total_interventions / max(total_frames, 1)
    failure_event_recall = total_alerted_failures / max(total_failure_events, 1)
    lead_time_median = float(np.median(lead_times)) if lead_times else float("nan")

    return {
        # Micro (frame-level, existing)
        "template_corruption_rate": round(template_corruption_rate, 4),
        "wrong_reinit_rate": round(wrong_reinit_rate, 4),
        "missed_safe_update_rate": round(missed_safe_update_rate, 4),
        "intervention_density": round(intervention_density, 2),
        "failure_event_recall": round(failure_event_recall, 4),
        "lead_time_median": lead_time_median,
        "total_frames": total_frames,
        # Macro (sequence-averaged)
        "macro_tcr": round(_nanmean(seq_tcr_list), 4) if seq_tcr_list else float("nan"),
        "macro_wrir": round(_nanmean(seq_wrir_list), 4) if seq_wrir_list else float("nan"),
        "n_seqs_evaluated": len(seq_tcr_list),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> "argparse.Namespace":
    import argparse
    p = argparse.ArgumentParser(
        description="Phase 2E: Policy threshold sweep for SALT-RD interventions."
    )
    p.add_argument("--preds", required=True,
                   help="Path to preds_val_*.json (calibrated model predictions)")
    p.add_argument("--labels", required=True,
                   help="Path to salt_rd_v2_labels.npz")
    p.add_argument("--eprocess", default=None,
                   help="Optional path to eprocess_val_*.json (for E-trace per sequence). "
                        "When omitted, e-process contribution is zeroed out.")
    p.add_argument("--memory", default=None,
                   help="Optional path to salt_rd_memory_sidecar.npz (for memory_margin per sequence). "
                        "When omitted, memory contribution is zeroed out.")
    p.add_argument("--output", default="saltr/results/policy_sweep_v2.json",
                   help="Output path for sweep results JSON")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[policy_sweep] Preds:    {args.preds}", flush=True)
    print(f"[policy_sweep] Labels:   {args.labels}", flush=True)
    print(f"[policy_sweep] E-process:{args.eprocess or '(none)'}", flush=True)
    print(f"[policy_sweep] Memory:   {args.memory or '(none)'}", flush=True)
    print(f"[policy_sweep] Output:   {args.output}", flush=True)

    summary = run_policy_sweep(
        preds_json_path=args.preds,
        labels_npz_path=args.labels,
        eprocess_json_path=args.eprocess,
        memory_sidecar_path=args.memory,
        output_path=args.output,
    )
    n = summary.get("n_configs", 0)
    configs = summary.get("configs", [])
    if configs:
        # Print the single best config by template_corruption_rate
        best = min(configs, key=lambda r: r.get("template_corruption_rate", 1.0))
        print(f"\n[policy_sweep] {n} configs evaluated.", flush=True)
        print(f"[policy_sweep] Best template_corruption_rate: "
              f"{best['template_corruption_rate']:.4f} "
              f"(fc_t={best['fc_threshold']}, reinit_t={best['reinit_threshold']}, "
              f"ep_a={best['eprocess_alpha']}, mem_t={best['mem_margin_threshold']})",
              flush=True)
        print(f"[policy_sweep] Best wrong_reinit_rate: {best['wrong_reinit_rate']:.4f}",
              flush=True)
    print(f"[policy_sweep] Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
