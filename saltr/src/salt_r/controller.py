from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from salt_r.actions import TrackerAction, ComputeAction, SearchAction, TemplateAction, RecoveryAction
from salt_r.evidence import EvidenceFrame, CandidateEvidence
from salt_r.feature_schema import FEATURE_SCHEMA_VERSION, validate_feature_matrix


@dataclass
class SALTRDDecision:
    action: TrackerAction
    risk_probs: dict[str, float] = field(default_factory=dict)
    action_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    selected_candidate: CandidateEvidence | None = None
    model_confidence: float = 0.0
    safety_fallback_applied: bool = False
    reason: str = ""


class SALTRDController:
    """
    Runtime controller. Takes EvidenceFrame, returns SALTRDDecision.
    No TSA. No TargetState.

    reinit_confidence_threshold: if > 0, fire REINIT when p(REINIT) >= threshold
    regardless of argmax. Aligns live inference with rollout_policy.py simulation.
    """

    def __init__(
        self,
        policy_net=None,
        feature_schema: str = FEATURE_SCHEMA_VERSION,
        reinit_confidence_threshold: float = 0.0,
    ):
        self._policy_net = policy_net
        self._feature_schema = feature_schema
        self._reinit_confidence_threshold = reinit_confidence_threshold
        self._frame_idx = 0
        self._window_size = int(getattr(policy_net, "window_size", 1) or 1)
        self._feature_window: deque[np.ndarray] = deque(maxlen=max(1, self._window_size))

    def reset(self) -> None:
        self._frame_idx = 0
        self._feature_window.clear()

    def step(self, evidence: EvidenceFrame) -> SALTRDDecision:
        """
        Produce a SALTRDDecision from an EvidenceFrame.

        If policy_net is None (no model loaded), returns a safe NOOP full-compute action.
        All decisions come from model output; no tracking thresholds here.
        """
        if self._policy_net is None:
            return self._safe_noop(reason="no_model_loaded")

        features = evidence.base_features
        # Safety: validate shape
        try:
            validate_feature_matrix(features, expected_dim=28)
        except ValueError as e:
            return self._safe_noop(reason=f"feature_shape_invalid:{e}")

        # Safety: NaN/Inf guard
        if not np.isfinite(features).all():
            return self._safe_noop(reason="features_not_finite")

        # Run model with the same left-padded temporal window used by train_policy.py.
        # Passing a single frame here changes GRU behavior and invalidates calibrated
        # action thresholds learned from 20-frame windows.
        try:
            import torch
            window = self._build_feature_window(features)
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            output = self._policy_net(x)
        except Exception as e:  # noqa: BLE001
            return self._safe_noop(reason=f"model_error:{e}")

        return self._decode_output(output, evidence)

    def _build_feature_window(self, features: np.ndarray) -> np.ndarray:
        """Return a left-zero-padded ``(window_size, 28)`` feature window."""
        feat = np.asarray(features, dtype=np.float32).copy()
        self._feature_window.append(feat)

        hist = list(self._feature_window)
        pad_len = self._window_size - len(hist)
        if pad_len > 0:
            padding = [np.zeros_like(feat) for _ in range(pad_len)]
            hist = padding + hist
        return np.stack(hist[-self._window_size:], axis=0).astype(np.float32, copy=False)

    def _decode_output(self, output: dict[str, Any], evidence: EvidenceFrame) -> SALTRDDecision:
        """Decode raw model output dict into a SALTRDDecision."""
        # Convert any tensor values to plain Python floats
        def _to_float(v: Any) -> float:
            try:
                return float(v)
            except Exception:
                return 0.0

        raw_risk = output.get("risk_probs", {})
        risk_probs = {k: _to_float(v) for k, v in raw_risk.items()}
        action_logits = output.get("action_logits", {})

        # Argmax decode per action head
        compute = self._decode_enum(action_logits.get("compute"), ComputeAction, ComputeAction.FULL)
        search = self._decode_enum(action_logits.get("search"), SearchAction, SearchAction.KEEP)
        template = self._decode_enum(action_logits.get("template"), TemplateAction, TemplateAction.KEEP_CURRENT)
        recovery = self._decode_enum(action_logits.get("recovery"), RecoveryAction, RecoveryAction.NONE)

        # Confidence threshold override: align with rollout_policy.py simulation.
        # If p(REINIT) >= reinit_confidence_threshold, fire REINIT regardless of argmax.
        # This is a model-output gate, not a tracking heuristic.
        if self._reinit_confidence_threshold > 0.0:
            rec_logits = action_logits.get("recovery")
            if rec_logits is not None:
                try:
                    import torch
                    import torch.nn.functional as F
                    t = rec_logits if isinstance(rec_logits, torch.Tensor) else torch.tensor(rec_logits)
                    rec_probs = F.softmax(t.detach().cpu().float().flatten(), dim=0)
                    reinit_prob = rec_probs[2].item()  # index 2 = REINIT
                    if reinit_prob >= self._reinit_confidence_threshold:
                        recovery = RecoveryAction.REINIT
                except Exception:
                    pass  # fall back to argmax result

        # Candidate selection: pick highest-scored candidate if reinit is requested
        selected_candidate = None
        if recovery == RecoveryAction.REINIT and evidence.candidates:
            candidate_scores = output.get("candidate_scores", [])
            if candidate_scores and len(candidate_scores) == len(evidence.candidates):
                best_idx = int(np.argmax(candidate_scores))
                selected_candidate = evidence.candidates[best_idx]
            elif evidence.candidates:
                selected_candidate = evidence.candidates[0]

        # Safety: if recovery=REINIT but no candidate available, fall back to SCORE_CANDIDATES
        if recovery == RecoveryAction.REINIT and selected_candidate is None:
            recovery = RecoveryAction.SCORE_CANDIDATES

        bbox_hint: tuple[float, float, float, float] | None = None
        if selected_candidate is not None:
            bbox_hint = selected_candidate.bbox

        action = TrackerAction(
            compute=compute,
            search=search,
            template=template,
            recovery=recovery,
            bbox_hint=bbox_hint,
        )

        return SALTRDDecision(
            action=action,
            risk_probs=risk_probs,
            action_probs=action_logits,
            selected_candidate=selected_candidate,
            model_confidence=float(output.get("confidence", 0.0)),
            safety_fallback_applied=False,
            reason="model_output",
        )

    @staticmethod
    def _decode_enum(logits, enum_cls, default):
        """Argmax decode logits tensor/array/dict -> enum value. Falls back to default on error."""
        if logits is None:
            return default
        try:
            # Convert torch tensor to numpy
            try:
                import torch
                if isinstance(logits, torch.Tensor):
                    logits = logits.detach().cpu().numpy().flatten()
            except ImportError:
                pass
            if isinstance(logits, dict):
                best_key = max(logits, key=logits.__getitem__)
                return enum_cls(best_key)
            # list/array: argmax by position using enum order
            idx = int(np.argmax(logits))
            members = list(enum_cls)
            return members[idx] if idx < len(members) else default
        except (ValueError, IndexError, TypeError, RuntimeError):
            return default

    def _safe_noop(self, reason: str = "") -> SALTRDDecision:
        """Return a safe full-compute no-op action."""
        return SALTRDDecision(
            action=TrackerAction(),  # all defaults: FULL compute, KEEP search, KEEP_CURRENT template, NONE recovery
            safety_fallback_applied=True,
            reason=reason,
        )
