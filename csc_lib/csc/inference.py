"""Online CSC inference wrapper for runtime use (composite outputs).

Wraps a trained :class:`CSCGRU` and exposes ``step(telemetry)`` that
takes per-frame telemetry, builds a causal feature, runs the model on
the rolling window and returns the predicted localization /
confidence states, the derived paper-state, the failure-risk score
and control hints.

No GT.  No future frames.  Strictly causal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from csc_lib.csc.config import CSCFeatureConfig, CSCTrainConfig
from csc_lib.csc.features import FEATURE_DIM, _State, build_runtime_feature, build_runtime_feature_into
from csc_lib.csc.labeling.label_schema import (
    AUX_FLAGS,
    ConfidenceState,
    DerivedState,
    LocalizationState,
    derive_state,
)
from csc_lib.csc.model import CSCGRU, LegacyCSCGRU, build_model

log = logging.getLogger(__name__)


@dataclass
class CSCPrediction:
    localization_probs: np.ndarray              # (3,)
    confidence_probs: np.ndarray                # (2,)
    predicted_localization: int                 # LocalizationState
    predicted_confidence: int                   # ConfidenceState
    derived_state: int                          # DerivedState (paper-class)
    risk_score: float                           # P(LOST)
    false_confirmed_flag: bool
    aux_probs: dict[str, float]
    should_freeze_template: bool = False
    should_expand_search: bool = False
    should_request_redetection: bool = False
    should_skip_template_update: bool = False
    latency_ms: float = 0.0
    # ---- V3 proactive forecast outputs (None when forecast heads disabled) ----
    failure_next_10_prob: Optional[float] = None
    false_confirmed_next_10_prob: Optional[float] = None
    lost_aware_next_10_prob: Optional[float] = None


@dataclass
class CSCControlPolicy:
    """Decision thresholds that map (loc, conf, aux, risk) → control hints."""

    risk_threshold: float = 0.5

    def freeze_template(self, derived: int, risk: float) -> bool:
        # Freeze when we believe the tracker is in a wrong-but-confident
        # state, or simply lost.
        return derived in (
            int(DerivedState.LOST_AWARE),
            int(DerivedState.FALSE_CONFIRMED),
        ) or risk >= self.risk_threshold

    def expand_search(self, derived: int) -> bool:
        return derived in (
            int(DerivedState.LOST_AWARE),
            int(DerivedState.CORRECT_UNCERTAIN),
        )

    def redetect(self, derived: int) -> bool:
        # Strongest hint to ask a detector for help.
        return derived in (int(DerivedState.LOST_AWARE), int(DerivedState.FALSE_CONFIRMED))


class CSCRuntime:
    """Causal online inference engine."""

    def __init__(
        self,
        model: CSCGRU,
        feature_cfg: CSCFeatureConfig,
        *,
        image_size: tuple[int, int] = (1280, 720),
        policy: Optional[CSCControlPolicy] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.feature_cfg = feature_cfg
        self.image_size = image_size
        self.policy = policy or CSCControlPolicy()
        self.device = device

        T = feature_cfg.window_size
        # Step 4: persistent window — numpy for safe in-place shift (memmove), shared view for torch
        self._window_np: np.ndarray = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        # Persistent torch view of the numpy array — no per-step allocation on CPU
        self._window_view: torch.Tensor = torch.from_numpy(self._window_np).unsqueeze(0)
        self._window_count: int = 0
        # Step 5: pre-allocated feature buffer — no np.array allocation per frame
        self._feat_buf: np.ndarray = np.zeros(FEATURE_DIM, dtype=np.float32)
        self._state = _State()
        self._traced_fn = None  # set by _jit_trace() after load

        self._cal_apce = None
        self._cal_psr = None
        self._cal_conf = None

    def _jit_trace(self) -> None:
        """Step 7: JIT-trace the model for faster CPU inference.

        Creates a traced module with last_step_only baked in and stores it
        as self._traced_fn.  step() uses it directly if present.
        Falls back silently if tracing fails.

        V3 forecast heads disable JIT tracing — the eager path through
        ``self.model.predict()`` returns forecast probabilities, which the
        traced wrapper does not expose.  CSC step latency is still well
        within budget without the trace (≤ 1 ms on CPU).
        """
        if getattr(self.model, "enable_forecast", False):
            log.info(
                "CSCRuntime: forecast heads enabled — skipping JIT trace, using eager path"
            )
            self._traced_fn = None
            return
        T = self.feature_cfg.window_size
        dummy = torch.zeros(1, T, FEATURE_DIM, dtype=torch.float32, device=self.device)
        try:
            # Temporarily disable grad on all parameters (required for tracing)
            orig_requires_grad = {n: p.requires_grad for n, p in self.model.named_parameters()}
            self.model.requires_grad_(False)

            with torch.inference_mode():
                # Wrap in a module to avoid lambda closure issues with parameters
                class _LastStepModule(torch.nn.Module):
                    def __init__(self_, inner):  # noqa: N805
                        super().__init__()
                        self_.inner = inner
                    def forward(self_, x: torch.Tensor):  # noqa: N805
                        o = self_.inner.forward(x, last_step_only=True)
                        return (o.localization_logits, o.confidence_logits,
                                o.aux_logits, o.derived_logits)

                wrapper = _LastStepModule(self.model)
                traced = torch.jit.trace(wrapper, (dummy,))

            # Restore grad flags
            for n, p in self.model.named_parameters():
                p.requires_grad_(orig_requires_grad[n])

            self._traced_fn = traced
            log.info("CSCRuntime: JIT-traced forward (last_step_only=True baked in)")
        except Exception as exc:
            self._traced_fn = None
            log.warning("CSCRuntime: JIT trace failed, using eager mode: %s", exc)

    # ------------------------------------------------------------------
    # Calibrator attachment
    # ------------------------------------------------------------------

    def attach_calibrators(
        self,
        *,
        apce_calibrator=None,
        psr_calibrator=None,
        confidence_calibrator=None,
    ) -> "CSCRuntime":
        """Attach pre-trained percentile calibrators for raw APCE/PSR/confidence.

        When a calibrator is attached, the corresponding raw value is mapped to
        [0, 1] via the empirical CDF before the clip_value clamp is applied.

        This corrects for tracker-specific scale mismatches between training
        (e.g. LaSOT) and test (e.g. DTB70) datasets.

        Parameters
        ----------
        apce_calibrator:
            :class:`~csc_lib.csc.calibration.PercentileFeatureCalibrator` for APCE.
        psr_calibrator:
            :class:`~csc_lib.csc.calibration.PercentileFeatureCalibrator` for PSR.
        confidence_calibrator:
            :class:`~csc_lib.csc.calibration.PercentileConfidenceCalibrator` for
            raw tracker confidence.

        Returns
        -------
        self  (for chaining)
        """
        self._cal_apce = apce_calibrator
        self._cal_psr = psr_calibrator
        self._cal_conf = confidence_calibrator
        active = [
            name
            for name, cal in (
                ("apce", apce_calibrator),
                ("psr", psr_calibrator),
                ("confidence", confidence_calibrator),
            )
            if cal is not None
        ]
        log.info("CSCRuntime: calibrators attached for features: %s", active or "(none)")
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calibrate(self, value: Optional[float], calibrator) -> Optional[float]:
        """Apply a loaded calibrator to a single raw value, returning float in [0, 1].

        Returns *value* unchanged when calibrator is None or value is None.
        """
        if calibrator is None or value is None:
            return value
        try:
            result = float(calibrator.transform(float(value)))
            # Clamp to [0, 1] — piecewise-linear interp can produce tiny
            # out-of-range values at the distribution tails.
            return max(0.0, min(1.0, result))
        except Exception:
            return value

    def reset(self, image_size: Optional[tuple[int, int]] = None) -> None:
        self._window_np.fill(0.0)
        self._window_count = 0
        self._state = _State()
        if image_size is not None:
            self.image_size = image_size

    def step(
        self,
        *,
        confidence: Optional[float] = None,
        apce: Optional[float] = None,
        psr: Optional[float] = None,
        pred_bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> CSCPrediction:
        import time

        t0 = time.perf_counter()

        # Calibrate raw values → [0, 1] percentile
        confidence_cal = self._calibrate(confidence, self._cal_conf)
        apce_cal       = self._calibrate(apce,       self._cal_apce)
        psr_cal        = self._calibrate(psr,        self._cal_psr)

        # Step 5: write feature into pre-allocated buffer (no np.array alloc)
        build_runtime_feature_into(
            self._feat_buf,
            confidence=confidence_cal,
            apce=apce_cal,
            psr=psr_cal,
            pred_bbox=pred_bbox,
            image_size=self.image_size,
            state=self._state,
        )
        np.clip(self._feat_buf, -self.feature_cfg.clip_value, self.feature_cfg.clip_value, out=self._feat_buf)

        # Step 4: ring-buffer window update via numpy (safe memmove for overlapping shift)
        if self._window_count == 0:
            self._window_np[:] = self._feat_buf      # first frame: fill all slots (causal pad)
        else:
            self._window_np[:-1] = self._window_np[1:]  # numpy memmove handles overlap correctly
            self._window_np[-1]  = self._feat_buf
        self._window_count += 1

        # _window_view is a persistent torch tensor sharing numpy memory — no copy on CPU
        x = self._window_view
        if self.device != "cpu":
            x = x.to(self.device)

        # V3 forecast outputs (None unless eager path runs and model has forecast heads)
        forecast_failure_val: Optional[float] = None
        forecast_fc_val: Optional[float] = None
        forecast_lost_val: Optional[float] = None

        with torch.inference_mode():
            if self._traced_fn is not None:
                loc_logits, conf_logits, aux_logits, der_logits = self._traced_fn(x)
                import torch.nn.functional as _F
                loc_probs_t  = _F.softmax(loc_logits[0, 0],  dim=-1)
                conf_probs_t = _F.softmax(conf_logits[0, 0], dim=-1)
                der_probs_t  = _F.softmax(der_logits[0, 0],  dim=-1)
                aux_probs_t  = torch.sigmoid(aux_logits[0, 0])
                risk_t       = der_probs_t[2:3] + der_probs_t[3:4]
                packed_np    = torch.cat([loc_probs_t, conf_probs_t, der_probs_t, aux_probs_t, risk_t]).cpu().numpy()
            else:
                out = self.model.predict(x, last_step_only=True)
                loc_p  = out["localization_probs"][0, 0]
                conf_p = out["confidence_probs"][0, 0]
                der_p  = out["derived_probs"][0, 0]
                aux_p  = out["aux_probs"][0, 0]
                risk_t = out["risk_score"][0:1, 0]
                packed_np = torch.cat([loc_p, conf_p, der_p, aux_p, risk_t]).cpu().numpy()
                # V3 forecast probs (last step). out values may be missing for V2 models.
                fail_p = out.get("failure_next_10_prob")
                fc_n10_p = out.get("false_confirmed_next_10_prob")
                lost_n10_p = out.get("lost_aware_next_10_prob")
                forecast_failure_val = float(fail_p[0, 0].cpu().item()) if fail_p is not None else None
                forecast_fc_val      = float(fc_n10_p[0, 0].cpu().item()) if fc_n10_p is not None else None
                forecast_lost_val    = float(lost_n10_p[0, 0].cpu().item()) if lost_n10_p is not None else None
        loc_probs_np  = packed_np[0:3]
        conf_probs_np = packed_np[3:5]
        der_probs_np  = packed_np[5:9]
        aux_probs_np  = packed_np[9:14]
        risk_val      = float(packed_np[14])

        loc_idx  = int(np.argmax(loc_probs_np))
        conf_idx = int(np.argmax(conf_probs_np))

        derived = derive_state(LocalizationState(loc_idx), ConfidenceState(conf_idx))
        fc = derived == DerivedState.FALSE_CONFIRMED

        freeze    = self.policy.freeze_template(int(derived), risk_val)
        expand    = self.policy.expand_search(int(derived))
        redetect  = self.policy.redetect(int(derived))
        skip_update = freeze or fc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return CSCPrediction(
            localization_probs=loc_probs_np,
            confidence_probs=conf_probs_np,
            predicted_localization=loc_idx,
            predicted_confidence=conf_idx,
            derived_state=int(derived),
            risk_score=risk_val,
            false_confirmed_flag=bool(fc),
            aux_probs={name: float(p) for name, p in zip(AUX_FLAGS, aux_probs_np)},
            should_freeze_template=freeze,
            should_expand_search=expand,
            should_request_redetection=redetect,
            should_skip_template_update=skip_update,
            latency_ms=latency_ms,
            failure_next_10_prob=forecast_failure_val,
            false_confirmed_next_10_prob=forecast_fc_val,
            lost_aware_next_10_prob=forecast_lost_val,
        )


def _is_legacy_checkpoint(state_dict: dict) -> bool:
    """Return True if the checkpoint was saved with the old 2-head V0 architecture."""
    return "head_state.weight" in state_dict and "head_risk.weight" in state_dict


def load_runtime(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    image_size: tuple[int, int] = (1280, 720),
    calibration_dir: Optional[Path] = None,
    tracker_name: Optional[str] = None,
) -> CSCRuntime:
    """Load a :class:`CSCRuntime` from a checkpoint file.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` checkpoint produced by ``train_csc.py``.
    device:
        Torch device string.
    image_size:
        ``(width, height)`` of the input video frames.  Overridden per-sequence
        by :py:meth:`CSCRuntime.reset`.
    calibration_dir:
        Optional directory containing pre-saved calibrator JSON files.
        When provided (and the files exist), percentile calibrators for APCE,
        PSR, and confidence are loaded and attached to the runtime so that
        raw feature values are mapped to [0, 1] before the clip_value clamp.
        File name convention::

            <calibration_dir>/<tracker_name>_<dataset>_apce.json
            <calibration_dir>/<tracker_name>_<dataset>_psr.json
            <calibration_dir>/<tracker_name>_<dataset>_confidence.json

        The checkpoint stem is used to infer ``<tracker_name>_<dataset>``
        when ``tracker_name`` is not explicitly given.  Example: checkpoint
        stem ``ortrack_lasot_tcn16`` → prefix ``ortrack_lasot``.
    tracker_name:
        Override the tracker/dataset prefix used when searching
        ``calibration_dir`` (e.g. ``"ortrack_lasot"``).  Ignored when
        ``calibration_dir`` is None.
    """
    blob = torch.load(checkpoint_path, map_location=device)
    cfg = CSCTrainConfig.from_dict(blob["config"])
    cfg.model.feature_dim = FEATURE_DIM

    state_dict = blob["state_dict"]
    if _is_legacy_checkpoint(state_dict):
        # V0 checkpoint: head_state (6-class) + head_risk (1-class binary).
        # Load via LegacyCSCGRU which remaps old heads to current CSCOutput.
        import warnings
        warnings.warn(
            f"[CSC] Loading legacy V0 checkpoint from {checkpoint_path}. "
            "head_state/head_risk will be remapped to current 3-head format. "
            "Results are EXPLORATORY — retrain on the new label schema for "
            "production use.",
            UserWarning,
            stacklevel=2,
        )
        model = LegacyCSCGRU(cfg.model)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = build_model(cfg.model)
        model.load_state_dict(state_dict)

    runtime = CSCRuntime(
        model=model,
        feature_cfg=cfg.feature,
        image_size=image_size,
        device=device,
    )

    # --- Lazily attach calibrators when calibration_dir is given ---
    if calibration_dir is not None:
        _attach_calibrators_from_dir(
            runtime,
            calibration_dir=Path(calibration_dir),
            checkpoint_path=Path(checkpoint_path),
            tracker_name=tracker_name,
        )

    # Step 7: JIT-trace the model forward for faster inference
    runtime._jit_trace()

    return runtime


def _attach_calibrators_from_dir(
    runtime: CSCRuntime,
    *,
    calibration_dir: Path,
    checkpoint_path: Path,
    tracker_name: Optional[str],
) -> None:
    """Try to load calibrators from *calibration_dir* and attach them to *runtime*.

    The naming convention is ``<prefix>_apce.json``, ``<prefix>_psr.json``,
    ``<prefix>_confidence.json``.  The prefix is either *tracker_name* (when
    given) or inferred from the checkpoint path.

    Inference order for the prefix (first non-empty match wins):
    1. *tracker_name* argument (explicit override).
    2. The checkpoint's **parent directory name** with model-type suffixes
       stripped — e.g. ``outputs/csc_training/ortrack_lasot_tcn16/
       checkpoint_best.pth`` → parent stem ``ortrack_lasot_tcn16`` →
       prefix ``ortrack_lasot``.
    3. The checkpoint **file stem** with model-type suffixes stripped — e.g.
       ``ortrack_lasot_tcn16.pth`` → prefix ``ortrack_lasot``.

    Missing files are silently skipped — calibration is best-effort.
    """
    from csc_lib.csc.calibration import (
        PercentileConfidenceCalibrator,
        PercentileFeatureCalibrator,
    )
    import re

    # Model-type suffix pattern to strip: _tcn16, _tcn32, _gru, _mlp, _smoke*, etc.
    _MODEL_SUFFIX_RE = re.compile(r"_(tcn\d+|gru\d*|mlp|smoke.*)$", re.IGNORECASE)

    def _strip_model_suffix(s: str) -> str:
        return _MODEL_SUFFIX_RE.sub("", s)

    if tracker_name:
        prefix = tracker_name
    else:
        # Try parent directory name first (most reliable for checkpoints named
        # ``checkpoint_best.pth`` inside a run directory like ``ortrack_lasot_tcn16/``).
        parent_name = checkpoint_path.parent.name
        parent_prefix = _strip_model_suffix(parent_name)

        # Fall back to checkpoint file stem if parent prefix looks like a plain
        # checkpoint name (e.g. parent is "csc_training" and stem carries info).
        file_prefix = _strip_model_suffix(checkpoint_path.stem)

        # Use whichever prefix is more specific (longer) and not a generic name.
        _generic = {"checkpoint_best", "checkpoint", "model", "best"}
        if parent_prefix.lower() in _generic or not parent_prefix:
            prefix = file_prefix
        else:
            prefix = parent_prefix

    cal_apce = None
    cal_psr = None
    cal_conf = None

    apce_path = calibration_dir / f"{prefix}_apce.json"
    if apce_path.exists():
        try:
            cal_apce = PercentileFeatureCalibrator.load(apce_path)
            log.info("CSCRuntime: loaded APCE calibrator from %s", apce_path)
        except Exception as exc:
            log.warning("CSCRuntime: failed to load APCE calibrator %s: %s", apce_path, exc)

    psr_path = calibration_dir / f"{prefix}_psr.json"
    if psr_path.exists():
        try:
            cal_psr = PercentileFeatureCalibrator.load(psr_path)
            log.info("CSCRuntime: loaded PSR calibrator from %s", psr_path)
        except Exception as exc:
            log.warning("CSCRuntime: failed to load PSR calibrator %s: %s", psr_path, exc)

    conf_path = calibration_dir / f"{prefix}_confidence.json"
    if conf_path.exists():
        try:
            cal_conf = PercentileConfidenceCalibrator.load(conf_path)
            log.info("CSCRuntime: loaded confidence calibrator from %s", conf_path)
        except Exception as exc:
            log.warning("CSCRuntime: failed to load confidence calibrator %s: %s", conf_path, exc)

    if cal_apce is None and cal_psr is None and cal_conf is None:
        log.info(
            "CSCRuntime: no calibrators found in %s with prefix %r — "
            "using raw feature values (may degrade on out-of-distribution data).",
            calibration_dir,
            prefix,
        )
    else:
        runtime.attach_calibrators(
            apce_calibrator=cal_apce,
            psr_calibrator=cal_psr,
            confidence_calibrator=cal_conf,
        )
