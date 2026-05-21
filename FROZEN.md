# FROZEN — Baseline and Permanent Design Decisions

**Last updated:** 2026-05-22

## Frozen design decisions

| Decision | Rule |
|---|---|
| No online Farneback flow in production | Flow indices 22–27 are zeroed by `zero_production_features()`. Farneback is offline teacher only. |
| CE layer 3 only, kr=0.50 | Multi-stage CE was tested and regressed. `_CE_LOC={3}` is the correct config. |
| LSTM disabled permanently | OnlineLSTMMotionPredictor showed zero benefit and false high-dynamic states. |
| No runtime thresholds on APCE/p_fc | All tracking decisions must come from learned action heads. |
| TSA permanently deleted | `src/uav_tracker/ml/tsa/` archived. No re-introduction allowed. |
| No center-freeze | Oracle audit: +0.000 hard AUC. Phase 7 regression: -0.036. Permanently killed. |
| Dynamic template update disabled | Oracle AUPRC too low, car7 regression risk. Stays off until learned template head passes gates. |

## What is allowed in src/uav_tracker/

- Bug fixes that would corrupt SALT-RD training data
- Adding telemetry fields to `TrackState` (zero-defaulted, no behavior change)
- `update_with_action(frame, action: TrackerAction)` control path

## What is NOT allowed

- New hand-coded thresholds for tracking state decisions
- Reintroducing TSA, DYNAMIC state, or VelocityDrift
- Any change that alters benchmark AUC without an explicit GO decision

## Ablation paths still present

| Component | Location | Status |
|---|---|---|
| CE pruning | `sglatrack.py` — `update_with_action` warns for non-FULL | disabled by default |
| LSTM | `src/uav_tracker/ml/motion_predictor/` | disabled permanently |

## Verified baseline (2026-05-22)

- 449 unit tests passing
- SALT-RD production_no_flow checkpoint: val fc AUROC 0.885
- SGLATrack checkpoint: `$UAV_WEIGHTS_ROOT/sglatrack/sglatrack_ep0297.pth.tar`
