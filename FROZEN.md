/b# FROZEN — src/uav_tracker/ is the baseline

**Date frozen:** 2026-05-19  
**Frozen by:** SALT-RD migration (HANDOFF_NEXT.md)

## What this means

`src/uav_tracker/` is the **frozen SGLATrack/SALT v3 baseline**.

After this file was created, only these changes are permitted:

1. **Config gate wiring** — runtime code reads `enable_ce`, `enable_dynamic`, `enable_velocity_drift`, `enable_salt_rd` from YAML and honors them. No logic changes.
2. **Telemetry extension** — `TrackState` gains `score_map_stats`, `motion_stats`, `flow_stats`, `appearance_stats`, `compute_mode` for SALT-RD feature collection. Fields are zero/empty-defaulted and do not change tracker behaviour.
3. **Bug fixes** — only if a bug would silently corrupt SALT-RD training data.

## What is NOT allowed after this file

- New TSA rules, scene classifiers, or routing logic.
- Changing CE/DYNAMIC/VelocityDrift behaviour (disable via config gate instead).
- Removing CE/DYNAMIC/VelocityDrift code (required for paper ablations).
- Any change that alters benchmark AUC without an explicit GO decision.

## Ablation paths that must NOT be deleted

| Path | Gate | Purpose |
|---|---|---|
| `src/uav_tracker/trackers/sglatrack.py` CE logic | `enable_ce: false` | CE ablation |
| `src/uav_tracker/ml/tsa/` DYNAMIC state | `enable_dynamic: false` | DYNAMIC state ablation |
| `src/uav_tracker/ml/tsa/velocity_drift.py` | `enable_velocity_drift: false` | VelocityDrift ablation |
| `src/uav_tracker/ml/motion_predictor/lstm_predictor.py` | `motion_predictor.enabled: false` | LSTM ablation |

## SALT-RD new code lives in

```
saltr/src/salt_r/          ← all SALT-RD implementation
saltr/configs/             ← SALT-RD configs (do not touch configs/prod/)
```

## Verified baseline state

- pytest: 174 passed, 5 deselected
- `uav-tracker doctor`: passed
- SGLATrack checkpoint: `$UAV_WEIGHTS_ROOT/sglatrack/sglatrack_ep0297.pth.tar`
