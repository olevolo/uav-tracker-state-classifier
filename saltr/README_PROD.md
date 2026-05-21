# SALT-RD — Production Reference

SALT-RD (Scene-Adaptive Learning of Tracking Reliability and Dynamicity) is a
GRU-based trust controller that runs alongside SGLATrack and gates template
updates and reinit decisions.

## Architecture (current)

No TSA. The controller loop is:

```
SGLATrack.update_with_action(frame, bbox)
    └── EvidenceExtractor  ← extracts 28-dim telemetry per frame
         └── SALTRDController  ← GRU forward, emits risk scores + action
```

The controller produces per-frame risk scores:

- **p_false_confirmed**: probability the tracker is drifted but still scoring high
- **p_imminent_failure_dynamic_10/20**: probability of track loss within 10/20 frames
- **p_recoverable**: probability the target is still locatable

Action dispatch via `stage3_policy()` using `SALTRDState` enum
(`TRUSTED`, `LOW_EVIDENCE`, `FALSE_CONFIRMED_RISK`, `REACQUIRE_NEEDED`).

## Production checkpoint

```
saltr/checkpoints/production_no_flow/saltrd_best.pt
```

| Property | Value |
|----------|-------|
| Architecture | GRU (hidden=64, layers=2, input=28) |
| Val fc AUROC | **0.885** |
| Diag fc AUROC | 0.598 |
| Stage 2 gate | wrir=0, msu=0.081, coverage=65.7% |

Oracle action audit: reinit policy +0.083 hard AUC → **BUILD policy confirmed**.

See `RESULTS.md` for full benchmark results across UAV123 / VisDrone / DTB70.

## LODO generalization checkpoints

```
saltr/checkpoints/lodo_no_uav123/saltrd_best.pt
saltr/checkpoints/lodo_no_dtb70/saltrd_best.pt
saltr/checkpoints/lodo_no_visdrone/saltrd_best.pt
```

## Key commands

### Advisory shadow eval (val split)

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.shadow_mode \
    --npz saltr/data/salt_rd_v2_labels.npz \
    --checkpoint saltr/checkpoints/production_no_flow/saltrd_best.pt \
    --split val \
    --advisory \
    --output saltr/results/shadow_mode_val.json
```

### Policy sweep

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.policy_sweep \
    --preds saltr/results/preds_all_v2_oof_teacher.json \
    --output saltr/results/policy_sweep.json
```

### Run benchmark with SALT-RD controller

```bash
.venv/bin/python scripts/fast_bench.py \
    --dataset uav123 \
    --advisory saltr/checkpoints/production_no_flow/saltrd_best.pt \
    --output outputs/bench_uav123_saltrd.json
```

### Attach controller to tracker (code)

```python
from salt_r.controller import SALTRDController

controller = SALTRDController(
    checkpoint="saltr/checkpoints/production_no_flow/saltrd_best.pt",
    device="cpu",
)
tracker.set_salt_rd_controller(controller)
# update_with_action() runs EvidenceExtractor + controller each frame
```

## Canonical eval results

```
saltr/results/eval_val_v2_retrained.json         — production val eval
saltr/results/eval_diagnostic_v2_retrained.json  — production diagnostic eval
saltr/results/oracle_action_audit.json           — oracle reinit audit (+0.083 hard AUC)
saltr/results/preds_all_v2_oof_teacher.json      — OOF predictions (canonical)
```

## Killed experiments (archived)

Experiments that were tried and killed are in `saltr/archive/killed/`:
- **v2.1 memory** — required GT sidecar, not online-deployable (KILL)
- **v2.2 sgla pos/peak** — diag AUROC 0.584/0.567 < baseline 0.598 (KILL)
- **v2.3 point** — diag AUROC 0.546 < baseline 0.598 (KILL)
- Pilot scripts: dino_identity, candidate_mining, point_teacher, point_sidecar, sgla_memory_extractor
