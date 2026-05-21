# SALT-RD — Production Reference

SALT-RD (Scene-Adaptive Learning of Tracking Reliability and Dynamicity) is a
GRU-based trust controller that runs alongside SGLATrack and gates template
updates and reinit decisions.

## What SALT-RD does

SALT-RD computes a 28-dim telemetry feature vector per frame from SGLATrack
outputs (APCE, PSR, IoU, confidence streaks, motion signals, etc.) and runs it
through a trained GRU model to produce per-frame risk scores:

- **p_false_confirmed**: probability the tracker is drifted but still scoring high
- **p_imminent_failure_dynamic_10/20**: probability of track loss within 10/20 frames
- **p_recoverable**: probability the target is still locatable

The `SALTRDAdvisor` wraps the model and exposes a `should_block_template_update()`
gate, implementing the 5-gate template guard protocol.

## Production checkpoint

```
saltr/checkpoints/production/saltrd_best.pt
```

| Property | Value |
|----------|-------|
| Architecture | GRU (hidden=64, layers=2, input=28) |
| Val AUROC | **0.885** |
| Diag AUROC | 0.598 |
| Stage 2 advisory | val cov=65.7%, FAR=8.5%, WRIR=0, MSU=0.081 |

See `saltr/checkpoints/production/README.md` for full details.

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
    --checkpoint saltr/checkpoints/production/saltrd_best.pt \
    --memory-sidecar saltr/data/salt_rd_memory_sidecar.npz \
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

### Run benchmark with advisory

```bash
.venv/bin/python scripts/fast_bench.py \
    --dataset uav123 \
    --advisory saltr/checkpoints/production/saltrd_best.pt \
    --output outputs/bench_uav123_advisory.json
```

### Attach advisor to tracker (code)

```python
from salt_r.advisor import SALTRDAdvisor

advisor = SALTRDAdvisor(
    checkpoint="saltr/checkpoints/production/saltrd_best.pt",
    device="cpu",
)
tracker.set_salt_rd_advisor(advisor)

# Each tracking step — advisor.step() is called automatically inside
# tracker.update_with_state(). Before template update:
if not advisor.should_block_template_update():
    tracker.try_update_template(frame, bbox, apce, psr, frame_idx, cosine_sim)
```

## Canonical eval results

```
saltr/results/eval_val_v2_retrained.json        — production val eval
saltr/results/eval_diagnostic_v2_retrained.json — production diagnostic eval
saltr/results/shadow_mode_val_v2_retrained_advisory.json — Stage 2 GO result
saltr/results/shadow_mode_val_v2_retrained_5gate.json    — 5-gate advisory
saltr/results/preds_all_v2_oof_teacher.json     — OOF predictions (canonical)
```

## Killed experiments (archived)

Experiments that were tried and killed are in `saltr/archive/killed/`:
- **v2.1 memory** — required GT sidecar, not online-deployable (KILL)
- **v2.2 sgla pos/peak** — diag AUROC 0.584/0.567 < baseline 0.598 (KILL)
- **v2.3 point** — diag AUROC 0.546 < baseline 0.598 (KILL)
- Pilot scripts: dino_identity, candidate_mining, point_teacher, point_sidecar, sgla_memory_extractor
