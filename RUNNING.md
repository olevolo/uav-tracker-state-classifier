# RUNNING.md — CSC Evaluation Framework

For rationale, architecture and Stage-1 gate criteria see `csc_implementation_artifacts/CLAUDE.md`.

---

## Quick cheatsheet

```
make verify                                           # check all 5 trackers
make baseline TRACKER=sglatrack DATASET=got10k SPLIT=val MAX_SEQ=5
make pipeline TRACKER=sglatrack DATASET=got10k SPLIT=val
make all-trackers-on DATASET=got10k SPLIT=val
python -u tools/csc.py                                # list all subcommands
python -u tools/csc.py baseline -h                   # wrapped tool's own help
make help                                             # Makefile target reference
```

---

## 1. Quick verify

Check that every tracker loads its real weights (no stub mode) before any
long-running experiment:

```bash
make verify
# or equivalently:
perl -e 'alarm 900; exec @ARGV' .venv/bin/python -u tools/csc.py verify-trackers
```

Expected output: a table with five rows all showing `OK YES`, then
`PASS — all 5 trackers loaded real weights`.  Any `NO` means a missing
`.pth.tar` file under `~/uav-tracker-weights/<tracker>/`.

---

## 2. Per-tracker quick smoke

Run two sequences on GOT-10k val to confirm the full stack works end-to-end:

```bash
# Baseline only (fast — no CSC)
make baseline TRACKER=sglatrack DATASET=got10k SPLIT=val MAX_SEQ=2

# With CSC (requires a trained checkpoint)
python -u tools/csc.py with-csc \
    --tracker sglatrack --dataset got10k --split val \
    --csc_checkpoint outputs/csc_training/sglatrack_got10k_csc_tcn16/checkpoint_best.pth \
    --max_sequences 2
```

Swap `TRACKER=` for any of: `sglatrack ostrack ortrack avtrack evptrack`.

---

## 3. End-to-end recipe for one (tracker, dataset) pair

```bash
make pipeline TRACKER=sglatrack DATASET=got10k SPLIT=val \
              CSC_CONFIG=configs/csc/csc_tcn16.yaml
```

The pipeline runs eight steps (skip-if-fresh logic avoids re-running completed
steps when you re-invoke it):

| Step | What happens                         | Output path |
|------|--------------------------------------|-------------|
| 1    | verify-trackers                      | (stdout)    |
| 2    | baseline: run tracker                | `outputs/baselines/<tracker>/<dataset>/<split>/` |
| 3    | calibrate: fit percentile calibrators| `outputs/calibration/<tracker>_<dataset>_*.json` |
| 4    | labels: build scene-state labels     | `outputs/csc_labels/<dataset>/<split>/` |
| 5    | train CSC                            | `outputs/csc_training/<tracker>_<dataset>_<cfg>/` |
| 6    | eval-csc: standalone metrics         | `<train_dir>/val_metrics.json` |
| 7    | profile: GFLOPs + latency            | `outputs/profile/<tracker>_<dataset>_<cfg>.json` |
| 8    | gate: Stage-1 criteria               | `<train_dir>/gate_report.json` |

### Approximate wall-clock times

| Machine        | baseline (GOT-10k val, 180 seq) | train (50 ep) | profile (5 seq) |
|----------------|----------------------------------|---------------|-----------------|
| GPU (A100)     | ~25 min                          | ~8 min        | ~2 min          |
| CPU-only       | ~4 h                             | ~35 min       | ~15 min         |

Add `MAX_SEQ=10` to cap sequences for a fast smoke run (~5 min GPU / ~20 min CPU).

---

## 4. Multi-tracker recipe

```bash
make all-trackers-on DATASET=got10k SPLIT=val
# optionally limit sequences:
make all-trackers-on DATASET=got10k SPLIT=val MAX_SEQ=10
```

This loops over all five trackers (`sglatrack ostrack ortrack avtrack evptrack`)
and calls `make baseline` for each in turn.

**Important:** each tracker runs in its own Python subprocess.  The tracker
libraries (`papers/code/<tracker>/lib`) all export identically named packages,
so two trackers cannot coexist in one Python process.  The loop is therefore
*sequential by design* — do not attempt to parallelise it.

After baselines, run `make pipeline` once per tracker to complete training:

```bash
for T in sglatrack ostrack ortrack avtrack evptrack; do
  make pipeline TRACKER=$T DATASET=got10k SPLIT=val
done
```

---

## 5. Output directory structure

```
outputs/
  baselines/<tracker>/<dataset>/<split>/
    predictions/<seq>.txt        ← frame-by-frame bbox
    telemetry/<seq>.jsonl        ← confidence / APCE / PSR per frame
    manifest.json                ← run metadata (FPS, git commit, …)

  calibration/
    <tracker>_<dataset>_confidence.json
    <tracker>_<dataset>_apce.json
    <tracker>_<dataset>_psr.json
    <tracker>_<dataset>.manifest.json

  csc_labels/<dataset>/<split>/
    labels.jsonl                 ← all frames, flat
    labels_per_sequence/<seq>.jsonl
    label_stats.json

  csc_training/<tracker>_<dataset>_<cfg_stem>/
    checkpoint_best.pth
    checkpoint_last.pth
    val_metrics.json
    gate_report.json
    training_log.jsonl

  csc_runs/<tracker>_<dataset>_<split>_<ckpt_stem>/
    predictions/<seq>.txt
    telemetry/<seq>.jsonl
    states/<seq>.jsonl           ← per-frame CSC output
    metrics.json

  profile/
    <tracker>_<dataset>_<cfg_stem>.json

  audit/
    <tracker>_<dataset>_<split>.png
    <tracker>_<dataset>_<split>_dist.csv

  _pipeline_<tracker>_<dataset>_<split>_<cfg_stem>/
    .pipeline_state.json         ← skip-if-fresh timestamps
```

---

## 6. How to add a new tracker

1. Copy an existing adapter, e.g.
   `salrtd/src/uav_tracker/trackers/ostrack.py` → `…/mytracker.py`.
2. Implement the three methods: `init(frame, bbox)`, `update(frame) → TrackState`,
   and decorate with `@TRACKERS.register("mytracker")`.
3. Add `"mytracker"` to the `_TRACKER_NAMES` list in:
   - `tools/run_baseline.py`
   - `tools/run_with_csc.py`
   - `tools/_verify_no_stubs.py` (the `for name in (…)` loop)
   - `tools/csc.py` (`--tracker choices=` in the pipeline parser)
   - `Makefile` (`TRACKERS_ALL` variable)
4. Place weights under `~/uav-tracker-weights/mytracker/<file>.pth.tar`.
5. Run `make verify` to confirm the tracker loads cleanly.

Existing adapters to use as templates:
- `salrtd/src/uav_tracker/trackers/ostrack.py` — attention-based tracker
- `salrtd/src/uav_tracker/trackers/avtrack.py` — lightweight CNN tracker

---

## UAV123 Final Evaluation

**Critical protocol constraint:** UAV123 is the final test set only.  Never
train CSC on UAV123.  Never tune thresholds directly on UAV123.

### Prerequisites

| Item | Path | Status |
|------|------|--------|
| UAV123 baseline predictions | `outputs/baselines/<tracker>/uav123/test/predictions/` | Required |
| UAV123 baseline telemetry | `outputs/baselines/<tracker>/uav123/test/telemetry/` | Required |
| Calibrator (ORTrack) | `outputs/calibration/ortrack_lasot_confidence.json` | Required |
| Calibrator (SGLATrack) | `outputs/calibration/sglatrack_got10k_confidence.json` | Required — but GOT-10k based (Issue 5) |
| CSC checkpoint | trained on LaSOT / GOT-10k, NOT UAV123 | Required |

### Run the pipeline

```bash
# 1. Set the safety gate (acknowledges no UAV123 training contamination)
export CSC_NOT_TRAINED_ON_UAV123=1

# 2. Run for ORTrack (recommended first — has LaSOT calibrator + LaSOT-trained checkpoint)
bash tools/run_uav123_final_eval.sh ortrack \
    outputs/csc_training/ortrack_lasot_tcn16/checkpoint_best.pth

# 3. Run for SGLATrack (GOT-10k calibrator — see calibration warning in logs)
bash tools/run_uav123_final_eval.sh sglatrack \
    <path_to_sglatrack_checkpoint>
```

The script runs 8 steps (skip-if-fresh logic).  Each step writes to:

```
outputs/eval/<tracker>/uav123/test/
  passive/                     ← CSC passive inference (states/, predictions/, telemetry/, metrics.json)
  labels/                      ← GT scene-state labels (eval-only, never used for training)
  tracking_metrics/            ← AUC, Precision@20, FPS
  episode_metrics/             ← Recall@5/10, FA/1000, mean detection delay
  paper_metrics/               ← FCR, FCD, TTFC, Recovery@30, State-Cond AUC
    paper_metrics.csv
    QUALITY_REPORT.md          ← paste-ready paper tables
    state_transition_matrix.csv
    state_conditioned_auc.csv
  FINAL_REPORT.md              ← assembled report with all tables
```

### Metric definitions

| Metric | Definition |
|--------|-----------|
| FCR | N_false_confirmed_frames / N_total_frames |
| FCD | Mean length (frames) of contiguous FALSE_CONFIRMED segments |
| TTFC | Mean(t_first_FC - t_last_CONFIRMED) per sequence |
| Recovery@30 | Fraction of FC episodes where state returns to CORRECT_CONFIRMED within 30 frames |
| UUR proxy | Fraction of frames where FC flag=True and CSC did NOT recommend skipping update |
| State-Conditioned AUC | AUC computed on the subset of frames predicted in each state |

### Known gaps and risks

1. **SGLATrack telemetry schema on UAV123 is incomplete.** SGLATrack emits only
   `confidence` (no `apce`, no `psr`) in UAV123 telemetry files.  The feature
   builder fills `apce=0, psr=0` silently.  This degrades CSC prediction quality
   for SGLATrack.  ORTrack emits the full schema.

2. **SGLATrack calibrator is GOT-10k, not LaSOT (Issue 5 in memory).**
   The confidence scale for SGLATrack on GOT-10k is `[0.012, 0.019]` vs ORTrack
   LaSOT `[0.45, 0.97]`.  This makes the `conf_threshold` meaningless for
   SGLATrack without re-fitting on LaSOT.  The LaSOT baseline for SGLATrack is
   currently running (PID 14213); re-fit calibration once it finishes.

3. **UUR requires tracker-side hook.**  True UUR needs to know when the tracker
   actually updated its template.  Only CSC's `should_skip_template_update`
   recommendation is currently logged.  The UUR value in the report is a proxy.

4. **UAV123@10fps is NOT yet supported.**  No loader, no dataset config entry.
   Must be added before final paper submission.  Register as a new dataset
   identical to UAV123 but loading every 10th frame.

5. **TTFC and FC-related metrics will be 0 / N/A** if the current checkpoint
   (FC_recall=0, see memory) fails to predict any FALSE_CONFIRMED frames.
   These metrics require a properly trained CSC checkpoint.  Re-run once a
   better checkpoint is available.

---

## DRY_RUN mode

Append `DRY_RUN=1` to any `make` command to print commands without running them:

```bash
make pipeline TRACKER=sglatrack DATASET=got10k DRY_RUN=1
```
