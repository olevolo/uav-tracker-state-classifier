# Bug & Dead Code Inventory

Open bugs only. All bugs marked ✅ Fixed in the 2026-05-19 review have been resolved.

---

## MEDIUM — stale infrastructure

### BUG-01: `generate_ml_labels.py` writes all-zero `flow_features` and `iou_trace`
**File:** `scripts/generate_ml_labels.py` lines 285–292
**Impact:** `uav123_labels.npz` has `flow_features/{seq}` = zeros for all 123
sequences. The script was run with KCF which has no APCE/PSR/entropy signals. Any
training that reads these fields silently falls back to proxy values instead of real
tracker responses. `iou_trace/{seq}` is also all zeros.
**Fix:** Run `generate_ml_labels.py` with SGLATrack to populate real
APCE/PSR/entropy into `flow_features`, and compute real IoU into `iou_trace`:
```bash
PYTHONPATH=src .venv/bin/python scripts/generate_ml_labels.py \
    --tracker sglatrack --output data/uav123_labels_sgla.npz
```
**Status:** ⚠️ Partially Fixed (2026-05-19) — `--tracker sglatrack` mode added.
The existing NPZ (generated with KCF) still has all-zero fields and needs regeneration.

---

### BUG-10: `configs/experiments/v2_full_ml.yaml` is a stale architecture config
**File:** `configs/experiments/v2_full_ml.yaml`
**Impact:** References the old V2 pipeline (`ml_scene_scheduler`) that was
superseded by the current architecture. This config cannot be run correctly with
the current codebase.
**Fix:** Delete `configs/experiments/v2_full_ml.yaml` (the archive copy already
exists at `configs/archive/v2_full_ml.yaml`).
**Status:** ⏳ Open — archive copy was created but the original was not removed.
