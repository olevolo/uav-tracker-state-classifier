#!/bin/bash
# V4 retrain pipeline (self-driving): wait for re-extraction → build v4 shards →
# train CSCv4 on MPS (Apple GPU) → quick diagnose. Launch in background after the
# two re-extract jobs are running. Idempotent-ish (overwrites its outputs).
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
LOG=outputs/_logs/v4_retrain_pipeline.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "STAGE 0: wait for re-extraction (lasot/train + uavdt_sot/test = last of each group)"
deadline=$(( $(date +%s) + 18000 ))   # 5h safety cap
while true; do
  ldone=$(grep -c 'done:' outputs/_logs/reextract_lasot_train.log 2>/dev/null || echo 0)
  udone=$(grep -c 'done:' outputs/_logs/reextract_uavdt_sot_test.log 2>/dev/null || echo 0)
  laN=$(find outputs/baselines_v4/sglatrack/lasot -path '*telemetry*' -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$ldone" -ge 1 ] && [ "$udone" -ge 1 ]; then say "extraction complete (lasot=$laN seqs)"; break; fi
  [ "$(date +%s)" -ge "$deadline" ] && { say "TIMEOUT waiting for extraction; proceeding with what exists (lasot=$laN)"; break; }
  sleep 60
done

say "STAGE 1: build v4 training shards from baselines_v4 telemetry"
$PY tools/v4_build_labels.py --telemetry_root outputs/baselines_v4 \
  --out outputs/csc_labels_v4/train_shards.jsonl \
  --calib_out outputs/csc_labels_v4/v4_feature_calibrators.json >> "$LOG" 2>&1 \
  || { say "FAIL build_labels"; exit 1; }
rows=$(wc -l < outputs/csc_labels_v4/train_shards.jsonl | tr -d ' ')
fc_rows=$($PY - <<'PY'
import json
n = 0
with open("outputs/csc_labels_v4/train_shards.jsonl") as fh:
    for line in fh:
        if line.strip() and int(json.loads(line).get("derived", -1)) == 3:
            n += 1
print(n)
PY
)
say "shards: ${rows} rows; FC rows=${fc_rows}"
[ "$fc_rows" -le 0 ] && { say "FAIL build_labels: zero FC rows in V4 shard"; exit 1; }

say "STAGE 2: train CSCv4 on MPS (25 epochs)"
$PY tools/train_csc_v4.py --shards outputs/csc_labels_v4/train_shards.jsonl \
  --out_dir outputs/csc_training_v4/csc_v4_r1 --device mps --epochs 25 --batch_size 256 >> "$LOG" 2>&1 \
  || { say "FAIL train"; exit 1; }

say "STAGE 3: quick diagnose (v4_diagnose on a held-out val passive, exploratory)"
$PY tools/v4_diagnose.py --selftest >> "$LOG" 2>&1 || true
say "===== V4 RETRAIN PIPELINE DONE ====="
echo "V4_RETRAIN_DONE" >> "$LOG"
