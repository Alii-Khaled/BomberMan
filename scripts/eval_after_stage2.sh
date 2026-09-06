#!/bin/bash
# Deferred eval: waits for Stage-2 training to finish, then runs the
# trimmed matrix (M5-M8 at 40 rounds x 2 seeds; M1-M4 already done),
# aggregates tables and renders report figures.
# Usage: nohup bash scripts/eval_after_stage2.sh > logs/eval_deferred.log 2>&1 &
# Eval uses CPU inference (tournament conditions) once training frees the machine.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

# Wait for training (max ~5h); training holds the GPU + CPU.
for i in $(seq 1 300); do
  if pgrep -f "sentinel --train 1" >/dev/null; then
    sleep 60
  else
    break
  fi
done
echo "=== training done (or timeout), starting eval ==="
SENTINEL_DEVICE=cpu bash scripts/run_sentinel_eval.sh
$PY scripts/aggregate_eval.py
$PY scripts/plot_eval.py
echo "=== eval + figures complete ==="
