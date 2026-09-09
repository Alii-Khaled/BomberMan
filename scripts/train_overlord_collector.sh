#!/bin/bash
# Overlord collector classroom (E43): economy with opponents, classic only.
# Base: O3s-best ep741 (EMA cleared, eps re-warm ~=0.15 env steps, C2/C5).
# Bonus: CRATE_EXTRA +0.15/extra-crate (cap 3) bundled in (probed 5/5 in
# scripts/probe_multicrate.py; stated Huber caveat — the stage's main lever
# is the collector distribution, never solo again).
#   C1 300: vs 3x coin_collector_agent (crate-race pressure, hunt stays warm)
#   + frozen gates: 100 rb (seeds 0+1) + 60 collector-matrix.
# Pre-registered gates: crates/round 11.6 -> 16+ with NO frozen regression
# vs 3.79 ship; miss either -> revert bonus, stop, ledger E44.
# Resumable: STAGE_C1_N=0 to skip straight to gates.
# Guard: watch_overlord_focused.sh (matches train_overlord_collector).
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export OVERLORD_OPT=adam OVERLORD_TUNED=1
export OVERLORD_BATCH=1024 OVERLORD_UTD=2 OVERLORD_EOR_UPDATES=12
export OVERLORD_EPS_DECAY=100000 OVERLORD_SAVE_EVERY=5
export OVERLORD_CHANNELS_LAST=1 OVERLORD_COMPILE=0
export OVERLORD_BASE=96 OVERLORD_FC=512 OVERLORD_NORM=bn OVERLORD_DEEP=0
C1=${STAGE_C1_N:-300}

mkdir -p results/archive logs
if [ "$C1" -gt 0 ]; then
echo "=== C1 (collector classroom: 3x coin_collector) N=$C1 ==="
$PY main.py play --no-gui --agents overlord coin_collector_agent coin_collector_agent coin_collector_agent --train 1 --scenario classic --n-rounds $C1 --save-stats results/overlord_collector_c1.json
$PY -c "
import torch, shutil
shutil.copy('agent_code/overlord/checkpoints/best.pt', 'results/archive/overlord_collector_best_C1.pt')
print('C1 best archived')
"
fi

echo "=== Frozen gates (CPU) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --seed 0 --save-stats results/overlord_collector_eval_rb_s0.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --seed 1 --save-stats results/overlord_collector_eval_rb_s1.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord coin_collector_agent coin_collector_agent coin_collector_agent --train 0 --continue-without-training --scenario classic --n-rounds 60 --seed 0 --save-stats results/overlord_collector_eval_cl.json
echo "Done. Metrics: agent_code/overlord/runs/metrics.csv"
