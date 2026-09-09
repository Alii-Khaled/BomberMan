#!/bin/bash
# Overlord W0 crate-light stage (E40): last training card.
# Base: O3s-best ep741 (EMA cleared, eps re-warm ~=0.15 env steps, C2/C5).
# W0 200 rounds solo crate-light (density 0.4, 9 coins — training-only
# scenario in settings.py) -> 30rd quick screen (stop if <2.5) ->
# frozen 100 rb classic + 60 warden-mix.
# Continue-gates (E40): coins lift vs O2s 1.34 with NO frozen-score
# regression vs 3.79 ship; anything else -> stop, ledger, report.
# Resumable: STAGE_W0_N=0 to skip straight to gates.
# Guard: watch_overlord_focused.sh (matches train_overlord_cratelight).
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export OVERLORD_OPT=adam OVERLORD_TUNED=1
export OVERLORD_BATCH=1024 OVERLORD_UTD=2 OVERLORD_EOR_UPDATES=12
export OVERLORD_EPS_DECAY=100000 OVERLORD_SAVE_EVERY=5
export OVERLORD_CHANNELS_LAST=1 OVERLORD_COMPILE=0
export OVERLORD_BASE=96 OVERLORD_FC=512 OVERLORD_NORM=bn OVERLORD_DEEP=0
W0=${STAGE_W0_N:-200}

mkdir -p results/archive logs
if [ "$W0" -gt 0 ]; then
echo "=== W0 (economy, solo crate-light) N=$W0 ==="
$PY main.py play --no-gui --agents overlord --train 1 --scenario crate-light --n-rounds $W0 --save-stats results/overlord_w0.json
$PY -c "
import torch, shutil
shutil.copy('agent_code/overlord/checkpoints/best.pt', 'results/archive/overlord_w0_best.pt')
print('W0 best archived')
"
fi

echo "=== Quick screen (30 rounds, CPU, vs 3x rb) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 30 --seed 0 --save-stats results/overlord_w0_screen.json
echo "=== Frozen gates (CPU) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --seed 0 --save-stats results/overlord_w0_eval_rb_s0.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --seed 1 --save-stats results/overlord_w0_eval_rb_s1.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord warden_v1 rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 60 --seed 0 --save-stats results/overlord_w0_eval_warden.json
echo "Done. Metrics: agent_code/overlord/runs/metrics.csv"
