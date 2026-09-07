#!/bin/bash
# Overlord short validation run (fresh init, aggressive L40S path).
# O1s 200 coin-heaven -> O2s 300 solo classic -> O3s 300 hunt ->
# O4s 500 vs 3x rule_based = 1300 rounds + frozen CPU eval.
# Resumable per stage: re-run with O*_N=0 to skip finished stages.
# Progress: tail -f logs/overlord_validation.log
# Metrics:  tail -n 5 agent_code/overlord/runs/metrics.csv
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export OVERLORD_OPT=adam OVERLORD_TUNED=1
export OVERLORD_BATCH=1024 OVERLORD_UTD=2 OVERLORD_EOR_UPDATES=12
export OVERLORD_EPS_DECAY=100000 OVERLORD_SAVE_EVERY=5
export OVERLORD_CHANNELS_LAST=1 OVERLORD_COMPILE=0
# Arch (fresh init — shapes baked at first build):
export OVERLORD_BASE=96 OVERLORD_FC=512 OVERLORD_NORM=bn OVERLORD_DEEP=0
O1=${STAGE_O1_N:-200}
O2=${STAGE_O2_N:-300}
O3=${STAGE_O3_N:-300}
O4=${STAGE_O4_N:-500}

reset_for_next_stage() {
  $PY -c "
import torch, os, shutil
shutil.copy('agent_code/overlord/checkpoints/best.pt', 'results/archive/overlord_val_best_$1.pt')
p = 'agent_code/overlord/checkpoints/last.pt'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
ckpt['best_ema'] = None; ckpt['ema_reward'] = None
ckpt['epsilon_steps'] = 63000  # ~= epsilon 0.25 re-warm on 100k decay
torch.save(ckpt, p + '.tmp'); os.replace(p + '.tmp', p)
print('reset for next stage after $1: best archived, EMA cleared, eps~=0.25')
"
}

mkdir -p results/archive logs
if [ "$O1" -gt 0 ]; then
echo "=== O1s (Task 1, coin-heaven navigation) N=$O1 ==="
$PY main.py play --no-gui --agents overlord --train 1 --scenario coin-heaven --n-rounds $O1 --save-stats results/overlord_val_stage1.json
reset_for_next_stage O1s
fi

if [ "$O2" -gt 0 ]; then
echo "=== O2s (Task 2, classic solo bombs/escape) N=$O2 ==="
$PY main.py play --no-gui --agents overlord --train 1 --scenario classic --n-rounds $O2 --save-stats results/overlord_val_stage2.json
reset_for_next_stage O2s
fi

if [ "$O3" -gt 0 ]; then
echo "=== O3s (Task 3, hunt peaceful + coin_collector) N=$O3 ==="
$PY main.py play --no-gui --agents overlord peaceful_agent coin_collector_agent --train 1 --scenario classic --n-rounds $O3 --save-stats results/overlord_val_stage3.json
reset_for_next_stage O3s
fi

if [ "$O4" -gt 0 ]; then
echo "=== O4s (Task 4, vs rule_based) N=$O4 ==="
$PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $O4 --save-stats results/overlord_val_stage4.json
fi

echo "=== Frozen eval (train=0, 100 rounds vs rule_based, CPU) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --save-stats results/overlord_val_eval.json
echo "Done. Metrics: agent_code/overlord/runs/metrics.csv"
