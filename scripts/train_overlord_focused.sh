#!/bin/bash
# Overlord focused W-curriculum (E26 program): fewer rounds, weakpoint-first.
# Base: ep_1400 weights in last.pt (EMA cleared, eps re-warm 0.40 applied at
# restore) + E28 crate-approach pull (probed 7/7 in scripts/probe_crate_pull.py).
#   W1 economy  300: solo classic (crate pull must convert to crates/coins)
#   W2 sparring 300: vs warden_v1 + 2x rule_based (discipline by example)
#   W3 gate     300: vs 3x rule_based (gate-matchup consolidation)
#   + frozen evals: 100 vs 3x rb (gate, comparable to 3.70) + 60 warden-mix.
# Resumable per stage: re-run with W*_N=0 to skip finished stages.
# Env: identical L40S aggressive path as validation (never change arch mid-run:
# shapes are baked — BASE/FC/NORM/DEEP must stay 96/512/bn/0).
# Launch detached: nohup bash scripts/train_overlord_focused.sh > logs/overlord_focused.log 2>&1 &
# Guard: bash scripts/watch_overlord_focused.sh (kill-switch only, no auto-launch).
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export OVERLORD_OPT=adam OVERLORD_TUNED=1
export OVERLORD_BATCH=1024 OVERLORD_UTD=2 OVERLORD_EOR_UPDATES=12
export OVERLORD_EPS_DECAY=100000 OVERLORD_SAVE_EVERY=5
export OVERLORD_CHANNELS_LAST=1 OVERLORD_COMPILE=0
export OVERLORD_BASE=96 OVERLORD_FC=512 OVERLORD_NORM=bn OVERLORD_DEEP=0
W1=${STAGE_W1_N:-300}
W2=${STAGE_W2_N:-300}
W3=${STAGE_W3_N:-300}

reset_for_next_stage() {
  $PY -c "
import torch, os, shutil
shutil.copy('agent_code/overlord/checkpoints/best.pt', 'results/archive/overlord_focused_best_$1.pt')
p = 'agent_code/overlord/checkpoints/last.pt'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
ckpt['best_ema'] = None; ckpt['ema_reward'] = None
ckpt['epsilon_steps'] = 63000  # ~= epsilon 0.40 re-warm on 100k decay
torch.save(ckpt, p + '.tmp'); os.replace(p + '.tmp', p)
print('reset for next stage after $1: best archived, EMA cleared, eps~=0.40')
"
}

mkdir -p results/archive logs
if [ "$W1" -gt 0 ]; then
echo "=== W1 (economy, solo classic) N=$W1 ==="
$PY main.py play --no-gui --agents overlord --train 1 --scenario classic --n-rounds $W1 --save-stats results/overlord_focused_w1.json
reset_for_next_stage W1
fi

if [ "$W2" -gt 0 ]; then
echo "=== W2 (warden sparring: warden + 2x rule_based) N=$W2 ==="
$PY main.py play --no-gui --agents overlord warden_v1 rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $W2 --save-stats results/overlord_focused_w2.json
reset_for_next_stage W2
fi

if [ "$W3" -gt 0 ]; then
echo "=== W3 (gate consolidation: 3x rule_based) N=$W3 ==="
$PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $W3 --save-stats results/overlord_focused_w3.json
fi

echo "=== Frozen eval vs 3x rule_based (100 rounds, CPU) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --save-stats results/overlord_focused_eval_rb.json
echo "=== Frozen eval vs warden-mix (60 rounds, CPU) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord warden_v1 rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 60 --save-stats results/overlord_focused_eval_warden.json
echo "Done. Metrics: agent_code/overlord/runs/metrics.csv"
