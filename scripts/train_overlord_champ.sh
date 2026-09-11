#!/bin/bash
# Overlord CHAMP retune (E31 Track 2): single-change discipline.
# Base: O3s-best ep741 in last.pt (EMA cleared, eps re-warm ~=0.20 env steps,
# C2 counting, C5 sparring-light). One change vs validation: opponent
# schedule + low re-warm. NO pull, NO arch change, NO UTD change.
#   R1 400: gate matchup 3x rule_based (consolidation)
#   R2a 200: warden_vN + 2x rule_based (discipline sparring)
#   R2b 200: sentinel + 2x rule_based (diverse learned foe; --train 1 => only overlord learns)
#   + frozen gates: 100 rb + 60 warden-mix + 60 sentinel-mix (CPU).
# Resumable: STAGE_R1_N=0 etc. Guard: watch_overlord_focused.sh (patched to
# match train_overlord_champ). No auto-launch — manual gates between stages.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
WARDEN="${WARDEN:-warden_v2}"
export OVERLORD_OPT=adam OVERLORD_TUNED=1
export OVERLORD_BATCH=1024 OVERLORD_UTD=2 OVERLORD_EOR_UPDATES=12
export OVERLORD_EPS_DECAY=100000 OVERLORD_SAVE_EVERY=5
export OVERLORD_CHANNELS_LAST=1 OVERLORD_COMPILE=0
export OVERLORD_BASE=96 OVERLORD_FC=512 OVERLORD_NORM=bn OVERLORD_DEEP=0
R1=${STAGE_R1_N:-400}
R2A=${STAGE_R2A_N:-200}
R2B=${STAGE_R2B_N:-200}

reset_for_next_stage() {
  $PY -c "
import torch, os, shutil
shutil.copy('agent_code/overlord/checkpoints/best.pt', 'results/archive/overlord_champ_best_$1.pt')
p = 'agent_code/overlord/checkpoints/last.pt'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
ckpt['best_ema'] = None; ckpt['ema_reward'] = None
ckpt['epsilon_steps'] = 84000  # ~=0.20 re-warm, env-step units (C2/C5)
torch.save(ckpt, p + '.tmp'); os.replace(p + '.tmp', p)
print('reset for next stage after $1: best archived, EMA cleared, eps~=0.20')
"
}

mkdir -p results/archive logs
if [ "$R1" -gt 0 ]; then
echo "=== R1 (gate consolidation: 3x rule_based) N=$R1 ==="
$PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $R1 --save-stats results/overlord_champ_r1.json
reset_for_next_stage R1
fi

if [ "$R2A" -gt 0 ]; then
echo "=== R2a (warden sparring) N=$R2A ==="
$PY main.py play --no-gui --agents overlord $WARDEN rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $R2A --save-stats results/overlord_champ_r2a.json
fi

if [ "$R2B" -gt 0 ]; then
echo "=== R2b (sentinel sparring) N=$R2B ==="
$PY main.py play --no-gui --agents overlord sentinel rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $R2B --save-stats results/overlord_champ_r2b.json
reset_for_next_stage R2
fi

echo "=== Frozen gates (CPU) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --save-stats results/overlord_champ_eval_rb.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord $WARDEN rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 60 --save-stats results/overlord_champ_eval_warden.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord sentinel rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 60 --save-stats results/overlord_champ_eval_sentinel.json
echo "Done. Metrics: agent_code/overlord/runs/metrics.csv"
