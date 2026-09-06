#!/bin/bash
# Overlord curriculum O1-O4 (Tasks 1-4, Project Description §4).
# Fresh start (pre-O state archived in results/archive/overlord_pre_O1/).
# Arm-A bundle baked into code (Huber td+aux, clip 1.0, tuned AdamW,
# epsilon-greedy 1.0->0.05/50k, margin gate, WAIT -0.30, NO stale JIT trace).
# Resumable per stage: re-run with O*_N=0 to skip finished stages, e.g.
#   STAGE_O1_N=0 STAGE_O2_N=0 bash scripts/train_overlord_curriculum.sh
# Between stages: best_ema reset (cross-regime EMA incomparable — sentinel
# E04/E10 lesson) + epsilon re-warm to ~0.25 (sentinel S5/S6 lesson).
# Env: OVERLORD_OPT=adam OVERLORD_TUNED=1, UTD/BATCH/EOR defaults.
# Device default: auto (CUDA on Colab when available, else CPU).
# AMP default: on for CUDA (OVERLORD_AMP=0 for fp32).
# Overnight: nohup bash scripts/train_overlord_curriculum.sh > logs/overlord_curriculum.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export OVERLORD_OPT=adam OVERLORD_TUNED=1
O1=${STAGE_O1_N:-750}
O2=${STAGE_O2_N:-1500}
O3=${STAGE_O3_N:-1000}
O4=${STAGE_O4_N:-2000}

reset_for_next_stage() {
  # $1 = tag (e.g. O1) — archive stage best, reset EMA + re-warm epsilon.
  $PY -c "
import torch, os, shutil
shutil.copy('agent_code/overlord/checkpoints/best.pt', 'results/archive/overlord_best_$1.pt')
p = 'agent_code/overlord/checkpoints/last.pt'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
ckpt['best_ema'] = None; ckpt['ema_reward'] = None
ckpt['epsilon_steps'] = 39474  # ~= epsilon 0.25 re-warm
torch.save(ckpt, p + '.tmp'); os.replace(p + '.tmp', p)
print('reset for next stage after $1: best archived, EMA cleared, eps=0.25')
"
}

if [ "$O1" -gt 0 ]; then
echo "=== O1 (Task 1, coin-heaven navigation) N=$O1 ==="
$PY main.py play --no-gui --agents overlord --train 1 --scenario coin-heaven --n-rounds $O1 --save-stats results/overlord_stage1.json
reset_for_next_stage O1
fi

if [ "$O2" -gt 0 ]; then
echo "=== O2 (Task 2, classic solo bombs/escape) N=$O2 ==="
$PY main.py play --no-gui --agents overlord --train 1 --scenario classic --n-rounds $O2 --save-stats results/overlord_stage2.json
reset_for_next_stage O2
fi

if [ "$O3" -gt 0 ]; then
echo "=== O3 (Task 3, hunt peaceful + coin_collector) N=$O3 ==="
$PY main.py play --no-gui --agents overlord peaceful_agent coin_collector_agent --train 1 --scenario classic --n-rounds $O3 --save-stats results/overlord_stage3.json
reset_for_next_stage O3
fi

if [ "$O4" -gt 0 ]; then
echo "=== O4 (Task 4, vs rule_based) N=$O4 ==="
$PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $O4 --save-stats results/overlord_stage4.json
fi

echo "=== Frozen eval (train=0, 100 rounds vs rule_based) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --save-stats results/overlord_eval.json
echo "Done. Metrics: agent_code/overlord/runs/metrics.csv"
