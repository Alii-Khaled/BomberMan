#!/bin/bash
# Sentinel curriculum Tasks 1-4 (Project Description §4/§6).
# Resumable: each stage appends to checkpoints/last.pt + runs/metrics.csv
# Override: SENTINEL_DEVICE=cpu for CPU-only, N override per stage.
# Device default: auto (CUDA on Colab when available, else CPU).
# AMP default: on for CUDA (SENTINEL_AMP=0 for fp32).
# Usage: bash scripts/train_sentinel_curriculum.sh
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
STAGE1=${STAGE1_N:-500}
STAGE2=${STAGE2_N:-1500}
STAGE3=${STAGE3_N:-1000}
STAGE4=${STAGE4_N:-2000}

echo "=== Stage 1 (Task 1, coin-heaven navigation) N=$STAGE1 ==="
$PY main.py play --no-gui --agents sentinel --train 1 --scenario coin-heaven --n-rounds $STAGE1 --save-stats results/sentinel_stage1.json

echo "=== Stage 2 (Task 2, classic solo bombs/escape) N=$STAGE2 ==="
$PY main.py play --no-gui --agents sentinel --train 1 --scenario classic --n-rounds $STAGE2 --save-stats results/sentinel_stage2.json

echo "=== Stage 3 (Task 3, hunt peaceful then coin_collector) N=$STAGE3 ==="
$PY main.py play --no-gui --agents sentinel peaceful_agent coin_collector_agent --train 1 --scenario classic --n-rounds $STAGE3 --save-stats results/sentinel_stage3.json

echo "=== Stage 4 (Task 4, vs rule_based) N=$STAGE4 ==="
$PY main.py play --no-gui --agents sentinel rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $STAGE4 --save-stats results/sentinel_stage4.json

echo "=== Frozen eval (train=0, 100 rounds vs rule_based) ==="
$PY main.py play --no-gui --agents sentinel rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --save-stats results/sentinel_eval.json
echo "Done. Metrics: agent_code/sentinel/runs/metrics.csv"
