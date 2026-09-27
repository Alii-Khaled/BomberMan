#!/usr/bin/env bash
# Blocks 2-4 of the recipe leg (continuation).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs results logs/e109
LEG="${1:?v1|v2}"
BETA=$([ "$LEG" = "v2" ] && echo 0.2 || echo 0.1)
OUTBASE="results/e109${LEG}.pt"

run_block() { # k opponents...
  local k="$1"
  shift
  nice -n 10 env HARVEY_DEVICE=cpu HARVEY_MODEL="$PWD/$OUTBASE" \
    HARVEY_RL_LR=5e-5 HARVEY_RL_BETA="$BETA" HARVEY_RL_STABLE=1 \
    HARVEY_RL_BOMB_TRACE=1 HARVEY_RL_OUT="$PWD/$OUTBASE" \
    HARVEY_RL_CSV="$PWD/results/e109${LEG}b${k}.csv" \
    APEX_TEACHER=Harvey APEX_DEMO_OUT="$PWD/results/e108_recordings" \
    python3 main.py play --no-gui --agents "$@" \
    --train 1 --scenario classic --n-rounds 75 \
    > "logs/e109${LEG}b${k}_train.log" 2>&1
  cp "$OUTBASE" "${OUTBASE}_b${k}.pt" 2>/dev/null || true
  echo "BLOCK ${LEG} ${k} DONE rc=$?"
}

run_block 2 Harvey rule_based_agent rule_based_agent warden_v2
run_block 3 Harvey apex_teacher rule_based_agent rule_based_agent
run_block 4 Harvey rule_based_agent coin_collector_agent warden_v1
echo "E109_${LEG}_LEG_DONE"
