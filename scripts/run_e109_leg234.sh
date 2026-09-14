#!/usr/bin/env bash
# E109 legs: blocks 2-4 continuation (block 1 done for v1/v2).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p /tmp/opencode/e109
LEG="${1:?v1|v2}"
BETA=$([ "$LEG" = "v2" ] && echo 0.2 || echo 0.1)
OUTBASE="results/e109${LEG}.pt"

run_block() { # k opponents...
  local k="$1"
  shift
  nice -n 10 env ARBITER_DEVICE=cpu ARBITER_MODEL="$PWD/$OUTBASE" \
    ARBITER_RL_LR=5e-5 ARBITER_RL_BETA="$BETA" ARBITER_RL_STABLE=1 \
    ARBITER_RL_BOMB_TRACE=1 ARBITER_RL_OUT="$PWD/$OUTBASE" \
    ARBITER_RL_CSV="$PWD/results/e109${LEG}b${k}.csv" \
    APEX_TEACHER=arbiter_ng APEX_DEMO_OUT=/tmp/opencode/e108 \
    python3 main.py play --no-gui --agents "$@" \
    --train 1 --scenario classic --n-rounds 75 \
    > "logs/e109${LEG}b${k}_train.log" 2>&1
  cp "$OUTBASE" "${OUTBASE}_b${k}.pt" 2>/dev/null || true
  echo "BLOCK ${LEG} ${k} DONE rc=$?"
}

run_block 2 arbiter_ng rule_based_agent rule_based_agent warden_v2
run_block 3 arbiter_ng apex_teacher rule_based_agent rule_based_agent
run_block 4 arbiter_ng rule_based_agent coin_collector_agent warden_v1
echo "E109_${LEG}_LEG_DONE"
