#!/usr/bin/env bash
# E109-B diag legs on the D1 ship (sequential, no-op learning).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results
for s in 0 1; do
  out="results/diagstats_e109d1_s${s}.json"
  [ -f "$out" ] && { echo "SKIP $out"; continue; }
  nice -n 10 env ARBITER_DEVICE=cpu ARBITER_DIAG="$PWD/results/diag_e109d1" \
    ARBITER_RL_LR=0 ARBITER_RL_OUT="/tmp/opencode/e109/scratch.pt" \
    python3 main.py play --no-gui \
    --agents arbiter_ng rule_based_agent rule_based_agent rule_based_agent \
    --train 1 --continue-without-training --scenario classic \
    --n-rounds 100 --seed "$s" --save-stats "$out" \
    > "logs/diagleg_e109_s${s}.log" 2>&1
  echo "DONE diag s$s rc=$?"
done
cp results/diag_e109d1_deaths.jsonl results/diag_e109d1_s0_deaths.jsonl 2>/dev/null
echo E109_DIAG_DONE
