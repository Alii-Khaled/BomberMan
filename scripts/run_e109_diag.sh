#!/usr/bin/env bash
# Diag legs on the current ship (sequential, no-op learning).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs results logs/e109
for s in 0 1; do
  out="results/diagstats_e109d1_s${s}.json"
  [ -f "$out" ] && { echo "SKIP $out"; continue; }
  nice -n 10 env HARVEY_DEVICE=cpu HARVEY_DIAG="$PWD/results/diag_e109d1" \
    HARVEY_RL_LR=0 HARVEY_RL_OUT="$PWD/logs/e109/scratch.pt" \
    python3 main.py play --no-gui \
    --agents Harvey rule_based_agent rule_based_agent rule_based_agent \
    --train 1 --continue-without-training --scenario classic \
    --n-rounds 100 --seed "$s" --save-stats "$out" \
    > "logs/diagleg_e109_s${s}.log" 2>&1
  echo "DONE diag s$s rc=$?"
done
cp results/diag_e109d1_deaths.jsonl results/diag_e109d1_s0_deaths.jsonl 2>/dev/null
echo E109_DIAG_DONE
