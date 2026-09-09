#!/bin/bash
# Reaper frozen gates (CPU, tournament conditions). Manual launch.
#   G1 100x2 vs 3x rule_based (Task-4 gate)
#   G2 60x2 vs warden_v1 + 2x rule_based (surpass-warden gate)
#   G3 60x2 vs sentinel + 2x rule_based
#   G4 40x2 vs 3x coin_collector + G5 40x2 vs 3x random (weak fields)
#   G6 Q_WEIGHT=0 ablation (ML-compliance: learned Q must add value)
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export REAPER_DEVICE=cpu

run() {  # tag, agents..., n-rounds
  local tag="$1"; shift
  local n="$1"; shift
  echo "=== gate/$tag N=$n ==="
  $PY main.py play --no-gui --agents reaper "$@" --train 0 \
    --continue-without-training --scenario classic \
    --n-rounds "$n" --save-stats "results/gate_reaper_${tag}.json"
}

for seed in 0 1; do
  run "rb_s${seed}" 100 rule_based_agent rule_based_agent rule_based_agent --seed $seed
  run "warden_s${seed}" 60 warden_v1 rule_based_agent rule_based_agent --seed $seed
  run "sentinel_s${seed}" 60 sentinel rule_based_agent rule_based_agent --seed $seed
  run "collect_s${seed}" 40 coin_collector_agent coin_collector_agent coin_collector_agent --seed $seed
  run "random_s${seed}" 40 random_agent random_agent random_agent --seed $seed
done
echo "=== ablation Q_WEIGHT=0 (heuristic-only) vs rb, seed 0 ==="
REAPER_Q_WEIGHT=0 $PY main.py play --no-gui --agents reaper rule_based_agent rule_based_agent rule_based_agent \
  --train 0 --continue-without-training --scenario classic \
  --n-rounds 100 --seed 0 --save-stats results/gate_reaper_q0_s0.json
echo "Done."
