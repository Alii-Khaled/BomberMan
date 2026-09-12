#!/usr/bin/env bash
# E99 feature-v2 self-distillation corpus: the SHIP arbiter acts, and
# arbiter_v2_dagger records (114-dim v2 feats, ship action) per tick.
# Consumed by: scripts/arbiter_extract.py --pkg=arbiter_v2
#              --reaper-include=arbiter_v2_self
# Usage: DAGGER_N=400 bash scripts/collect_arbiter_v2_self.sh
# Box-heavy (ship search every tick); keep other gates off.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
N=${DAGGER_N:-400}
PREFIX=${DEMO_PREFIX:-arbiter_v2_self}
mkdir -p results/demos logs

run() { # tag, n-rounds, opponents...
  local tag="$1"; shift
  local n="$1"; shift
  if [ "$n" -le 0 ]; then return 0; fi
  echo "=== arbiter_v2_self [$tag] N=$n ==="
  # shellcheck disable=SC2086
  ARBITER_DEMO_DIR="results/demos/${PREFIX}${tag}" \
    $PY main.py play --no-gui --agents arbiter_v2_dagger "$@" \
      --train 1 --continue-without-training --scenario classic \
      --n-rounds "$n" --save-stats "results/demos_${PREFIX}${tag}.json"
}

n_rb=$(( N * 50 / 100 ))
n_wm=$(( N * 25 / 100 ))
n_rn=$(( N * 125 / 1000 ))
n_cl=$(( N - n_rb - n_wm - n_rn ))
run ""    "$n_rb" rule_based_agent rule_based_agent rule_based_agent
run "_wm" "$n_wm" warden_v2 rule_based_agent rule_based_agent
run "_rn" "$n_rn" random_agent random_agent random_agent
run "_cl" "$n_cl" coin_collector_agent coin_collector_agent coin_collector_agent
echo "Done. v2-feature ship demos in results/demos/${PREFIX}*/ (~$N rounds)"
