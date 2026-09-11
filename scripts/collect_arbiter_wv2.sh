#!/usr/bin/env bash
# E98 ship-state warden_v2 label transfer: the SHIP arbiter acts, and at
# every ship-visited state arbiter_dagger also records warden_v2's move
# (acts_w) alongside the ship's own action. Training-time only.
#
# Rationale (E86/E88 lesson): label quality on the SHIP's own state
# distribution is the bottleneck — training on warden-visited states
# (collect_demos) shifts the prior off the G1 optimum. These rows keep
# the distribution and replace/augment the labels with the current best
# warden.
#
# Usage: DAGGER_N=300 bash scripts/collect_arbiter_wv2.sh
# Box-heavy: the ship's search runs every tick (~40 ms); keep N modest
# and avoid running other gates concurrently.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
WARDEN="${WARDEN:-warden_v2}"
N=${DAGGER_N:-300}
PREFIX=${DEMO_PREFIX:-da_warden}
export ARBITER_DAGGER_WARDEN=1
mkdir -p results/demos logs

run() { # tag, n-rounds, opponents...
  local tag="$1"; shift
  local n="$1"; shift
  if [ "$n" -le 0 ]; then return 0; fi
  echo "=== ship_wv2 [$tag] N=$n ==="
  # shellcheck disable=SC2086
  ARBITER_DEMO_DIR="results/demos/${PREFIX}${tag}" \
    $PY main.py play --no-gui --agents arbiter_dagger "$@" \
      --train 1 --continue-without-training --scenario classic \
      --n-rounds "$n" --save-stats "results/demos_${PREFIX}${tag}.json"
}

n_rb=$(( N * 50 / 100 ))
n_wm=$(( N * 25 / 100 ))
n_rn=$(( N * 125 / 1000 ))
n_cl=$(( N - n_rb - n_wm - n_rn ))
run ""    "$n_rb" rule_based_agent rule_based_agent rule_based_agent
run "_wm" "$n_wm" $WARDEN rule_based_agent rule_based_agent
run "_rn" "$n_rn" random_agent random_agent random_agent
run "_cl" "$n_cl" coin_collector_agent coin_collector_agent coin_collector_agent
echo "Done. Warden-labelled ship demos in results/demos/${PREFIX}*/ (~$N rounds)"
