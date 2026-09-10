#!/usr/bin/env bash
# P2-C self-distillation demos: the SHIP arbiter acts (exact ship
# defaults from the environment), arbiter_dagger records (feats, acts)
# to results/demos/arbiter_self[_<field>]/ (training-time only).
# Field split mirrors the gate distribution so the retrain sees the
# eval distribution (cf. collect_demos.sh E37/P4).
# Usage: DAGGER_N=200 bash scripts/collect_arbiter_self.sh
# Box-sequential: run ONLY when no other gate is active.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
N=${DAGGER_N:-200}
mkdir -p results/demos logs

run() { # tag, n-rounds, opponents...
  local tag="$1"; shift
  local n="$1"; shift
  if [ "$n" -le 0 ]; then return 0; fi
  echo "=== arbiter_self [$tag] N=$n ==="
  # shellcheck disable=SC2086
  ARBITER_DEMO_DIR="results/demos/arbiter_self${tag}" \
    $PY main.py play --no-gui --agents arbiter_dagger "$@" \
      --train 1 --continue-without-training --scenario classic \
      --n-rounds "$n" --save-stats "results/demos_arbiterself${tag}.json"
}

n_rb=$(( N * 50 / 100 ))
n_wm=$(( N * 25 / 100 ))
n_rn=$(( N * 125 / 1000 ))
n_cl=$(( N - n_rb - n_wm - n_rn ))
run ""    "$n_rb" rule_based_agent rule_based_agent rule_based_agent
run "_wm" "$n_wm" warden_v1 rule_based_agent rule_based_agent
run "_rn" "$n_rn" random_agent random_agent random_agent
run "_cl" "$n_cl" coin_collector_agent coin_collector_agent coin_collector_agent
echo "Done. Self-demo files in results/demos/arbiter_self*/ (~$N rounds)"
