#!/usr/bin/env bash
# E86 DAgger round 2: record the E80-teacher policy on a harder field
# mix (40 rule_based / 40 warden-mix / 20 coin_collector). Round-1
# corpus (arbiter_self*) was rb-heavy and light vs strong opponents.
set -u
PY=python3

run() { # tag, n-rounds, opponents...
  local tag="$1"; shift
  local n="$1"; shift
  if [ "$n" -le 0 ]; then return 0; fi
  echo "=== arbiter_self2 [$tag] N=$n ==="
  # shellcheck disable=SC2086
  ARBITER_DEMO_DIR="results/demos/arbiter_self2${tag}" \
    $PY main.py play --no-gui --agents arbiter_dagger "$@" \
      --train 1 --continue-without-training --scenario classic \
      --n-rounds "$n" --save-stats "results/demos_arbiterself2${tag}.json" \
      --silence-errors
}

run ""    40 rule_based_agent rule_based_agent rule_based_agent
run "_wm" 40 warden_v1 rule_based_agent rule_based_agent
run "_cl" 20 coin_collector_agent coin_collector_agent coin_collector_agent
echo "Done. E86 teacher demos in results/demos/arbiter_self2*/ (~100 rounds)"
