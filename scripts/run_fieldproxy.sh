#!/usr/bin/env bash
# Field-proxy matrix (P1.3): arbiter vs field archetypes that proxy the
# unseen tournament field (no Discord downloads available). Frozen CPU,
# 40 rounds x seeds 0,1 — same discipline as G2-G4, lighter N.
#   STRONG : arbiter warden_vN overlord sentinel      (hunters present)
#   RACER  : arbiter coin_collector x2 + overlord     (crate-race field)
#   WEAK   : arbiter peaceful random overlord         (free-kill farm)
#   TRAINED: arbiter apex reaper sentinel             (DQN-class field)
# Box-sequential: run ONLY when no other gate is active.
# Usage: bash scripts/run_fieldproxy.sh [TAG]  (TAG prefixes output files)
set -u
TAG="${1:-fieldproxy}"
WARDEN="${WARDEN:-warden_v2}"
for lobby in "strong:arbiter $WARDEN overlord sentinel" \
             "racer:arbiter coin_collector_agent coin_collector_agent overlord" \
             "weak:arbiter peaceful_agent random_agent overlord" \
             "trained:arbiter apex reaper sentinel"; do
  name="${lobby%%:*}"; agents="${lobby#*:}"
  # shellcheck disable=SC2086
  for seed in 0 1; do
    out="results/${TAG}_${name}_s${seed}.json"
    if [ -f "$out" ]; then echo "SKIP existing $out"; continue; fi
    echo "=== $out : $agents (seed $seed)"
    # shellcheck disable=SC2086
    python3 main.py play --no-gui --agents $agents \
      --train 0 --continue-without-training --scenario classic \
      --n-rounds 40 --seed "$seed" --save-stats "$out" --silence-errors \
      || { echo "FAILED $out"; exit 1; }
  done
done
echo "FIELDPROXY $TAG DONE"
