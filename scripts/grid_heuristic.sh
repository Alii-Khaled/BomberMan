#!/bin/bash
# Track-1 frozen heuristic grid (E31): one-knob variants + all-defaults
# baseline, 40 rounds seed 0 (paired arenas) vs 3x rule_based, CPU.
# Top-2 go to 100x2 validation. Artifacts: results/grid_<name>.json.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
N=40

run() {  # $1 = tag, rest = env assignments
  local tag="$1"; shift
  echo "=== grid/$tag ==="
  env "$@" OVERLORD_DEVICE=cpu \
    $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent \
      --train 0 --continue-without-training --scenario classic \
      --n-rounds $N --seed 0 --save-stats results/grid_$tag.json
}

run base
run wait045        OVERLORD_G_WAIT=0.45
run hunt07         OVERLORD_G_HUNT_BASE=0.7
run bombopp15      OVERLORD_G_BOMB_OPP=1.5
run bombcrate08    OVERLORD_G_BOMB_CRATE=0.8
run corridor25     OVERLORD_G_CORRIDOR=2.5
run revisit        OVERLORD_G_REVISIT3=0.7 OVERLORD_G_REVISIT2=0.25
run flee3          OVERLORD_G_FLEE_BOOST=3.0
run brepeat15      OVERLORD_G_BOMB_REPEAT=1.5
run coin06         OVERLORD_G_COIN=0.6
run late07         OVERLORD_G_LATE=0.7
echo "Grid done."
