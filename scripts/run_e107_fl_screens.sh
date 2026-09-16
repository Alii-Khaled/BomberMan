#!/usr/bin/env bash
# E107 C2 screens: ARBITER_FLEE_LOOK=1 (ship weights, env-only).
# STRONG 40x5 + UNSEEN legs 40x2, 8-wide.
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results /tmp/opencode/e107

run() { # tag btag seed rounds opponents...
  local tag="$1" btag="$2" seed="$3" rounds="$4"
  shift 4
  local out="results/tourney_${tag}_${btag}_s${seed}.json"
  local ldir="/tmp/opencode/e107/${tag}_${btag}_s${seed}"
  mkdir -p "$ldir"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  nice -n 10 env ARBITER_DEVICE=cpu ARBITER_FLEE_LOOK=1 \
    python3 scripts/tournament_eval.py --agents Harvy "$@" \
    --n-rounds "$rounds" --seed "$seed" --scenario classic \
    --log-dir "$ldir" --match-name "${tag}_${btag}_s${seed}" --out "$out" \
    > "logs/tourney_${tag}_${btag}_s${seed}.log" 2>&1
  echo "DONE $tag $btag s$seed rc=$?"
}

JOBS=(
  "e107fl strong 0 40 warden_v2 overlord sentinel"
  "e107fl strong 1 40 warden_v2 overlord sentinel"
  "e107fl strong 2 40 warden_v2 overlord sentinel"
  "e107fl strong 3 40 warden_v2 overlord sentinel"
  "e107fl strong 4 40 warden_v2 overlord sentinel"
  "e107fl ucow 0 40 unseen_coward unseen_coward unseen_coward"
  "e107fl ucow 1 40 unseen_coward unseen_coward unseen_coward"
  "e107fl ubom 0 40 unseen_bomber unseen_bomber unseen_bomber"
  "e107fl ubom 1 40 unseen_bomber unseen_bomber unseen_bomber"
  "e107fl urus 0 40 unseen_rusher unseen_rusher unseen_rusher"
  "e107fl urus 1 40 unseen_rusher unseen_rusher unseen_rusher"
  "e107fl urac 0 40 unseen_racer unseen_racer unseen_racer"
  "e107fl urac 1 40 unseen_racer unseen_racer unseen_racer"
)
for j in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run $j &
  while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n || true; done
done
wait
echo E107_FL_DONE
