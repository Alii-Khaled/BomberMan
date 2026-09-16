#!/usr/bin/env bash
# E108 D1 decision battery: e108rl.pt.ep225 (candidate) on the full
# pooled battery. Control = e107wl legs (current ship, same infra/day).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results /tmp/opencode/e108

run() { # tag btag seed rounds opponents...
  local tag="$1" btag="$2" seed="$3" rounds="$4"
  shift 4
  local out="results/tourney_${tag}_${btag}_s${seed}.json"
  local ldir="/tmp/opencode/e108/${tag}_${btag}_s${seed}"
  mkdir -p "$ldir"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  nice -n 10 env ARBITER_DEVICE=cpu ARBITER_NG_MODEL="$PWD/results/e108rl.pt.ep225" \
    python3 scripts/tournament_eval.py --agents Harvy "$@" \
    --n-rounds "$rounds" --seed "$seed" --scenario classic \
    --log-dir "$ldir" --match-name "${tag}_${btag}_s${seed}" --out "$out" \
    > "logs/tourney_${tag}_${btag}_s${seed}.log" 2>&1
  echo "DONE $tag $btag s$seed rc=$?"
}

JOBS=(
  "e108d1 g1 0 100 rule_based_agent rule_based_agent rule_based_agent"
  "e108d1 g1 1 100 rule_based_agent rule_based_agent rule_based_agent"
  "e108d1 strong 0 40 warden_v2 overlord sentinel"
  "e108d1 strong 1 40 warden_v2 overlord sentinel"
  "e108d1 strong 2 40 warden_v2 overlord sentinel"
  "e108d1 strong 3 40 warden_v2 overlord sentinel"
  "e108d1 strong 4 40 warden_v2 overlord sentinel"
  "e108d1 strong 5 40 warden_v2 overlord sentinel"
  "e108d1 strong 6 40 warden_v2 overlord sentinel"
  "e108d1 strong 7 40 warden_v2 overlord sentinel"
  "e108d1 strong 8 40 warden_v2 overlord sentinel"
  "e108d1 strong 9 40 warden_v2 overlord sentinel"
  "e108d1 umix 0 100 unseen_coward unseen_bomber unseen_rusher"
  "e108d1 umix 1 100 unseen_coward unseen_bomber unseen_rusher"
  "e108d1 ucow 0 40 unseen_coward unseen_coward unseen_coward"
  "e108d1 ucow 1 40 unseen_coward unseen_coward unseen_coward"
  "e108d1 ubom 0 40 unseen_bomber unseen_bomber unseen_bomber"
  "e108d1 ubom 1 40 unseen_bomber unseen_bomber unseen_bomber"
  "e108d1 urus 0 40 unseen_rusher unseen_rusher unseen_rusher"
  "e108d1 urus 1 40 unseen_rusher unseen_rusher unseen_rusher"
  "e108d1 urac 0 40 unseen_racer unseen_racer unseen_racer"
  "e108d1 urac 1 40 unseen_racer unseen_racer unseen_racer"
)
for j in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run $j &
  while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n || true; done
done
wait
echo E108_BATTERY_DONE
