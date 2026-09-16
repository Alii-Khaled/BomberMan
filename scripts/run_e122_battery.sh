#!/usr/bin/env bash
# E122 canonical gate: fresh same-session control (ship defaults) vs the
# certified coin-take arm. 66 games per arm / 1120 rounds each:
#   G1 100x2 | STRONG 40x10 | UMIX 100x2 | ucow/ubom/urus/urac 40x2
# 8 concurrent games, nice'd, engine logs to /tmp.
# Usage: bash scripts/run_e122_battery.sh <arm-tag> <env-assignment>
set -u
cd /home/jovyan/work/BomberMan
TAG="${1:?arm tag}"; ENVV="${2:-}"
mkdir -p logs results /tmp/opencode/e122bat

MAXJOBS=8

run() { # btag leg agent seed rounds opponents...
  local btag="$1" leg="$2" agent="$3" seed="$4" rounds="$5"
  shift 5
  local out="results/tourney_${TAG}_${btag}_${leg}_s${seed}.json"
  local ldir="/tmp/opencode/e122bat/${TAG}_${btag}_${leg}_s${seed}"
  mkdir -p "$ldir"
  [ -f "$out" ] && { echo "SKIP $out"; return 0; }
  nice -n 10 env ARBITER_DEVICE=cpu $ENVV \
    python3 scripts/tournament_eval.py --agents "$agent" "$@" \
    --n-rounds "$rounds" --seed "$seed" --scenario classic \
    --log-dir "$ldir" --match-name "${TAG}_${btag}_${leg}_s${seed}" \
    --out "$out" > "logs/tourney_${TAG}_${btag}_${leg}_s${seed}.log" 2>&1
  echo "DONE $TAG $btag $leg s$seed rc=$?"
}

JOBS=()
for conf in "ctl|" "arm|$ENVV"; do
  btag="${conf%%|*}"; e="${conf#*|}"
  for s in 0 1; do
    JOBS+=("$btag|g1|$s|100|rule_based_agent rule_based_agent rule_based_agent")
  done
  for s in 0 1 2 3 4 5 6 7 8 9; do
    JOBS+=("$btag|strong|$s|40|warden_v2 overlord sentinel")
  done
  for s in 0 1; do
    JOBS+=("$btag|umix|$s|100|unseen_coward unseen_bomber unseen_rusher")
    JOBS+=("$btag|ucow|$s|40|unseen_coward unseen_coward unseen_coward")
    JOBS+=("$btag|ubom|$s|40|unseen_bomber unseen_bomber unseen_bomber")
    JOBS+=("$btag|urus|$s|40|unseen_rusher unseen_rusher unseen_rusher")
    JOBS+=("$btag|urac|$s|40|unseen_racer unseen_racer unseen_racer")
  done
done
echo "jobs=${#JOBS[@]}"
for j in "${JOBS[@]}"; do
  IFS='|' read -r btag leg seed rounds opps <<< "$j"
  # shellcheck disable=SC2086
  run "$btag" "$leg" Harvy "$seed" "$rounds" $opps &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n || true; done
done
wait
echo E122BAT_ALL_DONE
