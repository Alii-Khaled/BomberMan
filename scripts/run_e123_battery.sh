#!/usr/bin/env bash
# E123 canonical gate (arm half only): composed candidate
#   ARBITER_COINTAKE=1 ARBITER_COINTAKE_D=3 ARBITER_SOLO_RADIUS=8
# vs the same-session E122 ctl legs (ship defaults, run this session).
set -u
cd /home/jovyan/work/BomberMan
TAG="${1:?tag}"; ENVV="${2:-}"
mkdir -p logs results /tmp/opencode/e123bat
MAXJOBS=6

run() { # leg agent seed rounds opponents...
  local leg="$1" agent="$2" seed="$3" rounds="$4"
  shift 4
  local out="results/tourney_${TAG}_${leg}_s${seed}.json"
  local ldir="/tmp/opencode/e123bat/${TAG}_${leg}_s${seed}"
  mkdir -p "$ldir"
  [ -f "$out" ] && { echo "SKIP $out"; return 0; }
  nice -n 10 env ARBITER_DEVICE=cpu $ENVV \
    python3 scripts/tournament_eval.py --agents "$agent" "$@" \
    --n-rounds "$rounds" --seed "$seed" --scenario classic \
    --log-dir "$ldir" --match-name "${TAG}_${leg}_s${seed}" \
    --out "$out" > "logs/tourney_${TAG}_${leg}_s${seed}.log" 2>&1
  echo "DONE $TAG $leg s$seed rc=$?"
}

JOBS=()
for s in 0 1; do
  JOBS+=("g1|$s|100|rule_based_agent rule_based_agent rule_based_agent")
done
for s in 0 1 2 3 4 5 6 7 8 9; do
  JOBS+=("strong|$s|40|warden_v2 overlord sentinel")
done
for s in 0 1; do
  JOBS+=("umix|$s|100|unseen_coward unseen_bomber unseen_rusher")
  JOBS+=("ucow|$s|40|unseen_coward unseen_coward unseen_coward")
  JOBS+=("ubom|$s|40|unseen_bomber unseen_bomber unseen_bomber")
  JOBS+=("urus|$s|40|unseen_rusher unseen_rusher unseen_rusher")
  JOBS+=("urac|$s|40|unseen_racer unseen_racer unseen_racer")
done
echo "jobs=${#JOBS[@]}"
for j in "${JOBS[@]}"; do
  IFS='|' read -r leg seed rounds opps <<< "$j"
  # shellcheck disable=SC2086
  run "$leg" arbiter_ng "$seed" "$rounds" $opps &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n || true; done
done
wait
echo E123BAT_ALL_DONE
