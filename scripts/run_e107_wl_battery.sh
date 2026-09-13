#!/usr/bin/env bash
# E107 C1 completion: wl arm needs g1 100x2 + strong s5-s9 (7 games).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results /tmp/opencode/e107

run() { # btag seed rounds opponents...
  local btag="$1" seed="$2" rounds="$3"
  shift 3
  local out="results/tourney_e107wl_${btag}_s${seed}.json"
  local ldir="/tmp/opencode/e107/e107wl_${btag}_s${seed}"
  mkdir -p "$ldir"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  nice -n 10 env ARBITER_DEVICE=cpu ARBITER_OPPMODEL=wardenlite \
    python3 scripts/tournament_eval.py --agents arbiter_ng "$@" \
    --n-rounds "$rounds" --seed "$seed" --scenario classic \
    --log-dir "$ldir" --match-name "e107wl_${btag}_s${seed}" --out "$out" \
    > "logs/tourney_e107wl_${btag}_s${seed}.log" 2>&1
  echo "DONE $btag s$seed rc=$?"
}

for s in 0 1; do
  run g1 "$s" 100 rule_based_agent rule_based_agent rule_based_agent &
done
for s in 5 6 7 8 9; do
  run strong "$s" 40 warden_v2 overlord sentinel &
done
wait
echo E107_WL_DONE
