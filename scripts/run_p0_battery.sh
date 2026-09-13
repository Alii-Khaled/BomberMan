#!/usr/bin/env bash
# P0 confirm batteries (P0cash/E104): for control(=ship) + 2 candidates.
#   G1 re-legs 100x2 (same-session-clean pooled metric)
#   STRONG 40x10  (lobby: <agent> warden_v2 overlord sentinel)
#   UNSEEN: UMIX 100x2 (coward bomber rusher) + 4 archetype legs 40x2
# 8 concurrent games max, nice'd; engine logs -> /tmp (local disk);
# agent logs stay at WARNING (tournament_eval patches settings).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results /tmp/opencode/p0bat

MAXJOBS=8

run() { # mtag btag agent model seed rounds opponents...
  local mtag="$1" btag="$2" agent="$3" model="$4" seed="$5" rounds="$6"
  shift 6
  local out="results/tourney_${mtag}_${btag}_s${seed}.json"
  local ldir="/tmp/opencode/p0bat/${mtag}_${btag}_s${seed}"
  mkdir -p "$ldir"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  if [ -n "$model" ]; then
    nice -n 10 env ARBITER_DEVICE=cpu ARBITER_NG_MODEL="$model" \
      python3 scripts/tournament_eval.py --agents "$agent" "$@" \
      --n-rounds "$rounds" --seed "$seed" --scenario classic \
      --log-dir "$ldir" --match-name "${mtag}_${btag}_s${seed}" \
      --out "$out" > "logs/tourney_${mtag}_${btag}_s${seed}.log" 2>&1
  else
    nice -n 10 env ARBITER_DEVICE=cpu \
      python3 scripts/tournament_eval.py --agents "$agent" "$@" \
      --n-rounds "$rounds" --seed "$seed" --scenario classic \
      --log-dir "$ldir" --match-name "${mtag}_${btag}_s${seed}" \
      --out "$out" > "logs/tourney_${mtag}_${btag}_s${seed}.log" 2>&1
  fi
  echo "DONE $mtag $btag s$seed rc=$?"
}

MODELS="ctl:arbiter: b1e150:arbiter_ng:$PWD/results/e104_b1.pt.ep150 ng20:arbiter_ng:$PWD/results/arbiter_ng_full.pt"

# build the 66-game job list, interleaving models for uniform load.
# Fields are '|'-separated so empty model strings survive word handling.
JOBS=()
for m in $MODELS; do
  mtag="${m%%:*}"; rest="${m#*:}"
  a="${rest%%:*}"; w="${rest#*:}"
  # G1 re-legs
  JOBS+=("$mtag|g1|$a|$w|0|100|rule_based_agent rule_based_agent rule_based_agent")
  JOBS+=("$mtag|g1|$a|$w|1|100|rule_based_agent rule_based_agent rule_based_agent")
  # STRONG 40x10
  for s in 0 1 2 3 4 5 6 7 8 9; do
    JOBS+=("$mtag|strong|$a|$w|$s|40|warden_v2 overlord sentinel")
  done
  # UNSEEN UMIX + legs
  for s in 0 1; do
    JOBS+=("$mtag|umix|$a|$w|$s|100|unseen_coward unseen_bomber unseen_rusher")
    JOBS+=("$mtag|ucow|$a|$w|$s|40|unseen_coward unseen_coward unseen_coward")
    JOBS+=("$mtag|ubom|$a|$w|$s|40|unseen_bomber unseen_bomber unseen_bomber")
    JOBS+=("$mtag|urus|$a|$w|$s|40|unseen_rusher unseen_rusher unseen_rusher")
    JOBS+=("$mtag|urac|$a|$w|$s|40|unseen_racer unseen_racer unseen_racer")
  done
done
echo "jobs=${#JOBS[@]}"

for j in "${JOBS[@]}"; do
  IFS='|' read -r mtag btag agent model seed rounds opps <<< "$j"
  # shellcheck disable=SC2086
  run "$mtag" "$btag" "$agent" "$model" "$seed" "$rounds" $opps &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n || true; done
done
wait
echo P0BAT_ALL_DONE
