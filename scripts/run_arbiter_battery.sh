#!/usr/bin/env bash
# Arbiter tournament battery (per-round winners/ranks + bootstrap CIs).
#
# Modes:
#   g1     : <agent> vs 3x rule_based_agent        (pooled score + G1 win rate)
#   strong : <agent> + warden_v2 + overlord + sentinel
#            (head-to-head vs warden_v2 in the STRONG field, one run gives all)
#   ab     : <agent> warden_v2 + 2x rule_based      (paired head-to-head)
#
# Usage:
#   AGENT=arbiter AGENT_TAG=_e88 bash scripts/run_arbiter_battery.sh g1 100 0 1 2 3 4 5
#   AGENT=arbiter AGENT_TAG=_e88 bash scripts/run_arbiter_battery.sh strong 40 0 1 2 3 4 5 6 7 8 9
#
# Outputs: results/tourney_<agent><tag>_<mode>_s<seed>.json
set -u
cd "$(dirname "$0")/.."
MODE="${1:?mode: g1|strong|ab}"
N="${2:?n-rounds}"
shift 2
SEEDS=("$@")
[ "${#SEEDS[@]}" -eq 0 ] && SEEDS=(0)
AGENT="${AGENT:-arbiter}"
TAG="${AGENT_TAG:-}"
STRONG_EXTRA="${STRONG_EXTRA:-warden_v2 overlord sentinel}"

case "$MODE" in
  g1)     OPPONENTS=(rule_based_agent rule_based_agent rule_based_agent) ;;
  strong) # shellcheck disable=SC2206
          OPPONENTS=($STRONG_EXTRA) ;;
  ab)     OPPONENTS=(warden_v2 rule_based_agent rule_based_agent) ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

for seed in "${SEEDS[@]}"; do
  out="results/tourney_${AGENT}${TAG}_${MODE}_s${seed}.json"
  log="logs/tourney_${AGENT}${TAG}_${MODE}_s${seed}.log"
  logdir="logs/${AGENT}${TAG}_${MODE}_s${seed}"
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP existing $out (FORCE=1 to rerun)"
    continue
  fi
  echo "=== $MODE seed=$seed: $AGENT ${OPPONENTS[*]} (N=$N)"
  mkdir -p "$logdir"
  python3 scripts/tournament_eval.py \
    --agents "$AGENT" "${OPPONENTS[@]}" \
    --n-rounds "$N" --seed "$seed" --scenario classic \
    --log-dir "$logdir" \
    --match-name "agent_battery_${AGENT}${TAG}_${MODE}_s${seed}" \
    --out "$out" >"$log" 2>&1 || { echo "FAILED seed $seed (see $log)"; exit 1; }
  tail -n 8 "$log"
done
echo "AGENT BATTERY $AGENT $MODE DONE"
