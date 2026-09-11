#!/usr/bin/env bash
# Warden tournament battery (versioned naming: always warden_vN).
#
# Runs scripts/tournament_eval.py (per-round winners/ranks + bootstrap CIs)
# for the warden under test:
#   g1     : warden vs 3x rule_based_agent      (score bar + G1 win rate)
#   strong : warden in the STRONG field-proxy   (arbiter + overlord + sentinel)
#   ab     : warden vs arbiter + 2x rule_based  (paired head-to-head)
#
# Usage:
#   WARDEN=warden_v1 bash scripts/run_warden_battery.sh g1 100 0 1
#   WARDEN=warden_v2 bash scripts/run_warden_battery.sh strong 40 0 1 2 3 4
#   WARDEN=warden_v2 bash scripts/run_warden_battery.sh ab 40 0 1
#
# Outputs: results/tourney_<warden>_<mode>_s<seed>.json
set -u
cd "$(dirname "$0")/.."
MODE="${1:?mode: g1|strong|ab}"
N="${2:?n-rounds}"
shift 2
SEEDS=("$@")
[ "${#SEEDS[@]}" -eq 0 ] && SEEDS=(0)
WARDEN="${WARDEN:-warden_v2}"
STRONG_EXTRA="${WARDEN_STRONG:-arbiter overlord sentinel}"
TAG="${WARDEN_TAG:-}"

case "$MODE" in
  g1)     OPPONENTS=(rule_based_agent rule_based_agent rule_based_agent) ;;
  strong) # shellcheck disable=SC2206
          OPPONENTS=($STRONG_EXTRA) ;;
  ab)     OPPONENTS=(arbiter rule_based_agent rule_based_agent) ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

for seed in "${SEEDS[@]}"; do
  out="results/tourney_${WARDEN}${TAG}_${MODE}_s${seed}.json"
  log="logs/tourney_${WARDEN}${TAG}_${MODE}_s${seed}.log"
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP existing $out (FORCE=1 to rerun)"
    continue
  fi
  echo "=== $MODE seed=$seed: $WARDEN ${OPPONENTS[*]} (N=$N)"
  python3 scripts/tournament_eval.py \
    --agents "$WARDEN" "${OPPONENTS[@]}" \
    --n-rounds "$N" --seed "$seed" --scenario classic \
    --match-name "warden_battery_${WARDEN}_${MODE}_s${seed}" \
    --out "$out" >"$log" 2>&1 || { echo "FAILED seed $seed (see $log)"; exit 1; }
  tail -n 8 "$log"
done
echo "WARDEN BATTERY $WARDEN $MODE DONE"
