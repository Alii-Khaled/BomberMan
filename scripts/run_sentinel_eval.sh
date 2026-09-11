#!/bin/bash
# Frozen evaluation matrix for the current sentinel model.
# 100 rounds x seeds {0,1,2} per matchup (report-grade CIs). Resumable:
# existing results/eval_<matchup>_seed<S>.json files are skipped.
# Usage: bash scripts/run_sentinel_eval.sh [matchup-id ...]
#   e.g. bash scripts/run_sentinel_eval.sh m3 m6   (run subset)
# Env: SENTINEL_DEVICE=cpu for eval (CPU inference is what the tournament uses).
set -e
PY="${PY:-python3}"
WARDEN="${WARDEN:-warden_v2}"
N=40
SEEDS="0 1"

declare -A MATCHUPS
MATCHUPS[m1]="sentinel --scenario coin-heaven"
MATCHUPS[m2]="sentinel --scenario classic"
MATCHUPS[m3]="sentinel random_agent random_agent random_agent --scenario classic"
MATCHUPS[m4]="sentinel peaceful_agent peaceful_agent peaceful_agent --scenario classic"
MATCHUPS[m5]="sentinel coin_collector_agent coin_collector_agent coin_collector_agent --scenario classic"
MATCHUPS[m6]="sentinel rule_based_agent rule_based_agent rule_based_agent --scenario classic"
MATCHUPS[m7]="sentinel overlord rule_based_agent rule_based_agent --scenario classic"
MATCHUPS[m8]="sentinel $WARDEN rule_based_agent rule_based_agent --scenario classic"

WANT="$*"
if [ -z "$WANT" ]; then WANT="m1 m2 m3 m4 m5 m6 m7 m8"; fi

for m in $WANT; do
  # shellcheck disable=SC2086
  read -ra SPEC <<< "${MATCHUPS[$m]}"
  if [ -z "${SPEC[*]}" ]; then echo "unknown matchup: $m"; exit 1; fi
  agents=()
  scenario="classic"
  i=0
  while [ $i -lt ${#SPEC[@]} ]; do
    case "${SPEC[$i]}" in
      --scenario) scenario="${SPEC[$((i+1))]}"; i=$((i+2));;
      *) agents+=("${SPEC[$i]}"); i=$((i+1));;
    esac
  done
  for s in $SEEDS; do
    out="results/eval_${m}_seed${s}.json"
    if [ -f "$out" ]; then echo "skip $out (exists)"; continue; fi
    echo "=== $m seed=$s (${agents[*]} | $scenario) ==="
    # shellcheck disable=SC2068
    $PY main.py play --no-gui --agents ${agents[@]} --train 0 \
      --continue-without-training --scenario "$scenario" \
      --n-rounds $N --seed "$s" --save-stats "$out"
  done
done
echo "Done. Aggregate with: python3 scripts/aggregate_eval.py"
