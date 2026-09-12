#!/usr/bin/env bash
# E101 ARBITER-NG league corpus: apex-format (img 12x17x17 + sc + act)
# recordings from a warden-heavy teacher league. Consumed by
# scripts/pretrain_arbiter_ng.py.
# Usage: bash scripts/collect_arbiter_ng_league.sh
# Resume-safe (apex_teacher continues round numbering per teacher dir).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
OUT="${APEX_DEMO_OUT:-results/apex_ng_demos}"
mkdir -p "$OUT" logs

collect() { # teacher, tag, n-rounds, opponents...
  local t="$1"; shift
  local tag="$1"; shift
  local n="$1"; shift
  if [ "$n" -le 0 ]; then return 0; fi
  local out="$OUT/$t"
  local have=0
  [ -d "$out" ] && have=$(ls "$out" 2>/dev/null | wc -l)
  if [ "$have" -ge "$n" ]; then
    echo "SKIP $t [$tag] have=$have >= $n"
    return 0
  fi
  echo "=== ng-league $t [$tag] N=$n (have $have)"
  # shellcheck disable=SC2086
  APEX_TEACHER="$t" APEX_DEMO_OUT="$OUT" \
    $PY main.py play --no-gui --agents apex_teacher "$@" \
      --train 1 --continue-without-training --scenario classic \
      --n-rounds "$n" --save-stats "results/apex_ng_${t}_${tag}.json" \
      2>&1 | tail -1
}

# warden-heavy split: rb anchor + warden-mix dominant + a little weak
collect arbiter   rb 200 rule_based_agent rule_based_agent rule_based_agent
collect arbiter   wm 150 warden_v2 rule_based_agent rule_based_agent
collect arbiter   rn 25  random_agent random_agent random_agent
collect arbiter   cl 25  coin_collector_agent coin_collector_agent coin_collector_agent
collect warden_v2 rb 200 rule_based_agent rule_based_agent rule_based_agent
collect warden_v2 wm 150 warden_v2 rule_based_agent rule_based_agent
collect warden_v2 rn 25  random_agent random_agent random_agent
collect warden_v2 cl 25  coin_collector_agent coin_collector_agent coin_collector_agent
collect overlord  rb 75  rule_based_agent rule_based_agent rule_based_agent
collect overlord  wm 75  warden_v2 rule_based_agent rule_based_agent
collect sentinel  rb 75  rule_based_agent rule_based_agent rule_based_agent
collect sentinel  wm 75  warden_v2 rule_based_agent rule_based_agent
collect coin_collector_agent rb 50 rule_based_agent rule_based_agent rule_based_agent
collect coin_collector_agent wm 50 warden_v2 rule_based_agent rule_based_agent
collect random_agent rb 50 rule_based_agent rule_based_agent rule_based_agent
echo "NG LEAGUE COLLECTION DONE -> $OUT"
