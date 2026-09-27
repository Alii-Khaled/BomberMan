#!/usr/bin/env bash
# Corpus widening: apex-format league recordings (resume-safe).
# Lanes run concurrently (3 games max), each lane sequential internally.
#   L1: ship self-play mirror (teacher=Harvey vs Harvey) + ship
#       field top-ups (warden-mix, rb-heavy)
#   L2: warden_v1 teacher (new hunter lineage) rb + warden-mix
#   L3: coin_collector top-up + random + collector warden-mix
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
OUT="$(pwd)/results/apex_ng_demos"
mkdir -p "$OUT" logs logs/e108

collect() { # teacher tag n opponents...
  local t="$1" tag="$2" n="$3"
  shift 3
  local have=0
  [ -d "$OUT/$t" ] && have=$(ls "$OUT/$t" 2>/dev/null | wc -l)
  local target
  case "$t:$tag" in
    Harvey:mirror) target=$n ;;
    *) target=$n ;;
  esac
  echo "=== $t [$tag] N=$n (have $have)"
  # shellcheck disable=SC2086
  APEX_TEACHER="$t" APEX_DEMO_OUT="$OUT" \
    nice -n 10 python3 main.py play --no-gui --agents apex_teacher "$@" \
      --train 1 --continue-without-training --scenario classic \
      --n-rounds "$n" --save-stats "results/apexng_${t}_${tag}.json" \
      > "logs/apexng_${t}_${tag}.log" 2>&1
  echo "DONE $t $tag rc=$?"
}

lane1() {
  collect Harvey mirror 150 Harvey rule_based_agent warden_v2
  collect Harvey wm2 100 warden_v2 rule_based_agent rule_based_agent
  collect Harvey rb2 100 rule_based_agent rule_based_agent rule_based_agent
}
lane2() {
  collect warden_v1 rb 150 rule_based_agent rule_based_agent rule_based_agent
  collect warden_v1 wm 75 warden_v2 rule_based_agent rule_based_agent
}
lane3() {
  collect coin_collector_agent rb2 100 rule_based_agent rule_based_agent rule_based_agent
  collect random_agent rb2 50 rule_based_agent rule_based_agent rule_based_agent
  collect coin_collector_agent wm2 50 warden_v2 rule_based_agent rule_based_agent
}

lane1 & lane2 & lane3 &
wait
for d in Harvey warden_v1 coin_collector_agent random_agent; do
  echo "$d: $(ls "$OUT/$d" 2>/dev/null | wc -l) files"
done
echo E108_COLLECT_DONE
