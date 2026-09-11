#!/bin/bash
# Re-collect apex-format BC demos (S3): the 2026-09-09 disk cleanup wiped
# results/apex_demos/ (400 npz, E51). Restores warden_vN/sentinel/overlord
# at E51 parity and ADDS coin_collector_agent (100) — the only agent in the
# repo realizing 3.39 crates/bomb, hence the only source of high-yield
# placement demonstrations for ARBITER's V (E62/S3 decision).
# Recorder: agent_code/apex_teacher/ delegates act() to APEX_TEACHER and
# writes B3 npz (img uint8 T,12,17,17 x4 + sc T,16 + act T,) to
# results/apex_demos/<teacher>/ (resume-safe: appends round IDs).
# Fields per teacher: rb 50% / warden-mix 25% / 3x collector 12.5% /
# crate-light 12.5% (E41 mixed-density coverage lesson). No random field:
# random agents die early and boards are unrepresentative.
# Usage: bash scripts/collect_apex_demos.sh [teacher ...]
#   (default: warden_v2 sentinel overlord coin_collector_agent)
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
WARDEN="${WARDEN:-warden_v2}"
OUT="$(pwd)/results/apex_demos"
W=${WARDEN_N:-200}
S=${SENTINEL_N:-100}
O=${OVERLORD_N:-100}
C=${COLLECTOR_N:-100}

mkdir -p results/apex_demos logs

collect() { # teacher, field-tag, scenario, opponents..., n-rounds
  local t="$1"; shift
  local tag="$1"; shift
  local sc="$1"; shift
  local n="${@: -1}"
  local opps="${@:1:$#-1}"
  if [ "$n" -le 0 ]; then return 0; fi
  echo "=== apex-demos $t [$tag/$sc] N=$n ==="
  # shellcheck disable=SC2086
  APEX_TEACHER="$t" APEX_DEMO_OUT="$OUT" \
    $PY main.py play --no-gui --agents apex_teacher $opps \
      --train 1 --continue-without-training --scenario "$sc" \
      --n-rounds "$n" --save-stats "results/apex_demos_${t}_${tag}.json" 2>&1 | tail -1
}

if [ "$#" -eq 0 ]; then set -- warden_v2 sentinel overlord coin_collector_agent; fi
for t in "$@"; do
  case "$t" in
    warden|warden_v1|warden_v2) N=$W ;;
    sentinel)            N=$S ;;
    overlord)            N=$O ;;
    coin_collector_agent) N=$C ;;
    *) echo "unknown teacher $t"; exit 1 ;;
  esac
  if [ "$N" -le 0 ]; then continue; fi
  n_rb=$(python3 -c "print(int($N*0.50))")
  n_wm=$(python3 -c "print(int($N*0.25))")
  n_cl=$(python3 -c "print(int($N*0.125))")
  n_cr=$(python3 -c "print(int($N*0.125))")
  n_rb=$(( n_rb + N - n_rb - n_wm - n_cl - n_cr ))
  collect "$t" rb classic rule_based_agent rule_based_agent rule_based_agent "$n_rb"
  collect "$t" wm classic $WARDEN rule_based_agent rule_based_agent "$n_wm"
  collect "$t" co classic coin_collector_agent coin_collector_agent coin_collector_agent "$n_cl"
  collect "$t" cr crate-light rule_based_agent rule_based_agent rule_based_agent "$n_cr"
done
echo "=== counts ==="
for d in results/apex_demos/*/; do echo "$d $(ls "$d" | wc -l) npz"; done
echo "Done."
