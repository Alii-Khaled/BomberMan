#!/bin/bash
# Collect BC demos for reaper from the teachers (warden/sentinel/overlord).
# The recorder delegates act() to the teacher and writes (features, action)
# npz files to results/demos/<teacher>/ (training-time only, never ships).
# Teachers act greedily (their own train flags are absent -> epsilon 0).
#
# E37/P4: demos are collected in the GATE fields (3x rb, warden-mix, 3x
# random, 3x collector) — NOT vs weak peaceful/collector lineups — so the BC
# distribution matches where the policy will be evaluated. Per-teacher
# budgets split across fields: WARDEN_N/SENTINEL_N/OVERLORD_N totals, or
# set FIELD_N=0 to skip a field. STAGE_DAGGER_N>0 appends a DAgger round:
# the current student (REAPER_STUDENT_PT, default my-saved-model.pt) acts
# and the teacher labels the visited states -> results/demos/<teacher>_dagger/.
# Usage: TEACHER_N=300 bash scripts/collect_demos.sh [teacher ...]
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
W=${WARDEN_N:-400}
S=${SENTINEL_N:-200}
O=${OVERLORD_N:-200}
# field split: rb / warden-mix / random / collector (fractions of teacher N)
FRB=${FIELD_RB:-0.50}
FWM=${FIELD_WM:-0.25}
FRN=${FIELD_RN:-0.125}
FCL=${FIELD_CL:-0.125}
STAGE_DAGGER_N=${STAGE_DAGGER_N:-0}

mkdir -p results/demos logs

collect() { # teacher, tag, opponents..., n-rounds
  local t="$1"; shift
  local tag="$1"; shift
  local n="${@: -1}"
  local opps="${@:1:$#-1}"
  if [ "$n" -le 0 ]; then return 0; fi
  echo "=== collecting $t [$tag] N=$n ==="
  # shellcheck disable=SC2086
  TEACHER=$t REAPER_DEMO_DIR="results/demos/${t}${tag}" \
    $PY main.py play --no-gui --agents reaper_teacher $opps \
      --train 1 --continue-without-training --scenario classic \
      --n-rounds "$n" --save-stats "results/demos_${t}${tag}.json"
}

if [ "$#" -eq 0 ]; then set -- warden sentinel overlord; fi
for t in "$@"; do
  case "$t" in
    warden)   N=$W ;;
    sentinel) N=$S ;;
    overlord) N=$O ;;
    *) echo "unknown teacher $t"; exit 1 ;;
  esac
  if [ "$N" -le 0 ]; then continue; fi
  # integer split (remainder -> rb field)
  n_rb=$(python3 -c "print(int($N*$FRB))")
  n_wm=$(python3 -c "print(int($N*$FWM))")
  n_rn=$(python3 -c "print(int($N*$FRN))")
  n_cl=$(python3 -c "print(int($N*$FCL))")
  n_rb=$(( n_rb + N - n_rb - n_wm - n_rn - n_cl ))
  collect "$t" ""   rule_based_agent rule_based_agent rule_based_agent "$n_rb"
  collect "$t" "_wm" warden_v1 rule_based_agent rule_based_agent "$n_wm"
  collect "$t" "_rn" random_agent random_agent random_agent "$n_rn"
  collect "$t" "_cl" coin_collector_agent coin_collector_agent coin_collector_agent "$n_cl"
  if [ "$STAGE_DAGGER_N" -gt 0 ]; then
    echo "=== dagger $t (student acts, teacher labels) N=$STAGE_DAGGER_N ==="
    TEACHER=$t REAPER_DAGGER=1 REAPER_DEMO_DIR="results/demos/${t}_dagger" \
      $PY main.py play --no-gui --agents reaper_teacher \
        rule_based_agent rule_based_agent rule_based_agent \
        --train 1 --continue-without-training --scenario classic \
        --n-rounds "$STAGE_DAGGER_N" --save-stats "results/demos_${t}_dagger.json"
  fi
done
echo "Done. Demo files in results/demos/"
