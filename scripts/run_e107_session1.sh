#!/usr/bin/env bash
# E107 Session-1 driver v2: A1 diag legs via main.py (--train 1 +
# --continue-without-training loads train.py so diag_dump_round fires;
# ARBITER_RL_LR=0 + scratch OUT = no-op learning, weights untouched),
# then C1 wardenlite screens (8-wide).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results /tmp/opencode/e107

diag() { # seed
  local seed="$1"
  local out="results/diagstats_e107b1_s${seed}.json"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  nice -n 10 env ARBITER_DEVICE=cpu \
    ARBITER_DIAG="$PWD/results/diag_e107b1" \
    ARBITER_RL_LR=0 ARBITER_RL_OUT="/tmp/opencode/e107/scratch.pt" \
    python3 main.py play --no-gui \
    --agents arbiter_ng rule_based_agent rule_based_agent rule_based_agent \
    --train 1 --continue-without-training --scenario classic \
    --n-rounds 100 --seed "$seed" --save-stats "$out" \
    > "logs/diagleg_s${seed}.log" 2>&1
  echo "DONE diag s$seed rc=$?"
}

diag 0
diag 1

# C1 arms: ARBITER_OPPMODEL=wardenlite (ship weights, env-only), 8-wide
run() { # tag btag seed rounds opponents...
  local tag="$1" btag="$2" seed="$3" rounds="$4"
  shift 4
  local out="results/tourney_${tag}_${btag}_s${seed}.json"
  local ldir="/tmp/opencode/e107/${tag}_${btag}_s${seed}"
  mkdir -p "$ldir"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  nice -n 10 env ARBITER_DEVICE=cpu ARBITER_OPPMODEL=wardenlite \
    python3 scripts/tournament_eval.py --agents arbiter_ng "$@" \
    --n-rounds "$rounds" --seed "$seed" --scenario classic \
    --log-dir "$ldir" --match-name "${tag}_${btag}_s${seed}" --out "$out" \
    > "logs/tourney_${tag}_${btag}_s${seed}.log" 2>&1
  echo "DONE $tag $btag s$seed rc=$?"
}

ARM_JOBS=(
  "e107wl strong 0 40 warden_v2 overlord sentinel"
  "e107wl strong 1 40 warden_v2 overlord sentinel"
  "e107wl strong 2 40 warden_v2 overlord sentinel"
  "e107wl strong 3 40 warden_v2 overlord sentinel"
  "e107wl strong 4 40 warden_v2 overlord sentinel"
  "e107wl ucow 0 40 unseen_coward unseen_coward unseen_coward"
  "e107wl ucow 1 40 unseen_coward unseen_coward unseen_coward"
  "e107wl ubom 0 40 unseen_bomber unseen_bomber unseen_bomber"
  "e107wl ubom 1 40 unseen_bomber unseen_bomber unseen_bomber"
  "e107wl urus 0 40 unseen_rusher unseen_rusher unseen_rusher"
  "e107wl urus 1 40 unseen_rusher unseen_rusher unseen_rusher"
  "e107wl urac 0 40 unseen_racer unseen_racer unseen_racer"
  "e107wl urac 1 40 unseen_racer unseen_racer unseen_racer"
)
for j in "${ARM_JOBS[@]}"; do
  # shellcheck disable=SC2086
  run $j &
  while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n || true; done
done
wait
echo E107_S1_DONE
