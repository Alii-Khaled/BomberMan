#!/usr/bin/env bash
# Phase-0 gate batch (p0cash): G1 100x2 CPU, fresh same-session control.
# Candidates: E104 NG-RL legs (b1 ep050/ep150, b2 ep100/ep200) + NG-1 BC ep13/ep20.
# Each game gets its own --log-dir (shared game.log rotation races under
# concurrency); every process is single-threaded (128-core box).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results

run() { # tag agent model seed
  local tag="$1" agent="$2" model="$3" seed="$4"
  local out="results/tourney_${tag}_g1_s${seed}.json"
  local ldir="logs/p0/${tag}_s${seed}"
  mkdir -p "$ldir"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  if [ -n "$model" ]; then
    ARBITER_DEVICE=cpu ARBITER_MODEL="$model" \
      python3 scripts/tournament_eval.py \
      --agents "$agent" rule_based_agent rule_based_agent rule_based_agent \
      --n-rounds 100 --seed "$seed" --scenario classic \
      --log-dir "$ldir" \
      --match-name "${tag}_s${seed}" --out "$out" \
      > "logs/tourney_${tag}_g1_s${seed}.log" 2>&1
  else
    ARBITER_DEVICE=cpu \
      python3 scripts/tournament_eval.py \
      --agents "$agent" rule_based_agent rule_based_agent rule_based_agent \
      --n-rounds 100 --seed "$seed" --scenario classic \
      --log-dir "$ldir" \
      --match-name "${tag}_s${seed}" --out "$out" \
      > "logs/tourney_${tag}_g1_s${seed}.log" 2>&1
  fi
  echo "DONE $tag s$seed rc=$?"
}

run p0gate_ctl    arbiter    ""                                     0 &
run p0gate_ctl    arbiter    ""                                     1 &
run p0gate_b1e050 arbiter_ng "$PWD/results/e104_b1.pt.ep050"        0 &
run p0gate_b1e050 arbiter_ng "$PWD/results/e104_b1.pt.ep050"        1 &
run p0gate_b1e150 arbiter_ng "$PWD/results/e104_b1.pt.ep150"        0 &
run p0gate_b1e150 arbiter_ng "$PWD/results/e104_b1.pt.ep150"        1 &
run p0gate_b2e100 arbiter_ng "$PWD/results/e104_b2.pt.ep100"        0 &
run p0gate_b2e100 arbiter_ng "$PWD/results/e104_b2.pt.ep100"        1 &
run p0gate_b2e200 arbiter_ng "$PWD/results/e104_b2.pt.ep200"        0 &
run p0gate_b2e200 arbiter_ng "$PWD/results/e104_b2.pt.ep200"        1 &
run p0gate_ng13   arbiter_ng "$PWD/results/arbiter_ng_full.pt.ep13" 0 &
run p0gate_ng13   arbiter_ng "$PWD/results/arbiter_ng_full.pt.ep13" 1 &
run p0gate_ng20   arbiter_ng "$PWD/results/arbiter_ng_full.pt"      0 &
run p0gate_ng20   arbiter_ng "$PWD/results/arbiter_ng_full.pt"      1 &
wait
echo P0GATE_ALL_DONE
