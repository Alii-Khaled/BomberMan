#!/usr/bin/env bash
# E108 D1 triage: e108rl snapshots vs fresh control, G1 40x1 s0 (cap 4).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results /tmp/opencode/e108

run() { # tag seed model
  local tag="$1" seed="$2" model="$3"
  local out="results/tourney_${tag}_s${seed}.json"
  local ldir="/tmp/opencode/e108/${tag}_s${seed}"
  mkdir -p "$ldir"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  nice -n 10 env ARBITER_DEVICE=cpu ARBITER_NG_MODEL="$model" \
    python3 scripts/tournament_eval.py \
    --agents Harvy rule_based_agent rule_based_agent rule_based_agent \
    --n-rounds 40 --seed "$seed" --scenario classic \
    --log-dir "$ldir" --match-name "${tag}_s${seed}" --out "$out" \
    > "logs/tourney_${tag}_s${seed}.log" 2>&1
  echo "DONE $tag s$seed rc=$?"
}

JOBS=(
  "e108ctl 0 $PWD/agent_code/Harvy/my-saved-model.pt"
  "e108rl075 0 $PWD/results/e108rl.pt.ep075"
  "e108rl150 0 $PWD/results/e108rl.pt.ep150"
  "e108rl225 0 $PWD/results/e108rl.pt.ep225"
  "e108rl300 0 $PWD/results/e108rl.pt.ep300"
)
for j in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run $j &
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n || true; done
done
wait
echo E108_TRIAGE_DONE
