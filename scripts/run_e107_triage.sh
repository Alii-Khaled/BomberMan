#!/usr/bin/env bash
# E107 B1-continuation triage screens: e107b1 snapshots vs fresh control,
# G1 40x1 s0 (and s1 for any arm that screens strong). Cap 4 (shares the
# box with the diag driver's sequential leg).
set -u
cd /home/jovyan/work/BomberMan
mkdir -p logs results /tmp/opencode/e107

run() { # tag seed model
  local tag="$1" seed="$2" model="$3"
  local out="results/tourney_${tag}_s${seed}.json"
  local ldir="/tmp/opencode/e107/${tag}_s${seed}"
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
  "e107ctl40 0 $PWD/agent_code/Harvy/my-saved-model.pt"
  "e107b1c150 0 $PWD/results/e107b1.pt.ep150"
  "e107b1c200 0 $PWD/results/e107b1.pt.ep200"
  "e107b1c250 0 $PWD/results/e107b1.pt.ep250"
  "e107b1c300 0 $PWD/results/e107b1.pt.ep300"
)
for j in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  run $j &
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n || true; done
done
wait
echo E107_TRIAGE_DONE
