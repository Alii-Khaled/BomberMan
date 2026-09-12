#!/usr/bin/env bash
# E101 ARBITER-NG self-driving pipeline: waits for the league corpus,
# trains NG-1 with per-epoch checkpoints, screens epochs at G1 40x2 with
# a same-session control, then runs NG-2 RL on the best-val model and
# screens that too. All artifacts under results/ (gitignored).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
R="$(pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== waiting for league corpus"
while ! grep -q "NG LEAGUE COLLECTION DONE" logs/collect_ng_league.log 2>/dev/null; do
  sleep 60
done
echo "=== corpus done; training NG-1 (20 epochs)"
OMP_NUM_THREADS=4 $PY scripts/pretrain_arbiter_ng.py \
  --dirs results/apex_demos,results/apex_ng_demos \
  --epochs 20 --batch 256 --cache results/arbiter_ng_scalars.npz \
  --out results/arbiter_ng_full.pt \
  --init-scalars agent_code/arbiter/my-saved-model.pt 2>&1 | tail -25

echo "=== screening NG-1 epochs (40x2) + control"
AGENT=arbiter AGENT_TAG=_ngctl3 bash scripts/run_arbiter_battery.sh g1 40 0 1 &
for ep in 02 04 06 10 20; do
  M="$R/results/arbiter_ng_full.pt.ep${ep}"
  [ -f "$M" ] || continue
  ARBITER_MODEL="$M" AGENT=arbiter_ng AGENT_TAG=_ngep${ep} \
    bash scripts/run_arbiter_battery.sh g1 40 0 1 &
done
wait
echo "=== NG-1 screens done; starting NG-2 RL (300 rounds, stable HPs)"
ARBITER_MODEL="$R/results/arbiter_ng_full.pt" \
ARBITER_RL_OUT=results/arbiter_ng_rl.pt ARBITER_RL_SAVE_EVERY=50 \
ARBITER_RL_CSV=results/arbiter_ng_rl.csv ARBITER_RL_LR=5e-5 ARBITER_RL_BETA=0.1 \
  $PY main.py play --no-gui --agents arbiter_ng rule_based_agent \
    rule_based_agent rule_based_agent --train 1 --scenario classic \
    --n-rounds 300 --seed 0 --save-stats results/arbiter_ng_rl_train.json \
    --silence-errors 2>&1 | tail -3

echo "=== screening NG-2 RL (40x2)"
ARBITER_MODEL="$R/results/arbiter_ng_rl.pt" AGENT=arbiter_ng AGENT_TAG=_ngrl \
  bash scripts/run_arbiter_battery.sh g1 40 0 1
echo "NG PIPELINE DONE"
