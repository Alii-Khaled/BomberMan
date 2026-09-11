#!/bin/bash
# Overlord hybrid cross-sparring (E46/H3) — QUEUED, DO NOT LAUNCH YET.
# Launches only on: (1) E36-confirmed reaper weights in agent_code/reaper/,
# (2) explicit go. New distribution (learned diverse foe), same classic
# regime — the one training mechanism not yet tried on overlord.
#   S1 250: vs reaper + 2x rule_based
#   S2 250: vs warden_vN + 2x rule_based
#   + frozen gates: 100 rb (seeds 0+1) + 60 reaper-mix + 60 warden-mix.
# Base (set at launch): o3sbest archive -> last.pt, EMA cleared, eps 0.15.
# Guard: watch_overlord_focused.sh (add train_overlord_hybspar match first).
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
WARDEN="${WARDEN:-warden_v2}"
export OVERLORD_OPT=adam OVERLORD_TUNED=1
export OVERLORD_BATCH=1024 OVERLORD_UTD=2 OVERLORD_EOR_UPDATES=12
export OVERLORD_EPS_DECAY=100000 OVERLORD_SAVE_EVERY=5
export OVERLORD_CHANNELS_LAST=1 OVERLORD_COMPILE=0
export OVERLORD_BASE=96 OVERLORD_FC=512 OVERLORD_NORM=bn OVERLORD_DEEP=0
S1=${STAGE_S1_N:-250}
S2=${STAGE_S2_N:-250}

echo "QUEUED (E46/H3): requires E36-confirmed reaper weights + explicit go."
echo "Base setup (run at launch): cp results/archive/overlord_val_best_O3s.pt agent_code/overlord/checkpoints/last.pt + EMA/eps reset."
exit 0

mkdir -p results/archive logs
if [ "$S1" -gt 0 ]; then
echo "=== S1 (reaper sparring) N=$S1 ==="
$PY main.py play --no-gui --agents overlord reaper rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $S1 --save-stats results/overlord_hyb_s1.json
fi

if [ "$S2" -gt 0 ]; then
echo "=== S2 (warden sparring) N=$S2 ==="
$PY main.py play --no-gui --agents overlord $WARDEN rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $S2 --save-stats results/overlord_hyb_s2.json
fi

echo "=== Frozen gates (CPU) ==="
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --seed 0 --save-stats results/overlord_hyb_eval_rb_s0.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord rule_based_agent rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 100 --seed 1 --save-stats results/overlord_hyb_eval_rb_s1.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord reaper rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 60 --seed 0 --save-stats results/overlord_hyb_eval_reaper.json
OVERLORD_DEVICE=cpu $PY main.py play --no-gui --agents overlord $WARDEN rule_based_agent rule_based_agent --train 0 --continue-without-training --scenario classic --n-rounds 60 --seed 0 --save-stats results/overlord_hyb_eval_warden.json
echo "Done. Metrics: agent_code/overlord/runs/metrics.csv"
