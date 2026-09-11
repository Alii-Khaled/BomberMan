#!/bin/bash
# Reaper curriculum (BC-init fine-tune). Resumable per stage; guard: manual.
#   C1 economy 150: solo classic (placement + survival calibration)
#   C2 hunt    200: vs peaceful + coin_collector
#   C3 mixed   500: vs rule_based + warden_vN + sentinel (robustness)
#   C4 gate    150: vs 3x rule_based (gate consolidation)
#   + frozen gates (CPU): see scripts/eval_reaper.sh
#
# Run isolation (E37/P4): REAPER_RUN_DIR redirects checkpoints/metrics/exports
# (see train.py), REAPER_TAG prefixes results/archive files so K parallel
# sweep jobs never clobber each other. SKIP_GATES=1 skips the final eval
# (the sweep does its own bake-off). BC init: if the run has no last.pt but
# REAPER_BC_INIT exists, it is installed as the weights-only start.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
WARDEN="${WARDEN:-warden_v2}"
export REAPER_OPT=adam REAPER_TUNED=1
export REAPER_BATCH=256 REAPER_UTD=1 REAPER_EOR_UPDATES=4
export REAPER_EPS_START=${REAPER_EPS_START:-0.10}
export REAPER_EPS_DECAY=50000 REAPER_SAVE_EVERY=5
export REAPER_DEMOS=${REAPER_DEMOS:-"results/demos/*/round_*.npz"}
export REAPER_DEMO_RATIO=${REAPER_DEMO_RATIO:-0.25}
export REAPER_BC_W=${REAPER_BC_W:-0.5}
C1=${STAGE_C1_N:-150}
C2=${STAGE_C2_N:-200}
C3=${STAGE_C3_N:-500}
C4=${STAGE_C4_N:-150}
RD=${REAPER_RUN_DIR:-agent_code/reaper}
TAG=${REAPER_TAG:-}
BC_INIT=${REAPER_BC_INIT:-agent_code/reaper/checkpoints/bc_last.pt}
SEED_ARG=""
if [ -n "${REAPER_SEED:-}" ]; then SEED_ARG="--seed $REAPER_SEED"; fi

reset_for_next_stage() {
  RD="$RD" TAG="$TAG" STAGE="$1" $PY -c "
import torch, os, shutil
rd = os.environ['RD']
p = os.path.join(rd, 'checkpoints', 'last.pt')
ckpt = torch.load(p, map_location='cpu', weights_only=False)
shutil.copy(p, 'results/archive/%sreaper_%s_best_snapshot.pt' % (os.environ['TAG'], os.environ['STAGE']))
ckpt['best_ema'] = None; ckpt['ema_reward'] = None
ckpt['epsilon_steps'] = 42500  # ~=0.10 re-warm, env-step units
torch.save(ckpt, p + '.tmp'); os.replace(p + '.tmp', p)
print('reset for next stage after %s: best archived, EMA cleared, eps~=0.10' % os.environ['STAGE'])
"
}

mkdir -p results/archive logs "$RD/checkpoints"
if [ ! -f "$RD/checkpoints/last.pt" ] && [ -f "$BC_INIT" ]; then
  echo "=== BC init: $BC_INIT -> $RD/checkpoints/last.pt ==="
  cp "$BC_INIT" "$RD/checkpoints/last.pt"
fi

if [ "$C1" -gt 0 ]; then
echo "=== [${TAG:-main}] C1 (solo economy) N=$C1 ==="
$PY main.py play --no-gui --agents reaper --train 1 --scenario classic --n-rounds $C1 $SEED_ARG --save-stats results/${TAG}reaper_c1.json
reset_for_next_stage C1
fi

if [ "$C2" -gt 0 ]; then
echo "=== [${TAG:-main}] C2 (hunt) N=$C2 ==="
$PY main.py play --no-gui --agents reaper peaceful_agent coin_collector_agent --train 1 --scenario classic --n-rounds $C2 $SEED_ARG --save-stats results/${TAG}reaper_c2.json
reset_for_next_stage C2
fi

if [ "$C3" -gt 0 ]; then
echo "=== [${TAG:-main}] C3 (mixed sparring) N=$C3 ==="
$PY main.py play --no-gui --agents reaper rule_based_agent $WARDEN sentinel --train 1 --scenario classic --n-rounds $C3 $SEED_ARG --save-stats results/${TAG}reaper_c3.json
reset_for_next_stage C3
fi

if [ "$C4" -gt 0 ]; then
echo "=== [${TAG:-main}] C4 (gate consolidation) N=$C4 ==="
$PY main.py play --no-gui --agents reaper rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds $C4 $SEED_ARG --save-stats results/${TAG}reaper_c4.json
reset_for_next_stage C4
fi

if [ "${SKIP_GATES:-0}" != "1" ] && [ -z "$TAG" ]; then
echo "=== Frozen gates (CPU) ==="
bash scripts/eval_reaper.sh
fi
echo "Done. Metrics: $RD/runs/metrics.csv"
