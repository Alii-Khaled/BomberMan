#!/bin/bash
# Reaper parallel hyperparameter sweep (E37/P4a).
# Launches K independent curriculum runs (scripts/train_reaper.sh) with
# isolated REAPER_RUN_DIR + REAPER_TAG each, sharing the L40S for gradient
# steps (tiny MLP) while rollout Python spreads over the 128 cores.
# Prereq: BC init at agent_code/reaper/checkpoints/bc_last.pt
#   (python3 scripts/pretrain_reaper.py --demos results/demos/*/*.npz ...)
# Usage: bash scripts/sweep_reaper.sh [job ...]
#   default: all 8 jobs. Each logs to logs/sweep_<job>.log (nohup'd).
#   SKIP_GATES=1 is forced; bake off with scripts/bakeoff_reaper.sh.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs results/sweeps

BC=agent_code/reaper/checkpoints/bc_last.pt
if [ ! -f "$BC" ]; then
  echo "missing BC init $BC - run pretrain_reaper.py first"
  exit 1
fi

# job_name | extra env for train_reaper.sh (REAPER_RUN_DIR/TAG auto-added)
JOBS=(
  "sw01_base|"
  "sw02_seed1|REAPER_SEED=1"
  "sw03_killheavy|REAPER_W_PHI_KILL=3.0 REAPER_W_PHI_COIN=0.5 REAPER_W_PHI_MOB=0.5"
  "sw04_lowheur|REAPER_HEUR_WEIGHT=0.25"
  "sw05_lowlr|REAPER_LR=0.0003"
  "sw06_shorthor|REAPER_GAMMA=0.985 REAPER_N_STEP=6"
  "sw07_longhor|REAPER_GAMMA=0.995 REAPER_N_STEP=10"
  "sw08_noshaping|REAPER_W_PHI_KILL=0 REAPER_W_PHI_COIN=0 REAPER_W_PHI_MOB=0"
)

launch() { # "name|env..."
  local spec="$1"
  local name="${spec%%|*}"
  local extra="${spec#*|}"
  # Absolute run dir: train.py resolves a relative REAPER_RUN_DIR against
  # the repo root (backend chdir's into the agent dir around callbacks),
  # but absolute here is unambiguous everywhere (checkpoints, metrics,
  # results JSONs, archive snapshots).
  local rd="$(pwd)/results/sweeps/$name"
  mkdir -p "$rd/checkpoints"
  if [ ! -f "$rd/checkpoints/last.pt" ]; then
    cp "$BC" "$rd/checkpoints/last.pt"
  fi
  echo "=== launching $name ($extra) ==="
  # shellcheck disable=SC2086
  env REAPER_RUN_DIR="$rd" REAPER_TAG="${name}_" SKIP_GATES=1 \
      REAPER_BC_INIT="$BC" $extra \
    nohup bash scripts/train_reaper.sh > "logs/sweep_${name}.log" 2>&1 &
  echo "$! > logs/sweep_${name}.pid"
  echo $! > "logs/sweep_${name}.pid"
}

if [ "$#" -gt 0 ]; then
  for j in "$@"; do
    for spec in "${JOBS[@]}"; do
      if [ "${spec%%|*}" = "$j" ]; then launch "$spec"; fi
    done
  done
else
  for spec in "${JOBS[@]}"; do launch "$spec"; done
fi
echo "Launched. Watch: tail -f logs/sweep_sw01_base.log ; progress: grep -h 'C[0-9] (' logs/sweep_*.log"
