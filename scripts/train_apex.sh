#!/bin/bash
# Apex training (OOM-safe defaults for 16 GiB Jupyter containers).
#
# WHY THIS EXISTS: running `main.py play --train` inside a notebook cell runs
# the full CNN+DQN loop in the Jupyter kernel process. With the legacy
# 300k float32 replay (~8 GiB raw + overhead) the cgroup (16G) OOMKills the
# whole singleuser server -> full Lab disconnect, no traceback. This script
# runs in a TERMINAL (detached), with capped replay + uint8 storage + sane
# batch/EOR, so the Lab server stays alive.
#
# Usage (in a terminal, NOT a notebook cell):
#   bash scripts/train_apex.sh
#   STAGE_A1_N=750 STAGE_A2_N=1500 bash scripts/train_apex.sh
#   STAGE_A1_N=0 STAGE_A2_N=0 STAGE_A3_N=750 bash scripts/train_apex.sh
#     (A3 = Task-3 hunting vs peaceful+collector; run
#     scripts/apex_archive_A2_for_A3.sh first for the EMA/epsilon reset)
#   APEX_DEMO=results/apex_demos_all STAGE_A1_N=0 STAGE_A2_N=0 STAGE_A3_N=0 \
#     STAGE_A4_N=750 bash scripts/train_apex.sh
#     (A4 = A3 + DQfD demos; run scripts/apex_archive_A3_for_A4.sh first)
#   nohup bash scripts/train_apex.sh > logs/apex_train.log 2>&1 &
#   tmux new -d -s apex 'bash scripts/train_apex.sh > logs/apex_A2.log 2>&1'
#   tmux attach -t apex   # reattach after the browser/terminal tab dies
#
# WHY tmux/detached: the Jupyter terminal tab (xterm.js) can die at random
# points on multi-hour runs while the backend keeps going. main.py now
# defaults to plain newline logs for --no-gui (APEX_TQDM=0) instead of a
# per-round tqdm \\r rewrite, but always launch detached so a dead tab
# never takes training down with it. If the tab dies: open a fresh one
# and run `tmux ls; tmux attach -t apex` — do NOT start a second training.
#
# Env overrides (all optional):
#   APEX_TQDM (default 0) — 0 = plain "[progress] round N/M" log every
#     APEX_LOG_EVERY (default 25); 1 = throttled tqdm bar (~1 update/min).
#     TQDM_DISABLE=1 / NO_TQDM=1 also force plain iteration in main.py.
#   APEX_LOG_EVERY (default 25) — newline progress cadence.
#   APEX_BUFFER (default 100000, max 300000) — replay cap. 100k uint8 ~= 0.7G.
#   APEX_BATCH (default 256 here, code default 512) — per-update batch.
#   APEX_UTD / APEX_EOR_UPDATES (defaults 1 / 2 here) — update fan-out.
#   APEX_SAVE_EVERY (default 5), APEX_EPS_DECAY (default 100000).
#   APEX_EXPORT_CWD=1 restores the legacy second my-saved-model.pt copy.
#   PY (default python3), SEED via APEX_SEED.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
# Unbuffered + faulthandler so a killed run still leaves a traceback/tail.
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
# Terminal-safe progress: plain newline logs, no live \\r bar (see main.py).
export APEX_TQDM=${APEX_TQDM:-0} APEX_LOG_EVERY=${APEX_LOG_EVERY:-25}
export APEX_OPT=${APEX_OPT:-adam} APEX_TUNED=${APEX_TUNED:-0}
export APEX_BATCH=${APEX_BATCH:-256} APEX_UTD=${APEX_UTD:-1} APEX_EOR_UPDATES=${APEX_EOR_UPDATES:-2}
export APEX_BUFFER=${APEX_BUFFER:-100000}
export APEX_EPS_DECAY=${APEX_EPS_DECAY:-100000} APEX_SAVE_EVERY=${APEX_SAVE_EVERY:-5}
export APEX_CHANNELS_LAST=${APEX_CHANNELS_LAST:-1} APEX_COMPILE=${APEX_COMPILE:-0}
export APEX_BASE=${APEX_BASE:-96} APEX_FC=${APEX_FC:-512} APEX_NORM=${APEX_NORM:-bn} APEX_DEEP=${APEX_DEEP:-0}
# DQfD demos (A4+): flat dir of B3 npz (see results/apex_demos_all symlink
# farm; train.py globs non-recursive). Empty = pure TD (A1-A3 behavior).
export APEX_DEMO=${APEX_DEMO:-} APEX_DEMO_W=${APEX_DEMO_W:-1.0}
# Absolutize: the backend chdirs into agent_code/<name>/ around every
# callback, so a relative APEX_DEMO would resolve nowhere (0 pairs, DQfD
# silently off — caught once on A4 launch). Anchor at repo root (pwd here).
if [ -n "${APEX_DEMO:-}" ] && [ "${APEX_DEMO#/}" = "${APEX_DEMO}" ]; then
  _abs="$(pwd)/${APEX_DEMO}"; [ -d "$_abs" ] && export APEX_DEMO="$_abs"
  unset _abs
fi
# Hard gate (S3): a configured-but-empty demo dir means DQfD is silently
# off (E60's 30-round pure-TD incident; train.py only warns). Refuse to
# launch rather than burn GPU hours on the wrong objective.
if [ -n "${APEX_DEMO:-}" ]; then
  # -readable (not just -name): the farm is symlinks by design, and a
  # dangling link must NOT count (wiped 2026-09-09 left 400 of them).
  _n=$(find "${APEX_DEMO}" -maxdepth 1 -name '*.npz' -readable 2>/dev/null | wc -l)
  if [ "$_n" -eq 0 ]; then
    echo "REFUSE: APEX_DEMO=${APEX_DEMO} holds 0 readable npz (dangling? wiped 2026-09-09 once already)"; exit 1
  fi
  echo "demo gate OK: $_n readable npz in ${APEX_DEMO}"
  unset _n
fi
A1=${STAGE_A1_N:-400}
A2=${STAGE_A2_N:-0}
A3=${STAGE_A3_N:-0}
A4=${STAGE_A4_N:-0}
SEED_ARG=""
if [ -n "${APEX_SEED:-}" ]; then SEED_ARG="--seed $APEX_SEED"; fi

echo "apex container-safe launch: BUFFER=$APEX_BUFFER BATCH=$APEX_BATCH UTD=$APEX_UTD EOR=$APEX_EOR_UPDATES"
echo "progress mode: APEX_TQDM=$APEX_TQDM (0=plain log every $APEX_LOG_EVERY rounds, 1=throttled bar)"
echo "free -h:"; free -h | head -n 3 || true
echo "df -h work:"; df -h /home/jovyan/work 2>/dev/null | tail -n 1 || df -h . | tail -n 1
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>&1 | head -n 5 || true
mkdir -p results/archive logs agent_code/apex/checkpoints
# Quota preflight: /home/jovyan/work is a small NFS volume (2G). Warn early
# instead of dying mid-run on checkpoint/metric writes (which are caught and
# only warn, but NFS stalls look like a hang).
_AVAIL_MB=$(df -m /home/jovyan/work 2>/dev/null | tail -n 1 | awk '{print $4}'); _AVAIL_MB=${_AVAIL_MB:-999999}
if [ "$_AVAIL_MB" -lt 500 ]; then
  echo "WARNING: only ${_AVAIL_MB}MB free on /home/jovyan/work — prune logs/game.log* / old checkpoints before long runs."
fi
du -sh logs results agent_code/apex/checkpoints 2>/dev/null || true
if [ -f agent_code/apex/checkpoints/last.pt ]; then
  echo "resume: checkpoints/last.pt present ($(du -h agent_code/apex/checkpoints/last.pt | cut -f1)) — training will resume, metrics.csv appends."
else
  echo "resume: no checkpoints/last.pt — fresh start."
fi
# Exit/signal trap: distinguishes clean finish from kill/crash. Runs on any
# exit so a dead terminal/tab still leaves code + resources in the log file.
log_resources() {
  echo "--- resources @ $(date -u '+%F %T UTC') exit=${1:-?} ---"
  free -h | head -n 3 || true
  df -h /home/jovyan/work 2>/dev/null | tail -n 1 || df -h . | tail -n 1
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ';'; echo || true
  tail -n 3 agent_code/apex/runs/metrics.csv 2>/dev/null || echo "no metrics yet"
}
trap 'rc=$?; log_resources $rc; echo "train_apex.sh exiting rc=$rc"; exit $rc' EXIT
trap 'echo "SIGHUP received (terminal/tab closed) — continuing detached under tmux/nohup"; ' HUP
trap 'echo "SIGTERM/SIGINT received"; exit 143' TERM INT

if [ "$A1" -gt 0 ]; then
echo "=== A1 (coin-heaven warmup) N=$A1 ==="
$PY main.py play --no-gui --agents apex coin_collector_agent coin_collector_agent coin_collector_agent --train 1 --scenario coin-heaven --n-rounds $A1 $SEED_ARG --save-stats results/apex_a1.json
fi

if [ "$A2" -gt 0 ]; then
echo "=== A2 (classic solo) N=$A2 ==="
$PY main.py play --no-gui --agents apex --train 1 --scenario classic --n-rounds $A2 $SEED_ARG --save-stats results/apex_a2.json
fi

if [ "$A3" -gt 0 ]; then
echo "=== A3 (Task-3 hunting: vs peaceful+collector) N=$A3 ==="
$PY main.py play --no-gui --agents apex peaceful_agent coin_collector_agent --train 1 --scenario classic --n-rounds $A3 $SEED_ARG --save-stats results/apex_a3.json
fi

if [ "$A4" -gt 0 ]; then
echo "=== A4 (DQfD demos ON: vs peaceful+collector) N=$A4 DEMO=${APEX_DEMO:-<off>} W=$APEX_DEMO_W ==="
$PY main.py play --no-gui --agents apex peaceful_agent coin_collector_agent --train 1 --scenario classic --n-rounds $A4 $SEED_ARG --save-stats results/apex_a4.json
fi

echo "Done. Metrics: agent_code/apex/runs/metrics.csv"
echo "NEVER run this from a notebook cell — terminal/nohup/tmux only."
