#!/bin/bash
# Frozen bake-off for reaper candidates (E37/P4a+P6).
# Usage: bash scripts/bakeoff_reaper.sh <run_dir> [<run_dir> ...]
# Each run_dir must contain checkpoints/best.pt (a sweep job dir or the
# main agent dir). For every candidate: stage an isolated agent copy
# agent_code/_bake_<name>/ with best.pt installed as my-saved-model.pt,
# run the frozen gates (CPU, --train 0), print pooled score/round, clean up.
# Gates (multi-field robustness, sentinel-mix excluded per protocol):
#   G1 3x rb 100x2 | G2 3x random 40x2 | G3 warden-mix 60x2 | G4 3x collector 40x2
# Plus Q0 ablation (REAPER_Q_WEIGHT=0) for the winner only (ML-compliance).
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
G1N=${G1N:-100}; G2N=${G2N:-40}; G3N=${G3N:-60}; G4N=${G4N:-40}

stage() { # run_dir -> bake name
  local rd="$1"
  local name="$2"
  local dest="agent_code/_bake_${name}"
  rm -rf "$dest"
  mkdir -p "$dest"
  cp agent_code/reaper/callbacks.py agent_code/reaper/features.py \
     agent_code/reaper/model.py agent_code/reaper/safety.py \
     agent_code/reaper/train.py agent_code/reaper/requirements.txt "$dest/"
  cp agent_code/reaper/avatar.png agent_code/reaper/bomb.png "$dest/" 2>/dev/null || true
  $PY -c "
import torch, os
ckpt = torch.load('$rd/checkpoints/best.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('q_net', ckpt) if isinstance(ckpt, dict) else ckpt
torch.save(sd, '$dest/my-saved-model.pt')
print('staged $name from $rd (ep=%s)' % (ckpt.get('episode', '?') if isinstance(ckpt, dict) else '?'))
"
  echo "$dest"
}

gate() { # bake_name, tag, n, opponents...
  local b="$1"; shift
  local tag="$1"; shift
  local n="$1"; shift
  for seed in 0 1; do
    $PY main.py play --no-gui --agents "_bake_${b}" "$@" --train 0 \
      --continue-without-training --scenario classic \
      --n-rounds "$n" --seed "$seed" \
      --save-stats "results/bake_${b}_${tag}_s${seed}.json" > /dev/null 2>&1
  done
  $PY -c "
import json
tot = rnd = 0
for s in (0, 1):
    d = json.load(open('results/bake_${b}_${tag}_s%s.json' % s))
    st = d['by_agent']['_bake_${b}']
    tot += st['score']; rnd += st['rounds']
print('  $tag pooled: %.3f' % (tot / max(rnd, 1)))
"
}

if [ "$#" -eq 0 ]; then echo "usage: $0 <run_dir>..."; exit 1; fi
for rd in "$@"; do
  name=$(basename "$rd")
  dest=$(stage "$rd" "$name")
  echo "== $name =="
  gate "$name" rb  "$G1N" rule_based_agent rule_based_agent rule_based_agent
  gate "$name" rn  "$G2N" random_agent random_agent random_agent
  gate "$name" wm  "$G3N" warden_v1 rule_based_agent rule_based_agent
  gate "$name" cl  "$G4N" coin_collector_agent coin_collector_agent coin_collector_agent
  rm -rf "$dest"
done
echo "Done. Winner gets the Q0 ablation + ship decision (Phase 6)."
