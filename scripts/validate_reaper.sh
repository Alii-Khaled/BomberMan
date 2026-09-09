#!/bin/bash
# Validate snapshot-screen winners at 100x2 (E36). Usage:
#   bash scripts/validate_reaper.sh "arm snap" [...]
# e.g. bash scripts/validate_reaper.sh "sw01_base ep_000400.pt" "sw07_longhor ep_000400.pt"
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export REAPER_DEVICE=cpu
for spec in "$@"; do
  arm="${spec%% *}"; snap="${spec##* }"
  tag="${arm}_${snap%.pt}"
  dest="agent_code/_val_${tag}"
  rm -rf "$dest"
  mkdir -p "$dest"
  cp agent_code/reaper/callbacks.py agent_code/reaper/features.py \
     agent_code/reaper/model.py agent_code/reaper/safety.py \
     agent_code/reaper/train.py agent_code/reaper/requirements.txt "$dest/"
  cp agent_code/reaper/avatar.png agent_code/reaper/bomb.png "$dest/" 2>/dev/null || true
  $PY -c "
import torch
ckpt = torch.load('results/sweeps/$arm/checkpoints/$snap', map_location='cpu', weights_only=False)
sd = ckpt.get('q_net', ckpt) if isinstance(ckpt, dict) else ckpt
torch.save(sd, '$dest/my-saved-model.pt')
print('validating $tag (ep=%s)' % (ckpt.get('episode', '?') if isinstance(ckpt, dict) else '?'))
"
  for seed in 0 1; do
    $PY main.py play --no-gui --agents "_val_${tag}" rule_based_agent rule_based_agent rule_based_agent \
      --train 0 --continue-without-training --scenario classic \
      --n-rounds 100 --seed "$seed" --save-stats "results/val_${tag}_s${seed}.json" > /dev/null 2>&1
  done
  $PY -c "
import json
tot = {k: 0 for k in ('score','rounds','coins','kills','suicides','crates','bombs')}
for s in (0, 1):
    d = json.load(open('results/val_${tag}_s%s.json' % s))
    st = d['by_agent']['_val_${tag}']
    for k in tot: tot[k] += st.get(k, 0)
r = tot['rounds']
print('$tag pooled: score %.3f coins %.2f kills %.3f sui %.3f crates %.1f (s0=%.2f s1=%.2f)' % (
    tot['score']/r, tot['coins']/r, tot['kills']/r, tot['suicides']/r, tot['crates']/r,
    json.load(open('results/val_${tag}_s0.json'))['by_agent']['_val_${tag}']['score']/100,
    json.load(open('results/val_${tag}_s1.json'))['by_agent']['_val_${tag}']['score']/100))
"
  rm -rf "$dest"
done
echo Done.
