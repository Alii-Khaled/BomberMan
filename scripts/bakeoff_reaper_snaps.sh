#!/bin/bash
# Snapshot bake-off screen for reaper sweep arms (E36 follow-up).
# best.pt = EMA argmax, which repeatedly failed to equal frozen-best
# (E13/E30) — so screen C1/C2/C3/C4 snapshots + last at 40 rounds seed 0
# (paired arenas), then validate top-2 at 100x2 via bakeoff_reaper.sh.
# Usage: bash scripts/bakeoff_reaper_snaps.sh
#   (arms + snapshots hardcoded below; prints pooled rb score per snap)
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export REAPER_DEVICE=cpu

screen() { # arm, snapfile, tag
  local arm="$1" snap="$2" tag="$3"
  local dest="agent_code/_snap_${tag}"
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
"
  $PY main.py play --no-gui --agents "_snap_${tag}" rule_based_agent rule_based_agent rule_based_agent \
    --train 0 --continue-without-training --scenario classic \
    --n-rounds 40 --seed 0 --save-stats "results/snap_${tag}_s0.json" > /dev/null 2>&1
  $PY -c "
import json
d = json.load(open('results/snap_${tag}_s0.json'))
st = d['by_agent']['_snap_${tag}']
r = st['rounds']
print('$tag: score %.3f coins %.2f kills %.3f sui %.3f' % (
    st['score']/r, st.get('coins',0)/r, st.get('kills',0)/r, st.get('suicides',0)/r))
"
  rm -rf "$dest"
}

for arm in sw01_base sw04_lowheur sw07_longhor sw08_noshaping; do
  for snap in ep_000200.pt ep_000400.pt ep_000600.pt ep_000800.pt ep_001000.pt last.pt; do
    screen "$arm" "$snap" "${arm}_${snap%.pt}"
  done
done
echo Done.
