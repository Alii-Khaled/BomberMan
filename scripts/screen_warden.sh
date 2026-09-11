#!/usr/bin/env bash
# Quick paired screen for warden candidates (main.py stats, same seed).
#
# Usage:
#   bash scripts/screen_warden.sh <seed> <rounds> tag:agent:ENV1=..,ENV2=.. [...]
# Example:
#   bash scripts/screen_warden.sh 0 20 v1:warden_v1: roll:warden_v2:
#
# Each config plays the classic 4-agent field (warden + 3x rule_based_agent)
# for <rounds> on <seed>; per-round warden stats are printed and the raw
# JSON lands in results/screen_<tag>_s<seed>.json.
set -u
cd "$(dirname "$0")/.."
KEY_PREFIX="${KEY_PREFIX:-warden}"
SEED="${1:?seed}"; shift
ROUNDS="${1:?rounds}"; shift
for cfg in "$@"; do
  tag="${cfg%%:*}"; rest="${cfg#*:}"
  agent="${rest%%:*}"; envs="${rest#*:}"
  [ "$agent" = "$rest" ] && agent="$tag"
  out="results/screen_${tag}_s${SEED}.json"
  log="logs/screen_${tag}_s${SEED}.log"
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP $out (FORCE=1 to rerun)"
  else
    echo "=== screen $tag agent=$agent env=[$envs] seed=$SEED N=$ROUNDS"
    # shellcheck disable=SC2086
    env $(echo "$envs" | tr ',' ' ') timeout 1800 python3 main.py play \
      --no-gui --agents "$agent" rule_based_agent rule_based_agent \
      rule_based_agent --train 0 --continue-without-training \
      --scenario classic --n-rounds "$ROUNDS" --seed "$SEED" \
      --save-stats "$out" --silence-errors >"$log" 2>&1 \
      || { echo "FAILED $tag (see $log)"; continue; }
  fi
  python3 - "$out" "$tag" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
best = None
for k, v in d['by_agent'].items():
    if k.startswith(KEY_PREFIX):
        best = (k, v); break
k, v = best
r = v.get('rounds', 1) or 1
st = v.get('steps', 1) or 1
print('%-12s %-10s score/rd %5.2f  k %.2f  sui %.2f  coins %.2f  '
      'crates %5.1f  bombs %5.1f  ms/step %5.1f'
      % (sys.argv[2], k, v.get('score', 0) / r, v.get('kills', 0) / r,
         v.get('suicides', 0) / r, v.get('coins', 0) / r,
         v.get('crates', 0) / r, v.get('bombs', 0) / r,
         1000.0 * v.get('time', 0) / st))
PY
done
