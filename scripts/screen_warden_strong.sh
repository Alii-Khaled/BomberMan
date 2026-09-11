#!/usr/bin/env bash
# STRONG-lobby screen for warden candidates (win rate / rank / score).
#
# Usage:
#   bash scripts/screen_warden_strong.sh <seed> <rounds> tag:agent:ENV1=..,ENV2=.. [...]
# Lobby: <warden> arbiter overlord sentinel (the field-proxy STRONG field).
set -u
cd "$(dirname "$0")/.."
SEED="${1:?seed}"; shift
ROUNDS="${1:?rounds}"; shift
for cfg in "$@"; do
  tag="${cfg%%:*}"; rest="${cfg#*:}"
  agent="${rest%%:*}"; envs="${rest#*:}"
  [ "$agent" = "$rest" ] && agent="$tag"
  out="results/strongscreen_${tag}_s${SEED}.json"
  log="logs/strongscreen_${tag}_s${SEED}.log"
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP $out (FORCE=1 to rerun)"
  else
    echo "=== strong screen $tag agent=$agent env=[$envs] seed=$SEED N=$ROUNDS"
    # shellcheck disable=SC2086
    env $(echo "$envs" | tr ',' ' ') timeout 3600 python3 scripts/tournament_eval.py \
      --agents "$agent" arbiter overlord sentinel \
      --n-rounds "$ROUNDS" --seed "$SEED" --scenario classic \
      --match-name "strongscreen_${tag}_s${SEED}" --out "$out" \
      >"$log" 2>&1 || { echo "FAILED $tag (see $log)"; continue; }
  fi
  python3 - "$out" "$tag" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
w = sys.argv[2]
for n, v in sorted(d['summary'].items()):
    if n.startswith('warden'):
        print('%-10s %-10s score %.3f  win %.3f  rank %.2f  kills %.2f  sui %d'
              % (w, n, v['score_mean'], v['win_rate'], v['mean_rank'],
                 v['kills_total'] / d['n_rounds'], v['suicides_total']))
PY
done
