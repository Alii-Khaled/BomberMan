#!/usr/bin/env bash
# E97 plant-gate probe runner: off vs on-subset-off + liveness + determinism.
# Usage: bash scripts/probe_arbiter_plant.sh
set -u
cd "$(dirname "$0")/.."
run() { # mode, k
  ARBITER_PLANT_OPP_MODE="$1" ARBITER_PLANT_OPP_K="$2" \
    python3 scripts/probe_arbiter_plant.py
}
run off 1
run off 1
run 1 1
run 2 1
run 2 2
python3 - <<'PY'
import json, os
def load(m):
    p = 'results/probe_plant_%s.json' % m
    with open(p) as f:
        return json.load(f)
off1, off2 = load('off'), load('off')
on1, on2, on2k2 = load('1'), load('2'), load('2')
fails = []
if off1 != off2:
    fails.append('determinism off!=off')
def as_set(v):
    return set(map(tuple, v)) if isinstance(v, list) else None
veto1 = veto2 = veto2k2 = 0
for k in off1:
    o = as_set(off1[k])
    if o is None:
        continue
    for tag, data, cnt in (('1', on1, 'v1'), ('2', on2, 'v2'), ('2k2', on2k2, 'v3')):
        s = as_set(data.get(k))
        if s is None:
            continue
        if not s.issubset(o):
            fails.append('%s adds permission at %s: %s vs %s' % (tag, k, sorted(s - o), sorted(o)))
        if s != o:
            if tag == '1':
                veto1 += 1
            elif tag == '2':
                veto2 += 1
            else:
                veto2k2 += 1
print('vetoed states: mode1=%d mode2=%d mode2K2=%d' % (veto1, veto2, veto2k2))
if veto1 == 0:
    fails.append('mode1 never vetoes (dead knob)')
if veto2 == 0:
    fails.append('mode2 never vetoes (dead knob)')
if fails:
    print('PLANT PROBE FAILED:')
    for f in fails[:5]:
        print(' -', f)
    raise SystemExit(1)
print('PLANT PROBE OK (subset-off, liveness, determinism)')
PY
