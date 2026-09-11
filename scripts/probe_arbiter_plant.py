#!/usr/bin/env python3
"""E97 Phase 3 probe: opponent-aware plant gate (ARBITER_PLANT_OPP).

Runs in a MODE-specific process (the knob is read at search import):
  ARBITER_PLANT_OPP_MODE=off|1|2 [ARBITER_PLANT_OPP_K=1..3]
Dumps the admitted bomb plans (search.gen_plans) for a fixed battery of
random + hand states to results/probe_plant_<MODE>.json.

Compare with scripts/probe_arbiter_plant.sh: on-subset-off
(never adds permission), liveness (each mode vetoes >=1 state), and
determinism (off == off).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from items import Bomb, Coin, Explosion
import settings as s

MODE = os.environ.get('ARBITER_PLANT_OPP_MODE', 'off').strip()
if MODE != 'off':
    os.environ['ARBITER_PLANT_OPP'] = MODE
OUT = os.path.join(REPO, 'results', 'probe_plant_%s.json' % MODE)

import arbiter.safety as AS          # noqa: E402
import arbiter.search as SR          # noqa: E402
import arbiter.sim as SIM            # noqa: E402

rng = np.random.default_rng(11)


def make_arena():
    a = np.zeros((17, 17), dtype=int)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = -1
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                a[x, y] = -1
    free = list(zip(*np.where(a == 0)))
    for (x, y) in free:
        if rng.random() < 0.75:
            a[x, y] = 1
    for (x, y) in [(1, 1), (1, 15), (15, 1), (15, 15)]:
        for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1),
                         (x, y + 1)]:
            if a[xx, yy] == 1:
                a[xx, yy] = 0
    return a


def free_tiles(arena, avoid=()):
    av = set(avoid)
    return [(x, y) for x in range(17) for y in range(17)
            if arena[x, y] == 0 and (x, y) not in av]


def build_state(n_agents=4, n_bombs=2, n_exp=0, n_coins=3):
    arena = make_arena()
    spots = free_tiles(arena)
    rng.shuffle(spots)
    ag = []
    for i in range(n_agents):
        (x, y) = spots.pop()
        ag.append({'x': x, 'y': y, 'alive': True,
                   'score': int(rng.integers(0, 6)),
                   'bombs_left': bool(rng.random() < 0.7), 'is_self': i == 0})
    occ = [(a['x'], a['y']) for a in ag]
    bombs = []
    for _ in range(n_bombs):
        cands = [t for t in free_tiles(arena, occ)]
        if not cands:
            break
        (x, y) = cands[int(rng.integers(len(cands)))]
        bombs.append([x, y, int(rng.integers(0, 5)), -1])
        occ.append((x, y))
    coins = []
    for _ in range(n_coins):
        cands = free_tiles(arena, occ)
        if not cands:
            break
        (x, y) = cands[int(rng.integers(len(cands)))]
        coins.append([x, y, 1])
    explosions = []
    for _ in range(n_exp):
        cands = free_tiles(arena, occ)
        if not cands:
            break
        (x, y) = cands[int(rng.integers(len(cands)))]
        explosions.append({'coords': [(x, y)], 'timer': 2, 'stage': 0,
                           'owner': -1})
    st = {'arena': arena, 'coins': coins, 'agents': ag, 'bombs': bombs,
          'explosions': explosions, 'step': int(rng.integers(0, 390)),
          'total_coins': 9}
    return SIM.to_game_state(st)


out = {}
for i in range(150):
    gs = build_state()
    try:
        safety = AS.action_safety(gs)
        plans = SR.gen_plans(gs, safety)
        tiles = sorted(set(tuple(p['bomb_at']) for p in plans
                           if p.get('bomb_at')))
        out['s%d' % i] = tiles
    except Exception as ex:
        out['s%d' % i] = {'error': str(ex)[:80]}

with open(OUT, 'w') as fh:
    json.dump(out, fh)
n_bomb = sum(1 for v in out.values() if isinstance(v, list) and v)
print('mode=%s dumped %d states (%d with >=1 bomb plan) -> %s'
      % (MODE, len(out), n_bomb, OUT))
