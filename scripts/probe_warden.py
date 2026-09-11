#!/usr/bin/env python3
"""Warden v2 probe: static gates before any game is played.

  G0  self-contained import (numpy only, no torch) + callbacks API.
  G1  sim.blast_coords == items.Bomb.get_blast_coords (exact, N trials).
  G2  sim.step full-state parity vs the engine's own step functions
      (no hidden coins), self-first seating.
  G3  chosen action is spec-valid on N random states.
  G4  safety.valid_mask parity with the spec for all 6 actions.
  G5  determinism: identical state + seed -> identical action.
  G6  latency: p50/p99 per act() call under the tournament budget.
  G7  warden_v2 package does not import torch (self-containment).

Usage: python3 scripts/probe_warden.py [--trials N]
Exit nonzero on any failure.
"""
import logging
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('WARDEN_SEED', '12345')

import numpy as np

import warden_v2.sim as SIM
import warden_v2.safety as SAFE
import warden_v2.callbacks as WC
from items import Bomb, Coin, Explosion
from environment import BombeRLeWorld
import settings as s

N = 300
for a in sys.argv[1:]:
    if a.startswith('--trials'):
        N = int(a.split('=')[1] if '=' in a else sys.argv[sys.argv.index(a) + 1])

fails = []
rng = np.random.default_rng(7)


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


_logger = logging.getLogger('probe_warden')
_logger.addHandler(logging.NullHandler())


class FakeAgent:
    def __init__(self, name, x, y, score=0, bombs_left=True):
        self.name = name
        self.x, self.y = int(x), int(y)
        self.dead = False
        self.score = int(score)
        self.total_score = int(score)
        self.bombs_left = bool(bombs_left)
        self.bomb_sprite = None
        self.avatar = None
        self.events = []
        self.trophies = []

    def add_event(self, e):
        self.events.append(e)

    def update_score(self, d):
        self.score += d
        self.total_score += d


class FakeWorld:
    tile_is_free = BombeRLeWorld.tile_is_free


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


def build_world(n_agents=4, n_bombs=2, n_exp=1, n_coins=3):
    arena = make_arena()
    spots = free_tiles(arena)
    rng.shuffle(spots)
    fagents = []
    for i in range(n_agents):
        (x, y) = spots.pop()
        fagents.append(FakeAgent('a%d' % i, x, y,
                                 score=int(rng.integers(0, 6)),
                                 bombs_left=bool(rng.random() < 0.7)))
    occ = [(a.x, a.y) for a in fagents]
    bombs = []
    for _ in range(n_bombs):
        cands = [t for t in free_tiles(arena, occ) if t not in occ]
        if not cands:
            break
        (x, y) = cands[int(rng.integers(len(cands)))]
        owner = fagents[int(rng.integers(len(fagents)))]
        bombs.append(Bomb((x, y), owner, int(rng.integers(0, 5)), 3, None))
        occ.append((x, y))
    coins = []
    for _ in range(n_coins):
        cands = [t for t in free_tiles(arena, occ)]
        if not cands:
            break
        (x, y) = cands[int(rng.integers(len(cands)))]
        coins.append(Coin((x, y), collectable=True))
    explosions = []
    for _ in range(n_exp):
        cands = free_tiles(arena, occ)
        if not cands:
            break
        (x, y) = cands[int(rng.integers(len(cands)))]
        owner = fagents[int(rng.integers(len(fagents)))]
        e = Explosion([(x, y)], [], owner, s.EXPLOSION_TIMER)
        if rng.random() < 0.5:
            e.timer = 1
        explosions.append(e)
    fw = FakeWorld()
    fw.arena = arena.copy()
    fw.bombs = bombs
    fw.coins = coins
    fw.active_agents = list(fagents)
    fw.agents = list(fagents)
    fw.explosions = explosions
    fw.step = int(rng.integers(0, 390))
    fw.running = True
    fw.logger = _logger
    st = {'arena': arena.copy(),
          'coins': [[c.x, c.y, int(c.collectable)] for c in coins],
          'agents': [{'x': a.x, 'y': a.y, 'alive': not a.dead,
                      'score': a.score, 'bombs_left': a.bombs_left,
                      'is_self': i == 0} for i, a in enumerate(fagents)],
          'bombs': [[b.x, b.y, b.timer,
                     fagents.index(b.owner)] for b in bombs],
          'explosions': [{'coords': list(e.blast_coords), 'timer': e.timer,
                          'stage': e.stage, 'owner': fagents.index(e.owner)}
                         for e in explosions],
          'step': fw.step, 'total_coins': 9}
    acts = []
    for a in fagents:
        opts = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
        if a.bombs_left:
            opts.append('BOMB')
        acts.append(opts[int(rng.integers(len(opts)))])
    return fw, fagents, st, acts


def engine_step(fw, fagents, acts):
    for a, act in zip(list(fw.active_agents), acts):
        BombeRLeWorld.perform_agent_action(fw, a, act)
    BombeRLeWorld.collect_coins(fw)
    BombeRLeWorld.update_explosions(fw)
    BombeRLeWorld.update_bombs(fw)
    BombeRLeWorld.evaluate_explosions(fw)
    fw.step += 1


def snapshot_world(fw, fagents):
    return {
        'arena': fw.arena.copy(),
        'coins': sorted([(c.x, c.y, bool(c.collectable)) for c in fw.coins]),
        'scores': [a.score for a in fagents],
        'alive': [not a.dead for a in fagents],
        'bl': [bool(a.bombs_left) for a in fagents],
        'apos': [(a.x, a.y) for a in fagents],
        'bombs': sorted([(b.x, b.y, b.timer, fagents.index(b.owner))
                         for b in fw.bombs]),
        'exp': sorted([(tuple(sorted(e.blast_coords)), e.stage, e.timer,
                        fagents.index(e.owner)) for e in fw.explosions]),
    }


def snapshot_sim(st):
    return {
        'arena': np.asarray(st['arena']).copy(),
        'coins': sorted([(c[0], c[1], bool(c[2])) for c in st['coins']]),
        'scores': [a['score'] for a in st['agents']],
        'alive': [bool(a['alive']) for a in st['agents']],
        'bl': [bool(a['bombs_left']) for a in st['agents']],
        'apos': [(a['x'], a['y']) for a in st['agents']],
        'bombs': sorted([(b[0], b[1], b[2], b[3]) for b in st['bombs']]),
        'exp': sorted([(tuple(sorted(e['coords'])), e['stage'], e['timer'],
                        e['owner']) for e in st['explosions']]),
    }


def cmp_snap(w, s):
    diffs = []
    if not np.array_equal(w['arena'], s['arena']):
        diffs.append('arena:%d' % int((w['arena'] != s['arena']).sum()))
    for k in ('coins', 'scores', 'alive', 'bl', 'apos', 'bombs', 'exp'):
        if w[k] != s[k]:
            diffs.append(k)
    return diffs


class Obj:
    pass


# ---- G0: imports / API ----
have_api = all(hasattr(WC, fn) for fn in ('setup', 'act'))
check('G0 callbacks API (setup/act)', have_api)
check('G0 no torch imported by warden_v2',
      not any(m == 'torch' or m.startswith('torch.') for m in sys.modules))


# ---- G1: blast exactness ----
ok = True
for _ in range(N * 5):
    arena = make_arena()
    xs, ys = np.where(arena != -1)
    i = int(rng.integers(len(xs)))
    x, y = int(xs[i]), int(ys[i])
    if set(SIM.blast_coords(arena, x, y)) != set(
            Bomb((x, y), None, 4, 3, None).get_blast_coords(arena)):
        ok = False
        break
check('G1 blast == engine (%d)' % (N * 5), ok)


# ---- G2: full-step parity (no hidden coins) ----
bad = 0
trials = 0
for _ in range(N // 2):
    fw, fagents, st, acts = build_world()
    engine_step(fw, fagents, acts)
    SIM.step(st, acts)
    trials += 1
    if cmp_snap(snapshot_world(fw, fagents), snapshot_sim(st)):
        bad += 1
check('G2 full-step parity (no hidden coins)', bad == 0,
      '%d/%d diverged' % (bad, trials))


# ---- G3: chosen action spec-valid ----
_DELTAS = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}


def spec_valid(gs):
    arena = np.asarray(gs['field'])
    _, _, bl, (x, y) = gs['self']
    bomb_cells = set((int(b[0][0]), int(b[0][1])) for b in gs['bombs'])
    other_cells = set((int(o[3][0]), int(o[3][1])) for o in gs['others'])
    out = {}
    for a, (dx, dy) in _DELTAS.items():
        nx, ny = x + dx, y + dy
        out[a] = (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]
                  and arena[nx, ny] == 0 and (nx, ny) not in bomb_cells
                  and (nx, ny) not in other_cells)
    out['WAIT'] = True
    out['BOMB'] = bool(bl)
    return out


bad_actions = []
n_bombs = 0
for _ in range(N):
    _, _, st, _ = build_world()
    gs = SIM.to_game_state(st)
    o = Obj()
    WC.setup(o)
    a = WC.act(o, gs)
    if a == 'BOMB':
        n_bombs += 1
    if not spec_valid(gs).get(a, False):
        bad_actions.append((a, gs['self'][3]))
check('G3 chosen action spec-valid (%d states, %d bombs)'
      % (N, n_bombs), not bad_actions, str(bad_actions[:3]))


# ---- G4: valid_mask parity with spec ----
bad_mask = []
for _ in range(N):
    _, _, st, _ = build_world()
    gs = SIM.to_game_state(st)
    arena = np.asarray(gs['field'])
    _, _, bl, (x, y) = gs['self']
    bombs = gs['bombs']
    others = [(int(o[3][0]), int(o[3][1])) for o in gs['others']]
    expm = gs['explosion_map']
    mask = SAFE.valid_mask(arena, (x, y), bl, bombs, others, expm)
    spec = spec_valid(gs)
    for a in spec:
        if bool(mask.get(a)) != bool(spec[a]):
            bad_mask.append((a, bool(mask.get(a)), bool(spec[a])))
check('G4 valid_mask == spec (all actions)', not bad_mask,
      str(bad_mask[:3]))


# ---- G5: determinism ----
det_bad = 0
for _ in range(max(20, N // 10)):
    _, _, st, _ = build_world()
    gs = SIM.to_game_state(st)
    o1, o2 = Obj(), Obj()
    WC.setup(o1)
    WC.setup(o2)
    if WC.act(o1, gs) != WC.act(o2, gs):
        det_bad += 1
check('G5 determinism (same state+seed)', det_bad == 0,
      '%d mismatches' % det_bad)


# ---- G6: latency ----
lat = []
for _ in range(N // 3):
    _, _, st, _ = build_world()
    gs = SIM.to_game_state(st)
    o = Obj()
    WC.setup(o)
    t0 = time.perf_counter()
    WC.act(o, gs)
    lat.append((time.perf_counter() - t0) * 1000.0)
lat = np.asarray(lat)
p50, p99 = float(np.percentile(lat, 50)), float(np.percentile(lat, 99))
print('      latency ms: p50 %.2f p99 %.2f max %.2f' % (p50, p99, lat.max()))
check('G6 latency p99 < 500 ms', p99 < 500.0)

print()
if fails:
    print('WARDEN PROBE FAILED:', ', '.join(fails))
    sys.exit(1)
print('WARDEN PROBE OK (%d gates)' % 7)
