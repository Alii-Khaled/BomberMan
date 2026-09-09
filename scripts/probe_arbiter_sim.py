#!/usr/bin/env python3
"""ARBITER P1 parity probe: sim.step vs the engine's OWN step functions.

Method (stronger than replay-matching): build fake micro-worlds, run
environment.BombeRLeWorld's unbound step functions on them AND
arbiter.sim.step on the equivalent SimState, compare everything.
  G1 blast_coords == items.Bomb.get_blast_coords (2000 trials, EXACT).
  G2 full-step parity, no hidden coins (300 trials): arena, coins,
      scores, alive, bombs_left, bombs, explosions. Engine order forced
      self-first so A1 (movement order) cannot trigger.
  G2b A1 characterization: self-last seating + contested tile ->
      divergence must be confined to the contested outcome.
  G3 hidden coins (200 trials): everything exact EXCEPT reveals, where
      sim books expectation (approximation A2) -> mean |exp-act| small.
  G4 from/to_game_state round-trip preserves observables (demo states).
  G5 latency: sim.step mean ms (P1 budget: thousands of steps/search).
Usage: python3 scripts/probe_arbiter_sim.py [--trials N]
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

import numpy as np

import arbiter.sim as SIM
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


_logger = logging.getLogger('probe_sim')
_logger.addHandler(logging.NullHandler())

try:
    from fallbacks import pygame as _pg
    _SURF = _pg.Surface((30, 30))
except Exception:
    _SURF = None


class FakeAgent:
    def __init__(self, name, x, y, score=0, bombs_left=True):
        self.name = name
        self.x, self.y = int(x), int(y)
        self.dead = False
        self.score = int(score)
        self.total_score = int(score)
        self.bombs_left = bool(bombs_left)
        self.bomb_sprite = None
        self.avatar = _SURF
        self.events = []
        self.trophies = []

    def add_event(self, e):
        self.events.append(e)

    def update_score(self, d):
        self.score += d
        self.total_score += d


class FakeWorld:
    # tile_is_free only touches .arena/.bombs/.active_agents — reuse the
    # engine's own method unbound so movement blocking is engine-exact.
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


def build_world(n_agents=4, n_bombs=2, n_exp=1, n_coins=3, hidden=False):
    """Fake world + equivalent SimState. Self is agents[0] (G2)."""
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
        coll = True
        if hidden and rng.random() < 0.5:
            arena[x, y] = 1  # hide under a crate
            coll = False
        coins.append(Coin((x, y), collectable=coll))
    explosions = []
    for _ in range(n_exp):
        cands = free_tiles(arena, occ)
        if not cands:
            break
        (x, y) = cands[int(rng.integers(len(cands)))]
        owner = fagents[int(rng.integers(len(fagents)))]
        e = Explosion([(x, y)], [], owner, s.EXPLOSION_TIMER)
        if rng.random() < 0.5:
            e.timer = 1  # aged one step (stage 0, 1 lethal step left)
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
    # equivalent SimState (agent 0 = self)
    st = {'arena': arena.copy(),
          'coins': [[c.x, c.y, int(c.collectable)] for c in coins],
          'agents': [{'x': a.x, 'y': a.y, 'alive': not a.dead,
                       'score': a.score, 'bombs_left': a.bombs_left,
                       'is_self': i == 0} for i, a in enumerate(fagents)],
          'bombs': [[b.x, b.y, b.timer,
                     fagents.index(b.owner)] for b in bombs],
          'explosions': [{'coords': list(e.blast_coords), 'timer': e.timer,
                           'stage': e.stage,
                           'owner': fagents.index(e.owner)}
                          for e in explosions],
          'step': fw.step, 'total_coins': 9}
    acts = []
    for i, a in enumerate(fagents):
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


# ---- G1: blast exactness ----
ok = True
for _ in range(2000):
    arena = make_arena()
    xs, ys = np.where(arena != -1)
    i = int(rng.integers(len(xs)))
    x, y = int(xs[i]), int(ys[i])
    if set(SIM.blast_coords(arena, x, y)) != set(
            Bomb((x, y), None, 4, 3, None).get_blast_coords(arena)):
        ok = False
        break
check('G1 blast == engine (2000)', ok)

# ---- G2: full-step parity, no hidden coins ----
bad = 0
trials = 0
for _ in range(N):
    fw, fagents, st, acts = build_world(
        n_agents=int(rng.integers(1, 5)),
        n_bombs=int(rng.integers(0, 4)),
        n_exp=int(rng.integers(0, 3)),
        n_coins=int(rng.integers(0, 6)), hidden=False)
    engine_step(fw, fagents, acts)
    SIM.step(st, acts)
    trials += 1
    d = cmp_snap(snapshot_world(fw, fagents), snapshot_sim(st))
    if d:
        bad += 1
        if bad <= 3:
            print('   trial diff:', d)
check('G2 full-step parity (no hidden)', bad == 0,
      '%d/%d mismatch' % (bad, trials))

# ---- G2b: A1 characterization (self-last + contested tile) ----
confined = True
seen_contest = 0
for _ in range(120):
    fw, fagents, st, acts = build_world(n_agents=2, n_bombs=0, n_exp=0,
                                        n_coins=0, hidden=False)
    # force adjacency: teleport a0 next to a1 on free tiles sharing a
    # common free neighbour (the contested tile)
    a0, a1 = fagents[0], fagents[1]
    # simpler robust construction: pick any free tile T with >=2 free
    # neighbours; put a0, a1 on two distinct neighbours of T
    tgt = None
    free = [(x, y) for x in range(17) for y in range(17)
            if fw.arena[x, y] == 0]
    rng.shuffle(free)
    for (tx, ty) in free:
        nbs = [(tx + dx, ty + dy) for dx, dy in
               ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nbs = [(x, y) for (x, y) in nbs
               if 0 <= x < 17 and 0 <= y < 17 and fw.arena[x, y] == 0]
        if len(nbs) >= 2:
            tgt = (tx, ty)
            a0.x, a0.y = nbs[0]
            a1.x, a1.y = nbs[1]
            st['agents'][0]['x'], st['agents'][0]['y'] = nbs[0]
            st['agents'][1]['x'], st['agents'][1]['y'] = nbs[1]
            break
    if tgt is None:
        continue
    seen_contest += 1
    a0s, a1s = (a0.x, a0.y), (a1.x, a1.y)
    # seating self-last: engine applies opp first
    fw.active_agents = [a1, a0]
    acts_ord = [acts[1], acts[0]]
    _dd = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
    mv0 = _dd.get((tgt[0] - a0.x, tgt[1] - a0.y), 'WAIT')
    mv1 = _dd.get((tgt[0] - a1.x, tgt[1] - a1.y), 'WAIT')
    engine_step(fw, fagents, [mv1, mv0])
    SIM.step(st, [mv0, mv1])  # sim: self-first by design (A1)
    w, q = snapshot_world(fw, fagents), snapshot_sim(st)
    # positions may differ ONLY by who won the contested tile
    rest_ok = (np.array_equal(w['arena'], q['arena'])
               and w['coins'] == q['coins'] and w['scores'] == q['scores']
               and w['alive'] == q['alive'] and w['bl'] == q['bl']
               and w['bombs'] == q['bombs'] and w['exp'] == q['exp'])
    if list(w['apos']) == list(q['apos']):
        pos_ok = True
    else:
        sw, ss = set(w['apos']), set(q['apos'])
        pos_ok = (tgt in sw and tgt in ss
                  and (sw ^ ss) <= {a0s, a1s, tgt})
    if not (rest_ok and pos_ok):
        confined = False
        break
check('G2b A1 confined to contested tile', confined and seen_contest > 20,
      '%d contests' % seen_contest)

# ---- G3: hidden coins (expectation, not exactness) ----
bad = 0
errs = []
for _ in range(200):
    fw, fagents, st, acts = build_world(
        n_agents=int(rng.integers(1, 5)),
        n_bombs=int(rng.integers(1, 4)),
        n_coins=int(rng.integers(2, 6)), hidden=True)
    engine_step(fw, fagents, acts)
    info = SIM.step(st, acts)
    w, q = snapshot_world(fw, fagents), snapshot_sim(st)
    # coins collectable may differ (hidden reveals); compare rest exactly
    rest = [k for k in ('scores', 'alive', 'bl', 'apos', 'bombs', 'exp')
            if w[k] != q[k]]
    if rest or not np.array_equal(w['arena'], q['arena']):
        bad += 1
    actual = sum(1 for c in w['coins'] if c[2]) - sum(
        1 for c in q['coins'] if c[2])
    # actual reveals vs sim expectation: sim books exp into info only
    errs.append(abs(info['revealed_exp'] - max(0, actual)))
check('G3 hidden: rest exact', bad == 0, '%d/200 mismatch' % bad)
check('G3 reveal expectation sane', float(np.mean(errs)) < 0.75,
      'mean|exp-act|=%.3f' % float(np.mean(errs)))

# ---- G4: round-trip on demo states ----
import glob
ok = True
for f in sorted(glob.glob(os.path.join(
        REPO, 'results/apex_demos/*/*.npz')))[:3]:
    d = np.load(f)
    img = d['img'].astype(np.float32) / 4.0
    sc = d['sc']
    for i in (0, len(img) // 2, len(img) - 1):
        t = img[i]
        arena = np.where(t[0] > 0.5, -1,
                         np.where(t[1] > 0.5, 1, 0)).astype(int)
        gs = {'round': 1, 'step': 5, 'field': arena,
              'bombs': [((int(x), int(y)),
                         max(0, int(round(t[5][x, y] * 4))))
                        for x, y in zip(*np.where(t[5] > 0))],
              'explosion_map': np.where(t[7] > 0.5, 1, 0),
              'coins': [(int(x), int(y))
                        for x, y in zip(*np.where(t[2] > 0.5))],
              'self': ('w', 0, True, (1, 1)),
              'others': []}
        st = SIM.from_game_state(gs)
        gs2 = SIM.to_game_state(st)
        if not np.array_equal(gs2['field'], arena):
            ok = False
        if sorted(gs2['coins']) != sorted(gs['coins']):
            ok = False
check('G4 round-trip observables', ok)

# ---- G5: latency ----
fw, fagents, st, acts = build_world()
t0 = time.perf_counter()
for _ in range(300):
    import copy
    SIM.step(copy.deepcopy(st), acts)
ms = (time.perf_counter() - t0) / 300 * 1000
print('G5 sim.step %.3f ms' % ms)
check('G5 step < 0.5ms', ms < 0.5, '%.3f' % ms)

print()
if fails:
    print('FAILED:', fails)
    sys.exit(1)
print('ALL SIM PROBE GROUPS PASS')
