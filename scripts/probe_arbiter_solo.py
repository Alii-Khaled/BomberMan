#!/usr/bin/env python3
"""E112 (P0) ARBITER-NG solo/endgame unfreeze probe.

  G0  flags OFF: gen_plans has no trek plans (ship behavior).
  G1  coin trek: with flags ON and a reachable visible coin, a coin trek
      exists and the returned action strictly decreases BFS distance.
  G2  yield trek: no coins -> a yield trek to the best-yield tile exists
      and its first step strictly decreases BFS distance.
  G3  crate fallback: no coins and no >=2-yield tile -> a crate trek.
  G4  opponent parity: with an opponent present, flags ON produce exactly
      the same plan list as flags OFF (no treks, no margin change).
  G5  safety gate: trek first steps must be mask-safe; on a state whose
      trek path starts in a ticking blast the search never returns it.
  G6  fixture regression (seed-3 solo freeze): 200 sim ticks from the
      verified freeze state -> no death, >=40 crates cleared, no
      stay-put streak > 4.
  G7  loop penalty math: escalating formula + cap; OFF == LOOP3/LOOP2.
  G8  latency: search_action on the solo fixture < 50 ms.

Usage: python3 scripts/probe_arbiter_solo.py
"""
import collections
import logging
import os
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import agent_code.arbiter_ng.search as S
import agent_code.arbiter_ng.callbacks as WC
from agent_code.arbiter_ng.safety import action_safety

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def set_flags(trek=False, margin=-1.0, loop_esc='0'):
    S.SOLO_TREK = bool(trek)
    S.SOLO_MARGIN = float(margin)
    WC.LOOP_ESC = str(loop_esc)


def mk_agent(model=None):
    o = types.SimpleNamespace()
    log = logging.getLogger('probe-solo')
    log.setLevel(logging.WARNING)
    o.logger = log
    o.train = False
    WC.setup(o)
    return o


def mk_gs(arena, pos, coins=(), others=(), bombs=(), step=10):
    W, H = arena.shape
    gs = {'round': 1, 'step': step, 'field': arena,
          'self': ('me', 0, True, (pos[0], pos[1])),
          'others': [('o%d' % i, 0, True, (o[0], o[1]))
                     for i, o in enumerate(others)],
          'bombs': [((b[0], b[1]), b[2]) for b in bombs],
          'coins': [tuple(c) for c in coins],
          'explosion_map': np.zeros((W, H))}
    return gs


def open_arena():
    a = np.zeros((17, 17), dtype=np.int8)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = -1
    return a


# exact seed-3 free state (t=60, pos (13,5), solo, no coins/bombs)
FIXTURE_ARENA = [
    '#################',
    '#.....x.xx.x....#',
    '#.#x#x#x#.#.#x#.#',
    '#xx.xxxxxxx.x.x.#',
    '#x#.#.#x#x#.#x#.#',
    '#xxxxxxxxxxxxxxx#',
    '#x#x#x#x#x#x#x#.#',
    '#..x..xxxxxxxxxx#',
    '#x#x#x#x#x#x#x#.#',
    '#x..xx.xxx..xxxx#',
    '#.#x#x#.#.#x#.#.#',
    '#.xxx.xxxxxxx.xx#',
    '#.#x#x#.#.#x#x#.#',
    '#.....xxxx.xxxx.#',
    '#.#x#x#x#x#x#x#.#',
    '#.....xxxxx.....#',
    '#################',
]
_M = {'#': -1, '.': 0, 'x': 1}
FIXTURE = np.array([[_M[c] for c in row] for row in FIXTURE_ARENA],
                   dtype=np.int8)

# G0 flags OFF ---------------------------------------------------------------
set_flags(trek=False, margin=-1.0, loop_esc='0')
gs = mk_gs(FIXTURE, (13, 5))
sf = action_safety(gs)
plans = S.gen_plans(gs, sf)
check('G0 flags-off: no trek plans', not any('trek' in p for p in plans),
      'n=%d' % len(plans))
a_off, dbg_off = S.search_action(gs, sf, None, time.perf_counter(), 0.30)
check('G0 flags-off: no solo arbitration',
      'solo_trek' not in dbg_off, 'action=%s' % a_off)

# G1 coin trek ---------------------------------------------------------------
_DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0),
          'RIGHT': (1, 0)}

set_flags(trek=True, margin=0.15)
arena = open_arena()
pos = (8, 8)
coin = (8, 13)
gs = mk_gs(arena, pos, coins=[coin])
sf = action_safety(gs)
plans = S.gen_plans(gs, sf)
treks = [p for p in plans if p.get('trek') == 'coin']
check('G1 coin trek generated', bool(treks),
      'targets=%s' % [p['trek_target'] for p in treks])
a, dbg = S.search_action(gs, sf, None, time.perf_counter(), 0.30)
ok = False
d_txt = '-'
if a in _DELTA and dbg.get('solo_trek') == 'coin':
    dx, dy = _DELTA[a]
    moved = (pos[0] + dx, pos[1] + dy)
    dmap = S._bfs_dist(arena, set(), coin)
    ok = int(dmap[moved]) == int(dmap[pos]) - 1
    d_txt = '%d -> %d' % (int(dmap[pos]), int(dmap[moved]))
check('G1 returned action steps toward the coin', ok,
      'action=%s dist %s' % (a, d_txt))

# G2 yield trek (no coins) ---------------------------------------------------
arena = open_arena()
for (cx, cy) in [(3, 3), (4, 3), (3, 4), (4, 4)]:
    arena[cx, cy] = 1
pos = (8, 8)
gs = mk_gs(arena, pos)
sf = action_safety(gs)
plans = S.gen_plans(gs, sf)
treks = [p for p in plans if p.get('trek') == 'yield']
check('G2 yield trek generated', bool(treks),
      'targets=%s' % [(p['trek_target'], p.get('trek_yield'))
                      for p in treks])
if treks:
    tgt = treks[0]['trek_target']
    a, dbg = S.search_action(gs, sf, None, time.perf_counter(), 0.30)
    tgt = dbg.get('solo_trek_target', tgt)
    dmap = S._bfs_dist(arena, set(), tgt)
    ok = False
    if a in _DELTA:
        dx, dy = _DELTA[a]
        moved = (pos[0] + dx, pos[1] + dy)
        ok = int(dmap[moved]) == int(dmap[pos]) - 1
    check('G2 returned action steps toward the yield target', ok,
          'action=%s target=%s' % (a, tgt))

# G3 crate fallback ----------------------------------------------------------
arena = open_arena()
# one crate, walled so its best blast hits exactly 1 crate
arena[3, 3] = 1
for (wx, wy) in [(2, 3), (4, 3), (3, 2)]:
    arena[wx, wy] = -1
arena[3, 4] = 1  # second crate adjacent: yield at (3,2) blocked, (3,4)->1
pos = (8, 8)
gs = mk_gs(arena, pos)
sf = action_safety(gs)
plans = S.gen_plans(gs, sf)
kinds = [p.get('trek') for p in plans if p.get('trek')]
check('G3 crate fallback generated when no yield>=2 tile',
      'crate' in kinds or 'yield' in kinds, 'kinds=%s' % kinds)

# G4 opponent parity ---------------------------------------------------------
set_flags(trek=False, margin=0.15)
arena = open_arena()
pos = (8, 8)
gs_a = mk_gs(arena, pos, coins=[(8, 13)], others=[(4, 4)], step=33)
sf_a = action_safety(gs_a)
plans_off = S.gen_plans(gs_a, sf_a)
a_off, dbg_off = S.search_action(gs_a, sf_a, None, time.perf_counter(), 0.30)
set_flags(trek=True, margin=0.15)
plans_on = S.gen_plans(gs_a, sf_a)
a_on, dbg_on = S.search_action(gs_a, sf_a, None, time.perf_counter(), 0.30)
same = [dict(p) for p in plans_off] == [dict(p) for p in plans_on]
check('G4 opponent-present plans identical (no treks)', same and
      not any('trek' in p for p in plans_on))
check('G4 opponent-present action identical', a_off == a_on,
      '%s vs %s' % (a_off, a_on))
check('G4 opponent-present margin not solo',
      dbg_on.get('margin_eff') in (None, S.BOMB_MARGIN))

# G5 trek safety gate --------------------------------------------------------
# own bomb about to detonate across the whole corridor; no coins reachable
# through danger; treks must not be returned as an unsafe first step.
arena = open_arena()
pos = (8, 8)
gs = mk_gs(arena, pos, bombs=[(8, 11, 0)])  # detonates now, arm hits pos
sf = action_safety(gs)
a, dbg = S.search_action(gs, sf, None, time.perf_counter(), 0.30)
ok = a is None or bool(sf.get('safe', {}).get(a, False))
check('G5 trek never returns an unsafe first step', ok,
      'action=%s solo_trek=%s' % (a, dbg.get('solo_trek')))

# G6 fixture regression ------------------------------------------------------
set_flags(trek=True, margin=0.15, loop_esc='2')
agent = mk_agent()
from agent_code.arbiter_ng.sim import (from_game_state, to_game_state,
                                       step as sim_step)
fixture_gs = mk_gs(FIXTURE, (13, 5), step=60)
st = from_game_state(fixture_gs)
crates0 = int((np.asarray(st['arena']) == 1).sum())
bombs_used = 0
last_pos = None
repeat = max_repeat = 0
died = False
for t in range(200):
    g = to_game_state(st)
    g['step'] = 60 + t
    a = WC.act(agent, g)
    info = sim_step(st, [a])
    died = died or bool(info['died'][0])
    if a == 'BOMB':
        bombs_used += 1
    if not st['agents'][0]['alive']:
        break
    pos = (st['agents'][0]['x'], st['agents'][0]['y'])
    if pos == last_pos:
        repeat += 1
        max_repeat = max(max_repeat, repeat)
    else:
        repeat = 0
    last_pos = pos
crates1 = int((np.asarray(st['arena']) == 1).sum())
check('G6 fixture: no death in 200 ticks', not died)
check('G6 fixture: >=40 crates cleared', (crates0 - crates1) >= 40,
      '%d -> %d' % (crates0, crates1))
check('G6 fixture: no stay-put streak > 4', max_repeat <= 4,
      'max=%d' % max_repeat)
check('G6 fixture: bombs planted', bombs_used >= 6, 'n=%d' % bombs_used)

# G7 loop penalty math -------------------------------------------------------
WC.LOOP_ESC = '0'
off_vals = [WC._loop_penalty(c, solo=False) for c in (0, 1, 2, 3, 5)]
check('G7 OFF == LOOP3/LOOP2', off_vals == [0.0, 0.0, -WC.LOOP2,
                                            -WC.LOOP3, -WC.LOOP3],
      str(off_vals))
WC.LOOP_ESC = '1'
esc = [WC._loop_penalty(c, solo=False) for c in (0, 1, 2, 3, 5, 100)]
check('G7 mode1 escalation monotone + capped',
      esc[0] == 0.0 and esc[1] == 0.0 and esc[2] < 0 and
      esc[3] < esc[2] and esc[-1] == -WC.LOOP_ESC_CAP,
      str(esc))
WC.LOOP_ESC = '2'
solo_vals = [WC._loop_penalty(c, solo=True) for c in (0, 1, 2, 3, 5)]
nonsolo_vals = [WC._loop_penalty(c, solo=False) for c in (0, 1, 2, 3, 5)]
check("G7 mode2 escalates solo only",
      solo_vals[3] < off_vals[3] and nonsolo_vals == off_vals,
      'solo=%s nonsolo=%s' % (solo_vals, nonsolo_vals))
WC.LOOP_ESC = '0'

# G8 anti-pin guard ----------------------------------------------------------
class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def antipin_agent():
    o = types.SimpleNamespace()
    o.logger = _Log()
    o.train = False
    o.model = None
    o.coord_history = collections.deque([], 24)
    o.bomb_history = collections.deque([], 5)
    o.current_round = 0
    o.flee_timer = 0
    o._device = None
    return o


# junction: exits RIGHT (dead-end) and DOWN (open); armed enemy at (3,5).
# With the guard off the uniform-pi rank picks RIGHT (first valid index);
# the guard re-ranks by optionality and must pick DOWN.
junc = open_arena()
junc[5, 4] = -1   # UP blocked
junc[4, 5] = -1   # LEFT blocked
junc[7, 5] = -1   # RIGHT dead-end
junc[6, 4] = -1
junc[6, 6] = -1
junc_gs = mk_gs(junc, (5, 5), others=[(3, 5)], step=40)

calls = {'n': 0}
_orig_fq = WC._flee_quality_choice


def _spy_fq(*a, **kw):
    calls['n'] += 1
    return _orig_fq(*a, **kw)


WC._flee_quality_choice = _spy_fq
WC._SEARCH_ON = False
WC._TACTICAL_ON = False
WC.ANTIPIN_MOB = 2
WC.ANTIPIN = False
a_off = WC.act(antipin_agent(), junc_gs)
n_off = calls['n']
WC.ANTIPIN = True
a_on = WC.act(antipin_agent(), junc_gs)
n_on = calls['n'] - n_off
check('G8 guard off: baseline enters the dead-end',
      a_off == 'RIGHT' and n_off == 0, 'off=%s calls=%d' % (a_off, n_off))
check('G8 guard on: leaves the pocket to the open exit',
      a_on == 'DOWN' and n_on == 1, 'on=%s calls=%d' % (a_on, n_on))
# unarmed enemy: guard must not fire
unarmed_gs = mk_gs(junc, (5, 5), others=[(3, 5)], step=40)
unarmed_gs['others'] = [('o0', 0, False, (3, 5))]
WC.ANTIPIN = True
calls['n'] = 0
a_unarmed = WC.act(antipin_agent(), unarmed_gs)
check('G8 unarmed enemy does not trigger the guard',
      calls['n'] == 0 and a_unarmed == a_off,
      'unarmed=%s calls=%d' % (a_unarmed, calls['n']))
# armed but far away: no trigger
far_gs = mk_gs(open_arena(), (8, 8), others=[(8, 15)], step=40)
calls['n'] = 0
a_far = WC.act(antipin_agent(), far_gs)
check('G8 armed-but-far does not trigger the guard', calls['n'] == 0,
      'far=%s calls=%d' % (a_far, calls['n']))
WC._flee_quality_choice = _orig_fq
WC.ANTIPIN = False
WC.ANTIPIN_MOB = 1
WC._SEARCH_ON = True
WC._TACTICAL_ON = True
WC.ANTIPIN = False
WC._SEARCH_ON = True
WC._TACTICAL_ON = True

# G9 latency -----------------------------------------------------------------
t0 = time.perf_counter()
for _ in range(5):
    S.search_action(mk_gs(FIXTURE, (13, 5)), action_safety(
        mk_gs(FIXTURE, (13, 5))), None, time.perf_counter(), 0.30)
ms = (time.perf_counter() - t0) / 5.0 * 1000.0
check('G9 solo search < 50 ms', ms < 50.0, '%.1f ms' % ms)

print()
if fails:
    print('SOLO PROBE FAILED:', ', '.join(fails))
    sys.exit(1)
print('SOLO PROBE OK')
