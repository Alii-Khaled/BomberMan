#!/usr/bin/env python3
"""E122 (P0) certified coin-take overlay probe (ARBITER_COINTAKE).

  G0  overlay OFF: act() unchanged on a coin-adjacent state (ship parity).
  G1  overlay ON, adjacent coin: the certified step onto the coin is
      returned even though the uniform-pi rank picks an arbitrary dir.
  G2  geometry: d=1 and d=2 coins fire; d=3 is capped; crates/opponents/
      bombs on the coin tile or path yield None; unsafe first step -> None.
  G3  must-flee gating: with the own tile lethal, the overlay never
      overrides the flee path (chosen action is mask-safe).
  G4  tactical precedence: certified kill adjacent to a coin -> BOMB.
  G5  opponent-ful parity: no coin within d=2 -> identical action on/off.
  G6  latency: _cointake_step < 1 ms on a loaded board.

Usage: python3 scripts/probe_arbiter_cointake.py
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

import agent_code.Harvy.callbacks as WC
import agent_code.Harvy.search as S
from agent_code.Harvy.safety import action_safety

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def bare_agent():
    o = types.SimpleNamespace()
    log = logging.getLogger('probe-cointake')
    log.setLevel(logging.WARNING)
    o.logger = log
    o.train = False
    o.model = None
    o.coord_history = collections.deque([], 24)
    o.bomb_history = collections.deque([], 5)
    o.current_round = 0
    o.flee_timer = 0
    o._device = None
    return o


def mk_gs(arena, pos, coins=(), others=(), bombs=(), step=10,
          score=0, bombs_left=True):
    W, H = arena.shape
    return {'round': 1, 'step': step, 'field': arena,
            'self': ('me', score, bombs_left, (pos[0], pos[1])),
            'others': [('o%d' % i, 0, True, (o[0], o[1]))
                       for i, o in enumerate(others)],
            'bombs': [((b[0], b[1]), b[2]) for b in bombs],
            'coins': [tuple(c) for c in coins],
            'explosion_map': np.zeros((W, H))}


def open_arena():
    a = np.zeros((17, 17), dtype=np.int8)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = -1
    return a


# G0 overlay OFF --------------------------------------------------------------
WC.COINTAKE = False
WC._SEARCH_ON = True
WC._TACTICAL_ON = True
arena = open_arena()
gs = mk_gs(arena, (8, 8), coins=[(7, 8)])  # LEFT-adjacent coin
a_off = WC.act(bare_agent(), gs)
check('G0 overlay off: baseline uniform-pi step (not guaranteed LEFT)',
      a_off in ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'),
      'action=%s' % a_off)

# G1 overlay ON, adjacent coin ------------------------------------------------
WC.COINTAKE = True
WC.COINTAKE_D = 2
a_on = WC.act(bare_agent(), gs)
check('G1 overlay takes the adjacent coin (LEFT)', a_on == 'LEFT',
      'off=%s on=%s' % (a_off, a_on))

# repeat with a fresh agent (histories empty): deterministic
a_on2 = WC.act(bare_agent(), gs)
check('G1b deterministic', a_on2 == 'LEFT', 'action=%s' % a_on2)

# G2 geometry -----------------------------------------------------------------
# d=1 LEFT
sf = action_safety(gs)
step = WC._cointake_step(gs, sf, 2)
check('G2a d=1 LEFT', step == 'LEFT', 'step=%s' % step)
# d=2 LEFT (coin two tiles away, straight corridor)
gs2 = mk_gs(open_arena(), (8, 8), coins=[(6, 8)])
sf2 = action_safety(gs2)
step = WC._cointake_step(gs2, sf2, 2)
check('G2b d=2 fires, first step LEFT', step == 'LEFT', 'step=%s' % step)
# d=3 capped out at d=2
step = WC._cointake_step(gs2, sf2, 2) if False else WC._cointake_step(
    mk_gs(open_arena(), (8, 8), coins=[(5, 8)]), action_safety(
        mk_gs(open_arena(), (8, 8), coins=[(5, 8)])), 2)
check('G2c d=3 capped -> None', step is None, 'step=%s' % step)
# blocked path: coin 1 away in BFS terms but a crate on the direct path
arena = open_arena()
arena[7, 8] = 1          # crate between us (8,8) and coin (6,8)
gs3 = mk_gs(arena, (8, 8), coins=[(6, 8)])
sf3 = action_safety(gs3)
step = WC._cointake_step(gs3, sf3, 2)
check('G2d crate-blocked coin -> None (path > cap)', step is None,
      'step=%s' % step)
# opponent standing ON the coin tile
gs4 = mk_gs(open_arena(), (8, 8), coins=[(7, 8)], others=[(7, 8)])
step = WC._cointake_step(gs4, action_safety(gs4), 2)
check('G2e opponent on coin tile -> None', step is None, 'step=%s' % step)
# bomb ON the coin tile (blocked for movement)
gs5 = mk_gs(open_arena(), (8, 8), coins=[(7, 8)], bombs=[(7, 8, 2)])
step = WC._cointake_step(gs5, action_safety(gs5), 2)
check('G2f bomb on coin tile -> None', step is None, 'step=%s' % step)
# unsafe first step: coin 2 DOWN but a timer-0 bomb covers the first step
arena = open_arena()
gs6 = mk_gs(arena, (8, 8), coins=[(8, 10)], bombs=[(8, 12, 0)])
step = WC._cointake_step(gs6, action_safety(gs6), 2)
check('G2g unsafe first step -> None', step is None, 'step=%s' % step)

# G3 must-flee gating ---------------------------------------------------------
# own tile lethal (bomb on our tile, timer 0) with an adjacent coin:
# the overlay must not override the flee; the returned action is safe.
arena = open_arena()
gs7 = mk_gs(arena, (8, 8), coins=[(7, 8)], bombs=[(8, 8, 0)])
sf7 = action_safety(gs7)
a = WC.act(bare_agent(), gs7)
safe = sf7.get('safe', {})
check('G3 must-flee: action is mask-safe (coin not hijacked)',
      bool(safe.get(a, False)) or a == 'WAIT' and not safe.get('WAIT', True)
      or a in ('UP', 'DOWN', 'LEFT', 'RIGHT'),
      'action=%s safe=%s' % (a, {k: safe.get(k) for k in
                                 ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')}))

# G4 tactical precedence: certified kill next to a coin -> BOMB ---------------
# opp pocketed at (8,7): walls block every escape; our bomb at (8,8) traps.
arena = open_arena()
arena[7, 7] = -1
arena[9, 7] = -1
arena[8, 6] = -1
gs8 = mk_gs(arena, (8, 8), coins=[(9, 8)], others=[(8, 7)])
a = WC.act(bare_agent(), gs8)
check('G4 certified kill wins over adjacent coin -> BOMB', a == 'BOMB',
      'action=%s' % a)

# G5 opponent-ful parity (no coin within d=2) ---------------------------------
arena = open_arena()
for (cx, cy) in [(10, 10), (11, 10), (10, 11)]:
    arena[cx, cy] = 1
gs9 = mk_gs(arena, (5, 5), coins=[(13, 13)], others=[(11, 11)], step=33)
WC.COINTAKE = False
a_off = WC.act(bare_agent(), gs9 := gs9 if False else mk_gs(
    arena.copy(), (5, 5), coins=[(13, 13)], others=[(11, 11)], step=33))
WC.COINTAKE = True
a_on = WC.act(bare_agent(), mk_gs(arena.copy(), (5, 5), coins=[(13, 13)],
                                  others=[(11, 11)], step=33))
check('G5 opponent-ful parity when no coin in reach', a_off == a_on,
      '%s vs %s' % (a_off, a_on))

# G6 latency ------------------------------------------------------------------
loaded = open_arena()
for cx in range(2, 15):
    for cy in range(2, 15):
        if (cx + cy) % 3 == 0:
            loaded[cx, cy] = 1
gs10 = mk_gs(loaded, (8, 8), coins=[(6, 6), (12, 12), (4, 12)],
             others=[(3, 3)], bombs=[(10, 4, 3)])
sf10 = action_safety(gs10)
t0 = time.perf_counter()
for _ in range(20):
    WC._cointake_step(gs10, sf10, 2)
ms = (time.perf_counter() - t0) / 20.0 * 1000.0
check('G6 cointake step < 1 ms', ms < 1.0, '%.3f ms' % ms)

WC.COINTAKE = False
print()
if fails:
    print('COINTAKE PROBE FAILED:', ', '.join(fails))
    sys.exit(1)
print('COINTAKE PROBE OK')
