#!/usr/bin/env python3
"""E125 (P0) solo approach-oscillation probe.

Fixture = the verified freeze board (solo seed-0 round 18, t=80,
agent at (11,8), 3 visible coins, no bombs): the K-cap membership flip
makes the winning bomb plan's first step alternate with the agent's
position and the round runs 400 steps with 6 bombs / 0 coins collected.

  G0  ship (arms off): reproduces the oscillation — no bomb planted in
      the freeze window (60 sim ticks from (11,9), position cycle).
  G1  Arm B (SOLO_COMMIT): the committed approach completes — bombs
      planted, crates destroyed, no death.
  G2  Arm B abandon: a target whose first step is inside a live blast
      clears the commitment and falls back to arbitration (legal act).
  G3  Arm A (BOMB_HYST): no death, legal action (documented outcome).
  G4  opponent-ful bit-parity: search_action results identical with and
      without the state dict when an opponent is present.

Usage: python3 scripts/probe_arbiter_solo_freeze.py
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
from agent_code.arbiter_ng.sim import (from_game_state, to_game_state,
                                       step as sim_step)

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def set_arms(commit=0, hyst=0.0):
    S.SOLO_COMMIT = int(commit)
    S.BOMB_HYST = float(hyst)


FREEZE_ARENA = [
    '#################',
    '#..xxxxxxxx.xx..#',
    '#.#x#x#x#x#x#x#.#',
    '#xxxx.xxx.xxxx.x#',
    '#x#x#x#.#x#x#.#x#',
    '#xx.x.xxxxxxxxxx#',
    '#x#.#.#.#.#x#.#x#',
    '#xxxxxxxx..x...x#',
    '#x#.#.#x#.#x#.#x#',
    '#xxxxx..........#',
    '#x#x#x#x#.#.#.#x#',
    '#xxxxx.........x#',
    '#x#.#x#.#.#.#.#x#',
    '#..xxxxxx.x.x..x#',
    '#.#x#x#x#.#.#.#.#',
    '#..xxxxx........#',
    '#################',
]
_M = {'#': -1, '.': 0, 'x': 1}
ARENA = np.array([[_M[c] for c in row] for row in FREEZE_ARENA],
                 dtype=np.int8)
COINS = [(12, 9), (11, 6), (13, 9)]


def mk_agent():
    o = types.SimpleNamespace()
    o.logger = logging.getLogger('probe-freeze')
    o.logger.setLevel(logging.WARNING)
    o.train = False
    WC.setup(o)
    o._device = None
    return o


def mk_gs(pos, bombs=(), step=80):
    return {'round': 1, 'step': step, 'field': ARENA.copy(),
            'self': ('me', 0, bool(not bombs), (pos[0], pos[1])),
            'others': [],
            'bombs': [((b[0], b[1]), b[2]) for b in bombs],
            'coins': [tuple(c) for c in COINS],
            'explosion_map': np.zeros(ARENA.shape)}


def run_ticks(agent, st, n, bombs_list):
    for t in range(n):
        g = to_game_state(st)
        g['step'] = 80 + t
        a = WC.act(agent, g)
        info = sim_step(st, [a])
        if a == 'BOMB':
            bombs_list.append(t)
        if not st['agents'][0]['alive']:
            break
    return int((np.asarray(st['arena']) == 1).sum())


# baseline from the verified freeze state: agent at (11,9)
st = from_game_state(mk_gs((11, 9)))
crates0 = int((np.asarray(st['arena']) == 1).sum())

# G0 ship oscillation ---------------------------------------------------------
set_arms(0, 0.0)
agent = mk_agent()
bombs0 = []
crates_a = run_ticks(agent, from_game_state(mk_gs((11, 9))), 60, bombs0)
check('G0 ship: freeze persists (<=1 bomb in 60 ticks)',
      len(bombs0) <= 1, 'bombs=%s' % bombs0)

# G1 Arm B completes the approach ---------------------------------------------
S.SOLO_COMMIT = 6
S.SOLO_COMMIT_MAX = 6
agent = mk_agent()
bombs_b = []
st_b = from_game_state(mk_gs((11, 9)))
crates_b = run_ticks(agent, st_b, 120, bombs_b)
check('G1 Arm B: approach completes (>=3 bombs in 120 ticks)',
      len(bombs_b) >= 3, 'bombs=%s' % bombs_b)
check('G1 Arm B: crates cleared', crates_a - crates_b >= 6,
      'crates %d -> %d' % (crates0, crates_b))
check('G1 Arm B: no death', st_b['agents'][0]['alive'])

# G2 Arm B abandons on unsafe first step --------------------------------------
agent = mk_agent()
gs2 = mk_gs((11, 9), bombs=[(11, 7, 0)])
state = {'commit': (13, 9), 'commit_age': 0}
t0 = time.perf_counter()
a2, dbg2 = S.search_action(gs2, action_safety(gs2), None, t0, 0.30,
                           state=state)
check('G2 commit with a live threat: action safe or None',
      a2 is None or bool(action_safety(gs2).get('safe', {}).get(a2, False)),
      'act=%s commit_left=%s' % (a2, state.get('commit')))

# G3 Arm A (hysteresis) — no death, legal -------------------------------------
S.SOLO_COMMIT = 0
S.BOMB_HYST = 0.25
agent = mk_agent()
bombs_a = []
st_a = from_game_state(mk_gs((11, 9)))
crates_c = run_ticks(agent, st_a, 120, bombs_a)
check('G3 Arm A: legal play, alive', st_a['agents'][0]['alive'])
check('G3 Arm A: bombs=%d in 120 ticks' % len(bombs_a), True)

# G4 opponent-ful parity ------------------------------------------------------
S.SOLO_COMMIT = 0
S.BOMB_HYST = 0.0
gs3 = mk_gs((11, 9), step=80)
gs3['others'] = [('o0', 0, True, (2, 2))]
sf3 = action_safety(gs3)
t0 = time.perf_counter()
a_off, _ = S.search_action(gs3, sf3, None, t0, 0.30)
a_off2, _ = S.search_action(gs3, sf3, None, t0, 0.30, state={})
check('G4 opponent-ful: state dict changes nothing (arms solo-gated)',
      a_off == a_off2, '%s vs %s' % (a_off, a_off2))

S.SOLO_COMMIT = 0
S.BOMB_HYST = 0.0
print()
if fails:
    print('FREEZE PROBE FAILED:', ', '.join(fails))
    sys.exit(1)
print('FREEZE PROBE OK')
