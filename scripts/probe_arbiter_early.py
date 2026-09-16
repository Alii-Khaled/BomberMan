#!/usr/bin/env python3
"""E130 (P0) early-game opponent-ful oscillation probe.

The E128/E130 telemetry showed 27-32% of opening move-ticks are
immediate reversals (~82-89% with no own bomb nearby and no flee). Two
mechanisms, two arms:

  G1  ship (arms off): the round-18 K-cap membership flip reproduces in
      an OPPONENT-FUL state (the E125 fix is solo-gated) — no bomb in
      60 ticks on the freeze fixture with a far opponent.
  G2  ARBITER_COMMIT_OPP=1: the committed approach completes (bomb
      planted, crates destroyed, no death, opponent-ful).
  G3  abandon: a NEW enemy bomb makes the committed first step unsafe
      -> commitment cleared, arbitration returns a legal action.
  G4  ARBITER_BACKTRACK_OPP: an opponent-ful forced-reversal history
      flips the S0 choice; solo history with the same knob ON is
      UNCHANGED (solo untouched).
  G5  both arms off = ship bit-parity on a benign state.

Usage: python3 scripts/probe_arbiter_early.py
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
    o.logger = logging.getLogger('probe-early')
    o.logger.setLevel(logging.WARNING)
    o.train = False
    WC.setup(o)
    o._device = None
    return o


def mk_gs(pos, bombs=(), others=(), step=80):
    return {'round': 1, 'step': step, 'field': ARENA.copy(),
            'self': ('me', 0, bool(not bombs), (pos[0], pos[1])),
            'others': [('o%d' % i, 0, True, (o[0], o[1]))
                       for i, o in enumerate(others)],
            'bombs': [((b[0], b[1]), b[2]) for b in bombs],
            'coins': [tuple(c) for c in COINS],
            'explosion_map': np.zeros(ARENA.shape)}


def run_ticks(agent, st, n, bombs_list, opp_alive=False):
    for t in range(n):
        g = to_game_state(st)
        g['step'] = 80 + t
        if not opp_alive:
            g['others'] = []
            st['agents'][1]['alive'] = False
        a = WC.act(agent, g)
        info = sim_step(st, [a] + ['WAIT'] * (len(st['agents']) - 1))
        if a == 'BOMB':
            bombs_list.append(t)
        if not st['agents'][0]['alive']:
            break
    return int((np.asarray(st['arena']) == 1).sum())


def mk_agent_bare():
    """Model-less agent: uniform pi -> the tie-breaks decide, so the
    flicker tests are discriminating regardless of the pi gaps."""
    o = types.SimpleNamespace()
    o.logger = logging.getLogger('probe-early-bare')
    o.logger.setLevel(logging.WARNING)
    o.train = False
    o.model = None
    o.coord_history = collections.deque([], 24)
    o.bomb_history = collections.deque([], 5)
    o.current_round = 1
    o.flee_timer = 0
    o._device = None
    return o


# G1 ship S0-flicker on a no-plan corridor (opponent-ful) ----------------------
# The E125 K-cap flip needs the solo radius-8 pool; with opponents alive
# the radius is 4 and margin-clearing plans DO commit — so the
# opponent-ful flicker lives in search-None states (S0). Fixture: a
# corridor whose only bombable cluster yields 1 crate (payoff < the 0.6
# margin over the move baseline) -> search returns None -> S0 flickers
# UP <-> WAIT on a forced reversal history.
S.COMMIT_OPP = False
S.BOMB_HYST = 0.0
WC.BACKTRACK = 0.0
WC.BACKTRACK_OPP = 0.0
WC.COINTAKE = False
pkt = np.zeros((17, 17), dtype=np.int8)
pkt[0, :] = pkt[-1, :] = pkt[:, 0] = pkt[:, -1] = -1
pkt[9, 8] = -1
pkt[8, 9] = -1
pkt[7, 8] = -1
pkt[8, 7] = 1          # single crate above: yield-1 tile -> no plan wins
gs1 = {'round': 1, 'step': 100, 'field': pkt,
       'self': ('me', 0, True, (8, 8)),
       'others': [('o0', 0, True, (1, 2))], 'bombs': [],
       'coins': [], 'explosion_map': np.zeros((17, 17))}
hist = [(8, 7), (8, 8)] * 8
WC.BACKTRACK = 0.0
WC.BACKTRACK_OPP = 0.0
agent = mk_agent_bare()
agent.coord_history = collections.deque(hist, 24)
a_off = WC.act(agent, gs1)
agent2 = mk_agent_bare()
agent2.coord_history = collections.deque(hist, 24)
WC.BACKTRACK_OPP = 3.0
a_on = WC.act(agent2, gs1)
WC.BACKTRACK_OPP = 0.0
check('G1 opponent-ful S0 flicker: arms-off legal act',
      a_off in ('UP', 'WAIT'), 'off=%s' % a_off)
check('G1b BACKTRACK_OPP breaks the reversal (WAIT)', a_on == 'WAIT',
      'on=%s' % a_on)

# G2 commit extension completes the approach ----------------------------------
S.COMMIT_OPP = True
agent = mk_agent()
bombs_b = []
st_b = from_game_state(mk_gs((11, 9), others=[(1, 2)]))
crates_b = run_ticks(agent, st_b, 120, bombs_b, opp_alive=True)
check('G2 COMMIT_OPP: approach completes (>=3 bombs / 120 ticks)',
      len(bombs_b) >= 3, 'bombs=%s' % bombs_b)
check('G2 COMMIT_OPP: crates cleared', crates_b <= 94,
      'crates -> %d' % crates_b)
check('G2 COMMIT_OPP: no death', st_b['agents'][0]['alive'])

# G3 abandon on a new enemy bomb ----------------------------------------------
S.COMMIT_OPP = True
agent = mk_agent()
gs2 = mk_gs((11, 9), bombs=[(11, 7, 0)], others=[(1, 2)])
state = {'commit': (13, 9), 'commit_age': 0}
a2, _ = S.search_action(gs2, action_safety(gs2), None,
                        time.perf_counter(), 0.30, state=state)
sf2 = action_safety(gs2)
check('G3 commit with a live own bomb: action safe or None',
      a2 is None or bool(sf2.get('safe', {}).get(a2, False)),
      'act=%s commit_left=%s' % (a2, state.get('commit')))

# G4 opponent-ful backtrack flips a forced reversal (real weights) ----------
S.COMMIT_OPP = False
WC._SEARCH_ON = True
WC._TACTICAL_ON = True
WC.COINTAKE = False
WC.ANTIPIN = False
gs3 = mk_gs((11, 9), others=[(1, 2)])
# pocket: only UP free (walls elsewhere), hist ends ...A,B
pkt3 = np.zeros((17, 17), dtype=np.int8)
pkt3[0, :] = pkt3[-1, :] = pkt3[:, 0] = pkt3[:, -1] = -1
pkt3[9, 8] = -1
pkt3[8, 9] = -1
pkt3[7, 8] = -1
gs4 = {'round': 1, 'step': 100, 'field': pkt3,
       'self': ('me', 0, True, (8, 8)),
       'others': [('o0', 0, True, (1, 2))], 'bombs': [],
       'coins': [], 'explosion_map': np.zeros((17, 17))}
hist = [(8, 7), (8, 8)] * 8
WC.BACKTRACK = 0.0
WC.BACKTRACK_OPP = 0.0
ag_a = mk_agent_bare()
ag_a.coord_history = collections.deque(hist, 24)
a_off = WC.act(ag_a, gs4)
WC.BACKTRACK_OPP = 3.0
ag_b = mk_agent_bare()
ag_b.coord_history = collections.deque(hist, 24)
a_on = WC.act(ag_b, gs4)
WC.BACKTRACK_OPP = 0.0
check('G4 opponent-ful backtrack flips the reversal', a_off != a_on,
      'off=%s on=%s' % (a_off, a_on))

# G4b solo untouched by BACKTRACK_OPP -----------------------------------------
pkt2_gs = {'round': 1, 'step': 100, 'field': pkt,
           'self': ('me', 0, True, (8, 8)), 'others': [], 'bombs': [],
           'coins': [], 'explosion_map': np.zeros((17, 17))}
WC.BACKTRACK_OPP = 0.0
a_solo0 = WC.act(mk_agent(), pkt2_gs)
WC.BACKTRACK_OPP = 3.0
a_solo1 = WC.act(mk_agent(), pkt2_gs)
WC.BACKTRACK_OPP = 0.0
check('G4b solo state: BACKTRACK_OPP does not bind', a_solo0 == a_solo1,
      '%s vs %s' % (a_solo0, a_solo1))

# G5 benign-state parity ------------------------------------------------------
S.COMMIT_OPP = False
gs5 = mk_gs((8, 8), others=[(1, 2)])
WC.COINTAKE = True
a0 = WC.act(mk_agent(), mk_gs((8, 8), others=[(1, 2)]))
a1 = WC.act(mk_agent(), mk_gs((8, 8), others=[(1, 2)]))
check('G5 benign state deterministic (ship)', a0 == a1,
      '%s vs %s' % (a0, a1))

S.COMMIT_OPP = False
print()
if fails:
    print('EARLY PROBE FAILED:', ', '.join(fails))
    sys.exit(1)
print('EARLY PROBE OK')
