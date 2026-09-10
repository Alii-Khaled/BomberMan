#!/usr/bin/env python3
"""ARBITER move-seed probe (E78: probe before games, E21 rule).

Verifies the MOVE_SEEDS policy in search_action on a constructed
mid-game state (model=None -> V skipped via the existing guard):
  1. default is 1 (validated P1 flow untouched).
  2. determinism: same setting twice -> identical action/scores.
  3. bomb invariance: best_bomb is EXACTLY equal under MOVE_SEEDS=1
     vs 3 (bomb plans stay single-seed with the identical j=0 seed
     formula — E71b veto calibration preserved).
  4. averaging is live: multi-seed move evaluation runs (plans scored,
     no crash) and tiny-budget exhaustion degrades gracefully.

Usage: python3 scripts/probe_arbiter_moveseeds.py
Exit nonzero on any failure.
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import arbiter.safety as AS
import arbiter.search as SR

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def midgame_state():
    f = np.zeros((17, 17), dtype=int)
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = -1
    for (cx, cy) in [(7, 5), (5, 7), (9, 9), (3, 3), (11, 5)]:
        f[cx, cy] = 1
    return {'round': 5, 'step': 120, 'field': f, 'bombs': [],
            'explosion_map': np.zeros((17, 17)), 'coins': [(8, 8)],
            'self': ('me', 2, True, (5, 5)),
            'others': [('o0', 1, True, (9, 7))], 'user_input': None}


def run(ms):
    SR.MOVE_SEEDS = ms
    gs = midgame_state()
    safety = AS.action_safety(gs)
    t0 = time.perf_counter()
    a, dbg = SR.search_action(gs, safety, None, t0, 5.0)
    return a, dbg


def main():
    check('default-one', SR.MOVE_SEEDS == 1,
          f'MOVE_SEEDS={SR.MOVE_SEEDS!r}')

    a1, d1 = run(1)
    a1b, d1b = run(1)
    check('deterministic',
          a1 == a1b and d1.get('best_move') == d1b.get('best_move')
          and d1.get('best_bomb') == d1b.get('best_bomb'),
          f'{a1}/{d1.get("best_move")}/{d1.get("best_bomb")}')
    check('plans-scored', d1.get('plans', 0) > 5,
          f"plans={d1.get('plans')}")

    a3, d3 = run(3)
    check('bomb-invariant',
          d1.get('best_bomb') == d3.get('best_bomb'),
          f"1:{d1.get('best_bomb')} vs 3:{d3.get('best_bomb')}")
    check('move-averaged-runs', d3.get('plans', 0) == d1.get('plans'),
          f"plans {d1.get('plans')} vs {d3.get('plans')}")
    check('returns-action', a3 in ('UP', 'DOWN', 'LEFT', 'RIGHT',
                                   'WAIT', None),
          f'->{a3!r}')

    # tiny budget: exhaustion path degrades without raising.
    SR.MOVE_SEEDS = 3
    gs = midgame_state()
    safety = AS.action_safety(gs)
    try:
        a, dbg = SR.search_action(gs, safety, None,
                                  time.perf_counter(), 0.0)
        check('exhaustion', True, f'->{a!r} exhausted={dbg["exhausted"]}')
    except Exception as ex:  # noqa: BLE001
        check('exhaustion', False, repr(ex))

    SR.MOVE_SEEDS = 1
    print('FAILURES:', fails if fails else 'none')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
