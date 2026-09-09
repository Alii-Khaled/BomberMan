"""E44 regression probe: multi-crate bonus is REMOVED.

C1 rejected it (crates 11.0 vs 16+ bar, bombs up with crates down).
Asserts ABSENCE: BOMB_GOOD intact, no CRATE_EXTRA anywhere.
Fails loudly; never touches training or checkpoints.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent_code.overlord.train import _custom, reward_from_events


def state(pos, crates=(), step=10):
    field = np.zeros((17, 17), dtype=int)
    for (x, y) in crates:
        field[x, y] = 1
    return {'field': field, 'coins': [], 'step': step,
            'self': ('t', 0, True, tuple(pos)), 'others': [], 'bombs': [],
            'explosion_map': np.zeros((17, 17), dtype=int)}


def check(name, cond):
    print(('PASS' if cond else 'FAIL') + ': ' + name)
    if not cond:
        raise SystemExit('probe failed: ' + name)


ev = _custom(state((8, 8), crates=[(9, 8), (10, 8), (11, 8)]), 'BOMB', state((8, 8)))
check('multi-crate: BOMB_GOOD only, no EXTRA',
      ev.count('BOMB_GOOD') == 1 and 'CRATE_EXTRA' not in ev)
check('reward pure BOMB_GOOD (0.4)',
      abs(reward_from_events(None, ev) - 0.4) < 1e-9)

ev = _custom(state((8, 8), crates=[(9, 8)]), 'BOMB', state((8, 8)))
check('single crate unchanged', ev == ['BOMB_GOOD'])

print('PROBE PASS (E44 bonus-absence confirmed, BOMB_GOOD intact)')
