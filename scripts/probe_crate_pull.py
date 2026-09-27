"""Regression probe: the crate-approach pull stays removed.

The CRATE_APPROACH/RETREAT +-0.05 shaping cut coins (1.09 vs 1.34) and
was reverted. This probe asserts its absence: no crate events on any
movement, zero crate-channel reward, coin pull intact.
Fails loudly (no silent except); never touches training or checkpoints.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent_code.overlord.train import _custom, reward_from_events


def state(pos, crates=(), coins=(), step=10):
    field = np.zeros((17, 17), dtype=int)
    for (x, y) in crates:
        field[x, y] = 1
    return {'field': field, 'coins': list(coins), 'step': step,
            'self': ('t', 0, True, tuple(pos)), 'others': [], 'bombs': [],
            'explosion_map': np.zeros((17, 17), dtype=int)}


def check(name, cond):
    print(('PASS' if cond else 'FAIL') + ': ' + name)
    if not cond:
        raise SystemExit('probe failed: ' + name)


def no_crate(ev):
    return 'CRATE_APPROACH' not in ev and 'CRATE_RETREAT' not in ev


# 1-3. former pull cases now emit nothing crate-related
ev = _custom(state((5, 5), crates=[(5, 8)]), 'DOWN', state((5, 6), crates=[(5, 8)]))
check('toward crate: no crate events', no_crate(ev))
check('toward crate: zero reward', reward_from_events(None, ev) == 0.0)

ev = _custom(state((5, 6), crates=[(5, 8)]), 'UP', state((5, 5), crates=[(5, 8)]))
check('away from crate: no crate events', no_crate(ev))

ev = _custom(state((5, 5), crates=[(5, 8)], coins=[(5, 2)]), 'DOWN',
             state((5, 6), crates=[(5, 8)], coins=[(5, 2)]))
check('conflict: only coin event survives',
      no_crate(ev) and ev == ['MOVE_AWAY_TARGET'])
check('conflict reward is pure coin (-0.06)',
      abs(reward_from_events(None, ev) - (-0.06)) < 1e-9)

    # 4. coin pull itself intact (the revert left the shaping intact)
ev = _custom(state((5, 5), coins=[(5, 8)]), 'DOWN', state((5, 6), coins=[(5, 8)]))
check('coin pull intact', ev == ['MOVE_TOWARD_TARGET'])
check('coin reward intact', reward_from_events(None, ev) == 0.06)

print('PROBE PASS: 4/4 (E29 pull-absence confirmed, coin shaping intact)')
