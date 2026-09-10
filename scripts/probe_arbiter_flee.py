#!/usr/bin/env python3
"""E90 probe: ARBITER_FLEE_Q post-plant flee quality.

Groups:
  G1 default off (ship-identical) and helper import.
  G2 unit: open-space/distance score picks the non-pocket move among the
     mask's safe options and ignores unsafe moves.
  G3 fallback: returns None when no safe move exists; returns the only
     safe move when it is the sole option.
  G4 act smoke: FLEE_Q=1 act() on a live threat state returns a legal
     action (threat state injected via flee_timer).
Exit nonzero on any failure.
"""
import importlib
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def reload_arb(flee_q):
    for mod in list(sys.modules):
        if mod.startswith('agent_code.arbiter'):
            del sys.modules[mod]
    if flee_q is None:
        os.environ.pop('ARBITER_FLEE_Q', None)
    else:
        os.environ['ARBITER_FLEE_Q'] = flee_q
    import agent_code.arbiter.callbacks as cb
    importlib.reload(cb)
    return cb


def board_with_pocket():
    """s1-style pocket: agent (11,9), bomb same tile, opponent below-left."""
    f = np.zeros((17, 17), dtype=int)
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                f[x, y] = -1
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = -1
    return f


def main():
    cb = reload_arb(None)
    check('G1 default FLEE_Q off', cb.FLEE_Q is False)
    check('G1 helper present', callable(cb._flee_quality_choice))

    pi = np.zeros(6, dtype=np.float64)
    fld = board_with_pocket()
    bombs = [((11, 9), 3)]
    others = [(10, 11)]
    valid = {'UP': True, 'RIGHT': True, 'DOWN': True, 'LEFT': True,
             'WAIT': True, 'BOMB': False}
    safe = {'UP': False, 'RIGHT': True, 'DOWN': True, 'LEFT': False,
            'WAIT': False, 'BOMB': False}
    a = cb._flee_quality_choice(fld, bombs, others, 11, 9,
                                valid, safe, pi)
    check('G2 picks open/away move (RIGHT)', a == 'RIGHT', str(a))

    safe2 = {'UP': False, 'RIGHT': False, 'DOWN': True, 'LEFT': False,
             'WAIT': False, 'BOMB': False}
    a2 = cb._flee_quality_choice(fld, bombs, others, 11, 9,
                                 valid, safe2, pi)
    check('G2 sole safe move returned', a2 == 'DOWN', str(a2))

    safe3 = {k: False for k in valid}
    a3 = cb._flee_quality_choice(fld, bombs, others, 11, 9,
                                 valid, safe3, pi)
    check('G3 no safe move -> None', a3 is None, str(a3))

    # G4 act smoke with an injected flee lock
    cb = reload_arb('1')
    check('G4 FLEE_Q on', cb.FLEE_Q is True)
    gs = {'round': 1, 'step': 10, 'field': fld,
          'self': ('me', 0, False, (11, 9)),
          'others': [('o', 0, True, (10, 11))], 'bombs': [((11, 9), 3)],
          'explosion_map': np.zeros((17, 17)), 'coins': []}
    self_ns = types.SimpleNamespace()
    self_ns.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None)
    self_ns.train = False
    cb.setup(self_ns)
    self_ns.flee_timer = 5
    act = cb.act(self_ns, gs)
    check('G4 act legal under flee lock',
          act in ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'), str(act))

    if fails:
        print('FAILED:', fails)
        return 1
    print('ALL FLEE-Q PROBE GROUPS PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
