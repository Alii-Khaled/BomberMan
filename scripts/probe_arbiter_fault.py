#!/usr/bin/env python3
"""ARBITER fault-injection probe (P1.1: gates before games, E21 rule).

Calls act() on malformed/degenerate game states and asserts it ALWAYS
returns a legal action string (never raises). The engine has no fallback
agent: an unguarded act() exception kills the whole tournament game
(environment.py:440-447), or benches us for the round under
--silence-errors. The P1.1 armor (act wrapper -> WAIT) must hold.

Usage: python3 scripts/probe_arbiter_fault.py
Exit nonzero on any failure.
"""
import logging
import os
import sys
import time
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
# Small budget: search must degrade gracefully, probe stays fast.
os.environ['ARBITER_TIME_BUDGET'] = '0.05'

import numpy as np

import arbiter.callbacks as AC

ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB')

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def base_state():
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return {'round': 1, 'step': 1, 'field': field,
            'bombs': [], 'explosion_map': np.zeros((17, 17)),
            'coins': [(3, 3)], 'self': ('me', 0, True, (1, 1)),
            'others': [], 'user_input': None}


def main():
    logging.basicConfig(level=logging.CRITICAL)
    t0 = time.perf_counter()
    fake = SimpleNamespace(train=False,
                           logger=logging.getLogger('probe_fault'))
    try:
        AC.setup(fake)
        check('setup-ok', True)
    except Exception as ex:  # noqa: BLE001
        check('setup-ok', False, repr(ex))
        print('FAILURES:', fails)
        return 1
    check('setup-time', time.perf_counter() - t0 < 60.0,
          f'{time.perf_counter() - t0:.1f}s')

    cases = {}
    gs = base_state()
    cases['base'] = (gs, None)
    gs = base_state()
    del gs['coins']
    cases['missing-coins'] = (gs, None)
    gs = base_state()
    gs['coins'] = None
    cases['coins-none'] = (gs, None)
    gs = base_state()
    gs['bombs'] = None
    cases['bombs-none'] = (gs, None)
    gs = base_state()
    gs['explosion_map'] = None
    cases['explosion-none'] = (gs, None)
    gs = base_state()
    gs['explosion_map'] = np.full((17, 17), np.nan)
    cases['explosion-nan'] = (gs, None)
    gs = base_state()
    gs['others'] = None
    cases['others-none'] = (gs, None)
    gs = base_state()
    del gs['round']
    del gs['step']
    cases['missing-round-step'] = (gs, None)
    gs = base_state()
    gs['step'] = 'bogus'
    cases['step-bogus'] = (gs, None)
    gs = base_state()
    gs['field'] = gs['field'].tolist()
    cases['field-as-list'] = (gs, None)
    # Walled in: all interior crates except spawn tile.
    gs = base_state()
    gs['field'][1:-1, 1:-1] = 1
    gs['field'][1, 1] = 0
    cases['walled-in'] = (gs, None)
    # Walled in + no bomb: must still answer (WAIT fallback chain).
    gs = base_state()
    gs['field'][1:-1, 1:-1] = 1
    gs['field'][1, 1] = 0
    gs['self'] = ('me', 0, False, (1, 1))
    cases['walled-in-no-bomb'] = (gs, None)
    # Total garbage: armor must convert to WAIT, never raise.
    cases['none-state'] = (None, 'WAIT')
    cases['empty-dict'] = ({}, 'WAIT')

    for name, (state, want) in cases.items():
        try:
            t1 = time.perf_counter()
            a = AC.act(fake, state)
            dt = time.perf_counter() - t1
        except Exception as ex:  # noqa: BLE001
            check(f'act-{name}', False, f'raised {ex!r}')
            continue
        ok = a in ACTIONS and (want is None or a == want)
        check(f'act-{name}', ok, f'-> {a!r} ({dt * 1000:.1f} ms)')
        if dt > 2.0:
            check(f'act-{name}-latency', False, f'{dt:.2f}s > 2s')

    # Success-path sanity: base state answered through the real policy
    # (not the armor) — the armor must be inert on valid input.
    try:
        impl_action = AC._act_impl(fake, base_state(),
                                   time.perf_counter())
        check('impl-path-live', impl_action in ACTIONS,
              f'-> {impl_action!r}')
    except Exception as ex:  # noqa: BLE001
        check('impl-path-live', False, repr(ex))

    print('FAILURES:', fails if fails else 'none')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
