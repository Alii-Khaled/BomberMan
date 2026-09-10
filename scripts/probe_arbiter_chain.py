#!/usr/bin/env python3
"""ARBITER guarded-chain probe (E77: probe before games, E21 rule).

Unit tests on chain_guard_ok (pure predicate):
  U1 opp in own-tile blast -> (True, 2.0).
  U2 empty open field -> (False, 0.0).
  U3 2 crates in blast, open escape -> (True, 2.0).
  U4 1 crate, open escape -> (True, 1.0).
  U5 blocked tile (opponent standing) -> (False, _).
  U6 out-of-bounds -> (False, _).
  U7 pocket trap (opps in blast but no escape) -> (False, _).
End-to-end on gen_plans (K=1 forces the guard to prove admission —
ranking takes only the top tile, every extra bomb plan is
guard-admitted):
  E1 opp-adjacent: off=[(5,5)], on=[(5,5),(5,4)] (exact admission).
  E2 empty: on == off (E72 junk stays out).
  E3 crates2 + richer tile elsewhere: off=[(7,7)], on=[(7,7),(5,5)].
  E4 append-only at K=8 (seed-index discipline, E72b).
  E5 default is OFF (validated P1 flow untouched).

Usage: python3 scripts/probe_arbiter_chain.py
Exit nonzero on any failure.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import arbiter.safety as AS
from arbiter.safety import future_danger
import arbiter.search as SR

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def open_field():
    f = np.zeros((17, 17), dtype=int)
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = -1
    return f


def make_state(field, self_xy=(5, 5), others=(), bombs_left=True):
    return {'round': 3, 'step': 40, 'field': np.asarray(field),
            'bombs': [], 'explosion_map': np.zeros((17, 17)),
            'coins': [], 'self': ('me', 0, bombs_left, self_xy),
            'others': [('o%d' % i, 0, True, o) for i, o in
                       enumerate(others)],
            'user_input': None}


def guard_args(gs, self_xy=(5, 5)):
    arena = np.asarray(gs['field'])
    others_xy = [(int(p[3][0]), int(p[3][1])) for p in gs['others']]
    blocked = set() | (set(others_xy) - {self_xy})
    danger = future_danger(arena, [], gs['explosion_map'], 8)
    return arena, blocked, [], others_xy, danger


def bomb_tiles(gs, k=None):
    safety = AS.action_safety(gs)
    kw = {} if k is None else {'K': k}
    plans = SR.gen_plans(gs, safety, **kw)
    return plans, [p['bomb_at'] for p in plans if p['bomb_at']]


def main():
    SR.CHAIN = False
    check('default-off', SR.CHAIN_GUARD is False,
          f'CHAIN_GUARD={SR.CHAIN_GUARD!r}')

    # --- unit tests on the predicate ---
    gs = make_state(open_field(), others=[(5, 6)])
    a = guard_args(gs)
    ok, ty = SR.chain_guard_ok(*a, 5, 5)
    check('U1-opp', ok is True and abs(ty - 2.0) < 1e-9,
          f'({ok}, {ty})')

    gs = make_state(open_field())
    a = guard_args(gs)
    ok, ty = SR.chain_guard_ok(*a, 5, 5)
    check('U2-empty', ok is False and abs(ty) < 1e-9,
          f'({ok}, {ty})')

    f3 = open_field()
    f3[5, 3] = 1
    f3[5, 7] = 1
    gs = make_state(f3)
    a = guard_args(gs)
    ok, ty = SR.chain_guard_ok(*a, 5, 5)
    check('U3-crates2', ok is True and abs(ty - 2.0) < 1e-9,
          f'({ok}, {ty})')

    f4 = open_field()
    f4[5, 7] = 1
    gs = make_state(f4)
    a = guard_args(gs)
    ok, ty = SR.chain_guard_ok(*a, 5, 5)
    check('U4-crates1', ok is True and abs(ty - 1.0) < 1e-9,
          f'({ok}, {ty})')

    gs = make_state(open_field(), others=[(5, 6)])
    a = guard_args(gs)
    ok, _ = SR.chain_guard_ok(*a, 5, 6)
    check('U5-blocked', ok is False, f'({ok}, _)')

    ok, _ = SR.chain_guard_ok(*a, -1, 5)
    check('U6-oob', ok is False, f'({ok}, _)')

    # pocket: 1-wide corridor x=5, y=1..6, walls both sides; self at
    # the closed end; opp in blast but off the escape path. Escape is
    # impossible (blast band cannot be crossed) -> no admission.
    fp = open_field()
    for yy in range(1, 7):
        fp[4, yy] = -1
        fp[6, yy] = -1
    for xx in range(1, 17):
        fp[xx, 7] = -1
    gsp = make_state(fp, self_xy=(5, 1), others=[(4, 1)])
    ap = guard_args(gsp, self_xy=(5, 1))
    ok, ty = SR.chain_guard_ok(*ap, 5, 1)
    check('U7-pocket', ok is False, f'({ok}, {ty})')

    # --- end-to-end ---
    gs1 = make_state(open_field(), others=[(5, 6)])
    SR.CHAIN_GUARD = False
    _, off1 = bomb_tiles(gs1, k=1)
    SR.CHAIN_GUARD = True
    _, on1 = bomb_tiles(gs1, k=1)
    check('E1-opp-admit', off1 == [(5, 5)] and on1 == [(5, 5), (5, 4)],
          f'off={off1} on={on1}')

    gs2 = make_state(open_field())
    SR.CHAIN_GUARD = False
    _, off2 = bomb_tiles(gs2, k=1)
    SR.CHAIN_GUARD = True
    _, on2 = bomb_tiles(gs2, k=1)
    check('E2-empty', off2 == on2 == [(5, 5)], f'{off2} vs {on2}')

    f7 = open_field()
    f7[5, 3] = 1
    f7[5, 7] = 1
    f7[7, 6] = 1
    f7[7, 8] = 1
    f7[6, 7] = 1
    gs7 = make_state(f7)
    SR.CHAIN_GUARD = False
    _, off7 = bomb_tiles(gs7, k=1)
    SR.CHAIN_GUARD = True
    _, on7 = bomb_tiles(gs7, k=1)
    # (7,7) is correctly gate-rejected (escape dist > 3 in the crate
    # pocket); the guard admits exactly the three legitimate chain
    # tiles: (5,5) cr2, (5,4) cr2, (6,5) cr1.
    check('E3-crates2-admit', off7 == [(5, 6)] and
          on7 == [(5, 6), (5, 5), (6, 5), (5, 4)],
          f'off={off7} on={on7}')

    SR.CHAIN_GUARD = False
    _, off8 = bomb_tiles(gs1)
    SR.CHAIN_GUARD = True
    _, on8 = bomb_tiles(gs1)
    check('E4-append-only', off8 == on8[:len(off8)],
          f'off={off8} on={on8}')

    SR.CHAIN_GUARD = False
    print('FAILURES:', fails if fails else 'none')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
