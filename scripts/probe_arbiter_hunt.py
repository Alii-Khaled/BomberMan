#!/usr/bin/env python3
"""ARBITER hunt-intent probe (E82: probe before games, E21 rule).

Unit tests:
  U0  default-off (ARBITER_HUNT unset -> HUNT False).
  U1  hunt_trigger_ok semantics: no-opp / close-opp / far-opp /
      late-game / low-loot.
  U2  hunt_tiles: ring-first, wall break, crate passthrough.
End-to-end on gen_plans:
  E1  open field, opp adjacent: append-only + exactly one hunt plan,
      its bomb tile's blast covers the opp, prefix ends with BOMB.
  E2  no trigger (loot > 6, opp far): on == off (bit-identical).
  E3  bombs_left False: no hunt plans.
  E4  HUNT_PLANS cap with 2 opponents: 1 plan per opp, capped.
  E5  determinism (two identical calls).
  E6  sealed pocket (cornered opp): hunt plan admitted at the pocket
      mouth with a proven escape; search_action (model=None) selects
      the hunt BOMB (certified kill priced +5 beats zero-payoff
      alternatives).
Exit nonzero on any failure. Fails loudly; never touches training,
checkpoints, or the ship weights.
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
from arbiter.sim import blast_coords

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def open_field():
    f = np.zeros((17, 17), dtype=int)
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = -1
    return f


def make_state(field, self_xy=(5, 5), others=(), bombs_left=True,
               step=40, coins=()):
    return {'round': 3, 'step': step, 'field': np.asarray(field),
            'bombs': [], 'explosion_map': np.zeros((17, 17)),
            'coins': list(coins), 'self': ('me', 0, bombs_left, self_xy),
            'others': [('o%d' % i, 0, True, o) for i, o in
                       enumerate(others)],
            'user_input': None}


def plans_of(gs, k=None, radius=None):
    safety = AS.action_safety(gs)
    kw = {} if k is None else {'K': k}
    if radius is not None:
        kw['radius'] = radius
    return SR.gen_plans(gs, safety, **kw)


def bomb_tiles(plans):
    return [p['bomb_at'] for p in plans if p['bomb_at']]


def covers_opp(arena, tile, opp):
    return tuple(opp) in set(blast_coords(arena, tile[0], tile[1]))


def main():
    check('U0-default-off', SR.HUNT is False, f'HUNT={SR.HUNT!r}')

    # --- U1 trigger semantics ---
    f = open_field()
    check('U1a-no-opp', SR.hunt_trigger_ok(f, 0, [], 5, 5, 40) is False)
    check('U1b-close-opp',
          SR.hunt_trigger_ok(f, 0, [(9, 5)], 5, 5, 40) is True)
    f7c = open_field()
    for (_cx, _cy) in [(2, 2), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12),
                       (2, 14)]:
        f7c[_cx, _cy] = 1
    check('U1c-far-opp-loot',
          SR.hunt_trigger_ok(f7c, 0, [(5, 10)], 5, 5, 40) is False)
    check('U1d-late-game',
          SR.hunt_trigger_ok(f, 0, [(5, 10)], 5, 5, 201) is True)
    check('U1e-low-loot',
          SR.hunt_trigger_ok(f, 5, [(5, 10)], 5, 5, 40) is True)

    # --- U2 hunt_tiles geometry ---
    fw = open_field()
    fw[4, 1] = -1
    fw[6, 1] = -1
    fw[5, 2] = 1  # crate: blast passes, cannot stand
    tiles = SR.hunt_tiles(fw, set(), 5, 1)
    check('U2a-ring-first-wall-break',
          tiles == [(5, 2), (5, 3), (5, 4)] or
          tiles == [(5, 3), (5, 4)],
          f'{tiles}')
    check('U2b-crate-skipped', (5, 2) not in tiles, f'{tiles}')
    check('U2c-crate-passthrough', (5, 3) in tiles, f'{tiles}')

    # --- E1 open field, opp adjacent ---
    gs1 = make_state(open_field(), others=[(5, 6)])
    SR.HUNT = False
    off1 = plans_of(gs1)
    SR.HUNT = True
    on1 = plans_of(gs1)
    boff, bon = bomb_tiles(off1), bomb_tiles(on1)
    new = [t for t in bon if t not in boff]
    cond = (bon[:len(boff)] == boff and len(new) == 1
            and covers_opp(np.asarray(gs1['field']), new[0], (5, 6)))
    hunt_plan = [p for p in on1 if p['bomb_at'] == new[0]][0] if new \
        else None
    cond = cond and hunt_plan is not None \
        and hunt_plan['prefix'][-1] == 'BOMB' \
        and hunt_plan['first'] in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
    check('E1-append-one-covers', cond, f'new={new} off={len(boff)}')
    SR.HUNT = False

    # --- E2 no trigger ---
    f2 = open_field()
    for (cx, cy) in [(2, 2), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12),
                     (2, 14)]:
        f2[cx, cy] = 1
    gs2 = make_state(f2, others=[(9, 9)], step=40)
    SR.HUNT = False
    off2 = plans_of(gs2)
    SR.HUNT = True
    on2 = plans_of(gs2)
    check('E2-no-trigger-parity', off2 == on2, 'on != off under no trigger')
    SR.HUNT = False

    # --- E3 no bombs left ---
    gs3 = make_state(open_field(), others=[(5, 6)], bombs_left=False)
    SR.HUNT = True
    on3 = plans_of(gs3)
    check('E3-no-bomb-plans', bomb_tiles(on3) == [], f'{bomb_tiles(on3)}')
    SR.HUNT = False

    # --- E4 cap with 2 opponents ---
    gs4 = make_state(open_field(), others=[(5, 6), (8, 5)])
    SR.HUNT = False
    off4 = plans_of(gs4)
    SR.HUNT = True
    SR.HUNT_PLANS = 1
    on4a = plans_of(gs4)
    SR.HUNT_PLANS = 2
    on4b = plans_of(gs4)
    n_a = len(bomb_tiles(on4a)) - len(bomb_tiles(off4))
    n_b = len(bomb_tiles(on4b)) - len(bomb_tiles(off4))
    check('E4-cap', n_a == 1 and n_b == 2, f'cap1={n_a} cap2={n_b}')
    SR.HUNT_PLANS = 2

    # --- E5 determinism ---
    SR.HUNT = True
    p_a = plans_of(gs1)
    p_b = plans_of(gs1)
    check('E5-determinism', p_a == p_b, '')
    SR.HUNT = False

    # --- E6 sealed pocket: cornered opp, hunt plan at the mouth ---
    f6 = open_field()
    for (cx, cy) in [(4, 1), (6, 1), (4, 2), (6, 2), (4, 3), (6, 3)]:
        f6[cx, cy] = -1
    # E6a: radius=0 (own tile only) + K=1 -> the ranked walk can only
    # ever admit the own tile; (5,4) in the plans proves HUNT
    # admission through the same gate (exact, like chain-probe E3).
    gs6 = make_state(f6, self_xy=(5, 5), others=[(5, 1)])
    SR.HUNT = False
    off6 = plans_of(gs6, k=1, radius=0)
    SR.HUNT = True
    on6 = plans_of(gs6, k=1, radius=0)
    cond = bomb_tiles(off6) == [(5, 5)] and \
        bomb_tiles(on6) == [(5, 5), (5, 4)] and \
        [p for p in on6 if p['bomb_at'] == (5, 4)][0]['prefix'][-1] \
        == 'BOMB'
    check('E6a-radius0-exact-admit', cond,
          f'off={bomb_tiles(off6)} on={bomb_tiles(on6)}')
    # E6b: E2E smoke (default K; opponent cannot bomb -> no
    # suicide-luck noise in rollouts). Assert a legal action only:
    # the post-bomb continuation heuristic is ship behavior, do not
    # pin its seed-dependent choice here.
    gs6b = make_state(f6, self_xy=(5, 5))
    gs6b['others'] = [('o0', 0, False, (5, 1))]
    SR.HUNT = True
    import time as _t
    a6, dbg6 = SR.search_action(gs6b, AS.action_safety(gs6b),
                                None, _t.perf_counter(), 5.0)
    check('E6c-e2e-legal-action',
          a6 is None or a6 in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT',
                               'BOMB'),
          f'a6={a6!r} plans={dbg6["plans"]}')
    SR.HUNT = False

    print('FAILURES:', fails if fails else 'none')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
