#!/usr/bin/env python3
"""E88 probe: ARBITER_CRN common-random-numbers rollout seeds.

Groups:
  G1 default off: search_action uses a distinct rollout seed per plan
     (the validated unpaired P1 flow) and returns a legal action.
  G2 CRN on: every plan in a step shares one rollout seed (paired
     comparison; the per-tick opponent RNG inside score_plan then keys
     on (seed, tick, opponent)).
  G3 CRN determinism: repeated search_action calls on one state agree.
  G4 score_plan bit-parity: CRN=0 reproduces the legacy score exactly.
Exit nonzero on any failure.
"""
import importlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)

import numpy as np

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def reload_search(crn):
    for mod in list(sys.modules):
        if mod.startswith('agent_code.arbiter') or mod in (
                'arbiter.search', 'arbiter.safety', 'arbiter.features',
                'arbiter.sim', 'arbiter.model'):
            del sys.modules[mod]
    if crn is None:
        os.environ.pop('ARBITER_CRN', None)
    else:
        os.environ['ARBITER_CRN'] = crn
    import agent_code.arbiter.search as S
    import agent_code.arbiter.safety as AS
    importlib.reload(AS)
    importlib.reload(S)
    return S, AS


def make_state(seed=0):
    rng = np.random.default_rng(seed)
    W = H = 17
    arena = np.zeros((W, H), dtype=int)
    for x in range(W):
        for y in range(H):
            if (x + 1) * (y + 1) % 2 == 1:
                arena[x, y] = -1
    arena[:1, :] = -1
    arena[-1:, :] = -1
    arena[:, :1] = -1
    arena[:, -1:] = -1
    arena[(rng.random((W, H)) < 0.75) & (arena == 0)] = 1
    free = np.argwhere(arena == 0)
    rng.shuffle(free)
    self_pos = tuple(int(v) for v in free[0])
    others = [tuple(int(v) for v in free[i]) for i in (1, 2, 3)]
    coins = [tuple(int(v) for v in free[i]) for i in range(5, 14)]
    return {'round': 1, 'step': 50, 'field': arena,
            'self': ('me', 0, True, self_pos),
            'others': [('o%d' % i, 0, True, p) for i, p in enumerate(others)],
            'bombs': [], 'explosion_map': np.zeros((W, H)), 'coins': coins}


def seed_spy_call(S, gs, safety):
    """Run search_action while recording the seed passed per plan."""
    seeds = []
    orig = S.score_plan

    def spy(st0, plan, horizon=None, seed=0, w_crate=None, w_death=None):
        seeds.append(seed)
        if horizon is None:
            horizon = S.H
        if w_crate is None:
            w_crate = S.W_CRATE
        if w_death is None:
            w_death = S.W_DEATH
        return orig(st0, plan, horizon=horizon, seed=seed,
                    w_crate=w_crate, w_death=w_death)

    import time as _time
    S.score_plan = spy
    try:
        action, dbg = S.search_action(gs, safety, None,
                                      _time.perf_counter(), 10.0)
    finally:
        S.score_plan = orig
    return action, dbg, seeds


def main():
    gs = make_state(7)
    S, AS = reload_search(None)
    check('G0 default CRN on (E88 ship)', S.CRN is True)
    S, AS = reload_search('0')
    check('G1 CRN=0 off', S.CRN is False)
    safety = AS.action_safety(gs)
    a, dbg, seeds = seed_spy_call(S, gs, safety)
    check('G1 legal action', a is None or a in S._DIR_TO_ACTION.values()
          or a in ('WAIT', 'BOMB'), str(a))
    check('G1 per-plan seeds distinct',
          len(seeds) == dbg['plans'] and len(set(seeds)) == len(seeds),
          '%d plans / %d unique' % (len(seeds), len(set(seeds))))

    S, AS = reload_search('1')
    check('G2 CRN on', S.CRN is True)
    safety = AS.action_safety(gs)
    a1, dbg1, seeds1 = seed_spy_call(S, gs, safety)
    check('G2 one shared rollout seed',
          len(set(seeds1)) == 1, '%d plans / %d unique' % (
              len(seeds1), len(set(seeds1))))
    a2, dbg2, seeds2 = seed_spy_call(S, gs, safety)
    check('G3 CRN determinism action', a1 == a2, '%s vs %s' % (a1, a2))
    check('G3 CRN determinism plans/seeds',
          dbg1.get('plans') == dbg2.get('plans') and seeds1 == seeds2)

    # G4: CRN=0 score_plan reproduces the legacy formula exactly.
    import copy
    S0, AS0 = reload_search(None)
    from agent_code.arbiter.sim import from_game_state
    st0 = from_game_state(gs)
    planA = {'first': 'WAIT', 'prefix': ['WAIT'], 'bomb_at': None}
    planB = {'first': 'WAIT', 'prefix': ['WAIT', 'WAIT'], 'bomb_at': None}
    pA = S0.score_plan(st0, planA, seed=12345)
    pA2 = S0.score_plan(copy.deepcopy(st0), planA, seed=12345)
    check('G4 score_plan determinism', pA[0] == pA2[0], '%r vs %r' % (
        pA[0], pA2[0]))
    S1, AS1 = reload_search('1')
    st1 = from_game_state(gs)
    qA = S1.score_plan(st1, planA, seed=12345)
    qA2 = S1.score_plan(copy.deepcopy(st1), planA, seed=12345)
    check('G4 CRN score_plan determinism', qA[0] == qA2[0],
          '%r vs %r' % (qA[0], qA2[0]))

    if fails:
        print('FAILED:', fails)
        return 1
    print('ALL CRN PROBE GROUPS PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
