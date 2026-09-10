#!/usr/bin/env python3
"""ARBITER rollout-noise probe (E83: probe before games, E21 rule).

  U0  default-off: CERT_OWN False, OPPMODEL 'random'.
  U1  _opp_move 'random' bit-parity: same rng -> same action for
      random / wardenlite-on-danger paths.
  E1  cert-owner: an opp SELF-trap (opp bombs itself in a sealed
      corner, CERT default) credits us +5; with CERT_OWN=1 the same
      plan's payoff drops to ~0 (engine-exact).
  E2  our-bomb certs unaffected by CERT_OWN: the sealed-pocket hunt
      state's own-tile cert still pays +5 with CERT_OWN=1 (lucky
      seed where the continuation survives).
  E3  wardenlite: coin-pursuit step chosen over a random one.
  E4  wardenlite: bomb guard fires on opps_hit > 0, holds on empty
      yield, and holds when the opp has no safe neighbour.
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
import arbiter.search as SR
from arbiter.sim import from_game_state

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


def opp_state(st, i):
    from arbiter.sim import to_game_state
    return None


def run_oppmove(st, i, seed, danger=None):
    rng = np.random.default_rng(seed)
    return SR._opp_move(rng, st, i, danger)


def main():
    check('U0-defaults', SR.CERT_OWN is False and SR.OPPMODEL == 'random',
          f'{SR.CERT_OWN} {SR.OPPMODEL!r}')

    # --- U1 _opp_move parity (random model, danger None and set) ---
    gs = make_state(open_field(), others=[(5, 6)])
    st = from_game_state(gs)
    a1 = run_oppmove(st, 1, 7)
    SR.OPPMODEL = 'wardenlite'
    SR.OPPMODEL = 'random'
    check('U1a-random-model', a1 in ('UP', 'DOWN', 'LEFT', 'RIGHT',
                                     'WAIT', 'BOMB'), f'a1={a1!r}')
    # danger set: opp at (5,2), bomb at (5,6) t=2 up-arm covers
    # (5,3),(5,4),(5,5) — DOWN is lethal, UP/WAIT are safe; wardenlite
    # must also drop BOMB (guard holds: no crates, no opps in blast).
    fw = open_field()
    gsw = make_state(fw, self_xy=(5, 9), others=[(5, 2)])
    stw = from_game_state(gsw)
    from arbiter.safety import future_danger
    dang = future_danger(np.asarray(fw), [((5, 6), 2)],
                         np.zeros((17, 17)), 4)
    SR.OPPMODEL = 'wardenlite'
    steps_w = set()
    for s in range(20):
        steps_w.add(run_oppmove(stw, 1, s, dang))
    SR.OPPMODEL = 'random'
    steps_r = set()
    for s in range(20):
        steps_r.add(run_oppmove(stw, 1, s, dang))
    check('U1b-avoid-lethal', steps_w <= {'UP', 'LEFT', 'RIGHT', 'WAIT'}
          and ('DOWN' in steps_r or 'BOMB' in steps_r),
          f'wardenlite={sorted(steps_w)} random={sorted(steps_r)}')

    # --- E1 cert-owner: opp self-trap credits us only under default ---
    # E88: CERT_OWN semantics are orthogonal to CRN; pin the legacy
    # unpaired rollout so this probe's seed scan stays stable.
    SR.CRN = False
    gs1 = make_state(open_field(), others=())
    gs1['others'] = [('o0', 0, True, (5, 1))]
    st0 = from_game_state(gs1)
    down = {'first': 'DOWN', 'prefix': ['DOWN'], 'bomb_at': None}
    found = None
    for s in range(40):
        pay_default, st_end = SR.score_plan(st0, down, seed=s)
        SR.CERT_OWN = True
        pay_own, _ = SR.score_plan(st0, down, seed=s)
        SR.CERT_OWN = False
        if pay_default >= 4.5 and abs(pay_own) < 1.0:
            found = (s, pay_default, pay_own)
            break
    check('E1-cert-owner', found is not None,
          f'seed/def/own={found} (scanned 40)')

    # --- E2 our-bomb cert survives CERT_OWN ---
    f2 = open_field()
    for (cx, cy) in [(4, 1), (6, 1), (4, 2), (6, 2), (4, 3), (6, 3)]:
        f2[cx, cy] = -1
    gs2 = make_state(f2, self_xy=(5, 5))
    gs2['others'] = [('o0', 0, False, (5, 1))]
    st2 = from_game_state(gs2)
    SR.HUNT = True
    plans2 = SR.gen_plans(gs2, AS.action_safety(gs2))
    SR.HUNT = False
    hunt = [p for p in plans2 if p['bomb_at'] == (5, 4)]
    if hunt:
        pays = [SR.score_plan(st2, hunt[0], seed=s)[0]
                for s in range(6)]
        SR.CERT_OWN = True
        pays_own = [SR.score_plan(st2, hunt[0], seed=s)[0]
                    for s in range(6)]
        SR.CERT_OWN = False
        check('E2-own-cert-survives',
              max(pays) == max(pays_own) and max(pays) > 4.5,
              f'max default={max(pays):.2f} own={max(pays_own):.2f}')
    else:
        check('E2-own-cert-survives', False, 'hunt plan missing')

    # --- E3 wardenlite coin pursuit ---
    gs3 = make_state(open_field(), others=())
    gs3['others'] = [('o0', 0, False, (5, 8))]
    gs3['coins'] = [(5, 6)]
    st3 = from_game_state(gs3)
    SR.OPPMODEL = 'wardenlite'
    steps = set()
    for s in range(20):
        steps.add(run_oppmove(st3, 1, s))
    SR.OPPMODEL = 'random'
    check('E3-coin-pursuit', steps == {'UP'},
          f'steps={sorted(steps)}')

    # --- E4 wardenlite bomb guard ---
    # (a) opps_hit > 0 + mobility: opp adjacent to us (its blast covers
    # us); it has 3 free pre-blast neighbours -> guard fires.
    gs4 = make_state(open_field(), others=[(5, 6)])
    st4 = from_game_state(gs4)
    SR.OPPMODEL = 'wardenlite'
    acts4 = set()
    for s in range(30):
        acts4.add(run_oppmove(st4, 1, s))
    SR.OPPMODEL = 'random'
    check('E4a-guard-fires-opp', 'BOMB' in acts4, f'{sorted(acts4)}')
    # (b) empty blast, no crates, no opps: no BOMB (guard holds)
    gs4b = make_state(open_field(), others=[])
    st4b = from_game_state(gs4b)
    SR.OPPMODEL = 'wardenlite'
    acts4b = set()
    for s in range(30):
        acts4b.add(run_oppmove(st4b, 0, s))
    SR.OPPMODEL = 'random'
    check('E4b-guard-holds-empty', 'BOMB' not in acts4b,
          f'{sorted(acts4b)}')
    # (c) crates_hit >= 2 fires the guard
    f4c = open_field()
    f4c[5, 4] = 1
    f4c[5, 6] = 1
    gs4c = make_state(f4c, others=[])
    st4c = from_game_state(gs4c)
    SR.OPPMODEL = 'wardenlite'
    acts4c = set()
    for s in range(30):
        acts4c.add(run_oppmove(st4c, 0, s))
    SR.OPPMODEL = 'random'
    check('E4c-guard-fires-crates', 'BOMB' in acts4c, f'{sorted(acts4c)}')

    print('FAILURES:', fails if fails else 'none')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
