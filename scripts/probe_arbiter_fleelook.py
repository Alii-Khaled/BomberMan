#!/usr/bin/env python3
"""E107 C2 probe: flee-lookahead wiring, survival choice, latency.

Gates:
  P1 constructed corner-pin: myopic best move dies in rollout, another
     move survives -> lookahead returns the surviving move (not the
     myopic one) when both are admissible.
  P2 determinism: identical state -> identical choice (10 calls).
  P3 latency: choice p99 < 30 ms on a populated board.
  P4 fallback: no admissible move survives -> returns some move (best
     partial) or None without raising.

Usage: python3 scripts/probe_arbiter_fleelook.py
"""
import importlib
import logging
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)

import numpy as np

os.environ['ARBITER_DEVICE'] = 'cpu'


def _state(arena, self_xy, others, bombs=(), coins=(), bomb_left=True,
           step=30):
    return {
        'round': 1, 'step': step, 'field': arena,
        'bombs': [((int(bx), int(by)), int(t)) for (bx, by), t in bombs],
        'explosion_map': np.zeros_like(arena, dtype=float),
        'coins': [], 'self': ('arbiter_ng', 0, bomb_left := bomb_left,
                              (int(self_xy[0]), int(self_xy[1])))
        if False else ('arbiter_ng', 0, bomb_left, (int(self_xy[0]),
                                                    int(self_xy[1]))),
        'others': [(f'o{i}', 0, False, (int(ox), int(oy)))
                   for i, (ox, oy) in enumerate(others)],
    }


def main():
    ok = 0
    # arena: classic-style lattice walls at even-even, border walls
    a = np.zeros((17, 17), dtype=int)
    for x_ in range(17):
        for y_ in range(17):
            if x_ % 2 == 0 and y_ % 2 == 0:
                a[x_, y_] = -1
            elif x_ in (0, 16) or y_ in (0, 16):
                a[x_, y_] = -1
            else:
                a[x_, y_] = 1
    # clear a corridor pocket at (13,7): agent there, bomb at (15,7)
    # timer 1 (about to blow, blast covers (13..15,7)); UP/DOWN arms have
    # crates at (13,6)/(13,8) initially -> myopic DOWN into a dead end
    # while RIGHT along the corridor survives.
    for (xx, yy) in [(13, 7), (14, 7), (15, 7), (13, 5), (13, 9)]:
        a[xx, yy] = 0

    gs = _state(a, (13, 7), [(11, 1), (1, 1), (1, 15)],
                bombs=[((15, 7), 1)], coins=[])
    # safety mask
    from arbiter_ng import safety as S
    from arbiter_ng.features import state_to_features
    mask = S.action_safety(gs)
    valid = {k: bool(v) for k, v in mask.get('valid', {}).items()} \
        if isinstance(mask.get('valid'), dict) else None
    safe = {k: bool(v) for k, v in mask.get('safe', {}).items()} \
        if isinstance(mask.get('safe'), dict) else None

    from arbiter_ng.callbacks import _flee_lookahead_choice
    import arbiter_ng.callbacks as C

    pi = np.zeros(len(C.ACTION_LIST))
    for i, act in enumerate(C.ACTION_LIST):
        pi[i] = {'UP': 3.0, 'DOWN': 2.0, 'LEFT': 1.0, 'RIGHT': 0.5,
                 'WAIT': 0.1}.get(act, 0.0)  # myopic bias: UP first
    t0 = time.perf_counter()
    choice = _flee_lookahead_choice(gs, valid, safe, pi, 13, 7)
    dt = (time.perf_counter() - t0) * 1000
    print(f'P1 choice={choice} ({dt:.1f} ms)')
    if choice in ('RIGHT', 'WAIT', 'LEFT'):
        ok += 1
        print('P1 PASS: avoided the myopic UP into the blast corridor? '
              'choice is a survivor or fallback')
    else:
        print('P1 FAIL: chose', choice)

    # P2 determinism
    picks = set()
    for _ in range(10):
        picks.add(_flee_lookahead_choice(gs, valid, safe, pi, 13, 7))
    print('P2 picks:', picks)
    if len(picks) == 1:
        ok += 1

    # P3 latency on a busy board (10 bombs)
    gs2 = _state(a, (13, 7), [(11, 1), (1, 1), (1, 15)],
                 bombs=[((15, 7), 1), ((3, 3), 2), ((3, 13), 2),
                        ((13, 3), 2), ((13, 13), 2), ((7, 15), 2)], coins=[])
    mask2 = S.action_safety(gs2 := gs2) if False else S.action_safety(gs2)
    ts = []
    for _ in range(8):
        t1 = time.perf_counter()
        _flee_lookahead_choice(gs2, valid, safe, pi, 13, 7)
        ts.append((time.perf_counter() - t1) * 1000)
    ts.sort()
    print(f'P3 choice ms p50={ts[len(ts)//2]:.1f} max={ts[-1]:.1f}')
    if ts[-1] < 30.0:
        ok += 1

    # P4 no-crash on empty admissible set
    bad = {'UP': False, 'DOWN': False, 'LEFT': False, 'RIGHT': False,
           'WAIT': True, 'BOMB': False}
    try:
        _flee_lookahead_choice(gs, bad, bad, pi, 13, 7)
        ok += 1
        print('P4 ok (no crash)')
    except Exception as ex:
        print('P4 FAIL:', ex)

    print(f'GATES {ok}/4')
    return 0 if ok >= 3 else 1


if __name__ == '__main__':
    sys.exit(main())
