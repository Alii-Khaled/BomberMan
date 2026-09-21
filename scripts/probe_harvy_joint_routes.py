#!/usr/bin/env python3
"""Focused probes for E134's dynamic opponent-route bomb evaluator."""

import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agent_code.Harvy import safety as S


def game(arena, self_pos, opponents):
    return {
        'field': np.asarray(arena, dtype=int),
        'self': ('Harvy', 0, True, self_pos),
        'others': [('opp%d' % i, 0, True, pos)
                   for i, pos in enumerate(opponents)],
        'bombs': [], 'coins': [],
        'explosion_map': np.zeros_like(arena, dtype=int),
        'step': 1, 'round': 1,
    }


def open_board(n=7):
    arena = np.zeros((n, n), dtype=int)
    arena[0, :] = arena[-1, :] = -1
    arena[:, 0] = arena[:, -1] = -1
    return arena


def main():
    # A currently occupied exit is not a permanent wall.  The old static
    # blocker certificate says "dead"; the dynamic route profile finds the
    # route that opens after the blocker moves.
    ar = np.full((7, 7), -1, dtype=int)
    for pos in [(3, 1), (3, 2), (3, 3), (4, 3), (4, 4), (4, 5),
                (5, 3), (5, 4), (5, 5)]:
        ar[pos] = 0
    danger = S.with_hypothetical_bomb(
        S.future_danger(ar, [], None, 6), ar, 3, 1, 6, 4, 3)
    bombs = [((3, 1), 4)]
    static, _ = S.escape_bfs((3, 3), ar, bombs, [(4, 3)], danger, 6)
    dynamic = S.escape_route_profile((3, 3), ar, bombs, danger, 6)
    assert not any(static.values()), 'fixture must reproduce false static trap'
    assert dynamic['survives'] and dynamic['routes'] > 0

    # A genuinely sealed target still has zero routes, so conservative kill
    # certification remains possible.
    sealed = np.full((7, 7), -1, dtype=int)
    for pos in [(3, 1), (3, 2), (3, 3)]:
        sealed[pos] = 0
    danger2 = S.with_hypothetical_bomb(
        S.future_danger(sealed, [], None, 6), sealed, 3, 1, 6, 4, 3)
    prof2 = S.escape_route_profile(
        (3, 3), sealed, [((3, 1), 4)], danger2, 6)
    assert not prof2['survives'] and prof2['routes'] == 0

    # Open space must not be over-vetoed.
    g_open = game(open_board(), (3, 3), [(5, 3)])
    old_mode = (S.JOINT_ROUTES, S.DYNAMIC_ROUTES, S.BODYBLOCK)
    S.JOINT_ROUTES = S.DYNAMIC_ROUTES = S.BODYBLOCK = True
    sf_open = S.action_safety(g_open, horizon=6)
    assert sf_open['safe']['BOMB']
    assert sf_open['joint_routes']['own_robust']

    # Reproduce a connected close-body-block state: legacy safety admits the
    # plant, while the minimax route evaluator finds no robust escape policy.
    body = np.asarray([
        [-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1, 0, 0, 0, 0, 0, 0,-1,-1],
        [-1, 0, 0,-1, 0,-1, 0,-1,-1],
        [-1, 0, 0,-1, 0, 0,-1, 0,-1],
        [-1,-1,-1, 0,-1,-1, 0, 0,-1],
        [-1, 0, 0, 0, 0,-1, 0, 0,-1],
        [-1,-1,-1, 0, 0, 0,-1, 0,-1],
        [-1, 0,-1, 0, 0,-1, 0,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1],
    ], dtype=int)
    g_body = game(body, (7, 3), [(5, 2)])
    S.JOINT_ROUTES = S.DYNAMIC_ROUTES = S.BODYBLOCK = False
    base = S.action_safety(g_body, horizon=6)
    S.JOINT_ROUTES = S.DYNAMIC_ROUTES = S.BODYBLOCK = True
    guarded = S.action_safety(g_body, horizon=6)
    assert base['safe']['BOMB'], 'legacy gate must admit fixture plant'
    assert not guarded['safe']['BOMB']
    assert not guarded['joint_routes']['own_robust']

    # Timing guard: the extra work is only reached for mask-safe plants and
    # must remain far below the engine's 0.5-second action limit.
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        S.joint_bomb_analysis(g_open, 6)
        samples.append((time.perf_counter() - t0) * 1000.0)
    med = float(np.median(samples))
    p95 = float(np.percentile(samples, 95))
    assert p95 < 50.0, 'joint route evaluator too slow: p95 %.2fms' % p95
    S.JOINT_ROUTES, S.DYNAMIC_ROUTES, S.BODYBLOCK = old_mode
    print('PASS joint routes: dynamic-exit, sealed-kill, open-safe, '
          'body-block-veto; median %.2fms p95 %.2fms' % (med, p95))


if __name__ == '__main__':
    main()
