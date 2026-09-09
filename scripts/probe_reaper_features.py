#!/usr/bin/env python3
"""Reaper probes (E21 rule: nothing ships without a direct unit probe).

Checks:
  1. feature shape/dtype/finiteness/determinism
  2. symmetry equivariance: features(transform(s)) == apply_aug(features(s))
     for all 8 dihedral transforms (validates augmentation path exactly)
  3. action mapping consistency under the same transforms
  4. safety parity: reaper safety == overlord safety on identical states
  5. opp_can_escape sanity: trapped corridor -> False, open field -> True
  6. augmentation permutations are bijections
Run from repo root: python3 scripts/probe_reaper_features.py
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def make_random_state(rng, n_bombs=2, n_others=2, n_coins=4):
    W = H = 17
    arena = np.zeros((W, H), dtype=int)
    for x in range(W):
        for y in range(H):
            if (x + 1) * (y + 1) % 2 == 1:
                arena[x, y] = -1
    # full border walls, exactly like environment.build_arena
    arena[0, :] = -1
    arena[-1, :] = -1
    arena[:, 0] = -1
    arena[:, -1] = -1
    free = [(x, y) for x in range(1, W - 1) for y in range(1, H - 1)
            if arena[x, y] == 0 and (x, y) not in
            [(1, 1), (1, H - 2), (W - 2, 1), (W - 2, H - 2)]]
    for (x, y) in rng.choice(free, size=40, replace=False):
        arena[x, y] = 1
    arena[1, 1] = arena[1, H - 2] = arena[W - 2, 1] = arena[W - 2, H - 2] = 0
    arena[2, 1] = arena[1, 2] = arena[2, H - 2] = arena[1, H - 3] = 0
    arena[W - 3, 1] = arena[W - 2, 2] = arena[W - 3, H - 2] = arena[W - 2, H - 3] = 0
    exp_map = np.zeros((W, H))
    for (x, y) in rng.choice(free, size=5, replace=False):
        exp_map[x, y] = rng.integers(1, 3)
    bombs = []
    for (x, y) in rng.choice(free, size=n_bombs, replace=False):
        if arena[x, y] == 0:
            bombs.append(((int(x), int(y)), int(rng.integers(0, 5))))
    coins = [tuple(int(v) for v in c) for c in rng.choice(free, size=n_coins, replace=False)]
    others = [('opp', int(rng.integers(0, 8)), bool(rng.integers(0, 2)),
               tuple(int(v) for v in rng.choice(free)))
              for _ in range(n_others)]
    self_pos = tuple(int(v) for v in rng.choice(free))
    while self_pos in [b[0] for b in bombs]:
        self_pos = tuple(int(v) for v in rng.choice(free))
    state = {
        'round': 1, 'step': int(rng.integers(1, 400)),
        'field': arena,
        'self': ('reaper', 0, bool(rng.integers(0, 2)), self_pos),
        'others': others,
        'bombs': bombs,
        'coins': coins,
        'explosion_map': exp_map,
        'user_input': None,
    }
    return state


def main():
    from agent_code.reaper.features import (state_to_features, FEATURE_DIM,
                                            apply_aug, map_action, transform_state,
                                            SYMS, N_SYMS, AUG_PERMS)
    from agent_code.reaper import safety as rs
    from agent_code.overlord import safety as osafety
    from agent_code.reaper.model import build_model, ACTION_LIST

    rng = np.random.default_rng(42)
    n_checks = 0

    # 1. shape/dtype/finite/deterministic
    for i in range(20):
        st = make_random_state(rng)
        s = rs.action_safety(st)
        f1 = state_to_features(st, s)
        f2 = state_to_features(st, s)
        assert f1.shape == (FEATURE_DIM,), f1.shape
        assert f1.dtype == np.float32
        assert np.isfinite(f1).all()
        assert np.array_equal(f1, f2)
    print('ok: shape/dtype/finiteness/determinism (20 states)')
    n_checks += 1

    # 6. permutations bijective
    for row in AUG_PERMS:
        assert len(set(row)) == FEATURE_DIM
    print('ok: augmentation permutations are bijections')
    n_checks += 1

    # 2+3. equivariance + action mapping
    for i in range(15):
        st = make_random_state(rng)
        s = rs.action_safety(st)
        own = None
        if st['self'][2] is False and st['bombs']:
            own = st['bombs'][0][0]
        f0 = state_to_features(st, s, own)
        for sym in range(N_SYMS):
            tst = transform_state(st, sym)
            ts = rs.action_safety(tst)
            town = None
            if own is not None:
                A, b = SYMS[sym]
                nx, ny = A @ np.array(own) + np.array(b)
                town = (int(nx), int(ny))
            ft = state_to_features(tst, ts, town)
            fp = apply_aug(f0, sym)
            if not np.allclose(ft, fp, atol=1e-5):
                bad = np.where(~np.isclose(ft, fp, atol=1e-5))[0]
                raise AssertionError(
                    f'sym {sym} state {i} mismatch at feats {bad[:8]}: '
                    f't={ft[bad[:4]]} p={fp[bad[:4]]}')
            # action mapping: BFS first-step of a move maps under the same delta rule
            A, b = SYMS[sym]
            for a in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
                d = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}[a]
                nd = tuple(int(v) for v in (A @ np.array(d)))
                inv = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
                assert map_action(a, sym) == inv[nd]
            assert map_action('WAIT', sym) == 'WAIT'
            assert map_action('BOMB', sym) == 'BOMB'
    print('ok: symmetry equivariance of features + action mapping (15 states x 8 syms)')
    n_checks += 1

    # 4. safety parity with overlord
    for i in range(15):
        st = make_random_state(rng)
        rs_out = rs.action_safety(st)
        os_out = osafety.action_safety(st)
        assert rs_out['valid'] == os_out['valid'], (rs_out['valid'], os_out['valid'])
        assert rs_out['safe'] == os_out['safe'], (rs_out['safe'], os_out['safe'])
    print('ok: reaper safety parity with overlord (15 states)')
    n_checks += 1

    # 5. opp_can_escape sanity
    W = H = 17
    arena = np.zeros((W, H), dtype=int)
    arena[0, :] = arena[-1, :] = arena[:, 0] = arena[:, -1] = -1
    # trapped corridor: opp at (2,2) with walls around except one exit that
    # the hypothetical bomb covers
    arena[2, 1] = -1
    arena[1, 2] = -1
    arena[3, 2] = -1
    arena[2, 3] = -1
    can, dist = rs.opp_can_escape(arena, [], (2, 2), (2, 2))
    assert can is False, 'opponent with no exits must not escape'
    # open field
    arena2 = np.zeros((W, H), dtype=int)
    arena2[0, :] = arena2[-1, :] = arena2[:, 0] = arena2[:, -1] = -1
    can2, dist2 = rs.opp_can_escape(arena2, [], (8, 8), (2, 2))
    assert can2 is True and dist2 < float('inf')
    print('ok: opp_can_escape trapped=False / open=True')
    n_checks += 1

    # model smoke: shapes + zero-head Q start
    m = build_model()
    q = m(torch_zeros(4, FEATURE_DIM))
    assert q.shape == (4, 6)
    assert float(q.abs().max()) == 0.0
    print('ok: model shapes + zero-init heads')
    n_checks += 1

    # 7. vectorized helpers == brute force (E37/P1 latency rewrite)
    from agent_code.reaper.features import (_blast_tiles, _blast_crate_counts,
                                            _blast_hits_opp_mask)
    from agent_code.reaper.safety import future_danger, first_lethal
    for trial in range(10):
        st = make_random_state(rng)
        ar = np.asarray(st['field'])
        W2, H2 = ar.shape
        mask, counts = _blast_crate_counts(ar)
        floor2 = [(x, y) for x in range(W2) for y in range(H2)
                  if ar[x, y] == 0]
        for x in range(W2):
            for y in range(H2):
                exp = sum(1 for (bx, by) in _blast_tiles(ar, x, y)
                          if ar[bx, by] == 1)
                assert counts[x, y] == exp, (trial, x, y)
        exp_adj = set()
        for x in range(W2):
            for y in range(H2):
                if ar[x, y] == 1:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W2 and 0 <= ny < H2 \
                                and ar[nx, ny] == 0:
                            exp_adj.add((nx, ny))
        assert set(map(tuple, np.argwhere(mask).tolist())) == exp_adj
        for (ox, oy) in [floor2[i] for i in
                         rng.choice(len(floor2), size=3, replace=False)]:
            om = _blast_hits_opp_mask(ar, ox, oy)
            for (x, y) in floor2:
                assert bool(om[x, y]) == ((ox, oy) in _blast_tiles(ar, x, y))
        d = future_danger(ar, st['bombs'], np.asarray(st['explosion_map']))
        fl = first_lethal(d)
        for x in range(W2):
            for y in range(H2):
                exp = next((t for t in range(d.shape[0]) if d[t, x, y]),
                           d.shape[0])
                assert fl[x, y] == exp, (trial, x, y)
    print('ok: vectorized blast/first-lethal == brute force (10 arenas)')
    n_checks += 1

    # 8. kill-block correctness (E37/P2): constructed trap + cross-checks
    from agent_code.reaper.features import _adj_kill_info, _blast_tiles as _bt
    W = H = 17
    arena = np.zeros((W, H), dtype=int)
    arena[0, :] = arena[-1, :] = arena[:, 0] = arena[:, -1] = -1
    # dead-end pocket at (5,5): only exit is (5,6)
    arena[4, 5] = arena[6, 5] = arena[5, 4] = -1
    st = {'round': 1, 'step': 100, 'field': arena,
          'self': ('reaper', 0, True, (5, 7)),
          'others': [('opp', 0, True, (5, 5))],
          'bombs': [], 'coins': [],
          'explosion_map': np.zeros((W, H)), 'user_input': None}
    s = rs.action_safety(st)
    f = state_to_features(st, s, None)
    # (5,6) is UP from (5,7) = DELTAS[0]; bomb there covers (5,5) whose
    # only exit is the bomb tile itself -> trapped
    assert f[68] == 1.0, f'trap UP, got {f[68]}'
    assert f[69] == 0.0 and f[70] == 0.0 and f[71] == 0.0
    assert f[63] == 1.0, 'scalar trap must mirror the directional mask'
    assert f[72] == 0.5, f'opps_hit UP, got {f[72]}'   # 1 opp / 2
    # DOWN (5,8) also covers (5,5) at range 3, but the opponent escapes
    # sideways out of that blast -> hit without trap
    assert f[73] == 0.5, f'opps_hit DOWN, got {f[73]}'
    assert f[69] == 0.0, 'DOWN must be hit-but-escapable, not a trap'
    assert f[74] == 0.0 and f[75] == 0.0
    # open floor all around -> every escape margin feasible
    assert all(v < 1.0 for v in f[80:84]), f'esc {f[80:84]}'
    assert f[91] < 1.0, 'min opp escape distance must be finite'
    assert f[93] == 1.0, 'PHI_kill must be 1 on a realized trap'
    assert f[84] == 1.0 and f[86] == 0.0, 'open self tile mobility'
    assert 0.0 < f[85] <= 1.0, 'reachable area must be positive'
    # same pocket but opponent already has an open exit -> no trap
    arena2 = arena.copy()
    arena2[6, 5] = 0
    st2 = dict(st, field=arena2)
    f2 = state_to_features(st2, rs.action_safety(st2), None)
    assert f2[68] == 0.0 and f2[63] == 0.0, 'open exit must clear the trap'
    assert f2[93] < 1.0, 'PHI_kill must drop without a trap'
    # cross-check _adj_kill_info crates/opps against brute force
    for i in range(10):
        stx = make_random_state(rng)
        arx = np.asarray(stx['field'])
        xx, yy = int(stx['self'][3][0]), int(stx['self'][3][1])
        oxx = [(int(a), int(b)) for (_, _, _, (a, b)) in stx['others']]
        _bset = set((int(a), int(b)) for ((a, b), _) in stx['bombs'])
        _oset = set(oxx)
        _, _bc = _blast_crate_counts(arx)
        tr, op, cr, es, moe = _adj_kill_info(
            arx, stx['bombs'], oxx, xx, yy, _bc, full=True)
        assert len(tr) == len(op) == len(cr) == len(es) == 4
        for d, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx, ny = xx + dx, yy + dy
            if not (0 <= nx < 17 and 0 <= ny < 17) or arx[nx, ny] != 0 \
                    or (nx, ny) in _bset or (nx, ny) in _oset:
                assert es[d] == 1.0, (i, d)
                assert cr[d] == 0.0 and op[d] == 0.0 and tr[d] == 0.0
                continue
            exp_c = sum(1 for (bx, by) in _bt(arx, nx, ny)
                        if arx[bx, by] == 1)
            assert abs(cr[d] - min(exp_c, 4) / 4.0) < 1e-6, (i, d)
            exp_o = sum(1 for (ox, oy) in oxx
                        if (ox, oy) in _bt(arx, nx, ny))
            assert abs(op[d] - min(exp_o, 2) / 2.0) < 1e-6, (i, d)
            assert set(tr) <= {0.0, 1.0} and set(es[d:d + 1]) <= {es[d]}
            assert 0.0 <= es[d] <= 1.0
    print('ok: kill-block trap/mask/crates/opps (constructed + 10 states)')
    n_checks += 1

    # 9. tactical overlay (E37/P5): exact kill proof + BOMB override
    from agent_code.reaper.safety import bomb_here_traps
    from agent_code.reaper import callbacks as rc
    # self ON the corridor exit (5,6): our bomb covers (5,5) whose only
    # exit is the bomb tile -> forced kill.
    st3 = dict(st, self=('reaper', 0, True, (5, 6)))
    s3 = rs.action_safety(st3)
    assert s3['safe']['BOMB'] is True
    traps, _ = bomb_here_traps(st3, s3)
    assert traps is True, 'bomb on the exit must trap the pocketed opp'
    # self one tile further out (5,7): the opponent escapes sideways out
    # of our blast -> no forced kill (checked in the timing harness).
    traps0, _ = bomb_here_traps(st, s)
    assert traps0 is False
    # override: zero-Q policy misses the kill, tactical takes it.
    import torch as _tt
    from types import SimpleNamespace as _SN
    from collections import deque as _dq
    _tt.set_num_threads(1)

    def _mkself():
        m = _SN(train=False, epsilon=0.0, coord_history=_dq([], 24),
                bomb_history=_dq([], 5), current_round=1, flee_timer=0,
                own_bomb=None)
        m.model = build_model()
        m.model.eval()
        return m
    rc.SEARCH = 'off'
    a_off = rc.act(_mkself(), dict(st3))
    rc.SEARCH = 'tactical'
    a_tac = rc.act(_mkself(), dict(st3))
    rc.SEARCH = 'off'
    assert a_tac == 'BOMB', f'tactical must take the forced kill, got {a_tac}'
    assert a_off != 'BOMB', 'zero-Q baseline must miss it (else no test)'
    print('ok: tactical bomb_here_traps proof + BOMB override')
    n_checks += 1

    print(f'ALL {n_checks} PROBE GROUPS PASSED')


def torch_zeros(*shape):
    import torch
    return torch.zeros(*shape)


if __name__ == '__main__':
    main()
