# Warden v2 — bounded rollout search (heuristic leaf, exact dynamics).
#
# Root candidates: the valid moves + bomb plans (own tile and BFS-reachable
# high-yield tiles). Each plan is scored by rolling the exact simulator
# forward with a cheap greedy continuation/opponent policy, paired across
# plans (CRN: opponent/continuation draws keyed by (rollout, tick, agent)
# so all plans see the same randomness). Leaf value uses exact events plus
# hand-crafted mobility/margin/coin terms — no learned network.
#
# The fast warden heuristic remains the fallback whenever the wall-clock
# budget (WARDEN_TIME_BUDGET, default 0.30 s) is exhausted or the search
# raises.

import time

import numpy as np

from . import safety as S
from . import sim as SIM


def _env(name, default, cast, lo=None, hi=None):
    import os
    try:
        v = cast(os.environ.get(name, str(default)))
    except Exception:
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


BUDGET = _env('WARDEN_TIME_BUDGET', 0.30, float, 0.02, 0.45)
H = _env('WARDEN_SEARCH_H', 10, int, 6, 16)
RADIUS = _env('WARDEN_SEARCH_R', 3, int, 1, 6)
PLANS = _env('WARDEN_SEARCH_PLANS', 12, int, 4, 32)
SEEDS = _env('WARDEN_SEARCH_SEEDS', 2, int, 1, 4)
MARGIN = _env('WARDEN_BOMB_MARGIN', 0.5, float)
ESC_DIRS = _env('WARDEN_PLANT_ESC', 1, int, 1, 4)

W_CRATE = _env('WARDEN_W_CRATE', 1.0, float)
W_COIN = _env('WARDEN_W_COIN', 1.0, float)
W_KILL = _env('WARDEN_W_KILL', 2.5, float)
W_REVEAL = _env('WARDEN_W_REVEAL', 1.0, float)
W_DEATH = _env('WARDEN_W_DEATH', 8.0, float)
W_MARGIN = _env('WARDEN_W_MARGIN', 0.6, float)
W_MOB = _env('WARDEN_W_MOB', 0.12, float)
W_ALIVE = _env('WARDEN_W_ALIVE', 0.8, float)

_DIR4 = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')


def _clone(simst):
    return {
        'arena': simst['arena'].copy(),
        'coins': [list(c) for c in simst['coins']],
        'agents': [dict(a) for a in simst['agents']],
        'bombs': [list(b) for b in simst['bombs']],
        'explosions': [{'coords': list(e['coords']), 'timer': e['timer'],
                        'stage': e['stage'], 'owner': e['owner']}
                       for e in simst['explosions']],
        'step': simst.get('step', 0),
        'total_coins': simst.get('total_coins', 9),
    }


def _sim_danger(simst):
    arena = simst['arena']
    expm = np.zeros(arena.shape, dtype=float)
    for e in simst['explosions']:
        if e['stage'] == 0:
            for (cx, cy) in e['coords']:
                expm[cx, cy] = max(expm[cx, cy], max(0, e['timer'] - 1))
    bombs = [((b[0], b[1]), b[2]) for b in simst['bombs'] if b[0] is not None]
    return S.future_danger(arena, bombs, expm, H)


def _nearest_target_dir(x, y, targets):
    best, bd = None, 10 ** 9
    for (tx, ty) in targets:
        d = abs(tx - x) + abs(ty - y)
        if d < bd:
            bd, best = d, (tx, ty)
    if best is None:
        return None
    tx, ty = best
    if abs(tx - x) >= abs(ty - y):
        return (1 if tx > x else -1, 0)
    return (0, 1 if ty > y else -1)


def _greedy_action(simst, i, danger, coins, rng, allow_bomb=True):
    """Cheap continuation/opponent policy: survive, chase coins, keep space."""
    arena = simst['arena']
    a = simst['agents'][i]
    if not a['alive']:
        return 'WAIT'
    W, Hh = arena.shape[0], arena.shape[1]
    act = SIM.valid_actions(simst, i)
    x, y = a['x'], a['y']
    others = [(a2['x'], a2['y']) for j, a2 in enumerate(simst['agents'])
              if j != i and a2['alive']]
    tstep = _nearest_target_dir(x, y, coins or others)

    best, bs = None, -1e9
    for action in act:
        if action == 'BOMB':
            continue
        dx, dy = SIM._DELTAS[action]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < Hh):
            continue
        lethal01 = danger[0, nx, ny] or danger[1, nx, ny]
        lethal2 = danger[2, nx, ny] if danger.shape[0] > 2 else False
        nfree = 0
        for ddx, ddy in _DIR4:
            mx, my = nx + ddx, ny + ddy
            if 0 <= mx < W and 0 <= my < Hh and arena[mx, my] == 0:
                nfree += 1
        sc = 0.0
        if not lethal01:
            sc += 3.0
        if not lethal2:
            sc += 1.0
        sc += W_MOB * min(4, nfree)
        if tstep is not None and (dx, dy) == tstep:
            sc += 1.0
        if action == 'WAIT':
            sc -= 0.4
        sc += 0.05 * rng.random()
        if sc > bs:
            bs, best = sc, action
    if best is None:
        best = 'WAIT'
    if allow_bomb and a['bombs_left']:
        blast = set(SIM.blast_coords(arena, x, y))
        hits = sum(1 for o in others if o in blast)
        if hits > 0:
            safe_move = any((not (danger[0, x + dx, y + dy]
                                   or danger[1, x + dx, y + dy]))
                            for dx, dy in _DIR4
                            if 0 <= x + dx < W and 0 <= y + dy < Hh
                            and arena[x + dx, y + dy] == 0)
            if safe_move and rng.random() < 0.5:
                return 'BOMB'
    return best


def _certified_kills(arena, bombs, bomb_tile, others_xy, danger_ne):
    blast = set(S.true_blast(arena, bomb_tile[0], bomb_tile[1]))
    n = 0
    for o in others_xy:
        if o in blast:
            can, _ = S.opp_can_escape(arena, bombs, o, bomb_tile,
                                      others_xy, danger=danger_ne)
            if not can:
                n += 1
    return n


def _rollout(simst0, prefix, rng_key, j, margin0):
    st_sim = _clone(simst0)
    n_agents = len(st_sim['agents'])
    crates = coins = kills = reveal = 0.0
    died = 0.0
    for t in range(H):
        if not st_sim['agents'][0]['alive']:
            break
        danger = _sim_danger(st_sim)
        coin_xy = [(c[0], c[1]) for c in st_sim['coins'] if c[2]]
        actions = ['WAIT'] * n_agents
        for i in range(n_agents):
            key = (rng_key ^ (j * 2654435761) ^ (t * 40503)
                   ^ (i * 2246822519)) & 0xffffffff
            rng = np.random.default_rng(key)
            if i == 0 and t < len(prefix):
                actions[0] = prefix[t]
            else:
                actions[i] = _greedy_action(st_sim, i, danger, coin_xy, rng,
                                            allow_bomb=(i != 0))
        info = SIM.step(st_sim, actions)
        crates += info['crates']
        coins += info['coins'][0]
        kills += info['kills'][0]
        reveal += info['revealed_exp']
        if info['died'][0]:
            died = 1.0
            break
    me = st_sim['agents'][0]
    opp_scores = [a['score'] for a in st_sim['agents'][1:]]
    margin_now = me['score'] - (max(opp_scores) if opp_scores else 0)
    mobility = 0
    if me['alive']:
        W, Hh = st_sim['arena'].shape[0], st_sim['arena'].shape[1]
        for dx, dy in _DIR4:
            nx, ny = me['x'] + dx, me['y'] + dy
            if 0 <= nx < W and 0 <= ny < Hh and st_sim['arena'][nx, ny] == 0:
                mobility += 1
    return (W_CRATE * crates + W_COIN * coins + W_KILL * kills
            + W_REVEAL * reveal - W_DEATH * died
            + W_MARGIN * (margin_now - margin0)
            + W_MOB * mobility + W_ALIVE * (1.0 if me['alive'] else 0.0))


def _bfs_path(arena, blocked, start, goal, limit=12):
    from collections import deque
    W, Hh = arena.shape[0], arena.shape[1]
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if start == goal:
        return []
    prev = {start: None}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        if len(prev) > 400:
            break
        for dx, dy in _DIR4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < Hh):
                continue
            if (nx, ny) in prev:
                continue
            if arena[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            prev[(nx, ny)] = (cx, cy)
            if (nx, ny) == goal:
                q.clear()
                break
            q.append((nx, ny))
    if goal not in prev:
        return None
    acts, cur = [], goal
    _d2a = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT',
            (1, 0): 'RIGHT'}
    while cur != start:
        px, py = prev[cur]
        acts.append(_d2a[(cur[0] - px, cur[1] - py)])
        cur = (px, py)
    acts.reverse()
    return acts[:limit]


def _bomb_plans(st, aux, deadline):
    """Certified bomb candidates: own tile + reachable high-yield tiles."""
    plans = []
    if not st['bombs_left']:
        return plans
    arena = st['arena']
    x, y = st['x'], st['y']
    bombs = st['bombs']
    others = st['others']
    danger = aux['danger']
    blocked = set(aux['bomb_cells']) | (set(others) - {(x, y)})
    if aux['can_escape_bomb'] and aux['valid'].get('BOMB') \
            and not aux['must_flee']:
        plans.append({'prefix': ['BOMB'], 'bomb_tile': (x, y),
                      'is_bomb': True})

    def certified(tx, ty):
        dh = S.with_hypothetical_bomb(danger, arena, tx, ty, S.HORIZON)
        bh = list(bombs) + [((tx, ty), 4)]
        sh, _dd = S.escape_bfs((tx, ty), arena, bh, others, dh, S.HORIZON)
        n = sum(1 for d in _DIR4 if sh.get(d, False))
        return n >= ESC_DIRS

    cand = []
    W, Hh = arena.shape[0], arena.shape[1]
    for cx in range(W):
        if time.monotonic() > deadline:
            break
        for cy in range(Hh):
            if arena[cx, cy] != 0 or (cx, cy) in blocked:
                continue
            d = abs(cx - x) + abs(cy - y)
            if d == 0 or d > RADIUS:
                continue
            blast = set(S.true_blast(arena, cx, cy))
            cr = sum(1 for (bx, by) in blast
                     if 0 <= bx < W and 0 <= by < Hh and arena[bx, by] == 1)
            opps = sum(1 for o in others if o in blast)
            coins_near = sum(1 for (ox, oy) in st['coins']
                             if abs(ox - cx) + abs(oy - cy) <= 2)
            score = cr + 2.0 * opps + 0.3 * coins_near - 0.1 * d
            if cr == 0 and opps == 0:
                continue
            cand.append((-score, d, (cx, cy)))
    cand.sort()
    cap = max(1, PLANS - len(plans) - 5)
    for _, _, (cx, cy) in cand[:cap]:
        if time.monotonic() > deadline:
            break
        if not certified(cx, cy):
            continue
        path = _bfs_path(arena, blocked, (x, y), (cx, cy))
        if path is None or len(path) > 11:
            continue
        plans.append({'prefix': path + ['BOMB'], 'bomb_tile': (cx, cy),
                      'is_bomb': True})
    return plans


def override(game_state, st, aux, self_obj, allow_bombs=True):
    """Return a searched action or None (caller falls back to heuristic)."""
    t0 = time.monotonic()
    deadline = t0 + BUDGET
    try:
        simst0 = SIM.from_game_state(game_state)
    except Exception:
        return None
    rng_key = ((int(getattr(self_obj, '_seed', 0)) * 1000003
                ^ int(st['round']) * 9176 ^ int(st['step']) * 7919)
               & 0xffffffff)
    plans = []
    for action in _ACTIONS:
        if aux['valid'].get(action):
            plans.append({'prefix': [action], 'bomb_tile': None,
                          'is_bomb': False})
    if allow_bombs:
        plans.extend(_bomb_plans(st, aux, deadline))
    if not plans:
        return None

    opp_scores = [int(s) for (_, s, _, _) in
                  (game_state.get('others', []) or [])]
    margin0 = int(st['score']) - (max(opp_scores) if opp_scores else 0)

    for p in plans:
        if time.monotonic() > deadline:
            break
        tot = 0.0
        for j in range(SEEDS):
            tot += _rollout(simst0, p['prefix'], rng_key, j, margin0)
        p['score'] = tot / SEEDS
        if p['is_bomb'] and p['bomb_tile'] is not None:
            try:
                danger_ne = S.future_danger(st['arena'], st['bombs'], None,
                                            S.HORIZON)
                ck = _certified_kills(st['arena'], st['bombs'], p['bomb_tile'],
                                      st['others'], danger_ne)
                p['score'] += 5.0 * ck
            except Exception:
                pass

    best_move = None
    best_bomb = None
    for p in plans:
        if 'score' not in p:
            continue
        if p['is_bomb']:
            if best_bomb is None or p['score'] > best_bomb['score']:
                best_bomb = p
        else:
            if best_move is None or p['score'] > best_move['score']:
                best_move = p
    if best_bomb is not None and (best_move is None
                                  or best_bomb['score'] >
                                  best_move['score'] + MARGIN):
        return best_bomb['prefix'][0]
    if best_move is not None:
        return best_move['prefix'][0]
    return None
