"""Bounded best-first search over exact-dynamics plans.

Plans are scored by exact payoff plus a learned leaf value. The network
does not vote on root actions; it prices consequences that the payoff
terms cannot see.

A plan is a committed own-action prefix (a BFS path to a bomb tile plus
BOMB, or a single move) followed by a cheap continuation to detonation.
Opponents follow an avoid-lethal-if-possible policy with seeded,
deterministic randomness per round and step. Kills count at full value
only when certified against optimal flight (opp_can_escape); kills
that depend on rollout luck are ignored.
Score(plan) = exact margin delta (crates*w + coins + 5*certified kills
- 8*own death + coin-reveal expectation) + V_BLEND * V(s_end).

Budgets: wall-clock (shared with act via HARVEY_TIME_BUDGET), a plan
cap, and a leaf-value cap (features cost 1.18 ms against 0.031 ms for
sim.step, so leaf evaluation is the binding constraint). Exhaustion
falls back to the policy ranking.
"""
import os
import time
from collections import deque

import numpy as np

_DELTAS4 = ((0, -1), (0, 1), (-1, 0), (1, 0))
_DIR_TO_ACTION = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT',
                  (1, 0): 'RIGHT'}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


H = _env_int('HARVEY_SEARCH_H', 6, 2, 12)
# Rollout averaging reduces opponent-policy noise; leaf V uses one rollout.
SEEDS = _env_int('HARVEY_SEEDS', 1, 1, 8)
# Move plans may average more seeds; bomb plans retain a single hard death veto.
MOVE_SEEDS = _env_int('HARVEY_MOVE_SEEDS', 1, 1, 8)
# Common random numbers compare plans against identical opponent draws.
CRN = os.environ.get('HARVEY_CRN', '1') == '1'
K = _env_int('HARVEY_SEARCH_K', 8, 0, 32)
RADIUS = _env_int('HARVEY_SEARCH_R', 4, 1, 8)
PLAN_CAP = _env_int('HARVEY_SEARCH_PLANS', 48, 1, 256)
# Leaf-value inference is disabled by default to preserve the search budget.
V_BLEND = _env_float('HARVEY_V_BLEND', 0.0)
# Explicit leaf-value ablation, equivalent to a zero V blend.
V_OFF = os.environ.get('HARVEY_V_OFF', '0') == '1'
W_CRATE = _env_float('HARVEY_W_CRATE', 0.1)
W_DEATH = _env_float('HARVEY_W_DEATH', 8.0)
# Search chooses a bomb only when it beats the best move by this score margin.
# The legacy HARVEY_BOMB_MARGIN name remains supported for old scripts.
BOMB_MARGIN = _env_float(
    'HARVEY_BOMB_SCORE_MARGIN',
    _env_float('HARVEY_BOMB_MARGIN', 0.6))
# Optional extra score margin for an immediate plant near an opponent.
DUEL_BOMB_MARGIN = _env_float('HARVEY_DUEL_BOMB_MARGIN', 0.0)
DUEL_BOMB_D = _env_int('HARVEY_DUEL_BOMB_D', 3, 1, 6)
# Prefer productive bomb tiles that do not strand the agent far from coins.
W_COIN_TILE = _env_float('HARVEY_W_COIN_TILE', 0.5)
# Low-yield bomb tiles may be required to clear a higher score margin.
# margin_eff = BOMB_MARGIN + YIELD_GAMMA * max(0, 2 - tile_yield),
# tile_yield = crates_in_blast + 2 * opps_in_blast at the bomb tile.
YIELD_GAMMA = _env_float('HARVEY_YIELD_GAMMA', 0.0)
# Ranking bonus per opponent in a candidate bomb's blast.
W_OPP_TILE = _env_float('HARVEY_W_OPP_TILE', 2.0)
# Zero accepts any escape; positive values cap distance to persistent safety.
ESC_DIST = _env_float('HARVEY_ESC_DIST', 3.0)
# Minimum distinct escape directions required by the search bomb gate.
PLANT_ESC = _env_int('HARVEY_PLANT_ESC', 1, 1, 4)
# Optional opponent-shadow gate; it vetoes bombs but never re-ranks moves.
PLANT_OPP = _env_int('HARVEY_PLANT_OPP', 0, 0, 4)
PLANT_OPP_K = _env_int('HARVEY_PLANT_OPP_K', 1, 1, 3)
# Optional BFS coin route scored as a move plan.
COINRUN = os.environ.get('HARVEY_COINRUN', '0') == '1'
# Optional chain candidates bypass ranking, but not the escape gate.
CHAIN = os.environ.get('HARVEY_CHAIN', '0') == '1'
# Guarded chain candidates additionally require immediate tactical value.
CHAIN_GUARD = os.environ.get('HARVEY_CHAIN_GUARD', '0') == '1'
# Optional expected credit for opponents with difficult but possible escapes.
TRAP_HARD = _env_float('HARVEY_TRAP_HARD', 0.0)
TRAP_P = _env_float('HARVEY_TRAP_P', 0.5)
# Scale certified-kill payoff without changing rollout dynamics.
KILL_P = _env_float('HARVEY_KILL_P', 1.0)
# Optional pursuit plans target free bomb tiles whose blast reaches a foe.
HUNT = os.environ.get('HARVEY_HUNT', '0') == '1'
HUNT_PLANS = _env_int('HARVEY_HUNT_PLANS', 2, 1, 4)
# Cap pursuit distance because only the destination receives full certification.
HUNT_DIST = _env_int('HARVEY_HUNT_DIST', 4, 1, 11)
# Optionally credit forced kills only for our or unknown-owner bombs.
CERT_OWN = os.environ.get('HARVEY_CERT_OWN', '0') == '1'
# Rollout opponents use either a cheap warden-like or random policy.
OPPMODEL = os.environ.get('HARVEY_OPPMODEL', 'wardenlite').strip().lower()
# Solo-only controls counter policy oscillation after all opponents are gone.
# SOLO_TREK adds exact BFS move plans; SOLO_MARGIN sets their bomb comparison.
SOLO_TREK = os.environ.get('HARVEY_SOLO_TREK', '0') == '1'
SOLO_MARGIN = _env_float('HARVEY_SOLO_MARGIN', 0.15)
SOLO_TREK_COINS = _env_int('HARVEY_SOLO_TREK_COINS', 2, 1, 4)
SOLO_TREK_YIELD_N = _env_int('HARVEY_SOLO_TREK_YIELD_N', 3, 1, 8)
SOLO_TREK_YIELD_MIN = _env_float('HARVEY_SOLO_TREK_YIELD_MIN', 2.0)
# Solo search may use a wider deterministic candidate radius; zero uses RADIUS.
SOLO_RADIUS = _env_int('HARVEY_SOLO_RADIUS', 8, 0, 16)
# A short-lived committed target prevents approach-direction oscillation.
# BOMB_HYST optionally biases arbitration toward the previous target.
SOLO_COMMIT = _env_int('HARVEY_SOLO_COMMIT', 6, 0, 8)
SOLO_COMMIT_MAX = _env_int('HARVEY_SOLO_COMMIT_MAX', 6, 1, 12)
BOMB_HYST = _env_float('HARVEY_BOMB_HYST', 0.0)
# Extend committed approaches to opponent-present states when enabled.
COMMIT_OPP = os.environ.get('HARVEY_COMMIT_OPP', '1') == '1'
# Optional opening-only bomb margin and opponent-pursuit window.
OPEN_MARGIN = _env_float('HARVEY_OPEN_MARGIN', -1.0)
OPEN_T = _env_int('HARVEY_OPEN_T', 100, 0, 400)
HUNT_OPEN = os.environ.get('HARVEY_HUNT_OPEN', '0') == '1'
HUNT_OPEN_STEP = _env_int('HARVEY_HUNT_OPEN_STEP', 150, 0, 400)
HUNT_OPEN_D = _env_int('HARVEY_HUNT_OPEN_D', 5, 1, 11)
# A positive duel distance widens candidates, plan cap, and move seeds nearby.
DUEL_D = _env_int('HARVEY_DUEL_D', 0, 0, 11)
DUEL_K = _env_int('HARVEY_DUEL_K', 12, 0, 32)
DUEL_PLAN_CAP = _env_int('HARVEY_DUEL_PLANS', 96, 1, 256)
DUEL_MOVE_SEEDS = _env_int('HARVEY_DUEL_MOVE_SEEDS', 3, 1, 8)


def duel_state(game_state):
    """True iff an armed opponent is within DUEL_D Manhattan steps."""
    if DUEL_D <= 0:
        return False
    try:
        _mx, _my = game_state['self'][3]
        for o in (game_state.get('others') or []):
            if o[2] and abs(int(o[3][0]) - int(_mx)) \
                    + abs(int(o[3][1]) - int(_my)) <= DUEL_D:
                return True
    except Exception:
        return False
    return False


def _snap(st):
    """Fast structural copy of a SimState (deepcopy is ~10x slower).

    Copies every container sim.step mutates: arena, coins, agents,
    bombs, explosions list + entries. explosion['coords'] is shared:
    step() only ever REPLACES coords (fresh blast_coords list), never
    mutates one in place (verified across step/certification reads).
    """
    return {
        'arena': st['arena'].copy(),
        'coins': [c[:] for c in st['coins']],
        'agents': [dict(a) for a in st['agents']],
        'bombs': [b[:] for b in st['bombs']],
        'explosions': [{'coords': e['coords'], 'timer': e['timer'],
                        'stage': e['stage'], 'owner': e['owner']}
                       for e in st['explosions']],
        'step': st['step'],
        'total_coins': st['total_coins'],
    }


def _bfs_path(arena, blocked, start, goal, limit=12):
    """Shortest path (list of first-step actions) start -> goal.
    blocked mirrors engine tile_is_free (bombs + agents + non-floor)."""
    W, Hh = arena.shape[0], arena.shape[1]
    if tuple(start) == tuple(goal):
        return []
    prev = {tuple(start): None}
    qa = {tuple(start): None}
    queue = deque([tuple(start)])
    while queue:
        cx, cy = queue.popleft()
        if len(prev) > 400:
            break
        for dx, dy in _DELTAS4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < Hh):
                continue
            if (nx, ny) in prev:
                continue
            if arena[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            prev[(nx, ny)] = (cx, cy)
            qa[(nx, ny)] = _DIR_TO_ACTION[(dx, dy)]
            if len(prev) > 0 and (nx, ny) == tuple(goal):
                queue.clear()
                break
            queue.append((nx, ny))
    if tuple(goal) not in prev:
        return None
    acts, cur = [], tuple(goal)
    while cur != tuple(start):
        acts.append(qa[cur])
        cur = prev[cur]
    acts.reverse()
    return acts[:limit]


def _bfs_dist(arena, blocked, start):
    W, Hh = arena.shape[0], arena.shape[1]
    dist = np.full((W, Hh), 10 ** 9, dtype=np.int32)
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < W and 0 <= sy < Hh):
        return dist
    dist[sx, sy] = 0
    queue = deque([(sx, sy)])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _DELTAS4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < Hh):
                continue
            if dist[nx, ny] != 10 ** 9:
                continue
            if arena[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            dist[nx, ny] = dist[cx, cy] + 1
            queue.append((nx, ny))
    return dist


def _opp_min_dist(arena, blocked, others_xy):
    """BFS distance to the nearest opponent over walkable tiles."""
    W, Hh = arena.shape[0], arena.shape[1]
    dist = np.full((W, Hh), 10 ** 9, dtype=np.int32)
    q = deque()
    for (ox, oy) in (others_xy or []):
        ox, oy = int(ox), int(oy)
        if 0 <= ox < W and 0 <= oy < Hh and dist[ox, oy] != 0:
            dist[ox, oy] = 0
            q.append((ox, oy))
    while q:
        cx, cy = q.popleft()
        for dx, dy in _DELTAS4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < Hh):
                continue
            if dist[nx, ny] != 10 ** 9:
                continue
            if arena[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            dist[nx, ny] = dist[cx, cy] + 1
            q.append((nx, ny))
    return dist


_SHADOW_CACHE = {'key': None, 'val': ()}


def _shadow_cells(arena, others_xy, k):
    """Tiles within k BFS steps of any opponent (per-step memoized)."""
    if not others_xy:
        return set()
    key = (arena.tobytes(), tuple(sorted((int(a), int(b))
                                         for (a, b) in others_xy)), int(k))
    if _SHADOW_CACHE['key'] != key:
        dist = _opp_min_dist(arena, set(), others_xy)
        cells = set(map(tuple, np.argwhere(dist <= int(k)).tolist()))
        _SHADOW_CACHE['key'] = key
        _SHADOW_CACHE['val'] = cells
    return _SHADOW_CACHE['val']


def chain_guard_ok(arena, blocked, bombs, others_xy, danger, tx, ty):
    """Admission predicate for chain tiles, matching the warden payoff rule.

    Returns (admit, tile_yield). Pure, so the probes can call it
    directly: a chain tile is admitted when it has a proven escape (the
    same recipe as the plan gate) and the warden want_bomb payoff guard
    holds (opps_hit > 0, or crates_hit >= 2 with hyp_dist <= 3, or
    crates_hit == 1 with hyp_dist <= 2).
    """
    from .safety import escape_bfs, with_hypothetical_bomb
    from .sim import blast_coords
    W, Hh = arena.shape[0], arena.shape[1]
    if not (0 <= tx < W and 0 <= ty < Hh):
        return False, 0.0
    if arena[tx, ty] != 0 or (tx, ty) in blocked:
        return False, 0.0
    try:
        blast = set(blast_coords(arena, tx, ty))
    except Exception:
        return False, 0.0
    opps = sum(1 for o in others_xy if o in blast)
    try:
        cr = sum(1 for (bx, by) in blast if arena[bx, by] == 1)
    except Exception:
        cr = 0
    tyield = float(cr) + 2.0 * float(opps)
    try:
        dh = with_hypothetical_bomb(danger, arena, tx, ty, 8)
        bh = list(bombs) + [((tx, ty), 4)]
        sh, dhyp = escape_bfs((tx, ty), arena, bh, others_xy, dh, 8)
        ok = any(sh.get(d, False)
                 for d in [(0, -1), (0, 1), (-1, 0), (1, 0)])
        hd = float(dhyp)
        if ESC_DIST > 0:
            ok = bool(ok and hd <= ESC_DIST)
    except Exception:
        ok, hd = False, float('inf')
    if not ok:
        return False, tyield
    if opps > 0 or (cr >= 2 and hd <= 3.0) or (cr == 1 and hd <= 2.0):
        return True, tyield
    return False, tyield


def _try_bomb_plan(arena, blocked, bombs, others_xy, danger,
                   plans, x, y, cx, cy, tile_yield):
    """Shared bomb-tile admission: BFS path plus the proven-escape gate.

    One gate serves ranked and chain tiles alike. Returns True on
    admission.
    """
    from .safety import escape_bfs, with_hypothetical_bomb
    path = _bfs_path(arena, blocked, (x, y), (cx, cy))
    if path is None:
        return False
    try:
        dh = with_hypothetical_bomb(danger, arena, cx, cy, 8)
        bh = list(bombs) + [((cx, cy), 4)]
        sh, dhyp = escape_bfs((cx, cy), arena, bh,
                              others_xy, dh, 8)
        _dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        n_esc = sum(1 for d in _dirs if sh.get(d, False))
        can = n_esc >= PLANT_ESC
        if ESC_DIST > 0:
            try:
                can = bool(can and float(dhyp) <= ESC_DIST)
            except Exception:
                can = False
        if can and PLANT_OPP > 0:
            # Certified escapes must also avoid each opponent's reach shadow.
            try:
                shadow = _shadow_cells(arena, others_xy, PLANT_OPP_K)
                sh_opp, d_opp = escape_bfs(
                    (cx, cy), arena, bh,
                    list(others_xy) + list(shadow), dh, 8)
                n_opp = sum(1 for d in _dirs if sh_opp.get(d, False))
                can = n_opp >= PLANT_OPP
                if can and ESC_DIST > 0:
                    can = bool(float(d_opp) <= max(ESC_DIST, 4.0))
            except Exception:
                can = False
    except Exception:
        can = False
    if not can:
        return False
    prefix = (path + ['BOMB'])[:12]
    plans.append({'first': prefix[0], 'prefix': prefix,
                  'bomb_at': (cx, cy),
                  'tile_yield': float(tile_yield.get(
                      (cx, cy), 2.0))})
    return True


def hunt_trigger_ok(arena, n_coins, others_xy, x, y, step):
    """Warden hunt predicate: opponents present
    AND (loot <= 6 | step > 200 | opponent within Manhattan 3)."""
    if not others_xy:
        return False
    loot = int((np.asarray(arena) == 1).sum()) + int(n_coins)
    dmin = min(abs(ox - x) + abs(oy - y) for (ox, oy) in others_xy)
    return bool(loot <= 6 or step > 200 or dmin <= 3)


def hunt_tiles(arena, blocked, ox, oy):
    """Free tiles whose hypothetical blast covers the opp tile (ox, oy).

    Rays from the opp tile in all 4 directions up to blast power 3:
    a bomb at tile T covers the opp iff the opp tile is reachable from
    T through non-wall tiles (crates pass blast but cannot be stood
    on; bombs/agents block standing but not the ray). Returned
    ring-first (dist 1, 2, 3), dir order fixed for determinism.
    """
    W, Hh = arena.shape[0], arena.shape[1]
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, 4):
            tx, ty = ox + dx * i, oy + dy * i
            if not (0 <= tx < W and 0 <= ty < Hh):
                break
            if arena[tx, ty] == -1:
                break
            if arena[tx, ty] != 0 or (tx, ty) in blocked:
                continue
            out.append((tx, ty))
    return out


def gen_plans(game_state, safety, K=K, radius=RADIUS):
    """Root candidate plans. Each plan: dict(first, prefix, bomb_at)."""
    from .safety import escape_bfs, future_danger, with_hypothetical_bomb
    from .sim import yield_field
    arena = np.asarray(game_state['field'])
    _, _, bombs_left, (x, y) = game_state['self']
    x, y = int(x), int(y)
    step = int(game_state.get('step', 0))
    bombs = game_state.get('bombs', []) or []
    bomb_set = set((int(bxy[0]), int(bxy[1])) for (bxy, _) in bombs)
    others = game_state.get('others', []) or []
    others_xy = [(int(p[3][0]), int(p[3][1])) for p in others]
    # engine tile_is_free blocks bombs AND agents: paths must too, or the
    # first step lands on an occupied tile (INVALID_ACTION).
    blocked = set(bomb_set) | (set(others_xy) - {(x, y)})
    plans = []
    valid = safety.get('valid', {})
    for a in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
        if valid.get(a):
            plans.append({'first': a, 'prefix': [a], 'bomb_at': None})
    if COINRUN:
        # Replan a short BFS route toward the nearest visible coin each step.
        try:
            _coins = [(int(c[0]), int(c[1])) for c in
                      (game_state.get('coins', []) or [])]
            _coins.sort(key=lambda c: abs(c[0] - x) + abs(c[1] - y))
            for (_tx, _ty) in _coins:
                _path = _bfs_path(arena, blocked, (x, y), (_tx, _ty),
                                  limit=12)
                if _path:
                    plans.append({'first': _path[0], 'prefix': _path,
                                  'bomb_at': None, 'coin_run': True})
                    break
        except Exception:
            pass
    if bombs_left and K > 0:
        try:
            yf = yield_field(arena)
        except Exception:
            yf = None
        if yf is not None:
            coins_xy = [(int(c[0]), int(c[1])) for c in
                        (game_state.get('coins', []) or [])]
            dist = _bfs_dist(arena, blocked, (x, y))
            cand = []
            W, Hh = arena.shape[0], arena.shape[1]
            for cx in range(W):
                for cy in range(Hh):
                    if arena[cx, cy] != 0 or (cx, cy) in blocked:
                        continue
                    dd = int(dist[cx, cy])
                    if dd > radius:
                        continue
                    near_coins = sum(1 for (ox, oy) in coins_xy
                                     if abs(ox - cx) <= 3
                                     and abs(oy - cy) <= 3)
                    cand.append((float(yf[cx, cy])
                                 + W_COIN_TILE * near_coins,
                                 -dd, (cx, cy)))
            cand.sort(reverse=True)
            try:
                danger = future_danger(arena, bombs,
                                       game_state.get('explosion_map'), 8)
            except Exception:
                danger = None
            # Add opponent blast value only after the cheap yield pre-ranking.
            from .sim import blast_coords
            rescored = []
            tile_yield = {}
            for key, negdd, (cx, cy) in cand[:max(K * 4, 16)]:
                try:
                    blast = set(blast_coords(arena, cx, cy))
                except Exception:
                    blast = set()
                opps = sum(1 for o in others_xy if o in blast)
                try:
                    crates = sum(1 for (bx, by) in blast
                                 if arena[bx, by] == 1)
                except Exception:
                    crates = 0
                tile_yield[(cx, cy)] = float(crates) + 2.0 * float(opps)
                rescored.append((key + W_OPP_TILE * opps, negdd,
                                 (cx, cy)))
            rescored.sort(reverse=True)
            # Append chain tiles so existing plans retain their seed indices.
            ordered = [(cx, cy) for _, _, (cx, cy) in rescored]
            cap = K
            if CHAIN:
                chain = []
                for _fx, _fy in ((x, y), (x + 1, y), (x - 1, y),
                                 (x, y + 1), (x, y - 1)):
                    if not (0 <= _fx < W and 0 <= _fy < Hh):
                        continue
                    if arena[_fx, _fy] != 0 or (_fx, _fy) in blocked:
                        continue
                    chain.append((_fx, _fy))
                _seen = set(ordered)
                ordered = list(ordered) + [t for t in chain
                                           if t not in _seen]
                cap = K + len(chain)
            # The K cap applies strictly to the ranked candidate walk.
            for (cx, cy) in ordered:
                if len([p for p in plans if p['bomb_at']]) >= cap:
                    break
                _try_bomb_plan(arena, blocked, bombs, others_xy,
                               danger, plans, x, y, cx, cy, tile_yield)
            if CHAIN_GUARD:
                # Append unique guarded-chain tiles through the common bomb gate.
                _taken = set(p['bomb_at'] for p in plans
                             if p['bomb_at'])
                _extra = []
                for (_fx, _fy) in ((x, y), (x + 1, y), (x - 1, y),
                                   (x, y + 1), (x, y - 1)):
                    _ok, _ty = chain_guard_ok(
                        arena, blocked, bombs, others_xy, danger,
                        _fx, _fy)
                    if not _ok:
                        continue
                    tile_yield[(_fx, _fy)] = _ty
                    if (_fx, _fy) not in _taken:
                        _extra.append((_fx, _fy))
                _cap2 = K + len(_extra)
                for (_fx, _fy) in _extra:
                    if len([p for p in plans
                            if p['bomb_at']]) >= _cap2:
                        break
                    _try_bomb_plan(arena, blocked, bombs, others_xy,
                                   danger, plans, x, y, _fx, _fy,
                                   tile_yield)
            # Add at most one gated pursuit plan per opponent, nearest first.
            if HUNT and others_xy and hunt_trigger_ok(
                    arena, len(coins_xy), others_xy, x, y, step):
                from .sim import blast_coords as _hblast
                _n0 = len([p for p in plans if p['bomb_at']])
                for (ox, oy) in sorted(
                        others_xy,
                        key=lambda o: abs(o[0] - x) + abs(o[1] - y)):
                    if len([p for p in plans if p['bomb_at']]) \
                            - _n0 >= HUNT_PLANS:
                        break
                    _cand = [t for t in hunt_tiles(arena, blocked,
                                                   ox, oy)
                             if int(dist[t]) <= HUNT_DIST
                             and t not in [p['bomb_at'] for p in plans
                                           if p['bomb_at']]]
                    _cand.sort(key=lambda t: int(dist[t]))
                    for (_tx, _ty) in _cand:
                        try:
                            _hb = set(_hblast(arena, _tx, _ty))
                            tile_yield[(_tx, _ty)] = float(sum(
                                1 for (bx, by) in _hb
                                if arena[bx, by] == 1)) \
                                + 2.0 * float(sum(
                                    1 for o in others_xy if o in _hb))
                        except Exception:
                            pass
                        if _try_bomb_plan(
                                arena, blocked, bombs, others_xy,
                                danger, plans, x, y, _tx, _ty,
                                tile_yield):
                            break
    # Opening pursuit plans share the normal admission and escape gate.
    if HUNT_OPEN and others_xy and step < HUNT_OPEN_STEP \
            and yf is not None:
        try:
            _near = sorted(((abs(ox - x) + abs(oy - y), ox, oy)
                            for (ox, oy) in others_xy
                            if abs(ox - x) + abs(oy - y) <= HUNT_OPEN_D))
            _n0 = len([p for p in plans if p['bomb_at']])
            for (_hd, ox, oy) in _near:
                if len([p for p in plans if p['bomb_at']]) - _n0 \
                        >= HUNT_PLANS:
                    break
                _cand = [t for t in hunt_tiles(arena, blocked, ox, oy)
                         if int(dist[t]) <= HUNT_DIST
                         and t not in [p['bomb_at'] for p in plans
                                       if p['bomb_at']]]
                _cand.sort(key=lambda t: int(dist[t]))
                for (_tx, _ty) in _cand:
                    try:
                        _hb = set(blast_coords(arena, _tx, _ty))
                        tile_yield[(_tx, _ty)] = float(sum(
                            1 for (bx, by) in _hb
                            if arena[bx, by] == 1)) \
                            + 2.0 * float(sum(1 for o in others_xy
                                              if o in _hb))
                    except Exception:
                        pass
                    if _try_bomb_plan(arena, blocked, bombs, others_xy,
                                      danger, plans, x, y, _tx, _ty,
                                      tile_yield):
                        break
        except Exception:
            pass
    # Solo trek order: visible coins, productive bomb tiles, then crates.
    # Append these plans so the candidate order and seed indices stay stable.
    if SOLO_TREK and not others_xy:
        try:
            from .sim import yield_field as _solo_yf
            try:
                _yf = _solo_yf(arena)
            except Exception:
                _yf = None
            _solo_coins = [(int(c[0]), int(c[1])) for c in
                           (game_state.get('coins', []) or [])]
            _solo_coins.sort(key=lambda c: abs(c[0] - x) + abs(c[1] - y))
            _n_coin = 0
            for (_tx, _ty) in _solo_coins:
                if _n_coin >= SOLO_TREK_COINS:
                    break
                _path = _bfs_path(arena, blocked, (x, y), (_tx, _ty),
                                  limit=12)
                if _path:
                    plans.append({'first': _path[0], 'prefix': _path,
                                  'bomb_at': None, 'trek': 'coin',
                                  'trek_target': (_tx, _ty)})
                    _n_coin += 1
            if _yf is not None:
                _dist = _bfs_dist(arena, blocked, (x, y))
                W, Hh = arena.shape[0], arena.shape[1]
                _yield_c = []
                _crate_c = []
                for cx in range(W):
                    for cy in range(Hh):
                        if arena[cx, cy] != 0 or (cx, cy) in blocked:
                            continue
                        dd = int(_dist[cx, cy])
                        if dd <= 0 or dd >= 10 ** 9:
                            continue
                        yv = float(_yf[cx, cy])
                        if yv >= SOLO_TREK_YIELD_MIN:
                            _yield_c.append((-yv, dd, (cx, cy)))
                        elif yv >= 1.0:
                            _crate_c.append((dd, (cx, cy)))
                _yield_c.sort()
                for _neg, _dd, (_tx, _ty) in \
                        _yield_c[:SOLO_TREK_YIELD_N]:
                    _path = _bfs_path(arena, blocked, (x, y), (_tx, _ty),
                                      limit=12)
                    if not _path:
                        continue
                    plans.append({'first': _path[0], 'prefix': _path,
                                  'bomb_at': None, 'trek': 'yield',
                                  'trek_target': (_tx, _ty),
                                  'trek_yield': -_neg})
                if not _yield_c:
                    _crate_c.sort()
                    for _dd, (_tx, _ty) in _crate_c[:1]:
                        _path = _bfs_path(arena, blocked, (x, y),
                                          (_tx, _ty), limit=12)
                        if _path:
                            plans.append({'first': _path[0],
                                          'prefix': _path,
                                          'bomb_at': None,
                                          'trek': 'crate',
                                          'trek_target': (_tx, _ty)})
        except Exception:
            pass
    return plans


def _sim_bombs(st):
    """Sim bombs [x,y,t,owner] -> game_state ((x,y),t) pairs for safety fns."""
    return [((b[0], b[1]), b[2]) for b in st['bombs']]


def _opp_move(rng, st, i, danger_now):
    """Cheap opponent model: avoid lethal tiles, then seeded random."""
    from .sim import valid_actions
    a = st['agents'][i]
    opts = valid_actions(st, i)
    if not opts:
        return 'WAIT'
    bad = set()
    try:
        d = np.asarray(danger_now)
        for act in opts:
            from .sim import _DELTAS
            dx, dy = _DELTAS[act]
            nx, ny = a['x'] + dx, a['y'] + dy
            if d.ndim == 3 and d.shape[0] > 2:
                if bool(d[0, nx, ny]) or bool(d[1, nx, ny]) \
                        or bool(d[2, nx, ny]):
                    bad.add(act)
    except Exception:
        pass
    good = [o for o in opts if o not in bad]
    if OPPMODEL == 'wardenlite':
        # Opponents bomb only with tactical value and pre-plant mobility.
        if 'BOMB' in good:
            try:
                from .sim import blast_coords
                blast = set(blast_coords(st['arena'], a['x'], a['y']))
                crates = sum(1 for (bx, by) in blast
                             if st['arena'][bx, by] == 1)
                opps = sum(1 for k, o in enumerate(st['agents'])
                           if o['alive'] and k != i
                           and (o['x'], o['y']) in blast)
                guard = opps > 0 or crates >= 2
                free = 0
                if guard:
                    d = np.asarray(danger_now) \
                        if danger_now is not None else None
                    from .sim import _DELTAS
                    W, Hh = st['arena'].shape[0], st['arena'].shape[1]
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = a['x'] + dx, a['y'] + dy
                        if not (0 <= nx < W and 0 <= ny < Hh):
                            continue
                        if st['arena'][nx, ny] != 0:
                            continue
                        if d is not None and d.ndim == 3 \
                                and d.shape[0] > 2 and (
                                    bool(d[0, nx, ny])
                                    or bool(d[1, nx, ny])):
                            continue
                        free += 1
                if not guard or free < 2:
                    good.remove('BOMB')
            except Exception:
                pass
        # movement: Manhattan-coin-pursuit among good moves
        if good and good != ['WAIT']:
            try:
                coins = [c for c in st['coins'] if c[2]]
                if coins:
                    from .sim import _DELTAS
                    cx, cy = min(((c[0], c[1]) for c in coins),
                                 key=lambda c: abs(c[0] - a['x'])
                                 + abs(c[1] - a['y']))
                    def _step_cost(act):
                        if act == 'WAIT':
                            return abs(cx - a['x']) + abs(cy - a['y'])
                        dx, dy = _DELTAS[act]
                        return abs(cx - a['x'] - dx) \
                            + abs(cy - a['y'] - dy)
                    best = min(_step_cost(o) for o in good)
                    good = [o for o in good if _step_cost(o) == best]
            except Exception:
                pass
    if not good:
        good = opts
    return good[int(rng.integers(len(good)))]


def _continuation(game_state, st, i, danger=None):
    """Cheap post-prefix policy. Danger-aware: if our own tile is lethal
    within 2 steps, flee to the valid move minimizing near-term danger
    (tie-break: coin-greedy); else coin-greedy. The value head prices
    the leaf, so no network runs here. The danger check comes first
    because a pure coin-greedy continuation walks bomb plans into their
    own blast and prices every bomb at -8."""
    arena = st['arena']
    bombs = [(b[0], b[1]) for b in st['bombs']]
    bomb_set = set(bombs)
    a = st['agents'][i]
    # flee first: own tile lethal within 2 steps?
    try:
        d = np.asarray(danger) if danger is not None else None
        if d is not None and d.ndim == 3 and d.shape[0] > 2:
            if bool(d[0, a['x'], a['y']]) or bool(d[1, a['x'], a['y']]) \
                    or bool(d[2, a['x'], a['y']]):
                from .sim import valid_actions
                best_a, best_key = 'WAIT', None
                for act in valid_actions(st, i):
                    from .sim import _DELTAS
                    dx, dy = _DELTAS[act]
                    nx, ny = a['x'] + dx, a['y'] + dy
                    key = (int(d[0, nx, ny]) + int(d[1, nx, ny])
                           + int(d[2, nx, ny]))
                    if best_key is None or key < best_key:
                        best_key, best_a = key, act
                return best_a
    except Exception:
        pass
    dist = _bfs_dist(np.asarray(arena), bomb_set, (a['x'], a['y']))
    best, best_d, best_a = None, 10 ** 9, None
    targets = [(c[0], c[1]) for c in st['coins'] if c[2]]
    if not targets:
        for cx in range(arena.shape[0]):
            for cy in range(arena.shape[1]):
                if arena[cx, cy] == 1:
                    for dx, dy in _DELTAS4:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < arena.shape[0] \
                                and 0 <= ny < arena.shape[1] \
                                and arena[nx, ny] == 0:
                            targets.append((nx, ny))
    from .sim import valid_actions
    valid = set(valid_actions(st, i))
    for (tx, ty) in targets:
        if not (0 <= tx < dist.shape[0] and 0 <= ty < dist.shape[1]):
            continue
        if dist[tx, ty] < best_d:
            # first step toward (tx,ty): neighbour with dist-1
            for dx, dy in _DELTAS4:
                nx, ny = tx - dx, ty - dy
                if 0 <= nx < dist.shape[0] and 0 <= ny < dist.shape[1] \
                        and dist[nx, ny] == dist[tx, ty] - 1:
                    act = _DIR_TO_ACTION.get((a['x'] - nx, a['y'] - ny))
                    if act is None:
                        # walk down the gradient from self instead
                        break
                    if act in valid:
                        best_d, best_a = int(dist[tx, ty]), act
                    break
            else:
                continue
    return best_a or 'WAIT'


def score_plan(st0, plan, horizon=H, seed=0, w_crate=W_CRATE,
               w_death=W_DEATH):
    """Exact rollout of a plan. Returns (margin_delta, end_state, info).
    Kills count only when certified against optimal flight."""
    from .sim import step as sim_step, margin, to_game_state
    from .safety import opp_can_escape, future_danger
    rng = None if CRN else np.random.default_rng(seed)
    st = _snap(st0)
    m0 = margin(st)
    prefix = list(plan['prefix'])
    payoff = 0.0
    certified_kills = 0
    frac_done = set()
    ticks = 0

    def _opp_act(i, tick, danger):
        if CRN:
            r = np.random.default_rng((int(seed), int(tick), int(i)))
            return _opp_move(r, st, i, danger)
        return _opp_move(rng, st, i, danger)

    # phase 1: committed prefix
    for act in prefix:
        if not st['agents'][0]['alive']:
            break
        acts = [act]
        try:
            danger = future_danger(st['arena'], _sim_bombs(st), None, 4)
        except Exception:
            danger = None
        for i in range(1, len(st['agents'])):
            if st['agents'][i]['alive']:
                acts.append(_opp_act(i, ticks, danger))
            else:
                acts.append('WAIT')
        info = sim_step(st, acts)
        payoff += W_CRATE * info['crates'] + info['revealed_exp']
        payoff += sum(info['coins'][0:1]) * 1.0
        ticks += 1
        if info['died'][0]:
            payoff -= w_death
            break
    # Simulate through detonation and lingering blast before scoring the leaf.
    settle = max(horizon, len(prefix) + 7)
    while st['agents'][0]['alive'] and ticks < settle:
        if not st['bombs'] and not any(
                e['stage'] == 0 for e in st['explosions']):
            break
        try:
            danger = future_danger(st['arena'], _sim_bombs(st), None, 4)
        except Exception:
            danger = None
        acts = [_continuation(None, st, 0, danger)]
        for i in range(1, len(st['agents'])):
            if st['agents'][i]['alive']:
                acts.append(_opp_act(i, ticks, danger))
            else:
                acts.append('WAIT')
        # Credit only opponents that cannot escape a relevant bomb under
        # optimal flight; enemy self-traps are not our kills.
        try:
            for b in st['bombs']:
                if b[2] > 2:
                    continue
                if CERT_OWN and b[3] not in (0, -1):
                    continue
                from .sim import blast_coords
                blast = set(blast_coords(st['arena'], b[0], b[1]))
                for j in range(1, len(st['agents'])):
                    oj = st['agents'][j]
                    if not oj['alive']:
                        continue
                    if (oj['x'], oj['y']) not in blast:
                        continue
                    others_xy = [(o['x'], o['y'])
                                 for k, o in enumerate(st['agents'])
                                 if k != j and o['alive']]
                    can, dist = opp_can_escape(
                        np.asarray(st['arena']), _sim_bombs(st),
                        (oj['x'], oj['y']), (b[0], b[1]), others_xy)
                    if not can:
                        certified_kills += 1
                        oj['alive'] = False
                        st['agents'][0]['score'] += 5
                    elif TRAP_HARD > 0 and j not in frac_done \
                            and b[2] <= 2:
                        try:
                            hard = float(dist) >= TRAP_HARD
                        except Exception:
                            hard = False
                        if hard:
                            frac_done.add(j)
                            certified_kills += TRAP_P
        except Exception:
            pass
        info = sim_step(st, acts)
        payoff += W_CRATE * info['crates'] + info['revealed_exp']
        payoff += sum(info['coins'][0:1]) * 1.0
        ticks += 1
        if info['died'][0]:
            payoff -= w_death
            break
    payoff += KILL_P * 5.0 * certified_kills
    payoff += margin(st) - m0 - (st['agents'][0]['score'] - st0['agents'][0]['score'])
    # margin() already includes score deltas; the last line adds the
    # OPPONENT-score movement only (own score counted once via margin).
    return payoff, st


def search_action(game_state, safety, model, t0, budget, state=None):
    """Bounded best-first over plans. Returns (action|None, debug).

    `state` is an optional dict the caller persists across ticks, used
    by the committed-approach and sticky-target handling. None or an
    empty dict gives the plain behavior.
    """
    import torch
    from .sim import from_game_state, to_game_state, margin
    from .features import state_to_features
    dbg = {'plans': 0, 'leaves': 0, 'exhausted': False}
    try:
        st0 = from_game_state(game_state)
    except Exception:
        return None, dbg
    # Widen the bounded search only when an armed opponent is nearby.
    _duel = duel_state(game_state)
    dbg['duel'] = _duel
    k_eff = DUEL_K if _duel else K
    plan_cap = DUEL_PLAN_CAP if _duel else PLAN_CAP
    mv_seeds = DUEL_MOVE_SEEDS if _duel else MOVE_SEEDS
    _solo_board = not (game_state.get('others') or [])
    r_eff = SOLO_RADIUS if (SOLO_RADIUS > 0 and _solo_board) else RADIUS
    # Continue a committed approach without re-arbitration while its target,
    # path length, first-step safety, bomb availability, and age remain valid.
    _commit_ok = _solo_board or COMMIT_OPP
    if state is not None and SOLO_COMMIT > 0 and _commit_ok:
        try:
            _tgt = state.get('commit')
            if _tgt is not None:
                _age = int(state.get('commit_age', 0)) + 1
                state['commit_age'] = _age
                if _age > SOLO_COMMIT_MAX:
                    state['commit'] = None
                else:
                    arena_c = np.asarray(st0['arena'])
                    cx, cy = int(_tgt[0]), int(_tgt[1])
                    _bl = bool(game_state['self'][2])
                    _bombs = game_state.get('bombs') or []
                    _bomb_set = set((int(b[0][0]), int(b[0][1]))
                                    for b in _bombs)
                    _others = [(int(o[3][0]), int(o[3][1]))
                               for o in (game_state.get('others') or [])]
                    _blk = _bomb_set | (set(_others) -
                                        {(int(game_state['self'][3][0]),
                                          int(game_state['self'][3][1]))})
                    if not _bl or arena_c[cx, cy] != 0 or (cx, cy) in _blk:
                        state['commit'] = None
                    else:
                        _path = _bfs_path(arena_c, _blk,
                                          (int(game_state['self'][3][0]),
                                           int(game_state['self'][3][1])),
                                          (cx, cy))
                        _act = _path[0] if _path else 'BOMB'
                        _ok = (bool(safety.get('valid', {})
                                    .get(_act, False))
                               and bool(safety.get('safe', {})
                                        .get(_act, False)))
                        if _ok:
                            dbg['commit'] = [cx, cy]
                            dbg['commit_age'] = _age
                            return _act, dbg
                        state['commit'] = None
        except Exception:
            state['commit'] = None
    plans = gen_plans(game_state, safety, K=k_eff, radius=r_eff)[:plan_cap]
    if not plans:
        return None, dbg
    rnd = int(game_state.get('round', 0))
    stp = int(game_state.get('step', 0))
    scored = []
    v_batch, v_idx = [], []
    dbg['seeds'] = SEEDS
    dbg['move_seeds'] = mv_seeds
    base_seed = 1000 * rnd + stp
    for pi_, plan in enumerate(plans):
        if (time.perf_counter() - t0) >= budget:
            dbg['exhausted'] = True
            break
        # Average move payoff across configured seeds, but apply the full
        # death penalty if any rollout dies. Bomb plans remain single-seed.
        n_seeds = mv_seeds if plan['bomb_at'] is None else 1
        pay_sum, end = 0.0, None
        died_any = False
        for j in range(n_seeds):
            if (time.perf_counter() - t0) >= budget:
                dbg['exhausted'] = True
                break
            # CRN omits plan index so competing plans share opponent draws.
            payoff_j, end_j = score_plan(
                st0, plan,
                seed=base_seed + 7919 * j + (0 if CRN else pi_))
            died_j = not end_j['agents'][0]['alive']
            pay_sum += payoff_j + (W_DEATH if died_j else 0.0)
            died_any = died_any or died_j
            if end is None:
                end = end_j
        if end is None:
            continue  # budget died mid-plan: drop it, keep the fallback
        scored.append([pay_sum / n_seeds - (W_DEATH if died_any else 0.0),
                       plan, end])
        v_batch.append(end)
        v_idx.append(len(scored) - 1)
    dbg['plans'] = len(scored)
    # one batched V evaluation over all leaves
    try:
        if v_batch and V_BLEND != 0.0 and not V_OFF:
            feats = np.stack([
                state_to_features(to_game_state(e), None)
                for e in v_batch]).astype(np.float32)
            with torch.no_grad():
                vv = model(torch.from_numpy(feats).to(
                    next(model.parameters()).device))[1]
                vv = np.asarray(vv.cpu().numpy(), dtype=np.float64) * 10.0
            for j, v, est in zip(v_idx, vv, v_batch):
                if not est['agents'][0]['alive']:
                    continue  # dead leaf: exact payoff only (V never saw
                    # dead states; the -8 already priced the death)
                scored[j][0] += V_BLEND * float(v)
                dbg['leaves'] += 1
    except Exception:
        pass
    if not scored:
        return None, dbg
    # Search owns bombs only; otherwise return None for policy-based movement.
    # Low-yield bomb tiles may receive an additional soft score margin.
    bombs = [r for r in scored if r[1]['bomb_at'] is not None]
    moves = [r for r in scored if r[1]['bomb_at'] is None]
    best_move = max([r[0] for r in moves], default=float('-inf'))
    # Expose the best move-plan action for optional policy distillation.
    if moves:
        _bm = max(moves, key=lambda r: r[0])
        dbg['best_move_first'] = _bm[1]['first']
        dbg['best_move_score'] = float(_bm[0])
    bombs.sort(key=lambda r: r[0], reverse=True)
    dbg['best_move'] = float(best_move) if moves else None
    dbg['best_bomb'] = float(bombs[0][0]) if bombs else None
    bombs = [r for r in scored if r[1]['bomb_at'] is not None]
    moves = [r for r in scored if r[1]['bomb_at'] is None]
    best_move = max([r[0] for r in moves], default=float('-inf'))
    # Recomputed block retained for behavior compatibility with the trained net.
    if moves:
        _bm = max(moves, key=lambda r: r[0])
        dbg['best_move_first'] = _bm[1]['first']
        dbg['best_move_score'] = float(_bm[0])
    bombs.sort(key=lambda r: r[0], reverse=True)
    dbg['best_move'] = float(best_move) if moves else None
    dbg['best_bomb'] = float(bombs[0][0]) if bombs else None
    if bombs:
        try:
            _y = float(bombs[0][1].get('tile_yield', 2.0))
        except Exception:
            _y = 2.0
        _solo = _solo_board
        if _solo and SOLO_MARGIN >= 0.0:
            # Solo bomb plans use the dedicated, usually lower margin.
            margin_eff = SOLO_MARGIN
        elif OPEN_MARGIN >= 0.0 \
                and not (game_state.get('coins') or []) \
                and int(game_state.get('step', 0)) < OPEN_T:
            # The opening margin applies only before a coin becomes visible.
            margin_eff = OPEN_MARGIN
        else:
            margin_eff = BOMB_MARGIN + YIELD_GAMMA * max(0.0, 2.0 - _y)
        dbg['margin_eff'] = float(margin_eff)
        dbg['best_yield'] = float(_y)
        dbg['phase'] = ('solo' if (_solo and SOLO_MARGIN >= 0.0)
                        else ('open' if (OPEN_MARGIN >= 0.0
                                         and margin_eff == OPEN_MARGIN)
                              else 'default'))
        # Save top targets for diagnostics and committed-approach state.
        try:
            dbg['committed_bomb_at'] = bombs[0][1].get('bomb_at')
            dbg['committed_first'] = bombs[0][1].get('first')
            if len(bombs) > 1:
                dbg['bomb2_at'] = bombs[1][1].get('bomb_at')
                dbg['bomb2_score'] = float(bombs[1][0])
        except Exception:
            pass
        # Optionally bias arbitration toward the previous committed target.
        _pick = bombs[0]
        if state is not None and BOMB_HYST > 0.0 and _solo_board:
            try:
                sticky = state.get('last_target')
                if sticky is not None:
                    for _r in bombs:
                        if _r[1].get('bomb_at') == sticky:
                            if _r[0] + BOMB_HYST > _pick[0]:
                                _pick = _r
                            break
            except Exception:
                pass
        # Risk-price only an immediate plant in a close duel. Applying the
        # surcharge after hysteresis keeps it on the selected plan, and
        # checking first == BOMB leaves approach movement intact.
        _duel_margin = 0.0
        if DUEL_BOMB_MARGIN > 0.0 and not _solo_board \
                and _pick[1].get('first') == 'BOMB':
            try:
                _x, _y = (int(game_state['self'][3][0]),
                           int(game_state['self'][3][1]))
                if any(abs(int(o[3][0]) - _x) + abs(int(o[3][1]) - _y)
                       <= DUEL_BOMB_D
                       for o in (game_state.get('others') or [])):
                    _duel_margin = DUEL_BOMB_MARGIN
                    margin_eff += _duel_margin
            except Exception:
                _duel_margin = 0.0
        dbg['duel_margin'] = float(_duel_margin)
        if (_pick[0] - best_move) > margin_eff:
            if state is not None and (_solo_board or COMMIT_OPP):
                try:
                    state['last_target'] = _pick[1].get('bomb_at')
                    if SOLO_COMMIT > 0:
                        state['commit'] = _pick[1].get('bomb_at')
                        state['commit_age'] = 0
                except Exception:
                    pass
            dbg['commit'] = None
            return _pick[1]['first'], dbg
    # In solo mode, a safe trek move may replace an oscillating policy fallback.
    if SOLO_TREK and not (game_state.get('others') or []):
        treks = [r for r in scored if r[1].get('trek')]
        if treks:
            treks.sort(key=lambda r: (r[0],
                                      r[1].get('trek_yield', 0.0)),
                       reverse=True)
            for _r in treks:
                _a = _r[1]['first']
                if not isinstance(_a, str):
                    continue
                if not safety.get('valid', {}).get(_a, False):
                    continue
                if not safety.get('safe', {}).get(_a, False):
                    continue
                dbg['solo_trek'] = _r[1].get('trek')
                dbg['solo_trek_target'] = _r[1].get('trek_target')
                return _a, dbg
    return None, dbg
