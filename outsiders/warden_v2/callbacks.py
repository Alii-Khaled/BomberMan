# Warden v2 — outsider sparring agent (heuristic, non-learning).
#
# Source of truth: outsiders/warden_v2/; agent_code/warden_v2 is a symlink
# so the game loader finds it.
#
# v2 over v1:
#   * corrected time-expanded escape solver (latest-lethal test + arrival
#     check, E88 semantics) and a boolean danger timeline (no window loss)
#   * certified-trap bombs (opponent in blast with no proven escape)
#   * optional bounded rollout search over exact-dynamics plans
#     (WARDEN_SEARCH=rollout, default) with a wall-clock budget and
#     graceful fallback to the v1-speed heuristic
#   * deterministic seeded RNG for reproducible A/B evaluation
# Everything is numpy-only, CPU-only, self-contained.

from collections import deque

import numpy as np

from . import safety as S

_BOMB_TIMER = 4
_DELTAS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
    'WAIT': (0, 0),
}
_DIR_TO_ACTION = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT',
                  (1, 0): 'RIGHT', (0, 0): 'WAIT'}
_MOVES = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
_DIR4 = ((0, -1), (0, 1), (-1, 0), (1, 0))


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


HORIZON = S.HORIZON
SEARCH = _env('WARDEN_SEARCH', 'off', str).strip().lower()
OPP_TRAP = bool(_env('WARDEN_OPP_TRAP', 0, int, 0, 1))
MOBILITY_W = _env('WARDEN_MOBILITY_W', 0.0, float)
LOOP_W = _env('WARDEN_LOOP_W', 1.0, float)
SAFE_BONUS = _env('WARDEN_SAFE_BONUS', 3.0, float)
PREF_BONUS = _env('WARDEN_PREF_BONUS', 1.5, float)
PREF_BONUS2 = _env('WARDEN_PREF_BONUS2', 0.7, float)
DEADEND_W = _env('WARDEN_DEADEND_W', 0.0, float)
COIN_FIRST = bool(_env('WARDEN_COIN_FIRST', 0, int, 0, 1))
SINGLE_CRATE = bool(_env('WARDEN_SINGLE_CRATE', 1, int, 0, 1))
SINGLE_CRATE_DIST = _env('WARDEN_SINGLE_CRATE_DIST', 2, int, 0, 4)
MULTI_CRATE_DIST = _env('WARDEN_MULTI_CRATE_DIST', 3, int, 0, 6)
WAIT_PENALTY = _env('WARDEN_WAIT_PENALTY', 0.0, float)
PLANT_ESC = _env('WARDEN_PLANT_ESC', 1, int, 1, 4)
FLEE_OPP_W = _env('WARDEN_FLEE_OPP_W', 0.0, float)
OPP_AVOID_W = _env('WARDEN_OPP_AVOID_W', 0.0, float)
CRATE_GUARD_DIST = _env('WARDEN_CRATE_GUARD_DIST', 0, int, 0, 6)


def setup(self):
    import os
    if os.environ.get('WARDEN_SEED'):
        seed = int(os.environ['WARDEN_SEED'])
    else:
        seed = int.from_bytes(os.urandom(4), 'little')
    self._seed = int(seed) & 0x7fffffff
    self.coord_hist = deque([], 24)
    self.bomb_hist = deque([], 5)
    self.current_round = -1
    self._rng = np.random.default_rng(self._seed)


def _make_rng(seed, round_id, step):
    return np.random.default_rng((int(seed) * 1000003 ^ int(round_id) * 9176
                                  ^ int(step) * 7919) & 0xffffffff)


def _bfs_first_steps(start, arena, bomb_cells):
    """BFS over free tiles. Returns (dist, first_step) maps."""
    w, h = arena.shape[0], arena.shape[1]
    dist = np.full((w, h), 10 ** 9, dtype=np.int32)
    first = {}
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < w and 0 <= sy < h):
        return dist, first
    dist[sx, sy] = 0
    queue = deque([(sx, sy)])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _DIR4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if dist[nx, ny] != 10 ** 9:
                continue
            if arena[nx, ny] != 0 or (nx, ny) in bomb_cells:
                continue
            dist[nx, ny] = dist[cx, cy] + 1
            first[(nx, ny)] = first.get((cx, cy), (dx, dy))
            queue.append((nx, ny))
    return dist, first


def _dist_to(dist, first, targets):
    """Shortest BFS distance and first step toward any target tile or
    its best free neighbour (blocked targets handled)."""
    best_d, best_step = 10 ** 9, None
    for tx, ty in targets:
        tx, ty = int(tx), int(ty)
        if not (0 <= tx < dist.shape[0] and 0 <= ty < dist.shape[1]):
            continue
        if dist[tx, ty] < 10 ** 9:
            if dist[tx, ty] == 0:
                return 0, (0, 0)
            if dist[tx, ty] < best_d:
                best_d, best_step = int(dist[tx, ty]), first.get((tx, ty))
            continue
        for dx, dy in _DIR4:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dist.shape[0] and 0 <= ny < dist.shape[1]:
                if dist[nx, ny] < 10 ** 9 and int(dist[nx, ny]) + 1 < best_d:
                    best_d = int(dist[nx, ny]) + 1
                    best_step = first.get((nx, ny))
    return best_d, best_step


def parse(game_state):
    arena = np.asarray(game_state['field'])
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state.get('bombs', []) or []
    coins = [(int(a), int(b)) for (a, b) in (game_state.get('coins', []) or [])]
    others = [(int(pxy[0]), int(pxy[1]))
              for (_, _, _, pxy) in (game_state.get('others', []) or [])]
    step = int(game_state.get('step', 0))
    explosion_map = game_state.get('explosion_map', None)
    return {'arena': arena, 'x': int(x), 'y': int(y), 'score': int(score),
            'bombs_left': bool(bombs_left), 'bombs': bombs, 'coins': coins,
            'others': others, 'step': step, 'explosion_map': explosion_map,
            'round': int(game_state.get('round', 0))}


def decide(st, bomb_hist=()):
    """Fast heuristic decision (v1 rules + v2 corrections).

    Returns (want_bomb, aux); aux carries the derived state for the
    search layer (danger, escape sets, targets, bomb payoff)."""
    arena, x, y = st['arena'], st['x'], st['y']
    bombs, others, coins = st['bombs'], st['others'], st['coins']
    bombs_left = st['bombs_left']
    bomb_cells = set((int(b[0][0]), int(b[0][1])) for b in bombs)
    other_cells = set(others)

    valid = S.valid_mask(arena, (x, y), bombs_left, bombs, others,
                         st['explosion_map'])
    danger = S.future_danger(arena, bombs, st['explosion_map'], HORIZON)
    safe_first, esc_dist = S.escape_bfs((x, y), arena, bombs, others,
                                        danger, HORIZON)
    safe_moves = set()
    for d, ok in safe_first.items():
        if ok and d in _DIR_TO_ACTION:
            safe_moves.add(_DIR_TO_ACTION[d])
    must_flee = bool(danger[0, x, y] or danger[1, x, y])

    # hypothetical own bomb: proven escape + distance
    dh = S.with_hypothetical_bomb(danger, arena, x, y, HORIZON)
    bombs_h = list(bombs) + [((x, y), _BOMB_TIMER)]
    sh, hyp_dist = S.escape_bfs((x, y), arena, bombs_h, others, dh, HORIZON)
    n_esc = sum(1 for d in _DIR4 if sh.get(d, False))
    can_escape_bomb = n_esc >= PLANT_ESC

    w, h = arena.shape[0], arena.shape[1]
    crates = [(cx, cy) for cx in range(w) for cy in range(h)
              if arena[cx, cy] == 1]
    crate_adj = set()
    for cx, cy in crates:
        for dx, dy in _DIR4:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and arena[nx, ny] == 0:
                crate_adj.add((nx, ny))
    dist, first = _bfs_first_steps((x, y), arena, bomb_cells)

    _, coin_step = _dist_to(dist, first, coins)
    _, crate_step = _dist_to(dist, first, sorted(crate_adj) or crates)
    _, opp_step = _dist_to(dist, first, others)

    loot = len(crates) + len(coins)
    hunt = bool(others) and (loot <= 6 or st['step'] > 200 or
                             min(abs(ox - x) + abs(oy - y)
                                 for ox, oy in others) <= 3)

    blast_now = set(S.true_blast(arena, x, y))
    opps_hit = sum(1 for o in others if o in blast_now)
    crates_hit = sum(1 for c in crates if c in blast_now)
    want_bomb = False
    trap_kill = False
    guard = (valid['BOMB'] and can_escape_bomb and not must_flee
             and (x, y) not in list(bomb_hist)[-3:])
    if guard and CRATE_GUARD_DIST > 0 and opps_hit == 0:
        dmin = min(abs(ox - x) + abs(oy - y) for (ox, oy) in others) \
            if others else 99
        if dmin <= CRATE_GUARD_DIST:
            guard = False
    if guard:
        if opps_hit > 0:
            want_bomb = True
        elif crates_hit >= 2 and hyp_dist <= MULTI_CRATE_DIST:
            want_bomb = True
        elif SINGLE_CRATE and crates_hit == 1 and hyp_dist <= SINGLE_CRATE_DIST:
            want_bomb = True
    if OPP_TRAP and valid['BOMB'] and can_escape_bomb and not must_flee:
        danger_ne = S.future_danger(arena, bombs, None, HORIZON)
        for o in others:
            if o in blast_now:
                can, _ = S.opp_can_escape(arena, bombs, o, (x, y), others,
                                          HORIZON, danger=danger_ne)
                if not can:
                    trap_kill = True
                    want_bomb = True
                    break

    preferred = []
    if COIN_FIRST:
        if coins and coin_step in _DIR_TO_ACTION:
            preferred.append(_DIR_TO_ACTION[coin_step])
        if crate_adj and crate_step in _DIR_TO_ACTION:
            preferred.append(_DIR_TO_ACTION[crate_step])
        elif crates and crate_step in _DIR_TO_ACTION:
            preferred.append(_DIR_TO_ACTION[crate_step])
        if hunt and opp_step in _DIR_TO_ACTION:
            preferred.append(_DIR_TO_ACTION[opp_step])
    else:
        if hunt and opp_step in _DIR_TO_ACTION:
            preferred.append(_DIR_TO_ACTION[opp_step])
        if coins and coin_step in _DIR_TO_ACTION:
            preferred.append(_DIR_TO_ACTION[coin_step])
        if crate_adj and crate_step in _DIR_TO_ACTION:
            preferred.append(_DIR_TO_ACTION[crate_step])
        elif crates and crate_step in _DIR_TO_ACTION:
            preferred.append(_DIR_TO_ACTION[crate_step])

    aux = {'danger': danger, 'safe_moves': safe_moves, 'valid': valid,
           'must_flee': must_flee, 'safe_first': safe_first,
           'hyp_dist': hyp_dist, 'can_escape_bomb': can_escape_bomb,
           'blast_now': blast_now, 'opps_hit': opps_hit,
           'crates_hit': crates_hit, 'trap_kill': trap_kill,
           'preferred': preferred, 'hunt': hunt, 'dist': dist,
           'crate_adj': crate_adj, 'crates': crates,
           'bomb_cells': bomb_cells, 'other_cells': other_cells}
    return want_bomb, aux


def choose_fast(st, aux, coord_hist, want_bomb):
    """v1 move scoring on top of decide()'s derived state."""
    arena, x, y = st['arena'], st['x'], st['y']
    danger = aux['danger']
    valid = aux['valid']

    def loop_penalty(action):
        dx, dy = _DELTAS[action]
        tile = (x + dx, y + dy)
        count = list(coord_hist).count(tile)
        return (-0.6 if count >= 3 else (-0.2 if count == 2 else 0.0)) * LOOP_W

    def danger_cost(action):
        dx, dy = _DELTAS[action]
        tx, ty = x + dx, y + dy
        if not (0 <= tx < danger.shape[1] and 0 <= ty < danger.shape[2]):
            return -100.0
        for t in range(danger.shape[0]):
            if danger[t, tx, ty]:
                if t <= 1:
                    return -100.0
                if t <= 3:
                    return -2.0
                break
        return 0.0

    def mobility(action):
        if MOBILITY_W <= 0.0 and DEADEND_W <= 0.0:
            return 0.0
        dx, dy = _DELTAS[action]
        tx, ty = x + dx, y + dy
        n = 0
        for ddx, ddy in _DIR4:
            nx, ny = tx + ddx, ty + ddy
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] \
                    and arena[nx, ny] == 0 and (nx, ny) not in aux['bomb_cells']:
                n += 1
        out = MOBILITY_W * min(4, n)
        if DEADEND_W > 0 and n <= 1:
            out -= DEADEND_W
        return out

    def flee_term(action):
        if FLEE_OPP_W <= 0.0 or not st['others']:
            return 0.0
        dx, dy = _DELTAS[action]
        tx, ty = x + dx, y + dy
        d = min(abs(ox - tx) + abs(oy - ty) for (ox, oy) in st['others'])
        return FLEE_OPP_W * min(3, d)

    def avoid_term(action):
        if OPP_AVOID_W <= 0.0 or not st['others']:
            return 0.0
        dx, dy = _DELTAS[action]
        tx, ty = x + dx, y + dy
        n = sum(1 for (ox, oy) in st['others']
                if abs(ox - tx) + abs(oy - ty) <= 1)
        return -OPP_AVOID_W * n

    preferred = aux['preferred']
    candidates = []
    for action in _MOVES:
        if not valid.get(action):
            continue
        score = 0.0
        if action in aux['safe_moves']:
            score += SAFE_BONUS
        if preferred and action == preferred[0]:
            score += PREF_BONUS
        elif action in preferred:
            score += PREF_BONUS2
        score += danger_cost(action)
        score += loop_penalty(action)
        score += mobility(action)
        score += avoid_term(action)
        if action == 'WAIT':
            score -= WAIT_PENALTY
        if aux['must_flee']:
            score += flee_term(action)
            if action == 'WAIT':
                score -= 5.0
        candidates.append((score, action))
    if aux['must_flee']:
        safe_candidates = [(sc, a) for sc, a in candidates
                           if a in aux['safe_moves']]
        if safe_candidates:
            candidates = safe_candidates
    return candidates


def act(self, game_state):
    st = parse(game_state)
    if st['round'] != self.current_round:
        self.current_round = st['round']
        self.coord_hist = deque([], 24)
        self.bomb_hist = deque([], 5)
        self._rng = _make_rng(self._seed, st['round'], 0)

    want_bomb, aux = decide(st, self.bomb_hist)
    action = None
    if SEARCH != 'off':
        try:
            from . import search as SR
            action = SR.override(game_state, st, aux, self)
        except Exception:
            action = None

    if action is None:
        if want_bomb:
            action = 'BOMB'
        else:
            candidates = choose_fast(st, aux, self.coord_hist, want_bomb)
            if candidates:
                best = max(sc for sc, _ in candidates)
                tied = sorted(a for sc, a in candidates if sc == best)
                action = tied[int(self._rng.integers(len(tied)))]
            else:
                action = 'WAIT'

    if action == 'BOMB':
        self.bomb_hist.append((st['x'], st['y']))
        self.coord_hist.append((st['x'], st['y']))
    elif action in _DELTAS:
        dx, dy = _DELTAS[action]
        self.coord_hist.append((st['x'] + dx, st['y'] + dy))
    return action
