# Warden v1 — outsider sparring agent (heuristic, non-learning)
#
# Source of truth lives in outsiders/warden_v1/; agent_code/warden_v1 is a
# symlink so the game loader finds it. See outsiders/README.md.
#
# Design (written from scratch for this project):
#   * wall-aware blast computation (blast stops at stone walls)
#   * timer-based danger map over a short horizon, not just ticking bombs
#   * time-expanded escape search before every BOMB decision
#   * target priority: coins -> crate-adjacent tiles -> opponents (late hunt)
#   * strict bomb discipline: only with proven escape route + real payoff
#   * loop avoidance via short coordinate history
#
# No torch, no multiprocessing, no absolute paths. Fast (<10ms/step).

from collections import deque

import numpy as np

_BOMB_TIMER = 4
_BLAST_POWER = 3
_HORIZON = 6

_DELTAS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
    'WAIT': (0, 0),
}


def setup(self):
    np.random.seed()
    self.coord_hist = deque([], 24)
    self.bomb_hist = deque([], 5)
    self.current_round = 0


def _blast_tiles(arena, x, y, power=_BLAST_POWER):
    """Tiles hit by a bomb at (x, y). Stops at stone walls and board edges."""
    tiles = [(x, y)]
    w, h = arena.shape[0], arena.shape[1]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < w and 0 <= ny < h):
                break
            if arena[nx, ny] == -1:
                break
            tiles.append((nx, ny))
    return tiles


def _danger_timeline(arena, bombs, explosion_map, horizon=_HORIZON):
    """Map tile -> earliest step at which it is lethal (inf if safe).

    Bombs kill at their timer step and one step after (lingering explosion).
    Active explosions kill now and next step.
    """
    danger = {}
    if explosion_map is not None:
        exp = np.asarray(explosion_map)
        if exp.shape == arena.shape:
            for x in range(arena.shape[0]):
                for y in range(arena.shape[1]):
                    if exp[x, y] > 0:
                        danger[(x, y)] = 0
    for (xb, yb), t in (bombs or []):
        try:
            t = int(t)
        except Exception:
            t = _BOMB_TIMER
        for step in (t, t + 1):
            if 0 <= step <= horizon:
                for tile in _blast_tiles(arena, int(xb), int(yb)):
                    if tile not in danger or step < danger[tile]:
                        danger[tile] = step
    return danger


def _free(arena, x, y, bomb_cells, other_cells):
    w, h = arena.shape[0], arena.shape[1]
    if not (0 <= x < w and 0 <= y < h):
        return False
    if arena[x, y] != 0:
        return False
    if (x, y) in bomb_cells:
        return False
    if (x, y) in other_cells:
        return False
    return True


def _escape_search(start, arena, bomb_cells, other_cells, danger, horizon=_HORIZON):
    """Time-expanded BFS. Returns (safe_first_moves, best_dist).

    safe_first_moves: set of first-step deltas leading to a survivable path.
    best_dist: steps to the nearest tile with no known future danger (inf).
    """
    sx, sy = int(start[0]), int(start[1])
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
    visited = set()
    queue = deque()
    # seed first steps (t=1); start tile itself may already be lethal
    for dx, dy in moves:
        nx, ny = sx + dx, sy + dy
        if (nx, ny) == (sx, sy):
            blocked = False
        else:
            blocked = not _free(arena, nx, ny, bomb_cells, other_cells - {(sx, sy)})
        if blocked:
            continue
        if (nx, ny, 1) in visited:
            continue
        visited.add((nx, ny, 1))
        queue.append((nx, ny, 1, (dx, dy)))
    safe_first = set()
    best_dist = float('inf')
    while queue:
        cx, cy, ct, first = queue.popleft()
        lethal_at = danger.get((cx, cy), float('inf'))
        if lethal_at == float('inf') or ct < lethal_at:
            # reaches a tile with no future danger -> safe line found
            if lethal_at == float('inf'):
                safe_first.add(first)
                if ct < best_dist:
                    best_dist = ct
                continue
        if ct >= horizon:
            # survived the whole horizon along this path
            safe_first.add(first)
            continue
        if ct >= lethal_at:
            continue  # died on this tile, do not expand
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            nt = ct + 1
            if nt > horizon or (nx, ny, nt) in visited:
                continue
            if not _free(arena, nx, ny, bomb_cells, other_cells - {(sx, sy)}):
                if (nx, ny) != (cx, cy):
                    continue
            if nt >= danger.get((nx, ny), float('inf')):
                continue  # tile detonates on arrival: dead path
            visited.add((nx, ny, nt))
            queue.append((nx, ny, nt, first))
    return safe_first, best_dist


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
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
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


def _dist_to(dist, first, targets, start):
    """Shortest BFS distance and first step toward any target tile or neighbour."""
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
        # blocked target (crate/opponent): approach the best free neighbour
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dist.shape[0] and 0 <= ny < dist.shape[1]:
                if dist[nx, ny] < 10 ** 9 and int(dist[nx, ny]) + 1 < best_d:
                    best_d = int(dist[nx, ny]) + 1
                    best_step = first.get((nx, ny))
    return best_d, best_step


def act(self, game_state):
    if game_state.get('round', 0) != getattr(self, 'current_round', 0):
        self.coord_hist = deque([], 24)
        self.bomb_hist = deque([], 5)
        self.current_round = game_state.get('round', 0)

    arena = np.asarray(game_state['field'])
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state.get('bombs', []) or []
    coins = [(int(a), int(b)) for (a, b) in (game_state.get('coins', []) or [])]
    others = [(int(pxy[0]), int(pxy[1])) for (_, _, _, pxy) in (game_state.get('others', []) or [])]
    step = int(game_state.get('step', 0))
    explosion_map = game_state.get('explosion_map', None)

    bomb_cells = set((int(bxy[0]), int(bxy[1])) for (bxy, _) in bombs)
    other_cells = set(others)
    danger = _danger_timeline(arena, bombs, explosion_map)

    # --- valid actions (mirrors environment movement rules) ---
    valid = {}
    for action, (dx, dy) in _DELTAS.items():
        if action == 'BOMB':
            continue
        nx, ny = x + dx, y + dy
        ok = _free(arena, nx, ny, bomb_cells, other_cells - {(x, y)})
        if action == 'WAIT':
            ok = (x, y) not in bomb_cells
            try:
                ok = ok and not (np.asarray(explosion_map)[x, y] >= 1)
            except Exception:
                pass
        valid[action] = ok
    try:
        bomb_ok = bool(bombs_left) and not (np.asarray(explosion_map)[x, y] >= 1)
    except Exception:
        bomb_ok = bool(bombs_left)
    valid['BOMB'] = bomb_ok

    # --- escape analysis ---
    safe_first, _ = _escape_search((x, y), arena, bomb_cells, other_cells, danger)
    delta_to_action = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT',
                       (1, 0): 'RIGHT', (0, 0): 'WAIT'}
    safe_moves = set(delta_to_action[d] for d in safe_first if d in delta_to_action)

    # hypothetical own bomb: can we still get out?
    hyp_danger = dict(danger)
    for tile in _blast_tiles(arena, x, y):
        for s in (_BOMB_TIMER, _BOMB_TIMER + 1):
            if tile not in hyp_danger or s < hyp_danger[tile]:
                hyp_danger[tile] = s
    hyp_bombs = set(bomb_cells) | {(x, y)}
    hyp_safe, hyp_dist = _escape_search((x, y), arena, hyp_bombs, other_cells, hyp_danger)
    can_escape_bomb = any(d != (0, 0) for d in hyp_safe)

    # immediate threat: standing where it blows up now/next
    must_flee = danger.get((x, y), float('inf')) <= 1

    # --- targets ---
    w, h = arena.shape[0], arena.shape[1]
    crates = [(cx, cy) for cx in range(w) for cy in range(h) if arena[cx, cy] == 1]
    crate_adj = set()
    for cx, cy in crates:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and arena[nx, ny] == 0:
                crate_adj.add((nx, ny))
    dist, first = _bfs_first_steps((x, y), arena, bomb_cells)

    _, coin_step = _dist_to(dist, first, coins, (x, y))
    _, crate_step = _dist_to(dist, first, sorted(crate_adj) or crates, (x, y))
    _, opp_step = _dist_to(dist, first, others, (x, y))

    crates_coins_left = len(crates) + len(coins)
    hunt = bool(others) and (crates_coins_left <= 6 or step > 200 or
                             min(abs(ox - x) + abs(oy - y) for ox, oy in others) <= 3)

    # --- bomb decision (strict discipline) ---
    blast_now = set(_blast_tiles(arena, x, y))
    opps_hit = sum(1 for o in others if o in blast_now)
    crates_hit = sum(1 for c in crates if c in blast_now)
    want_bomb = False
    if valid['BOMB'] and can_escape_bomb and not must_flee and (x, y) not in list(self.bomb_hist)[-3:]:
        if opps_hit > 0:
            want_bomb = True
        elif crates_hit >= 2 and hyp_dist <= 3:
            want_bomb = True
        elif crates_hit == 1 and hyp_dist <= 2:
            want_bomb = True

    # --- move scoring ---
    step_to_action = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT',
                      (1, 0): 'RIGHT', (0, 0): 'WAIT'}
    preferred = []
    if hunt and opp_step in step_to_action:
        preferred.append(step_to_action[opp_step])
    if coins and coin_step in step_to_action:
        preferred.append(step_to_action[coin_step])
    if crate_adj and crate_step in step_to_action:
        preferred.append(step_to_action[crate_step])
    elif crates and crate_step in step_to_action:
        preferred.append(step_to_action[crate_step])

    def loop_penalty(action):
        dx, dy = _DELTAS[action]
        tile = (x + dx, y + dy)
        count = list(self.coord_hist).count(tile)
        return -0.6 if count >= 3 else (-0.2 if count == 2 else 0.0)

    def danger_cost(action):
        dx, dy = _DELTAS[action]
        tile = (x + dx, y + dy)
        lethal = danger.get(tile, float('inf'))
        if lethal <= 1:
            return -100.0
        if lethal <= 3:
            return -2.0
        return 0.0

    candidates = []
    for action in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
        if not valid.get(action):
            continue
        score = 0.0
        if action in safe_moves:
            score += 3.0
        if preferred and action == preferred[0]:
            score += 1.5
        elif action in preferred:
            score += 0.7
        score += danger_cost(action)
        score += loop_penalty(action)
        if must_flee and action == 'WAIT':
            score -= 5.0
        candidates.append((score, action))

    # fleeing overrides greed: drop non-safe moves when threatened
    if must_flee:
        safe_candidates = [(sc, a) for sc, a in candidates if a in safe_moves]
        if safe_candidates:
            candidates = safe_candidates

    if want_bomb:
        self.bomb_hist.append((x, y))
        return 'BOMB'
    if candidates:
        best_score = max(sc for sc, _ in candidates)
        tied = sorted(a for sc, a in candidates if sc == best_score)
        # deterministic tie-break with seeded RNG
        choice = tied[np.random.randint(len(tied))]
        dx, dy = _DELTAS[choice]
        if choice != 'WAIT':
            self.coord_hist.append((x + dx, y + dy))
        else:
            self.coord_hist.append((x, y))
        return choice
    self.coord_hist.append((x, y))
    return 'WAIT'
