# Unseen sparring agents — EVAL ONLY, never shipped in a tournament zip.
# unseen_racer: pure coin-runner — BFS to the nearest revealed coin every
# step, never bombs, ignores opponents except avoiding lethal cells.
# Behavior axis: score-obsessed racer (unlike coin_collector_agent, which
# bombs for coins and fights; unlike peaceful_agent, which is uniform
# random).
"""Coin-racer sparring agent (eval-only outsider)."""
from collections import deque

import numpy as np

ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0),
         'WAIT': (0, 0)}
STEP_OF = {(1, 0): 'RIGHT', (-1, 0): 'LEFT', (0, 1): 'DOWN', (0, -1): 'UP'}


def setup(self):
    self.logger.info('unseen_racer setup (eval-only outsider)')


def _rng(round_id, step, name):
    s = (round_id * 7919 + step * 104729
         + sum(ord(c) for c in name) * 31) % (2 ** 31)
    return np.random.default_rng(s)


def _free(arena, x, y, bombs, occupied=()):
    if not (0 <= x < arena.shape[0] and 0 <= y < arena.shape[1]):
        return False
    if arena[x, y] != 0:
        return False
    if (x, y) in bombs or (x, y) in occupied:
        return False
    return True


def _bomb_blast(arena, bx, by):
    cells = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = bx + dx, by + dy
        for _ in range(3):
            if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]):
                break
            if arena[nx, ny] == -1:
                break
            cells.add((nx, ny))
            if arena[nx, ny] == 1:
                break
            nx += dx
            ny += dy
    return cells


def _bfs_step(arena, blocked, start, targets):
    W, H = arena.shape
    tset = set(targets)
    if not tset or tuple(start) in tset:
        return None
    prev = {tuple(start): None}
    qa = {}
    q = deque([tuple(start)])
    while q:
        cx, cy = q.popleft()
        if len(prev) > 500:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if (nx, ny) in prev or arena[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            prev[(nx, ny)] = (cx, cy)
            qa[(nx, ny)] = STEP_OF[(dx, dy)]
            if (nx, ny) in tset:
                cur, first = (nx, ny), None
                while prev[cur] is not None:
                    first = qa[cur]
                    cur = prev[cur]
                return first
            q.append((nx, ny))
    return None


def act(self, game_state):
    try:
        arena = game_state['field']
        W, H = arena.shape
        _, _, _bomb_allowed, (x, y) = game_state['self']
        bombs = {tuple(b[0]): b[1] for b in game_state['bombs']}
        exp = game_state['explosion_map']
        coins = {tuple(c) for c in game_state['coins']}
        occupied = {tuple(o[3]) for o in game_state['others']}
        rng = _rng(game_state['round'], game_state['step'], 'racer')

        lethal, risky = set(), set()
        for cx in range(W):
            for cy in range(H):
                if exp[cx, cy] > 0:
                    lethal.add((cx, cy))
        for (bx, by), t in bombs.items():
            cells = _bomb_blast(arena, bx, by)
            if t <= 1:
                lethal |= cells
            else:
                risky |= cells
        danger = lethal | risky

        def safe_step(nx, ny):
            return (_free(arena, nx, ny, bombs, occupied)
                    and (nx, ny) not in danger)

        def ok_step(nx, ny):
            return (_free(arena, nx, ny, bombs, occupied)
                    and (nx, ny) not in lethal)

        # flee first: any danger on the board -> run to a safe tile
        if danger:
            goals = [(cx, cy) for cx in range(W) for cy in range(H)
                     if _free(arena, cx, cy, bombs, occupied)
                     and (cx, cy) not in danger]
            step = _bfs_step(arena, set(bombs.keys()) | occupied,
                             (x, y), goals)
            if step is not None and ok_step(x + DELTA[step][0],
                                            y + DELTA[step][1]):
                return step
            opts = [a for a in ACTIONS if ok_step(x + DELTA[a][0],
                                                  y + DELTA[a][1])]
            if opts:
                return opts[rng.integers(0, len(opts))]
            return 'WAIT'

        if coins:
            blocked = set(bombs.keys()) | occupied
            step = _bfs_step(arena, blocked, (x, y), coins)
            if step is not None and ok_step(x + DELTA[step][0],
                                            y + DELTA[step][1]):
                return step
        opts = [a for a in ACTIONS if safe_step(x + DELTA[a][0],
                                                y + DELTA[a][1])]
        if not opts:
            opts = [a for a in ACTIONS if ok_step(x + DELTA[a][0],
                                                  y + DELTA[a][1])]
        if not opts:
            return 'WAIT'
        return opts[rng.integers(0, len(opts))]
    except Exception:
        return 'WAIT'
