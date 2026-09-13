# Unseen sparring agents — EVAL ONLY, never shipped in a tournament zip.
# unseen_bomber: drops a bomb on cooldown whenever a <=3-step escape path
# exists, then FLEES to safety before wandering. Behavior axis:
# reckless-aggressive volume bomber (imperfect, not suicidal).
"""Bomb-every-cooldown sparring agent (eval-only outsider)."""
from collections import deque

import numpy as np

ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0),
         'WAIT': (0, 0)}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def setup(self):
    self.logger.info('unseen_bomber setup (eval-only outsider)')


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
    for dx, dy in DIRS4:
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


def _first_step_toward(arena, x, y, bombs, occupied, goals, blocked_extra,
                       depth):
    """First action of the shortest free path to any goal cell (or None)."""
    gset = set(goals)
    if not gset or (x, y) in gset:
        return None
    prev = {(x, y): None}
    act_of = {}
    q = deque([(x, y)])
    while q and len(prev) < 600:
        cx, cy = q.popleft()
        d = 0
        for dx, dy in DIRS4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]):
                continue
            if (nx, ny) in prev or (nx, ny) in blocked_extra:
                continue
            if not _free(arena, nx, ny, bombs, occupied):
                continue
            prev[(nx, ny)] = (cx, cy)
            act_of[(nx, ny)] = {(1, 0): 'RIGHT', (-1, 0): 'LEFT',
                                (0, 1): 'DOWN', (0, -1): 'UP'}[(dx, dy)]
            if (nx, ny) in gset:
                cur, first = (nx, ny), None
                while prev[cur] is not None:
                    first = act_of[cur]
                    cur = prev[cur]
                return first
            if len(prev) < depth * 8 + 40:
                q.append((nx, ny))
    return None


def act(self, game_state):
    try:
        arena = game_state['field']
        W, H = arena.shape
        _, _, bomb_allowed, (x, y) = game_state['self']
        bombs = {tuple(b[0]): b[1] for b in game_state['bombs']}
        exp = game_state['explosion_map']
        coins = {tuple(c) for c in game_state['coins']}
        occupied = {tuple(o[3]) for o in game_state['others']}

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

        rng = _rng(game_state['round'], game_state['step'], 'bomber')

        def safe(nx, ny):
            return (_free(arena, nx, ny, bombs, occupied)
                    and (nx, ny) not in danger)

        def ok(nx, ny):
            return (_free(arena, nx, ny, bombs, occupied)
                    and (nx, ny) not in lethal)

        # 1) plant if allowed and a <=3-step exit path exists
        if bomb_allowed:
            new_blast = _bomb_blast(arena, x, y)
            # goal: any safe tile outside the blast reachable in <=3 steps
            seen = {(x, y)}
            q = deque([(x, y, 0)])
            plant = False
            while q:
                cx, cy, d = q.popleft()
                if d >= 3:
                    continue
                for dx, dy in DIRS4:
                    nx, ny = cx + dx, cy + dy
                    if not _free(arena, nx, ny, bombs, occupied) \
                            or (nx, ny) in seen:
                        continue
                    if (nx, ny) in lethal or (nx, ny) in risky:
                        continue
                    if (nx, ny) not in new_blast:
                        plant = True
                        break
                    seen.add((nx, ny))
                    q.append((nx, ny, d + 1))
                if plant:
                    break
            if plant:
                return 'BOMB'

        # 2) flee if any danger is on the board: run to the nearest tile
        # that is currently safe (outside all blast zones)
        if danger:
            goals = [(cx, cy) for cx in range(W) for cy in range(H)
                     if _free(arena, cx, cy, bombs, occupied)
                     and (cx, cy) not in danger]
            step = _first_step_toward(arena, x, y, bombs, occupied,
                                      goals, set(), 7)
            if step is not None and ok(x + DELTA[step][0], y + DELTA[step][1]):
                return step
            # danger but no path: take any non-lethal step
            opts = [a for a in ACTIONS if ok(x + DELTA[a][0], y + DELTA[a][1])]
            if opts:
                return opts[rng.integers(0, len(opts))]
            return 'WAIT'

        # 3) wander: slight coin pull, else random walk
        opts = [a for a in ACTIONS if safe(x + DELTA[a][0], y + DELTA[a][1])]
        if not opts:
            opts = [a for a in ACTIONS if ok(x + DELTA[a][0], y + DELTA[a][1])]
        if not opts:
            return 'WAIT'
        scored = []
        for a in opts:
            nx, ny = x + DELTA[a][0], y + DELTA[a][1]
            s = rng.random() * 2.0
            if (nx, ny) in coins:
                s += 5.0
            scored.append((s, a))
        scored.sort(reverse=True)
        return scored[0][1]
    except Exception:
        return 'WAIT'
