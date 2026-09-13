# Unseen sparring agents — EVAL ONLY, never shipped in a tournament zip.
# unseen_rusher: pure hunter — BFS to the nearest opponent every step,
# plants when an opponent stands on its blast line (with a <=3-step exit
# path), FLEES while bombs are live, resumes the chase after.
# Ignores coins entirely. Behavior axis: opponent-obsessed rusher.
"""Opponent-rusher sparring agent (eval-only outsider)."""
from collections import deque

import numpy as np

ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0),
         'WAIT': (0, 0)}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
STEP_OF = {(1, 0): 'RIGHT', (-1, 0): 'LEFT', (0, 1): 'DOWN', (0, -1): 'UP'}


def setup(self):
    self.logger.info('unseen_rusher setup (eval-only outsider)')


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
    gset = set(goals)
    if not gset or (x, y) in gset:
        return None
    prev = {(x, y): None}
    act_of = {}
    q = deque([(x, y)])
    while q and len(prev) < 600:
        cx, cy = q.popleft()
        for dx, dy in DIRS4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]):
                continue
            if (nx, ny) in prev or (nx, ny) in blocked_extra:
                continue
            if not _free(arena, nx, ny, bombs, occupied):
                continue
            prev[(nx, ny)] = (cx, cy)
            act_of[(nx, ny)] = STEP_OF[(dx, dy)]
            if (nx, ny) in gset:
                cur, first = (nx, ny), None
                while prev[cur] is not None:
                    first = act_of[cur]
                    cur = prev[cur]
                return first
            q.append((nx, ny))
    return None


def act(self, game_state):
    try:
        arena = game_state['field']
        W, H = arena.shape
        _, _, bomb_allowed, (x, y) = game_state['self']
        bombs = {tuple(b[0]): b[1] for b in game_state['bombs']}
        exp = game_state['explosion_map']
        others = [tuple(o[3]) for o in game_state['others']]
        occupied = {o for o in others}
        rng = _rng(game_state['round'], game_state['step'], 'rusher')

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

        def ok(nx, ny):
            return (_free(arena, nx, ny, bombs, occupied)
                    and (nx, ny) not in lethal)

        # 1) plant: opponent on the blast line (Manhattan<=3, same
        # row/col) and a <=3-step exit path exists
        if bomb_allowed and others:
            for (ox, oy) in others:
                if (ox == x and abs(oy - y) <= 3) or \
                        (oy == y and abs(ox - x) <= 3):
                    new_blast = _bomb_blast(arena, x, y)
                    seen = {(x, y)}
                    q = deque([(x, y, 0)])
                    plant = False
                    while q and not plant:
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
                        return 'BOMB'
                    break

        # 2) flee while danger exists
        if danger:
            goals = [(cx, cy) for cx in range(W) for cy in range(H)
                     if _free(arena, cx, cy, bombs, occupied)
                     and (cx, cy) not in danger]
            step = _first_step_toward(arena, x, y, bombs, occupied,
                                      goals, set(), 7)
            if step is not None and ok(x + DELTA[step][0], y + DELTA[step][1]):
                return step
            opts = [a for a in ACTIONS if ok(x + DELTA[a][0], y + DELTA[a][1])]
            if opts:
                return opts[rng.integers(0, len(opts))]
            return 'WAIT'

        # 3) chase: first step of the shortest path to the nearest
        # opponent (bombs never on field here, blocked set is trivial)
        step = _first_step_toward(arena, x, y, bombs, occupied,
                                  others, set(), 12)
        if step is not None:
            return step
        opts = [a for a in ACTIONS if ok(x + DELTA[a][0], y + DELTA[a][1])]
        if not opts:
            return 'WAIT'
        return opts[rng.integers(0, len(opts))]
    except Exception:
        return 'WAIT'
