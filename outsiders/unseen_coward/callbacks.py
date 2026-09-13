# Unseen sparring agents — EVAL ONLY, never shipped in a tournament zip.
# unseen_coward: never bombs; maximizes distance from opponents and bombs,
# prefers edges/corners. Behavior axis: passive-defensive (unlike
# rule_based/warden/collector — no balancing, no hunting, no economy).
"""Coward-fleer sparring agent (eval-only outsider)."""
import os
from collections import deque

import numpy as np

ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0),
         'WAIT': (0, 0)}


def setup(self):
    self.logger.info('unseen_coward setup (eval-only outsider)')
    self.coord_hist = []


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


def _blast_cells(arena, bombs):
    """Cells covered by any current bomb blast (per-bomb cross, stop at walls)."""
    cells = set()
    for (bx, by), _t in bombs:
        cells.add((bx, by))
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


def _threat_dists(arena, threats):
    """BFS dist map to nearest threat cell (threats = list of (x, y))."""
    W, H = arena.shape
    dist = np.full((W, H), 999, dtype=np.int32)
    q = deque()
    for (x, y) in threats:
        if 0 <= x < W and 0 <= y < H and dist[x, y] > 0:
            dist[x, y] = 0
            q.append((x, y))
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and arena[nx, ny] == 0 \
                    and dist[nx, ny] == 999:
                dist[nx, ny] = dist[cx, cy] + 1
                q.append((nx, ny))
    return dist


def act(self, game_state):
    try:
        arena = game_state['field']
        W, H = arena.shape
        _, _, bomb_allowed, (x, y) = game_state['self']
        bombs = {tuple(b[0]): b[1] for b in game_state['bombs']}
        exp = game_state['explosion_map']
        others = [tuple(o[3]) for o in game_state['others']]

        # lethal = live explosions + imminent (timer<=1) blasts;
        # longer fuses are 'risky' (penalized, not forbidden)
        lethal, risky = set(), set()
        for cx in range(W):
            for cy in range(H):
                if exp[cx, cy] > 0:
                    lethal.add((cx, cy))
        for (bx, by), t in bombs.items():
            cells = _blast_cells(arena, [((bx, by), t)])
            if t <= 1:
                lethal |= cells
            else:
                risky |= cells

        rng = _rng(game_state['round'], game_state['step'], 'coward')
        occ = {tuple(o) for o in others}
        opts = []
        for a in ACTIONS:
            dx, dy = DELTA[a]
            nx, ny = x + dx, y + dy
            if not _free(arena, nx, ny, bombs, occ) or (nx, ny) in lethal:
                continue
            risky_pen = 6.0 if (nx, ny) in risky else 0.0
            # mobility proxy: free neighbors of the target
            mob = sum(1 for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if _free(arena, nx + ddx, ny + ddy, bombs, occ))
            if mob == 0 and a != 'WAIT':
                continue
            opts.append((a, nx, ny, mob, risky_pen))
        if not opts:
            return 'WAIT'

        # score: far from opponents (BFS dist), far from bombs, edge bonus
        opp_d = _threat_dists(arena, occ) if occ else None
        bomb_xy = list(bombs.keys())
        bomb_d = _threat_dists(arena, bomb_xy) if bomb_xy else None
        best, best_s = None, None
        for (a, nx, ny, mob, risky_pen) in opts:
            s = mob * 0.5
            if opp_d is not None:
                s += min(int(opp_d[nx, ny]), 12) * 1.0
            if bomb_d is not None:
                s += min(int(bomb_d[nx, ny]), 12) * 1.5
            s += (0 if 0 < nx < W - 1 and 0 < ny < H - 1 else 1.5)
            s -= risky_pen
            s += rng.random() * 0.1
            if best_s is None or s > best_s:
                best, best_s = a, s
        # loop breaker: penalize immediate back-and-forth
        if len(self.coord_hist) >= 2 and best is not None:
            if self.coord_hist[-1] == (x, y):
                rng.shuffle(opts)
                for (a, nx, ny, _m, _r) in opts:
                    if (nx, ny) != self.coord_hist[-2]:
                        self.coord_hist.append((x, y))
                        return a
        self.coord_hist.append((x, y))
        if len(self.coord_hist) > 8:
            self.coord_hist = self.coord_hist[-8:]
        return best if best is not None else 'WAIT'
    except Exception:
        return 'WAIT'
