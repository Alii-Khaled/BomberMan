# Warden v2 — shared hazard / escape / validity helpers (numpy only).
#
# Adapted from warden_v1's inline helpers with the E88 escape-solver
# correctness fix (latest-lethal test + arrival check) and boolean
# time-axis danger (non-contiguous windows handled exactly). Geometry
# and timing replicate environment.py exactly:
#   bomb dropped at t -> detonates during step t+4 -> lethal t+4, t+5.
#   explosion_map > 0 marks a live blast at t=0 and t=1 (conservative:
#   the timer-1 encoding loses the final dangerous frame).
from collections import deque

import numpy as np

BOMB_TIMER = 4
POWER = 3


def _env_int(name, default, lo, hi):
    import os
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


HORIZON = _env_int('WARDEN_HORIZON', 8, 4, 14)


def true_blast(arena, x, y, power=POWER):
    """== items.Bomb.get_blast_coords (stops at stone walls only)."""
    coords = [(int(x), int(y))]
    W, H = arena.shape[0], arena.shape[1]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = int(x) + dx * i, int(y) + dy * i
            if not (0 <= nx < W and 0 <= ny < H):
                break
            if arena[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def future_danger(arena, bombs, explosion_map, horizon=HORIZON,
                  bomb_timer=BOMB_TIMER, power=POWER):
    """danger[t, x, y] for t=0..horizon (bool)."""
    W, H = arena.shape[0], arena.shape[1]
    danger = np.zeros((horizon + 1, W, H), dtype=bool)
    if explosion_map is not None:
        exp = np.asarray(explosion_map) > 0
        if exp.shape == (W, H):
            danger[0] |= exp
            if horizon >= 1:
                danger[1] |= exp
    for (xb, yb), t in (bombs or []):
        try:
            t = int(t)
        except Exception:
            t = bomb_timer
        for dt in (t, t + 1):
            if 0 <= dt <= horizon:
                for (cx, cy) in true_blast(arena, int(xb), int(yb), power):
                    danger[dt, cx, cy] = True
    return danger


def with_hypothetical_bomb(danger, arena, x, y, horizon=HORIZON,
                           bomb_timer=BOMB_TIMER, power=POWER):
    out = danger.copy()
    for dt in (bomb_timer, bomb_timer + 1):
        if 0 <= dt <= horizon:
            for (cx, cy) in true_blast(arena, x, y, power):
                out[dt, cx, cy] = True
    return out


def last_lethal(danger, horizon=HORIZON):
    """t of latest lethal frame per tile, -1 if safe through horizon."""
    d = np.asarray(danger)
    any_d = d.any(axis=0)
    return np.where(any_d, horizon - d[::-1].argmax(axis=0), -1).astype(np.int32)


def escape_bfs(pos, arena, bombs, others_xy, danger, horizon=HORIZON):
    """Time-expanded BFS (E88 semantics). Returns (safe_first, dist).

    safe_first[delta] True if that first step leads to a path that
    survives to the horizon or reaches a tile with no future danger.
    dist = steps to the nearest fully-safe tile (inf if none).
    """
    x0, y0 = int(pos[0]), int(pos[1])
    W, H = arena.shape[0], arena.shape[1]
    if not (0 <= x0 < W and 0 <= y0 < H):
        return {}, float('inf')

    ar = np.asarray(arena)
    free_b = (ar == 0).reshape(-1).tobytes()
    dng = np.asarray(danger, dtype=bool)
    dang_b = dng.reshape(-1).tobytes()
    ll_flat = last_lethal(dng, horizon).reshape(-1).tolist()
    plane = W * H

    bomb_cells = set()
    for (xy, _t) in (bombs or []):
        bomb_cells.add((int(xy[0]), int(xy[1])))
    other_cells = set((int(a), int(b)) for (a, b) in (others_xy or []))

    moves = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
    t1 = 1 if horizon >= 1 else 0
    n_t = horizon + 1
    visited = bytearray(W * H * n_t)
    visited[(x0 * H + y0) * n_t + 0] = 1

    q = deque()
    for mi, (dx, dy) in enumerate(moves):
        nx, ny = x0 + dx, y0 + dy
        if not (0 <= nx < W and 0 <= ny < H):
            continue
        key = (nx, ny)
        if not free_b[nx * H + ny]:
            continue
        if key in bomb_cells:
            continue
        if key in other_cells and key != (x0, y0):
            continue
        vi = (nx * H + ny) * n_t + t1
        if visited[vi]:
            continue
        visited[vi] = 1
        q.append((nx, ny, t1, mi))

    safe_first = set()
    found_safe = set()
    dist_to_safe = float('inf')

    while q:
        cx, cy, ct, fmi = q.popleft()
        ci = cx * H + cy
        if dang_b[ct * plane + ci]:
            continue
        future_hit = ll_flat[ci] >= ct
        if not future_hit:
            found_safe.add(fmi)
            if ct < dist_to_safe:
                dist_to_safe = ct
        if ct == horizon:
            safe_first.add(fmi)
            continue
        if ct >= 1 and not future_hit:
            safe_first.add(fmi)
        nt = ct + 1
        di = nt * plane
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            nci = nx * H + ny
            if not free_b[nci]:
                continue
            key = (nx, ny)
            if key in bomb_cells:
                continue
            if key in other_cells and key != (x0, y0):
                continue
            if dang_b[di + nci]:
                continue
            vi = nci * n_t + nt
            if visited[vi]:
                continue
            visited[vi] = 1
            q.append((nx, ny, nt, fmi))

    result = {}
    for mi, d in enumerate(moves):
        result[d] = (mi in safe_first) or (mi in found_safe)
    return result, dist_to_safe


def opp_can_escape(arena, bombs, opp_pos, bomb_pos, others_xy=None,
                   horizon=HORIZON, bomb_timer=BOMB_TIMER, power=POWER,
                   danger=None):
    """Can an opponent survive a bomb planted at bomb_pos? (certified kill
    when False). WAIT counts as a survivable first move."""
    ox, oy = int(opp_pos[0]), int(opp_pos[1])
    bx, by = int(bomb_pos[0]), int(bomb_pos[1])
    if danger is None:
        danger = future_danger(arena, bombs, None, horizon, bomb_timer, power)
    danger = with_hypothetical_bomb(danger, arena, bx, by, horizon,
                                    bomb_timer, power)
    bombs_h = list(bombs or []) + [((bx, by), bomb_timer)]
    others = [o for o in (others_xy or []) if (int(o[0]), int(o[1])) != (ox, oy)]
    safe, dist = escape_bfs((ox, oy), arena, bombs_h, others, danger, horizon)
    if safe.get((0, 0), False):
        return True, dist
    return any(safe.get(d, False) for d in
               ((0, -1), (0, 1), (-1, 0), (1, 0))), dist


def tile_free(arena, x, y, bomb_cells, other_cells):
    W, H = arena.shape[0], arena.shape[1]
    if not (0 <= x < W and 0 <= y < H):
        return False
    if arena[x, y] != 0:
        return False
    if (x, y) in bomb_cells:
        return False
    if (x, y) in other_cells:
        return False
    return True


def valid_mask(arena, self_xy, bombs_left, bombs, others, explosion_map):
    """Engine-valid action flags for our agent."""
    x, y = int(self_xy[0]), int(self_xy[1])
    bomb_cells = set((int(b[0][0]), int(b[0][1])) for b in bombs)
    other_cells = set((int(o[0]), int(o[1])) for o in others) - {(x, y)}
    valid = {}
    for action, (dx, dy) in (('UP', (0, -1)), ('DOWN', (0, 1)),
                             ('LEFT', (-1, 0)), ('RIGHT', (1, 0))):
        valid[action] = tile_free(arena, x + dx, y + dy, bomb_cells, other_cells)
    wait_ok = (x, y) not in bomb_cells
    try:
        wait_ok = wait_ok and not (np.asarray(explosion_map)[x, y] >= 1)
    except Exception:
        pass
    valid['WAIT'] = bool(wait_ok)
    try:
        bomb_ok = bool(bombs_left) and not (np.asarray(explosion_map)[x, y] >= 1)
    except Exception:
        bomb_ok = bool(bombs_left)
    valid['BOMB'] = bool(bomb_ok)
    return valid
