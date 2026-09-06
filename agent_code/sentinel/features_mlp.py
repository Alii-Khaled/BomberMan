"""MLP features for sentinel (~46 dims, normalized). Numpy only."""
from collections import deque
import numpy as np

FEATURE_DIM = 46


def _bfs_dist_dir(arena, bombs, start):
    """BFS over free tiles (arena==0, not bomb). Returns dist map + first-step dir map."""
    W, H = arena.shape[0], arena.shape[1]
    bomb_set = set((int(xy[0]), int(xy[1])) for (xy, _t) in (bombs or []))
    dist = np.full((W, H), 9999, dtype=np.int32)
    first = np.full((W, H, 2), 0, dtype=np.int32)  # first step dx,dy from start
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < W and 0 <= sy < H):
        return dist, first
    dist[sx, sy] = 0
    q = deque([(sx, sy)])
    # order: UP DOWN LEFT RIGHT for determinism
    neigh = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    while q:
        cx, cy = q.popleft()
        for dx, dy in neigh:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if dist[nx, ny] != 9999:
                continue
            if arena[nx, ny] != 0:
                continue
            if (nx, ny) in bomb_set:
                continue
            dist[nx, ny] = dist[cx, cy] + 1
            if cx == sx and cy == sy:
                first[nx, ny] = (dx, dy)
            else:
                first[nx, ny] = first[cx, cy]
            q.append((nx, ny))
    return dist, first


def _nearest(dist, first, targets, start):
    best_d, best_dir = 9999, (0, 0)
    sx, sy = int(start[0]), int(start[1])
    for (tx, ty) in targets:
        tx, ty = int(tx), int(ty)
        W, H = dist.shape
        if not (0 <= tx < W and 0 <= ty < H):
            continue
        d = int(dist[tx, ty])
        # if target itself blocked (crate/opponent), use best neighbour dist+1
        if d >= 9999:
            cand = 9999
            cdir = (0, 0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < W and 0 <= ny < H and dist[nx, ny] < 9999:
                    nd = int(dist[nx, ny]) + 1
                    if nd < cand:
                        cand = nd
                        cdir = tuple(first[nx, ny]) if (nx, ny) != (sx, sy) else (dx * -1, dy * -1)
                        # fallback: point toward neighbour
                        if cdir == (0, 0):
                            # direction from start toward neighbour
                            cdir = (int(np.sign(nx - sx)), int(np.sign(ny - sy)))
            d, bdir = cand, cdir
        else:
            bdir = tuple(first[tx, ty]) if (tx, ty) != (sx, sy) else (0, 0)
        if d < best_d:
            best_d, best_dir = d, bdir
    return best_d, best_dir


def dir_to_onehot(d):
    m = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
    v = [0, 0, 0, 0]
    if tuple(d) in m:
        v[m[tuple(d)]] = 1
    return v


def state_to_features(game_state, safety_info=None):
    """Return float32[FEATURE_DIM]. None -> zeros (for terminal)."""
    if game_state is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    arena = np.asarray(game_state['field'])
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state.get('bombs', []) or []
    coins = game_state.get('coins', []) or []
    others = game_state.get('others', []) or []
    others_xy = [(int(xy[0]), int(xy[1])) for (n, s, b, xy) in others]
    exp_map = np.asarray(game_state.get('explosion_map', np.zeros_like(arena, dtype=float)))
    step = float(game_state.get('step', 0))

    W, H = arena.shape[0], arena.shape[1]

    # immediate 4 dirs: wall, crate, bomb, other, explosion, danger-now
    danger_now = None
    if safety_info is not None and 'danger' in safety_info:
        danger_now = safety_info['danger'][0]
    else:
        # cheap approx: explosion + bomb pos
        danger_now = (exp_map >= 1)
    bomb_set = set((int(xy[0]), int(xy[1])) for (xy, _t) in bombs)
    other_set = set(others_xy)
    feats = []
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H):
            feats += [1, 0, 0, 0, 0, 1]  # wall, danger
            continue
        feats.append(1.0 if arena[nx, ny] == -1 else 0.0)
        feats.append(1.0 if arena[nx, ny] == 1 else 0.0)
        feats.append(1.0 if (nx, ny) in bomb_set else 0.0)
        feats.append(1.0 if (nx, ny) in other_set else 0.0)
        feats.append(1.0 if exp_map[nx, ny] >= 1 else 0.0)
        feats.append(1.0 if danger_now[nx, ny] else 0.0)
    # =24

    dist, first = _bfs_dist_dir(arena, bombs, (x, y))
    # crate-adjacent tiles as targets (stand next to crate to bomb it)
    crate_adj = set()
    crates = []
    for cx in range(1, W - 1):
        for cy in range(1, H - 1):
            if arena[cx, cy] == 1:
                crates.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and arena[nx, ny] == 0:
                        crate_adj.add((nx, ny))
    d_coin, dir_coin = _nearest(dist, first, [(int(a), int(b)) for (a, b) in coins], (x, y))
    d_crate, dir_crate = _nearest(dist, first, list(crate_adj) if crate_adj else crates, (x, y))
    d_opp, dir_opp = _nearest(dist, first, others_xy, (x, y))

    def norm_d(d):
        return min(float(d), 20.0) / 20.0 if d < 9999 else 1.0

    feats.append(norm_d(d_coin))
    feats += dir_to_onehot(dir_coin)
    feats.append(norm_d(d_crate))
    feats += dir_to_onehot(dir_crate)
    feats.append(norm_d(d_opp))
    feats += dir_to_onehot(dir_opp)
    # = 5*3=15 -> total 39

    # bomb context
    # min timer in LOS (wall-aware)
    min_t = 5.0
    for (xy, t) in bombs:
        xb, yb = int(xy[0]), int(xy[1])
        if xb == x:
            step_ok = True
            for yy in range(min(y, yb) + 1, max(y, yb)):
                if arena[x, yy] == -1:
                    step_ok = False
                    break
            if step_ok and abs(yb - y) <= 3:
                min_t = min(min_t, float(t))
        if yb == y:
            step_ok = True
            for xx in range(min(x, xb) + 1, max(x, xb)):
                if arena[xx, y] == -1:
                    step_ok = False
                    break
            if step_ok and abs(xb - x) <= 3:
                min_t = min(min_t, float(t))
    feats.append(min_t / 5.0)
    if safety_info is not None:
        feats.append(1.0 if safety_info.get('can_escape_if_bomb') else 0.0)
        feats.append(min(float(safety_info.get('crates_hit_if_bomb', 0)), 4.0) / 4.0)
        feats.append(min(float(safety_info.get('opps_hit_if_bomb', 0)), 2.0) / 2.0)
    else:
        feats += [0.0, 0.0, 0.0]
    feats.append(1.0 if bombs_left else 0.0)
    feats.append(min(step, 400.0) / 400.0)
    feats.append(min(float((arena == 1).sum()), 100.0) / 100.0)
    # =7 -> total 46
    arr = np.asarray(feats, dtype=np.float32)
    assert arr.shape[0] == FEATURE_DIM, f"{arr.shape}"
    return arr
