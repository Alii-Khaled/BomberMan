"""ARBITER vendored copy of agent_code/reaper/features.py (E48 rule: the
tournament ships agent_code/arbiter/ alone — no cross-agent imports).
98-dim engineered vector (numpy only), incl. the 8-symmetry dihedral
group (SYMS/AUG_PERMS/apply_aug/map_action) and the vectorized
whole-board crate-yield map _blast_crate_counts. Untouched apart from
this header; probe parity vs reaper in scripts/probe_arbiter.py.

Original docstring follows:

Layout (indices), directions in DELTAS order (UP DOWN LEFT RIGHT):
  0-23   per-dir context x4: wall, crate, bomb, other,
         explosion, danger-now                     [6 x 4]
  24     BFS dist to nearest coin (norm)           [scalar]
  25-28  BFS first-step dirs to coin (mask x4)
  29     BFS dist to nearest crate-adjacent tile   [scalar]
  30-33  BFS first-step dirs to crate-adjacent tile (mask x4)
  34     BFS dist to nearest opponent             [scalar]
  35-38  BFS first-step dirs to opponent (mask x4)
  39     BFS dist to nearest good bomb spot (crate-adjacent tile whose
         hypothetical blast hits >=2 crates)      [scalar]
  40-43  BFS first-step dirs to good bomb spot (mask x4)
  44     dist to safe tile (escape solver)        [scalar]
  45-48  safe first moves (mask x4)
  49     min bomb timer in line-of-sight (norm)
  50     bomb-here hits opponent (opps in own blast / 2)
         (E37: repurposed — the old can_escape_if_bomb was mask-invariant
         and therefore a constant-1 dead input whenever BOMB was legal)
  51     crates_hit_if_bomb (norm)
  52     opps_hit_if_bomb (norm)
  53     bombs_left
  54     step (norm)
  55     crates count (norm)
  56     own bomb active
  57     own bomb timer (norm)
  58     dist to own bomb (norm)
  59     own bomb blast hits self
  60     dist to nearest opponent (norm)
  61     opponents alive (norm)
  62     nearest opponent in dead-end
  63     trap available (adjacent spot hits an opponent that cannot escape)
  64     max crates_hit over adjacent bomb spots (norm)
  65     coins left (norm)
  66     score margin vs best opponent (norm, signed)
  67     hunt flag (crates+coins low or late game)
  68-71  per-dir TRAP mask: bomb at the adjacent tile in dir d traps an
         opponent (in blast + cannot escape)       [mask x4]
  72-75  per-dir opps_hit: opponents in blast of a bomb at the adjacent
         tile in dir d (/2)                        [x4]
  76-79  per-dir crates_hit: crates hit by a bomb at the adjacent tile
         in dir d (/4)                             [x4]
  80-83  per-dir escape margin: own escape distance after move-then-bomb
         in dir d (/8; 1.0 = infeasible)           [x4]
  84     own free-neighbour count (/4)
  85     own safely-reachable area within 6 steps (/60)
  86     own in dead-end (free_nb <= 1)
  87     BFS dist to nearest >=3-way junction (/10)
  88     nearest opponent has a bomb left
  89     opponents without a bomb left (/3)
  90     nearest opponent free-neighbour count (/4)
  91     min opponent escape distance under our best adjacent bomb (/8)
  92     nearest opponent cornered (free_nb <= 2)
  93     kill-progress potential PHI_kill in [0,1] (trap 1.0 > threat >
         pressure; the dense signal that makes rare kills learnable)
  94     coin closeness (1 - dist_to_coin)
  95     steps remaining (1 - step/400)
  96     live bombs on board (/4)
  97     score margin vs NEAREST opponent (norm, signed)
  98-101 E99 per-dir opponent seal risk: 1/(1+BFS-dist) of any opponent
         to the adjacent tile in dir d (walkability incl. bombs)
  102    min opponent BFS distance to our tile (/14; 1.0 = none)
  103    opponents that can reach our tile within 2 steps (/3)
  104    opponents within 3 steps (/3)
  105    opponent-reachable floor area within 5 steps of us (/60)
  106    best coin-race margin over reachable coins:
         max clip((opp_dist - our_dist)/6, -1, 1) (positive = we win)
  107    visible coins an opponent reaches strictly sooner (/3)
  108    post-plant escape-direction count n_esc (/4)
  109    best post-plant escape lane width (free neighbours, /4)
  110    current escape corridor depth (consecutive free tiles, /3)
  111    last-lethal time at our tile (/horizon; 0 = safe)
  112    any existing bomb (timer<4) whose blast overlaps our bomb-here blast
  113    nearest opponent distance to our best safe escape tile (/14)

Symmetry support: the 8 dihedral transforms of the 17x17 board map the
dir-indexed segments onto each other via a fixed permutation (built once);
scalars are invariant. ``apply_aug`` / ``transform_state`` / ``map_action``
let training augment transitions by recomputing/permuting features exactly.
"""
from collections import deque

import numpy as np

FEATURE_DIM = 114

DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]          # UP DOWN LEFT RIGHT
DELTA_TO_DIR = {tuple(d): i for i, d in enumerate(DELTAS)}

# --- dihedral group D4 on a 17x17 board (coords 0..16) ---------------------
# each symmetry = (A, b): new = A @ old + b, keeps the wall lattice intact
SYMS = [
    (np.array([[1, 0], [0, 1]]), (0, 0)),     # identity
    (np.array([[0, 1], [-1, 0]]), (0, 16)),   # rot90
    (np.array([[-1, 0], [0, -1]]), (16, 16)), # rot180
    (np.array([[0, -1], [1, 0]]), (16, 0)),   # rot270
    (np.array([[-1, 0], [0, 1]]), (16, 0)),   # flip horizontal
    (np.array([[1, 0], [0, -1]]), (0, 16)),   # flip vertical
    (np.array([[0, 1], [1, 0]]), (0, 0)),     # transpose
    (np.array([[0, -1], [-1, 0]]), (16, 16)), # anti-transpose
]
N_SYMS = len(SYMS)


def _delta_to_dir(d):
    return DELTA_TO_DIR[tuple(int(v) for v in d)]


def _build_aug_perms():
    """perm[s, i] = index in the ORIGINAL feature vector of transformed
    feature i under symmetry s (i.e. augmented[s][i] = original[perm[s, i]])."""
    perms = np.tile(np.arange(FEATURE_DIM, dtype=np.int64), (N_SYMS, 1))
    for s, (A, _b) in enumerate(SYMS):
        p = perms[s]
        # transformed direction nd = A @ d corresponds to original direction d
        new_dir = [_delta_to_dir(A @ np.array(d)) for d in DELTAS]
        for d in range(4):
            nd = new_dir[d]
            for k in range(6):                     # per-dir context
                p[nd * 6 + k] = d * 6 + k
        for base in (24, 29, 34, 39):              # (dist, onehot4) groups
            for d in range(4):
                p[base + 1 + new_dir[d]] = base + 1 + d
        for d in range(4):                         # first safe dir one-hot
            p[45 + new_dir[d]] = 45 + d
        for base in (68, 72, 76, 80):              # per-dir kill features
            for d in range(4):
                p[base + new_dir[d]] = base + d
        for base in (98,):                         # E99 per-dir seal risk
            for d in range(4):
                p[base + new_dir[d]] = base + d
    assert all(len(set(row)) == FEATURE_DIM for row in perms), "perm not bijective"
    return perms


AUG_PERMS = _build_aug_perms()


def apply_aug(feats, sym):
    """Permute a feature vector under symmetry s (scalars untouched)."""
    return np.asarray(feats, dtype=np.float32)[AUG_PERMS[sym]]


def map_action(action, sym):
    """Map an action under symmetry s. WAIT/BOMB are invariant."""
    if action in ('WAIT', 'BOMB'):
        return action
    A, _b = SYMS[sym]
    d = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}[action]
    nd = tuple(int(v) for v in (A @ np.array(d)))
    inv = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
    return inv[nd]


def transform_state(game_state, sym):
    """Return a shallow-copied game_state with every coordinate transformed."""
    A, b = SYMS[sym]
    gs = dict(game_state)
    arena = np.asarray(game_state['field'])
    W, H = arena.shape
    new_arena = np.zeros_like(arena)
    for x in range(W):
        for y in range(H):
            nx, ny = A @ np.array([x, y]) + np.array(b)
            new_arena[nx, ny] = arena[x, y]
    gs['field'] = new_arena

    def txy(xy):
        nx, ny = A @ np.array([int(xy[0]), int(xy[1])]) + np.array(b)
        return (int(nx), int(ny))

    gs['bombs'] = [(txy(xy), t) for (xy, t) in (game_state.get('bombs') or [])]
    gs['coins'] = [txy(c) for c in (game_state.get('coins') or [])]
    name, score, bombs_left, xy = game_state['self']
    gs['self'] = (name, score, bombs_left, txy(xy))
    gs['others'] = [(n, s, bl, txy(xy))
                    for (n, s, bl, xy) in (game_state.get('others') or [])]
    if game_state.get('explosion_map') is not None:
        em = np.asarray(game_state['explosion_map'])
        new_em = np.zeros_like(em)
        for x in range(W):
            for y in range(H):
                nx, ny = A @ np.array([x, y]) + np.array(b)
                new_em[nx, ny] = em[x, y]
        gs['explosion_map'] = new_em
    return gs


# --- BFS helpers ------------------------------------------------------------

def _bfs_dist4(arena, bombs, start):
    """BFS distance maps from each first-step direction.

    Returns dist4: (4, W, H) int32 — dist4[d, t] = distance from start to t
    via first step DELTAS[d] (9999 unreachable). Canonical BFS distances
    are order-independent, so the per-direction maps are equivariant under
    the board symmetries (no tie-break ambiguity).
    """
    W, H = arena.shape[0], arena.shape[1]
    bomb_set = set((int(xy[0]), int(xy[1])) for (xy, _t) in (bombs or []))
    sx, sy = int(start[0]), int(start[1])
    dist4 = np.full((4, W, H), 9999, dtype=np.int32)
    if not (0 <= sx < W and 0 <= sy < H):
        return dist4
    from collections import deque
    for d, (dx, dy) in enumerate(DELTAS):
        nx, ny = sx + dx, sy + dy
        if not (0 <= nx < W and 0 <= ny < H):
            continue
        if arena[nx, ny] != 0 or (nx, ny) in bomb_set:
            continue
        dist = dist4[d]
        dist[nx, ny] = 1
        q = deque([(nx, ny)])
        while q:
            cx, cy = q.popleft()
            for ex, ey in DELTAS:
                px, py = cx + ex, cy + ey
                if not (0 <= px < W and 0 <= py < H):
                    continue
                if dist[px, py] != 9999:
                    continue
                if arena[px, py] != 0:
                    continue
                if (px, py) in bomb_set:
                    continue
                dist[px, py] = dist[cx, cy] + 1
                q.append((px, py))
    return dist4


def _bfs_dist_multi(arena, bombs, starts):
    """Multi-source BFS distance map (min over starts). 9999 unreachable.

    Blocked = non-floor tiles and bomb tiles (same walkability contract as
    the opponent/escape BFS used everywhere else). Used by the E99
    interference/coin-race feature block.
    """
    from collections import deque
    W, H = arena.shape[0], arena.shape[1]
    dist = np.full((W, H), 9999, dtype=np.int32)
    bomb_set = set((int(xy[0]), int(xy[1])) for (xy, _t) in (bombs or []))
    q = deque()
    for (sx, sy) in (starts or []):
        sx, sy = int(sx), int(sy)
        if 0 <= sx < W and 0 <= sy < H and dist[sx, sy] != 0:
            dist[sx, sy] = 0
            q.append((sx, sy))
    while q:
        cx, cy = q.popleft()
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if dist[nx, ny] != 9999:
                continue
            if arena[nx, ny] != 0 or (nx, ny) in bomb_set:
                continue
            dist[nx, ny] = dist[cx, cy] + 1
            q.append((nx, ny))
    return dist


def _nearest_multi(dist4, targets, start):
    """(dist_norm, dir_mask4) toward nearest target (union over min-dist
    first-step directions, blocked targets approached via best neighbour).
    Unreachable -> (1.0, zeros). Equivariant under board symmetries."""
    W, H = dist4.shape[1], dist4.shape[2]
    best_d = 9999
    best_mask = np.zeros(4, dtype=np.float32)
    sx, sy = int(start[0]), int(start[1])
    for (tx, ty) in targets:
        tx, ty = int(tx), int(ty)
        if not (0 <= tx < W and 0 <= ty < H):
            continue
        if (tx, ty) == (sx, sy):
            # already standing on the target: distance 0, no move needed
            d, mask = 0, np.zeros(4, dtype=np.float32)
        else:
            ds = dist4[:, tx, ty].astype(np.int64)
            if (ds < 9999).any():
                pass  # free tile reachable via at least one first step
            else:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = tx + dx, ty + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        ds = np.minimum(ds, dist4[:, nx, ny] + 1)
            d = int(ds.min())
            if d >= 9999:
                continue
            mask = (ds == d).astype(np.float32)
        if d < best_d:
            best_d, best_mask = d, mask
        elif d == best_d:
            best_mask = np.maximum(best_mask, mask)
    return min(float(best_d), 20.0) / 20.0 if best_d < 9999 else 1.0, best_mask


# --- main feature builder ----------------------------------------------------

def _blast_tiles(arena, x, y, power=3):
    tiles = [(x, y)]
    W, H = arena.shape[0], arena.shape[1]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < W and 0 <= ny < H):
                break
            if arena[nx, ny] == -1:
                break
            tiles.append((nx, ny))
    return tiles


def _blast_crate_counts(arena, power=3):
    """Blast value of EVERY tile at once (numpy, no Python loops).

    Returns (crate_adj_mask, blast_crates) where blast_crates[x, y] = number
    of crates a bomb at (x, y) would hit (origin + 4 arms, blast stops at
    stone walls only — same convention as _blast_tiles/true_blast).
    crate_adj_mask = free tiles adjacent to at least one crate.
    Borders are stone walls, so arm scans never run off the board.
    """
    ar = np.asarray(arena)
    W, H = ar.shape[0], ar.shape[1]
    wall = (ar == -1).astype(np.int32)
    crate = (ar == 1).astype(np.int32)
    floor = (ar == 0)

    # free tiles touching a crate (padded shifts, no wrap-around)
    adj = np.zeros((W, H), dtype=bool)
    adj[1:, :] |= crate[:-1, :] != 0
    adj[:-1, :] |= crate[1:, :] != 0
    adj[:, 1:] |= crate[:, :-1] != 0
    adj[:, :-1] |= crate[:, 1:] != 0
    crate_adj_mask = adj & floor

    kh = np.arange(power + 1).reshape(1, -1, 1)      # arm lengths, (1,P+1,1)
    # --- horizontal: one leading zero column keeps edge indices in range
    wc = np.zeros((W, H + 1), dtype=np.int32)
    wc[:, 1:] = np.cumsum(wall, axis=1)
    cc = np.zeros((W, H + 1), dtype=np.int32)
    cc[:, 1:] = np.cumsum(crate, axis=1)
    jj = np.arange(H).reshape(1, 1, -1)               # (1,1,H)
    Wi = np.arange(W).reshape(-1, 1, 1)               # (W,1,1) row index
    end_r = np.minimum(jj + 1 + kh, H)                # (1,P+1,H)
    end_l = np.maximum(jj - kh, 0)
    clear_r = (wc[Wi, end_r] - wc[Wi, jj + 1]) == 0    # (W,P+1,H)
    reach_r = clear_r.sum(axis=1) - 1                 # clear arm length 0..P
    clear_l = (wc[Wi, jj] - wc[Wi, end_l]) == 0
    reach_l = clear_l.sum(axis=1) - 1
    hits_r = cc[Wi, end_r] - cc[Wi, jj + 1]
    hits_l = cc[Wi, jj] - cc[Wi, end_l]
    # --- vertical: same accumulation along rows
    wc2 = np.zeros((W + 1, H), dtype=np.int32)
    wc2[1:, :] = np.cumsum(wall, axis=0)
    cc2 = np.zeros((W + 1, H), dtype=np.int32)
    cc2[1:, :] = np.cumsum(crate, axis=0)
    ii = np.arange(W).reshape(-1, 1)                  # (W,1)
    kv = np.arange(power + 1).reshape(1, -1)          # (1,P+1)
    end_d = np.minimum(ii + 1 + kv, W)                # (W,P+1)
    end_u = np.maximum(ii - kv, 0)
    clear_d = (wc2[end_d, :] - wc2[ii + 1, :]) == 0   # (W,P+1,H)
    reach_d = clear_d.sum(axis=1) - 1
    clear_u = (wc2[ii, :] - wc2[end_u, :]) == 0
    reach_u = clear_u.sum(axis=1) - 1
    hits_d = cc2[end_d, :] - cc2[ii + 1, :]
    hits_u = cc2[ii, :] - cc2[end_u, :]
    # crates hit = origin + arm contents within the CLEAR reach
    Widx = np.arange(W)[:, None]
    Hidx = np.arange(H)[None, :]
    blast = crate.astype(np.int32)
    blast = blast + hits_r[Widx, np.clip(reach_r, 0, power), Hidx]
    blast = blast + hits_l[Widx, np.clip(reach_l, 0, power), Hidx]
    blast = blast + hits_d[Widx, np.clip(reach_d, 0, power), Hidx]
    blast = blast + hits_u[Widx, np.clip(reach_u, 0, power), Hidx]
    return crate_adj_mask, blast


def _adj_kill_info(arena, bombs, others_xy, x, y, blast_counts, full=True):
    """Directional kill table for move-then-bomb (DELTAS order: U D L R).

    For each of the 4 adjacent tiles returns:
      trap[d]   1.0 if a bomb there traps an opponent (in blast + cannot
                escape) — the directionalized f[63]
      opps[d]   opponents in blast there / 2
      crates[d] crates hit there / 4 (from the vectorized table)
      esc[d]    own escape distance after move-then-bomb there / 8;
                1.0 = infeasible / not a floor spot
    plus min_opp_esc = minimum opponent escape distance over all opponents
    in any adjacent blast (inf if none).

    The expensive part (own escape BFS per spot + opponent escape BFS per
    threatened opponent) runs only when full=True; the caller gates it on
    `bombs_left and opponents present`. All conditions are rotation-
    invariant, so equivariance is preserved.
    """
    from .safety import (danger_no_explosion, opp_can_escape, escape_bfs,
                         with_hypothetical_bomb)
    trap = [0.0, 0.0, 0.0, 0.0]
    opps = [0.0, 0.0, 0.0, 0.0]
    crates = [0.0, 0.0, 0.0, 0.0]
    esc = [1.0, 1.0, 1.0, 1.0]
    min_opp_esc = float('inf')
    ar = np.asarray(arena)
    W, H = ar.shape[0], ar.shape[1]
    x, y = int(x), int(y)
    bomb_set = set((int(xy[0]), int(xy[1])) for (xy, _t) in (bombs or []))
    other_set = set((int(a), int(b)) for (a, b) in (others_xy or []))
    spots = []
    for d, (dx, dy) in enumerate(DELTAS):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H):
            continue
        if ar[nx, ny] != 0 or (nx, ny) in bomb_set or (nx, ny) in other_set:
            continue
        spots.append((d, nx, ny))
        try:
            crates[d] = min(float(blast_counts[nx, ny]), 4.0) / 4.0
        except (IndexError, TypeError, ValueError):
            pass
    if not spots or not others_xy:
        return trap, opps, crates, esc, min_opp_esc
    opp_masks = [_blast_hits_opp_mask(ar, int(ox), int(oy))
                 for (ox, oy) in others_xy]
    for d, nx, ny in spots:
        hits = [i for i, m in enumerate(opp_masks) if m[nx, ny]]
        opps[d] = min(float(len(hits)), 2.0) / 2.0
    if not full:
        return trap, opps, crates, esc, min_opp_esc
    shared = danger_no_explosion(ar, bombs)
    base_bombs = list(bombs or [])
    for d, nx, ny in spots:
        # own escape after moving to (nx,ny) and bombing there. WAIT is
        # excluded on purpose: staying on your own bomb tile dies at t=4.
        dh = with_hypothetical_bomb(shared, ar, nx, ny)
        bh = base_bombs + [((nx, ny), 4)]
        try:
            sh, dist_h = escape_bfs((nx, ny), ar, bh, others_xy, dh)
        except Exception:
            continue
        try:
            dh_f = float(dist_h)
        except (TypeError, ValueError):
            dh_f = float('inf')
        can_self = any(sh.get(dd, False)
                       for dd in ((0, -1), (0, 1), (-1, 0), (1, 0)))
        if can_self and dh_f < 9999:
            esc[d] = min(dh_f, 8.0) / 8.0
        hits = [i for i, m in enumerate(opp_masks) if m[nx, ny]]
        for i in hits:
            ox, oy = others_xy[i]
            try:
                can, dist_o = opp_can_escape(ar, bombs, (int(ox), int(oy)),
                                             (nx, ny), others_xy,
                                             danger=shared)
                do_f = float(dist_o)
            except Exception:
                continue
            if do_f < min_opp_esc:
                min_opp_esc = do_f
            if not can:
                trap[d] = 1.0
    return trap, opps, crates, esc, min_opp_esc


def _blast_hits_opp_mask(arena, ox, oy, power=3):
    """Bool board: tiles whose hypothetical blast covers opponent (ox, oy).

    Vectorized complement to _blast_tiles: an origin (x, y) hits the
    opponent iff aligned within blast range with no stone wall between.
    (For an origin equal to the opponent tile the range test is vacuous
    and True, matching _blast_tiles which always includes the origin.)
    """
    ar = np.asarray(arena)
    W, H = ar.shape[0], ar.shape[1]
    wall = (ar == -1).astype(np.int32)
    m = np.zeros((W, H), dtype=bool)
    if not (0 <= ox < W and 0 <= oy < H):
        return m
    # same row ox: origin col y hits iff |y-oy|<=P and the cols between the
    # two tiles hold no wall (origin/opponent are non-wall by construction).
    wc = np.zeros((W, H + 1), dtype=np.int32)
    wc[:, 1:] = np.cumsum(wall, axis=1)
    y0 = np.arange(H)
    hi = np.maximum(y0, oy)
    lo = np.minimum(y0, oy)
    m[ox, :] |= (np.abs(y0 - oy) <= power) & \
        ((wc[ox, hi + 1] - wc[ox, lo + 1]) == 0)
    # same column oy: symmetric along rows.
    wc2 = np.zeros((W + 1, H), dtype=np.int32)
    wc2[1:, :] = np.cumsum(wall, axis=0)
    x0 = np.arange(W)
    hi2 = np.maximum(x0, ox)
    lo2 = np.minimum(x0, ox)
    m[:, oy] |= (np.abs(x0 - ox) <= power) & \
        ((wc2[hi2 + 1, oy] - wc2[lo2 + 1, oy]) == 0)
    return m


def state_to_features(game_state, safety_info=None, own_bomb=None):
    if game_state is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    from .safety import true_blast, opp_can_escape
    arena = np.asarray(game_state['field'])
    W, H = arena.shape[0], arena.shape[1]
    _, score, bombs_left, (x, y) = game_state['self']
    x, y = int(x), int(y)
    bombs = game_state.get('bombs') or []
    coins = [(int(a), int(b)) for (a, b) in (game_state.get('coins') or [])]
    others = [(n, s, b, (int(xy[0]), int(xy[1])))
              for (n, s, b, xy) in (game_state.get('others') or [])]
    others_xy = [xy for (_, _, _, xy) in others]
    exp_map = np.asarray(game_state.get('explosion_map', np.zeros((W, H))))
    step = float(game_state.get('step', 0))
    bomb_set = set((int(xy[0]), int(xy[1])) for (xy, _t) in bombs)
    other_set = set(others_xy)

    danger_now = None
    if safety_info is not None and 'danger' in safety_info:
        danger_now = safety_info['danger'][0]
    else:
        danger_now = exp_map >= 1

    f = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 0-23 per-dir context
    for d, (dx, dy) in enumerate(DELTAS):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H):
            f[d * 6 + 0] = 1.0
            f[d * 6 + 5] = 1.0
            continue
        f[d * 6 + 0] = 1.0 if arena[nx, ny] == -1 else 0.0
        f[d * 6 + 1] = 1.0 if arena[nx, ny] == 1 else 0.0
        f[d * 6 + 2] = 1.0 if (nx, ny) in bomb_set else 0.0
        f[d * 6 + 3] = 1.0 if (nx, ny) in other_set else 0.0
        f[d * 6 + 4] = 1.0 if exp_map[nx, ny] >= 1 else 0.0
        f[d * 6 + 5] = 1.0 if danger_now[nx, ny] else 0.0

    dist4 = _bfs_dist4(arena, bombs, (x, y))

    # crate-adjacent free tiles + per-tile blast values (vectorized)
    crate_adj_mask, blast_counts = _blast_crate_counts(arena)
    crate_xy = np.argwhere(np.asarray(arena) == 1)
    crates = sorted(map(tuple, crate_xy.tolist()))
    crate_adj = sorted(map(tuple, np.argwhere(crate_adj_mask).tolist()))

    # good bomb spots: crate-adjacent tiles whose blast hits >=2 crates
    good_spots = sorted(map(tuple, np.argwhere(
        crate_adj_mask & (blast_counts >= 2)).tolist()))
    if not good_spots and len(crates) <= 8:
        good_spots = list(crate_adj)

    f[24], mask = _nearest_multi(dist4, coins, (x, y))
    f[25:29] = mask
    f[29], mask = _nearest_multi(dist4, crate_adj if crate_adj else crates, (x, y))
    f[30:34] = mask
    f[34], mask = _nearest_multi(dist4, others_xy, (x, y))
    f[35:39] = mask
    f[39], mask = _nearest_multi(dist4, good_spots if good_spots else crate_adj or crates, (x, y))
    f[40:44] = mask

    # 44-48 safe move info from the escape solver (mask of safe first moves)
    if safety_info is not None:
        f[44] = min(float(safety_info.get('dist_to_safe', float('inf'))), 20.0) / 20.0 \
            if float(safety_info.get('dist_to_safe', float('inf'))) < 9999 else 1.0
        safe_map = safety_info.get('safe', {})
        dir_name = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        for d in range(4):
            if safe_map.get(dir_name[d], False):
                f[45 + d] = 1.0

    # 49-55 bomb/safety scalars
    min_t = 5.0
    for (xy, t) in bombs:
        xb, yb = int(xy[0]), int(xy[1])
        t = min(max(int(t), 0), 4)
        if xb == x and abs(yb - y) <= 3:
            clear = all(arena[x, yy] != -1 for yy in range(min(y, yb) + 1, max(y, yb)))
            if clear:
                min_t = min(min_t, float(t))
        if yb == y and abs(xb - x) <= 3:
            clear = all(arena[xx, y] != -1 for xx in range(min(x, xb) + 1, max(x, xb)))
            if clear:
                min_t = min(min_t, float(t))
    f[49] = min_t / 5.0
    if safety_info is not None:
        # f[50]: bomb-HERE escape margin (1 = immediate escape, 0 =
        # infeasible). Replaces can_escape_if_bomb, which is mask-invariant
        # and therefore a dead constant-1 input whenever BOMB is legal.
        try:
            _dh = float(safety_info.get('dist_hyp', float('inf')))
        except (TypeError, ValueError):
            _dh = float('inf')
        f[50] = 1.0 - min(_dh, 8.0) / 8.0 if _dh < 9999 else 0.0
        f[51] = min(float(safety_info.get('crates_hit_if_bomb', 0)), 4.0) / 4.0
        f[52] = min(float(safety_info.get('opps_hit_if_bomb', 0)), 2.0) / 2.0
    f[53] = 1.0 if bombs_left else 0.0
    f[54] = min(step, 400.0) / 400.0
    f[55] = min(float(len(crates)), 100.0) / 100.0

    # 56-59 own bomb
    if own_bomb is not None and not bombs_left:
        obx, oby = int(own_bomb[0]), int(own_bomb[1])
        ob_timer = 4.0
        for (xy, t) in bombs:
            if int(xy[0]) == obx and int(xy[1]) == oby:
                ob_timer = float(t)
                break
        f[56] = 1.0
        f[57] = min(max(ob_timer, 0.0), 4.0) / 4.0
        f[58] = min(float(abs(obx - x) + abs(oby - y)), 20.0) / 20.0
        f[59] = 1.0 if (x, y) in set(true_blast(arena, obx, oby)) else 0.0

    # 60-64 opponent model (+ 68-97 kill-centric block, E37/P2)
    # own mobility (floor neighbours that are not bombs / opponents)
    own_free_nb = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and arena[nx, ny] == 0 \
                and (nx, ny) not in bomb_set and (nx, ny) not in other_set:
            own_free_nb += 1
    f[84] = min(float(own_free_nb), 4.0) / 4.0
    f[86] = 1.0 if own_free_nb <= 1 else 0.0
    # reachable area within 6 steps (pure mobility, ignores danger/opps)
    try:
        _seen, _q = {(x, y)}, deque([(x, y, 0)])
        while _q:
            _cx, _cy, _cd = _q.popleft()
            if _cd >= 6:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                _nx, _ny = _cx + dx, _cy + dy
                if 0 <= _nx < W and 0 <= _ny < H and arena[_nx, _ny] == 0 \
                        and (_nx, _ny) not in bomb_set \
                        and (_nx, _ny) not in _seen:
                    _seen.add((_nx, _ny))
                    _q.append((_nx, _ny, _cd + 1))
        f[85] = min(float(len(_seen)), 60.0) / 60.0
    except Exception:
        pass
    # dist to nearest >=3-way junction (reuses the BFS maps)
    try:
        _jn = np.zeros((W, H), dtype=bool)
        _fl = (np.asarray(arena) == 0)
        _nb = np.zeros((W, H), dtype=np.int32)
        _nb[1:, :] += _fl[:-1, :]
        _nb[:-1, :] += _fl[1:, :]
        _nb[:, 1:] += _fl[:, :-1]
        _nb[:, :-1] += _fl[:, 1:]
        _jn = _fl & (_nb >= 3)
        _jl = sorted(map(tuple, np.argwhere(_jn).tolist()))
        _jd, _ = _nearest_multi(dist4, _jl, (x, y))
        f[87] = _jd
    except Exception:
        pass
    if others_xy:
        od = min(abs(ox - x) + abs(oy - y) for (ox, oy) in others_xy)
        f[60] = min(float(od), 20.0) / 20.0
        to = min(others_xy, key=lambda o: abs(o[0] - x) + abs(o[1] - y))
        free_nb = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = to[0] + dx, to[1] + dy
            if 0 <= nx < W and 0 <= ny < H and arena[nx, ny] == 0 and (nx, ny) not in bomb_set:
                free_nb += 1
        f[62] = 1.0 if free_nb == 1 else 0.0
        try:
            _oi = others_xy.index(to)
            _obl = bool(others[_oi][2])
            _nscore = float(others[_oi][1])
        except (ValueError, IndexError, TypeError):
            _obl, _nscore = False, 0.0
        f[88] = 1.0 if _obl else 0.0
        f[90] = min(float(free_nb), 4.0) / 4.0
        f[92] = 1.0 if free_nb <= 2 else 0.0
        f[97] = float(np.clip((score - _nscore) / 5.0, -1.0, 1.0))
        # directional kill table (E37/P2 core). Expensive escape part runs
        # only when we can actually bomb.
        _trap, _op, _cr, _es, _moe = _adj_kill_info(
            arena, bombs, others_xy, x, y, blast_counts,
            full=bool(bombs_left))
        f[68:72] = _trap
        f[72:76] = _op
        f[76:80] = _cr
        f[80:84] = _es
        f[63] = 1.0 if max(_trap) > 0 else 0.0
        f[91] = min(float(_moe), 8.0) / 8.0 if _moe < 9999 else 1.0
        # kill-progress potential: trap (1.0) > threat (<=0.5) > pressure.
        # Dense everywhere an opponent exists; the Phase-3 shaping signal.
        if max(_trap) > 0:
            f[93] = 1.0
        else:
            _threat = 0.5 * max(_op) if _op else 0.0
            _press = 0.25 * max(0.0, 1.0 - min(float(od), 10.0) / 10.0)
            f[93] = max(_threat, _press)
    f[61] = min(float(len(others)), 3.0) / 3.0
    try:
        f[89] = min(float(sum(1 for (_, _, _bl, _) in others if not _bl)),
                    3.0) / 3.0
    except Exception:
        pass
    # max crates_hit over adjacent bomb spots (from the vectorized table)
    mx = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H) or arena[nx, ny] != 0:
            continue
        if (nx, ny) in bomb_set or (nx, ny) in other_set:
            continue
        mx = max(mx, int(blast_counts[nx, ny]))
    f[64] = min(float(mx), 4.0) / 4.0

    # 65-67 endgame (+ 94-96 potentials)
    f[65] = min(float(len(coins)), 20.0) / 20.0
    opp_scores = [s for (_, s, _, _) in others] or [0]
    f[66] = float(np.clip((score - max(opp_scores)) / 5.0, -1.0, 1.0))
    f[67] = 1.0 if (len(crates) + len(coins) <= 4 or step > 250) else 0.0
    f[94] = 1.0 - float(f[24])
    f[95] = 1.0 - min(step, 400.0) / 400.0
    f[96] = min(float(len(bombs)), 4.0) / 4.0

    # --- E99 feature-v2 block (98-113): interference + coin race + post
    # plant quality. Opponent reachability is computed from the state alone
    # (available at search leaves too); post-plant/danger terms need the
    # safety_info and degrade to 0 when absent, like 44-52.
    try:
        opp_map = (_bfs_dist_multi(arena, bombs, others_xy)
                   if others_xy else None)
    except Exception:
        opp_map = None
    our_dist = dist4.min(axis=0)
    if opp_map is not None:
        for d, (dx, dy) in enumerate(DELTAS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                od = int(opp_map[nx, ny])
                if od < 9999:
                    f[98 + d] = 1.0 / (1.0 + od)
        odt = int(opp_map[x, y])
        f[102] = min(odt, 14.0) / 14.0 if odt < 9999 else 1.0
        f[103] = min(int((opp_map <= 2).sum()), 3) / 3.0
        f[104] = min(int((opp_map <= 3).sum()), 3) / 3.0
        try:
            _free = (np.asarray(arena) == 0)
            f[105] = min(int(((opp_map <= 5) & _free).sum()), 60) / 60.0
        except Exception:
            pass
        # coin race: ours vs the nearest opponent on the same coin
        if coins:
            margins = []
            contested = 0
            for (cx, cy) in coins:
                dm = int(our_dist[cx, cy]) if our_dist[cx, cy] < 9999 else 9999
                do = int(opp_map[cx, cy]) if opp_map[cx, cy] < 9999 else 9999
                if dm < 9999 and do < 9999:
                    margins.append(float(np.clip((do - dm) / 6.0, -1.0, 1.0)))
                if dm < 9999 and do < dm:
                    contested += 1
            if margins:
                f[106] = max(margins)
            f[107] = min(contested, 3) / 3.0
    if safety_info is not None:
        danger = safety_info.get('danger')
        if danger is not None:
            dng = np.asarray(danger)
            if dng.ndim == 3:
                col = dng[:, x, y]
                lv = np.where(col)[0]
                if len(lv):
                    f[111] = (int(lv[-1]) + 1) / float(dng.shape[0])
        try:
            f[108] = min(float(safety_info.get('n_esc_hyp', 0)), 4.0) / 4.0
        except (TypeError, ValueError):
            pass
        sh = safety_info.get('safe_hyp') or {}
        safe_map = safety_info.get('safe') or {}
        best_w, depth = 0, 0
        for (ddx, ddy), okd in sh.items():
            if not okd or (ddx, ddy) == (0, 0):
                continue
            nx, ny = x + ddx, y + ddy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            cnt = 0
            for ex, ey in DELTAS:
                tx, ty = nx + ex, ny + ey
                if 0 <= tx < W and 0 <= ty < H and arena[tx, ty] == 0 \
                        and (tx, ty) not in bomb_set:
                    cnt += 1
            best_w = max(best_w, cnt)
        for _a, _d in (('UP', (0, -1)), ('DOWN', (0, 1)),
                       ('LEFT', (-1, 0)), ('RIGHT', (1, 0))):
            if not safe_map.get(_a, False):
                continue
            ddx, ddy = _d
            cx0, cy0, k = x + ddx, y + ddy, 0
            while 0 <= cx0 < W and 0 <= cy0 < H and k < 3 \
                    and arena[cx0, cy0] == 0 and (cx0, cy0) not in bomb_set:
                k += 1
                cx0 += ddx
                cy0 += ddy
            depth = max(depth, k)
        f[109] = min(best_w, 4) / 4.0
        f[110] = min(depth, 3) / 3.0
        blast_hyp = safety_info.get('blast_if_bomb')
        if blast_hyp:
            try:
                for (bxy, bt) in bombs:
                    if int(bt) < 4:
                        if set(true_blast(arena, int(bxy[0]), int(bxy[1]))
                               ).intersection(blast_hyp):
                            f[112] = 1.0
                            break
            except Exception:
                pass
        if opp_map is not None:
            try:
                best = 9999
                for _a, _d in (('UP', (0, -1)), ('DOWN', (0, 1)),
                               ('LEFT', (-1, 0)), ('RIGHT', (1, 0))):
                    if safe_map.get(_a, False):
                        od = int(opp_map[x + _d[0], y + _d[1]])
                        best = min(best, od)
                if best < 9999:
                    f[113] = min(best, 14.0) / 14.0
            except Exception:
                pass
    return f
