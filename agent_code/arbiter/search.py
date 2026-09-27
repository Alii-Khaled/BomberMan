"""Arbiter rollout search with optional learned leaf-value scoring.

Plan = committed own-action prefix (BFS path to a bomb tile + BOMB, or a
single move) + cheap continuation to detonation settle. Opponents use an
avoid-lethal-if-possible + random policy (seeded, deterministic per
round/step). Kills count at full value only when certified against
optimal flight (opp_can_escape); rollout-luck kills are ignored.
Score(plan) = exact margin delta (crates*w + coins + 5*certified kills
- 8*own death + coin-reveal expectation) + V_BLEND * V(s_end).

Search shares ARBITER_TIME_BUDGET with act and caps plans and leaf calls.
If no plan qualifies, the callback uses its policy fallback. Opponent
responses and hidden-coin rewards are approximations.
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


H = _env_int('ARBITER_SEARCH_H', 6, 2, 12)
# Rollout seeds per plan: the exact-payoff part of a plan score carries
# opponent-policy RNG noise, and averaging over seeds shrinks it. V runs
# on the primary rollout's leaf only (leaf features cost 1.18 ms, so V
# over all seeds would break the step budget). Default 1.
SEEDS = _env_int('ARBITER_SEEDS', 1, 1, 8)
# Move-plan seeds: average the exact-payoff part over MOVE_SEEDS
# opponent-policy rollouts for MOVE plans only, single seed for bomb
# plans. Seed-averaging steadies move arbitration but dilutes the
# single-seed W_DEATH veto on bombs (veto calibration assumes one
# seed), so bombs keep the single-seed path (j=0 seed formula) and
# moves buy the coin-side variance reduction. Default 1.
MOVE_SEEDS = _env_int('ARBITER_MOVE_SEEDS', 1, 1, 8)
# Common random numbers: opponent rollouts key on (rollout seed, tick,
# opponent) instead of (rollout seed, plan), so every plan in a step is
# compared against the same opponent draws (paired comparison, lower
# variance on plan differences). Default ON; ARBITER_CRN=0 restores
# unpaired plan-indexed seeding.
CRN = os.environ.get('ARBITER_CRN', '1') == '1'
K = _env_int('ARBITER_SEARCH_K', 8, 0, 32)
RADIUS = _env_int('ARBITER_SEARCH_R', 4, 1, 8)
PLAN_CAP = _env_int('ARBITER_SEARCH_PLANS', 48, 1, 256)
V_BLEND = _env_float('ARBITER_V_BLEND', 1.0)
# V_OFF mirrors callbacks' ARBITER_V_OFF (V0 ablation): when set, the
# search skips leaf evaluation exactly as if V_BLEND were 0.
V_OFF = os.environ.get('ARBITER_V_OFF', '0') == '1'
W_CRATE = _env_float('ARBITER_W_CRATE', 0.1)
W_DEATH = _env_float('ARBITER_W_DEATH', 8.0)
# Bomb-vs-move arbitration: search owns bomb decisions (placement is
# what pi cannot do); pi owns moves (V RMSE +-2.4 dwarfs 1-step value
# gaps, so V-ranked moves are noise against pi's sharp policy). A bomb
# plan executes only if it beats the best move plan by more than
# BOMB_MARGIN; else search returns None and pi decides the move.
# The score margin reads ARBITER_BOMB_SCORE_MARGIN; the legacy
# ARBITER_BOMB_MARGIN name is still honored (README-documented
# meaning) so existing scripts keep working. safety.py's escape-dir
# count reads its own ARBITER_BOMB_ESC_MARGIN knob.
# Default 0.6: the strict escape solver makes certified kills and bomb
# admission stricter, and a 0.2 margin over-admits bombs against it.
BOMB_MARGIN = _env_float(
    'ARBITER_BOMB_SCORE_MARGIN',
    _env_float('ARBITER_BOMB_MARGIN', 0.6))
# Coin proximity in bomb-tile ranking: blast yield alone strands the
# agent far from the coins it reveals. Prefer high-yield tiles near
# collectable coins.
W_COIN_TILE = _env_float('ARBITER_W_COIN_TILE', 0.5)
# Yield-scaled bomb margin: junk bombs (plants with opps_hit=0 almost
# never kill) should clear a higher bar than valuable ones.
# margin_eff = BOMB_MARGIN + YIELD_GAMMA * max(0, 2 - tile_yield),
# tile_yield = crates_in_blast + 2 * opps_in_blast at the bomb tile.
# Soft: the margin scales the bar, nothing is forbidden outright.
# Default 0 (off).
YIELD_GAMMA = _env_float('ARBITER_YIELD_GAMMA', 0.0)
# Opponent blast bonus in tile ranking: search leans toward crate bombs
# and under-kills relative to pi/warden, which bomb opportunistically
# on opps_hit>0. Bonus per opponent in blast; certification at scoring
# (opp_can_escape) keeps it honest. Default 2.0.
W_OPP_TILE = _env_float('ARBITER_W_OPP_TILE', 2.0)
# Escape strictness: 0 = any escape first-step suffices;
# N > 0 = require escape_bfs dist_to_safe <= N. Default 3.0.
ESC_DIST = _env_float('ARBITER_ESC_DIST', 3.0)
# Minimum post-plant escape directions required by the search's bomb
# gate. 1 = the original any() rule. The mask's copy of this idea
# cannot govern search-selected bombs (the mask marks bombs the search
# then picks anyway); this knob targets the gate that admits search
# bombs.
PLANT_ESC = _env_int('ARBITER_PLANT_ESC', 1, 1, 4)
# Opponent-aware plant certification: an opponent body block can cut a
# planted bomb's certified escape. 0 = off. N >= 1 requires at least N
# post-plant escape directions that avoid the opponents' K-step shadow
# (tiles within ARBITER_PLANT_OPP_K BFS steps of any opponent). It
# vetoes bombs only and never re-ranks.
PLANT_OPP = _env_int('ARBITER_PLANT_OPP', 0, 0, 4)
PLANT_OPP_K = _env_int('ARBITER_PLANT_OPP_K', 1, 1, 3)
# Coin-race move plan: a committed BFS path to the nearest reachable
# visible coin, scored by the exact rollout like any plan. It competes
# as a MOVE plan (bomb_at None), and a valuable coin run raises the
# bomb bar via best_move. Default 0 (off).
COINRUN = os.environ.get('ARBITER_COINRUN', '0') == '1'
# Chain-bomb priority: own tile + 4 neighbours always enter bomb
# candidacy (they bypass ranking and the K cap). The re-bomb step is
# the cheapest volume lever: the tile we stand on needs no travel.
# The proven-escape gate below still applies to each. Default 0 (off).
CHAIN = os.environ.get('ARBITER_CHAIN', '0') == '1'
# Guarded chain: own tile + 4 neighbours enter candidacy only under
# warden's want_bomb guard (opps_hit > 0, or crates_hit >= 2 with
# hyp_dist <= 3, or crates_hit == 1 with hyp_dist <= 2). An unguarded
# chain fires junk (the 6-step lockout displaces good bombs), and junk
# score filtering does not help: the guard belongs on admission.
# Appended after ranked candidates so the ranked plans keep their
# rollout-seed indices and the K cap covers ranked plans only. The
# proven-escape gate below still applies. Default 0 (off).
CHAIN_GUARD = os.environ.get('ARBITER_CHAIN_GUARD', '0') == '1'
# Opportunistic-trap credit: certified kills (no escape at any timer)
# pay 1.0; opponents whose escape needs >= TRAP_HARD steps while a
# bomb detonates within 2 pay TRAP_P (expected value, they stay alive
# in sim). Default off (0 = certified only). Per-bomb kill rates
# already match warden; the gap is bomb volume, and strict
# certification vetoes contested bombs.
TRAP_HARD = _env_float('ARBITER_TRAP_HARD', 0.0)
TRAP_P = _env_float('ARBITER_TRAP_P', 0.5)
# Hunt intent: pursuit bomb plans. When the warden hunt trigger holds
# (opponents present and [loot <= 6 | step > 200 | opp within Manhattan
# 3]), each opponent gets one bomb plan at the best free tile whose
# hypothetical blast covers the opponent (BFS path + BOMB prefix),
# admitted through the same _try_bomb_plan gate as every other tile
# (one gate, no special cases). Appended after the ranked walk so
# ranked plans keep their rollout-seed indices; the bomb cap extends
# by len(hunt plans). Default 0 (off).
HUNT = os.environ.get('ARBITER_HUNT', '0') == '1'
HUNT_PLANS = _env_int('ARBITER_HUNT_PLANS', 2, 1, 4)
# Pursuit distance cap: 4 = the ranked walk's radius, the same
# path-staleness class (certifies the bomb tile, not the path);
# receding-horizon re-scoring covers the rest.
HUNT_DIST = _env_int('ARBITER_HUNT_DIST', 4, 1, 11)
# Cert-owner discipline: the certification loop runs over all sim
# bombs, and the rollout opp model drops bombs randomly (~0.5
# suicides/round/opp), so an opp self-trap would get certified and
# +5-credited to us although the engine pays nobody. CERT_OWN=1
# restricts the cert trigger to our own bombs (owner 0; pre-existing
# bombs are owner -1 = unknown and stay certifiable) while keeping
# the escape check global. Default 0 (off).
CERT_OWN = os.environ.get('ARBITER_CERT_OWN', '0') == '1'
# Rollout opponent model: 'random' (default: avoid-lethal-myopic +
# uniform incl. BOMB) or 'wardenlite' (avoid-lethal, then
# Manhattan-coin-pursuit step, bombs only under a cheap warden guard:
# opps_hit > 0 | crates_hit >= 2, and only with a free neighbour
# outside the new blast). The overlay prices arbitration against
# strong unseen agents; O(1) per opp-tick (no extra BFS).
OPPMODEL = os.environ.get('ARBITER_OPPMODEL', 'random').strip().lower()


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
    """Warden-rule admission predicate for chain tiles.

    Returns (admit, tile_yield). Pure function: a chain tile is
    admitted iff it has a proven escape (same recipe
    as the plan gate) AND warden's want_bomb payoff guard holds
    (opps_hit > 0, or crates_hit >= 2 with hyp_dist <= 3, or
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
    """Shared bomb-tile admission: BFS path + proven-escape gate.

    Single gate authority for ranked and chain tiles alike (one gate,
    no special cases). Returns True on admission.
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
            # escapes must also avoid the opponents' k-step shadow.
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
    """Warden hunt predicate (pure): opponents present
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
        # coin-race plan: committed BFS path to the nearest reachable
        # visible coin (prefix <= 12; replan next step).
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
            # stage 2: opponent blast bonus on the top slice (kill-seeking;
            # true_blast per tile is cheap, escape BFS below is not)
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
            # chain set: own tile + 4 neighbours enter candidacy
            # (gated by CHAIN; the escape check below applies). Appended
            # after ranked candidates so the ranked plans keep their
            # rollout-seed indices: the chain is a pure max-addition,
            # never an RNG perturbation. The K cap covers ranked plans;
            # chain tiles get extras.
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
            # Ranked walk: cap K strictly. The enumeration order and the
            # cap fix the rollout-seed indices in search_action.
            for (cx, cy) in ordered:
                if len([p for p in plans if p['bomb_at']]) >= cap:
                    break
                _try_bomb_plan(arena, blocked, bombs, others_xy,
                               danger, plans, x, y, cx, cy, tile_yield)
            if CHAIN_GUARD:
                # guarded chain: pure max-addition after the ranked
                # walk. Tiles the ranked walk already admitted are
                # skipped (no duplicates); chain gets its own budget of
                # K + len(extra) total bomb plans. Gate authority stays
                # in _try_bomb_plan (one gate, no special cases).
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
            # hunt-intent: one pursuit bomb plan per opponent (max
            # HUNT_PLANS total), nearest opp first. Candidate tiles
            # ring-first, then by BFS dist from us; one gate authority
            # (_try_bomb_plan). Pure max-addition after the ranked walk.
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
    return plans


def _sim_bombs(st):
    """Sim bombs [x,y,t,owner] -> game_state ((x,y),t) pairs for safety fns."""
    return [((b[0], b[1]), b[2]) for b in st['bombs']]


def _opp_move(rng, st, i, danger_now):
    """Avoid-lethal-if-possible + random (seeded); wardenlite overlay."""
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
        # bombs only under a cheap warden guard, and only when not
        # cornered (pre-blast safe mobility >= 2 incl. WAIT, the escape
        # proxy; the opp's own blast seals its neighbours by
        # construction, so mobility is checked before dropping).
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
    """Cheap post-prefix policy. Danger-aware: if own tile is lethal
    within 2 steps, flee to the valid move minimizing near-term danger
    (tie-break: coin-greedy); else coin-greedy. No net, V prices the
    leaf. The danger check keeps bomb plans from walking into their own
    blast, which would price every bomb at -8."""
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
    import copy
    from .sim import step as sim_step, margin, to_game_state
    from .safety import opp_can_escape, future_danger
    rng = None if CRN else np.random.default_rng(seed)
    st = copy.deepcopy(st0)
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
    # phase 2: settle to detonation with danger-aware continuation.
    # Settle-to-quiet (not a fixed horizon): bomb plans must see their
    # consequences, and truncating before detonation scores them on
    # garbage pre-blast V. Cap = prefix + 7 (detonation at +4/+5,
    # linger +1).
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
        # certify: opponents currently in any soon-detonating blast that
        # cannot escape -> forced kills (optimal-flight assumption).
        # CERT_OWN restricts the trigger to our bombs (owner 0;
        # pre-existing bombs owner -1 = unknown stay certifiable),
        # since the engine pays nobody for an opp self-trap.
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
    payoff += 5.0 * certified_kills
    payoff += margin(st) - m0 - (st['agents'][0]['score'] - st0['agents'][0]['score'])
    # margin() already includes score deltas; the last line adds the
    # opponent-score movement only (own score counted once via margin).
    return payoff, st


def search_action(game_state, safety, model, t0, budget):
    """Bounded best-first over plans. Returns (action|None, debug)."""
    import torch
    from .sim import from_game_state, to_game_state, margin
    from .features import state_to_features
    dbg = {'plans': 0, 'leaves': 0, 'exhausted': False}
    try:
        st0 = from_game_state(game_state)
    except Exception:
        return None, dbg
    plans = gen_plans(game_state, safety)[:PLAN_CAP]
    if not plans:
        return None, dbg
    rnd = int(game_state.get('round', 0))
    stp = int(game_state.get('step', 0))
    scored = []
    v_batch, v_idx = [], []
    dbg['seeds'] = SEEDS
    dbg['move_seeds'] = MOVE_SEEDS
    base_seed = 1000 * rnd + stp
    for pi_, plan in enumerate(plans):
        if (time.perf_counter() - t0) >= budget:
            dbg['exhausted'] = True
            break
        # Average the exact-payoff part over opponent-policy rollouts
        # (opponent RNG is the noise source). MOVE plans use MOVE_SEEDS
        # rollouts (coin-side variance reduction); BOMB plans stay
        # single-seed (W_DEATH veto calibration). Death is never
        # averaged: a plan that kills us in any seed takes the full
        # W_DEATH veto, since ruin is not compensable by upside. V
        # prices the primary rollout's leaf only.
        n_seeds = MOVE_SEEDS if plan['bomb_at'] is None else 1
        pay_sum, end = 0.0, None
        died_any = False
        for j in range(n_seeds):
            if (time.perf_counter() - t0) >= budget:
                dbg['exhausted'] = True
                break
            # CRN drops the plan index from the seed; the per-tick
            # opponent RNG inside score_plan then keys on (seed, tick,
            # opponent), sharing draws across plans.
            payoff_j, end_j = score_plan(
                st0, plan,
                seed=base_seed + 7919 * j + (0 if CRN else pi_))
            died_j = not end_j['agents'][0]['alive']
            pay_sum += payoff_j + (W_DEATH if died_j else 0.0)
            died_any = died_any or died_j
            if end is None:
                end = end_j
        if end is None:
            continue  # budget died mid-plan: drop it, keep the base fallback
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
    # Arbitration: search owns BOMB decisions only. A bomb plan executes
    # iff it beats the best move plan by BOMB_MARGIN; else None -> pi
    # decides the move. Move plans are the baseline. The margin scales
    # with the bomb tile's pre-rollout yield, so junk tiles clear a
    # higher bar (still arbitration).
    bombs = [r for r in scored if r[1]['bomb_at'] is not None]
    moves = [r for r in scored if r[1]['bomb_at'] is None]
    best_move = max([r[0] for r in moves], default=float('-inf'))
    bombs.sort(key=lambda r: r[0], reverse=True)
    dbg['best_move'] = float(best_move) if moves else None
    dbg['best_bomb'] = float(bombs[0][0]) if bombs else None
    if bombs:
        try:
            _y = float(bombs[0][1].get('tile_yield', 2.0))
        except Exception:
            _y = 2.0
        margin_eff = BOMB_MARGIN + YIELD_GAMMA * max(0.0, 2.0 - _y)
        dbg['margin_eff'] = float(margin_eff)
        dbg['best_yield'] = float(_y)
        if (bombs[0][0] - best_move) > margin_eff:
            return bombs[0][1]['first'], dbg
    return None, dbg
