"""Harvey blast, danger, and time-expanded escape analysis.

The implementation mirrors items.py bomb geometry and environment.py
movement rules while accounting for:
 - wall-aware blast (rule_based ignores walls in bomb_map)
 - future danger for t=0..H (rule_based only checks timer==0)
 - time-expanded BFS escape (rule_based uses same-row/col heuristic)

The hot escape loop uses flat byte/int buffers and a precomputed latest-
lethal map to stay within the per-step tournament time limit.
"""
from collections import deque
import os

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVE_DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0), 'WAIT': (0, 0)}
BOMB_TIMER_DEFAULT = 4
BOMB_POWER_DEFAULT = 3


def _env_int(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


# Eight ticks cover every known bomb blast and its lingering explosion.
HORIZON = _env_int('REAPER_HORIZON', 8, 4, 12)

# Minimum distinct first-step escape directions required after planting.
# BOMB_MARGIN remains an import-compatible alias for older scripts.
ESC_MARGIN = _env_int('ARBITER_BOMB_ESC_MARGIN', 1, 1, 4)
BOMB_MARGIN = ESC_MARGIN

# Optional joint-route analysis models moving agents and adversarial body blocks.
JOINT_ROUTES = os.environ.get('ARBITER_JOINT_ROUTES', '0') == '1'
DYNAMIC_ROUTES = os.environ.get(
    'ARBITER_DYNAMIC_ROUTES', '1' if JOINT_ROUTES else '0') == '1'
BODYBLOCK = os.environ.get(
    'ARBITER_BODYBLOCK', '1' if JOINT_ROUTES else '0') == '1'
JOINT_HORIZON = _env_int('ARBITER_JOINT_HORIZON', 6, 4, 8)
JOINT_BODY_D = _env_int('ARBITER_JOINT_BODY_D', 3, 1, 12)
_ROUTE_CAP = 100000


def true_blast(arena, x, y, power=BOMB_POWER_DEFAULT):
    """Exact blast coords replicating items.py (stops at stone wall -1 only)."""
    coords = [(x, y)]
    W, H = arena.shape[0], arena.shape[1]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < W and 0 <= ny < H):
                break
            if arena[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def future_danger(arena, bombs, explosion_map, horizon=HORIZON,
                  bomb_timer=BOMB_TIMER_DEFAULT, power=BOMB_POWER_DEFAULT):
    """Return bool array danger[t,x,y] for t=0..horizon.

    - t=0 includes current explosion_map>0 cells (plus one extra step of
      persistence since explosion_map timer-1 encoding loses timer==1 case).
    - bombs ((xb,yb),t): blast active at step t and t+1 (detonation step +
      lingering lethal step). Clamp t to horizon.
    """
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
                           bomb_timer=BOMB_TIMER_DEFAULT, power=BOMB_POWER_DEFAULT):
    """Copy of danger with an extra own bomb placed now at (x,y)."""
    out = danger.copy()
    for dt in (bomb_timer, bomb_timer + 1):
        if 0 <= dt <= horizon:
            for (cx, cy) in true_blast(arena, x, y, power):
                out[dt, cx, cy] = True
    return out


def first_lethal(danger, horizon=HORIZON):
    """first_lethal[x, y] = earliest t in 0..horizon that is lethal, else
    horizon+1 (sentinel). Kept for API compatibility; the escape BFS uses
    last_lethal below (earliest is insufficient when danger windows are
    non-contiguous, e.g. a timer-0 bomb at t=0..1 plus another at t=3..4)."""
    d = np.asarray(danger)
    any_d = d.any(axis=0)
    fl = np.where(any_d, d.argmax(axis=0), horizon + 1).astype(np.int32)
    return fl


def last_lethal(danger, horizon=HORIZON):
    """last_lethal[x, y] = latest t in 0..horizon that is lethal, else -1.

    The correct O(1) predicate for "is this tile lethal at some t >= ct"
    is `last_lethal >= ct`: danger windows may be non-contiguous (a
    timer-0 bomb marks t=0 and t=1; a farther bomb marks t=3 and t=4), so
    the earliest lethal time alone can miss a later window."""
    d = np.asarray(danger)
    any_d = d.any(axis=0)
    ll = np.where(any_d, horizon - d[::-1].argmax(axis=0), -1).astype(np.int32)
    return ll


def escape_bfs(pos, arena, bombs, others_xy, danger, horizon=HORIZON):
    """Time-expanded BFS. Returns (safe_time, dist_to_safe).

    safe_time[(dx,dy)] True if first step leads to a path surviving to horizon
    or reaching a tile safe for all future known danger.
    dist_to_safe: steps to nearest tile with no future danger, inf if none.

    Hot path: everything is flattened to bytes/ints once per call (arena,
    danger, first-lethal) so the BFS inner loop is pure Python integer work.
    """
    x0, y0 = int(pos[0]), int(pos[1])
    W, H = arena.shape[0], arena.shape[1]
    if not (0 <= x0 < W and 0 <= y0 < H):
        return {}, float('inf')

    ar = np.asarray(arena)
    free_b = (ar == 0).reshape(-1).tobytes()          # 1 == walkable floor
    dng = np.asarray(danger, dtype=bool)
    dang_b = dng.reshape(-1).tobytes()                # (t*W + x)*H + y
    # E88 fix: "lethal at some t >= ct" needs the LAST lethal time, not the
    # first (non-contiguous bomb windows), and arrival lethality must be
    # checked BEFORE a tile is marked safe. Pre-fix, a tile lethal at
    # t=0..1 (timer-0 bomb) reached at ct=1 was added to found_safe.
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
            continue                                   # lethal on arrival
        # lethal at some t in [ct, horizon]?  (ll is the latest lethal t)
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
    # WAIT may be unsafe but start could still be ok short-term; keep as computed
    return result, dist_to_safe


def escape_route_profile(pos, arena, bombs, danger, horizon=JOINT_HORIZON):
    """Count survivable time-expanded routes without static agent blockers.

    The old trap certificate passed every current agent position to
    ``escape_bfs`` as a permanent wall.  That can declare a kill merely
    because an agent occupies an exit *now*, although it moves away before
    detonation.  This profile intentionally gives the escaping player the
    benefit of that doubt: only arena walls/crates, bombs and timed danger
    constrain it.  Counts are capped because only relative route abundance is
    useful to diagnostics and bomb scoring.
    """
    try:
        ar = np.asarray(arena)
        dng = np.asarray(danger, dtype=bool)
        W, Hh = ar.shape
        x0, y0 = int(pos[0]), int(pos[1])
        if not (0 <= x0 < W and 0 <= y0 < Hh):
            return {'survives': False, 'routes': 0, 'positions': 0,
                    'first_steps': 0}
        bomb_cells = {(int(xy[0]), int(xy[1]))
                      for (xy, _t) in (bombs or [])}
        moves = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
        # (position, first move) -> number of routes reaching it.
        cur = {((x0, y0), None): 1}
        for t in range(1, int(horizon) + 1):
            nxt = {}
            for ((cx, cy), first), count in cur.items():
                for dx, dy in moves:
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < W and 0 <= ny < Hh):
                        continue
                    if ar[nx, ny] != 0 or (nx, ny) in bomb_cells:
                        continue
                    if t < dng.shape[0] and bool(dng[t, nx, ny]):
                        continue
                    f = (dx, dy) if first is None else first
                    key = ((nx, ny), f)
                    nxt[key] = min(_ROUTE_CAP,
                                   nxt.get(key, 0) + int(count))
            cur = nxt
            if not cur:
                break
        routes = min(_ROUTE_CAP, sum(cur.values()))
        return {
            'survives': bool(cur),
            'routes': int(routes),
            'positions': len({p for (p, _f) in cur}),
            'first_steps': len({f for (_p, f) in cur if f is not None}),
        }
    except Exception:
        # Fail open: an experimental certificate must never invent a kill or
        # veto a bomb because its own diagnostic calculation failed.
        return {'survives': True, 'routes': _ROUTE_CAP, 'positions': 0,
                'first_steps': 0}


def _joint_bodyblock_survives(arena, bombs, danger, self_pos, opp_pos,
                              horizon=JOINT_HORIZON):
    """Whether self has a survival policy against one moving body blocker.

    This is a small finite-horizon minimax game.  At each tick Harvey chooses a
    move, then the opponent is allowed the worst survivable simultaneous move.
    Same-destination and edge-swap conflicts are treated as a successful block
    (the conservative engine-order interpretation).  The opponent may move;
    unlike the old BFS it is never frozen in its current cell.
    """
    try:
        ar = np.asarray(arena)
        dng = np.asarray(danger, dtype=bool)
        W, Hh = ar.shape
        bomb_cells = {(int(xy[0]), int(xy[1]))
                      for (xy, _t) in (bombs or [])}
        moves = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))

        def next_cells(pos, t):
            out = []
            for dx, dy in moves:
                nx, ny = pos[0] + dx, pos[1] + dy
                if not (0 <= nx < W and 0 <= ny < Hh):
                    continue
                if ar[nx, ny] != 0 or (nx, ny) in bomb_cells:
                    continue
                if t < dng.shape[0] and bool(dng[t, nx, ny]):
                    continue
                out.append((nx, ny))
            return tuple(out)

        solo_cache = {}
        def solo_wins(t, spos):
            key = (t, spos)
            if key in solo_cache:
                return solo_cache[key]
            if t >= horizon:
                return True
            ans = any(solo_wins(t + 1, sn)
                      for sn in next_cells(spos, t + 1))
            solo_cache[key] = ans
            return ans

        memo = {}
        def wins(t, spos, opos):
            key = (t, spos, opos)
            if key in memo:
                return memo[key]
            if t >= horizon:
                return True
            own_next = next_cells(spos, t + 1)
            if not own_next:
                memo[key] = False
                return False
            opp_next = next_cells(opos, t + 1)
            if not opp_next:
                ans = any(solo_wins(t + 1, sn) for sn in own_next)
                memo[key] = ans
                return ans
            for sn in own_next:
                robust = True
                for on in opp_next:
                    if sn == on or (sn == opos and on == spos):
                        robust = False
                        break
                    if not wins(t + 1, sn, on):
                        robust = False
                        break
                if robust:
                    memo[key] = True
                    return True
            memo[key] = False
            return False

        s0 = (int(self_pos[0]), int(self_pos[1]))
        o0 = (int(opp_pos[0]), int(opp_pos[1]))
        return bool(wins(0, s0, o0))
    except Exception:
        return True


def joint_bomb_analysis(game_state, horizon=JOINT_HORIZON):
    """Evaluate a bomb planted HERE using dynamic opponent escape routes.

    Returns a compact dict used by ``action_safety`` and diagnostics.  Forced
    kills are conservative: an opponent is forced only when it has no route
    even after all moving agents are removed as blockers.  Own robustness is
    checked separately against each nearby opponent as an adversarial moving
    body blocker.
    """
    try:
        arena = np.asarray(game_state['field'])
        _, _, bombs_left, (x, y) = game_state['self']
        x, y = int(x), int(y)
        if not bombs_left:
            return {'own_robust': False, 'own_routes': 0,
                    'forced_kills': 0, 'opponent_routes': []}
        bombs = list(game_state.get('bombs') or [])
        exp = game_state.get('explosion_map')
        danger = future_danger(arena, bombs, exp, horizon)
        danger_h = with_hypothetical_bomb(
            danger, arena, x, y, horizon, BOMB_TIMER_DEFAULT,
            BOMB_POWER_DEFAULT)
        bombs_h = bombs + [((x, y), BOMB_TIMER_DEFAULT)]
        own = escape_route_profile((x, y), arena, bombs_h, danger_h,
                                   horizon)
        blast = set(true_blast(arena, x, y))
        opp_routes = []
        forced = 0
        robust = bool(own['survives'])
        for opp in (game_state.get('others') or []):
            ox, oy = int(opp[3][0]), int(opp[3][1])
            prof = escape_route_profile((ox, oy), arena, bombs_h,
                                        danger_h, horizon)
            threatened = (ox, oy) in blast
            if threatened and not prof['survives']:
                forced += 1
            opp_routes.append({
                'pos': [ox, oy], 'threatened': bool(threatened),
                'routes': int(prof['routes']),
                'positions': int(prof['positions']),
            })
            if robust and abs(ox - x) + abs(oy - y) <= JOINT_BODY_D:
                robust = _joint_bodyblock_survives(
                    arena, bombs_h, danger_h, (x, y), (ox, oy), horizon)
        return {
            'own_robust': bool(robust),
            'own_routes': int(own['routes']),
            'own_first_steps': int(own['first_steps']),
            'forced_kills': int(forced),
            'opponent_routes': opp_routes,
        }
    except Exception:
        return {'own_robust': True, 'own_routes': _ROUTE_CAP,
                'forced_kills': 0, 'opponent_routes': []}


def danger_no_explosion(arena, bombs, horizon=HORIZON,
                        bomb_timer=BOMB_TIMER_DEFAULT,
                        power=BOMB_POWER_DEFAULT):
    """Shared precomputed danger for the trap scan.

    Same convention as opp_can_escape's internal danger (no explosion map:
    an opponent standing in a current explosion is already dead). Callers
    that evaluate many hypothetical bomb spots compute this ONCE per step
    and pass it via `danger=` instead of rebuilding it per spot.
    """
    return future_danger(arena, bombs, None, horizon, bomb_timer, power)


def opp_can_escape(arena, bombs, opp_pos, bomb_pos, others_xy=None,
                   horizon=HORIZON, bomb_timer=BOMB_TIMER_DEFAULT,
                   power=BOMB_POWER_DEFAULT, danger=None):
    """Can an opponent standing at opp_pos survive a bomb at bomb_pos?

    Adds the hypothetical bomb to danger + bomb set, then runs the
    time-expanded escape BFS from the opponent's tile. WAIT counts as a
    survivable first move (the opponent may be safe standing still).
    Returns (can_survive, dist_to_safe).

    Used by features.py as the trap signal: opps_hit with no escape
    for the opponent = a kill opportunity. Pass a precomputed `danger`
    (from danger_no_explosion) to skip the per-call rebuild.
    """
    try:
        ox, oy = int(opp_pos[0]), int(opp_pos[1])
        bx, by = int(bomb_pos[0]), int(bomb_pos[1])
        if danger is None:
            danger = future_danger(arena, bombs, None, horizon,
                                   bomb_timer, power)
        danger = with_hypothetical_bomb(danger, arena, bx, by, horizon,
                                        bomb_timer, power)
        bombs_hyp = list(bombs or []) + [((bx, by), bomb_timer)]
        # E134 joint-route mode removes live agents from the opponent's
        # static obstacle set.  A kill is certified only if the opponent has
        # no escape even when currently occupied exits may open next tick.
        blockers = [] if DYNAMIC_ROUTES else (others_xy or [])
        safe_first, dist = escape_bfs((ox, oy), arena, bombs_hyp,
                                      blockers, danger, horizon)
        can = any(safe_first.get(d, False)
                  for d in ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)))
        return bool(can), float(dist)
    except Exception:
        return True, float('inf')


def bomb_here_traps(game_state, safety=None, danger=None):
    """Does bombing HERE right now guarantee a kill? (E37/P5 tactical.)

    For each opponent in our current blast, runs the exact opponent escape
    BFS against a hypothetical bomb at our tile. Returns (traps, min_esc)
    where traps=True iff some opponent cannot escape. Exact computation —
    no learning, no approximation — so a True here is a forced +5.
    Shares one danger map across opponents (same convention as the trap
    feature: no explosion map). Returns (False, inf) when we have no bomb,
    nobody is in our blast, or anything fails.
    """
    try:
        arena = np.asarray(game_state['field'])
        _, _, bombs_left, (x, y) = game_state['self']
        if not bombs_left:
            return False, float('inf')
        x, y = int(x), int(y)
        if safety is None:
            blast = set(true_blast(arena, x, y))
        else:
            blast = safety.get('blast_if_bomb') or set()
        others = game_state.get('others') or []
        opps = [(int(xy[0]), int(xy[1]))
                for (_, _, _, xy) in others
                if (int(xy[0]), int(xy[1])) in blast]
        if not opps:
            return False, float('inf')
        bombs = game_state.get('bombs', [])
        others_xy = [xy for (_, _, _, xy) in others]
        if danger is None:
            danger = danger_no_explosion(arena, bombs)
        min_esc = float('inf')
        for (ox, oy) in opps:
            can, dist = opp_can_escape(arena, bombs, (ox, oy), (x, y),
                                       others_xy, danger=danger)
            try:
                if float(dist) < min_esc:
                    min_esc = float(dist)
            except (TypeError, ValueError):
                pass
            if not can:
                return True, min_esc
        return False, min_esc
    except Exception:
        return False, float('inf')


def action_safety(game_state, horizon=HORIZON, power=BOMB_POWER_DEFAULT,
                  bomb_timer=BOMB_TIMER_DEFAULT):
    """High-level helper used by act(). Returns dict with:
    valid, safe (first-step survivable), can_escape_if_bomb, danger, blast_if_bomb.
    """
    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state.get('bombs', [])
    others_xy = [xy for (n, s, b, xy) in game_state.get('others', [])]
    explosion_map = game_state.get('explosion_map', np.zeros_like(arena))
    bomb_set = set((int(xy[0]), int(xy[1])) for (xy, _t) in bombs)
    other_set = set((int(a), int(b)) for (a, b) in others_xy)

    danger = future_danger(arena, bombs, explosion_map, horizon, bomb_timer, power)

    # validity mirrors environment.perform_agent_action + rule_based valid check
    cands = {'UP': (x, y - 1), 'DOWN': (x, y + 1), 'LEFT': (x - 1, y),
             'RIGHT': (x + 1, y), 'WAIT': (x, y)}
    valid = {}
    W, H = arena.shape[0], arena.shape[1]
    for a, (nx, ny) in cands.items():
        ok = True
        if not (0 <= nx < W and 0 <= ny < H):
            ok = False
        elif arena[nx, ny] != 0:
            ok = False
        elif (nx, ny) in bomb_set:
            ok = False
        elif (nx, ny) in other_set and (nx, ny) != (x, y):
            ok = False
        elif np.asarray(explosion_map)[nx, ny] >= 1:
            ok = False
        valid[a] = ok
    # BOMB valid if allowed by engine (bombs_left) and not standing in explosion
    valid['BOMB'] = bool(bombs_left) and np.asarray(explosion_map)[x, y] < 1

    safe_first, dist_safe = escape_bfs((x, y), arena, bombs, others_xy, danger, horizon)
    inv = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT', (0, 0): 'WAIT'}
    safe = {}
    for d, ok_first in safe_first.items():
        a = inv[d]
        safe[a] = bool(ok_first and valid.get(a, False))
    for a in cands:
        if a not in safe:
            safe[a] = False
    # BOMB safety: can we escape if we drop now?
    # NOTE: after dropping, own tile becomes a bomb tile (blocked for re-entry).
    # escape_bfs blocks bomb tiles, so pass bombs+own for hypothetical.
    danger_hyp = with_hypothetical_bomb(danger, arena, x, y, horizon, bomb_timer, power)
    bombs_hyp = list(bombs or []) + [((x, y), bomb_timer)]
    safe_hyp, dist_hyp = escape_bfs((x, y), arena, bombs_hyp, others_xy, danger_hyp, horizon)
    # Count distinct first moves, not merely the number of complete routes.
    _bomb_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    n_esc = sum(1 for d in _bomb_dirs if safe_hyp.get(d, False))
    can_escape = n_esc >= ESC_MARGIN
    # WAIT is not an escape from own bomb (staying dies at t=4/5 if in blast)
    # so exclude (0,0) from can_escape.
    # BOMB is safe iff valid and escape exists and current tile escapable
    safe['BOMB'] = bool(valid['BOMB'] and can_escape)
    # In a one-wide corridor, require a quick exit plus payoff or a trapped foe.
    blast_tmp = set(true_blast(arena, x, y, power))
    crates_tmp = sum(1 for (cx, cy) in blast_tmp if arena[cx, cy] == 1)
    opps_tmp = sum(1 for (ox, oy) in others_xy if (int(ox), int(oy)) in blast_tmp)
    try:
        up_free = (0 <= y - 1 < H and arena[x, y - 1] == 0 and (x, y - 1) not in bomb_set)
        dn_free = (0 <= y + 1 < H and arena[x, y + 1] == 0 and (x, y + 1) not in bomb_set)
        lf_free = (0 <= x - 1 < W and arena[x - 1, y] == 0 and (x - 1, y) not in bomb_set)
        rt_free = (0 <= x + 1 < W and arena[x + 1, y] == 0 and (x + 1, y) not in bomb_set)
        in_corridor = ((not up_free and not dn_free) or (not lf_free and not rt_free))
        if in_corridor and valid['BOMB'] and can_escape:
            if opps_tmp == 0:
                # need fast escape + real payoff, else forbid
                if not (dist_hyp <= 2 and crates_tmp >= 1):
                    safe['BOMB'] = False
                    can_escape = False
    except Exception:
        pass
    # An escape distance equal to the bomb timer is too late in engine order.
    if valid.get('BOMB', False) and can_escape:
        try:
            if float(dist_hyp) > 3:
                safe['BOMB'] = False
                can_escape = False
        except Exception:
            pass
    # CRITICAL: staying (WAIT/BOMB) dies if current tile explodes THIS step.
    # Moving away can still save you, but staying cannot.
    try:
        if bool(danger[0, x, y]):
            safe['WAIT'] = False
            safe['BOMB'] = False
            can_escape = False
    except Exception:
        pass

    blast = set(true_blast(arena, x, y, power))
    crates_hit = sum(1 for (cx, cy) in blast if arena[cx, cy] == 1)
    opps_hit = sum(1 for (ox, oy) in others_xy if (ox, oy) in blast)
    try:
        dist_hyp_val = float(dist_hyp)
    except Exception:
        dist_hyp_val = float('inf')
    joint = None
    if (DYNAMIC_ROUTES or BODYBLOCK) and safe.get('BOMB', False):
        try:
            joint = joint_bomb_analysis(game_state, JOINT_HORIZON)
            joint['bodyblock_veto'] = bool(
                BODYBLOCK and not joint.get('own_robust', True))
            if joint['bodyblock_veto']:
                safe['BOMB'] = False
                can_escape = False
        except Exception:
            joint = None
    out = {'valid': valid, 'safe': safe, 'danger': danger,
           'can_escape_if_bomb': can_escape, 'dist_to_safe': dist_safe,
           'dist_hyp': dist_hyp_val,
           'crates_hit_if_bomb': crates_hit, 'opps_hit_if_bomb': opps_hit,
           'blast_if_bomb': blast}
    if joint is not None:
        out['joint_routes'] = joint
    return out
