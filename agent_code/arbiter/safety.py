"""ARBITER vendored copy of agent_code/reaper/safety.py (E48 rule).
Exact blast + time-expanded escape solver (numpy only, no torch).
Untouched apart from this header; probe parity vs reaper in
scripts/probe_arbiter.py.

Original docstring follows:

Mirrors items.py:Bomb.get_blast_coords and environment.py movement rules,
but fixes rule_based_agent flaws:
 - wall-aware blast (rule_based ignores walls in bomb_map)
 - future danger for t=0..H (rule_based only checks timer==0)
 - time-expanded BFS escape (rule_based uses same-row/col heuristic)

Latency (E37/P1): the tournament allows 0.5 s/step on ONE Ryzen 5 2600
thread and an overrun also costs the NEXT step (environment.py:448-460),
so the escape solver is written against flat byte/int buffers instead of
numpy scalar indexing, and the "is this tile lethal at any t >= ct" scan is
collapsed to an O(1) lookup against a precomputed first-lethal map. Both
are semantics-preserving: `scripts/probe_reaper_features.py` group 4 still
asserts exact `valid`/`safe` parity with the frozen overlord copy.
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


# Default stays 8 so the overlord parity probe is exact; bombs can only
# threaten t <= BOMB_TIMER+1 == 5, so REAPER_HORIZON=6 is the A/B candidate.
HORIZON = _env_int('REAPER_HORIZON', 8, 4, 12)

# E87: minimum post-plant first-step escape directions for BOMB to be
# certified safe. 1 == ship behavior (any()); the gate run uses 2.
BOMB_MARGIN = _env_int('ARBITER_BOMB_MARGIN', 1, 1, 4)


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
    horizon+1 (sentinel). Lets the escape BFS answer "is this tile lethal at
    some t >= ct?" in O(1) instead of rescanning the whole time axis."""
    d = np.asarray(danger)
    any_d = d.any(axis=0)
    fl = np.where(any_d, d.argmax(axis=0), horizon + 1).astype(np.int32)
    return fl


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
    fl_flat = first_lethal(dng, horizon).reshape(-1).tolist()
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
        fl = fl_flat[ci]
        # lethal at some t in [ct, horizon]?  (fl is the earliest lethal t)
        future_hit = fl <= horizon and fl >= ct
        if not future_hit:
            found_safe.add(fmi)
            if ct < dist_to_safe:
                dist_to_safe = ct
        if ct == horizon:
            safe_first.add(fmi)
            continue
        if dang_b[ct * plane + ci]:
            continue                                   # died on this tile
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
        safe_first, dist = escape_bfs((ox, oy), arena, bombs_hyp,
                                      others_xy or [], danger, horizon)
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
    # E87 (ARBITER_BOMB_MARGIN, default 1 = ship-identical): require at
    # least M post-plant first-step escape DIRECTIONS (not just one).
    # Phase A diagnosis: single-escape plants died 18/39 (46%); esc>=2
    # plants 3/1379 (0.2%). n_esc >= 1 is exactly the legacy any().
    _bomb_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    n_esc = sum(1 for d in _bomb_dirs if safe_hyp.get(d, False))
    can_escape = n_esc >= BOMB_MARGIN
    # WAIT is not an escape from own bomb (staying dies at t=4/5 if in blast)
    # so exclude (0,0) from can_escape.
    # BOMB is safe iff valid and escape exists and current tile escapable
    safe['BOMB'] = bool(valid['BOMB'] and can_escape)
    # Corridor discipline: 1-wide corridor bombings needing long outrun are the
    # #1 suicide cause. Allow only if quick off-ramp (dist<=2) + crates, or kill.
    # Compute blast/crates/opps first for this decision.
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
    # E14b margin rule (sentinel audit 60rd: dist_hyp 4.0 plants 23/27 fatal
    # vs 2.4% at <=3.0 — a dist==timer escape always loses the race).
    if valid.get('BOMB', False) and can_escape:
        try:
            if float(dist_hyp) > 3:
                safe['BOMB'] = False
                can_escape = False
        except Exception:
            pass
    # NOTE (E21 lesson): a crate-payoff gate was tried on sentinel and
    # REJECTED after a true live test (200rd pooled neutral-to-negative:
    # vetoed slots don't convert without a crate-approach pull). Margin gate
    # only here; payoff stays an open research item, not a mask rule.
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
    return {'valid': valid, 'safe': safe, 'danger': danger,
            'can_escape_if_bomb': can_escape, 'dist_to_safe': dist_safe,
            'dist_hyp': dist_hyp_val,
            'crates_hit_if_bomb': crates_hit, 'opps_hit_if_bomb': opps_hit,
            'blast_if_bomb': blast}
