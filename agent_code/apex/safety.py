"""Exact blast + time-expanded escape solver (numpy only, no torch).

Mirrors items.py:Bomb.get_blast_coords and environment.py movement rules,
but fixes rule_based_agent flaws:
 - wall-aware blast (rule_based ignores walls in bomb_map)
 - future danger for t=0..H (rule_based only checks timer==0)
 - time-expanded BFS escape (rule_based uses same-row/col heuristic)
"""
from collections import deque
import numpy as np

import os as _os


def _env_flag(name):
    return _os.environ.get(name, '0') == '1'


# E47 warden-discipline transplants (default off = validated ship behavior;
# probes assert default-off identity on real states before any screen):
#  _PAYOFF_TIER: non-opp bombs need crates>=2 or fast escape (dist<=2).
#    Warden bombs 1-crate spots only with hyp_dist<=2, 2+ crates with <=3;
#    overlord allows 1-crate at dist-3 (E27: 2.7% vs 1.4% lethal).
#  _MUSTFLEE1: refuse BOMB while own tile is lethal NEXT step (danger[1]).
#    Warden never bombs under threat<=1; overlord vetoes only danger[0].
_PAYOFF_TIER = _env_flag('OVERLORD_G_PAYOFF_TIER')
_MUSTFLEE1 = _env_flag('OVERLORD_G_MUSTFLEE1')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVE_DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0), 'WAIT': (0, 0)}
BOMB_TIMER_DEFAULT = 4
BOMB_POWER_DEFAULT = 3
HORIZON = 8


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


def escape_bfs(pos, arena, bombs, others_xy, danger, horizon=HORIZON):
    """Time-expanded BFS. Returns (safe_time, dist_to_safe).

    safe_time[(dx,dy)] True if first step leads to a path surviving to horizon
    or reaching a tile safe for all future known danger.
    dist_to_safe: steps to nearest tile with no future danger, inf if none.
    """
    x0, y0 = int(pos[0]), int(pos[1])
    W, H = arena.shape[0], arena.shape[1]
    bomb_set = set()
    for (xy, _t) in (bombs or []):
        bomb_set.add((int(xy[0]), int(xy[1])))
    other_set = set((int(a), int(b)) for (a, b) in (others_xy or []))

    def blocked(nx, ny):
        if not (0 <= nx < W and 0 <= ny < H):
            return True
        if arena[nx, ny] != 0:
            return True
        if (nx, ny) in bomb_set:
            return True
        if (nx, ny) in other_set and (nx, ny) != (x0, y0):
            return True
        return False

    # BFS over (x,y,t); first-step tracking
    visited = np.zeros((W, H, horizon + 1), dtype=bool)
    # first move index: 0..4 for UP/DOWN/LEFT/RIGHT/WAIT
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
    q = deque()
    # seed t=0 (even if currently dangerous, we still try to flee)
    if not (0 <= x0 < W and 0 <= y0 < H):
        return {}, float('inf')
    visited[x0, y0, 0] = True
    for mi, (dx, dy) in enumerate(moves):
        nx, ny = x0 + dx, y0 + dy
        if blocked(nx, ny):
            continue
        if danger[1, nx, ny] if horizon >= 1 else danger[0, nx, ny]:
            # stepping into blast next step is death unless it clears;
            # still enqueue only if survivable (checked at pop), skip here
            # to keep branching small, but allow if no alternative (handled below)
            pass
        if not visited[nx, ny, 1 if horizon >= 1 else 0]:
            visited[nx, ny, 1 if horizon >= 1 else 0] = True
            q.append((nx, ny, 1 if horizon >= 1 else 0, mi))
    # also consider staying if horizon==0
    safe_first = {}
    # BFS
    parent_first = {}
    dist_to_safe = float('inf')
    # check if start itself is future-safe
    start_safe = not any(danger[t, x0, y0] for t in range(min(2, horizon + 1)))
    # standard BFS
    found_safe = {}
    while q:
        cx, cy, ct, fmi = q.popleft()
        # is this tile safe for all remaining known danger?
        future_hit = False
        for t in range(ct, horizon + 1):
            if danger[t, cx, cy]:
                future_hit = True
                break
        if not future_hit:
            if fmi not in found_safe:
                found_safe[fmi] = ct
            if ct < dist_to_safe:
                dist_to_safe = ct
        if ct == horizon:
            if fmi not in safe_first:
                # survived to horizon without hitting danger along path?
                safe_first[fmi] = True
            continue
        # expand if current tile not dangerous at ct
        if danger[ct, cx, cy]:
            continue
        if fmi not in safe_first and ct >= 1 and not future_hit:
            safe_first[fmi] = True
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            nt = ct + 1
            if nt > horizon or visited[nx, ny, nt]:
                continue
            if blocked(nx, ny):
                continue
            if danger[nt, nx, ny]:
                continue
            visited[nx, ny, nt] = True
            q.append((nx, ny, nt, fmi))
    # map first-move index to delta
    deltas = moves
    result = {}
    for mi, d in enumerate(deltas):
        result[d] = (mi in safe_first) or (mi in found_safe)
    # WAIT may be unsafe but start could still be ok short-term; keep as computed
    return result, dist_to_safe


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
    can_escape = any(safe_hyp.get(d, False) for d in [(0, -1), (0, 1), (-1, 0), (1, 0)])
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
    # E47 tiered payoff (warden discipline, default off): non-opp bombs
    # need crates>=2 or fast escape (dist<=2).
    if _PAYOFF_TIER and valid.get('BOMB', False) and can_escape:
        try:
            if opps_tmp == 0 and not (crates_tmp >= 2 or float(dist_hyp) <= 2):
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
    # E47 must-flee strictness (warden discipline, default off): refuse
    # BOMB while own tile is lethal NEXT step (threat<=1 gates warden's
    # want_bomb; overlord only vetoed danger[0]). Selection veto only —
    # the hypothetical-escape verdict is untouched.
    try:
        if _MUSTFLEE1 and horizon >= 1 and bool(danger[1, x, y]):
            safe['BOMB'] = False
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
