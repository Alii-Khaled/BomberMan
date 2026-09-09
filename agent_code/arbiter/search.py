"""ARBITER bounded best-first search (P1): exact-dynamics plans ranked by
exact payoff + learned leaf value. The net NEVER votes on root actions
(E62 constraint) — it evaluates consequences the heuristic cannot see.

Plan = committed own-action prefix (BFS path to a bomb tile + BOMB, or a
single move) + cheap continuation to detonation settle. Opponents use an
avoid-lethal-if-possible + random policy (seeded, deterministic per
round/step). Kills count at full value only when certified against
optimal flight (opp_can_escape); rollout-luck kills are ignored.
Score(plan) = exact margin delta (crates*w + coins + 5*certified kills
- 8*own death + coin-reveal expectation) + V_BLEND * V(s_end).

Budgets: wall-clock (shares ARBITER_TIME_BUDGET with act), plan cap,
leaf-V cap (features cost 1.18 ms vs sim.step 0.031 ms — V is the
binding constraint). Exhaustion degrades to the S0 ranking.
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
# Rollout seeds per plan (E71): the exact-payoff part of a plan score
# carries opponent-policy RNG noise; averaging over seeds shrinks it.
# V is evaluated on the primary rollout's leaf only (leaf features cost
# 1.18 ms — V over all seeds would break the step budget). Default 1 =
# exactly the validated P1 behavior (same seed formula).
SEEDS = _env_int('ARBITER_SEEDS', 1, 1, 8)
# Move-plan seeds (E78): average the exact-payoff part over
# MOVE_SEEDS opponent-policy rollouts for MOVE plans only, single
# seed for bomb plans. E71 showed seed-averaging lifts coins
# (2.50, stabler move arbitration) but dilutes the single-seed
# W_DEATH veto on bombs (+0.065 sui, E71b: veto calibration is
# single-seed). Bombs keep exactly the validated single-seed path
# (j=0 seed formula unchanged -> bit-identical bomb scores);
# moves buy the coin-side variance reduction. Default 1 = P1 flow.
MOVE_SEEDS = _env_int('ARBITER_MOVE_SEEDS', 1, 1, 8)
K = _env_int('ARBITER_SEARCH_K', 8, 0, 32)
RADIUS = _env_int('ARBITER_SEARCH_R', 4, 1, 8)
PLAN_CAP = _env_int('ARBITER_SEARCH_PLANS', 48, 1, 256)
V_BLEND = _env_float('ARBITER_V_BLEND', 1.0)
# V_OFF mirrors callbacks' ARBITER_V_OFF (V0 ablation): when set, the
# search skips leaf evaluation exactly as if V_BLEND were 0. (Lesson:
# an earlier V0 gate used V_OFF, which only zeroed the logged S0 value
# while search kept evaluating — always ablate via the path under test.)
V_OFF = os.environ.get('ARBITER_V_OFF', '0') == '1'
W_CRATE = _env_float('ARBITER_W_CRATE', 0.1)
W_DEATH = _env_float('ARBITER_W_DEATH', 8.0)
# Bomb-vs-move arbitration (P1 redesign): search OWNS bomb decisions
# (placement is what pi cannot do); pi owns moves (V RMSE +-2.4 dwarfs
# 1-step value gaps, so V-ranked moves are noise vs pi's sharp policy).
# A bomb plan executes only if it beats the best move plan by more than
# BOMB_MARGIN; else search returns None and S0 (pi) decides the move.
BOMB_MARGIN = _env_float('ARBITER_BOMB_MARGIN', 0.2)
# Coin proximity in bomb-tile ranking: blast yield alone strands the
# agent far from the coins it reveals (P1 screen: crates 20.8 but coins
# 0.85). Prefer high-yield tiles near collectable coins.
W_COIN_TILE = _env_float('ARBITER_W_COIN_TILE', 0.5)
# Yield-scaled bomb margin (E73): junk bombs (opps_hit=0 plants kill
# with P=0.003, E16) should clear a HIGHER bar, not the same 0.2.
# margin_eff = BOMB_MARGIN + YIELD_GAMMA * max(0, 2 - tile_yield),
# tile_yield = crates_in_blast + 2 * opps_in_blast at the bomb tile.
# Soft (arbitration, not veto — the veto saga stays closed) and
# default 0 = validated P1 behavior.
YIELD_GAMMA = _env_float('ARBITER_YIELD_GAMMA', 0.0)
# Opponent blast bonus in tile ranking (P1c: search bombs for crates and
# under-kills 0.20 vs S0 0.39 — pi/warden bombs opportunistically on
# opps_hit>0). Bonus per opponent in blast; certification at scoring
# (opp_can_escape) keeps it honest. Default 2.0.
W_OPP_TILE = _env_float('ARBITER_W_OPP_TILE', 2.0)
# Escape strictness: 0 = any escape first-step suffices;
# N > 0 = require escape_bfs dist_to_safe <= N. SHIP default 3.0
# (validated G1 3.95; P1d grid W2/E3 beat W2/E0 on both seeds).
ESC_DIST = _env_float('ARBITER_ESC_DIST', 3.0)
# Coin-race move plan (E75): a committed BFS path to the nearest
# reachable visible coin, scored by the exact rollout like any plan.
# Competes as a MOVE plan (bomb_at None) — a valuable coin run
# correctly raises the bomb bar via best_move. Default 0 = P1 flow.
COINRUN = os.environ.get('ARBITER_COINRUN', '0') == '1'
# Chain-bomb priority (E72): own tile + 4 neighbours always enter
# bomb candidacy (bypass ranking AND the K cap). The re-bomb step is
# the cheapest volume lever — the tile we stand on needs no travel.
# The proven-escape gate below still applies to each. Default 0 =
# validated P1 candidate flow.
CHAIN = os.environ.get('ARBITER_CHAIN', '0') == '1'
# Guarded chain (E77): own tile + 4 neighbours enter candidacy ONLY
# under warden's want_bomb guard (opps_hit > 0, or crates_hit >= 2
# with hyp_dist <= 3, or crates_hit == 1 with hyp_dist <= 2).
# E72's unguarded chain fired junk (6-step lockout displaces good
# bombs, -0.38); E73 proved junk filtering is score-neutral — the
# missing piece is guarding the ADMISSION, not the arbitration.
# Appended after ranked candidates (E72b seed-index discipline: the K
# cap covers ranked plans only; ranked plans keep their rollout-seed
# indices). Proven-escape gate below still applies. Default 0.
CHAIN_GUARD = os.environ.get('ARBITER_CHAIN_GUARD', '0') == '1'
# Opportunistic-trap credit (trap cycle): certified kills (no escape at
# any timer) pay 1.0; opponents whose escape needs >= TRAP_HARD steps
# while a bomb detonates within 2 pay TRAP_P (expected value — they stay
# alive in sim). Default off (0 = certified-only). Rationale: per-bomb
# kill rates match warden (1.5-1.6%) — the gap is VOLUME (20.4 vs 29.5
# bombs/rd), and strict certification vetoes contested bombs.
TRAP_HARD = _env_float('ARBITER_TRAP_HARD', 0.0)
TRAP_P = _env_float('ARBITER_TRAP_P', 0.5)


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


def chain_guard_ok(arena, blocked, bombs, others_xy, danger, tx, ty):
    """E77 warden-rule admission predicate for chain tiles.

    Returns (admit, tile_yield). Pure function (directly probed):
    a chain tile is admitted iff it has a proven escape (same recipe
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

    Single gate authority for ranked and chain tiles alike (E14b
    lesson: one gate, no special cases). Returns True on admission.
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
        can = any(sh.get(d, False)
                  for d in [(0, -1), (0, 1), (-1, 0), (1, 0)])
        if ESC_DIST > 0:
            try:
                can = bool(can and float(dhyp) <= ESC_DIST)
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


def gen_plans(game_state, safety, K=K, radius=RADIUS):
    """Root candidate plans. Each plan: dict(first, prefix, bomb_at)."""
    from .safety import escape_bfs, future_danger, with_hypothetical_bomb
    from .sim import yield_field
    arena = np.asarray(game_state['field'])
    _, _, bombs_left, (x, y) = game_state['self']
    x, y = int(x), int(y)
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
        # E75 coin-race plan: committed BFS path to the nearest
        # reachable visible coin (prefix ≤ 12; replan next step).
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
            # E72 chain set: own tile + 4 neighbours enter candidacy
            # (gated by CHAIN; the escape check below applies). E72b:
            # APPENDED after ranked candidates (not prepended) so the
            # ranked plans keep their rollout-seed indices — the chain
            # is a pure max-addition, never an RNG perturbation. The K
            # cap covers ranked plans; chain tiles get extras.
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
            # Ranked walk: cap K strictly. With CHAIN_GUARD off this is
            # bit-identical to the validated P1 flow (same order, same
            # cap -> same rollout-seed indices in search_action).
            for (cx, cy) in ordered:
                if len([p for p in plans if p['bomb_at']]) >= cap:
                    break
                _try_bomb_plan(arena, blocked, bombs, others_xy,
                               danger, plans, x, y, cx, cy, tile_yield)
            if CHAIN_GUARD:
                # E77 guarded chain: pure max-addition AFTER the ranked
                # walk (E72b semantics done right — the first draft
                # inflated the ranked cap instead; probe forensics).
                # Tiles the ranked walk already admitted are skipped
                # (no duplicates); chain gets its own budget of
                # K + len(extra) total bomb plans. Gate authority stays
                # in _try_bomb_plan (E14b: one gate, no special cases).
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
    return plans


def _sim_bombs(st):
    """Sim bombs [x,y,t,owner] -> game_state ((x,y),t) pairs for safety fns."""
    return [((b[0], b[1]), b[2]) for b in st['bombs']]


def _opp_move(rng, st, i, danger_now):
    """Avoid-lethal-if-possible + random (seeded)."""
    from .sim import valid_actions
    a = st['agents'][i]
    opts = valid_actions(st, i)
    if not opts:
        return 'WAIT'
    try:
        d = np.asarray(danger_now)
        bad = set()
        for act in opts:
            from .sim import _DELTAS
            dx, dy = _DELTAS[act]
            nx, ny = a['x'] + dx, a['y'] + dy
            if d.ndim == 3 and d.shape[0] > 2:
                if bool(d[0, nx, ny]) or bool(d[1, nx, ny]) \
                        or bool(d[2, nx, ny]):
                    bad.add(act)
        good = [o for o in opts if o not in bad]
        if good:
            opts = good
    except Exception:
        pass
    return opts[int(rng.integers(len(opts)))]


def _continuation(game_state, st, i, danger=None):
    """Cheap post-prefix policy. Danger-aware: if own tile is lethal
    within 2 steps, flee to the valid move minimizing near-term danger
    (tie-break: coin-greedy); else coin-greedy. No net — V prices the
    leaf. (Fix: coin-greedy continuation walked bomb plans into their
    own blast, pricing every bomb at -8 — bombs never won.)"""
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
    rng = np.random.default_rng(seed)
    st = copy.deepcopy(st0)
    m0 = margin(st)
    prefix = list(plan['prefix'])
    payoff = 0.0
    certified_kills = 0
    frac_done = set()
    ticks = 0
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
                acts.append(_opp_move(rng, st, i, danger))
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
    # Settle-to-quiet (not fixed horizon): bomb plans MUST see their
    # consequences — truncating before detonation scores them on garbage
    # pre-blast V. Cap = prefix + 7 (detonation at +4/+5, linger +1).
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
                acts.append(_opp_move(rng, st, i, danger))
            else:
                acts.append('WAIT')
        # certify: opponents currently in any soon-detonating blast that
        # cannot escape -> forced kills (optimal-flight assumption)
        try:
            for b in st['bombs']:
                if b[2] > 2:
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
    # NOTE: margin() already includes score deltas; the last line adds the
    # OPPONENT-score movement only (own score counted once via margin).
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
        # E71/E78: average the exact-payoff part over opponent-policy
        # rollouts (opponent RNG is the noise source). MOVE plans use
        # MOVE_SEEDS rollouts (coin-side variance reduction); BOMB
        # plans stay single-seed (E71b: W_DEATH veto calibration).
        # Death is NOT averaged: a plan that kills us in ANY seed
        # takes the full W_DEATH veto — ruin is not compensable by
        # upside. V prices the primary rollout's leaf only.
        n_seeds = MOVE_SEEDS if plan['bomb_at'] is None else 1
        pay_sum, end = 0.0, None
        died_any = False
        for j in range(n_seeds):
            if (time.perf_counter() - t0) >= budget:
                dbg['exhausted'] = True
                break
            payoff_j, end_j = score_plan(
                st0, plan, seed=base_seed + 7919 * j + pi_)
            died_j = not end_j['agents'][0]['alive']
            pay_sum += payoff_j + (W_DEATH if died_j else 0.0)
            died_any = died_any or died_j
            if end is None:
                end = end_j
        if end is None:
            continue  # budget died mid-plan: drop it, keep S0 fallback
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
    # Arbitration (P1 redesign): search owns BOMB decisions only. A bomb
    # plan executes iff it beats the best move plan by BOMB_MARGIN; else
    # None -> S0 (pi) decides the move. Move plans are the baseline.
    # E73: the margin scales with the bomb tile's pre-rollout yield —
    # junk tiles must clear a higher bar (soft, still arbitration).
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
