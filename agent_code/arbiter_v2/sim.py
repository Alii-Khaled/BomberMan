"""ARBITER exact forward simulator (numpy + engine semantics, no torch).

Replicates environment.py do_step order EXACTLY per simulated step:
  1. apply actions sequentially (movement via tile_is_free rules;
     BOMB iff bombs_left; else WAIT/INVALID = stand still),
  2. collect_coins (+1 standing on collectable),
  3. update_explosions (timer -= 1; 0 -> next stage; stage->1 restores
     owner's bombs_left),
  4. update_bombs (timer <= 0 -> detonate: wall-stop/crate-passthrough
     blast, clear crates, reveal hidden coins, spawn stage-0 explosion;
     else timer -= 1),
  5. evaluate_explosions (stage-0 blast tiles kill; owner +5 per victim,
     own blast = suicide, no points).
Bomb cycle: dropped at t (timer 4->3 same step) -> detonates during
step t+4 -> lethal t+4,t+5 -> bombs_left back at t+6. Matches the mask's
danger model (safety.py) and the brief (BOMB_TIMER 4, POWER 3).

TWO intentional approximations (both documented + probed):
  A1 movement order: engine applies in seating order; sim applies self
     (index 0) then opponents in others-order. Matters only when two
     agents contest one tile in one step (measured rare; replanning
     every step washes single-step artifacts out).
  A2 hidden coins: game_state hides coins under crates. The engine
     reveals them deterministically; the sim books EXPECTED coins per
     cleared crate (p = remaining-hidden / remaining-crates, all terms
     observable: total 9 - visible - collected_est, where collected_est
     = total_score - 5 * observed_deaths). Exact payoffs (crates, kills,
     deaths, collections of visible coins) are unaffected.
Everything else (blast geometry, timers, scoring, blocking) is exact
and parity-probed against the engine's own step functions in
scripts/probe_arbiter_sim.py.
"""
import numpy as np

POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2
REWARD_COIN = 1
REWARD_KILL = 5
TOTAL_COINS_CLASSIC = 9
MAX_STEPS = 400

_DELTAS = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0),
           'RIGHT': (1, 0), 'WAIT': (0, 0), 'BOMB': (0, 0)}
ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB')


def blast_coords(arena, x, y, power=POWER):
    """== items.Bomb.get_blast_coords (wall-stop, crate-passthrough)."""
    out = [(int(x), int(y))]
    W, H = arena.shape[0], arena.shape[1]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = int(x) + dx * i, int(y) + dy * i
            if not (0 <= nx < W and 0 <= ny < H):
                break
            if arena[nx, ny] == -1:
                break
            out.append((nx, ny))
    return out


def from_game_state(gs, total_coins=TOTAL_COINS_CLASSIC):
    """SimState from a live game_state. Agent 0 is always self."""
    arena = np.asarray(gs['field']).astype(np.int8).copy()
    _, self_score, bombs_left, (sx, sy) = gs['self']
    coins = [[int(cx), int(cy), 1]
             for (cx, cy) in (gs.get('coins', []) or [])]
    agents = [{'x': int(sx), 'y': int(sy), 'alive': True,
               'score': int(self_score), 'bombs_left': bool(bombs_left),
               'is_self': True}]
    for (n, s, b, xy) in (gs.get('others', []) or []):
        agents.append({'x': int(xy[0]), 'y': int(xy[1]), 'alive': True,
                       'score': int(s), 'bombs_left': bool(b),
                       'is_self': False})
    bombs = [[int(bxy[0]), int(bxy[1]), int(t), -1]
             for (bxy, t) in (gs.get('bombs', []) or [])]
    expm = np.asarray(gs.get('explosion_map',
                             np.zeros_like(arena, dtype=float)))
    # explosion_map stores max(timer-1): invert to timer (min 1 while live).
    # Stage is unobservable; assume stage 0 (lethal) iff value >= 1, else
    # stage-0-with-1-step-left (the timer-1 encoding loss case the mask
    # compensates). Detonation ownership unknown -> owner -1 (no score).
    explosions = []
    for (ex, ey) in zip(*np.where(expm > 0)):
        explosions.append({'coords': [(int(ex), int(ey))],
                           'timer': int(expm[int(ex), int(ey)]) + 1,
                           'stage': 0, 'owner': -1})
    return {'arena': arena, 'coins': coins, 'agents': agents,
            'bombs': bombs, 'explosions': explosions,
            'step': int(gs.get('step', 0)), 'total_coins': int(total_coins)}


def to_game_state(st):
    """game_state-like dict from a SimState (for V features at leaves)."""
    arena = np.asarray(st['arena'])
    expm = np.zeros(arena.shape, dtype=float)
    for e in st['explosions']:
        for (cx, cy) in e['coords']:
            if expm[cx, cy] < e['timer'] - 1:
                expm[cx, cy] = max(0, e['timer'] - 1)
    me = st['agents'][0]
    others = []
    for a in st['agents'][1:]:
        if a['alive']:
            others.append(('o', a['score'], a['bombs_left'],
                           (a['x'], a['y'])))
    return {'round': 1, 'step': st['step'], 'field': arena.copy(),
            'bombs': [((b[0], b[1]), b[2]) for b in st['bombs']],
            'explosion_map': expm,
            'coins': [(c[0], c[1]) for c in st['coins'] if c[2]],
            'self': ('me', me['score'], me['bombs_left'],
                     (me['x'], me['y'])),
            'others': others}


def tile_is_free(st, x, y, ignore_agent=None):
    arena = st['arena']
    W, H = arena.shape[0], arena.shape[1]
    if not (0 <= x < W and 0 <= y < H):
        return False
    if arena[x, y] != 0:
        return False
    for b in st['bombs']:
        if b[0] == x and b[1] == y:
            return False
    for i, a in enumerate(st['agents']):
        if not a['alive'] or i == ignore_agent:
            continue
        if a['x'] == x and a['y'] == y:
            return False
    return True


def valid_actions(st, i):
    """Engine-valid actions for agent i (mask lives elsewhere)."""
    a = st['agents'][i]
    out = []
    for act in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        dx, dy = _DELTAS[act]
        if tile_is_free(st, a['x'] + dx, a['y'] + dy, ignore_agent=i):
            out.append(act)
    out.append('WAIT')
    if a['bombs_left']:
        out.append('BOMB')
    return out


def _reveal_p(st):
    """Expected P(hidden coin | cleared crate) — approximation A2."""
    arena = st['arena']
    crates = int((arena == 1).sum())
    if crates <= 0:
        return 0.0
    visible = sum(1 for c in st['coins'] if c[2])
    total_score = sum(a['score'] for a in st['agents'])
    deaths = sum(1 for a in st['agents'] if not a['alive'])
    collected = total_score - REWARD_KILL * deaths - 0
    hidden = st['total_coins'] - visible - collected
    if hidden <= 0:
        return 0.0
    return min(1.0, hidden / crates)


def step(st, actions):
    """One exact engine step. actions aligned with st['agents'].
    Returns info dict with exact payoffs + expected coin reveals."""
    arena = st['arena']
    st['step'] = st.get('step', 0) + 1
    info = {'crates': 0, 'revealed_exp': 0.0, 'coins': [0] * len(actions),
            'kills': [0] * len(actions), 'died': [False] * len(actions),
            'score0': st['agents'][0]['score']}
    # 1. apply actions sequentially (self first: approximation A1)
    order = [0] + [i for i in range(1, len(actions)) if i < len(st['agents'])]
    for i in order:
        a = st['agents'][i]
        if not a['alive']:
            continue
        act = actions[i] if i < len(actions) else 'WAIT'
        if act in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
            dx, dy = _DELTAS[act]
            if tile_is_free(st, a['x'] + dx, a['y'] + dy, ignore_agent=i):
                a['x'] += dx
                a['y'] += dy
        elif act == 'BOMB' and a['bombs_left']:
            st['bombs'].append([a['x'], a['y'], BOMB_TIMER, i])
            a['bombs_left'] = False
        # WAIT / INVALID: stand still (engine-identical)
    # 2. collect coins
    for c in st['coins']:
        if not c[2]:
            continue
        for i, a in enumerate(st['agents']):
            if a['alive'] and a['x'] == c[0] and a['y'] == c[1]:
                c[2] = 0
                a['score'] += REWARD_COIN
                info['coins'][i] += 1
                break
    # 3. progress explosions
    keep = []
    for e in st['explosions']:
        e['timer'] -= 1
        if e['timer'] <= 0:
            e['stage'] += 1
            if e['stage'] == 1:
                o = e['owner']
                if 0 <= o < len(st['agents']):
                    st['agents'][o]['bombs_left'] = True
                e['timer'] = EXPLOSION_TIMER
            elif e['stage'] >= 2:
                continue
        keep.append(e)
    st['explosions'] = keep
    # 4. bombs: detonate at timer<=0 else tick
    p_reveal = _reveal_p(st)
    keep_b = []
    for b in st['bombs']:
        if b[0] is None:
            continue
        if b[2] <= 0:
            owner = b[3]
            coords = blast_coords(arena, b[0], b[1])
            for (cx, cy) in coords:
                if arena[cx, cy] == 1:
                    arena[cx, cy] = 0
                    info['crates'] += 1
                    info['revealed_exp'] += p_reveal
            st['explosions'].append({'coords': coords,
                                     'timer': EXPLOSION_TIMER,
                                     'stage': 0, 'owner': owner})
        else:
            b[2] -= 1
            keep_b.append(b)
    st['bombs'] = keep_b
    # 5. evaluate stage-0 explosions (set semantics: collect then remove)
    hit = set()
    for e in st['explosions']:
        if e['stage'] != 0:
            continue
        for (cx, cy) in e['coords']:
            for i, a in enumerate(st['agents']):
                if a['alive'] and a['x'] == cx and a['y'] == cy:
                    hit.add(i)
    for i in sorted(hit):
        st['agents'][i]['alive'] = False
        info['died'][i] = True
    for i in sorted(hit):
        # credit each lethal explosion owner once per victim (engine pays
        # +5 per (explosion, victim); overlapping blasts each pay)
        seen = set()
        for e in st['explosions']:
            if e['stage'] != 0:
                continue
            if (st['agents'][i]['x'], st['agents'][i]['y']) in e['coords']:
                o = e['owner']
                if o != i and (o, i) not in seen and o >= 0:
                    seen.add((o, i))
                    if o < len(st['agents']):
                        st['agents'][o]['score'] += REWARD_KILL
                        info['kills'][o] += 1
    info['score1'] = st['agents'][0]['score']
    return info


def margin(st):
    """Own score minus best opponent (dead opponents keep frozen scores,
    exactly as the tournament totals them)."""
    me = st['agents'][0]['score']
    opps = [a['score'] for a in st['agents'][1:]]
    return me - (max(opps) if opps else 0)


def yield_field(arena):
    """Exact crates-per-bomb for EVERY tile (vectorized, 0.13 ms)."""
    from .features import _blast_crate_counts
    _, blast = _blast_crate_counts(np.asarray(arena))
    return blast
