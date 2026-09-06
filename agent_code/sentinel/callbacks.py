"""Sentinel: reliable Dueling-MLP + exact safety mask + heuristic blend.

ML-compliant: act() always queries the learned DuelingMLP; heuristic is a
small additive prior so the agent is strong even before convergence and the
learned Q dominates after training.
"""
from collections import deque
import os
import random
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from .safety import action_safety, true_blast
from .features_mlp import state_to_features
from .model import build_model, ACTION_LIST


def _flee_scores(game_state):
    """Strong escape override: commit away from live bombs (wall-aware blast).

    Returns dict action->bonus. Called last so it dominates coin/hunt when
    threatened. Fixes 'flee then wander back' suicides.
    """
    scores = {a: 0.0 for a in ACTION_LIST}
    arena = game_state['field']
    _, _, _, (x, y) = game_state['self']
    bombs = game_state.get('bombs', []) or []
    if not bombs:
        return scores
    moves = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
    threatened = False
    for (xy, t) in bombs:
        try:
            t = int(t)
        except Exception:
            t = 4
        if t > 4:
            continue
        xb, yb = int(xy[0]), int(xy[1])
        blast = set(true_blast(arena, xb, yb, 3))
        if (x, y) not in blast and not ((xb == x or yb == y) and abs(xb - x) + abs(yb - y) <= 4):
            continue
        # this bomb threatens us (or nearby); urgency grows as timer shrinks
        threatened = True
        urgency = 1.2 + (4 - min(max(t, 0), 4)) * 0.7  # 1.2..4.0
        for a, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            in_blast_now = (nx, ny) in blast
            # distance from bomb center
            od = abs(x - xb) + abs(y - yb)
            nd = abs(nx - xb) + abs(ny - yb)
            if in_blast_now:
                # stepping (or staying) inside blast: heavy penalty unless leaving next step
                # allow only if it increases distance (fleeing through edge)
                if nd <= od:
                    scores[a] -= 2.0 * urgency
                else:
                    scores[a] -= 0.6 * urgency
            else:
                if nd > od:
                    scores[a] += 1.0 * urgency
                elif nd < od:
                    scores[a] -= 0.8 * urgency
        # on top of own bomb: must leave immediately, penalize WAIT/BOMB extra
        if (x, y) == (xb, yb):
            scores['WAIT'] -= 2.5
            scores['BOMB'] -= 2.5
    # corner bonus: prefer moves that leave the row/col entirely (turn a corner)
    if threatened:
        for (xy, t) in bombs:
            xb, yb = int(xy[0]), int(xy[1])
            if xb == x and yb != y:
                scores['LEFT'] += 0.5
                scores['RIGHT'] += 0.5
            if yb == y and xb != x:
                scores['UP'] += 0.5
                scores['DOWN'] += 0.5
    return scores

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
# Heuristic dominates before training (Q random ~ +-0.5); after training Q grows
# to +-5..10 and dominates. 0.2 keeps early play strong yet ML-compliant.
Q_WEIGHT = 0.2


def _heuristic_scores(game_state, safety):
    scores = {a: 0.0 for a in ACTION_LIST}
    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    coins = game_state.get('coins', []) or []
    others = game_state.get('others', []) or []
    others_xy = [(int(xy[0]), int(xy[1])) for (n, s, b, xy) in others]
    step = int(game_state.get('step', 0))
    W, H = arena.shape[0], arena.shape[1]

    # direction hints via BFS first-step (reuse features' BFS? cheap recompute via dist)
    # Simple Manhattan + wall-aware preference: score moves reducing distance
    def manhattan(ax, ay, targets):
        if not targets:
            return None, None
        best, bd = None, 1e9
        for (tx, ty) in targets:
            d = abs(int(tx) - ax) + abs(int(ty) - ay)
            if d < bd:
                bd, best = d, (int(tx), int(ty))
        return best, bd

    # coin seeking
    if coins:
        bt, _ = manhattan(x, y, [(int(a), int(b)) for (a, b) in coins])
        if bt is not None:
            for a, (dx, dy) in {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}.items():
                nx, ny = x + dx, y + dy
                nd = abs(bt[0] - nx) + abs(bt[1] - ny)
                od = abs(bt[0] - x) + abs(bt[1] - y)
                if nd < od:
                    scores[a] += 0.5
                elif nd > od:
                    scores[a] -= 0.25
    # crate adjacency: prefer staying/bombing near crates early
    crate_near = sum(1 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if 0 <= x + dx < W and 0 <= y + dy < H and arena[x + dx, y + dy] == 1)
    # opponent hunting weight grows over time / when crates+coins low
    crates_left = int((arena == 1).sum())
    coins_left = len(coins)
    hunt_w = 0.5 + 0.8 * min(1.0, step / 200.0)
    if crates_left + coins_left <= 4 or step > 250:
        hunt_w += 0.4
    if others_xy:
        # nearest opp
        ops = sorted(others_xy, key=lambda o: abs(o[0] - x) + abs(o[1] - y))
        tgt = ops[0]
        for a, (dx, dy) in {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}.items():
            nx, ny = x + dx, y + dy
            nd = abs(tgt[0] - nx) + abs(tgt[1] - ny)
            od = abs(tgt[0] - x) + abs(tgt[1] - y)
            if nd < od:
                scores[a] += 0.5 * hunt_w
            elif nd > od:
                scores[a] -= 0.2 * hunt_w
    # center control late (cut off escapes)
    if step > 280 and others_xy:
        for a, (dx, dy) in {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}.items():
            nx, ny = x + dx, y + dy
            if abs(nx - 8) + abs(ny - 8) < abs(x - 8) + abs(y - 8):
                scores[a] += 0.12
    # bomb logic (rule-based discipline: dead-end / adjacent kill / safe pocket only)
    if safety['can_escape_if_bomb'] and bombs_left:
        b = 0.0
        # adjacent kill always worth it (can_escape already ensures survival)
        if safety.get('opps_hit_if_bomb', 0) > 0:
            b += 1.1 * min(hunt_w, 1.5)
        elif crate_near > 0:
            # crate-only bombs: require safe pocket (fast escape) or dead-end.
            # Transit corridor bombs (dist_hyp>=2) are the #1 suicide cause.
            free_nb = sum(1 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                          if 0 <= x + dx < W and 0 <= y + dy < H and arena[x + dx, y + dy] == 0)
            dist_hyp = float(safety.get('dist_hyp', 9))
            is_dead_end = (free_nb == 1)
            if is_dead_end:
                b += 0.5 + 0.12 * min(crate_near, 3) + 0.2
            elif dist_hyp <= 2:
                b += 0.5 + 0.12 * min(crate_near, 3)
            else:
                b -= 1.0  # transit: do not bomb, keep moving to pocket
        # don't bomb every step on same tile (handled via history penalty in act)
        scores['BOMB'] += b
    else:
        scores['BOMB'] -= 0.7
    # WAIT penalty (avoid camping) unless no safe move.
    # E18: -0.12 let WAIT win strictly over flat early moves (30% early WAIT,
    # 321 safe-stuck sits/30rd with zero bombs ticking). Mask still picks WAIT
    # when it is the only safe move, so survival is unaffected.
    scores['WAIT'] -= 0.30
    # light dead-end avoidance: only punish true cul-de-sac
    for a, (dx, dy) in {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and arena[nx, ny] == 0:
            nb = sum(1 for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if 0 <= nx + ddx < W and 0 <= ny + ddy < H and arena[nx + ddx, ny + ddy] == 0)
            if nb == 0:
                # E18: early dead-ends hold crates — full avoidance strands
                # the agent at spawn (displacement 3.2 tiles by step 50).
                scores[a] -= 0.10 if step < 150 else 0.35  # cul-de-sac
    return scores


def setup(self):
    self.logger.debug('sentinel setup')
    try:
        import torch as _t
        _t.set_num_threads(1)
        _t.set_num_interop_threads(1)
    except Exception:
        pass
    np.random.seed()
    self.model = build_model()
    # load weights relative to this file (tournament-safe)
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, 'my-saved-model.pt'),
                 os.path.join(os.getcwd(), 'my-saved-model.pt')):
        if os.path.isfile(cand):
            try:
                import torch as _t
                sd = _t.load(cand, map_location='cpu', weights_only=False)
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                self.model.load_state_dict(sd, strict=False)
                self.logger.info(f'sentinel loaded {cand}')
                break
            except Exception as ex:
                self.logger.warning(f'sentinel load failed {cand}: {ex}')
    try:
        self.model.eval()
    except Exception:
        pass
    self.bomb_history = deque([], 5)
    self.coord_history = deque([], 24)
    self.current_round = 0
    self._rng = np.random.RandomState()
    self.flee_timer = 0
    # Exploration rate (overwritten by train.setup_training when train=True).
    # Tournament (train=False) stays greedy.
    self.epsilon = 0.0


def act(self, game_state):
    # new round reset
    try:
        rnd = int(game_state.get('round', 0))
    except Exception:
        rnd = 0
    if rnd != getattr(self, 'current_round', 0):
        self.bomb_history = deque([], 5)
        self.coord_history = deque([], 24)
        self.current_round = rnd
        self.flee_timer = 0

    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state.get('bombs', []) or []
    safety = action_safety(game_state)
    valid, safe = safety['valid'], safety['safe']
    # track own-bomb flee commitment: bombs list lacks owner, so use
    # heuristic — if we bombed recently and any bomb is within 6 steps of
    # our history, stay in flee mode. Decrement each step.
    flee_locked = getattr(self, 'flee_timer', 0) > 0
    if flee_locked:
        self.flee_timer = max(0, self.flee_timer - 1)

    # loop detection: penalize revisits
    loop_pen = {}
    for a in ACTION_LIST:
        loop_pen[a] = 0.0
    nxt = {'UP': (x, y - 1), 'DOWN': (x, y + 1), 'LEFT': (x - 1, y),
           'RIGHT': (x + 1, y), 'WAIT': (x, y), 'BOMB': (x, y)}
    for a, c in nxt.items():
        cnt = list(self.coord_history).count(c)
        if cnt >= 3:
            loop_pen[a] -= 0.4
        elif cnt == 2:
            loop_pen[a] -= 0.15
    if (x, y) in list(self.bomb_history)[-3:]:
        loop_pen['BOMB'] -= 0.8

    heu = _heuristic_scores(game_state, safety)
    for a in ACTION_LIST:
        heu[a] += loop_pen[a]
    # flee override dominates when threatened
    try:
        flee = _flee_scores(game_state)
        flee_boost = 1.0
        if flee_locked:
            # commit to escape: double flee, zero greed, forbid new bombs
            flee_boost = 2.0
            heu['BOMB'] -= 3.0
            # suppress coin/hunt pull by damping non-flee moves (flee already added)
            for a in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
                if flee.get(a, 0.0) <= 0:
                    heu[a] -= 0.5
        for a in ACTION_LIST:
            heu[a] += flee_boost * flee[a]
    except Exception:
        pass
    # corridor discipline: bombing in a 1-wide corridor (vertical walls both
    # sides) needs 4-step outrun; forbid unless killing or crates>=2 with clear runway.
    try:
        W, H = arena.shape[0], arena.shape[1]
        up_blocked = not (0 <= y - 1 < H and arena[x, y - 1] == 0)
        dn_blocked = not (0 <= y + 1 < H and arena[x, y + 1] == 0)
        lf_blocked = not (0 <= x - 1 < W and arena[x - 1, y] == 0)
        rt_blocked = not (0 <= x + 1 < W and arena[x + 1, y] == 0)
        in_corridor = (up_blocked and dn_blocked) or (lf_blocked and rt_blocked)
        if in_corridor and not flee_locked:
            if safety.get('opps_hit_if_bomb', 0) == 0:
                # need at least 2 crates to justify corridor risk, else forbid
                if safety.get('crates_hit_if_bomb', 0) < 2:
                    heu['BOMB'] -= 1.5
    except Exception:
        pass

    # learned Q
    q = np.zeros(len(ACTION_LIST), dtype=np.float64)
    try:
        if _HAS_TORCH and getattr(self, 'model', None) is not None:
            import torch as _t
            f = state_to_features(game_state, safety)
            with _t.no_grad():
                t = _t.from_numpy(f).unsqueeze(0)
                qv = self.model(t).squeeze(0).cpu().numpy()
            # clip crazy inits
            q = np.clip(np.asarray(qv, dtype=np.float64), -3, 3)
    except Exception:
        q = np.zeros(len(ACTION_LIST))

    heu_arr = np.array([heu[a] for a in ACTION_LIST], dtype=np.float64)
    total = heu_arr + Q_WEIGHT * q

    # Epsilon-greedy exploration during training only (grade: exploration proof).
    # Random choice respects the safety mask: safe+valid > valid > WAIT.
    try:
        if getattr(self, 'train', False):
            eps = float(getattr(self, 'epsilon', 0.0) or 0.0)
            if eps > 0 and random.random() < eps:
                safe_valid = [a for a in ACTION_LIST if valid.get(a) and safe.get(a)]
                pool = safe_valid or [a for a in ACTION_LIST if valid.get(a)] or ['WAIT']
                a = random.choice(pool)
                if a == 'BOMB':
                    self.bomb_history.append((x, y))
                    self.flee_timer = 5
                else:
                    self.coord_history.append(nxt[a])
                return a
    except Exception:
        pass

    # mask: prefer safe&valid; allow unsafe only if nothing safe
    order = sorted(range(len(ACTION_LIST)), key=lambda i: total[i], reverse=True)
    for i in order:
        a = ACTION_LIST[i]
        if valid.get(a, False) and safe.get(a, False):
            if a == 'BOMB':
                self.bomb_history.append((x, y))
                self.flee_timer = 5  # commit next 5 steps to escape
            if a in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
                self.coord_history.append(nxt[a])
            return a
    # fallback: any valid (even unsafe) to avoid INVALID; prefer non-BOMB
    for i in order:
        a = ACTION_LIST[i]
        if valid.get(a, False):
            if a == 'BOMB':
                self.bomb_history.append((x, y))
            else:
                self.coord_history.append(nxt[a])
            return a
    # last resort
    self.coord_history.append((x, y))
    return 'WAIT'
