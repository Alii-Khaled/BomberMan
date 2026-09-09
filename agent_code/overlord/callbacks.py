"""Overlord: CNN ceiling agent + exact safety mask + aggressive late-game hunting.

Same safety core as sentinel, stronger spatial representation. Always queries
the learned CNN; heuristic prior is smaller so learned spatial policy dominates.
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
from .features_cnn import state_to_tensor
from .model import build_model, scalars_from_state, ACTION_LIST

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
Q_WEIGHT = 1.0


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Track-1 frozen grid knobs (E31): env overrides, defaults = shipped values.
# With env unset the policy is behavior-identical to the 3.79 ship.
G_WAIT = _env_float('OVERLORD_G_WAIT', 0.30)
G_HUNT_BASE = _env_float('OVERLORD_G_HUNT_BASE', 0.5)
G_BOMB_OPP = _env_float('OVERLORD_G_BOMB_OPP', 1.1)
G_BOMB_CRATE = _env_float('OVERLORD_G_BOMB_CRATE', 0.5)
G_CORRIDOR = _env_float('OVERLORD_G_CORRIDOR', 1.5)
G_REVISIT3 = _env_float('OVERLORD_G_REVISIT3', 0.45)
G_REVISIT2 = _env_float('OVERLORD_G_REVISIT2', 0.15)
G_FLEE_BOOST = _env_float('OVERLORD_G_FLEE_BOOST', 2.0)
G_BOMB_REPEAT = _env_float('OVERLORD_G_BOMB_REPEAT', 0.9)
G_COIN = _env_float('OVERLORD_G_COIN', 0.45)
G_LATE = _env_float('OVERLORD_G_LATE', 0.4)
# E45 additive openness bonus (user hypothesis: open 4-neighbourhood
# maximizes blast tiles + shortens escapes). Default 0.0 = ship behavior
# (running training never sets it, so live runs are unaffected).
G_OPENNESS = _env_float('OVERLORD_G_OPENNESS', 0.0)
# E39 late-hunt veto (E16 leg never live-tested): 1 = refuse BOMB with no
# crates but opponents in blast after step 250 (P(kill) ~0.007 there).
# Default 0 = ship behavior.
G_LATEHUNT_VETO = _env_float('OVERLORD_G_LATEHUNT_VETO', 0)


def _latehunt_vetoed(game_state, safety):
    """True when the E39 veto forbids BOMB this step."""
    if not G_LATEHUNT_VETO:
        return False
    try:
        return (int(game_state.get('step', 0)) > 250
                and safety.get('crates_hit_if_bomb', 0) == 0
                and safety.get('opps_hit_if_bomb', 0) >= 1)
    except Exception:
        return False


def _flee_scores(game_state):
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
        threatened = True
        urgency = 1.2 + (4 - min(max(t, 0), 4)) * 0.7
        for a, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            in_blast_now = (nx, ny) in blast
            od = abs(x - xb) + abs(y - yb)
            nd = abs(nx - xb) + abs(ny - yb)
            if in_blast_now:
                if nd <= od:
                    scores[a] -= 2.0 * urgency
                else:
                    scores[a] -= 0.6 * urgency
            else:
                if nd > od:
                    scores[a] += 1.0 * urgency
                elif nd < od:
                    scores[a] -= 0.8 * urgency
        if (x, y) == (xb, yb):
            scores['WAIT'] -= 2.5
            scores['BOMB'] -= 2.5
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


def _heuristic(game_state, safety):
    scores = {a: 0.0 for a in ACTION_LIST}
    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    coins = game_state.get('coins', []) or []
    others = game_state.get('others', []) or []
    others_xy = [(int(xy[0]), int(xy[1])) for (n, s, b, xy) in others]
    step = int(game_state.get('step', 0))
    W, H = arena.shape[0], arena.shape[1]
    moves = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
    # coin pull
    if coins:
        cx, cy = min(((int(a), int(b)) for (a, b) in coins),
                     key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        for a, (dx, dy) in moves.items():
            nd = abs(cx - (x + dx)) + abs(cy - (y + dy))
            od = abs(cx - x) + abs(cy - y)
            if nd < od:
                scores[a] += G_COIN
            elif nd > od:
                scores[a] -= 0.2
    # hunt: stronger than sentinel, esp. late / 1v1
    crates_left = int((arena == 1).sum())
    hunt_w = G_HUNT_BASE + 0.8 * min(1.0, step / 200.0)
    if crates_left + len(coins) <= 4 or step > 250:
        hunt_w += G_LATE
    if others_xy:
        tgt = min(others_xy, key=lambda o: abs(o[0] - x) + abs(o[1] - y))
        for a, (dx, dy) in moves.items():
            nd = abs(tgt[0] - (x + dx)) + abs(tgt[1] - (y + dy))
            od = abs(tgt[0] - x) + abs(tgt[1] - y)
            if nd < od:
                scores[a] += 0.5 * hunt_w
            elif nd > od:
                scores[a] -= 0.2 * hunt_w
    # center control late (cut off escapes)
    if step > 280 and others_xy:
        for a, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            cd_new = abs(nx - 8) + abs(ny - 8)
            cd_old = abs(x - 8) + abs(y - 8)
            if cd_new < cd_old:
                scores[a] += 0.12
    # bomb (rule-based discipline: safe pocket / dead-end / kill only)
    crate_near = sum(1 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if 0 <= x + dx < W and 0 <= y + dy < H and arena[x + dx, y + dy] == 1)
    if safety.get('can_escape_if_bomb') and bombs_left:
        b = 0.0
        if safety.get('opps_hit_if_bomb', 0):
            b += G_BOMB_OPP * min(hunt_w, 1.5)
        elif crate_near:
            free_nb = sum(1 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                          if 0 <= x + dx < W and 0 <= y + dy < H and arena[x + dx, y + dy] == 0)
            dist_hyp = float(safety.get('dist_hyp', 9))
            if free_nb == 1:
                b += G_BOMB_CRATE + 0.12 * min(crate_near, 3) + 0.2
            elif dist_hyp <= 2:
                b += G_BOMB_CRATE + 0.12 * min(crate_near, 3)
            else:
                b -= 1.0
        # E45 additive openness: bomb score += G_OPENNESS per open neighbour
        # (blast coverage + short escapes). Pure addition — dead-end bonus
        # above untouched; default 0.0 reproduces ship exactly.
        try:
            _open_nb = sum(1 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if 0 <= x + dx < W and 0 <= y + dy < H and arena[x + dx, y + dy] == 0)
            b += G_OPENNESS * _open_nb
        except Exception:
            pass
        scores['BOMB'] += b
    else:
        scores['BOMB'] -= 0.7
    scores['WAIT'] -= G_WAIT  # E20-port: -0.12 let WAIT win over flat moves
    return scores


def setup(self):
    self.logger.debug('overlord setup')
    try:
        import torch as _t
        _t.set_num_threads(1)
        _t.set_num_interop_threads(1)
    except Exception:
        pass
    np.random.seed()
    self.model = build_model()
    here = os.path.dirname(os.path.abspath(__file__))
    loaded = False
    for cand in (os.path.join(here, 'my-saved-model.pt'),
                 os.path.join(os.getcwd(), 'my-saved-model.pt')):
        if os.path.isfile(cand):
            try:
                import torch as _t
                obj = _t.load(cand, map_location='cpu', weights_only=False)
                sd = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
                # support both raw and traced? traced can't load state_dict; try raw first
                try:
                    self.model.load_state_dict(sd, strict=False)
                    loaded = True
                except Exception:
                    # maybe full model saved
                    pass
                self.logger.info(f'overlord loaded {cand}')
                break
            except Exception as ex:
                self.logger.warning(f'overlord load failed {cand}: {ex}')
    try:
        self.model.eval()
    except Exception:
        pass
    # optional JIT fast path (eval-only: during training use the live model,
    # which syncs from q_net — a setup-time trace would act stale forever)
    self.fast = None
    try:
        import torch as _t
        if loaded and not getattr(self, 'train', False):
            with _t.no_grad():
                dummy_img = _t.zeros(1, 12, 17, 17)
                dummy_sc = _t.zeros(1, 8)
                self.fast = _t.jit.trace(self.model, (dummy_img, dummy_sc))
                self.fast.eval()
    except Exception as ex:
        self.logger.debug(f'overlord trace skipped: {ex}')
        self.fast = None
    self.bomb_history = deque([], 5)
    self.coord_history = deque([], 24)
    self.current_round = 0
    self.flee_timer = 0


def act(self, game_state):
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
    safety = action_safety(game_state)
    valid, safe = safety['valid'], safety['safe']
    flee_locked = getattr(self, 'flee_timer', 0) > 0
    if flee_locked:
        self.flee_timer = max(0, self.flee_timer - 1)
    nxt = {'UP': (x, y - 1), 'DOWN': (x, y + 1), 'LEFT': (x - 1, y),
           'RIGHT': (x + 1, y), 'WAIT': (x, y), 'BOMB': (x, y)}
    heu = _heuristic(game_state, safety)
    for a, c in nxt.items():
        cnt = list(self.coord_history).count(c)
        if cnt >= 3:
            heu[a] -= G_REVISIT3
        elif cnt == 2:
            heu[a] -= G_REVISIT2
    if (x, y) in list(self.bomb_history)[-3:]:
        heu['BOMB'] -= G_BOMB_REPEAT
    try:
        flee = _flee_scores(game_state)
        boost = G_FLEE_BOOST if flee_locked else 1.0
        if flee_locked:
            heu['BOMB'] -= 3.0
            for a in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
                if flee.get(a, 0.0) <= 0:
                    heu[a] -= 0.5
        for a in ACTION_LIST:
            heu[a] += boost * flee[a]
    except Exception:
        pass
    try:
        W, H = arena.shape[0], arena.shape[1]
        up_b = not (0 <= y - 1 < H and arena[x, y - 1] == 0)
        dn_b = not (0 <= y + 1 < H and arena[x, y + 1] == 0)
        lf_b = not (0 <= x - 1 < W and arena[x - 1, y] == 0)
        rt_b = not (0 <= x + 1 < W and arena[x + 1, y] == 0)
        if ((up_b and dn_b) or (lf_b and rt_b)) and not flee_locked:
            if safety.get('opps_hit_if_bomb', 0) == 0 and safety.get('crates_hit_if_bomb', 0) < 2:
                heu['BOMB'] -= G_CORRIDOR
    except Exception:
        pass

    q = np.zeros(len(ACTION_LIST))
    try:
        if _HAS_TORCH and getattr(self, 'model', None) is not None:
            import torch as _t
            img = state_to_tensor(game_state, safety)
            sc = scalars_from_state(game_state, safety)
            with _t.no_grad():
                ti = _t.from_numpy(img).unsqueeze(0)
                ts = _t.from_numpy(sc).unsqueeze(0)
                mdl = getattr(self, 'fast', None) or self.model
                out = mdl(ti, ts)
                qv = out[0] if isinstance(out, (tuple, list)) else out
                q = np.clip(np.asarray(qv.squeeze(0).cpu().numpy(), dtype=np.float64), -4, 4)
    except Exception:
        pass
    heu_arr = np.array([heu[a] for a in ACTION_LIST], dtype=np.float64)
    total = heu_arr + Q_WEIGHT * q
    # Epsilon-greedy exploration during training only (sentinel-proven):
    # random choice respects the safety mask: safe+valid > valid > WAIT.
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
    order = sorted(range(len(ACTION_LIST)), key=lambda i: total[i], reverse=True)
    # killer exception: allow stepping toward kill even if marginally unsafe? No:
    # keep strict safety, but prefer BOMB when opps_hit and escape exists (already safe-flagged)
    _veto = _latehunt_vetoed(game_state, safety)
    for i in order:
        a = ACTION_LIST[i]
        if a == 'BOMB' and _veto:
            continue
        if valid.get(a) and safe.get(a):
            if a == 'BOMB':
                self.bomb_history.append((x, y))
                self.flee_timer = 5
            else:
                self.coord_history.append(nxt[a])
            return a
    for i in order:
        a = ACTION_LIST[i]
        if a == 'BOMB' and _veto:
            continue
        if valid.get(a):
            if a == 'BOMB':
                self.bomb_history.append((x, y))
            else:
                self.coord_history.append(nxt[a])
            return a
    self.coord_history.append((x, y))
    return 'WAIT'
