"""Apex: synthesis agent — warden target priority + BFS directions,
overlord-grade safety mask, CNN Q (16-scalar head) at full authority.

Decision skeleton follows warden (proven 5.35 system): strict tiered bomb
discipline, safe-first move scoring with danger gradient, deterministic
play. Learned Q (Q_WEIGHT=1.0) modulates on top; safety mask (overlord
core: margin gate, corridor rule) constrains selection. Heuristic drives
early play (dueling heads zero-init); Q grows via BC + RL.
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

from .safety import action_safety
from .features_cnn import state_to_tensor
from .features_extra import extra_scalars
from .model import build_model, ACTION_LIST

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Frozen-probe + training-arm knobs (defaults = shipped A2 behavior).
# APEX_Q_WEIGHT: 1.0 ship; 0.0 = heuristic-only Q0 ablation (E36 pattern).
# APEX_RELAX_TIER: 0 = strict tiers (ship); 1 = allow crates==1 & dist<=3
#   (single approved relaxed arm — intent-volume A/B, frozen-gated).
# APEX_WAIT: additive WAIT penalty (default -0.30 = E20 value).
# APEX_MASK_ALWAYS: 1 = every move must be mask-safe (ship); 0 = mask
#   filters only when must_flee (warden semantics, E62/S2 sweep).
# APEX_Q_CLIP: abs bound on the learned Q before it is added to the
#   heuristic total (default 4.0; S2 arm tests 0.5 — E61 Q-delta −0.42).
Q_WEIGHT = _env_float('APEX_Q_WEIGHT', 1.0)
APEX_RELAX_TIER = os.environ.get('APEX_RELAX_TIER', '0') == '1'
APEX_WAIT = _env_float('APEX_WAIT', 0.30)
APEX_MASK_ALWAYS = os.environ.get('APEX_MASK_ALWAYS', '1') == '1'
Q_CLIP = _env_float('APEX_Q_CLIP', 4.0)

_DELTAS = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0),
           'RIGHT': (1, 0), 'WAIT': (0, 0)}
_STEP_TO_ACTION = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT',
                   (1, 0): 'RIGHT', (0, 0): 'WAIT'}


def _bfs_first_steps(start, arena, bomb_cells):
    """BFS over free tiles. Returns (dist, first_step) maps."""
    w, h = arena.shape[0], arena.shape[1]
    dist = np.full((w, h), 10 ** 9, dtype=np.int32)
    first = {}
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < w and 0 <= sy < h):
        return dist, first
    dist[sx, sy] = 0
    queue = deque([(sx, sy)])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if dist[nx, ny] != 10 ** 9:
                continue
            if arena[nx, ny] != 0 or (nx, ny) in bomb_cells:
                continue
            dist[nx, ny] = dist[cx, cy] + 1
            first[(nx, ny)] = first.get((cx, cy), (dx, dy))
            queue.append((nx, ny))
    return dist, first


def _dist_to(dist, first, targets):
    """Shortest BFS distance and first step toward any target/neighbour."""
    best_d, best_step = 10 ** 9, None
    for tx, ty in targets:
        tx, ty = int(tx), int(ty)
        if not (0 <= tx < dist.shape[0] and 0 <= ty < dist.shape[1]):
            continue
        if dist[tx, ty] < 10 ** 9:
            if dist[tx, ty] == 0:
                return 0, (0, 0)
            if dist[tx, ty] < best_d:
                best_d, best_step = int(dist[tx, ty]), first.get((tx, ty))
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dist.shape[0] and 0 <= ny < dist.shape[1]:
                if dist[nx, ny] < 10 ** 9 and int(dist[nx, ny]) + 1 < best_d:
                    best_d = int(dist[nx, ny]) + 1
                    best_step = first.get((nx, ny))
    return best_d, best_step


def setup(self):
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
                sd = obj.get('q_net', obj) if isinstance(obj, dict) else obj
                # full payloads (q_net) and raw state dicts both accepted;
                # partial transfer (trunk-only) loads strict=False.
                self.model.load_state_dict(sd, strict=False)
                loaded = True
                self.logger.info(f'apex loaded {cand}')
                break
            except Exception as ex:
                self.logger.warning(f'apex load failed {cand}: {ex}')
    try:
        self.model.eval()
    except Exception:
        pass
    self.fast = None
    try:
        import torch as _t
        if loaded and not getattr(self, 'train', False):
            with _t.no_grad():
                dummy_img = _t.zeros(1, 12, 17, 17)
                dummy_sc = _t.zeros(1, 16)
                self.fast = _t.jit.trace(self.model, (dummy_img, dummy_sc))
                self.fast.eval()
    except Exception as ex:
        self.logger.debug(f'apex trace skipped: {ex}')
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
    arena = np.asarray(game_state['field'])
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state.get('bombs', []) or []
    coins = [(int(a), int(b)) for (a, b) in (game_state.get('coins', []) or [])]
    others = [(int(p[0]), int(p[1])) for (_, _, _, p) in (game_state.get('others', []) or [])]
    step = int(game_state.get('step', 0))
    explosion_map = game_state.get('explosion_map', None)

    safety = action_safety(game_state)
    valid, safe = safety['valid'], safety['safe']
    flee_locked = getattr(self, 'flee_timer', 0) > 0
    if flee_locked:
        self.flee_timer = max(0, self.flee_timer - 1)

    bomb_cells = set((int(bxy[0]), int(bxy[1])) for (bxy, _) in bombs)
    other_cells = set(others)
    # danger timeline from the mask's array: earliest lethal step per tile
    danger = safety.get('danger', None)
    lethal = {}
    try:
        d = np.asarray(danger)
        for t in range(d.shape[0]):
            for cx, cy in zip(*np.where(d[t])):
                if (int(cx), int(cy)) not in lethal:
                    lethal[(int(cx), int(cy))] = int(t)
    except Exception:
        pass
    must_flee = lethal.get((x, y), float('inf')) <= 1

    # hypothetical own bomb (mask verdicts, E27-validated)
    can_escape_bomb = bool(safety.get('can_escape_if_bomb', False))
    dist_hyp = float(safety.get('dist_hyp', 9))
    opps_hit = int(safety.get('opps_hit_if_bomb', 0))
    crates_hit = int(safety.get('crates_hit_if_bomb', 0))

    # --- targets (warden priority: hunt -> coins -> crate-adj, BFS) ---
    crates = [(cx, cy) for cx in range(arena.shape[0])
              for cy in range(arena.shape[1]) if arena[cx, cy] == 1]
    crate_adj = set()
    for cx, cy in crates:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] \
                    and arena[nx, ny] == 0:
                crate_adj.add((nx, ny))
    dist, first = _bfs_first_steps((x, y), arena, bomb_cells)
    _, coin_step = _dist_to(dist, first, coins)
    _, crate_step = _dist_to(dist, first, sorted(crate_adj) or crates)
    _, opp_step = _dist_to(dist, first, others)
    hunt = bool(others) and (len(crates) + len(coins) <= 6 or step > 200 or
                             min(abs(ox - x) + abs(oy - y) for ox, oy in others) <= 3)

    # --- bomb decision (strict tiers, training-time mask) ---
    # APEX_RELAX_TIER=1 (approved single arm): crates==1 & dist<=3 allowed.
    # Default 0 reproduces A2 ship exactly.
    want_bomb = False
    if valid.get('BOMB') and can_escape_bomb and not must_flee \
            and (x, y) not in list(self.bomb_history)[-3:]:
        if opps_hit > 0:
            want_bomb = True
        elif crates_hit >= 2 and dist_hyp <= 3:
            want_bomb = True
        elif crates_hit == 1 and dist_hyp <= (3 if APEX_RELAX_TIER else 2):
            want_bomb = True

    # --- move scoring (warden system: safe + preferred + danger + loop) ---
    preferred = []
    if hunt and opp_step in _STEP_TO_ACTION:
        preferred.append(_STEP_TO_ACTION[opp_step])
    if coins and coin_step in _STEP_TO_ACTION:
        preferred.append(_STEP_TO_ACTION[coin_step])
    if crate_adj and crate_step in _STEP_TO_ACTION:
        preferred.append(_STEP_TO_ACTION[crate_step])
    elif crates and crate_step in _STEP_TO_ACTION:
        preferred.append(_STEP_TO_ACTION[crate_step])

    def loop_penalty(action):
        dx, dy = _DELTAS[action]
        tile = (x + dx, y + dy)
        count = list(self.coord_history).count(tile)
        return -0.6 if count >= 3 else (-0.2 if count == 2 else 0.0)

    def danger_cost(action):
        dx, dy = _DELTAS[action]
        tile = (x + dx, y + dy)
        let = lethal.get(tile, float('inf'))
        if let <= 1:
            return -100.0
        if let <= 3:
            return -2.0
        return 0.0

    scores = {}
    for action in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
        if not valid.get(action):
            continue
        sc = 0.0
        try:
            if action in _safe_moves(safety, valid):
                sc += 3.0
        except Exception:
            pass
        if preferred and action == preferred[0]:
            sc += 1.5
        elif action in preferred:
            sc += 0.7
        try:
            sc += danger_cost(action)
        except Exception:
            pass
        try:
            sc += loop_penalty(action)
        except Exception:
            pass
        if must_flee and action == 'WAIT':
            sc -= 5.0
        scores[action] = sc
    if must_flee:
        safec = {a: sc for a, sc in scores.items()
                 if a in _safe_moves(safety, valid)}
        if safec:
            scores = safec
    scores['WAIT'] = scores.get('WAIT', 0.0) - APEX_WAIT  # activity (E20 value 0.30)

    # --- learned Q (16-scalar net, full authority) ---
    q = np.zeros(len(ACTION_LIST))
    try:
        if _HAS_TORCH and getattr(self, 'model', None) is not None:
            import torch as _t
            img = state_to_tensor(game_state, safety)
            sc8 = scalars_from_state(game_state, safety)
            ex8 = extra_scalars(game_state)
            sc = np.concatenate([sc8, ex8]).astype(np.float32)
            with _t.no_grad():
                ti = _t.from_numpy(img).unsqueeze(0)
                ts = _t.from_numpy(sc).unsqueeze(0)
                mdl = getattr(self, 'fast', None) or self.model
                out = mdl(ti, ts)
                qv = out[0] if isinstance(out, (tuple, list)) else out
                q = np.clip(np.asarray(qv.squeeze(0).cpu().numpy(),
                                       dtype=np.float64), -Q_CLIP, Q_CLIP)
    except Exception:
        pass
    _order = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}
    total = {a: scores.get(a, 0.0) + Q_WEIGHT * float(q[_order[a]])
             for a in ACTION_LIST}
    nxt = {'UP': (x, y - 1), 'DOWN': (x, y + 1), 'LEFT': (x - 1, y),
           'RIGHT': (x + 1, y), 'WAIT': (x, y), 'BOMB': (x, y)}

    # epsilon-greedy exploration during training (safety-masked pool)
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
    if want_bomb and valid.get('BOMB') and safe.get('BOMB'):
        self.bomb_history.append((x, y))
        self.flee_timer = 5
        return 'BOMB'
    order = sorted([a for a in ACTION_LIST if a != 'BOMB'],
                   key=lambda a: total[a], reverse=True)
    # E62/S2: mask filters every move by default (ship behavior);
    # APEX_MASK_ALWAYS=0 restores warden semantics (mask binds only when
    # must_flee). The valid-only fallback is unchanged in both modes.
    if APEX_MASK_ALWAYS or must_flee:
        for a in order:
            if valid.get(a) and safe.get(a):
                self.coord_history.append(nxt[a])
                return a
    for a in order:
        if valid.get(a):
            self.coord_history.append(nxt[a])
            return a
    self.coord_history.append((x, y))
    return 'WAIT'


def _safe_moves(safety, valid):
    """Move actions flagged safe (and valid) by the mask."""
    out = set()
    try:
        for a in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
            if safety.get('safe', {}).get(a, False) and valid.get(a, False):
                out.add(a)
    except Exception:
        pass
    return out


def scalars_from_state(game_state, safety_info=None):
    from .features_cnn import scalars_from_state as _sc8
    return _sc8(game_state, safety_info)
