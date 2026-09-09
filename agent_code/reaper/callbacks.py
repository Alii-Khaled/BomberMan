"""Reaper act policy: learned Q (weight 1.0) + thin heuristic + strict mask.

Thin heuristic = safety-driven flee scores, loop penalties, WAIT penalty,
bomb-repeat penalty. Everything else (pathfinding, placement, hunting,
endgame) is the network's job — the Q was behavior-cloned from a strong
teacher then RL-refined, and the features give it BFS pathfinding +
placement value + opponent-model signals.

Safety mask from safety.py (proven: margin gate, exact blast, time-
expanded escape). Fallback without torch: mask + heuristic only.
"""
from collections import deque
import os
import random
import time

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from .safety import action_safety, true_blast
from .features import state_to_features, FEATURE_DIM
from .model import build_model, ACTION_LIST

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


Q_WEIGHT = _env_float('REAPER_Q_WEIGHT', 1.0)
HEUR_WEIGHT = _env_float('REAPER_HEUR_WEIGHT', 0.5)
HEUR_MAX = _env_float('REAPER_HEUR_MAX', 2.0)
G_WAIT = _env_float('REAPER_G_WAIT', 0.30)
G_REVISIT3 = _env_float('REAPER_G_REVISIT3', 0.45)
G_REVISIT2 = _env_float('REAPER_G_REVISIT2', 0.15)
G_BOMB_REPEAT = _env_float('REAPER_G_BOMB_REPEAT', 0.9)
G_FLEE = _env_float('REAPER_G_FLEE', 1.0)
# Post-bomb flee escalation. Default 1.0 = original behavior (a 20-round A/B
# showed 1.6 ~ 1.0 within noise on the zero-Q policy); tune via the Phase-4
# sweep once a trained Q is in the loop.
G_FLEE_LOCK = _env_float('REAPER_G_FLEE_LOCK', 1.0)
Q_CLIP = _env_float('REAPER_Q_CLIP', 50.0)
TIME_BUDGET = _env_float('REAPER_TIME_BUDGET', 0.12)
# Hunting prior: bonus for stepping toward a tile where a bomb would trap
# an opponent (feats[68:72]). Bounded and Q-overridable: guides an
# uncertain Q toward traps, negligible once the Q is confident. Env-gated
# for ablation (REAPER_TRAP_BONUS=0 removes it).
TRAP_BONUS = _env_float('REAPER_TRAP_BONUS', 1.5)
# Inference-time tactical overlay: off | tactical. Tactical = exact
# guaranteed-kill BOMB override (bomb_here_traps) + least-bad fallback
# when nothing is mask-safe. Off during training (clean Q0 ablation);
# evaluated as an inference overlay in the frozen bake-off.
SEARCH = os.environ.get('REAPER_SEARCH', 'off').strip().lower()
_DIR_TO_ACTION = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT',
                  (1, 0): 'RIGHT'}


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


def setup(self):
    self.logger.debug('reaper setup')
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
                obj = _t.load(cand, map_location='cpu', weights_only=True)
                sd = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
                if 'q_net' in sd:
                    sd = sd['q_net']
                self.model.load_state_dict(sd, strict=False)
                loaded = True
                self.logger.info(f'reaper loaded {cand}')
                break
            except Exception as ex:
                self.logger.warning(f'reaper load failed {cand}: {ex}')
    try:
        self.model.eval()
    except Exception:
        pass
    self.coord_history = deque([], 24)
    self.bomb_history = deque([], 5)
    self.current_round = 0
    self.flee_timer = 0
    self.own_bomb = None


def act(self, game_state):
    t0 = time.perf_counter()
    try:
        rnd = int(game_state.get('round', 0))
    except Exception:
        rnd = 0
    if rnd != getattr(self, 'current_round', 0):
        self.coord_history = deque([], 24)
        self.bomb_history = deque([], 5)
        self.current_round = rnd
        self.flee_timer = 0
        self.own_bomb = None
    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    safety = action_safety(game_state)
    valid, safe = safety['valid'], safety['safe']
    flee_locked = getattr(self, 'flee_timer', 0) > 0
    if flee_locked:
        self.flee_timer = max(0, self.flee_timer - 1)
    nxt = {'UP': (x, y - 1), 'DOWN': (x, y + 1), 'LEFT': (x - 1, y),
           'RIGHT': (x + 1, y), 'WAIT': (x, y), 'BOMB': (x, y)}

    heu = {a: 0.0 for a in ACTION_LIST}
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
        boost = G_FLEE_LOCK if flee_locked else G_FLEE
        if flee_locked:
            heu['BOMB'] -= 3.0
            for a in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
                if flee.get(a, 0.0) <= 0:
                    heu[a] -= 0.5
        for a in ACTION_LIST:
            heu[a] += boost * flee[a]
    except Exception:
        pass
    heu['WAIT'] -= G_WAIT
    # Bounded soft prior: with zero Q this is a uniform positive rescale
    # (argmax-preserving); with a trained Q it breaks ties but never
    # outvotes the network. The hard safety mask is the real guarantee.
    heu_arr = np.array([heu[a] for a in ACTION_LIST], dtype=np.float64)
    peak = float(np.abs(heu_arr).max())
    if peak > HEUR_MAX > 0:
        heu_arr *= HEUR_MAX / peak

    # Time budget: the tournament gives 0.5 s/step on one slow thread and
    # an overrun also taxes the NEXT step (environment.py:448-460), so we
    # spend at most ~60% of TIME_BUDGET on features and ~90% total. If the
    # budget is gone we rank by mask + heuristic (Q=0), which is safe.
    q = np.zeros(len(ACTION_LIST), dtype=np.float64)
    feats = None
    try:
        if _HAS_TORCH and getattr(self, 'model', None) is not None \
                and (time.perf_counter() - t0) < TIME_BUDGET * 0.6:
            own_bomb = getattr(self, 'own_bomb', None)
            feats = state_to_features(game_state, safety, own_bomb)
            if (time.perf_counter() - t0) < TIME_BUDGET * 0.9:
                import torch as _t
                with _t.no_grad():
                    tfeat = _t.from_numpy(feats.astype(np.float32)).unsqueeze(0)
                    qv = self.model(tfeat)
                    q = np.asarray(qv.squeeze(0).cpu().numpy(),
                                   dtype=np.float64)
                    q = np.nan_to_num(q, nan=0.0, posinf=Q_CLIP,
                                      neginf=-Q_CLIP)
                    if Q_CLIP > 0:
                        q = np.clip(q, -Q_CLIP, Q_CLIP)
    except Exception:
        pass

    total = HEUR_WEIGHT * heu_arr + Q_WEIGHT * q
    # Hunting prior (base policy): nudge toward trap tiles. Bounded,
    # Q-overridable, mask still filters. Uses feats[68:72] (DELTAS order).
    try:
        if TRAP_BONUS and feats is not None and len(feats) > 71:
            for d, dd in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
                tv = float(feats[68 + d])
                if tv > 0:
                    a = _DIR_TO_ACTION[dd]
                    total[ACTION_TO_IDX[a]] += TRAP_BONUS * tv
    except Exception:
        pass

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
                    self.own_bomb = (x, y)
                else:
                    self.coord_history.append(nxt[a])
                return a
    except Exception:
        pass

    # Tactical overlay (REAPER_SEARCH=tactical, inference only): an exact
    # guaranteed kill (+5, forced) dominates any Q estimate, so BOMB
    # overrides the ranking when bomb_here_traps proves one. Budget-guarded
    # like everything else.
    try:
        if SEARCH == 'tactical' and valid.get('BOMB') and safe.get('BOMB') \
                and (time.perf_counter() - t0) < TIME_BUDGET * 0.95:
            from .safety import bomb_here_traps
            _traps, _ = bomb_here_traps(game_state, safety)
            if _traps:
                self.bomb_history.append((x, y))
                self.flee_timer = 5
                self.own_bomb = (x, y)
                return 'BOMB'
    except Exception:
        pass

    order = sorted(range(len(ACTION_LIST)), key=lambda i: total[i], reverse=True)
    for i in order:
        a = ACTION_LIST[i]
        if valid.get(a) and safe.get(a):
            if a == 'BOMB':
                self.bomb_history.append((x, y))
                self.flee_timer = 5
                self.own_bomb = (x, y)
            else:
                self.coord_history.append(nxt[a])
                if bombs_left:
                    self.own_bomb = None
            return a
    # Least-bad fallback: nothing is mask-safe. Prefer the valid move the
    # network fears least, with an explicit penalty for stepping into
    # immediate danger (the Q alone may not price a forced death).
    try:
        if SEARCH == 'tactical':
            _danger_now = safety.get('danger')
            _best, _best_key = None, None
            for i in order:
                a = ACTION_LIST[i]
                if not valid.get(a):
                    continue
                _key = float(total[i])
                try:
                    if _danger_now is not None and a in nxt:
                        _nx, _ny = nxt[a]
                        if bool(_danger_now[0, _nx, _ny]):
                            _key -= 10.0
                except Exception:
                    pass
                if _best is None or _key > _best_key:
                    _best, _best_key = a, _key
            if _best is not None:
                if _best == 'BOMB':
                    self.bomb_history.append((x, y))
                    self.own_bomb = (x, y)
                else:
                    self.coord_history.append(nxt[_best])
                    if bombs_left:
                        self.own_bomb = None
                return _best
    except Exception:
        pass
    for i in order:
        a = ACTION_LIST[i]
        if valid.get(a):
            if a == 'BOMB':
                self.bomb_history.append((x, y))
                self.own_bomb = (x, y)
            else:
                self.coord_history.append(nxt[a])
                if bombs_left:
                    self.own_bomb = None
            return a
    self.coord_history.append((x, y))
    return 'WAIT'
