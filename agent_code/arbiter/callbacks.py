"""ARBITER act policy (S0): learned policy prior + thin survival
skeleton + warden-semantics mask filter. NO heuristic/Q blending.

Structural rule from A4 (E62): the net NEVER shares a vote with a
heuristic over root actions (apex double-counted warden into Q-delta
-0.42). Here pi ranks, V evaluates leaves (P1), and the heuristic
carries ONLY survival content: must_flee override, loop tie-break,
bomb-repeat tie-break — all bounded to <=0.5, an order below typical
logit gaps. Everything else (navigation, economy, combat, placement)
is the network's job.

Decision (S0, ARBITER_SEARCH=off):
  1. mask = action_safety(game_state) -> valid / safe / danger timeline.
  2. pi logits over 98-dim features (single forward also yields V, logged).
  3. must_flee (own tile lethal t<=1): restrict to safe moves (warden).
     else: rank valid moves by pi (S2 arm1: unconditional mask filter
     cost ~0.4 — the mask binds only under threat).
  4. Loop/bomb-repeat tie-breaks (bounded), BOMB gated on mask-safe.
  5. Budget guard (ARBITER_TIME_BUDGET 0.30): features+forward must fit
     in 60%/90%; on exhaustion degrade to mask + tie-breaks (pi=uniform).
P1 adds the bounded best-first search between steps 2 and 3
(ARBITER_SEARCH=search, agent_code/arbiter/search.py); the tactical proven-kill override
(ARBITER_SEARCH=tactical) is an inference-only exact computation.
'search+tactical' (E69) runs both: forced +5 first, search decides rest.
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

from .safety import action_safety, bomb_here_traps
from .features import state_to_features, FEATURE_DIM
from .features import transform_state, map_action
from .model import build_model, ACTION_LIST

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


# S0 knobs. ARBITER_SEARCH: search (SHIP default — validated G1 3.95)
# | tactical (proven-kill overlay, inference only) | off (S0 fallback)
# | search+tactical (E69: exact forced-kill overlay runs first, search
# decides everything else — a certified +5 should never lose a
# BOMB_MARGIN arbitration against a move plan).
SEARCH = os.environ.get('ARBITER_SEARCH', 'search').strip().lower()
# Substring flags (exact for all four mode strings: 'search' has no
# 'tactical' in it, 'tactical' has no 'search' in it, 'off' has neither).
_SEARCH_ON = 'search' in SEARCH
_TACTICAL_ON = 'tactical' in SEARCH
TIME_BUDGET = _env_float('ARBITER_TIME_BUDGET', 0.30)
# Ablation switches (E36/Q0 pattern): PI_OFF=1 forces a uniform prior
# (pi0 ablation — isolates the learned prior); V_OFF=1 forces V=0
# (V0 ablation — isolates the learned evaluator, load-bearing in P1).
PI_OFF = os.environ.get('ARBITER_PI_OFF', '0') == '1'
V_OFF = os.environ.get('ARBITER_V_OFF', '0') == '1'
LOOP3 = _env_float('ARBITER_LOOP3', 0.45)
LOOP2 = _env_float('ARBITER_LOOP2', 0.15)
BOMB_REPEAT = _env_float('ARBITER_BOMB_REPEAT', 0.9)
TRAP_BONUS = _env_float('ARBITER_TRAP_BONUS', 0.0)
# Dihedral TTA for pi (E74, SHIP default — best config on primary +
# warden-mix + collectors simultaneously): average the prior over the
# 8 exact board symmetries. Set ARBITER_TTA=0 for the single-forward
# P1 ablation.
TTA = os.environ.get('ARBITER_TTA', '1') == '1'
_DELTAS = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0),
           'RIGHT': (1, 0), 'WAIT': (0, 0), 'BOMB': (0, 0)}


def setup(self):
    # P1.1 safe defaults FIRST: if model construction/loading below
    # raises, act() still degrades to uniform-pi + mask (never crashes —
    # the engine has no fallback agent, so a setup crash kills the whole
    # run). model=None is a supported degradation (pi uniform, V
    # skipped via the getattr/model-None guards in act/search).
    self.model = None
    self.coord_history = deque([], 24)
    self.bomb_history = deque([], 5)
    self.current_round = 0
    self.flee_timer = 0
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
                if isinstance(obj, dict):
                    for key in ('arbiter', 'pi_v_net', 'state_dict',
                                'q_net', 'model'):
                        if key in obj and isinstance(obj[key], dict):
                            obj = obj[key]
                            break
                self.model.load_state_dict(obj, strict=False)
                loaded = True
                self.logger.info(f'arbiter loaded {cand}')
                break
            except Exception as ex:
                self.logger.warning(f'arbiter load failed {cand}: {ex}')
    try:
        self.model.eval()
    except Exception:
        pass
    if not loaded:
        try:
            self.logger.info('arbiter: no weights, pi uniform + V zero '
                             '(heuristic skeleton drives play)')
        except Exception:
            pass
    self.coord_history = deque([], 24)
    self.bomb_history = deque([], 5)
    self.current_round = 0
    self.flee_timer = 0


def _must_flee(game_state, danger):
    """Own tile lethal within 1 step (warden convention)."""
    try:
        _, _, _, (x, y) = game_state['self']
        d = np.asarray(danger)
        if d.ndim == 3 and d.shape[0] > 1:
            return bool(d[0, int(x), int(y)] or d[1, int(x), int(y)])
    except Exception:
        pass
    return False


def act(self, game_state):
    """Exception-armored entry (P1.1): any failure -> WAIT fallback.

    The success path is bit-identical to the validated flow (see
    _act_impl); the armor only converts a tournament-killing crash
    (the engine has no fallback agent — an unguarded act exception
    kills the whole game, or benches us for the round under
    --silence-errors) into a single WAIT.
    """
    t0 = time.perf_counter()
    try:
        return _act_impl(self, game_state, t0)
    except Exception:
        try:
            self.logger.warning('arbiter act failed; WAIT fallback')
        except Exception:
            pass
        return 'WAIT'


def _act_impl(self, game_state, t0):
    try:
        rnd = int(game_state.get('round', 0))
    except Exception:
        rnd = 0
    if rnd != getattr(self, 'current_round', 0):
        self.coord_history = deque([], 24)
        self.bomb_history = deque([], 5)
        self.current_round = rnd
        self.flee_timer = 0
    arena = np.asarray(game_state['field'])
    _, _, bombs_left, (x, y) = game_state['self']
    x, y = int(x), int(y)

    safety = action_safety(game_state)
    valid, safe = safety.get('valid', {}), safety.get('safe', {})
    flee_locked = getattr(self, 'flee_timer', 0) > 0
    if flee_locked:
        self.flee_timer = max(0, self.flee_timer - 1)
    must_flee = _must_flee(game_state, safety.get('danger'))
    nxt = {'UP': (x, y - 1), 'DOWN': (x, y + 1), 'LEFT': (x - 1, y),
           'RIGHT': (x + 1, y), 'WAIT': (x, y), 'BOMB': (x, y)}

    # --- learned prior (single forward yields pi + V) ---
    pi = np.full(len(ACTION_LIST), 1.0 / len(ACTION_LIST),
                 dtype=np.float64)
    v = 0.0
    try:
        if _HAS_TORCH and getattr(self, 'model', None) is not None \
                and not PI_OFF \
                and (time.perf_counter() - t0) < TIME_BUDGET * 0.6:
            if TTA:
                # E74: symmetry-averaged prior. Features are exactly
                # equivariant, so pi[a] = mean_s fwd(T_s(gs))[T_s(a)];
                # V is an invariant scalar, averaged directly. Any
                # failure or budget pressure falls back to uniform pi
                # (mask + tie-breaks decide, as in the S0 path).
                _pairs = []
                for _s in range(8):
                    if (time.perf_counter() - t0) >= TIME_BUDGET * 0.9:
                        break
                    try:
                        if _s == 0:
                            _gs, _sf = game_state, safety
                        else:
                            _gs = transform_state(game_state, _s)
                            _sf = action_safety(_gs)
                        _pairs.append(
                            (_s, state_to_features(_gs, _sf)))
                    except Exception:
                        continue
                if _pairs and (time.perf_counter() - t0) \
                        < TIME_BUDGET * 0.9:
                    import torch as _t
                    with _t.no_grad():
                        _tb = _t.from_numpy(np.stack(
                            [f.astype(np.float32) for (_, f) in _pairs]))
                        _mdl = getattr(self, 'model')
                        _lg, _vv = _mdl(_tb)
                        _lg = np.asarray(_lg.cpu().numpy(),
                                          dtype=np.float64)
                        _vv = np.asarray(_vv.cpu().numpy(),
                                          dtype=np.float64)
                    _acc = np.zeros(len(ACTION_LIST), dtype=np.float64)
                    for (_s, _), _row in zip(_pairs, _lg):
                        _row = np.nan_to_num(_row, nan=0.0, posinf=50.0,
                                              neginf=-50.0)
                        for _a in ACTION_LIST:
                            try:
                                _j = ACTION_TO_IDX[map_action(_a, _s)]
                            except Exception:
                                continue
                            _acc[ACTION_TO_IDX[_a]] += _row[_j]
                    pi = _acc / max(1, len(_pairs))
                    v = float(np.mean(_vv[:len(_pairs)]))
                    if V_OFF:
                        v = 0.0
            else:
                feats = state_to_features(game_state, safety)
                if (time.perf_counter() - t0) < TIME_BUDGET * 0.9:
                    import torch as _t
                    with _t.no_grad():
                        tf = _t.from_numpy(
                            feats.astype(np.float32)).unsqueeze(0)
                        mdl = getattr(self, 'model')
                        logits, vv = mdl(tf)
                        pi = np.asarray(logits.squeeze(0).cpu().numpy(),
                                        dtype=np.float64)
                        pi = np.nan_to_num(pi, nan=0.0, posinf=50.0,
                                            neginf=-50.0)
                        v = float(np.asarray(vv.squeeze(0).cpu().numpy()))
                        if V_OFF:
                            v = 0.0
    except Exception:
        pass
    try:
        self._last_v = float(v)
    except Exception:
        pass

    # --- epsilon-greedy exploration during training (masked pool) ---
    try:
        if getattr(self, 'train', False):
            eps = float(getattr(self, 'epsilon', 0.0) or 0.0)
            if eps > 0 and random.random() < eps:
                pool = [a for a in ACTION_LIST
                        if valid.get(a) and safe.get(a)] \
                    or [a for a in ACTION_LIST if valid.get(a)] \
                    or ['WAIT']
                a = random.choice(pool)
                if a == 'BOMB':
                    self.bomb_history.append((x, y))
                    self.flee_timer = 5
                else:
                    self.coord_history.append(nxt[a])
                return a
    except Exception:
        pass

    # --- tactical proven-kill override (inference only, exact) ---
    # Runs in 'tactical' AND 'search+tactical' (E69) modes. A forced +5
    # never loses a BOMB_MARGIN arbitration: mask-safe already implies a
    # certified escape, so deferring to the search can only veto it.
    try:
        if _TACTICAL_ON and valid.get('BOMB') and safe.get('BOMB') \
                and (time.perf_counter() - t0) < TIME_BUDGET * 0.95:
            _traps, _ = bomb_here_traps(game_state, safety)
            if _traps:
                self.bomb_history.append((x, y))
                self.flee_timer = 5
                return 'BOMB'
    except Exception:
        pass
    # --- P1 bounded search: exact plans + learned leaves (E62) ---
    # The net never votes on root actions; search selects, V evaluates.
    # Returns None on budget exhaust or failure -> S0 ranking below.
    try:
        if _SEARCH_ON:
            from .search import search_action
            remaining = TIME_BUDGET - (time.perf_counter() - t0)
            if remaining > 0.05:
                a, dbg = search_action(game_state, safety,
                                       getattr(self, 'model', None),
                                       t0, remaining)
                try:
                    self._last_search = dbg
                except Exception:
                    pass
                if a in ACTION_LIST:
                    return _commit(self, a, x, y, nxt, bombs_left)
    except Exception:
        pass

    # --- rank: pi, warden filter semantics, bounded tie-breaks only ---
    def tiebreak(a):
        t = 0.0
        try:
            cnt = list(self.coord_history).count(nxt[a])
            if cnt >= 3:
                t -= LOOP3
            elif cnt == 2:
                t -= LOOP2
            if a == 'BOMB' and (x, y) in list(self.bomb_history)[-3:]:
                t -= BOMB_REPEAT
            if flee_locked and a == 'BOMB':
                t -= 3.0
        except Exception:
            pass
        return t

    order = sorted(range(len(ACTION_LIST)),
                   key=lambda i: (pi[i] + tiebreak(ACTION_LIST[i]),
                                  -i),
                   reverse=True)
    # Warden semantics (S2 arm1, +0.43 pooled): the mask binds moves only
    # under threat; otherwise every valid move is rankable by pi.
    if must_flee or flee_locked:
        for i in order:
            a = ACTION_LIST[i]
            if valid.get(a) and safe.get(a):
                return _commit(self, a, x, y, nxt, bombs_left)
    for i in order:
        a = ACTION_LIST[i]
        if a == 'BOMB':
            if valid.get(a) and safe.get(a):
                return _commit(self, a, x, y, nxt, bombs_left)
            continue
        if valid.get(a):
            return _commit(self, a, x, y, nxt, bombs_left)
    # Least-bad fallback: nothing valid ranked — take any valid move.
    for i in order:
        a = ACTION_LIST[i]
        if valid.get(a):
            return _commit(self, a, x, y, nxt, bombs_left)
    self.coord_history.append((x, y))
    return 'WAIT'


def _commit(self, a, x, y, nxt, bombs_left):
    """Bookkeeping on a chosen action (flee escalation included)."""
    try:
        if a == 'BOMB':
            self.bomb_history.append((x, y))
            self.flee_timer = 5
        else:
            self.coord_history.append(nxt[a])
    except Exception:
        pass
    return a
