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
import json
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

# E86 Phase-A death diagnostics (ARBITER_DIAG=path prefix; unset = no-op,
# zero behavior change). When set, act() appends a compact per-tick
# snapshot (position, action, safety mask, nearby bombs, opponents,
# arena deltas, explosion map) to an in-memory buffer and end_of_round
# dumps the tail to <prefix>_deaths.jsonl for every round in which the
# agent died (SURVIVED_ROUND absent). Purely additive logging.
_DIAG = os.path.abspath(
    os.environ.get('ARBITER_DIAG', '').strip()) if \
    os.environ.get('ARBITER_DIAG', '').strip() else ''

# ARBITER_POLICY: 'pi' (ship: learned prior ranks S0 moves) or 'warden'
# (E97 experiment: warden_v2's fast heuristic ranks S0 moves). Only the
# move fallback changes — search/tactical still own BOMB decisions and
# action_safety's mask still binds. Default 'pi' == ship behavior.
POLICY = os.environ.get('ARBITER_POLICY', 'pi').strip().lower()


def _diag_snapshot(self, game_state, t0_unused=0.0):
    """Compact per-tick record for post-hoc death attribution."""
    try:
        _, _, _, (x, y) = game_state['self']
        x, y = int(x), int(y)
        rec = {
            't': int(game_state.get('step', 0)),
            'pos': [x, y],
            'safe': self._diag_last_mask,
            'flee': int(getattr(self, 'flee_timer', 0) > 0),
            'must_flee': bool(self._diag_must_flee),
        }
        ev = getattr(self, '_diag_last_events', None)
        if ev:
            rec['ev'] = ev
        bombs = []
        for b in (game_state.get('bombs') or []):
            (bx, by), ttl = b
            if abs(bx - x) <= 5 and abs(by - y) <= 5:
                bombs.append([int(bx), int(by), int(ttl)])
        rec['bombs'] = bombs
        rec['mine'] = [(int(bx), int(by))
                       for (bx, by) in list(self.bomb_history)[-5:]]
        rec['others'] = [[o[0], int(o[3][0]), int(o[3][1])]
                         for o in (game_state.get('others') or [])]
        fld = game_state.get('field')
        h = hash(fld.tobytes()) if fld is not None else 0
        if h != getattr(self, '_diag_field_hash', -1):
            self._diag_field_hash = h
            rec['field'] = [[int(v) for v in row] for row in fld]
        em = game_state.get('explosion_map')
        if em is not None:
            cells = np.argwhere(np.asarray(em) > 0)
            if len(cells):
                rec['expl'] = [[int(cx), int(cy)]
                               for cx, cy in cells]
        return rec
    except Exception:
        return None


def diag_dump_round(self, last_action, events):
    """Called from train.py end_of_round (the dispatched hook).

    Writes <prefix>_deaths.jsonl: a round_meta line + the full per-tick
    buffer for EVERY round (survival context enables the missed-kill
    inventory; death rounds carry KILLED_SELF / GOT_KILLED in the
    accumulated round events).
    """
    if not _DIAG:
        return
    try:
        ev_str = ' '.join(str(ev) for ev in (events or []))
        survived = 'SURVIVED_ROUND' in ev_str
        killer = ('SURVIVED' if survived else
                  ('KILLED_SELF' if 'KILLED_SELF' in ev_str else
                   ('GOT_KILLED' if 'GOT_KILLED' in ev_str else 'UNKNOWN')))
        buf = getattr(self, '_diag_buf', None) or []
        meta = {
            'type': 'round_meta',
            'killer': killer,
            'killed_opp': 'KILLED_OPPONENT' in ev_str,
            'last_action': last_action,
            'n_ticks': len(buf),
        }
        with open(_DIAG + '_deaths.jsonl', 'a') as f:
            f.write(json.dumps(meta) + '\n')
            for rec in buf:
                if rec is None:
                    continue
                rec2 = dict(rec)
                rec2['type'] = 'tick'
                try:
                    f.write(json.dumps(rec2) + '\n')
                except Exception:
                    try:
                        f.write(json.dumps(rec2, default=str) + '\n')
                    except Exception:
                        pass
        self._diag_buf = []
    except Exception:
        pass


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
# E90 flee quality: while fleeing (own bomb ticking or must-flee), pick
# among the mask's safe moves by optionality (free neighbours at the
# destination + distance from opponents) instead of raw pi order.
# Certified-only (only re-ranks already-valid+safe moves) and
# feature-neutral (state_to_features is untouched). Default 0 =
# ship-identical.
FLEE_Q = os.environ.get('ARBITER_FLEE_Q', '0') == '1'
_DELTAS = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0),
           'RIGHT': (1, 0), 'WAIT': (0, 0), 'BOMB': (0, 0)}
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _flee_quality_choice(arena, bombs, others_xy, x, y, valid, safe, pi):
    """Best safe flee move by open-space + opponent-distance (E90).

    Only considers moves already marked valid+safe by the mask. Score =
    free non-occupied 4-neighbours at the destination + 0.25 * min
    Manhattan distance to an opponent; pi breaks ties. Returns None when
    no safe move exists.
    """
    try:
        W, H = arena.shape[0], arena.shape[1]
        bomb_set = set()
        for _xy, _t in (bombs or []):
            bomb_set.add((int(_xy[0]), int(_xy[1])))
        other_set = set((int(o[0]), int(o[1])) for o in (others_xy or []))
        best_a, best_key = None, None
        for a in ACTION_LIST:
            if a == 'BOMB' or not (valid.get(a) and safe.get(a)):
                continue
            dx, dy = _DELTAS[a]
            nx, ny = x + dx, y + dy
            open_nb = 0
            for ddx, ddy in _DIRS4:
                tx, ty = nx + ddx, ny + ddy
                if not (0 <= tx < W and 0 <= ty < H):
                    continue
                if arena[tx, ty] != 0:
                    continue
                if (tx, ty) in bomb_set or (tx, ty) in other_set:
                    continue
                open_nb += 1
            if other_set:
                dmin = min(abs(nx - ox) + abs(ny - oy)
                           for (ox, oy) in other_set)
            else:
                dmin = 20
            key = (open_nb + 0.25 * dmin, float(pi[ACTION_TO_IDX[a]]))
            if best_key is None or key > best_key:
                best_key, best_a = key, a
        return best_a
    except Exception:
        return None


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
    # Candidate gating: ARBITER_MODEL points at an eval-only weights file
    # (never used in tournament zips, defaults identical to the ship path).
    _cands = []
    _env_model = os.environ.get('ARBITER_MODEL', '').strip()
    if _env_model:
        # Agent callbacks run with cwd = the agent dir, so a relative
        # candidate path must be anchored to the repo root or it would be
        # silently skipped and the ship weights loaded instead.
        if not os.path.isabs(_env_model):
            _root = os.path.dirname(os.path.dirname(here))
            _env_model = os.path.join(_root, _env_model)
        if os.path.isfile(_env_model):
            _cands.append(_env_model)
        else:
            try:
                self.logger.warning(
                    f'ARBITER_MODEL not found: {_env_model}')
            except Exception:
                pass
    _cands += [os.path.join(here, 'my-saved-model.pt'),
               os.path.join(os.getcwd(), 'my-saved-model.pt')]
    for cand in _cands:
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
    # E97 hybrid: warden_v2 move prior (default off == ship behavior).
    self._warden = None
    if POLICY == 'warden':
        try:
            import types
            from agent_code.warden_v2 import callbacks as _wc
            _w = types.SimpleNamespace()
            _wc.setup(_w)
            self._warden = _w
            self._warden_mod = _wc
            self.logger.info('arbiter policy=warden (hybrid S0 moves)')
        except Exception as ex:
            self._warden = None
            try:
                self.logger.warning(f'warden policy unavailable: {ex}')
            except Exception:
                pass


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


def _warden_move(self, game_state, valid, safe, threat):
    """Best warden_v2 move (BOMB excluded) that is valid for this agent.

    E97 hybrid: warden_v2's heuristic carries the crate/coin/kill
    navigation; arbiter's search carries bomb placement. Returns None
    when the hook is off or warden has no admissible move.
    """
    w = getattr(self, '_warden', None)
    if w is None:
        return None
    try:
        wc = self._warden_mod
        st = wc.parse(game_state)
        if st['round'] != getattr(w, 'current_round', -1):
            w.current_round = st['round']
            w.coord_hist = deque([], 24)
            w.bomb_hist = deque([], 5)
        _want, aux = wc.decide(st, w.bomb_hist)
        cands = wc.choose_fast(st, aux, w.coord_hist, _want, own_live=False)
        for _sc, a in sorted(cands, key=lambda p: p[0], reverse=True):
            if a == 'BOMB' or not valid.get(a):
                continue
            if threat and not safe.get(a):
                continue
            return a
    except Exception:
        return None
    return None


def _warden_hist_update(self, action, x, y, nxt):
    """Mirror the executed action into warden's loop-avoidance histories."""
    try:
        w = getattr(self, '_warden', None)
        if w is None:
            return
        w.coord_hist.append(nxt[action] if action != 'BOMB' else (x, y))
    except Exception:
        pass


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
        a = _act_impl(self, game_state, t0)
        if _DIAG:
            rec = _diag_snapshot(self, game_state)
            if rec is not None:
                rec['act'] = a
                buf = getattr(self, '_diag_buf', None)
                if buf is None:
                    buf = self._diag_buf = []
                buf.append(rec)
        return a
    except Exception:
        try:
            self.logger.warning('arbiter act failed; WAIT fallback',
                                exc_info=True)
        except Exception:
            pass
        return 'WAIT'


def _act_impl(self, game_state, t0):
    try:
        rnd = int(game_state.get('round', 0))
    except Exception:
        rnd = 0
    try:
        self._rl_step = int(game_state.get('step', 0))
    except Exception:
        self._rl_step = 0
    if rnd != getattr(self, 'current_round', 0):
        self.coord_history = deque([], 24)
        self.bomb_history = deque([], 5)
        self.current_round = rnd
        self.flee_timer = 0
        if _DIAG:
            self._diag_buf = []
            self._diag_field_hash = -1
    arena = np.asarray(game_state['field'])
    _, _, bombs_left, (x, y) = game_state['self']
    x, y = int(x), int(y)

    safety = action_safety(game_state)
    valid, safe = safety.get('valid', {}), safety.get('safe', {})
    if _DIAG:
        self._diag_last_mask = [
            ''.join('1' if valid.get(k) else '0' for k in ACTION_LIST),
            ''.join('1' if safe.get(k) else '0' for k in ACTION_LIST)]
        self._diag_must_flee = bool(_must_flee(game_state,
                                               safety.get('danger')))
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

    # --- S0 policy: warden_v2 move prior (E97 hybrid, default off) ---
    # Search/tactical above may already have committed a BOMB or plan;
    # this only replaces the pi-ranked move fallback.
    if POLICY == 'warden':
        _wa = _warden_move(self, game_state, valid, safe,
                           must_flee or flee_locked)
        if _wa is not None:
            _warden_hist_update(self, _wa, x, y, nxt)
            return _commit(self, _wa, x, y, nxt, bombs_left)

    # --- E101 RL: on-policy masked-softmax sample (train mode only) ---
    if getattr(self, '_rl', False) and getattr(self, 'train', False):
        try:
            from .rl_policy import sample_action
            _rf = state_to_features(game_state, safety)
            _ra = sample_action(self, pi, valid, safe, must_flee,
                                flee_locked, _rf)
            if _ra is not None:
                return _commit(self, _ra, x, y, nxt, bombs_left)
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
        if FLEE_Q:
            _fa = _flee_quality_choice(
                arena, game_state.get('bombs') or [],
                [(o[3][0], o[3][1]) for o in
                 (game_state.get('others') or [])],
                x, y, valid, safe, pi)
            if _fa is not None:
                return _commit(self, _fa, x, y, nxt, bombs_left)
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
