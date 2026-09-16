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
from .features import transform_tensor, AUG_PERMS, N_CHANNELS
from .model import build_model, ACTION_LIST

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
# NG flat layout: raveled board tensor ++ scalars (features.FEATURE_DIM).
_TENSOR_DIM = N_CHANNELS * 17 * 17

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

# ARBITER_PERF: timing-bucket profiler (default off, zero behavior
# change). When set to a path prefix, act() records one JSONL line per
# tick with ms buckets (safety / feat(TTA recompute) / fwd / tac /
# search / total), a duel-vs-trek flag (armed opponent <= Manhattan 4)
# and the number of TTA symmetries completed; the buffer flushes on
# round change (and from train.py end_of_round for the final round).
_PERF_RAW = os.environ.get('ARBITER_PERF', '').strip()
_PERF = os.path.abspath(_PERF_RAW) if _PERF_RAW else ''
_PERF_ON = bool(_PERF)

# E100c device knob: 'cuda:0' (main CUDA device) by default when CUDA is
# available, resolved per-agent in setup() with automatic CPU fallback;
# ARBITER_DEVICE=cpu pins the ship CPU behavior.
DEVICE_NAME = os.environ.get('ARBITER_DEVICE', 'cuda:0').strip() or 'cuda:0'

# Gap diagnostics (E122 Phase 0): ARBITER_GAP_DIAG=path prefix records a
# compact per-tick trace aimed at the three observed behavior gaps
# (adjacent-coin misses, opening tempo, solo endgame waste). Purely
# additive logging; unset = no-op, zero behavior change. The buffer
# flushes on round change and from train.py end_of_round.
_GAP_RAW = os.environ.get('ARBITER_GAP_DIAG', '').strip()
_GAP = os.path.abspath(_GAP_RAW) if _GAP_RAW else ''
_GAP_ON = bool(_GAP)

# E102: trace search/tactical-committed BOMB steps into the RL trace
# (ARBITER_RL_BOMB_TRACE=1) so the move policy gets a gradient on the
# states where bombs are planted (66% of deaths were own-bomb).
BOMB_TRACE = os.environ.get('ARBITER_RL_BOMB_TRACE', '0') == '1'


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


def _perf_acc(self, bucket, t_start):
    """Add ms to a timing bucket (ARBITER_PERF only; no-op otherwise)."""
    if not _PERF_ON:
        return
    try:
        acc = getattr(self, '_perf_acc', None)
        if acc is None:
            acc = self._perf_acc = {}
        acc[bucket] = acc.get(bucket, 0.0) + \
            (time.perf_counter() - t_start) * 1000.0
    except Exception:
        pass


def _perf_record(self, game_state, action, t0):
    """Append the per-tick perf record (ARBITER_PERF only)."""
    if not _PERF_ON:
        return
    try:
        acc = getattr(self, '_perf_acc', None) or {}
        total = (time.perf_counter() - t0) * 1000.0
        duel = 0
        try:
            _, _, _, (px, py) = game_state['self']
            for o in (game_state.get('others') or []):
                if o[2] and abs(int(o[3][0]) - int(px)) \
                        + abs(int(o[3][1]) - int(py)) <= 4:
                    duel = 1
                    break
        except Exception:
            pass
        rec = {
            'type': 'perf',
            'round': int(game_state.get('round', 0)),
            'step': int(game_state.get('step', 0)),
            'act': action,
            'duel': duel,
            'tta_n': int(getattr(self, '_perf_tta_n', 0) or 0),
            'ms': {'total': round(total, 3),
                   **{k: round(v, 3) for k, v in sorted(acc.items())}},
        }
        buf = getattr(self, '_perf_buf', None)
        if buf is None:
            buf = self._perf_buf = []
        buf.append(rec)
        self._perf_acc = {}
        self._perf_tta_n = 0
    except Exception:
        pass


def perf_flush_round(self):
    """Dump accumulated perf records (ARBITER_PERF only)."""
    if not _PERF_ON:
        return
    try:
        buf = getattr(self, '_perf_buf', None) or []
        if not buf:
            return
        with open(_PERF + '_perf.jsonl', 'a') as f:
            for rec in buf:
                f.write(json.dumps(rec) + '\n')
        self._perf_buf = []
    except Exception:
        pass


def gap_flush_round(self):
    """Dump accumulated gap-diagnosis records (ARBITER_GAP_DIAG only).

    Writes a round_meta line + the tick buffer for the round that just
    ended (or is about to be replaced). Called from the round-change
    block of _act_impl and from train.py end_of_round.
    """
    if not _GAP_ON:
        return
    try:
        buf = getattr(self, '_gap_buf', None) or []
        meta = getattr(self, '_gap_meta', None) or {}
        if not buf and not meta:
            return
        with open(_GAP + '_gap.jsonl', 'a') as f:
            m = dict(meta)
            m['type'] = 'round_meta'
            f.write(json.dumps(m) + '\n')
            for rec in buf:
                try:
                    f.write(json.dumps(rec) + '\n')
                except Exception:
                    try:
                        f.write(json.dumps(rec, default=str) + '\n')
                    except Exception:
                        pass
        self._gap_buf = []
        self._gap_meta = {}
    except Exception:
        pass


def _gap_bfs_from(arena, blocked, start):
    """BFS distance map from (start) over free tiles (engine blocking:
    non-floor + bombs + other agents). Reused by the gap recorder and
    the certified coin-take overlay."""
    from .search import _bfs_dist
    return _bfs_dist(arena, blocked, start)


def _cointake_step(game_state, safety, d_cap):
    """Exact certified coin-take (E122): first step onto the shortest
    mask-safe path to the nearest visible collectable coin reachable
    within d_cap steps. Returns the action or None. Exact +1 when it
    fires: the coin tile is free, so reaching it collects deterministically.
    """
    try:
        arena = np.asarray(game_state['field'])
        _, _, _, (x, y) = game_state['self']
        x, y = int(x), int(y)
        bombs = game_state.get('bombs') or []
        bomb_set = set((int(b[0][0]), int(b[0][1])) for b in bombs)
        others_xy = [(int(o[3][0]), int(o[3][1]))
                     for o in (game_state.get('others') or [])]
        blocked = bomb_set | (set(others_xy) - {(x, y)})
        coins = [(int(c[0]), int(c[1]))
                 for c in (game_state.get('coins') or [])]
        if not coins:
            return None
        coins.sort(key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        W, H = arena.shape[0], arena.shape[1]
        dist = _gap_bfs_from(arena, blocked, (x, y))
        best = None
        for (cx, cy) in coins:
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            d = int(dist[cx, cy])
            if d > d_cap:
                continue
            if best is None or d < best[0]:
                best = (d, cx, cy)
            if best[0] <= 1:
                break
        if best is None:
            return None
        d, cx, cy = best
        dc = _gap_bfs_from(arena, blocked, (cx, cy))
        dself = int(dc[x, y])
        if dself != d:
            return None
        for act, (dx, dy) in (('UP', (0, -1)), ('DOWN', (0, 1)),
                              ('LEFT', (-1, 0)), ('RIGHT', (1, 0))):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if int(dc[nx, ny]) != d - 1:
                continue
            if not (safety.get('valid', {}).get(act, False)
                    and safety.get('safe', {}).get(act, False)):
                continue
            return act
    except Exception:
        return None
    return None


def _gap_record(self, game_state, action):
    """Compact per-tick gap record (ARBITER_GAP_DIAG only)."""
    if not _GAP_ON:
        return
    try:
        _, score, _, (x, y) = game_state['self']
        x, y = int(x), int(y)
        rnd = int(game_state.get('round', 0))
        step = int(game_state.get('step', 0))
        arena = np.asarray(game_state['field'])
        W, H = arena.shape[0], arena.shape[1]
        coins_now = [(int(c[0]), int(c[1]))
                     for c in (game_state.get('coins') or [])]
        seen = getattr(self, '_gap_seen', None)
        if seen is None:
            seen = self._gap_seen = set()
        seen.update(coins_now)
        coll = len(seen) - len([c for c in coins_now if c in seen])
        hidden = max(0, 9 - len(seen))
        bombs = game_state.get('bombs') or []
        others = game_state.get('others') or []
        others_xy = [(int(o[3][0]), int(o[3][1])) for o in others]
        blocked = set((int(b[0][0]), int(b[0][1])) for b in bombs) \
            | (set(others_xy) - {(x, y)})
        dist = _gap_bfs_from(arena, blocked, (x, y))
        coin_rec = None
        d_cap = 2
        if coins_now:
            bd, bc = None, None
            for (cx, cy) in coins_now:
                if 0 <= cx < W and 0 <= cy < H:
                    d = int(dist[cx, cy])
                    if d < (10 ** 9) and d <= d_cap \
                            and (bd is None or d < bd):
                        bd, bc = d, (cx, cy)
            if bd is not None:
                # dirs on a shortest self->coin path: neighbor n of self
                # with coin-rooted BFS dc[n] == dc[self] - 1 (the overlay's
                # own derivation; the self-rooted map cannot certify this).
                dc = _gap_bfs_from(arena, blocked, bc)
                dirs = []
                for act, (dx, dy) in (('UP', (0, -1)), ('DOWN', (0, 1)),
                                      ('LEFT', (-1, 0)), ('RIGHT', (1, 0))):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H \
                            and int(dc[nx, ny]) == bd - 1:
                        dirs.append(act)
                ok = any(
                    getattr(self, '_gap_valid', {}).get(a2, False)
                    and getattr(self, '_gap_safe', {}).get(a2, False)
                    for a2 in dirs) if dirs else False
                cnt = list(getattr(self, 'coord_history', [])).count(bc) \
                    if bd > 0 else None
                coin_rec = {'d': int(bd), 'dirs': dirs, 'ok': bool(ok),
                            'cnt': cnt}
        pi = getattr(self, '_gap_pi', None)
        act_rank = coin_rank = -1
        if pi is not None:
            order = sorted(range(len(pi)), key=lambda i: -float(pi[i]))
            act_rank = order.index(ACTION_TO_IDX.get(action, 5)) \
                if action in ACTION_TO_IDX else -1
            if coin_rec and coin_rec['dirs']:
                cr = min(order.index(ACTION_TO_IDX[d2])
                         for d2 in coin_rec['dirs'] if d2 in ACTION_TO_IDX)
                coin_rank = int(cr)
        dbg = getattr(self, '_last_search', None)
        srch = None
        if isinstance(dbg, dict):
            srch = {'plans': dbg.get('plans'),
                    'exh': int(bool(dbg.get('exhausted'))),
                    'bm': dbg.get('best_move'),
                    'bb': dbg.get('best_bomb'),
                    'me': dbg.get('margin_eff'),
                    'y': dbg.get('best_yield')}
        rec = {
            'type': 'tick', 't': step, 'pos': [x, y], 'act': action,
            'crates': int((arena == 1).sum()),
            'coins_now': len(coins_now), 'seen': len(seen),
            'coll': int(coll), 'hidden': int(max(0, 9 - len(seen))),
            'nb': len(bombs), 'nopp': len(others),
            'bl': int(bool(game_state['self'][2])),
            'dopp': min((abs(ox - x) + abs(oy - y) for (ox, oy)
                         in others_xy), default=-1),
            'flee': int(getattr(self, 'flee_timer', 0) > 0),
            'mf': int(bool(getattr(self, '_gap_mf', False))),
            'decided': int(bool(getattr(self, '_search_decided', False))),
            'srch': srch,
            'coin': coin_rec,
            'act_rank': act_rank, 'coin_rank': coin_rank,
        }
        buf = getattr(self, '_gap_buf', None)
        if buf is None:
            buf = self._gap_buf = []
        buf.append(rec)
        if not getattr(self, '_gap_meta', None):
            self._gap_meta = {'round': rnd, 'start_crates':
                              int((arena == 1).sum())}
        # MAX_STEPS==400 (settings.py): flush the final round of a frozen
        # eval battery (no end_of_round dispatch in eval mode).
        if step >= 400:
            gap_flush_round(self)
    except Exception:
        pass


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
# E112 (P0) escalating loop penalty. Modes: '0' = ship (bounded
# LOOP3/LOOP2), '1' = escalate everywhere, '2' = escalate ONLY when no
# opponent is alive. The ship's bounded tie-breaks are an order of
# magnitude below the pi logit gaps on OOD solo states (verified: a
# 363-step UP/DOWN ping-pong with pi gaps > 1.0). G1 screens: mode 1
# costs kills/suicides in opponent-ful play (4.40 -> 3.98, suic 8 -> 12),
# so opponent-ful behavior keeps the validated bounded pair; the solo
# freeze gets the escalation (L5 screen 6.38 -> 7.72). E112 promotion:
# default '2'; '0' restores ship behavior (+ ARBITER_SOLO_MARGIN=-1).
LOOP_ESC = os.environ.get('ARBITER_LOOP_ESC', '2').strip() or '0'
LOOP_ESC_STEP = _env_float('ARBITER_LOOP_ESC_STEP', 0.5)
LOOP_ESC_CAP = _env_float('ARBITER_LOOP_ESC_CAP', 3.0)
# E114 (P2) anti-pin guard (default off). E109 death attribution: 59% of
# deaths are corner pins, 40/48 from ENEMY bombs with esc=0 at the plant
# tick — we are already standing in a pocket an armed enemy can seal.
# Survival-content steering only (E62): when an armed opponent is within
# ANTIPIN_D Manhattan steps and our own safe-mobility is <= ANTIPIN_MOB,
# re-rank the safe moves by open-space + distance to the armed opponent
# (the validated _flee_quality_choice recipe, here triggered by pin risk
# instead of a live flee). Never selects an unsafe move; never vetoes a
# certified bomb.
ANTIPIN = _env_float('ARBITER_ANTIPIN', 0.0) > 0.0
ANTIPIN_D = int(_env_float('ARBITER_ANTIPIN_D', 2))
ANTIPIN_MOB = int(_env_float('ARBITER_ANTIPIN_MOB', 1))
ANTIPIN_DEADEND = _env_float('ARBITER_ANTIPIN_DEADEND', 0.0) > 0.0
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
# E107 C2: 7-tick exact-sim survival lookahead on survival-critical moves
# (must_flee / flee_locked). Each admissible successor is rolled out with
# the search's own opponent model; only moves whose rollout survives are
# eligible (pi order wins). Default 0 = ship behavior (myopic flee).
FLEE_LOOK = os.environ.get('ARBITER_FLEE_LOOK', '0') == '1'
# E122 (P0) certified coin-take overlay. The search owns bombs (and solo
# trek plans when SOLO_TREK is on) but never moves outside that; the move
# fallback is the pi rank, whose bounded tie-breaks can reject a coin tile
# that was recently visited and whose prior is OOD in solo/thin states.
# When a visible collectable coin is reachable within COINTAKE_D steps of
# mask-safe BFS path and the first step is mask-valid+safe, take it:
# exact certified +1 (E69 tactical precedent; 1-step replan-every-step,
# E75 lesson). Default 0 = ship behavior; d is swept {1,2,3}.
# E122 (P0) certified coin-take overlay. The search owns bombs (and solo
# trek plans when SOLO_TREK is on) but never moves outside that; the move
# fallback is the pi rank, whose bounded tie-breaks can reject a coin tile
# that was recently visited and whose prior is OOD in solo/thin states.
# When a visible collectable coin is reachable within COINTAKE_D steps of
# mask-safe BFS path and the first step is mask-valid+safe, take it:
# exact certified +1 (E69 tactical precedent; 1-step replan-every-step,
# E75 lesson). E122 ship: default 1 with d=3 (pooled canonical gate
# +0.097/+0.015 win vs fresh control, no leg regression; d sweep 1/2/3
# picked 3: +80 vs +26/+21 pooled screen points). ARBITER_COINTAKE=0
# restores pre-E122 behavior.
COINTAKE = os.environ.get('ARBITER_COINTAKE', '1') == '1'
COINTAKE_D = int(_env_float('ARBITER_COINTAKE_D', 3))
# E123 (P0) solo backtrack penalty. The E112 escalation saturates: once
# both ping-pong endpoints sit at the cap (-3.0), their penalties cancel
# and the pi gaps dominate again (the residual endgame A<->B / WAIT
# waste). The backtrack term is state-DEPENDENT: it fires only when the
# destination is the tile we just came from (immediate reversal),
# escalating with recent re-visits, so it keeps discriminating where the
# visit-count penalty cannot. Solo-gated and skipped while fleeing.
# 0 = off (ship); suggested sweep {0.5, 1.5, 3.0}.
BACKTRACK = _env_float('ARBITER_BACKTRACK', 0.0)
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


def _loop_penalty(cnt, solo=False):
    """E112 tie-break loop penalty for a destination visited `cnt` times.

    Pure and directly probed. LOOP_ESC '1' escalates always, '2' escalates
    only in solo states (no opponent alive), '0' keeps the validated
    bounded LOOP3/LOOP2 pair.
    """
    if (LOOP_ESC == '1') or (LOOP_ESC == '2' and solo):
        if cnt >= 2:
            return -min(LOOP_ESC_STEP * (cnt - 1), LOOP_ESC_CAP)
        return 0.0
    if cnt >= 3:
        return -LOOP3
    if cnt == 2:
        return -LOOP2
    return 0.0


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
    # P0-battery: ARBITER_NG_MODEL takes precedence so lobbies can hold
    # the ship arbiter (ARBITER_MODEL) and this agent (candidate) at once.
    _cands = []
    _env_model = (os.environ.get('ARBITER_NG_MODEL', '').strip()
                  or os.environ.get('ARBITER_MODEL', '').strip())
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
    self._device = None
    if _HAS_TORCH:
        try:
            import torch as _t
            self._device = _t.device(DEVICE_NAME)
            self.model.to(self._device)
        except Exception:
            self._device = None
    try:
        self.model.eval()
    except Exception:
        pass
    # Inc 1 (E119): jit-traced fast path — bit-exact vs eager (probe
    # parity 0), ~20% faster per forward; falls back to eager silently.
    self._fast = None
    if _HAS_TORCH and self.model is not None:
        try:
            import torch as _t
            _dev = self._device if self._device is not None \
                else _t.device('cpu')
            _dummy = _t.zeros((1, FEATURE_DIM), dtype=_t.float32,
                              device=_dev)
            self._fast = _t.jit.trace(self.model, (_dummy,))
            self._fast.eval()
        except Exception:
            self._fast = None
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


def _flee_lookahead_choice(game_state, valid, safe, pi, x, y):
    """E107 C2: exact-sim survival lookahead for survival-critical moves.

    For each admissible successor (pi order), roll out ~7 ticks with the
    search's own opponent model (CRN-style shared per-tick draws) and a
    danger-aware continuation for us. Returns the first move whose
    rollout survives, else None (caller falls back to the myopic rank).
    """
    try:
        from .sim import from_game_state, step as sim_step, valid_actions
        from .search import _opp_move, _continuation, _sim_bombs, _snap
        from .safety import future_danger
    except Exception:
        return None
    cands = [a for a in ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT')
             if valid.get(a) and (a == 'WAIT' or safe.get(a))]
    if not cands:
        return None
    cands.sort(key=lambda a: -float(pi[ACTION_TO_IDX.get(a, 0)]))
    try:
        st0 = from_game_state(game_state)
    except Exception:
        return None
    base_seed = 1000 * int(game_state.get('round', 0)) \
        + int(game_state.get('step', 0))
    best, best_key = None, None
    for a in cands:
        st = _snap(st0)
        alive_ticks = 0
        try:
            danger = future_danger(st['arena'], _sim_bombs(st), None, 4)
        except Exception:
            danger = None
        acts = [a]
        for i in range(1, len(st['agents'])):
            if st['agents'][i]['alive']:
                r = np.random.default_rng((base_seed, 0, i))
                acts.append(_opp_move(r, st, i, danger))
            else:
                acts.append('WAIT')
        try:
            sim_step(st, acts)
        except Exception:
            continue
        if not st['agents'][0]['alive']:
            continue
        alive_ticks = 1
        while alive_ticks < 7 and st['agents'][0]['alive']:
            try:
                danger = future_danger(st['arena'], _sim_bombs(st), None, 4)
            except Exception:
                danger = None
            acts = [_continuation(None, st, 0, danger)]
            for i in range(1, len(st['agents'])):
                if st['agents'][i]['alive']:
                    r = np.random.default_rng(
                        (base_seed, alive_ticks, i))
                    acts.append(_opp_move(r, st, i, danger))
                else:
                    acts.append('WAIT')
            try:
                sim_step(st, acts)
            except Exception:
                break
            if st['agents'][0]['alive']:
                alive_ticks += 1
        if st['agents'][0]['alive'] and alive_ticks >= 6:
            return a  # first (pi-best) surviving rollout wins
        key = (alive_ticks,)
        if best_key is None or key > best_key:
            best, best_key = a, key
    return best if best is not None else None


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
        if _PERF_ON:
            _perf_record(self, game_state, a, t0)
        if _GAP_ON:
            _gap_record(self, game_state, a)
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
        if _PERF_ON:
            perf_flush_round(self)
        if _GAP_ON:
            gap_flush_round(self)
        self.coord_history = deque([], 24)
        self.bomb_history = deque([], 5)
        self.current_round = rnd
        self.flee_timer = 0
        if _DIAG:
            self._diag_buf = []
            self._diag_field_hash = -1
        if _GAP_ON:
            self._gap_seen = set()
    arena = np.asarray(game_state['field'])
    _, _, bombs_left, (x, y) = game_state['self']
    x, y = int(x), int(y)

    _pt = time.perf_counter() if _PERF_ON else 0.0
    safety = action_safety(game_state)
    _perf_acc(self, 'safety', _pt)
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
                # E74/E119: symmetry-averaged prior. pi[a] = mean_s
                # fwd(T_s(gs))[T_s(a)]; V is an invariant scalar,
                # averaged directly. Inc 1 computes the features ONCE
                # (canonical view) and derives the other 7 views by the
                # exact dihedral gather (transform_tensor + AUG_PERMS) —
                # the pretrain-augmentation semantics, ~90x cheaper than
                # the old 8x state/safety/features recompute. Divergence
                # vs the old path is confined to the escape-tie-break
                # block f[45..48] (probe_ng_tta_equiv.py: 0/685
                # must-flee argmax flips, 0.1% trek-only at n=2000).
                # Budget gates and the uniform-pi fallback are unchanged.
                _pt = time.perf_counter() if _PERF_ON else 0.0
                _pairs = []
                try:
                    _f0 = state_to_features(game_state, safety)
                    _tb0 = _f0[:_TENSOR_DIM].reshape(N_CHANNELS, 17, 17)
                    _s0 = _f0[_TENSOR_DIM:]
                    for _s in range(8):
                        if (time.perf_counter() - t0) \
                                >= TIME_BUDGET * 0.9:
                            break
                        if _s == 0:
                            _pairs.append((_s, _f0))
                        else:
                            _pairs.append((_s, np.concatenate(
                                [transform_tensor(_tb0, _s).ravel(),
                                 _s0[AUG_PERMS[_s]]])
                                .astype(np.float32)))
                except Exception:
                    _pairs = []
                _perf_acc(self, 'feat', _pt)
                if _PERF_ON:
                    self._perf_tta_n = len(_pairs)
                if _pairs and (time.perf_counter() - t0) \
                        < TIME_BUDGET * 0.9:
                    import torch as _t
                    _pt = time.perf_counter() if _PERF_ON else 0.0
                    with _t.inference_mode():
                        _tb = _t.from_numpy(np.stack(
                            [f.astype(np.float32) for (_, f) in _pairs]))
                        _dev = getattr(self, '_device', None)
                        if _dev is not None:
                            _tb = _tb.to(_dev)
                        _mdl = getattr(self, '_fast', None) \
                            or getattr(self, 'model')
                        _lg, _vv = _mdl(_tb)
                        _lg = np.asarray(_lg.cpu().numpy(),
                                          dtype=np.float64)
                        _vv = np.asarray(_vv.cpu().numpy(),
                                          dtype=np.float64)
                    _perf_acc(self, 'fwd', _pt)
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
                _pt = time.perf_counter() if _PERF_ON else 0.0
                feats = state_to_features(game_state, safety)
                _perf_acc(self, 'feat', _pt)
                if (time.perf_counter() - t0) < TIME_BUDGET * 0.9:
                    import torch as _t
                    _pt = time.perf_counter() if _PERF_ON else 0.0
                    with _t.inference_mode():
                        tf = _t.from_numpy(
                            feats.astype(np.float32)).unsqueeze(0)
                        _dev = getattr(self, '_device', None)
                        if _dev is not None:
                            tf = tf.to(_dev)
                        mdl = getattr(self, '_fast', None) \
                            or getattr(self, 'model')
                        logits, vv = mdl(tf)
                        pi = np.asarray(logits.squeeze(0).cpu().numpy(),
                                        dtype=np.float64)
                        pi = np.nan_to_num(pi, nan=0.0, posinf=50.0,
                                            neginf=-50.0)
                        v = float(np.asarray(vv.squeeze(0).cpu().numpy()))
                        if V_OFF:
                            v = 0.0
                    _perf_acc(self, 'fwd', _pt)
    except Exception:
        pass
    try:
        self._last_v = float(v)
    except Exception:
        pass
    if _GAP_ON:
        try:
            self._gap_pi = np.asarray(pi, dtype=np.float64).copy()
        except Exception:
            pass
    if _GAP_ON:
        try:
            self._gap_mf = bool(must_flee)
            self._gap_valid = valid
            self._gap_safe = safe
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
            _pt = time.perf_counter() if _PERF_ON else 0.0
            _traps, _ = bomb_here_traps(game_state, safety)
            _perf_acc(self, 'tac', _pt)
            if _traps:
                _rl_trace_bomb(self, game_state, safety, valid)
                self._search_decided = True
                self.bomb_history.append((x, y))
                self.flee_timer = 5
                return 'BOMB'
    except Exception:
        pass
    # --- P1 bounded search: exact plans + learned leaves (E62) ---
    # The net never votes on root actions; search selects, V evaluates.
    # Returns None on budget exhaust or failure -> S0 ranking below.
    # E121: _search_decided marks search/tactical-owned commits for the
    # ExIt recorder (zero behavior change); _last_search reset to avoid
    # stale debug leaking between ticks.
    self._last_search = None
    self._search_decided = False
    try:
        if _SEARCH_ON:
            from .search import search_action
            remaining = TIME_BUDGET - (time.perf_counter() - t0)
            if remaining > 0.05:
                _pt = time.perf_counter() if _PERF_ON else 0.0
                a, dbg = search_action(game_state, safety,
                                       getattr(self, 'model', None),
                                       t0, remaining)
                _perf_acc(self, 'search', _pt)
                try:
                    self._last_search = dbg
                except Exception:
                    pass
                if a in ACTION_LIST:
                    if a == 'BOMB':
                        _rl_trace_bomb(self, game_state, safety, valid)
                    self._search_decided = True
                    return _commit(self, a, x, y, nxt, bombs_left)
    except Exception:
        pass

    # --- E122 certified coin-take overlay (default off) ---
    # The search never owns moves outside the solo trek; the move fallback
    # is the pi rank, whose tie-breaks can reject a coin tile that was
    # recently visited. When a visible collectable coin is reachable in
    # <= COINTAKE_D steps and the first step of the shortest path is
    # mask-valid + mask-safe, take it: an exact certified +1 (same class
    # as the E69 tactical overlay; 1-step, replan-every-step per E75).
    # Never overrides a live must-flee or a search/tactical commit.
    if COINTAKE and not must_flee and not getattr(self, '_search_decided',
                                                  False) \
            and (time.perf_counter() - t0) < TIME_BUDGET * 0.6:
        try:
            _ca = _cointake_step(game_state, safety, COINTAKE_D)
        except Exception:
            _ca = None
        if _ca is not None:
            return _commit(self, _ca, x, y, nxt, bombs_left)

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
    # E114 (P2) anti-pin guard: leave a pocket BEFORE an armed opponent
    # can seal it. Survival content only: re-ranks valid+safe moves,
    # never overrides a live flee and never picks an unsafe tile.
    if ANTIPIN and not must_flee and not flee_locked:
        try:
            _armed = [(int(o[3][0]), int(o[3][1]))
                      for o in (game_state.get('others') or []) if o[2]]
            _near = [(ox, oy) for (ox, oy) in _armed
                     if abs(ox - x) + abs(oy - y) <= ANTIPIN_D]
            if _near:
                _n_safe = 0
                for _d in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
                    if valid.get(_d) and safe.get(_d):
                        _n_safe += 1
                _pocket = _n_safe <= ANTIPIN_MOB
                if _pocket and ANTIPIN_DEADEND:
                    _free = 0
                    for _dx, _dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        _nx, _ny = x + _dx, y + _dy
                        if 0 <= _nx < arena.shape[0] and \
                                0 <= _ny < arena.shape[1] and \
                                arena[_nx, _ny] == 0:
                            _free += 1
                    _pocket = _free <= 1
                if _pocket:
                    _v2 = dict(valid)
                    _v2['WAIT'] = False
                    _s2 = dict(safe)
                    _s2['WAIT'] = False
                    _pa = _flee_quality_choice(
                        arena, game_state.get('bombs') or [], _near,
                        x, y, _v2, _s2, pi)
                    if _pa is not None:
                        return _commit(self, _pa, x, y, nxt, bombs_left)
        except Exception:
            pass
    _solo_now = not (game_state.get('others') or [])

    def tiebreak(a):
        t = 0.0
        try:
            cnt = list(self.coord_history).count(nxt[a])
            t += _loop_penalty(cnt, solo=_solo_now)
            if BACKTRACK > 0 and _solo_now and a in ('UP', 'DOWN',
                                                     'LEFT', 'RIGHT') \
                    and not (flee_locked or must_flee):
                h = list(self.coord_history)
                if len(h) >= 2 and nxt[a] == h[-2]:
                    n_recent = h[-8:].count(nxt[a])
                    t -= BACKTRACK * (1.0 + 0.5 * max(0, n_recent - 1))
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
        # E107 C2 retune: survival lookahead only when the tile itself is
        # lethal (must_flee); flee_timer alone stays on the myopic rank
        # (the first version hijacked all post-plant moves and collapsed
        # the economy: -1.8..-5.5 across batteries).
        if FLEE_LOOK and must_flee:
            try:
                _fl = _flee_lookahead_choice(game_state, valid, safe,
                                             pi, x, y)
            except Exception:
                _fl = None
            if _fl is not None:
                return _commit(self, _fl, x, y, nxt, bombs_left)
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


def _rl_trace_bomb(self, game_state, safety, valid):
    """E102: record a search/tactical-committed BOMB step for REINFORCE.

    Allowed set = every mask-valid action at this state (BOMB included);
    chosen = BOMB. Returns flow via returns-to-go as for move steps.
    """
    if not BOMB_TRACE:
        return
    if not (getattr(self, '_rl', False) and getattr(self, 'train', False)):
        return
    try:
        allowed = [ACTION_TO_IDX[a] for a in ACTION_LIST if valid.get(a)]
        if not allowed or ACTION_TO_IDX['BOMB'] not in allowed:
            return
        f = state_to_features(game_state, safety)
        self._rl_trace.append(
            (int(getattr(self, 'current_round', 0)),
             int(getattr(self, '_rl_step', 0)),
             np.asarray(f, dtype=np.float32).copy(),
             tuple(allowed), allowed.index(ACTION_TO_IDX['BOMB'])))
    except Exception:
        pass
