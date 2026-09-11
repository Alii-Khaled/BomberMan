"""Arbiter self-distillation recorder (P2-C): the SHIP arbiter acts,
and every acted state is saved as (98-dim feats, executed action) for
offline pi retraining. Training-time only — never ships.

Delegation is exact (same self object, same env/defaults as the ship),
so the recorded distribution IS the ship's own visitation. Labels are
the ship's executed actions (search first-steps included) — this is
DAgger-style self-distillation: supervision on the student's own
distribution, mixed with the teacher corpus at extract time
(ARBITER_PI_TEACHERS="0 1 2 4").

Action encoding matches arbiter.model.ACTION_LIST exactly
(['UP','RIGHT','DOWN','LEFT','WAIT','BOMB']).
"""
import os
import sys
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
_ACTION_TO_IDX = {a: i for i, a in enumerate(_ACTION_LIST)}
# E98 label transfer: record warden_v2's move at every ship-visited state
# (ARBITER_DAGGER_WARDEN=1) into acts_w. The ship still acts exactly as
# the ship (pure observer; no behavior change).
_DAG_WARDEN = os.environ.get('ARBITER_DAGGER_WARDEN', '0') == '1'


def _warden_label(self, game_state, safety):
    """Warden_v2's preferred non-BOMB move index at this state (observer)."""
    w = getattr(self, '_dag_warden', None)
    if w is None:
        return None
    wc = self._dag_wmod
    st = wc.parse(game_state)
    if st['round'] != getattr(w, 'current_round', -1):
        w.current_round = st['round']
        w.coord_hist = deque([], 24)
        w.bomb_hist = deque([], 5)
    _want, aux = wc.decide(st, w.bomb_hist)
    cands = wc.choose_fast(st, aux, w.coord_hist, _want, own_live=False)
    valid = safety.get('valid', {}) if safety else {}
    for _sc, a in sorted(cands, key=lambda p: p[0], reverse=True):
        if a == 'BOMB':
            continue
        if valid and not valid.get(a):
            continue
        return _ACTION_TO_IDX.get(a)
    return None


def setup(self):
    np.random.seed()
    try:
        from agent_code.arbiter.features import state_to_features
        from agent_code.arbiter.safety import action_safety
        import agent_code.arbiter.callbacks as arb
        import agent_code.arbiter.model as arbm
        assert list(arbm.ACTION_LIST) == _ACTION_LIST, arbm.ACTION_LIST
    except Exception as ex:
        raise RuntimeError(f'arbiter_dagger cannot import arbiter: {ex}')
    self._state_to_features = state_to_features
    self._action_safety = action_safety
    self._arb = arb
    # Ship policy setup on this same object (model, histories, timers).
    # Recorder state below uses _dag_* names exclusively (no collision
    # with arbiter's model/coord_history/bomb_history/current_round).
    self._arb.setup(self)
    self._dag_feats = []
    self._dag_acts = []
    self._dag_acts_w = []
    self._dag_warden = None
    if _DAG_WARDEN:
        try:
            import types
            from agent_code.warden_v2 import callbacks as _wc
            _w = types.SimpleNamespace()
            _wc.setup(_w)
            self._dag_warden = _w
            self._dag_wmod = _wc
            self.logger.info('arbiter_dagger recording warden_v2 labels')
        except Exception as ex:
            self._dag_warden = None
            self.logger.warning(f'warden labels unavailable: {ex}')
    self.logger.info('arbiter_dagger delegating to ship arbiter')


def act(self, game_state):
    action = self._arb.act(self, game_state)
    if action not in _ACTION_TO_IDX:
        action = 'WAIT'
    try:
        safety = self._action_safety(game_state)
        feats = self._state_to_features(game_state, safety)
        self._dag_feats.append(np.asarray(feats, dtype=np.float32))
        self._dag_acts.append(_ACTION_TO_IDX[action])
        if _DAG_WARDEN:
            _wl = _warden_label(self, game_state, safety)
            self._dag_acts_w.append(
                _wl if _wl is not None else _ACTION_TO_IDX['WAIT'])
    except Exception:
        pass
    return action
