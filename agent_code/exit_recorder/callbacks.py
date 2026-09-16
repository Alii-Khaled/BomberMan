"""ExIt self-distillation recorder (E121): the SHIP arbiter_ng acts, and
every state where the SEARCH produced a decision is saved as
(flat 3566-dim NG feats, search-arbitrated label) for offline pi
distillation. Training-time only — never ships.

Labels are the EXACT-ROLLOUT improver's choice (ExIt / AlphaZero-style
policy iteration, step 1):
  - search committed a BOMB (arbitration or the executed action was the
    search/tactical bomb first-step) -> label = BOMB index;
  - otherwise the top-scoring MOVE plan's first step (dbg
    'best_move_first') -> label = that move index.
Rows where the search exhausted its budget or produced no decision are
SKIPPED (no reliable improver signal). The executed action is recorded
alongside (acts) for the DAgger-style mix, matching arbiter_dagger.

Delegation is exact (same self object, same env/defaults as the ship),
so the recorded distribution IS the ship's own visitation.

Flat layout matches arbiter_ng.features.FEATURE_DIM
(12x17x17 tensor raveled ++ 98 scalars).
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
BOMB_IDX = _ACTION_TO_IDX['BOMB']


def setup(self):
    np.random.seed()
    try:
        from agent_code.arbiter_ng.features import state_to_features
        from agent_code.arbiter_ng.safety import action_safety
        import agent_code.arbiter_ng.callbacks as arb
        import agent_code.arbiter_ng.model as arbm
        assert list(arbm.ACTION_LIST) == _ACTION_LIST, arbm.ACTION_LIST
    except Exception as ex:
        raise RuntimeError(f'exit_recorder cannot import arbiter_ng: {ex}')
    self._state_to_features = state_to_features
    self._action_safety = action_safety
    self._arb = arb
    self._arbm = arbm
    # Ship policy setup on this same object (model, histories, timers).
    # Recorder state uses _exit_* names exclusively.
    self._arb.setup(self)
    self._exit_feats = []
    self._exit_labels = []
    self._exit_acts = []
    self.logger.info('exit_recorder delegating to ship arbiter_ng')


def act(self, game_state):
    action = self._arb.act(self, game_state)
    if action not in _ACTION_TO_IDX:
        action = 'WAIT'
    try:
        safety = self._action_safety(game_state)
        feats = self._state_to_features(game_state, safety)
        dbg = getattr(self, '_last_search', None)
        decided = bool(getattr(self, '_search_decided', False))
        label = None
        if decided:
            # search/tactical-owned commit: the executed first-step IS
            # the improver's choice (bomb-plan path steps included)
            label = _ACTION_TO_IDX[action]
        elif dbg is not None and not dbg.get('exhausted') \
                and dbg.get('plans', 0) > 0 \
                and dbg.get('best_move_first') in _ACTION_TO_IDX:
            label = _ACTION_TO_IDX[dbg['best_move_first']]
        if label is not None:
            self._exit_feats.append(np.asarray(feats, dtype=np.float32))
            self._exit_labels.append(int(label))
            self._exit_acts.append(_ACTION_TO_IDX[action])
    except Exception:
        pass
    return action
