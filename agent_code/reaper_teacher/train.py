"""Reaper_teacher training callbacks: record (features, action) demos.

game_events_occurred gets the state the agent acted on (old_game_state)
and its action; features are computed with reaper's feature builder plus
the recorder's own-bomb tracker. Rounds flush to npz files:
  <REAPER_DEMO_DIR>/round_%06d.npz  {feats: float32[N,98], acts: uint8[N]}

In DAGGER mode (REAPER_DAGGER=1, see callbacks.py) the STUDENT acts but
the recorded action is the TEACHER's label for that state — consume-and-
clear the label so a skipped step (death) can never leak a stale label
into the next step.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
_ACTION_TO_IDX = {a: i for i, a in enumerate(_ACTION_LIST)}


def _demo_dir():
    d = os.environ.get('REAPER_DEMO_DIR', '').strip()
    if not d:
        teach = os.environ.get('TEACHER', 'warden_v2').strip().lower()
        d = os.path.join(_ROOT, 'results', 'demos', teach)
    if not os.path.isabs(d):
        d = os.path.join(_ROOT, d)
    os.makedirs(d, exist_ok=True)
    return d


def setup_training(self):
    from agent_code.reaper.features import state_to_features
    from agent_code.reaper.safety import action_safety
    self._state_to_features = state_to_features
    self._action_safety = action_safety
    self._own_bomb = None
    self._round_feats = []
    self._round_acts = []
    self._round_id = 0
    self._demo_dir = _demo_dir()
    try:
        # Resume-safe: continue numbering after existing files so a
        # restarted collection appends instead of silently overwriting.
        _ex = [f for f in os.listdir(self._demo_dir)
               if f.startswith('round_') and f.endswith('.npz')]
        _nums = [int(f[6:12]) for f in _ex if f[6:12].isdigit()]
        if _nums:
            self._round_id = max(_nums)
    except Exception:
        pass
    self.logger.info(f'reaper_teacher recording demos to {self._demo_dir}')


def _consume_label(self, fallback):
    """Teacher label for the current step in DAGGER mode, else fallback.

    Consume-and-clear: a step that never reaches game_events_occurred
    (death) must not leak its label into a later step.
    """
    try:
        if getattr(self, '_dagger', False):
            label = getattr(self, '_dagger_label', None)
            self._dagger_label = None
            if label in _ACTION_TO_IDX:
                return label
    except Exception:
        pass
    return fallback


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return
    try:
        safety = self._action_safety(old_game_state)
        feats = self._state_to_features(old_game_state, safety, self._own_bomb)
        self._round_feats.append(feats.astype(np.float32))
        self._round_acts.append(
            _ACTION_TO_IDX[_consume_label(self, self_action)])
    except Exception:
        pass
    try:
        if self_action == 'BOMB':
            _, _, _, (x, y) = old_game_state['self']
            self._own_bomb = (int(x), int(y))
        else:
            _, _, bl, _ = old_game_state['self']
            if not bl:
                nbl = new_game_state['self'][2] if new_game_state is not None else True
                if nbl:
                    self._own_bomb = None
    except Exception:
        pass


def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is not None:
        try:
            import events as e
            _survived = e.SURVIVED_ROUND in (events or [])
        except Exception:
            _survived = False
        if getattr(self, '_dagger', False) and _survived:
            # Survived: game_events_occurred already recorded this exact
            # state with the teacher label — pushing again here would store
            # a CONFLICTING (student-action) duplicate. Skip. (The died
            # case still pushes below: the fatal step never reached
            # game_events_occurred, and its teacher label is waiting.)
            pass
        else:
            try:
                safety = self._action_safety(last_game_state)
                feats = self._state_to_features(last_game_state, safety, self._own_bomb)
                self._round_feats.append(feats.astype(np.float32))
                self._round_acts.append(
                    _ACTION_TO_IDX[_consume_label(self, last_action)])
            except Exception:
                pass
    n = len(self._round_feats)
    if n:
        self._round_id += 1
        path = os.path.join(self._demo_dir, f'round_{self._round_id:06d}.npz')
        try:
            np.savez_compressed(
                path,
                feats=np.asarray(self._round_feats, dtype=np.float32),
                acts=np.asarray(self._round_acts, dtype=np.uint8),
            )
            self.logger.info(f'reaper_teacher saved {n} samples to {path}')
        except Exception as ex:
            self.logger.warning(f'reaper_teacher save failed: {ex}')
    self._round_feats = []
    self._round_acts = []
    self._own_bomb = None
