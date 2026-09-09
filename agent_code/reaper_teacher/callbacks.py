"""Reaper teacher recorder: delegates act() to a teacher agent and records
(features, action) samples for BC pretraining. Training-time only — never
ships to the tournament.

Teacher selection: TEACHER env = warden | sentinel | overlord (default
warden). The recorder reuses the teacher's own ``setup``/``act`` on the
same self object, so delegation is exact.

DAgger mode (REAPER_DAGGER=1): the STUDENT (reaper with current weights)
acts, and the teacher only labels the visited state. This puts teacher
supervision on the student's own state distribution, fixing the
off-distribution blindness of pure teacher demos. The recorder's train.py
saves (features, teacher_action) in this mode. Student weights come from
REAPER_STUDENT_PT (defaults to agent_code/reaper/my-saved-model.pt).
"""
import os
import sys
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TEACHER = os.environ.get('TEACHER', 'warden').strip().lower()
_DAGGER = os.environ.get('REAPER_DAGGER', '0') == '1'
_STUDENT_PT = os.environ.get('REAPER_STUDENT_PT', '').strip()
_ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
_ACTION_TO_IDX = {a: i for i, a in enumerate(_ACTION_LIST)}


def setup(self):
    np.random.seed()
    try:
        from agent_code.reaper.features import state_to_features
        from agent_code.reaper.safety import action_safety
        self._state_to_features = state_to_features
        self._action_safety = action_safety
    except Exception as ex:
        raise RuntimeError(f'reaper_teacher cannot import reaper features: {ex}')
    module = None
    if _TEACHER == 'warden':
        import agent_code.warden_v1.callbacks as module
    elif _TEACHER == 'sentinel':
        import agent_code.sentinel.callbacks as module
    elif _TEACHER == 'overlord':
        import agent_code.overlord.callbacks as module
    else:
        raise RuntimeError(f'unknown TEACHER={_TEACHER}')
    self._teacher = module
    self._teacher.setup(self)
    self._dagger = _DAGGER
    self._dagger_label = None
    self._student_round = 0
    if self._dagger:
        import agent_code.reaper.callbacks as student
        self._student = student
        self._student.setup(self)
        # Point the student at the requested weights (default: whatever
        # reaper.setup loaded from my-saved-model.pt).
        if _STUDENT_PT and os.path.isfile(_STUDENT_PT):
            try:
                import torch as _t
                obj = _t.load(_STUDENT_PT, map_location='cpu',
                              weights_only=True)
                sd = obj.get('state_dict', obj) \
                    if isinstance(obj, dict) else obj
                if isinstance(sd, dict) and 'q_net' in sd:
                    sd = sd['q_net']
                self.model.load_state_dict(sd, strict=False)
                self.model.eval()
                self.logger.info(f'reaper_teacher dagger student={_STUDENT_PT}')
            except Exception as ex:
                self.logger.warning(f'dagger student load failed: {ex}')
        # Deterministic student policy (no epsilon exploration): DAgger
        # labels the student's own greedy distribution.
        try:
            self.epsilon = 0.0
        except Exception:
            pass
        self.logger.info(f'reaper_teacher DAGGER on (student=reaper, '
                         f'teacher={_TEACHER})')
    self._own_bomb = None
    self._last_action = 'WAIT'
    if not self._dagger:
        self.logger.info(f'reaper_teacher delegating to {_TEACHER}')


def act(self, game_state):
    if getattr(self, '_dagger', False):
        # Teacher FIRST so its round-reset logic (keyed on the shared
        # self.current_round) runs correctly; the student's reset is then
        # handled manually below with its own tracker, since the teacher
        # has already advanced self.current_round.
        try:
            teacher_action = self._teacher.act(self, game_state)
        except Exception:
            teacher_action = None
        try:
            rnd = int(game_state.get('round', 0))
        except Exception:
            rnd = 0
        if rnd != getattr(self, '_student_round', 0):
            self._student_round = rnd
            self.coord_history = deque([], 24)
            self.bomb_history = deque([], 5)
            self.flee_timer = 0
            self.own_bomb = None
            self.current_round = rnd
        student_action = self._student.act(self, game_state)
        if teacher_action is None:
            teacher_action = student_action
        self._dagger_label = teacher_action
        self._last_action = student_action
        return student_action
    action = self._teacher.act(self, game_state)
    self._last_action = action
    return action
