"""Apex demo recorder (B3): delegating wrapper around a teacher policy.

TEACHER env: warden_v1 | sentinel | overlord (default warden_v1).
Records per step: img uint8 (12x17x17, x4 scale, LOSSLESS per E48),
sc float32 (16: overlord-8 + apex-extras-8), act uint8 (teacher action).
One npz per round -> APEX_DEMO_OUT/<teacher>/round_%06d.npz.
Deleted after the demo collection (temporary harness, E14b pattern).

The teacher acts every game (natural teacher distribution); the recorder
only observes. Teacher state lives on a private namespace (histories
evolve there, bit-identical to solo teacher play).
"""
import importlib
import logging
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HERE = os.path.dirname(os.path.abspath(__file__))
# NOTE (E51): SequentialAgentBackend chdirs into agent_code/<name>/ around
# EVERY callback (agents.py:304) — resolve the out dir from __file__ so
# recordings land in results/ regardless of cwd (E27 chdir lesson, third
# occurrence: 297 files initially landed under agent_code/apex_teacher/).
# NOTE (E51b): act() flushes on round-change; train.py:end_of_round()
# flushes the final round (without it the last round per invocation was
# silently dropped: 150 played -> 149 files).
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
TEACHER = os.environ.get('APEX_TEACHER', 'warden_v1')
OUT = os.environ.get('APEX_DEMO_OUT', os.path.join(_REPO, 'results', 'apex_demos'))

ORDER = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
A2I = {a: i for i, a in enumerate(ORDER)}


def _apex_features(game_state):
    from agent_code.apex.safety import action_safety
    from agent_code.apex.features_cnn import state_to_tensor
    from agent_code.apex.model import scalars_from_state
    from agent_code.apex.features_extra import extra_scalars
    try:
        s = action_safety(game_state)
    except Exception:
        s = None
    img = state_to_tensor(game_state, s).astype(np.float32)
    sc8 = scalars_from_state(game_state, s).astype(np.float32)
    ex8 = extra_scalars(game_state).astype(np.float32)
    sc = np.concatenate([sc8, ex8]).astype(np.float32)
    return img, sc


def setup(self):
    self.logger.info('apex_teacher setup (teacher=%s)' % TEACHER)
    self._t = types.SimpleNamespace()
    self._t.logger = logging.getLogger('apex_teacher.%s' % TEACHER)
    self._t.train = False
    mod = importlib.import_module('agent_code.%s.callbacks' % TEACHER)
    mod.setup(self._t)
    self._teacher_act = mod.act
    self._buf = []
    self._cur_round = None
    self._outdir = os.path.join(OUT, TEACHER)
    os.makedirs(self._outdir, exist_ok=True)
    try:
        existing = [f for f in os.listdir(self._outdir)
                    if f.startswith('round_') and f.endswith('.npz')]
        self._next = max([int(f[6:12]) for f in existing] + [0]) + 1
    except Exception:
        self._next = 1


def _flush(self):
    if not self._buf:
        return
    try:
        imgs = np.stack([b[0] for b in self._buf])
        scs = np.stack([b[1] for b in self._buf])
        acts = np.asarray([b[2] for b in self._buf], dtype=np.uint8)
        tmp = os.path.join(self._outdir, 'round_%06d.npz.part' % self._next)
        dst = os.path.join(self._outdir, 'round_%06d.npz' % self._next)
        # NOTE: np.savez_compressed appends '.npz' to plain string paths,
        # so write through a file handle (no extension mangling).
        with open(tmp, 'wb') as _fh:
            np.savez_compressed(_fh,
                                img=(imgs * 4.0 + 0.5).astype(np.uint8),
                                sc=scs.astype(np.float32), act=acts)
        os.replace(tmp, dst)
        self._next += 1
    except Exception as ex:
        try:
            self.logger.warning('apex_teacher flush failed: %s' % ex)
        except Exception:
            pass
    self._buf = []


def act(self, game_state):
    try:
        rnd = int(game_state.get('round', 0))
    except Exception:
        rnd = 0
    if self._cur_round is None:
        self._cur_round = rnd
    if rnd != self._cur_round:
        _flush(self)
        self._cur_round = rnd
    a = self._teacher_act(self._t, game_state)
    try:
        img, sc = _apex_features(game_state)
        if a in A2I and img.shape == (12, 17, 17) and sc.shape == (16,):
            self._buf.append((img, sc, A2I[a]))
    except Exception:
        pass
    return a
