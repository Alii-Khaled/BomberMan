"""E114 solo-DAgger recorder (training-only, never ships).

The SHIP policy (agent_code.Harvy, env flags apply) plays the game;
at every step where no opponent is alive the state is recorded with the
coin_collector_agent (or ARBITER_SOLO_LABEL teacher) action as the label.
Output is apex-format npz per round (img 12x17x17 uint8, sc float32[16],
act uint8) consumed by scripts/pretrain_arbiter_ng.py with
--dirs results/apex_demos,results/apex_ng_demos,<APEX_DEMO_OUT>.

Purpose: the BC prior is OOD in solo states (E112: the agent freezes);
E112's inference fix (solo margin + loop escalation) handles the
mechanics, this corpus teaches the prior the farming policy directly so
2-opponent/mixed states inherit it too.

Env:
  APEX_DEMO_OUT   output root (default results/apex_solo_dagger)
  SOLO_LABEL      label teacher (default coin_collector_agent)
  SOLO_DAGGER_MIN_STEP  skip the opening steps (default 0)
"""
import importlib
import logging
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

TEACHER = os.environ.get('SOLO_LABEL', 'coin_collector_agent')
OUT = os.environ.get('APEX_DEMO_OUT', os.path.join(_REPO, 'results',
                                                   'apex_solo_dagger'))
MIN_STEP = int(os.environ.get('SOLO_DAGGER_MIN_STEP', '0'))

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
    self.logger.info('solo_dagger setup (label=%s)' % TEACHER)
    # acting policy: the ship itself (same self object, exact behavior)
    import agent_code.Harvy.callbacks as arb
    arb.setup(self)
    self._dag_arb = arb
    # labeling policy: private namespace so its histories evolve along
    # the observed solo trajectory (observer-only, never acts)
    self._lab = types.SimpleNamespace()
    self._lab.logger = logging.getLogger('solo_dagger.%s' % TEACHER)
    self._lab.train = False
    mod = importlib.import_module('agent_code.%s.callbacks' % TEACHER)
    mod.setup(self._lab)
    self._lab_mod = mod
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
    self._n_skipped = 0


def _flush(self):
    if not self._buf:
        return
    try:
        imgs = np.stack([b[0] for b in self._buf])
        scs = np.stack([b[1] for b in self._buf])
        acts = np.asarray([b[2] for b in self._buf], dtype=np.uint8)
        tmp = os.path.join(self._outdir, 'round_%06d.npz.part' % self._next)
        dst = os.path.join(self._outdir, 'round_%06d.npz' % self._next)
        with open(tmp, 'wb') as _fh:
            np.savez_compressed(_fh,
                                img=(imgs * 4.0 + 0.5).astype(np.uint8),
                                sc=scs.astype(np.float32), act=acts)
        os.replace(tmp, dst)
        self._next += 1
    except Exception as ex:
        try:
            self.logger.warning('solo_dagger flush failed: %s' % ex)
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
    # ship acts (env flags apply exactly as in play)
    a = self._dag_arb.act(self, game_state)
    # record only solo states, labeled by the teacher
    try:
        step = int(game_state.get('step', 0))
        if step >= MIN_STEP and not (game_state.get('others') or []):
            from agent_code.Harvy.safety import action_safety
            lab_a = self._lab_mod.act(self._lab, game_state)
            sf = action_safety(game_state)
            if lab_a in A2I and sf.get('valid', {}).get(lab_a, False):
                img, sc = _apex_features(game_state)
                if img.shape == (12, 17, 17) and sc.shape == (16,):
                    self._buf.append((img, sc, A2I[lab_a]))
            else:
                self._n_skipped += 1
    except Exception:
        pass
    return a
