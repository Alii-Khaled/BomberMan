# ARBITER-RL (E100): on-policy masked-softmax sampling for policy-gradient
# fine-tuning. Active only when train.py sets self._rl (default ship path
# untouched).
import os

import numpy as np


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


TEMP = _env_float('ARBITER_RL_TEMP', 1.0)
NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def sample_action(self, pi, valid, safe, must_flee, flee_locked, feats):
    """Sample a move from masked softmax(pi/T); record the training trace.

    Admissible = valid, and safe under flee semantics (ship S0 filter).
    Records (round, step, feats, allowed_idx, chosen_j) for train.py,
    which recomputes the log-probs with gradients at round end.
    """
    threat = bool(must_flee or flee_locked)
    idx = []
    for i, a in enumerate(NAMES):
        if a == 'BOMB':
            continue
        if not valid.get(a):
            continue
        if threat and not safe.get(a):
            continue
        idx.append(i)
    if not idx:
        return None
    logits = np.asarray(pi, dtype=np.float64)[idx] / max(TEMP, 1e-3)
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    rng = np.random.default_rng(
        (int(getattr(self, '_seed', 0)) * 1000003
         ^ int(getattr(self, 'current_round', 0)) * 9176
         ^ int(getattr(self, '_rl_step', 0)) * 7919) & 0xffffffff)
    j = int(rng.choice(len(idx), p=p))
    try:
        self._rl_trace.append((int(self.current_round),
                               int(getattr(self, '_rl_step', 0)),
                               np.asarray(feats, dtype=np.float32).copy(),
                               tuple(int(i) for i in idx), int(j)))
    except Exception:
        pass
    return NAMES[idx[j]]
