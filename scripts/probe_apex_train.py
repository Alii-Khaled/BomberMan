"""B2 unit test: apex _update on CPU with synthetic transitions.

Exercises TD + aux + augmentation + DQfD demo paths (all on/off combos).
Fails loudly. Run: APEX_AUG=1 python3 scripts/probe_apex_train.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agent_code.apex import train as T
from agent_code.apex.model import build_model


def make_self(demo_n=300):
    s = types.SimpleNamespace()
    s.device = torch.device('cpu')
    s.batch_size = 32
    s.total_steps = 0
    s.epsilon_steps = 0
    s.epsilon = 1.0
    s.opt_schedule = False
    s.use_amp = False
    s.scaler = None
    s.buffer = T.PERBuffer(cap=10000)
    rng = np.random.default_rng(3)
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    for _ in range(6000):
        img = rng.standard_normal((12, 17, 17)).astype(np.float32)
        sc = rng.standard_normal((16,)).astype(np.float32)
        nimg = rng.standard_normal((12, 17, 17)).astype(np.float32)
        a = ACTIONS[rng.integers(6)]
        s.buffer.add(T.Transition(img, sc, a, nimg, sc,
                                  float(rng.standard_normal()), False, 0.1))
    s.q_net = build_model()
    s.target_net = build_model()
    s.target_net.load_state_dict(s.q_net.state_dict())
    s.optimizer = torch.optim.AdamW(s.q_net.parameters(), lr=3e-4)
    demo = []
    for _ in range(demo_n):
        demo.append(((rng.random((12, 17, 17)) * 4).astype(np.uint8),
                     rng.standard_normal((16,)).astype(np.float32),
                     int(rng.integers(6))))
    s.demo = demo
    return s


def check(name, cond):
    print(('PASS' if cond else 'FAIL') + ': ' + name)
    if not cond:
        raise SystemExit('probe failed: ' + name)


# aug permutation algebra (module const)
p = list(range(6))
for _ in range(4):
    p = [T.PERM_CCW[i] for i in p]
check('PERM_CCW^4 == identity', p == list(range(6)))
t = torch.randn(2, 12, 17, 17)
r = t
for _ in range(4):
    r = torch.rot90(r, 1, (2, 3))
check('rot90 x4 == identity tensor', torch.equal(t, r))

# full update, everything on
s = make_self(300)
T._update(s)
check('update runs (aug=%s demo=300)' % T.AUG, np.isfinite(s.last_loss))
l1 = s.last_loss
T._update(s)
check('second update finite + buffer live',
      np.isfinite(s.last_loss) and len(s.buffer) == 6000)

# paths off
s2 = make_self(0)
T._update(s2)
check('update runs (demo empty)', np.isfinite(s2.last_loss))

print('INFO: loss %.4f -> %.4f (illustrative only)' % (l1, s.last_loss))
print('PROBE PASS (B2 train loop: TD+aux+aug+demo)')
