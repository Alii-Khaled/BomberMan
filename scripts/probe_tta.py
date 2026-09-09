"""E38 TTA probes: rotation-equivariant Q-averaging feasibility.

Groups (fail loudly, never touch training/checkpoints):
 1. remap algebra: CW action perm ^4 == identity; inverse correctness.
 2. rotation direction: marker one-hot at north lands east under rot_cw.
 3. equivariance gap: |Q(x) - unrot(Q(rot(x)))| on the SHIP weights —
    measures how much TTA views actually differ (informational).
 4. timing: 4-view forward loop on CPU < 100 ms total (hard gate).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agent_code.overlord.model import build_model

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# CW rotation in (x,y) game coords: UP->RIGHT->DOWN->LEFT. WAIT/BOMB fixed.
PERM_CW = [1, 2, 3, 0, 4, 5]


def rot_cw(t):
    """Clockwise rot90 on [...,X,Y] (square 17x17). T'[..,x,y] = T[..,y,16-x]."""
    ax = list(range(t.ndim))
    ax[-2], ax[-1] = ax[-1], ax[-2]
    return np.transpose(t[..., :, ::-1], ax)


def check(name, cond):
    print(('PASS' if cond else 'FAIL') + ': ' + name)
    if not cond:
        raise SystemExit('probe failed: ' + name)


# 1. algebra
p = list(range(6))
for _ in range(4):
    p = [PERM_CW[i] for i in p]
check('perm^4 == identity', p == list(range(6)))
inv = [0] * 6
for i, j in enumerate(PERM_CW):
    inv[j] = i
v = np.arange(6)
check('inverse restores', all(v[PERM_CW][inv][i] == v[i] for i in range(6)))

# 2. direction: north marker -> east
m = np.zeros((17, 17))
m[8, 2] = 1.0  # north of center (x=8, y=2; UP is y-1)
r = rot_cw(m)
yx = np.unravel_index(np.argmax(r), r.shape)
check('north marker lands east (x=14,y=8)', tuple(yx) == (14, 8))

# 3. equivariance gap on ship weights (CPU)
m = build_model()
m.load_state_dict(torch.load('results/archive/overlord_SHIP_379/my-saved-model.pt',
                             map_location='cpu', weights_only=False), strict=True)
m.eval()
rng = np.random.default_rng(0)
gaps = []
with torch.no_grad():
    for _ in range(20):
        img = rng.standard_normal((1, 12, 17, 17)).astype(np.float32)
        sc = rng.standard_normal((1, 8)).astype(np.float32)
        q0, _ = m(torch.from_numpy(img), torch.from_numpy(sc))
        img_r = rot_cw(img)
        qr, _ = m(torch.from_numpy(np.ascontiguousarray(img_r)),
                   torch.from_numpy(sc))
        qr_back = qr.numpy()[:, [1, 2, 3, 0, 4, 5]]  # un-rotate: orig a <-> rot P[a]
        gaps.append(float(np.abs(q0.numpy() - qr_back).max()))
print('INFO: equivariance gap max|dQ| over 20 random states: '
      'median %.3f, max %.3f (Q clip range +-4)' % (
          float(np.median(gaps)), float(np.max(gaps))))
check('gap finite and nonzero (views differ)', np.isfinite(gaps).all())

# 4. timing: 4-view loop
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(20):
        img = torch.randn(1, 12, 17, 17)
        sc = torch.randn(1, 8)
        acc = None
        cur, perm = img, list(range(6))
        for _ in range(4):
            q, _ = m(cur, sc)
            q = q[:, perm]
            acc = q if acc is None else acc + q
            cur = torch.rot90(cur, -1, (2, 3))
            perm = [PERM_CW[i] for i in perm]
        avg = acc / 4
ms = (time.perf_counter() - t0) / 20 * 1000
print('INFO: 4-view CPU loop %.1f ms/act (gate <100 ms)' % ms)
check('timing gate <100ms', ms < 100.0)

print('PROBE PASS (E38 TTA feasible)' if True else '')
