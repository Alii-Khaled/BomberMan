#!/usr/bin/env python3
"""E101 ARBITER-NG probe: package gates before any game.

  G0  dims: FEATURE_DIM 3566 = 12*17*17 + 98; AUG_PERMS bijective.
  G1  tensor augmentation consistency: board_tensor(transform_state(s))
      == transform_tensor(board_tensor(s), s) for all 8 symmetries.
  G2  scalar augmentation parity (ship property; <=1% allowance for the
      known 1/300 escape-mask asymmetry).
  G3  flat input finite; model forward shapes; zero-init pi uniform.
  G4  action_safety valid mask == spec.
  G5  latency: features < 3 ms, model batch1 < 2 ms (1 thread), act p99
      < 500 ms.
Usage: python3 scripts/probe_arbiter_ng.py [--trials N]
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch

torch.set_num_threads(1)
import agent_code.arbiter_ng.features as F
import agent_code.arbiter_ng.safety as SAFE
import agent_code.arbiter_ng.model as M
import agent_code.arbiter_ng.callbacks as WC

N = 60
for a in sys.argv[1:]:
    if a.startswith('--trials'):
        N = int(a.split('=')[1] if '=' in a else sys.argv[sys.argv.index(a) + 1])

fails = []
rng = np.random.default_rng(23)


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def mkstate():
    a = np.zeros((17, 17), dtype=int)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = -1
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                a[x, y] = -1
    for (x, y) in zip(*np.where(a == 0)):
        if rng.random() < 0.75:
            a[x, y] = 1
    for (x, y) in [(1, 1), (1, 15), (15, 1), (15, 15)]:
        for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1),
                         (x, y + 1)]:
            if a[xx, yy] == 1:
                a[xx, yy] = 0
    free = list(zip(*np.where(a == 0)))
    rng.shuffle(free)
    ag = [(f'a{i}', int(rng.integers(0, 5)), bool(rng.random() < 0.7),
           free[i]) for i in range(4)]
    bombs = [((int(free[4][0]), int(free[4][1])), int(rng.integers(0, 5)))]
    coins = [(int(free[5][0]), int(free[5][1]))]
    return {'round': 1, 'step': int(rng.integers(1, 390)), 'field': a,
            'bombs': bombs, 'coins': coins,
            'explosion_map': np.zeros_like(a), 'self': ag[0],
            'others': ag[1:]}


check('G0 FEATURE_DIM == 12*289+98', F.FEATURE_DIM == 12 * 17 * 17 + 98,
      str(F.FEATURE_DIM))
check('G0 AUG_PERMS bijective (8x98)',
      F.AUG_PERMS.shape == (8, 98)
      and all(len(set(r)) == 98 for r in F.AUG_PERMS))

bad_t = bad_s = 0
for _ in range(N):
    gs = mkstate()
    sf = SAFE.action_safety(gs)
    t0 = F.board_tensor(gs, sf)
    s0 = F.scalar_features(gs, sf)
    for s in range(8):
        gst = F.transform_state(gs, s)
        sft = SAFE.action_safety(gst)
        tt = F.board_tensor(gst, sft)
        if not np.allclose(tt, F.transform_tensor(t0, s), atol=1e-6):
            bad_t += 1
        ss = F.scalar_features(gst, sft)
        if not np.allclose(ss, F.apply_aug(s0, s), atol=1e-5):
            bad_s += 1
            break
check('G1 tensor augmentation exact (%d x8)' % N, bad_t == 0,
      '%d mismatches' % bad_t)
check('G2 scalar augmentation <= 1% (ship parity)', bad_s <= max(1, N // 100),
      '%d/%d' % (bad_s, N))

f = F.state_to_features(mkstate(), SAFE.action_safety(mkstate()))
check('G3 flat input finite/shape', f.shape == (F.FEATURE_DIM,)
      and np.isfinite(f).all())
model = M.build_model()
model.eval()
x = torch.zeros(1, F.FEATURE_DIM)
with torch.no_grad():
    pi, v = model(x)
check('G3 zero-init pi uniform + v 0',
      float(pi.std()) < 1e-6 and abs(float(v)) < 1e-6)

bad = []
for _ in range(N):
    gs = mkstate()
    sf = SAFE.action_safety(gs)
    arena = np.asarray(gs['field'])
    _, _, _bl, (x0, y0) = gs['self']
    bcells = set((int(b[0][0]), int(b[0][1])) for b in gs['bombs'])
    ocells = set((int(o[3][0]), int(o[3][1])) for o in gs['others'])
    for a, (dx, dy) in (('UP', (0, -1)), ('DOWN', (0, 1)),
                        ('LEFT', (-1, 0)), ('RIGHT', (1, 0))):
        nx, ny = x0 + dx, y0 + dy
        spec = (0 <= nx < 17 and 0 <= ny < 17 and arena[nx, ny] == 0
                and (nx, ny) not in bcells and (nx, ny) not in ocells)
        if bool(sf['valid'].get(a)) != bool(spec):
            bad.append((a, spec))
check('G4 valid mask == spec', not bad, str(bad[:3]))

lat = []
for _ in range(N):
    gs = mkstate()
    sf = SAFE.action_safety(gs)
    t0 = time.perf_counter()
    F.state_to_features(gs, sf)
    lat.append((time.perf_counter() - t0) * 1000)
feat_ms = float(np.median(lat))
lat = []
xs = torch.zeros(1, F.FEATURE_DIM)
with torch.no_grad():
    for _ in range(N):
        t0 = time.perf_counter()
        model(xs)
        lat.append((time.perf_counter() - t0) * 1000)
fwd_ms = float(np.median(lat))
print('      features %.2f ms | forward %.3f ms' % (feat_ms, fwd_ms))
check('G5 features < 3 ms', feat_ms < 3.0)
check('G5 model batch1 < 2 ms', fwd_ms < 2.0)

import types
o = types.SimpleNamespace()
o.logger = type('L', (), {'info': lambda *a, **k: None,
                          'warning': lambda *a, **k: None})()
o.model = model
o.train = False
o.coord_history = __import__('collections').deque([], 24)
o.bomb_history = __import__('collections').deque([], 5)
o.current_round = 0
o.flee_timer = 0
lat = []
for _ in range(max(20, N // 2)):
    gs = mkstate()
    t0 = time.perf_counter()
    WC.act(o, gs)
    lat.append((time.perf_counter() - t0) * 1000)
p99 = float(np.percentile(lat, 99))
print('      act p99 %.1f ms' % p99)
check('G5 act p99 < 500 ms', p99 < 500.0)

print()
if fails:
    print('NG PROBE FAILED:', ', '.join(fails))
    sys.exit(1)
print('NG PROBE OK')
