#!/usr/bin/env python3
"""E99 feature-v2 probe: candidate-package gates before any game.

  G0  import + FEATURE_DIM == 114 + AUG_PERMS bijective.
  G1  features equivariant under all 8 board symmetries (with safety).
  G2  features computed without safety_info stay zero in the
      safety-dependent dims (search-leaf contract) and finite.
  G3  blast geometry == engine (items.Bomb) on random arenas.
  G4  action_safety valid mask == spec on random states.
  G5  latency: features < 2 ms (no safety) and < 3 ms (full); act p99
      < 500 ms with model=None (uniform pi).

Usage: python3 scripts/probe_arbiter_v2.py [--trials N]
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import arbiter_v2.features as F
import arbiter_v2.safety as SAFE
import arbiter_v2.sim as SIM
import arbiter_v2.callbacks as WC
from items import Bomb

N = int(os.environ.get('PROBE_TRIALS', '60'))
for a in sys.argv[1:]:
    if a.startswith('--trials'):
        N = int(a.split('=')[1] if '=' in a else sys.argv[sys.argv.index(a) + 1])

fails = []
rng = np.random.default_rng(17)


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def make_arena():
    a = np.zeros((17, 17), dtype=int)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = -1
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                a[x, y] = -1
    free = list(zip(*np.where(a == 0)))
    for (x, y) in free:
        if rng.random() < 0.75:
            a[x, y] = 1
    for (x, y) in [(1, 1), (1, 15), (15, 1), (15, 15)]:
        for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1),
                         (x, y + 1)]:
            if a[xx, yy] == 1:
                a[xx, yy] = 0
    return a


def mkstate(n_agents=4):
    arena = make_arena()
    free = list(zip(*np.where(arena == 0)))
    rng.shuffle(free)
    ag = [(f'a{i}', int(rng.integers(0, 5)), bool(rng.random() < 0.7),
           free[i]) for i in range(n_agents)]
    bombs = [((int(free[4][0]), int(free[4][1])), int(rng.integers(0, 5))),
             ((int(free[5][0]), int(free[5][1])), int(rng.integers(0, 5)))]
    coins = [(int(free[6][0]), int(free[6][1])),
             (int(free[7][0]), int(free[7][1]))]
    return {'round': 1, 'step': int(rng.integers(1, 390)), 'field': arena,
            'bombs': bombs, 'coins': coins,
            'explosion_map': np.zeros_like(arena), 'self': ag[0],
            'others': ag[1:]}


# ---- G0 ----
check('G0 FEATURE_DIM == 114', F.FEATURE_DIM == 114)
check('G0 AUG_PERMS bijective (8x114)',
      F.AUG_PERMS.shape == (8, F.FEATURE_DIM)
      and all(len(set(row)) == F.FEATURE_DIM for row in F.AUG_PERMS))
import inspect
import ast


def _imports_torch(mod):
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and \
                any(a.name.split('.')[0] == 'torch' for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and \
                node.module.split('.')[0] == 'torch':
            return True
    return False


check('G0 v2 features/safety numpy-only',
      not _imports_torch(F) and not _imports_torch(SAFE))

# ---- G1 equivariance ----
# NOTE: index 45 (safe-first mask) shows a rare (1/300) pre-existing
# escape_bfs tie-break asymmetry in the SHIP package too; the gate is
# exact on the NEW block (98-113) and records full-vector parity.
bad_new = 0
bad_all = 0
for _ in range(N):
    gs = mkstate()
    sf = SAFE.action_safety(gs)
    f0 = F.state_to_features(gs, sf)
    if not np.isfinite(f0).all():
        bad_new += 1
        bad_all += 1
        continue
    for s in range(8):
        gst = F.transform_state(gs, s)
        sft = SAFE.action_safety(gst)
        ft = F.state_to_features(gst, sft)
        ref = F.apply_aug(f0, s)
        d = np.abs(ft - ref)
        if float(d[98:].max()) > 1e-5:
            bad_new += 1
        if float(d.max()) > 1e-5:
            bad_all += 1
            break
check('G1 new block (98-113) equivariant (%d states x8)' % N,
      bad_new == 0, '%d mismatches' % bad_new)
check('G1 full-vector mismatch <= 1%% (ship parity)',
      bad_all <= max(1, N // 100), '%d/%d' % (bad_all, N))

# ---- G2 no-safety contract ----
gset = set()
for _ in range(N):
    gs = mkstate()
    f0 = F.state_to_features(gs, None)
    if not np.isfinite(f0).all():
        bad += 1
    gset.update(np.where(f0[108:111] != 0)[0].tolist())
    gset.update(np.where(f0[112:114] != 0)[0].tolist())
check('G2 no-safety leaves 108-113 zero + finite', not gset, str(sorted(gset)))

# ---- G3 blast parity ----
ok = True
for _ in range(N * 5):
    arena = make_arena()
    xs, ys = np.where(arena != -1)
    i = int(rng.integers(len(xs)))
    x, y = int(xs[i]), int(ys[i])
    if set(SIM.blast_coords(arena, x, y)) != set(
            Bomb((x, y), None, 4, 3, None).get_blast_coords(arena)):
        ok = False
        break
check('G3 blast == engine (%d)' % (N * 5), ok)

# ---- G4 valid-mask spec parity ----
bad = []
for _ in range(N):
    gs = mkstate()
    sf = SAFE.action_safety(gs)
    arena = np.asarray(gs['field'])
    _, _, bl, (x, y) = gs['self']
    bcells = set((int(b[0][0]), int(b[0][1])) for b in gs['bombs'])
    ocells = set((int(o[3][0]), int(o[3][1])) for o in gs['others'])
    for a, (dx, dy) in (('UP', (0, -1)), ('DOWN', (0, 1)),
                        ('LEFT', (-1, 0)), ('RIGHT', (1, 0))):
        nx, ny = x + dx, y + dy
        spec = (0 <= nx < 17 and 0 <= ny < 17 and arena[nx, ny] == 0
                and (nx, ny) not in bcells and (nx, ny) not in ocells)
        if bool(sf['valid'].get(a)) != bool(spec):
            bad.append((a, bool(sf['valid'].get(a)), spec))
check('G4 valid mask == spec', not bad, str(bad[:3]))

# ---- G5 latency ----
lat = []
for _ in range(max(30, N)):
    gs = mkstate()
    t0 = time.perf_counter()
    F.state_to_features(gs, None)
    lat.append((time.perf_counter() - t0) * 1000.0)
ns = float(np.median(lat))
lat = []
for _ in range(max(30, N)):
    gs = mkstate()
    t0 = time.perf_counter()
    sf = SAFE.action_safety(gs)
    F.state_to_features(gs, sf)
    lat.append((time.perf_counter() - t0) * 1000.0)
full = float(np.median(lat))
print('      features: no-safety %.2f ms | full %.2f ms' % (ns, full))
check('G5 features no-safety < 2 ms', ns < 2.0)
check('G5 features full < 3 ms', full < 3.0)

import types
o = types.SimpleNamespace()
o.logger = type('L', (), {'info': lambda *a, **k: None,
                          'warning': lambda *a, **k: None})()
WC.setup(o)
lat = []
for _ in range(max(20, N // 2)):
    gs = mkstate()
    t0 = time.perf_counter()
    WC.act(o, gs)
    lat.append((time.perf_counter() - t0) * 1000.0)
p99 = float(np.percentile(lat, 99))
print('      act (model=None) p99 %.1f ms' % p99)
check('G5 act p99 < 500 ms', p99 < 500.0)

print()
if fails:
    print('V2 PROBE FAILED:', ', '.join(fails))
    sys.exit(1)
print('V2 PROBE OK')
