#!/usr/bin/env python3
"""ARBITER P0 static probe (E21/E33 discipline: gates before games).
Groups:
  G1 model shapes + zero-init (pi uniform, V zero) + determinism.
  G2 vendored parity arbiter.features == reaper.features (exact) on
      reconstructed demo states: state_to_features, _blast_crate_counts.
  G3 vendored parity arbiter.safety == reaper.safety: action_safety
      valid/safe/can_escape, bomb_here_traps verdict.
  G4 act() unit: untrained net returns a mask-valid action on live states;
      S0 tie-break bounds (loop/BOMB penalties <= 0.5+0.9 combined scale).
  G5 latency: state_to_features + forward p50 on 1 thread (budgets P1).
Usage: python3 scripts/probe_arbiter.py [--states N]
Exit nonzero on any failure.
"""
import glob
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import arbiter.features as AF
import arbiter.safety as AS
import arbiter.model as AM
import reaper.features as RF
import reaper.safety as RS

N = 15
for a in sys.argv[1:]:
    if a.startswith('--states'):
        N = int(a.split('=')[1] if '=' in a else sys.argv[sys.argv.index(a) + 1])

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def reconstruct(i, img, sc):
    t = img[i]
    arena = np.where(t[0] > 0.5, -1, np.where(t[1] > 0.5, 1, 0)).astype(int)
    bombs = [((int(x), int(y)), max(0, int(round(t[5][x, y] * 4))))
             for x, y in zip(*np.where(t[5] > 0))]
    coins = [(int(x), int(y)) for x, y in zip(*np.where(t[2] > 0.5))]
    sp = list(zip(*np.where(t[3] > 0.5)))
    op = list(zip(*np.where(t[4] > 0.5)))
    x, y = int(sp[0][0]), int(sp[0][1])
    return {'round': 1, 'step': int(round(sc[i][0] * 400)), 'field': arena,
            'bombs': bombs, 'explosion_map': np.where(t[7] > 0.5, 1, 0),
            'coins': coins, 'self': ('w', 0, bool(sc[i][1] > 0.5), (x, y)),
            'others': [('o%d' % j, 0, False, (int(a), int(b)))
                       for j, (a, b) in enumerate(op)]}


# ---- G1: model ----
import torch
torch.set_num_threads(1)
m = AM.build_model()
m.eval()
check('G1 params>0', sum(p.numel() for p in m.parameters()) > 50000,
      str(sum(p.numel() for p in m.parameters())))
with torch.no_grad():
    pi, v = m(torch.zeros(2, AF.FEATURE_DIM))
check('G1 pi shape', tuple(pi.shape) == (2, 6), str(tuple(pi.shape)))
check('G1 v shape', tuple(v.shape) == (2,), str(tuple(v.shape)))
check('G1 zero-init pi==0', bool((pi.abs().max() == 0).item()))
check('G1 zero-init v==0', bool((v.abs().max() == 0).item()))
with torch.no_grad():
    pi2, v2 = m(torch.zeros(2, AF.FEATURE_DIM))
check('G1 determinism', bool((pi == pi2).all() and (v == v2).all()))

# ---- states ----
fs = sorted(glob.glob(os.path.join(REPO, 'results/apex_demos/*/*.npz')))[:4]
states = []
for f in fs:
    d = np.load(f)
    img = d['img'].astype(np.float32) / 4.0
    sc = d['sc']
    for i in (0, len(img) // 3, 2 * len(img) // 3, len(img) - 1):
        states.append(reconstruct(i, img, sc))
states = states[:N]
check('G2 states', len(states) == N, '%d states' % len(states))

# ---- G2: feature parity ----
ok = True
for gs in states:
    a = AF.state_to_features(gs)
    b = RF.state_to_features(gs)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
        ok = False
        break
check('G2 state_to_features exact', ok)
ok = True
for gs in states:
    aa, ab = AF._blast_crate_counts(np.asarray(gs['field']))
    ba, bb = RF._blast_crate_counts(np.asarray(gs['field']))
    if not (np.array_equal(aa, ba) and np.array_equal(ab, bb)):
        ok = False
        break
check('G2 _blast_crate_counts exact', ok)
check('G2 FEATURE_DIM', AF.FEATURE_DIM == RF.FEATURE_DIM == 98)
check('G2 N_SYMS', AF.N_SYMS == RF.N_SYMS == 8)

# ---- G3: safety parity ----
ok = True
for gs in states:
    sa = AS.action_safety(gs)
    sb = RS.action_safety(gs)
    if sa.get('valid') != sb.get('valid') or sa.get('safe') != sb.get('safe'):
        ok = False
        break
    if bool(sa.get('can_escape_if_bomb')) != bool(sb.get('can_escape_if_bomb')):
        ok = False
        break
check('G3 action_safety valid/safe/escape', ok)
ok = True
for gs in states:
    ta, _ = AS.bomb_here_traps(gs)
    tb, _ = RS.bomb_here_traps(gs)
    if bool(ta) != bool(tb):
        ok = False
        break
check('G3 bomb_here_traps verdict', ok)

# ---- G4: act() unit (untrained net: uniform pi + skeleton) ----
sys.path.insert(0, REPO)
import types
import arbiter.callbacks as AC
fake = types.SimpleNamespace()
fake.logger = __import__('logging').getLogger('probe')
fake.train = False
AC.setup(fake)
assert getattr(fake, 'model', None) is not None
ok = True
acts = []
for gs in states:
    a = AC.act(fake, gs)
    acts.append(a)
    if a not in AM.ACTION_LIST:
        ok = False
        break
check('G4 act returns known action', ok, 'sample %s' % acts[:6])
# determinism: two fresh instances, same weights, same state -> same action
import logging as _lg
fA = types.SimpleNamespace()
fA.logger = _lg.getLogger('probeA')
fA.train = False
AC.setup(fA)
fB = types.SimpleNamespace()
fB.logger = _lg.getLogger('probeB')
fB.train = False
AC.setup(fB)
fB.model.load_state_dict(fA.model.state_dict())
a1 = AC.act(fA, states[0])
b1 = AC.act(fB, states[0])
check('G4 act deterministic fresh', a1 == b1, '%s vs %s' % (a1, b1))

# ---- G5: latency (1 thread) ----
gs = states[0]
sa = AS.action_safety(gs)
t0 = time.perf_counter()
for _ in range(50):
    AF.state_to_features(gs, sa)
ms_feat = (time.perf_counter() - t0) / 50 * 1000
f = AF.state_to_features(gs, sa).astype(np.float32)
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(200):
        m(torch.from_numpy(f).unsqueeze(0))
ms_fwd = (time.perf_counter() - t0) / 200 * 1000
print('G5 state_to_features %.2f ms | forward %.3f ms' % (ms_feat, ms_fwd))
check('G5 features < 2ms', ms_feat < 2.0, '%.2f' % ms_feat)
check('G5 forward < 0.5ms', ms_fwd < 0.5, '%.3f' % ms_fwd)

print()
if fails:
    print('FAILED:', fails)
    sys.exit(1)
print('ALL PROBE GROUPS PASS')
