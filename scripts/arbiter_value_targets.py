#!/usr/bin/env python3
"""E99 Phase B: exact-outcome V targets from the search's own rollouts.

For sampled states (apex demos by default) roll the exact simulator with
the SAME continuation/opponent model the search uses at leaves
(`score_plan` with an empty prefix), then label V with the realized score
margin delta (clipped +-10). This attacks the E66 V-null at its root:
the legacy label (teacher margin-to-go at round end) has ~2.6 std noise;
these targets are short-horizon and model-consistent.

Output cache (compatible with pretrain_arbiter.py, teacher id 6):
  feats (N, D_v2), feats98 (N, 98 or NaN), acts (N,), vlabel (N,),
  teacher (N,), round (N,), is_val (N,)
Use with: ARBITER_V_TEACHERS="6" (pi teachers should exclude 6).

Usage: python3 scripts/arbiter_value_targets.py --pkg=arbiter_v2 \
         --out results/arbiter_v2_vtargets.npz [--horizon 10] [--stride 1]
         [--limit N] [--base-cache results/arbiter_v2_cache.npz]
"""
import glob
import importlib
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))

import numpy as np

PKG = os.environ.get('ARBITER_PKG', 'arbiter_v2')
OUT = os.path.join(REPO, 'results', 'arbiter_v2_vtargets.npz')
HORIZON = 10
STRIDE = 1
LIMIT = 0
BASE = None
for a in sys.argv[1:]:
    if a.startswith('--pkg='):
        PKG = a.split('=', 1)[1]
    elif a.startswith('--out='):
        OUT = a.split('=', 1)[1]
    elif a.startswith('--horizon='):
        HORIZON = int(a.split('=', 1)[1])
    elif a.startswith('--stride='):
        STRIDE = int(a.split('=', 1)[1])
    elif a.startswith('--limit='):
        LIMIT = int(a.split('=', 1)[1])
    elif a.startswith('--base-cache='):
        BASE = a.split('=', 1)[1]

SIM = importlib.import_module(PKG + '.sim')
SR = importlib.import_module(PKG + '.search')
AF = importlib.import_module(PKG + '.features')
AF98 = importlib.import_module('arbiter.features')


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


def main():
    files = sorted(glob.glob(os.path.join(REPO, 'results/apex_demos/*/*.npz')))
    print('apex files: %d | pkg=%s horizon=%d stride=%d limit=%d'
          % (len(files), PKG, HORIZON, STRIDE, LIMIT or 0))
    F, F98, V, A, T, R = [], [], [], [], [], []
    rnd = 0
    t0 = time.time()
    for fi, f in enumerate(files):
        d = np.load(f)
        img = d['img'].astype(np.float32) / 4.0
        sc = d['sc'].astype(np.float32)
        n = len(d['act'])
        for i in range(0, n, STRIDE):
            try:
                gs = reconstruct(i, img, sc)
                st0 = SIM.from_game_state(gs)
                m0 = SIM.margin(st0)
                payload = {'prefix': []}
                _pay, st_end = SR.score_plan(st0, payload, horizon=HORIZON,
                                             seed=(rnd * 131 + i))
                vlab = float(np.clip(SIM.margin(st_end) - m0, -10.0, 10.0))
                fv2 = AF.state_to_features(gs, None)
                f98 = AF98.state_to_features(gs, None)
            except Exception:
                continue
            F.append(np.asarray(fv2, dtype=np.float32))
            F98.append(np.asarray(f98, dtype=np.float32))
            V.append(np.float32(vlab))
            A.append(np.uint8(0))
            T.append(np.uint8(6))
            R.append(np.int32(rnd))
            if LIMIT and len(F) >= LIMIT:
                break
        rnd += 1
        if LIMIT and len(F) >= LIMIT:
            break
        if (fi + 1) % 100 == 0:
            print('  %d/%d files | %d rows | %.0fs'
                  % (fi + 1, len(files), len(F), time.time() - t0),
                  flush=True)
    F = np.stack(F)
    F98 = np.stack(F98)
    V = np.asarray(V, dtype=np.float32)
    A = np.asarray(A, dtype=np.uint8)
    T = np.asarray(T, dtype=np.uint8)
    R = np.asarray(R, dtype=np.int32)
    is_val = (R % 10 == 0)
    if BASE:
        z = np.load(BASE)
        nb = len(z['acts'])
        F = np.concatenate([z['feats'].astype(np.float32), F])
        F98 = np.concatenate([np.full((nb, 98), np.nan, dtype=np.float32),
                              F98])
        V = np.concatenate([z['vlabel'].astype(np.float32), V])
        A = np.concatenate([z['acts'].astype(np.uint8), A])
        T = np.concatenate([z['teacher'].astype(np.uint8), T])
        R = np.concatenate([z['round'].astype(np.int32), R])
        is_val = np.concatenate([z['is_val'], is_val])
        print('merged base cache: +%d rows -> %d total' % (nb, len(A)))
    tmp = OUT + '.part'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, feats=F, feats98=F98, acts=A, vlabel=V,
                            teacher=T, round=R, is_val=is_val)
    os.replace(tmp, OUT)
    vv = V[T == 6]
    print('rows %d (targets %d, val %d) | vlabel mean %.3f std %.3f '
          'min %.1f max %.1f | %.0fs'
          % (len(A), len(vv), int(is_val[T == 6].sum()),
             float(vv.mean()), float(vv.std()), float(vv.min()),
             float(vv.max()), time.time() - t0))
    print('wrote', OUT, '%.1f MB' % (os.path.getsize(OUT) / 1e6))


if __name__ == '__main__':
    main()
