#!/usr/bin/env python3
"""E105 NGQ dataset: offline (x, a, r, r3, next, done) from the NG-1 BC
cache + the raw demo img arrays.

x = (img/4 raveled 3468) ++ cached 98-dim NG scalars (the exact eval input
layout). r = score-margin delta between consecutive ticks (engine score
units), r3 = gamma-discounted 3-step return with terminal handling, next
= row index of the successor state (itself when terminal).

Usage: python3 scripts/build_ngq_dataset.py \
    [--cache results/arbiter_ng_scalars.npz] [--out results/ngq_ds]
"""
import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)

import numpy as np

TENSOR_DIM = 12 * 17 * 17
FEATURE_DIM = TENSOR_DIM + 98
GAMMA = 0.99
NSTEP = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='results/arbiter_ng_scalars.npz')
    ap.add_argument('--dirs', default='results/apex_demos,'
                                     'results/apex_ng_demos')
    ap.add_argument('--out', default='results/ngq_ds')
    a = ap.parse_args()

    t0 = time.time()
    out_base = a.out if os.path.isabs(a.out) \
        else os.path.join(REPO, a.out)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    z = np.load(os.path.join(REPO, a.cache), allow_pickle=False)
    scalars, acts = z['scalars'], z['acts']
    margin, fidx, tick = z['margin'], z['fidx'], z['tick']
    n = int(z['n'])
    files = [f.decode() if isinstance(f, bytes) else str(f)
             for f in z['files']]
    print('cache rows=%d files=%d' % (n, len(files)))

    x = np.lib.format.open_memmap(
        out_base + '_x.npy', mode='w+',
        dtype=np.float32, shape=(n, FEATURE_DIM))
    r = np.zeros(n, dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    nxt = np.zeros(n, dtype=np.int32)

    # file-major row groups (cache was built over sorted(files))
    starts = np.searchsorted(fidx, np.arange(len(files) + 1))
    for fi in range(len(files)):
        lo, hi = starts[fi], (starts[fi + 1] if fi + 1 <= len(files) else n)
        if lo == hi:
            continue
        d = np.load(files[fi])
        img = d['img'].astype(np.float32) / 4.0
        # emit rows in tick order (cache preserved order within file)
        rows = np.arange(lo, hi)
        tt = tick[rows]
        assert np.all(np.diff(tt) >= 0), 'row order violated'
        timg = img[tt].reshape(hi - lo, TENSOR_DIM)
        x[lo:hi, :TENSOR_DIM] = timg
        x[lo:hi, TENSOR_DIM:] = scalars[rows]
        # rewards/next within the file block
        for k in range(lo, hi):
            if k + 1 < hi and tick[k + 1] == tick[k] + 1:
                nxt[k] = k + 1
                r[k] = margin[k + 1] - margin[k]
            else:
                nxt[k] = k
                done[k] = True
        if (fi + 1) % 200 == 0:
            print('  %d/%d files (%.0fs)' % (fi + 1, len(files),
                                             time.time() - t0), flush=True)
    assert starts[len(files)] == n, 'coverage gap'

    # gamma-discounted NSTEP returns walking next pointers
    r3 = np.zeros(n, dtype=np.float32)
    nxt3 = np.zeros(n, dtype=np.int32)
    done3 = np.zeros(n, dtype=bool)
    disc = np.array([GAMMA ** j for j in range(NSTEP)], dtype=np.float32)
    for k in range(n):
        acc, cur, dflag = 0.0, k, False
        for j in range(NSTEP):
            if dflag:
                break
            acc += disc[j] * float(r[cur])
            cur = int(nxt[cur])
            dflag = bool(done[cur]) or cur == int(nxt[cur])
        r3[k] = acc
        nxt3[k] = cur
        done3[k] = dflag
        if (k + 1) % 100000 == 0:
            print('  nstep %d/%d (%.0fs)' % (k + 1, n, time.time() - t0),
                  flush=True)

    np.save(out_base + '_a.npy', acts)
    np.save(out_base + '_r.npy', r)
    np.save(out_base + '_done.npy', done)
    np.save(out_base + '_next.npy', nxt)
    np.save(out_base + '_r3.npy', r3)
    np.save(out_base + '_next3.npy', nxt3)
    np.save(out_base + '_done3.npy', done3)
    with open(out_base + '_meta.json', 'w') as fh:
        json.dump({'n': n, 'files': len(files), 'gamma': GAMMA,
                   'nstep': NSTEP, 'feature_dim': FEATURE_DIM}, fh)
    print('done: %d rows (%.0fs) x=%s' % (n, time.time() - t0, x.shape))


if __name__ == '__main__':
    main()
