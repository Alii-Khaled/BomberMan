#!/usr/bin/env python3
"""NG-TTA gather-exactness probe (Inc 0/1 gate, E119).

The ship's 8x-recompute TTA (state/safety/features recomputed per
symmetry) is replaced by ONE canonical compute + 7 exact dihedral
gathers (transform_tensor + AUG_PERMS) — the pretrain-augmentation
semantics. Verified invariants this probe enforces:

  G1 divergence confinement: any feature diff vs the recompute path is
     confined to the escape-tie-break block f[45..48] (flat indices
     3513..3516) — the escape BFS's first-move claim order is the only
     non-equivariant detail; board tensor + all other scalars are exact.
  G2 raw logit parity: views with identical features must have
     bit-identical logits (<=1e-6); views whose f[45..48] flipped are
     accounted for by G1/G3.
  G3 TTA-accumulated pi: must-flee argmax flips == 0, total flip rate
     <= 0.5% (measured 0.10% trek-only at n=2000).
  G4 gather strictly faster than recompute (measured 26.1 -> 0.3 ms).

Corpus: engine-style fuzzed states (standard lattice + random crates),
stratified to cover solo / post-plant / must-flee states.
Usage: python3 scripts/probe_ng_tta_equiv.py [--states N]
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch

import arbiter_ng.features as F
from arbiter_ng.safety import action_safety
from arbiter_ng.model import build_model

N_STATES = 1000
for a in sys.argv[1:]:
    if a.startswith('--states'):
        N_STATES = int(a.split('=')[1] if '=' in a
                       else sys.argv[sys.argv.index(a) + 1])

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)


def build_arena(rng, crate_density=0.35):
    a = np.zeros((17, 17), dtype=int)
    a[rng.random((17, 17)) < crate_density] = 1
    a[:1, :] = a[-1:, :] = a[:, :1] = a[:, -1:] = -1
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                a[x, y] = -1
    for (sx, sy) in ((1, 1), (1, 15), (15, 1), (15, 15)):
        a[sx, sy] = 0
    return a


def fuzz_state(rng):
    arena = build_arena(rng)
    free = [(x, y) for x in range(17) for y in range(17)
            if arena[x, y] == 0]
    rng.shuffle(free)
    if len(free) < 12:
        return None
    corners = [(1, 1), (1, 15), (15, 1), (15, 15)]
    agents = [corners[i] for i in rng.permutation(4)[:rng.integers(1, 5)]]
    self_xy = agents[0]
    others_xy = agents[1:]
    take = len(others_xy) + rng.integers(0, 10)
    coins = [tuple(c) for c in free[len(others_xy):len(others_xy) + take]
             if tuple(c) not in agents]
    rest = [c for c in free[len(others_xy) + take:] if tuple(c) not in agents]
    rng.shuffle(rest)
    n_bombs = int(rng.integers(0, 5))
    bombs = []
    for i in range(min(n_bombs, len(rest))):
        bx, by = rest[i]
        if rng.random() < 0.5:
            cand = [(x, y) for (x, y) in rest[i + 1:]
                    if abs(x - self_xy[0]) + abs(y - self_xy[1]) <= 3]
            if cand:
                bx, by = cand[int(rng.integers(0, len(cand)))]
        bombs.append(((bx, by), int(rng.integers(0, 5))))
    bombs_left = bool(rng.random() < 0.7)
    if not bombs_left and bombs:
        bombs[0] = ((self_xy[0], self_xy[1]), int(rng.integers(0, 4)))
    exp_map = np.zeros((17, 17), dtype=float)
    if rng.random() < 0.4 and len(rest) > 2:
        for (ex, ey) in rest[:int(rng.integers(1, 6))]:
            exp_map[ex, ey] = float(rng.integers(1, 3))
    others = [('opp%d' % i, int(rng.integers(0, 20)),
               bool(rng.random() < 0.8), xy)
              for i, xy in enumerate(others_xy)]
    return {
        'round': int(rng.integers(1, 60)),
        'step': int(rng.integers(0, 400)),
        'field': arena,
        'self': ('self', int(rng.integers(0, 20)), bombs_left, self_xy),
        'others': others,
        'bombs': bombs,
        'coins': coins,
        'user_input': None,
        'explosion_map': exp_map,
    }


def _tta_accum(logits):
    """Mirror the callbacks' TTA logit accumulation (map_action remap)."""
    acts = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    ai = {a: i for i, a in enumerate(acts)}
    acc = np.zeros(6, dtype=np.float64)
    for s in range(len(logits)):
        row = np.nan_to_num(logits[s], nan=0.0, posinf=50.0, neginf=-50.0)
        for a in acts:
            try:
                j = ai[F.map_action(a, s)]
            except Exception:
                continue
            acc[ai[a]] += row[j]
    return acc / max(1, len(logits))


def main():
    model = build_model()
    wpath = os.path.join(REPO, 'agent_code', 'arbiter_ng',
                         'my-saved-model.pt')
    if os.path.isfile(wpath):
        obj = torch.load(wpath, map_location='cpu', weights_only=True)
        if isinstance(obj, dict):
            for key in ('arbiter', 'pi_v_net', 'state_dict', 'model'):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        model.load_state_dict(obj, strict=False)
        print('INFO: loaded ship weights', wpath)
    else:
        print('INFO: no weights at %s (random-init logit parity)' % wpath)
    model.eval()

    rng = np.random.default_rng(7)
    states = []
    while len(states) < N_STATES:
        gs = fuzz_state(rng)
        if gs is not None:
            states.append(gs)

    n_solo = sum(1 for gs in states if not gs['others'])
    n_post = sum(1 for gs in states if not gs['self'][2])
    n_flee = 0
    for gs in states:
        sf = action_safety(gs)
        d = np.asarray(sf.get('danger'))
        x, y = gs['self'][3]
        if d.ndim == 3 and d.shape[0] > 1 and (d[0, x, y] or d[1, x, y]):
            n_flee += 1
    print('INFO: corpus solo=%d post-plant=%d must-flee=%d / %d'
          % (n_solo, n_post, n_flee, len(states)))
    check('corpus covers solo states', n_solo >= 20)
    check('corpus covers post-plant states', n_post >= 40)
    check('corpus covers must-flee states', n_flee >= 40)

    max_feat_diff = 0.0
    bad_idx = {}
    max_logit_diff = 0.0
    max_logit_diff_clean = 0.0
    n_flip_views = 0
    n_flee_flip = 0
    n_flip = 0
    max_pi_diff = 0.0
    t_recompute = 0.0
    t_gather = 0.0

    with torch.inference_mode():
        for gs in states:
            sf0 = action_safety(gs)
            f0 = F.state_to_features(gs, sf0)
            t0b = f0[:F.N_CHANNELS * 17 * 17].reshape(F.N_CHANNELS, 17, 17)
            s0 = f0[F.N_CHANNELS * 17 * 17:]
            rowsA, rowsB = [], []
            tw = time.perf_counter()
            for s in range(8):
                gs_s = F.transform_state(gs, s)
                sf_s = action_safety(gs_s)
                rowsA.append(F.state_to_features(gs_s, sf_s))
            t_recompute += time.perf_counter() - tw
            tw = time.perf_counter()
            for s in range(8):
                if s == 0:
                    rowsB.append(f0)
                else:
                    rowsB.append(np.concatenate(
                        [F.transform_tensor(t0b, s).ravel(),
                         s0[F.AUG_PERMS[s]]]).astype(np.float32))
            t_gather += time.perf_counter() - tw

            la, _ = model(torch.from_numpy(np.stack(rowsA)))
            lb, _ = model(torch.from_numpy(np.stack(rowsB)))
            la, lb = la.numpy(), lb.numpy()
            max_logit_diff = max(max_logit_diff,
                                 float(np.abs(la - lb).max()))
            pa, pb = _tta_accum(la), _tta_accum(lb)
            max_pi_diff = max(max_pi_diff, float(np.abs(pa - pb).max()))
            if int(np.argmax(pa)) != int(np.argmax(pb)):
                n_flip += 1
                d = np.asarray(sf0.get('danger'))
                x, y = gs['self'][3]
                if d.ndim == 3 and d.shape[0] > 1 \
                        and (d[0, x, y] or d[1, x, y]):
                    n_flee_flip += 1
            for vi, (fA, fB) in enumerate(zip(rowsA, rowsB)):
                dd = np.abs(fA - fB)
                if float(dd.max()) > max_feat_diff:
                    max_feat_diff = float(dd.max())
                if float(dd.max()) > 1e-9:
                    n_flip_views += 1
                    for i in np.nonzero(dd > 1e-9)[0]:
                        bad_idx[int(i)] = bad_idx.get(int(i), 0) + 1
                else:
                    dl = float(np.abs(la[vi] - lb[vi]).max())
                    max_logit_diff_clean = max(max_logit_diff_clean, dl)

    print('INFO: feature diff max = %.3g; divergent indices: %s'
          % (max_feat_diff, dict(sorted(bad_idx.items()))))
    check('G1 divergence confined to f[45..48] (3513..3516)',
          all(3513 <= i <= 3516 for i in bad_idx))
    print('INFO: logit parity: clean-view max|dlogit| = %.3g '
          '(%d flipped views, raw max %.3g)'
          % (max_logit_diff_clean, n_flip_views, max_logit_diff))
    check('G2 clean-view logit parity <=1e-6',
          max_logit_diff_clean <= 1e-6)
    print('INFO: TTA pi flips %d/%d (must-flee %d/%d), max|dpi| = %.3g'
          % (n_flip, len(states), n_flee_flip, n_flee, max_pi_diff))
    check('G3 must-flee argmax flips == 0', n_flee_flip == 0)
    check('G3 total flip rate <= 0.5%', n_flip <= 0.005 * len(states))
    n = len(states)
    ms_recompute = t_recompute / n * 1000.0
    ms_gather = t_gather / n * 1000.0
    print('INFO: TTA feature stage per step: recompute %.1f ms, '
          'gather %.1f ms (%.0fx faster)'
          % (ms_recompute, ms_gather, ms_recompute / max(ms_gather, 1e-9)))
    check('G4 gather faster than recompute', ms_gather < ms_recompute)

    if fails:
        print('PROBE FAIL: ' + ', '.join(fails))
        raise SystemExit(1)
    print('PROBE PASS (NG TTA gather exact; Inc 1 refactor is '
          'quality-equivalent within the measured tie-break envelope)')


if __name__ == '__main__':
    main()
