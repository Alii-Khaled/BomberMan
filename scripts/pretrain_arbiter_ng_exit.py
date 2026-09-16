#!/usr/bin/env python3
"""E121 ExIt distillation: fit arbiter_ng's pi to the SEARCH's own
arbitrated first-steps recorded by agent_code/exit_recorder.

This is expert-iteration (ExIt) step 1: the exact-rollout search is the
improver, pi is the distilled student, and the recorded distribution is
the ship's own visitation (on-manifold, unlike the E115 collector-label
attempt). Warm start from the ship weights, CE-only on pi (the search
owns bomb pricing at inference; V stays untouched), dihedral
augmentation via transform_tensor + AUG_PERMS + map_action.

Usage:
  python3 scripts/pretrain_arbiter_ng_exit.py [--epochs 8] [--batch 256]
      [--lr 5e-5] [--out results/arbiter_ng_exit.pt]
      [--dirs results/demos/arbiter_ng_exit] [--init agent_code/arbiter_ng/my-saved-model.pt]
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

import agent_code.arbiter_ng.features as F
import agent_code.arbiter_ng.model as M

ACTION_LIST = list(M.ACTION_LIST)
TENSOR_DIM = F.N_CHANNELS * 17 * 17


def load_rows(dirs):
    feats, labels, fidx = [], [], []
    files = []
    for d in dirs.split(','):
        files += sorted(glob.glob(os.path.join(REPO, d.strip(), '*.npz')))
    print('loading %d demo files...' % len(files))
    for fi, f in enumerate(files):
        z = np.load(f)
        f_ = np.asarray(z['feats'], dtype=np.float32)
        lb = np.asarray(z['labels'], dtype=np.int64)
        if f_.ndim != 2 or f_.shape[1] != TENSOR_DIM + 98:
            print('  skip %s (bad shape %s)' % (f, f_.shape))
            continue
        feats.append(f_)
        labels.append(lb)
        fidx.append(np.full(len(lb), fi, dtype=np.int64))
    X = np.concatenate(feats) if feats else np.zeros((0, TENSOR_DIM + 98),
                                                     np.float32)
    Y = np.concatenate(labels) if labels else np.zeros(0, np.int64)
    FI = np.concatenate(fidx) if fidx else np.zeros(0, np.int64)
    return X, Y, FI, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(REPO, 'results',
                                                  'arbiter_ng_exit.pt'))
    ap.add_argument('--dirs',
                    default='results/demos/arbiter_ng_exit')
    ap.add_argument('--init',
                    default=os.path.join(REPO, 'agent_code', 'arbiter_ng',
                                         'my-saved-model.pt'))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    X, Y, FI, files = load_rows(a.dirs)
    N = len(Y)
    is_val = (FI % 10 == 0)
    print('rows %d | val %d (%.1f%%) | files %d | label mix %s'
          % (N, int(is_val.sum()), 100 * is_val.mean(), len(files),
             np.bincount(Y, minlength=6).tolist()))
    if N < 1000:
        print('WARNING: tiny corpus (%d rows)' % N)

    act_perm = np.zeros((F.N_SYMS, 6), dtype=np.int64)
    for s in range(F.N_SYMS):
        for i, an in enumerate(ACTION_LIST):
            act_perm[s, i] = ACTION_LIST.index(F.map_action(an, s))

    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)
    torch.manual_seed(a.seed)
    model = M.build_model().to(dev)
    if a.init and os.path.isfile(a.init):
        obj = torch.load(a.init, map_location='cpu', weights_only=True)
        if isinstance(obj, dict):
            for key in ('arbiter', 'pi_v_net', 'state_dict', 'q_net',
                        'model'):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        model.load_state_dict(obj, strict=True)
        print('warm start from', a.init)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ce = torch.nn.CrossEntropyLoss()

    tr = np.where(~is_val)[0]
    va = np.where(is_val)[0]

    def batch_xy(idxs, sym):
        xb = X[idxs].copy()
        y = Y[idxs].copy()
        if sym != 0:
            tens = xb[:, :TENSOR_DIM].reshape(-1, F.N_CHANNELS, 17, 17)
            tens = np.stack([F.transform_tensor(t, sym) for t in tens])
            xb[:, :TENSOR_DIM] = tens.reshape(len(idxs), -1)
            xb[:, TENSOR_DIM:] = X[idxs][:, TENSOR_DIM:][:,
                                                         F.AUG_PERMS[sym]]
            y = act_perm[sym][y]
        return (torch.from_numpy(np.ascontiguousarray(xb)).to(dev),
                torch.from_numpy(np.ascontiguousarray(y)).to(dev))

    def run(idxs, train):
        model.train(train)
        order = rng.permutation(len(idxs)) if train \
            else np.arange(len(idxs))
        hits = tot = 0
        loss_sum = 0.0
        for st in range(0, len(idxs), a.batch):
            bi = idxs[order[st:st + a.batch]]
            if len(bi) < 2:
                continue
            sym = int(rng.integers(0, F.N_SYMS)) if train else 0
            xb, yb = batch_xy(bi, sym)
            opt.zero_grad(set_to_none=True)
            logits, _v = model(xb)
            loss = ce(logits, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            loss_sum += float(loss.detach()) * len(bi)
            hits += int((logits.argmax(-1) == yb).sum())
            tot += len(bi)
        return loss_sum / max(tot, 1), hits / max(tot, 1)

    best = -1.0
    for ep in range(a.epochs):
        t0 = time.perf_counter()
        tl, ta = run(tr, True)
        vl, vacc = run(va, False)
        print('ep %d/%d %.0fs train loss %.4f acc %.3f | val loss %.4f '
              'acc %.3f' % (ep + 1, a.epochs,
                            time.perf_counter() - t0, tl, ta, vl, vacc),
              flush=True)
        if vacc > best:
            best = vacc
            tmp = a.out + '.part'
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       tmp)
            os.replace(tmp, a.out)
            with open(os.path.splitext(a.out)[0] + '.meta.json', 'w') as fh:
                json.dump({'val_acc': vacc, 'epoch': ep + 1, 'lr': a.lr,
                           'rows': N, 'n_files': len(files),
                           'recipe': 'E121 exit distill (CE, warm ship)'},
                          fh, indent=1)
        ep_out = '%s.ep%02d' % (a.out, ep + 1)
        tmp = ep_out + '.part'
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   tmp)
        os.replace(tmp, ep_out)
    print('saved best %s (val_acc %.3f)' % (a.out, best))


if __name__ == '__main__':
    main()
