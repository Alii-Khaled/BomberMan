#!/usr/bin/env python3
"""E101 ARBITER-NG stage-1 training: expert-iteration BC on apex-format
league recordings (img 12x17x17 uint8 + sc + act).

Pipeline: reconstruct game_state per tick -> action_safety -> 98-dim
scalar cache (with safety, matching inference) + dihedral augmentation
on the fly (spatial transforms on the board tensor, AUG_PERMS on the
scalars, map_action on the label). Loss = CE(pi) + aux_w * MSE(aux,
margin/10) with the margin from sc[:,12] (E65 channel).

Usage:
  python3 scripts/pretrain_arbiter_ng.py [--epochs 20] [--batch 256]
      [--lr 1e-3] [--aux-w 0.1] [--out results/arbiter_ng.pt]
      [--dirs results/apex_demos,results/apex_ng_demos]
      [--init-scalars agent_code/arbiter/my-saved-model.pt] [--rebuild]
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
import agent_code.arbiter_ng.safety as SAFE

ACTION_LIST = list(M.ACTION_LIST)


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


def build_cache(dirs, cache, rebuild):
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache, allow_pickle=False)
        if int(z['n']) > 0:
            print('cache exists: %d rows (%s)' % (int(z['n']), cache))
            return z
    files = []
    for d in dirs.split(','):
        files += sorted(glob.glob(os.path.join(REPO, d.strip(), '*/*.npz')))
    print('featurizing %d files...' % len(files))
    S, A, MG, FI, FT = [], [], [], [], []
    t0 = time.time()
    for fi, f in enumerate(files):
        d = np.load(f)
        img = d['img'].astype(np.float32) / 4.0
        sc = d['sc'].astype(np.float32)
        act = d['act'].astype(np.int64)
        margins = sc[:, 12].astype(np.float64) * 10.0
        for i in range(len(act)):
            try:
                gs = reconstruct(i, img, sc)
                sf = SAFE.action_safety(gs)
                S.append(F.scalar_features(gs, sf).astype(np.float32))
                A.append(np.uint8(act[i]))
                MG.append(np.float32(margins[i]))
                FI.append(np.int32(fi))
                FT.append(np.int32(i))
            except Exception:
                continue
        if (fi + 1) % 100 == 0:
            print('  %d/%d files | %d rows | %.0fs'
                  % (fi + 1, len(files), len(S), time.time() - t0),
                  flush=True)
    z = dict(scalars=np.stack(S), acts=np.asarray(A),
             margin=np.asarray(MG), fidx=np.asarray(FI),
             tick=np.asarray(FT), n=np.int64(len(S)),
             files=np.asarray(files))
    tmp = cache + '.part'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **z)
    os.replace(tmp, cache)
    print('wrote cache %s (%.1f MB, %d rows)' % (
        cache, os.path.getsize(cache) / 1e6, len(S)))
    return np.load(cache, allow_pickle=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--aux-w', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(REPO, 'results',
                                                  'arbiter_ng.pt'))
    ap.add_argument('--dirs', default='results/apex_demos,results/apex_ng_demos')
    ap.add_argument('--cache', default=os.path.join(REPO, 'results',
                                                    'arbiter_ng_scalars.npz'))
    ap.add_argument('--init-scalars', default=None)
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--save-every', type=int, default=1,
                    help='also keep <out>.epNN checkpoints (0 = best only)')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    z = build_cache(a.dirs, a.cache, a.rebuild)
    scal, acts, margin, fidx, tick = (z['scalars'], z['acts'].astype(np.int64),
                                      z['margin'], z['fidx'], z['tick'])
    files = [str(x) for x in z['files']]
    N = len(acts)
    is_val = (fidx % 10 == 0)
    print('rows %d | val %d (%.1f%%) | files %d'
          % (N, int(is_val.sum()), 100.0 * is_val.mean(), len(files)))

    # load tensors into RAM (uint8, one array per file)
    imgs = {}
    t0 = time.time()
    for fi in range(len(files)):
        d = np.load(files[fi])
        imgs[fi] = d['img']  # uint8 (T,12,17,17)
    print('loaded %d img arrays (%.0fs)' % (len(imgs), time.time() - t0))

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
    if a.init_scalars:
        obj = torch.load(a.init_scalars, map_location='cpu', weights_only=True)
        if isinstance(obj, dict):
            for key in ('arbiter', 'pi_v_net', 'state_dict', 'q_net', 'model'):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        msd = model.state_dict()
        copied = []
        for src, dst in (('trunk.0.weight', 'mlp.0.weight'),
                         ('trunk.0.bias', 'mlp.0.bias'),
                         ('trunk.2.weight', 'mlp.2.weight'),
                         ('trunk.2.bias', 'mlp.2.bias')):
            if src in obj and dst in msd:
                v, t = obj[src], msd[dst]
                if v.dim() == 2:
                    r = min(v.shape[0], t.shape[0])
                    c = min(v.shape[1], t.shape[1])
                    t[:r, :c] = v[:r, :c]
                else:
                    r = min(v.shape[0], t.shape[0])
                    t[:r] = v[:r]
                copied.append(dst)
        print('init-scalars copied:', copied)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))

    def batch_xy(idxs, sym):
        fid = fidx[idxs]
        tk = tick[idxs]
        tens = np.empty((len(idxs), 12, 17, 17), dtype=np.float32)
        for k in range(len(idxs)):
            tens[k] = imgs[int(fid[k])][int(tk[k])].astype(np.float32) / 4.0
        if sym != 0:
            tens = np.stack([F.transform_tensor(t, sym) for t in tens])
        s = scal[idxs]
        if sym != 0:
            s = s[:, F.AUG_PERMS[sym]]
        x = np.concatenate([tens.reshape(len(idxs), -1), s], axis=1)
        y = act_perm[sym][acts[idxs]] if sym != 0 else acts[idxs]
        return (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev),
                torch.from_numpy(margin[idxs] / 10.0).to(dev))

    tr = np.where(~is_val)[0]
    va = np.where(is_val)[0]

    def run(idxs, train):
        model.train(train)
        order = rng.permutation(len(idxs)) if train else np.arange(len(idxs))
        hits = tot = 0
        loss_sum = 0.0
        for st in range(0, len(idxs), a.batch):
            bi = idxs[order[st:st + a.batch]]
            sym = int(rng.integers(0, F.N_SYMS)) if train else 0
            xb, yb, mb = batch_xy(bi, sym)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                logits, _v, aux = model.forward_all(xb)
                loss = ce(logits, yb) + a.aux_w * mse(aux, mb)
            if train:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
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
              'acc %.3f' % (ep + 1, a.epochs, time.perf_counter() - t0,
                            tl, ta, vl, vacc), flush=True)
        if vacc > best:
            best = vacc
            tmp = a.out + '.part'
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       tmp)
            os.replace(tmp, a.out)
            with open(os.path.splitext(a.out)[0] + '.meta.json', 'w') as fh:
                json.dump({'val_acc': vacc, 'epoch': ep + 1, 'lr': a.lr,
                           'aux_w': a.aux_w, 'rows': N,
                           'n_files': len(files)}, fh, indent=1)
        if a.save_every and (ep + 1) % a.save_every == 0:
            ep_out = '%s.ep%02d' % (a.out, ep + 1)
            tmp = ep_out + '.part'
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       tmp)
            os.replace(tmp, ep_out)
    print('saved best %s (val_acc %.3f)' % (a.out, best))


if __name__ == '__main__':
    main()
