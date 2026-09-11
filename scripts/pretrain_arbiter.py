#!/usr/bin/env python3
"""ARBITER P0 warm start: joint pi (CE) + V (margin-to-go regression).

Data: results/arbiter_p0_cache.npz (scripts/arbiter_extract.py).
  pi rows: teacher in ARBITER_PI_TEACHERS (default 0,1,2 = warden /
      sentinel / overlord — collector's 0.59-suicide actions stay OUT
      of the prior; E62 design note).
  V rows: finite vlabel AND teacher in ARBITER_V_TEACHERS (default
      0,1,2,3 — collector states teach high-yield positions).
Augmentation: per-batch symmetry s in 0..7 over the 8 dihedral perms
  (reaper-proven equivariant); actions remapped, V labels invariant.
Loss: CE(pi, a) + ARBITER_V_W * MSE(V, vlabel/10). E35 recipe: Adam
  1e-3, batch 1024, 5 epochs, CUDA+AMP (CPU fallback).
Gate: val_acc >= 0.5 (E35 got 0.82 on the 3-teacher mix).
Out: agent_code/arbiter/my-saved-model.pt (raw state_dict) + meta json.
Usage: python3 scripts/pretrain_arbiter.py [--epochs N] [--out PATH]
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))

import numpy as np

import arbiter.model as AM
import arbiter.features as AF

ACTION_LIST = AM.ACTION_LIST


def parse_env_set(name, default):
    raw = os.environ.get(name)
    if not raw:
        return set(default)
    return {int(x) for x in raw.replace(',', ' ').split() if x.strip() != ''}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--v-w', type=float,
                    default=float(os.environ.get('ARBITER_V_W', '1.0')))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache', default=os.path.join(REPO, 'results',
                                                    'arbiter_p0_cache.npz'))
    ap.add_argument('--out', default=os.path.join(REPO, 'agent_code',
                                                  'arbiter',
                                                  'my-saved-model.pt'))
    # Optional warm start (E97): load existing weights before training so
    # a short low-LR fine-tune can shift calibration without the
    # catastrophic distribution shift of a from-scratch retrain.
    ap.add_argument('--init', default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    z = np.load(a.cache)
    feats, acts = z['feats'], z['acts'].astype(np.int64)
    vlab, teacher, is_val = (z['vlabel'].astype(np.float32),
                             z['teacher'], z['is_val'])
    pi_t = parse_env_set('ARBITER_PI_TEACHERS', {0, 1, 2})
    v_t = parse_env_set('ARBITER_V_TEACHERS', {0, 1, 2, 3})
    pi_mask = np.array([t in pi_t for t in teacher])
    v_mask = np.isfinite(vlab) & np.array([t in v_t for t in teacher])
    print('cache rows %d | pi rows %d (teachers %s) | V rows %d | val %.1f%%' % (
        len(feats), int(pi_mask.sum()), sorted(pi_t), int(v_mask.sum()),
        100.0 * is_val.mean()))

    tr_pi = np.where(pi_mask & ~is_val)[0]
    va_pi = np.where(pi_mask & is_val)[0]
    tr_v = np.where(v_mask & ~is_val)[0]
    va_v = np.where(v_mask & is_val)[0]
    assert len(tr_pi) and len(va_pi) and len(tr_v) and len(va_v)
    print('train pi %d / val pi %d | train V %d / val V %d' % (
        len(tr_pi), len(va_pi), len(tr_v), len(va_v)))

    # index-level action remap under each symmetry (names -> indices once)
    act_perm = np.zeros((AF.N_SYMS, 6), dtype=np.int64)
    for s in range(AF.N_SYMS):
        for i, an in enumerate(ACTION_LIST):
            act_perm[s, i] = ACTION_LIST.index(AF.map_action(an, s))
    aug_perms = AF.AUG_PERMS

    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)
    torch.manual_seed(a.seed)
    model = AM.build_model().to(dev)
    if a.init:
        obj = torch.load(a.init, map_location='cpu', weights_only=True)
        if isinstance(obj, dict):
            for key in ('arbiter', 'pi_v_net', 'state_dict', 'q_net',
                        'model'):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        missing = model.load_state_dict(obj, strict=False)
        print('init %s (missing=%d unexpected=%d)'
              % (a.init, len(missing.missing_keys),
                 len(missing.unexpected_keys)))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))
    F = torch.from_numpy(feats)
    A = torch.from_numpy(acts)
    Y = torch.from_numpy(vlab / 10.0)
    has_v = torch.from_numpy(np.isfinite(vlab))

    def run_epoch(idxs, train):
        model.train(train)
        tot_loss, tot_ce, tot_mse, tot_hit, tot_n, tot_vn = (0.0,) * 6
        order = rng.permutation(len(idxs)) if train else np.arange(len(idxs))
        for st in range(0, len(idxs), a.batch):
            bi = idxs[order[st:st + a.batch]]
            s = int(rng.integers(0, AF.N_SYMS)) if train else 0
            xb = F[bi][:, aug_perms[s]].to(dev)
            ab = torch.from_numpy(act_perm[s][A[bi].numpy()]).to(dev)
            yb = Y[bi].to(dev)
            vb = has_v[bi].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                logits, vv = model(xb)
                loss_ce = ce(logits, ab)
                if bool(vb.any()):
                    loss_mse = mse(vv[vb], yb[vb])
                else:
                    loss_mse = vv.sum() * 0.0
                loss = loss_ce + a.v_w * loss_mse
            if train:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            bs = len(bi)
            tot_loss += float(loss.detach()) * bs
            tot_ce += float(loss_ce.detach()) * bs
            tot_mse += float(loss_mse.detach()) * bs if bool(vb.any()) else 0.0
            tot_hit += int((logits.argmax(-1) == ab).sum())
            tot_n += bs
            tot_vn += int(vb.sum())
        return (tot_loss / tot_n, tot_ce / tot_n,
                tot_mse / max(tot_n, 1), tot_hit / tot_n, tot_vn)

    for ep in range(a.epochs):
        t0 = time.perf_counter()
        tr = run_epoch(tr_pi, True)
        va = run_epoch(va_pi, False)
        # V validation on V rows (labels exist there by construction)
        va_vv = run_epoch(va_v, False)
        print('ep %d/%d %.0fs loss %.4f (ce %.4f mse %.4f) acc %.3f | '
              'val loss %.4f acc %.3f | V-val mse %.4f (n=%d)' % (
                  ep + 1, a.epochs, time.perf_counter() - t0,
                  tr[0], tr[1], tr[2], tr[3], va[0], va[3], va_vv[2],
                  va_vv[4]), flush=True)
    final_va, final_vmse = va[3], va_vv[2]
    tmp = a.out + '.part'
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, a.out)
    with open(os.path.splitext(a.out)[0] + '.meta.json', 'w') as fh:
        json.dump({'val_acc': final_va, 'val_mse': final_vmse,
                   'epochs': a.epochs, 'lr': a.lr, 'v_w': a.v_w,
                   'pi_teachers': sorted(pi_t), 'v_teachers': sorted(v_t),
                   'n_pi': len(tr_pi) + len(va_pi),
                   'n_v': len(tr_v) + len(va_v)}, fh, indent=1)
    print('saved %s (val_acc %.3f val_mse %.4f)' % (a.out, final_va,
                                                    final_vmse))
    if final_va < 0.5:
        print('GATE MISS: val_acc < 0.5 — do not trust this init blindly')
    else:
        print('GATE PASS: val_acc >= 0.5')


if __name__ == '__main__':
    main()
