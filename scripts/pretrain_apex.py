#!/usr/bin/env python3
"""B4: behavior-clone teachers into ApexNet (CE over Q logits).

Usage:
  python3 scripts/pretrain_apex.py --demos 'results/apex_demos/*/*.npz' \
      --epochs 5 --out agent_code/apex/my-saved-model.pt [--init trunk|fresh] [--no-aug]

Loss: CrossEntropy(softmax(Q(s)), a_demo) + 0.1 * aux-alive (aux target 0:
keeps the danger head finite, no supervision signal).
Init: trunk (o3sbest conv weights, head re-init; default) or fresh.
Aug: per-batch random CCW rotation with label remap (default on).
Saves raw state dict + meta json. Gate: val_acc >= 0.5 (E35 got 0.82 on
the MLP; CNN-from-teachers is harder — floor, not target).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agent_code.apex.model import build_model

PERM_CCW = [3, 0, 1, 2, 4, 5]
HEAD_KEYS = {'fc.0.weight', 'fc.0.bias', 'v.weight', 'v.bias',
             'adv.weight', 'adv.bias', 'aux.weight', 'aux.bias'}


def load_demos(pattern, cap=200000):
    imgs, scs, acts = [], [], []
    for f in sorted(glob.glob(pattern)):
        try:
            z = np.load(f)
            n = min(len(z['act']), len(z['img']))
            for i in range(n):
                if len(acts) >= cap:
                    break
                imgs.append(z['img'][i])
                scs.append(z['sc'][i].astype(np.float32))
                acts.append(int(z['act'][i]))
        except Exception:
            continue
        if len(acts) >= cap:
            break
    return ((np.stack(imgs).astype(np.float32) / 4.0) if imgs else np.zeros((0, 12, 17, 17), np.float32),
            np.stack(scs).astype(np.float32) if scs else np.zeros((0, 16), np.float32),
            np.asarray(acts, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos', default='results/apex_demos/*/*.npz')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='agent_code/apex/my-saved-model.pt')
    ap.add_argument('--init', default='trunk', choices=['trunk', 'fresh'])
    ap.add_argument('--no-aug', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev)

    imgs, scs, acts = load_demos(a.demos)
    print('samples: %d' % len(acts))
    assert len(acts) > 1000, 'too few demos (need >1000, got %d)' % len(acts)
    perm = rng.permutation(len(acts))
    cut = int(0.9 * len(acts))
    tr, va = perm[:cut], perm[cut:]
    print('train %d / val %d' % (len(tr), len(va)))

    m = build_model().to(dev)
    if a.init == 'trunk':
        ckpt = torch.load('results/archive/overlord_val_best_O3s.pt',
                          map_location='cpu', weights_only=False)['q_net']
        body = {k: v for k, v in ckpt.items() if k not in HEAD_KEYS}
        missing = m.load_state_dict(body, strict=False).missing_keys
        assert set(missing) == HEAD_KEYS, missing
        print('trunk init: o3sbest convs in, head fresh')
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    ce = torch.nn.CrossEntropyLoss()

    def run_epoch(idxs, train):
        m.train(train)
        tot_loss, tot_acc, n = 0.0, 0, 0
        order = rng.permutation(len(idxs)) if train else np.arange(len(idxs))
        for s in range(0, len(idxs), a.batch):
            bi = idxs[order[s:s + a.batch]]
            I = torch.from_numpy(imgs[bi]).to(dev)
            S = torch.from_numpy(scs[bi]).to(dev)
            A = torch.from_numpy(acts[bi]).to(dev)
            if train and not a.no_aug:
                k = int(rng.integers(4))
                if k:
                    I = torch.rot90(I, k, (2, 3)).contiguous()
                    for _ in range(k):
                        A = torch.tensor([PERM_CCW[int(x)] for x in A],
                                         device=dev)
            if train:
                opt.zero_grad(set_to_none=True)
            q, aux = m(I, S)
            loss = ce(q, A) + 0.1 * (aux.pow(2).mean())
            acc = (q.argmax(1) == A).float().mean().item()
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            tot_loss += float(loss.item()) * len(bi)
            tot_acc += acc * len(bi)
            n += len(bi)
        return tot_loss / n, tot_acc / n

    for ep in range(a.epochs):
        tl, ta = run_epoch(tr, True)
        vl, va = run_epoch(va, False)
        print('ep %d: train loss %.4f acc %.3f | val loss %.4f acc %.3f'
              % (ep, tl, ta, vl, va))
    final_va = va
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    torch.save(sd, a.out + '.tmp')
    os.replace(a.out + '.tmp', a.out)
    with open(a.out + '.meta.json', 'w') as f:
        json.dump({'val_acc': final_va, 'epochs': a.epochs, 'lr': a.lr,
                   'init': a.init, 'aug': not a.no_aug,
                   'samples': len(acts)}, f)
    print('saved %s (val_acc %.3f)' % (a.out, final_va))
    if final_va < 0.5:
        print('GATE MISS: val_acc < 0.5 — do not trust this init blindly')
        return 2
    print('GATE PASS: val_acc >= 0.5')
    return 0


if __name__ == '__main__':
    sys.exit(main())
