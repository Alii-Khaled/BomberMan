#!/usr/bin/env python3
"""E105 NGQ trainer: offline Double-DQN over the ngq dataset.

The Q head trains into the model's `pi` weight slot, so the existing
arbiter_ng agent loads the result unchanged (ARBITER_MODEL). Trunk is
warm-started from the NG-1 BC checkpoint; V/aux heads stay zero (V_BLEND
default 0 for NG keeps the search pure-exact).

Usage: python3 scripts/train_ngq_dqn.py [--init results/arbiter_ng_full.pt.ep04]
"""
import argparse
import json
import glob
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))
sys.path.insert(0, REPO)

import numpy as np
import torch
import torch.nn.functional as F

import agent_code.arbiter_ng.model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='results/ngq_ds')
    ap.add_argument('--init', default='results/arbiter_ng_full.pt.ep04')
    ap.add_argument('--out', default='results/arbiter_ngq.pt')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-4)
    a = ap.parse_args()

    t0 = time.time()
    ds_base = a.ds if os.path.isabs(a.ds) else os.path.join(REPO, a.ds)
    meta = json.load(open(ds_base + '_meta.json'))
    n = int(meta['n'])
    x = np.load(ds_base + '_x.npy', mmap_mode='r')
    acts = np.load(ds_base + '_a.npy')
    r3 = np.load(ds_base + '_r3.npy')
    nxt3 = np.load(ds_base + '_next3.npy')
    done3 = np.load(ds_base + '_done3.npy')
    print('dataset rows=%d (%.0fs)' % (n, time.time() - t0))

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = M.build_model().to(dev)
    if os.path.exists(os.path.join(REPO, a.init)):
        obj = torch.load(os.path.join(REPO, a.init), map_location='cpu',
                         weights_only=True)
        model.load_state_dict(obj, strict=False)
        print('warm start: %s' % a.init)
    # re-zero pi: the BC logits are action preferences (softmax), Q needs
    # a fresh value head (zero-init = no prior on return scale)
    M.nn.init.zeros_(model.pi.weight)
    M.nn.init.zeros_(model.pi.bias)
    tgt = M.build_model().to(dev)
    tgt.load_state_dict(model.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    acts_t = torch.from_numpy(acts.astype(np.int64)).to(dev)
    r3_t = torch.from_numpy(r3).to(dev)
    done3_t = torch.from_numpy(done3).to(dev)
    gam = 0.99

    steps_per_epoch = max(1, n // a.batch)
    rng = np.random.default_rng(7)
    for ep in range(1, a.epochs + 1):
        ep_loss = 0.0
        tgt.load_state_dict(model.state_dict())
        for _ in range(steps_per_epoch):
            idx = rng.integers(0, n, size=a.batch)
            xb = torch.from_numpy(np.asarray(x[idx])).to(dev)
            xb_next = torch.from_numpy(
                np.asarray(x[nxt3[idx]])).to(dev)
            qb, _ = model(xb)
            qa = qb.gather(1, acts_t[idx].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                qn, _ = model(xb_next)
                a_star = qn.argmax(dim=1)
                yq, _ = tgt(xb_next)
                y_max = yq.gather(1, a_star.unsqueeze(1)).squeeze(1)
                y = r3_t[idx] + (gam ** 3) * \
                    (1.0 - done3_t[idx].float()) * y_max
            loss = F.smooth_l1_loss(qa, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach())
        print('ep%02d loss=%.4f (%.0fs)' % (ep, ep_loss / steps_per_epoch,
                                            time.time() - t0), flush=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(REPO, a.out + '.ep%02d' % ep))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               os.path.join(REPO, a.out))
    print('saved %s' % a.out)


if __name__ == '__main__':
    main()
