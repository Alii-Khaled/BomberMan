#!/usr/bin/env python3
"""Behavior-clone pretrain for reaper from teacher demos.

Loads demo npz files (feats/acts), trains the reaper MLP with cross-
entropy over 8x dihedral-augmented batches, saves weights as
agent_code/reaper/my-saved-model.pt + checkpoints/bc_last.pt (payload
format resumable by train.py's setup_training).

Usage (from repo root):
  python3 scripts/pretrain_reaper.py --demos results/demos/warden/round_*.npz \
      --epochs 5 --batch 512 --lr 1e-3 --tau 1.0 --val-split 0.02
"""
import argparse
import glob
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos', nargs='+', required=True,
                    help='globs of demo npz files (feats/acts)')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--val-split', type=float, default=0.02)
    ap.add_argument('--steps-per-epoch', type=int, default=2000)
    ap.add_argument('--out-dir', default=os.path.join(_ROOT, 'agent_code', 'reaper'))
    args = ap.parse_args()

    import torch
    from agent_code.reaper.features import FEATURE_DIM, apply_aug, map_action, N_SYMS
    from agent_code.reaper.model import build_model, ACTION_LIST

    files = []
    for pat in args.demos:
        files += sorted(glob.glob(pat)) or sorted(glob.glob(os.path.expanduser(pat)))
    if not files:
        raise SystemExit('no demo files found')
    feats, acts = [], []
    for f in files:
        d = np.load(f)
        F = np.asarray(d['feats'], dtype=np.float32)
        A = np.asarray(d['acts'])
        if F.ndim == 2 and F.shape[1] == FEATURE_DIM:
            feats.append(F)
            acts.append(A)
    F = np.concatenate(feats, axis=0)
    A = np.concatenate(acts, axis=0)
    A = np.array([int(a) for a in A], dtype=np.int64)
    n_val = max(1, int(len(F) * args.val_split))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(F))
    val_F, val_A = F[perm[:n_val]], A[perm[:n_val]]
    tr_F, tr_A = F[perm[n_val:]], A[perm[n_val:]]
    print(f'demos: {len(F)} samples ({len(tr_F)} train / {len(val_F)} val)')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    tau = max(float(args.tau), 1e-3)
    rng = np.random.default_rng(1)

    def aug_batch(Fb, Ab, n):
        idx = rng.integers(0, len(Fb), size=n)
        X = Fb[idx].copy()
        Y = Ab[idx].copy()
        syms = rng.integers(0, N_SYMS, size=n)
        for i, s in enumerate(syms):
            X[i] = apply_aug(X[i], int(s))
            Y[i] = ACTION_LIST.index(map_action(ACTION_LIST[int(Y[i])], int(s)))
        return torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)

    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for _ in range(args.steps_per_epoch):
            X, Y = aug_batch(tr_F, tr_A, args.batch)
            logits = model(X) / tau
            loss = lossf(logits, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n += 1
        model.eval()
        with torch.no_grad():
            nv = min(len(val_F), 1024)
            Xv = torch.from_numpy(val_F[:nv].copy()).to(dev)
            pred = model(Xv).argmax(dim=1).cpu().numpy()
            acc = float((pred == val_A[:nv]).mean())
        model.train()
        print(f'epoch {ep + 1}/{args.epochs} loss={tot / n:.4f} val_acc={acc:.3f}')

    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    model_path = os.path.join(out, 'my-saved-model.pt')
    torch.save(sd, model_path)
    ckpt_dir = os.path.join(out, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {'q_net': sd, 'target_net': sd, 'episode': 0,
               'total_steps': 0, 'epsilon_steps': 0}
    bc_path = os.path.join(ckpt_dir, 'bc_last.pt')
    torch.save(payload, bc_path)
    print(f'saved {model_path} and {bc_path}')


if __name__ == '__main__':
    main()
