#!/usr/bin/env python3
"""E99 Phase B gate: V RMSE on exact-outcome targets.

Compares the legacy ship V (98-dim) and an optional candidate V
(package feature dim) against the exact-outcome labels (teacher id 6)
in a target cache produced by scripts/arbiter_value_targets.py.

Usage:
  python3 scripts/eval_arbiter_v.py --cache results/arbiter_v2_vcache.npz \
      [--cand results/arbiter_v2_v_256.pt --pkg=arbiter_v2 --hid 256] [--split val]
"""
import argparse
import importlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))

import numpy as np


def build(pkg, hid=None):
    if hid:
        os.environ['ARBITER_HID1'] = os.environ['ARBITER_HID2'] = \
            os.environ['ARBITER_HID3'] = str(hid)
    AM = importlib.import_module(pkg + '.model')
    return AM.build_model()


def load_state(model, path):
    import torch
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict):
        for key in ('arbiter', 'pi_v_net', 'state_dict', 'q_net', 'model'):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    model.load_state_dict(obj, strict=False)
    model.eval()
    return model


def rmse(model, feats, y):
    import torch
    with torch.no_grad():
        _pi, v = model(torch.from_numpy(feats))
    v = v.cpu().numpy()
    e = v - y
    return float(np.sqrt(np.mean(e * e))), float(np.corrcoef(v, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--legacy', default=os.path.join(
        REPO, 'agent_code', 'arbiter', 'my-saved-model.pt'))
    ap.add_argument('--cand', default=None)
    ap.add_argument('--pkg', default='arbiter_v2')
    ap.add_argument('--hid', type=int, default=None)
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    a = ap.parse_args()

    z = np.load(a.cache)
    T = z['teacher']
    m = (T == 6) & np.isfinite(z['vlabel'])
    if a.split == 'val':
        m = m & z['is_val']
    elif a.split == 'train':
        m = m & ~z['is_val']
    y = z['vlabel'][m].astype(np.float64)
    print('targets n=%d split=%s | y mean %.3f std %.3f'
          % (len(y), a.split, y.mean(), y.std()))

    f98 = z['feats98'][m].astype(np.float32)
    ok98 = np.isfinite(f98).all(axis=1)
    if ok98.any():
        ml = build('arbiter')
        ml = load_state(ml, a.legacy)
        r, c = rmse(ml, f98[ok98], y[ok98])
        print('legacy V (98-dim): RMSE %.4f corr %.4f (n=%d)'
              % (r, c, int(ok98.sum())))
    if a.cand:
        f = z['feats'][m].astype(np.float32)
        mc = build(a.pkg, a.hid)
        mc = load_state(mc, a.cand)
        r, c = rmse(mc, f, y)
        print('candidate V (%s): RMSE %.4f corr %.4f (n=%d)'
              % (a.pkg, r, c, len(y)))


if __name__ == '__main__':
    main()
