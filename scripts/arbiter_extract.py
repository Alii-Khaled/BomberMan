#!/usr/bin/env python3
"""ARBITER P0 extraction: materialize the joint pi/V training cache.

Inputs (S3 corpora, both validated 0-bad):
  apex-format  results/apex_demos/<teacher>/*.npz  (img u8 Tx12x17x17 x4,
      sc Tx16, act T) -> 98-dim feats via reconstruction + state_to_features
      (validated lossless, S3 probe) + margin-to-go labels from sc[:,12]
      (score_margin, clipped [-1,1]): vlabel_t = 10*(margin_T - margin_t).
  reaper-format results/demos/<teacher>[_<field>]/*.npz (feats Tx98,
      acts T,) -> pi rows only (no outcome channel: vlabel NaN).

Output: results/arbiter_p0_cache.npz with feats f32 (N,98), acts u8 (N,),
  vlabel f32 (N,) [NaN = pi-only], teacher u8 (N,), round i32 (N,),
  is_val bool (N,) [by-round split: hash(round_id)%10==0, ~10%].
  N ~= 340K pi rows (216K reaper + 123K apex), 123K V rows.

Teacher ids: 0 warden / 1 sentinel / 2 overlord / 3 collector.
  pi trains on {0,1,2} by default (collector's 0.59-suicide actions must
  not enter the prior); V trains on all four (collector states teach
  high-yield positions). Splits are env-overridable in pretrain.
Usage: python3 scripts/arbiter_extract.py [--out PATH] [--skip-reaper]
       [--reaper-include=prefix1,prefix2]

--skip-reaper (E88): omit all results/demos/* reaper-format rows.
--reaper-include (E88): keep only reaper-format dirs whose basename
starts with one of the comma-separated prefixes, e.g.
--reaper-include=arbiter_self_e88 keeps the corrected-feature self
corpus and drops the old buggy-feature warden/sentinel/overlord/
arbiter_self rows.
"""
import glob
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))

import numpy as np

# --pkg=arbiter_v2 selects the feature module (E99 feature-v2 candidate);
# reaper-format rows must carry that package's FEATURE_DIM.
import importlib
_PKG = 'arbiter'
for _a in sys.argv[1:]:
    if _a == '--pkg':
        _PKG = sys.argv[sys.argv.index(_a) + 1]
    elif _a.startswith('--pkg='):
        _PKG = _a.split('=', 1)[1]
AF = importlib.import_module(_PKG + '.features')

OUT = os.path.join(REPO, 'results', 'arbiter_p0_cache.npz')
SKIP_REAPER = '--skip-reaper' in sys.argv
REAPER_INCLUDE = None
for a in sys.argv[1:]:
    if a == '--out':
        OUT = sys.argv[sys.argv.index(a) + 1]
    elif a.startswith('--reaper-include='):
        REAPER_INCLUDE = [p for p in a.split('=', 1)[1].split(',') if p]

TEACHER_ID = {'warden': 0, 'warden_v1': 0, 'warden_v2': 0, 'sentinel': 1, 'overlord': 2,
              'coin_collector_agent': 3, 'collector': 3,
              'arbiter_self': 4, 'arbiter': 4,  # P2-C self-distillation
              'arbiter_v2_self': 4,  # E99 feature-v2 self corpus
              'da_warden': 5}  # E98 ship-state warden_v2 label transfer
# --warden-labels (E98): for dirs recorded with ARBITER_DAGGER_WARDEN=1,
# train pi on warden_v2's action at the ship's own visited states.
WARDEN_LABELS = '--warden-labels' in sys.argv


def teacher_of(path):
    base = os.path.basename(os.path.dirname(path)).lower()
    for k, v in TEACHER_ID.items():
        if base == k or base.startswith(k + '_'):
            return v
    raise ValueError('unknown teacher dir: %s' % path)


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
    F, A, V, T, R = [], [], [], [], []
    rnd = 0
    # apex-format: feats via reconstruction + margin-to-go labels
    apex_files = sorted(glob.glob(os.path.join(REPO, 'results/apex_demos/*/*.npz')))
    print('apex-format files: %d' % len(apex_files))
    for fi, f in enumerate(apex_files):
        d = np.load(f)
        img = d['img'].astype(np.float32) / 4.0
        sc = d['sc'].astype(np.float32)
        act = d['act'].astype(np.int64)
        assert set(np.unique(act)).issubset(set(range(6))), f
        margins = sc[:, 12].astype(np.float64) * 10.0
        vlab = (margins[-1] - margins).astype(np.float32)
        tid = teacher_of(f)
        feats = np.stack([AF.state_to_features(reconstruct(i, img, sc))
                          for i in range(len(act))]).astype(np.float32)
        F.append(feats)
        A.append(act.astype(np.uint8))
        V.append(vlab)
        T.append(np.full(len(act), tid, dtype=np.uint8))
        R.append(np.full(len(act), rnd, dtype=np.int32))
        rnd += 1
        if (fi + 1) % 100 == 0:
            print('  apex %d/%d' % (fi + 1, len(apex_files)), flush=True)
    # reaper-format: feats+acts direct, pi-only
    reaper_files = sorted(glob.glob(os.path.join(REPO, 'results/demos/*/*.npz')))
    if SKIP_REAPER:
        reaper_files = []
    if REAPER_INCLUDE:
        reaper_files = [
            f for f in reaper_files
            if any(os.path.basename(os.path.dirname(f)).startswith(p)
                   for p in REAPER_INCLUDE)]
    print('reaper-format files: %d%s%s' % (
        len(reaper_files),
        ' (skipped)' if SKIP_REAPER else '',
        ' (include=%s)' % ','.join(REAPER_INCLUDE) if REAPER_INCLUDE else ''))
    for f in reaper_files:
        d = np.load(f)
        feats = d['feats'].astype(np.float32)
        act = d['acts'].astype(np.int64)
        tid = teacher_of(f)
        if WARDEN_LABELS and 'acts_w' in d.files:
            act = d['acts_w'].astype(np.int64)
            tid = TEACHER_ID['da_warden']
        assert feats.shape[1] == AF.FEATURE_DIM and len(act) == len(feats), f
        assert set(np.unique(act)).issubset(set(range(6))), f
        F.append(feats)
        A.append(act.astype(np.uint8))
        V.append(np.full(len(act), np.nan, dtype=np.float32))
        T.append(np.full(len(act), tid, dtype=np.uint8))
        R.append(np.full(len(act), rnd, dtype=np.int32))
        rnd += 1
    F = np.concatenate(F)
    A = np.concatenate(A)
    V = np.concatenate(V)
    T = np.concatenate(T)
    R = np.concatenate(R)
    is_val = (R % 10 == 0)
    assert np.isfinite(F).all()
    print('rows: %d pi (V rows: %d) | rounds: %d | val rows: %d (%.1f%%)' % (
        len(F), int(np.isfinite(V).sum()), rnd, int(is_val.sum()),
        100.0 * is_val.mean()))
    print('teachers:', {int(t): int((T == t).sum()) for t in sorted(set(T.tolist()))})
    print('V label: mean %.3f std %.3f min %.2f max %.2f' % (
        float(np.nanmean(V)), float(np.nanstd(V)), float(np.nanmin(V)),
        float(np.nanmax(V))))
    tmp = OUT + '.part'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, feats=F, acts=A, vlabel=V, teacher=T,
                            round=R, is_val=is_val)
    os.replace(tmp, OUT)
    print('wrote', OUT, '%.1f MB' % (os.path.getsize(OUT) / 1e6))


if __name__ == '__main__':
    main()
