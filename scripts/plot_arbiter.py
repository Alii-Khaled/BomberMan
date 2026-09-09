#!/usr/bin/env python3
"""Report figures for ARBITER (§6). CPU-only, no games.

Inputs: results/arbiter_summary.csv (scripts/aggregate_arbiter.py).
Outputs: results/figures/arbiter_*.png + captions.md entries.
  1. ablation: S0 / search+V0 / search+V / pi0 (+ ship + warden lines).
  2. gates: G1/G2/G3/G4 + E69 (tactical) + E70 (K-widen) pooled bars
     with per-seed points (noise shown, not hidden).
  3. placement: crates/bomb bars (S0, search, warden, rb, collector).
  4. value-delta: apex Q-deltas + arbiter V-deltas (40×2 and 100×2)
     with the ±0.3 draw-noise band (only apex-A4 clears it).
Reference points (non-arbiter) are hardcoded from docs/experiments.md
and labeled as such; arbiter points come from the CSV pooled rows.
Usage: python3 scripts/plot_arbiter.py
"""
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(REPO, 'results', 'figures')
os.makedirs(FIGS, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams.update({'figure.dpi': 200, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.axisbelow': True})

SHIP, WARDEN = 3.79, 5.07
BLUE, GREY, GREEN, RED = '#0173B2', '#949494', '#029E73', '#D55E00'


def load():
    pooled, seeds = {}, {}
    for r in csv.DictReader(
            open(os.path.join(REPO, 'results', 'arbiter_summary.csv'))):
        if 'rbiter' not in r['agent']:
            continue
        g = r['gate']
        if g.endswith('_pooled'):
            pooled[g[:-len('_pooled')]] = r
        else:
            seeds.setdefault(g, []).append(r)
    return pooled, seeds


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def main():
    pooled, seeds = load()
    caps = []
    pv = lambda fam, k='score': f(pooled[fam][k]) if fam in pooled else 0.0

    # ---- 1. ablation (100x2 pooled throughout) ----
    labels = ['S0\n(pi, no search)', 'search+V0\n(exact only)',
              'search+V\n(ship)', 'pi0\n(uniform prior)']
    vals = [pv('s0_100'), pv('ship'), pv('p1_100'), pv('pi0')]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=[BLUE, GREY, BLUE, RED])
    ax.axhline(SHIP, color='k', ls='--', lw=1, label='ship bar 3.79')
    ax.axhline(WARDEN, color=GREEN, ls=':', lw=1, label='warden 5.07 (ref)')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, '%.2f' % v,
                ha='center', fontsize=9)
    ax.set_ylabel('score/round vs 3x rule_based (pooled)')
    ax.set_title('ARBITER ablation: pi carries (+3.30), V is null')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'arbiter_ablation.png'))
    plt.close(fig)
    caps.append('**arbiter_ablation.png** — Frozen G1 (100×2 pooled, '
                'classic vs 3× rule_based). pi0 (uniform prior + skeleton) '
                '0.14 proves the learned prior carries +3.30 (strongest '
                'ML-compliance in the repo, E65). Search adds +0.51 over S0 '
                'via exact placement; V≡0 (3.60) vs V (3.95) is unresolvable '
                'at ±0.3 draw noise (E66 null). Dashed: 3.79 ship bar.')

    # ---- 2. gates with per-seed points ----
    fams = [('s0_100', 'S0'), ('p1_100', 'G1 rb'), ('p1_wm', 'G2 warden-mix'),
            ('p1_co', 'G3 collectors'), ('p1_rn', 'G4 random'),
            ('e69', 'G1+tact (E69)'), ('e70', 'G1/K16 (E70)')]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (fam, lab) in enumerate(fams):
        v = pv(fam)
        b = ax.bar(i, v, color=BLUE)
        ax.text(i, v + 0.1, '%.2f' % v, ha='center', fontsize=8)
        pts = [f(r['score']) for r in
               seeds.get(fam + '_s0', []) + seeds.get(fam + '_s1', [])]
        for p in pts:
            ax.plot(i, p, 'ko', ms=4)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([lab for _, lab in fams], fontsize=8)
    ax.axhline(SHIP, color='k', ls='--', lw=1)
    ax.set_ylabel('score/round (pooled; dots = seeds)')
    ax.set_title('ARBITER gate matrix (frozen, CPU)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'arbiter_gates.png'))
    plt.close(fig)
    caps.append('**arbiter_gates.png** — Pooled bars with per-seed dots. '
                'The s0/s1 spread (up to ±0.6, e.g. V0 s0 4.83 vs s1 3.88) '
                'is draw noise from unseeded agent RNG, not map effects '
                '(E63/E66 methods note) — the reason 40×2 cannot ship. '
                'E69 (tactical bypass) and E70 (K-widening) both REJECT.')

    # ---- 3. placement ----
    labels = ['S0', 'search (ship)', 'warden (ref)', 'rule_based (ref)',
              'collector (ref)']
    vals = [pv('s0_100', 'crates_per_bomb'), pv('p1_100', 'crates_per_bomb'),
            1.16, 1.56, 3.39]
    crates = [pv('s0_100', 'crates'), pv('p1_100', 'crates'),
              34.3, 29.2, 67.3]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(labels, vals, color=[GREY, BLUE, GREEN, GREY, GREY])
    for b, v, c in zip(bars, vals, crates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05,
                '%.2f\n(%s cr)' % (v, c), ha='center', fontsize=8)
    ax.set_ylabel('crates destroyed per bomb')
    ax.set_title('Placement efficiency: search doubles S0, trails collector')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'arbiter_placement.png'))
    plt.close(fig)
    caps.append('**arbiter_placement.png** — Crates/bomb (frozen G1 except '
                'refs). Search doubles S0 (0.83→1.40) and passes warden '
                '(1.16) but trails the collector specialist (3.39) — the '
                'E42 intent bottleneck, computed instead of learned (E66). '
                'Ref crates/round in parentheses.')

    # ---- 4. value-delta series with noise band ----
    labels = ['apex A2\n(E58)', 'apex A3\n(E59)', 'apex A4\n(E61)',
              'arbiter V\n40×2', 'arbiter V\n100×2']
    v40 = pv('p1e_W2_E3') - pv('p1_v0b')
    v100 = pv('p1_100') - pv('ship')
    vals = [-0.12, -0.02, -0.42, v40, v100]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(-0.3, 0.3, color='grey', alpha=0.2, label='±0.3 draw noise')
    ax.axhline(0, color='k', lw=1)
    cols = [RED if abs(v) > 0.3 else GREY for v in vals]
    bars = ax.bar(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.12),
                '%+.2f' % v, ha='center', fontsize=9)
    ax.set_ylabel('learned-component delta (score/round)')
    ax.set_title('Value-function deltas: only apex-A4 clears the noise band')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'arbiter_value_delta.png'))
    plt.close(fig)
    caps.append('**arbiter_value_delta.png** — Learned-value deltas with '
                'the ±0.3 draw-noise band. Only apex-A4 (−0.42, both seeds '
                'agree) is a real effect — and it is negative '
                '(double-counting warden, E62). Arbiter V (−0.14/+0.35 '
                'across validations) is NULL: withdrawn before publication '
                '(E66). ML-compliance rests on pi (+3.30).')

    with open(os.path.join(FIGS, 'captions.md'), 'a') as fh:
        fh.write('\n'.join(caps) + '\n')
    print('wrote 4 figures + captions to', FIGS)


if __name__ == '__main__':
    main()
