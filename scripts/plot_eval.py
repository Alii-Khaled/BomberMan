"""Report figures for the sentinel project (Methods/Results).

Inputs: agent_code/sentinel/runs/metrics.csv, results/eval_summary_ci.csv,
        results/eval_rounds.csv
Outputs: results/figures/*.png + *.pdf, results/figures/captions.md
Usage: .venv/bin/python scripts/plot_eval.py

Conventions:
  * Fixed agent -> color map (sentinel always saturated, opponents muted).
  * Training curves are split by detected curriculum stage; no rolling
    window or EMA ever crosses a stage boundary (regimes are incomparable:
    coin-heaven ~50 reward vs classic ~negative).
  * Game-score proxy = coins + 5*kills per episode (tournament scoring).
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
FIGS = os.path.join(RESULTS, 'figures')
os.makedirs(FIGS, exist_ok=True)

plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams.update({
    'figure.dpi': 200,
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

# Fixed color per agent (stable across all figures; sentinel highlighted).
AGENT_COLORS = {
    'sentinel': '#0173B2',
    'rule_based_agent': '#DE8F05',
    'warden_v1': '#029E73',
    'warden_v2': '#029E73',
    'coin_collector_agent': '#CC78BC',
    'peaceful_agent': '#949494',
    'random_agent': '#CA9161',
    'overlord': '#D55E00',
}
FALLBACK_COLORS = ['#56B4E9', '#949494', '#0173B2', '#029E73', '#DE8F05',
                   '#CC78BC', '#D55E00', '#CA9161']


def agent_color(name, i=0):
    for key, color in AGENT_COLORS.items():
        if name == key or name.startswith(key):
            return color
    return FALLBACK_COLORS[i % len(FALLBACK_COLORS)]


STAGE_NAMES = ['Stage 1: coin-heaven solo', 'Stage 2: classic solo',
               'Stage 3: hunting (peaceful+collector)',
               'Stage 4: vs rule_based', 'Stage 5', 'Stage 6']
# Known curriculum boundaries (episode numbers). Resumes/smoke runs inside a
# stage reset the buffer too, so pure buffer-drop detection oversplits —
# these anchors pin the true stages; auto-detection only adds splits beyond
# the last anchor.
STAGE_BOUNDARIES = [1, 501, 2001, 3041]
MATCHUP_ORDER = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']
MATCHUP_LABELS = {
    'm1': 'solo\ncoin-heaven', 'm2': 'solo\nclassic', 'm3': 'vs 3x\nrandom',
    'm4': 'vs 3x\npeaceful', 'm5': 'vs 3x\ncollector',
    'm6': 'vs 3x\nrule-based', 'm7': 'vs overlord\n+2x rule-based',
    'm8': 'vs warden\n+2x rule-based',
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(rows, key, cast=float):
    return [cast(r[key]) for r in rows]


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name + '.png'))
    fig.savefig(os.path.join(FIGS, name + '.pdf'))
    plt.close(fig)


def rolling(vals, w):
    out = []
    for i in range(len(vals)):
        seg = vals[max(0, i - w + 1):i + 1]
        out.append(sum(seg) / len(seg))
    return out


def rolling_median(vals, w):
    out = []
    for i in range(len(vals)):
        seg = sorted(vals[max(0, i - w + 1):i + 1])
        out.append(seg[len(seg) // 2])
    return out


def detect_stages(rows):
    """Split episodes at replay-buffer resets (curriculum stage = new process).

    Robust to concurrent-writer interleaving: sorts by episode, drops
    duplicate episode numbers (last write wins), then detects buffer drops.
    Returns (ordered_rows, [(start, end_exclusive, label)]). Never empty.
    """
    tagged = []
    for i, r in enumerate(rows):
        try:
            tagged.append((int(r['episode']), i))
        except (ValueError, TypeError, KeyError):
            pass
    if not tagged:
        return rows, [(0, len(rows), STAGE_NAMES[0])]
    tagged.sort()
    order, last_ep = [], None
    for ep, i in tagged:
        if ep == last_ep:
            order[-1] = i  # duplicate episode: last write wins
        else:
            order.append(i)
            last_ep = ep
    orows = [rows[i] for i in order]
    max_ep = max(int(r['episode']) for r in orows)
    # Anchor on known curriculum boundaries; auto-detect only beyond them.
    bounds = sorted({b for b in STAGE_BOUNDARIES if 1 <= b <= max_ep})
    if not bounds:
        bounds = [1]
    # buffer-drop splits only past the last anchor
    last_anchor = bounds[-1]
    bufs = {}
    for k, r in enumerate(orows):
        try:
            bufs.setdefault(int(r['episode']), float(r.get('buffer') or 0))
        except (ValueError, TypeError):
            pass
    eps_sorted = sorted(bufs)
    for prev, cur in zip(eps_sorted, eps_sorted[1:]):
        if cur < last_anchor:
            continue
        if bufs[cur] < 0.5 * bufs[prev] and bufs[prev] > 1000:
            if cur not in bounds:
                bounds.append(cur)
    bounds = sorted(bounds)
    # convert episode bounds to row positions
    ep_of = [int(r['episode']) for r in orows]
    pos = []
    for b in bounds:
        pos.extend([k for k, e in enumerate(ep_of) if e >= b][:1])
    pos = sorted(set(pos)) + [len(orows)]
    stages = []
    for s in range(len(pos) - 1):
        if pos[s] >= pos[s + 1]:
            continue
        label = STAGE_NAMES[s] if s < len(STAGE_NAMES) else f'Stage {s + 1}'
        stages.append((pos[s], pos[s + 1], label))
    return orows, stages


def game_score(rows):
    return [c + 5 * k for c, k in zip(fnum(rows, 'coins', int),
                                      fnum(rows, 'kills', int))]


def fig_performance_split():
    path = 'agent_code/sentinel/runs/metrics.csv'
    if not os.path.isfile(path):
        print('skip performance_split (no metrics.csv)')
        return
    rows = read_csv(path)
    if len(rows) < 10:
        print('skip performance_split (too few episodes)')
        return
    rows, stages = detect_stages(rows)
    n = len(stages)
    scores, coins = game_score(rows), fnum(rows, 'coins', int)
    suic = fnum(rows, 'suicides', int)
    ep = fnum(rows, 'episode', int)

    fig, axes = plt.subplots(3, n, figsize=(3.8 * n + 1.5, 7),
                             sharex='col', squeeze=False)
    for j, (a, b, label) in enumerate(stages):
        e = ep[a:b]
        # raw faint + rolling bold, window restarts at the boundary
        for i, (vals, title, unit, w) in enumerate((
                (scores[a:b], 'Game score / episode', 'coins + 5·kills', 25),
                (coins[a:b], 'Coins / episode', 'coins', 25),
                (suic[a:b], 'Suicides / episode', 'rolling-50 rate', 50))):
            ax = axes[i][j]
            roll = rolling(vals, w)
            ax.scatter(e, vals, s=4, alpha=0.25, color=AGENT_COLORS['sentinel'])
            ax.plot(e, roll, lw=1.8, color=AGENT_COLORS['sentinel'])
            if i == 0:
                ax.set_title(f'{label}\n(ep {e[0]}–{e[-1]}, n={len(e)})')
            ax.set_ylabel(f'{title}\n[{unit}]')
            if vals and max(vals) == min(vals):
                ax.set_ylim(min(vals) - 0.5, max(vals) + 0.5)
            # recovery slope on the last stage's coins panel
            if i == 1 and j == n - 1 and len(e) > 30:
                x0, x1 = e[10], e[-1]
                y0 = sum(roll[10:20]) / 10
                y1 = sum(roll[-10:]) / 10
                slope = (y1 - y0) / max(1, x1 - x0) * 100
                ax.annotate(f'recovery {slope:+.1f} coins/100ep',
                            xy=(x1, y1), xytext=(0.05, 0.88),
                            textcoords='axes fraction', fontsize=8,
                            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    for j in range(n):
        axes[2][j].set_xlabel('training episode')
    fig.suptitle('Sentinel skill over training (per-stage scales; windows never cross stages)')
    savefig(fig, 'training_performance_split')
    print('stages:', [(s, ep[a], ep[b - 1]) for a, b, s in stages])


def fig_dynamics():
    path = 'agent_code/sentinel/runs/metrics.csv'
    if not os.path.isfile(path):
        return
    rows = read_csv(path)
    if len(rows) < 10:
        return
    rows, stages = detect_stages(rows)
    ep = fnum(rows, 'episode', int)
    loss = fnum(rows, 'loss')
    eps = fnum(rows, 'epsilon')
    steps = fnum(rows, 'total_steps', int)
    has_duty = 'duty_pct' in rows[0]
    duty = [float(r['duty_pct']) if r.get('duty_pct') not in (None, '') else float('nan')
            for r in rows] if has_duty else None

    nrows = 4 if has_duty else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(7.5, 2.4 * nrows + 0.5), sharex=True)
    # loss: rolling median, log scale, skip buffer-refill zeros
    med = rolling_median([v if v > 0 else float('nan') for v in loss], 25)
    import math
    med = [v for v in med]
    axes[0].plot(ep, med, lw=1.2, color=AGENT_COLORS['sentinel'])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('TD loss [rolling median-25, log]')
    axes[1].plot(ep, eps, lw=1.2, color=AGENT_COLORS['rule_based_agent'])
    axes[1].set_ylabel('epsilon (exploration)')
    axes[2].plot(ep, steps, lw=1.2, color=AGENT_COLORS['warden_v1'])
    axes[2].set_ylabel('gradient steps (cumulative)')
    if has_duty:
        import math as _m
        droll = []
        for i in range(len(duty)):
            seg = [v for v in duty[max(0, i - 24):i + 1] if v == v]
            droll.append(sum(seg) / len(seg) if seg else float('nan'))
        axes[3].plot(ep, droll, lw=1.2, color=AGENT_COLORS['overlord'])
        axes[3].set_ylim(0, 100)
        axes[3].set_ylabel('GPU duty % [rolling-25]')
        axes[3].set_xlabel('training episode')
    else:
        axes[2].set_xlabel('training episode')
    for ax in axes:
        # shade stages + boundary lines
        bounds = [ep[a] for a, _, _ in stages] + [ep[-1]]
        for k in range(len(stages)):
            ax.axvspan(bounds[k], bounds[k + 1] if k + 1 < len(bounds) else ep[-1],
                       color='grey', alpha=0.06)
            if k > 0:
                ax.axvline(bounds[k], color='black', lw=1.0, ls='--')
    axes[0].set_title('Sentinel learning dynamics (shaded = curriculum stages)')
    savefig(fig, 'training_dynamics')


def ci_table():
    rows = read_csv(os.path.join(RESULTS, 'eval_summary_ci.csv'))
    t = defaultdict(dict)
    for r in rows:
        t[(r['matchup'], r['agent'])][r['metric']] = (float(r['mean']), float(r['ci95']))
    return t


def matchup_agents(t, m):
    agents = sorted({a for (mm, a) in t if mm == m},
                    key=lambda a: (not a.startswith('sentinel'), a))
    return agents


def fig_head_to_head():
    t = ci_table()
    present = [m for m in MATCHUP_ORDER if any(mm == m for mm, _ in t)]
    if not present:
        print('skip head_to_head (no data)')
        return
    n = len(present)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.6 * cols + 1, 3.2 * rows_n + 0.8),
                             squeeze=False)
    for idx, m in enumerate(present):
        ax = axes[idx // cols][idx % cols]
        agents = matchup_agents(t, m)
        means = [t[(m, a)]['score_pr'][0] for a in agents]
        cis = [t[(m, a)]['score_pr'][1] for a in agents]
        colors = [agent_color(a, i) for i, a in enumerate(agents)]
        bars = ax.bar(range(len(agents)), means, yerr=cis, capsize=3, color=colors)
        short = [a.split('_')[0] if '_' in a else a for a in agents]
        ax.set_xticks(range(len(agents)), short, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('score / round')
        # dominance delta: sentinel minus best opponent
        s = means[agents.index('sentinel')] if 'sentinel' in agents else None
        opp = max([v for a, v in zip(agents, means) if a != 'sentinel'], default=None)
        title = MATCHUP_LABELS.get(m, m).replace('\n', ' ')
        if s is not None and opp is not None:
            title += f'  (Δ {s - opp:+.2f})'
        ax.set_title(title, fontsize=9)
        for b, a in zip(bars, agents):
            if a == 'sentinel' or a.startswith('sentinel'):
                b.set_edgecolor('black')
                b.set_linewidth(1.5)
    for idx in range(n, rows_n * cols):
        axes[idx // cols][idx % cols].axis('off')
    fig.suptitle('Head-to-head score/round ±95% CI (black edge = sentinel; Δ vs best opponent)')
    savefig(fig, 'eval_head_to_head')


def fig_safety_panel():
    t = ci_table()
    present = [m for m in MATCHUP_ORDER if ('sentinel' in [a for (mm, a) in t if mm == m])]
    if not present:
        print('skip safety_panel (no data)')
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, metric, title in zip(axes,
                                 ('suicides_pr', 'steps_pr', 'invalid_pr'),
                                 ('Suicides / round', 'Steps / round (survival)',
                                  'Invalid actions / round')):
        means = [t[(m, 'sentinel')][metric][0] for m in present]
        cis = [t[(m, 'sentinel')][metric][1] for m in present]
        ax.bar(range(len(present)), means, yerr=cis, capsize=3,
               color=AGENT_COLORS['sentinel'])
        ax.set_xticks(range(len(present)),
                      [MATCHUP_LABELS.get(m, m) for m in present], fontsize=7)
        ax.set_title(title)
    fig.suptitle('Sentinel safety panel ±95% CI')
    savefig(fig, 'eval_safety_panel')


def fig_distributions():
    path = os.path.join(RESULTS, 'eval_rounds.csv')
    if not os.path.isfile(path):
        print('skip distributions (no eval_rounds.csv)')
        return
    rows = read_csv(path)
    by_m = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ('steps', 'coins', 'kills'):
            by_m[r['matchup']][k].append(int(r[k]))
    order = [m for m in MATCHUP_ORDER if m in by_m]
    if not order:
        print('skip distributions (no data)')
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, key, title in zip(axes, ('steps', 'coins', 'kills'),
                              ('Round length (steps)', 'Game coins / round',
                               'Game kills / round')):
        ax.boxplot([by_m[m][key] for m in order], tick_labels=order, showfliers=False)
        ax.set_title(f'{title} (n={[len(by_m[m][key]) for m in order][0]} rounds)')
    fig.suptitle('Per-round distributions per matchup')
    savefig(fig, 'eval_distributions')


def fig_thinktime():
    t = ci_table()
    present = [m for m in MATCHUP_ORDER if ('sentinel' in [a for (mm, a) in t if mm == m])]
    if not present:
        print('skip thinktime (no data)')
        return
    means = [t[(m, 'sentinel')]['ms_per_step'][0] for m in present]
    cis = [t[(m, 'sentinel')]['ms_per_step'][1] for m in present]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(present)), means, yerr=cis, capsize=3,
           color=AGENT_COLORS['sentinel'])
    ax.axhline(500, color='red', ls='--', lw=1.2, label='tournament limit 500 ms')
    ax.set_yscale('log')
    ax.set_xticks(range(len(present)), [MATCHUP_LABELS.get(m, m) for m in present],
                  fontsize=8)
    ax.set_ylabel('think time ms/step [log]')
    ax.set_title('Sentinel inference cost vs tournament limit')
    ax.legend(fontsize=8)
    savefig(fig, 'eval_thinktime')


CAPTIONS = {
    'training_performance_split': 'Skill over training, split by curriculum stage '
        '(columns have independent y-scales; rolling windows restart at stage '
        'boundaries). Game score = coins + 5·kills per episode. Faint dots are '
        'raw episodes, bold lines rolling means (w=25, suicide rate w=50).',
    'training_dynamics': 'Learning diagnostics: TD loss (rolling median, log scale; '
        'gap = buffer refill after stage restart), epsilon schedule, cumulative '
        'gradient steps. Shaded bands are curriculum stages.',
    'eval_head_to_head': 'Frozen-model score per round ±95% CI across seeds, all '
        'agents per matchup. Black edge marks sentinel; Δ is sentinel minus the '
        'best opponent.',
    'eval_safety_panel': 'Safety: suicides, survival (round length) and invalid '
        'actions per round ±95% CI for sentinel across matchups.',
    'eval_distributions': 'Per-round game totals (all agents) per matchup; '
        'boxes hide outliers for readability.',
    'eval_thinktime': 'Sentinel inference cost per step (log scale) against the '
        '500 ms tournament limit.',
}


def write_captions():
    with open(os.path.join(FIGS, 'captions.md'), 'w') as f:
        f.write('# Figure captions (report paste-ready)\n\n')
        for name in sorted(CAPTIONS):
            png = os.path.join(FIGS, name + '.png')
            status = 'rendered' if os.path.isfile(png) else 'pending data'
            f.write(f'## {name} ({status})\n{CAPTIONS[name]}\n\n')


def main():
    need_eval = os.path.isfile(os.path.join(RESULTS, 'eval_summary_ci.csv'))
    fig_performance_split()
    fig_dynamics()
    if need_eval:
        fig_head_to_head()
        fig_safety_panel()
        fig_distributions()
        fig_thinktime()
    else:
        print('eval CSVs missing: run eval matrix + aggregate_eval.py first')
    write_captions()
    print('figures in', FIGS)


if __name__ == '__main__':
    main()
