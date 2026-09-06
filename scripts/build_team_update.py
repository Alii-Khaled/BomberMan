"""Team progress update PDF: visual, AI- and stages-focused.

Reads live state (metrics, stage JSONs, eval CI, figures) and writes
results/team_update.pdf via matplotlib PdfPages (no extra deps):
  p1 cover: title banner + KPI cards + curriculum timeline
  p2 AI approach: architecture, learning loop, infra wins
  p3 stages: per-stage goal/config/outcome/what-was-learned cards
  p4-5 training figures | p6 frozen eval + outlook
Usage: .venv/bin/python scripts/build_team_update.py
"""
import csv
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_eval import read_csv, detect_stages  # noqa: E402

RESULTS = 'results'
FIGS = os.path.join(RESULTS, 'figures')
OUT = os.path.join(RESULTS, 'team_update.pdf')

BLUE = '#0173B2'
TEAL = '#029E73'
ORANGE = '#DE8F05'
GREY = '#949494'
INK = '#1a1a1a'

STAGE_META = [
    {'key': 'Stage 1', 'name': 'coin-heaven solo', 'task': 'Task 1: navigation',
     'cfg': '0 crates, 50 free coins, solo',
     'learn': 'BFS pathfinding features -> movement; nearest-coin pursuit. '
              'Converged: coins 11 -> 51/round, zero bombs needed, zero deaths.'},
    {'key': 'Stage 2', 'name': 'classic solo', 'task': 'Task 2: bombs + escape',
     'cfg': '0.75 crate density, 9 hidden coins, solo',
     'learn': 'Crate-adjacent bombing, escape-route planning, suicide avoidance. '
              'Regime shock at ep 501, then suicide rate 0.85 -> ~0.3.'},
    {'key': 'Stage 3', 'name': 'hunting', 'task': 'Task 3: vs peaceful + coin_collector',
     'cfg': 'classic, 2 opponents',
     'learn': 'Kill setups, opponent blast timing, hunting vs economy tradeoff. '
              'Result: 102 kills (0.49/round); economy still lags combat.'},
    {'key': 'Stage 4', 'name': 'vs rule_based', 'task': 'Task 4: full combat (LIVE)',
     'cfg': 'classic, 3x rule_based + Lion + UTD3 + batch 512',
     'learn': 'Combat under pressure, kill stealing/avoidance, endgame hunting. '
              'Gate: beat rule_based score in frozen eval.'},
]


def stage_means(rows, stages):
    out = []
    for a, b, label in stages:
        seg = rows[a:b]
        n = max(1, len(seg))
        m = lambda k: sum(float(r.get(k) or 0) for r in seg) / n
        out.append({'ep': (int(seg[0]['episode']), int(seg[-1]['episode']), n),
                    'coins': m('coins'), 'kills': m('kills'),
                    'suicides': m('suicides'),
                    'score': m('coins') + 5 * m('kills')})
    return out


def banner(ax, title, subtitle):
    ax.add_patch(Rectangle((0, 0.82), 1, 0.18, fc=BLUE, ec='none',
                           transform=ax.transAxes))
    ax.text(0.04, 0.93, title, fontsize=20, weight='bold', color='white',
            va='center', transform=ax.transAxes)
    ax.text(0.04, 0.86, subtitle, fontsize=10, color='white', va='center',
            transform=ax.transAxes)


def kpi_row(fig, y, kpis):
    n = len(kpis)
    for i, (val, lab) in enumerate(kpis):
        ax = fig.add_axes([0.04 + i * 0.92 / n, y, 0.92 / n - 0.02, 0.10])
        ax.axis('off')
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0.02',
                                    fc='#f2f6fa', ec=BLUE, lw=1.2,
                                    transform=ax.transAxes))
        ax.text(0.5, 0.62, val, fontsize=15, weight='bold', color=INK,
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.28, lab, fontsize=8, color='#444444',
                ha='center', va='center', transform=ax.transAxes)


def page_cover(pdf, rows, last, means):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    banner(ax, 'Bomberman RL — team update',
           datetime.now().strftime('%Y-%m-%d %H:%M') +
           '  ·  Sentinel (Dueling DQN)  ·  RX 6900 XT / DirectML')
    ep = int(last['episode'])
    steps = int(last['total_steps'])
    duty = last.get('duty_pct', '—')
    kpi_row(fig, 0.66, [
        (f'{ep}', 'training episodes'),
        (f'{steps / 1e6:.2f}M', 'gradient steps'),
        (f"{means[2]['kills']:.2f}/rd" if len(means) > 2 else '—', 'Stage-3 kills/round'),
        (f'{duty}%', 'GPU duty (Stage 4)'),
    ])
    # curriculum timeline (widths ~ episode counts)
    tax = fig.add_axes([0.06, 0.46, 0.88, 0.16])
    tax.axis('off')
    tax.text(0, 0.88, 'Curriculum', fontsize=12, weight='bold', color=INK)
    total = sum(m['ep'][2] for m in means)
    x, colors = 0.0, [TEAL, TEAL, TEAL, ORANGE]
    for i, m in enumerate(means):
        w = max(0.06, m['ep'][2] / total * 0.96)
        tax.add_patch(FancyBboxPatch((x, 0.05), w, 0.62, boxstyle='round,pad=0.01',
                                     fc=colors[i] if i < len(colors) else GREY,
                                     ec='none'))
        nm = STAGE_META[i]['key'] if i < len(STAGE_META) else ''
        tax.text(x + w / 2, 0.45, nm, fontsize=9, weight='bold', color='white',
                 ha='center', va='center')
        tax.text(x + w / 2, 0.20, f"ep {m['ep'][0]}-{m['ep'][1]}",
                 fontsize=7.5, color='white', ha='center', va='center')
        x += w + 0.01
    # AI snapshot box
    bax = fig.add_axes([0.06, 0.03, 0.88, 0.40])
    bax.axis('off')
    bax.text(0, 0.96, 'AI snapshot', fontsize=12, weight='bold', color=INK)
    bullets = [
        'Agent: DuelingMLP (46 handcrafted feats -> 256 -> 256 -> V+A head).',
        'Learning: N-step (3) Double DQN + prioritized replay + safety mask +',
        '    heuristic blend; epsilon-greedy exploration (now floored at 0.05).',
        'Optimizer path: stock Adam -> DML-safe DMLAdam -> benchmarked Lion',
        '    winner (1.57x faster step) + warmup-cosine schedule (Stage 4).',
        'Throughput kit: 3 updates/step, batch 512, GPU-duty logging (95%).',
        'Safety net: atomic versioned checkpoints (worst loss: 1 round).',
    ]
    bax.text(0, 0.86, '\n'.join(bullets), fontsize=10, va='top', color=INK,
             family='monospace')
    pdf.savefig(fig)
    plt.close(fig)


def page_stages(pdf, means):
    fig = plt.figure(figsize=(8.27, 11.69))
    outer = fig.add_axes([0, 0, 1, 1])
    outer.axis('off')
    outer.add_patch(Rectangle((0, 0.86), 1, 0.14, fc='#0173B2', ec='none',
                              transform=outer.transAxes))
    outer.text(0.04, 0.945, 'Training stages — what each one teaches',
               fontsize=16, weight='bold', color='white', va='center',
               transform=outer.transAxes)
    outer.text(0.04, 0.885,
               'One row per curriculum stage: goal, setup, measured outcome',
               fontsize=9, color='white', va='center', transform=outer.transAxes)
    for i in range(4):
        ax = fig.add_axes([0.06, 0.68 - i * 0.16, 0.88, 0.145])
        ax.axis('off')
        meta = STAGE_META[i] if i < len(STAGE_META) else {}
        m = means[i] if i < len(means) else None
        live = (i == 3)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0.01',
                                    fc='#fff8ec' if live else '#f2f6fa',
                                    ec=ORANGE if live else BLUE, lw=1.5))
        ax.text(0.02, 0.84, f"{meta.get('key', '')}: {meta.get('name', '')}",
                fontsize=11, weight='bold', color=INK, transform=ax.transAxes)
        ax.text(0.02, 0.68, meta.get('task', ''), fontsize=8.5,
                style='italic', color='#444444', transform=ax.transAxes)
        ax.text(0.02, 0.52, f"Setup: {meta.get('cfg', '')}", fontsize=8.5,
                transform=ax.transAxes)
        if m:
            ax.text(0.02, 0.36,
                    f"Measured: score {m['score']:.2f}  ·  coins {m['coins']:.2f}  ·  "
                    f"kills {m['kills']:.2f}  ·  suicides {m['suicides']:.2f} /round  "
                    f"(n={m['ep'][2]}, ep {m['ep'][0]}-{m['ep'][1]})",
                    fontsize=8.5, weight='bold', color=BLUE, transform=ax.transAxes)
        ax.text(0.02, 0.14, f"Learned: {meta.get('learn', '')}", fontsize=8,
                va='top', color=INK, transform=ax.transAxes, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def page_fig(pdf, png, title, note=''):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=13, weight='bold')
    img = plt.imread(os.path.join(FIGS, png))
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    ax.imshow(img)
    ax.axis('off')
    if note:
        fig.text(0.08, 0.04, note, fontsize=8, style='italic')
    pdf.savefig(fig)
    plt.close(fig)


def main():
    rows = read_csv('agent_code/sentinel/runs/metrics.csv')
    orows, stages = detect_stages(rows)
    means = stage_means(orows, stages)
    last = orows[-1]
    with PdfPages(OUT) as pdf:
        page_cover(pdf, orows, last, means)
        page_stages(pdf, means)
        page_fig(pdf, 'training_performance_split.png',
                 'Training skill (per-stage scales)',
                 'Game score = coins + 5·kills. Rolling windows restart at stage '
                 'boundaries; no smoothing crosses regimes.')
        page_fig(pdf, 'training_dynamics.png', 'Learning dynamics',
                 'TD loss (log) rises with combat complexity; epsilon floored '
                 'since Stage 1; duty panel shows Stage-4 utilization work.')
        page_fig(pdf, 'eval_head_to_head.png',
                 'Frozen eval vs opponents (±95% CI)',
                 'NOTE: pre-Stage-4 model snapshot; re-eval runs after Stage 4. '
                 'Next: finish Stage 4 -> frozen eval -> report sections 5/6 '
                 '(agent code 21.09, report 28.09).')
    print(f'wrote {OUT} ({os.path.getsize(OUT) // 1024} KiB)')


if __name__ == '__main__':
    main()
