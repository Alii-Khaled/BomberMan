"""Aggregate frozen-eval JSONs into report tables.

Reads results/eval_<matchup>_seed<seed>.json (main.py --save-stats format:
by_agent lifetime stats + by_round per-round game totals).
Writes:
  results/eval_summary.csv    - per seed x matchup x agent, per-round rates
  results/eval_summary_ci.csv - mean/std/95% CI across seeds
  results/eval_rounds.csv     - per-round game totals (for distributions)
Usage: .venv/bin/python scripts/aggregate_eval.py
"""
import csv
import glob
import json
import math
import os
import re

RESULTS = 'results'
MATCHUP_LABELS = {
    'm1': 'solo coin-heaven',
    'm2': 'solo classic',
    'm3': 'vs 3x random',
    'm4': 'vs 3x peaceful',
    'm5': 'vs 3x coin_collector',
    'm6': 'vs 3x rule_based',
    'm7': 'vs overlord+2x rule_based',
    'm8': 'vs warden_v1+2x rule_based',
}


def per_round_rates(stats, rounds):
    rounds = max(1, int(rounds))
    out = {}
    for k in ('score', 'coins', 'kills', 'suicides', 'bombs', 'moves', 'invalid', 'steps'):
        out[k + '_pr'] = stats.get(k, 0) / rounds
    t = stats.get('time', 0.0)
    steps = stats.get('steps', 0)
    out['ms_per_step'] = (1000.0 * t / steps) if steps else 0.0
    return out


def main():
    files = sorted(glob.glob(os.path.join(RESULTS, 'eval_*_seed*.json')))
    if not files:
        print('no eval files found in results/')
        return
    summary_rows = []
    round_rows = []
    for path in files:
        m = re.search(r'eval_(m\d+)_seed(\d+)', os.path.basename(path))
        if not m:
            continue
        matchup, seed = m.group(1), int(m.group(2))
        with open(path) as f:
            data = json.load(f)
        by_agent = data.get('by_agent', {})
        by_round = data.get('by_round', {})
        for agent, stats in by_agent.items():
            rounds = stats.get('rounds', len(by_round)) or len(by_round) or 1
            row = {'matchup': matchup, 'label': MATCHUP_LABELS.get(matchup, matchup),
                   'seed': seed, 'agent': agent, 'rounds': rounds}
            row.update(per_round_rates(stats, rounds))
            summary_rows.append(row)
        for i, (rid, rstats) in enumerate(by_round.items()):
            round_rows.append({'matchup': matchup, 'seed': seed, 'round_idx': i,
                               'steps': rstats.get('steps', 0),
                               'coins': rstats.get('coins', 0),
                               'kills': rstats.get('kills', 0),
                               'suicides': rstats.get('suicides', 0)})
    with open(os.path.join(RESULTS, 'eval_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(os.path.join(RESULTS, 'eval_rounds.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(round_rows[0].keys()))
        w.writeheader()
        w.writerows(round_rows)

    # rollup across seeds
    from collections import defaultdict
    groups = defaultdict(list)
    for r in summary_rows:
        groups[(r['matchup'], r['label'], r['agent'])].append(r)
    metrics = [c for c in summary_rows[0] if c.endswith('_pr') or c == 'ms_per_step']
    ci_rows = []
    for (matchup, label, agent), rows in sorted(groups.items()):
        n = len(rows)
        for met in metrics:
            vals = [r[met] for r in rows]
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
            ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
            ci_rows.append({'matchup': matchup, 'label': label, 'agent': agent,
                            'metric': met, 'n_seeds': n,
                            'mean': round(mean, 4), 'std': round(std, 4),
                            'ci95': round(ci, 4)})
    with open(os.path.join(RESULTS, 'eval_summary_ci.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ci_rows[0].keys()))
        w.writeheader()
        w.writerows(ci_rows)
    print(f'aggregated {len(files)} files -> eval_summary.csv, eval_summary_ci.csv, eval_rounds.csv')

    # LaTeX table: sentinel row per matchup
    lines = ['\\begin{tabular}{lccccc}',
             '\\hline',
             'Matchup & Rounds & Score/round & Coins/round & Kills/round & Suicides/round \\\\',
             '\\hline']
    for (matchup, label, agent), rows in sorted(groups.items()):
        if 'sentinel' not in agent:
            continue
        n = len(rows)
        total_rounds = sum(r['rounds'] for r in rows)

        def fmt(met):
            vals = [r[met] for r in rows]
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
            return f'{mean:.2f} $\\pm$ {std:.2f}'

        lines.append(f"{label} & {total_rounds} & {fmt('score_pr')} & {fmt('coins_pr')} & "
                     f"{fmt('kills_pr')} & {fmt('suicides_pr')} \\\\")
    lines += ['\\hline', '\\end{tabular}']
    with open(os.path.join(RESULTS, 'eval_summary.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote eval_summary.tex')


if __name__ == '__main__':
    main()
