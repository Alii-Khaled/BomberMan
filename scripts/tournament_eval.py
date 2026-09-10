#!/usr/bin/env python3
"""Tournament-mode win-rate harness (E88/P1.3).

The repo's gates score pooled points/round; the project tournament ranks
agents by TOTAL score, so we also need per-round winners and ranks. This
runner drives BombeRLeWorld directly (no per-round agent scores exist in
--save-stats output) and records, per round:

  * score, and the cumulative kills/suicides/coins from lifetime stats
  * round winner (strict max score; ties share the win)
  * rank (1 = best; ties get the average rank)

Outputs a JSON with per-round scores + per-agent summary (mean score,
round-win rate, mean rank, bootstrap 95% CIs over rounds) and prints a
table. Evaluation-only: never part of the tournament agent zip.

Usage:
  python3 scripts/tournament_eval.py --agents arbiter rule_based_agent \
      rule_based_agent rule_based_agent --n-rounds 40 --seed 0 \
      --out results/tourney_ship_rb_s0.json
"""
import argparse
import json
import os
import sys
from time import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np


def _boot_ci(vals, n=2000, rng=None):
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return [0.0, 0.0]
    if rng is None:
        rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agents', nargs='+', required=True)
    p.add_argument('--n-rounds', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--scenario', default='classic')
    p.add_argument('--out', default=None)
    p.add_argument('--log-dir', default=os.path.join(REPO, 'logs'))
    p.add_argument('--match-name', default=None)
    p.add_argument('--silence', action='store_true', default=True)
    a = p.parse_args()

    from environment import BombeRLeWorld, WorldArgs
    args = WorldArgs(
        no_gui=True, fps=0, turn_based=False, update_interval=0.1,
        save_replay=False, replay=False, make_video=False,
        continue_without_training=True, log_dir=a.log_dir, save_stats=False,
        match_name=a.match_name, seed=a.seed, silence_errors=True,
        scenario=a.scenario)
    agents = [(name, False) for name in a.agents]
    t0 = time()
    world = BombeRLeWorld(args, agents)
    names = [ag.name for ag in world.agents]
    per_round = []
    for r in range(a.n_rounds):
        world.new_round()
        while world.running:
            world.do_step()
        scores = {ag.name: int(ag.score) for ag in world.agents}
        per_round.append(scores)
        if (r + 1) % 25 == 0 or (r + 1) == a.n_rounds:
            print('[tourney] round %d/%d elapsed=%.0fs' % (
                r + 1, a.n_rounds, time() - t0), flush=True)
    lifetime = {ag.name: dict(ag.lifetime_statistics) for ag in world.agents}
    world.end()

    rng = np.random.default_rng(a.seed + 977)
    summary = {}
    win_matrix = {n: [] for n in names}
    rank_matrix = {n: [] for n in names}
    for rd in per_round:
        vals = [rd[n] for n in names]
        top = max(vals)
        winners = [n for n in names if rd[n] == top]
        for n in names:
            win_matrix[n].append(1.0 / len(winners) if n in winners else 0.0)
        order = sorted(range(len(names)), key=lambda i: -vals[i])
        # average rank over ties
        ranks = {}
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[names[order[k]]] = avg
            i = j + 1
        for n in names:
            rank_matrix[n].append(ranks[n])
    for n in names:
        sc = [rd[n] for rd in per_round]
        lt = lifetime.get(n, {})
        summary[n] = {
            'score_total': int(sum(sc)),
            'score_mean': float(np.mean(sc)),
            'score_ci95': _boot_ci(sc, rng=rng),
            'round_wins': float(np.sum(win_matrix[n])),
            'win_rate': float(np.mean(win_matrix[n])),
            'win_rate_ci95': _boot_ci(win_matrix[n], rng=rng),
            'mean_rank': float(np.mean(rank_matrix[n])),
            'kills_total': int(lt.get('kills', 0) or 0),
            'suicides_total': int(lt.get('suicides', 0) or 0),
            'coins_total': int(lt.get('coins', 0) or 0),
            'bombs_total': int(lt.get('bombs', 0) or 0),
            'crates_total': int(lt.get('crates', 0) or 0),
        }

    out = {
        'agents': names, 'n_rounds': a.n_rounds, 'seed': a.seed,
        'scenario': a.scenario, 'runtime_s': round(time() - t0, 2),
        'per_round': per_round, 'summary': summary,
    }
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)

    print('\n%-24s %8s %8s %8s %8s %8s' % (
        'agent', 'mean', 'winrate', 'rank', 'kills/rd', 'coins/rd'))
    print('-' * 68)
    for n in names:
        s = summary[n]
        print('%-24s %8.3f %8.3f %8.2f %8.3f %8.3f' % (
            n, s['score_mean'], s['win_rate'], s['mean_rank'],
            s['kills_total'] / a.n_rounds, s['coins_total'] / a.n_rounds))
    if a.out:
        print('wrote', a.out)


if __name__ == '__main__':
    main()
