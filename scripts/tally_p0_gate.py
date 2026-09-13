#!/usr/bin/env python3
"""Phase-0 gate tally: pooled G1 100x2 means + win rates, candidates vs control.

Usage: python3 scripts/tally_p0_gate.py
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TAGS = ['p0gate_ctl', 'p0gate_b1e050', 'p0gate_b1e150', 'p0gate_b2e100',
        'p0gate_b2e200', 'p0gate_ng13', 'p0gate_ng20']


def load(tag):
    rounds, wins, strict_wins = [], 0, 0
    for s in (0, 1):
        f = os.path.join(REPO, f'results/tourney_{tag}_g1_s{s}.json')
        if not os.path.isfile(f):
            return None
        d = json.load(open(f))
        me = d['agents'][0]
        for r in d['per_round']:
            rounds.append(r[me])
            if r[me] == max(r.values()):
                strict_wins += 1
            if r[me] >= max(r.values()):
                wins += 1
    n = len(rounds)
    return {'tag': tag, 'n': n, 'score': sum(rounds) / n,
            'win': strict_wins / n, 'win_tied': wins / n}


def main():
    rows = [load(t) for t in TAGS]
    rows = [r for r in rows if r]
    ctl = next((r for r in rows if r['tag'] == 'p0gate_ctl'), None)
    print(f"{'tag':<18}{'n':>5}{'score':>8}{'win':>7}{'win_tied':>9}"
          f"{'d_ctl':>8}{'d_wt':>7}")
    for r in rows:
        ds = (r['score'] - ctl['score']) if ctl else float('nan')
        dw = (r['win_tied'] - ctl['win_tied']) if ctl else float('nan')
        print(f"{r['tag']:<18}{r['n']:>5}{r['score']:>8.3f}{r['win']:>7.3f}"
              f"{r['win_tied']:>9.3f}{ds:>+8.3f}{dw:>+9.3f}")
    if ctl:
        print('\ncontrol = ship (arbiter, E88 defaults), fresh same-session.')
    return rows


if __name__ == '__main__':
    main()
