#!/usr/bin/env python3
"""E106 battery tally: pooled multi-battery score/win per model + deltas.

Batteries: g1 (100x2), strong (40x10), umix (100x2),
ucow/ubom/urus/urac (40x2 each). Pooled = all rounds concatenated.
Decision: promote iff candidate pooled beats control on score AND win rate.

Usage: python3 scripts/tally_p0_battery.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ['ctl', 'b1e150', 'ng20']
BATTERIES = ['g1', 'strong', 'umix', 'ucow', 'ubom', 'urus', 'urac']


def load(model, battery):
    """Return list of (seed, rounds dict, first-agent name) per seed json."""
    out = []
    for s in range(10):
        f = os.path.join(REPO, f'results/tourney_{model}_{battery}_s{s}.json')
        if not os.path.isfile(f):
            continue
        d = json.load(open(f))
        me = d['agents'][0]
        out.append((s, d['per_round'], me))
    return out


def pooled(model):
    rows = {}
    for b in BATTERIES:
        legs = load(model, b)
        if not legs:
            rows[b] = None
            continue
        scores, strict_wins, tied_wins, n = [], 0, 0, 0
        for _s, per_round, me in legs:
            for r in per_round:
                n += 1
                scores.append(r[me])
                if r[me] == max(r.values()):
                    strict_wins += 1
                if r[me] >= max(r.values()):
                    tied_wins += 1
        rows[b] = {'n': n, 'score': sum(scores) / n,
                   'win': strict_wins / n, 'win_tied': tied_wins / n}
    allr = [v for v in rows.values() if v]
    tot_n = sum(v['n'] for v in allr)
    pooled_s = sum(v['score'] * v['n'] for v in allr) / tot_n
    pooled_w = sum(v['win'] * v['n'] for v in allr) / tot_n
    return rows, {'n': tot_n, 'score': pooled_s, 'win': pooled_w,
                  'batteries': [b for b in BATTERIES if rows[b]]}


def main():
    res = {}
    for m in MODELS:
        rows, pooled_m = pooled(m)
        res[m] = (rows, pooled_m)
        have = ', '.join(f"{b}:{v['n']}" for b, v in rows.items() if v)
        print(f'{m}: batteries [{have}]')
        if not all(rows[b] for b in BATTERIES):
            missing = [b for b in BATTERIES if not rows[b]]
            print(f'  MISSING legs for {m}: {missing}')
    print()
    hdr = f"{'model':<10}" + ''.join(f'{b:>10}' for b in BATTERIES) \
        + f"{'POOLED':>10}{'win':>8}"
    print(hdr)
    for m in MODELS:
        rows, pooled_m = res[m]
        line = f"{m:<10}"
        for b in BATTERIES:
            v = rows[b]
            line += f"{v['score']:>10.3f}" if v else f"{'--':>10}"
        line += f"{pooled_m['score']:>10.3f}{pooled_m['win']:>8.3f}"
        print(line)
    ctl = res['ctl'][1]
    print('\nvs control (pooled score / pooled win):')
    for m in MODELS[1:]:
        p = res[m][1]
        ds = p['score'] - res['ctl'][1]['score']
        dw = p['win'] - res['ctl'][1]['win']
        verdict = 'PROMOTE' if (ds > 0 and dw >= 0) else 'reject'
        print(f"  {m}: {ds:+.3f} / {dw:+.3f} -> {verdict}")


if __name__ == '__main__':
    main()
