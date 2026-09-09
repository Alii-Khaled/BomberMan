#!/usr/bin/env python3
"""Aggregate ARBITER frozen-gate JSONs into report tables (§6).

Reads results/gate_arbiter_*.json (main.py --save-stats format).
Writes results/arbiter_summary.csv — one row per file x agent with
per-round rates + pooled rows per gate family.
Usage: python3 scripts/aggregate_arbiter.py
"""
import csv
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'results', 'arbiter_summary.csv')

KEYS = ['score', 'coins', 'kills', 'suicides', 'crates', 'bombs',
        'moves', 'invalid', 'time', 'rounds', 'steps']


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, 'results',
                                            'gate_arbiter_*.json'))):
        tag = os.path.basename(f)[len('gate_arbiter_'):-len('.json')]
        try:
            d = json.load(open(f))
        except Exception as ex:
            print('skip %s: %s' % (tag, ex))
            continue
        for name, st in d.get('by_agent', {}).items():
            r = st.get('rounds', 0) or 1
            row = {'gate': tag, 'agent': name, 'rounds': st.get('rounds', 0)}
            for k in KEYS:
                v = st.get(k, 0)
                row[k] = round(v / r, 4) if k not in ('rounds',) else v
            # derived: crates/bomb, ms/step
            b = st.get('bombs', 0)
            row['crates_per_bomb'] = round(st.get('crates', 0) / b, 4) if b else 0
            m = st.get('moves', 0)
            row['ms_per_step'] = round(st.get('time', 0) / m * 1000, 2) if m else 0
            rows.append(row)
    # pooled rows per gate x agent (s0+s1, 20+40+100N agnostic)
    fams = {}
    for row in rows:
        fam = re.sub(r'_s\d+$', '', row['gate'])
        fams.setdefault((fam, row['agent']), []).append(row)
    for (fam, agent), rs in sorted(fams.items()):
        if len(rs) < 2:
            continue
        n = sum(r['rounds'] for r in rs)
        prow = {'gate': fam + '_pooled', 'agent': agent, 'rounds': n}
        for k in KEYS:
            if k == 'rounds':
                continue
            prow[k] = round(sum(r[k] * r['rounds'] for r in rs) / n, 4)
        b = sum(r['bombs'] * r['rounds'] for r in rs)
        prow['crates_per_bomb'] = round(
            sum(r['crates'] * r['rounds'] for r in rs) / b, 4) if b else 0
        m = sum(r['moves'] * r['rounds'] for r in rs)
        prow['ms_per_step'] = round(
            sum(r['time'] * r['rounds'] for r in rs) / m * 1000, 2) if m else 0
        rows.append(prow)
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['gate', 'agent', 'rounds',
                                           'score', 'coins', 'kills',
                                           'suicides', 'crates', 'bombs',
                                           'moves', 'invalid', 'time',
                                           'steps', 'crates_per_bomb',
                                           'ms_per_step'])
        w.writeheader()
        w.writerows(rows)
    print('wrote %s (%d rows)' % (OUT, len(rows)))


if __name__ == '__main__':
    main()
