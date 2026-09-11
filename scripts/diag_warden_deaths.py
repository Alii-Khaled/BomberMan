#!/usr/bin/env python3
"""Warden death attribution from WARDEN_DIAG_DIR jsonl + game.log.

The diag hook (WARDEN_DIAG_DIR) records one decision line per act() call;
game.log records the engine's terminal death lines (own bomb vs agent X's
bomb) with the enclosing round/step. Joining them attributes every death
to the decision that led into it.

Taxonomy (own-bomb deaths):
  own_pinned        no certified safe first move at the fatal decision
  own_left_safety   a safe move existed but another action was chosen
  own_solver_miss   chosen action was certified safe yet the engine killed
  enemy             opponent's bomb (with or without a safe move)
  unresolved        no diag line found for the fatal round/step

Usage:
  WARDEN_SEED=123 WARDEN_DIAG_DIR=$PWD/logs/diag_warden_e95 \
    python3 scripts/tournament_eval.py --agents warden_v2 arbiter overlord \
    sentinel --n-rounds 20 --seed 0 --log-dir logs/diag_warden_e95 \
    --match-name diag_warden_e95_strong_s0 \
    --out results/diag_warden_e95_strong_s0.json
  python3 scripts/diag_warden_deaths.py
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RE_ROUND = re.compile(r'STARTING ROUND #(\d+)')
RE_STEP = re.compile(r'STARTING STEP (\d+)')
RE_OWN = re.compile(r'Agent <([^>]+)> blown up by own bomb')
RE_ENEMY = re.compile(r'Agent <([^>]+)> blown up by agent <([^>]+)>\'s bomb')


def parse_deaths(path, agent):
    deaths = []
    round_id, step = 0, 0
    with open(path, errors='replace') as fh:
        for line in fh:
            m = RE_ROUND.search(line)
            if m:
                round_id = int(m.group(1))
                continue
            m = RE_STEP.search(line)
            if m:
                step = int(m.group(1))
                continue
            m = RE_OWN.search(line)
            if m:
                if m.group(1) == agent:
                    deaths.append({'round': round_id, 'step': step,
                                   'kind': 'own', 'killer': agent})
                continue
            m = RE_ENEMY.search(line)
            if m:
                if m.group(1) == agent:
                    deaths.append({'round': round_id, 'step': step,
                                   'kind': 'enemy', 'killer': m.group(2)})
    return deaths


def load_diag(path):
    by_round = {}
    if not os.path.exists(path):
        return by_round
    with open(path, errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            by_round.setdefault(int(rec['round']), []).append(rec)
    for recs in by_round.values():
        recs.sort(key=lambda r: int(r['step']))
    return by_round


def fatal_record(by_round, round_id, step):
    recs = by_round.get(round_id) or []
    best = None
    for r in recs:
        if int(r['step']) <= step:
            best = r
        else:
            break
    return best


def classify(death, rec):
    if rec is None:
        return 'unresolved'
    if death['kind'] == 'enemy':
        return 'enemy'
    if not rec['safe_moves']:
        return 'own_pinned'
    if rec['action'] not in rec['safe_moves']:
        return 'own_left_safety'
    return 'own_solver_miss'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game-log', default=os.path.join(
        REPO, 'logs', 'diag_warden_e95', 'game.log'))
    p.add_argument('--diag', default=os.path.join(
        REPO, 'logs', 'diag_warden_e95', 'warden_diag.jsonl'))
    p.add_argument('--agent', default='warden_v2')
    p.add_argument('--tourney', default=os.path.join(
        REPO, 'results', 'diag_warden_e95_strong_s0.json'))
    p.add_argument('--out-prefix', default=os.path.join(
        REPO, 'results', 'diag_warden_e95'))
    a = p.parse_args()

    deaths = parse_deaths(a.game_log, a.agent)
    by_round = load_diag(a.diag)

    rows = []
    counts = {}
    for d in deaths:
        rec = fatal_record(by_round, d['round'], d['step'])
        cls = classify(d, rec)
        counts[cls] = counts.get(cls, 0) + 1
        rows.append({'round': d['round'], 'step': d['step'], 'class': cls,
                     'killer': d['killer'], 'rec': rec})

    own = sum(1 for r in rows if r['class'].startswith('own'))
    enemy = counts.get('enemy', 0)
    print('deaths=%d own=%d enemy=%d unresolved=%d'
          % (len(rows), own, enemy, counts.get('unresolved', 0)))
    for cls in ('own_pinned', 'own_left_safety', 'own_solver_miss',
                'enemy', 'unresolved'):
        if cls in counts:
            print('  %-16s %d' % (cls, counts[cls]))

    if a.tourney and os.path.exists(a.tourney):
        t = json.load(open(a.tourney))
        v = t['summary'].get(a.agent, {})
        n = t.get('n_rounds') or 1
        print('tourney: score %.2f kills %.2f/rd sui %.2f/rd bombs %.1f/rd '
              'coins %.2f/rd'
              % (v.get('score_mean', 0), v.get('kills_total', 0) / n,
                 v.get('suicides_total', 0) / n,
                 v.get('bombs_total', 0) / n, v.get('coins_total', 0) / n))

    detail = []
    for r in rows:
        rec = r['rec'] or {}
        detail.append({
            'round': r['round'], 'step': r['step'], 'class': r['class'],
            'killer': r['killer'],
            'pos': [rec.get('x'), rec.get('y')],
            'action': rec.get('action'),
            'safe_moves': rec.get('safe_moves'),
            'esc_dist': rec.get('esc_dist'),
            'must_flee': rec.get('must_flee'),
            'danger_own_t0': rec.get('danger_own_t0'),
            'danger_own_t1': rec.get('danger_own_t1'),
            'plants': rec.get('plants'),
        })
    out = {'agent': a.agent, 'deaths': len(rows), 'own': own,
           'enemy': enemy, 'counts': counts, 'detail': detail}
    with open(a.out_prefix + '_attribution.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    with open(a.out_prefix + '_deaths.md', 'w') as fh:
        fh.write('# Warden death attribution (%s)\n\n' % a.agent)
        fh.write('deaths=%d own=%d enemy=%d unresolved=%d\n\n'
                 % (len(rows), own, enemy, counts.get('unresolved', 0)))
        for cls in ('own_pinned', 'own_left_safety', 'own_solver_miss',
                    'enemy', 'unresolved'):
            if cls in counts:
                fh.write('- %s: %d\n' % (cls, counts[cls]))
        fh.write('\n| r.step | class | pos | action | safe | esc | '
                 'dng0/1 | plants |\n|---|---|---|---|---|---|---|---|\n')
        for d in detail:
            fh.write('| %d.%d | %s | %s | %s | %s | %s | %s | %s |\n'
                     % (d['round'], d['step'], d['class'], d['pos'],
                        d['action'], ','.join(d['safe_moves'] or []),
                        d['esc_dist'], '%s/%s' % (d['danger_own_t0'],
                                                  d['danger_own_t1']),
                        d['plants']))
    print('wrote %s_attribution.json + %s_deaths.md' % (a.out_prefix,
                                                        a.out_prefix))
    return 0


if __name__ == '__main__':
    sys.exit(main())
