#!/usr/bin/env python3
"""Canonical battery/triage runner + tally (E109+).

Replaces the ad-hoc bash drivers (three field-splitting bugs taught us
better). Runs games nice'd on CPU with per-game /tmp log dirs, skips
existing outputs, wave-limited concurrency, then prints a pooled tally.

Usage:
  python3 scripts/battery.py --tag TAG [--model PATH] [--control CTL]
      [--legs "g1:2:100,strong:10:40,umix:2:100,ucow:2:40,ubom:2:40,
               urus:2:40,urac:2:40"] [--jobs 8] [--tally-only]
  legspec: name:seeds:rounds[:opp1+opp2+opp3]
  Defaults per battery name:
    g1     100rds  rule_based_agent x3
    strong 40rds   warden_v2 overlord sentinel
    umix   100rds  unseen_coward unseen_bomber unseen_rusher
    ucow/ubom/urus/urac 40rds 3x archetype
    ctl40  40rds   rule_based_agent x3   (triage control)
  Outputs: results/tourney_<tag>_<battery>_s<seed>.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = '/tmp/opencode/battery'

DEFAULT_LEGS = [('g1', 2, 100, 'rule_based_agent rule_based_agent rule_based_agent'),
                ('strong', 10, 40, 'warden_v2 overlord sentinel'),
                ('umix', 2, 100, 'unseen_coward unseen_bomber unseen_rusher'),
                ('ucow', 2, 40, 'unseen_coward unseen_coward unseen_coward'),
                ('ubom', 2, 40, 'unseen_bomber unseen_bomber unseen_bomber'),
                ('urus', 2, 40, 'unseen_rusher unseen_rusher unseen_rusher'),
                ('urac', 2, 40, 'unseen_racer unseen_racer unseen_racer')]
TRIAGE_LEGS = [('g1', 1, 40, 'rule_based_agent rule_based_agent rule_based_agent')]


def parse_legs(spec):
    if not spec:
        return DEFAULT_LEGS
    legs = []
    for part in spec.split(','):
        f = part.split(':')
        name, seeds, rounds = f[0], int(f[1]), int(f[2])
        opps = f[3].replace('+', ' ') if len(f) > 3 else dict(
            (l[0], l[3]) for l in DEFAULT_LEGS)[name]
        legs.append((name, seeds, rounds, opps))
    return legs


def out_path(tag, name, seed):
    return os.path.join(REPO, 'results', f'tourney_{tag}_{name}_s{seed}.json')


def run_leg(tag, name, seed, rounds, opps, model, jobs_left):
    out = out_path(tag, name, seed)
    if os.path.isfile(out):
        return
    ldir = f'/tmp/opencode/battery/{tag}_{name}_s{seed}'
    os.makedirs(ldir, exist_ok=True)
    env = dict(os.environ)
    env.update({'ARBITER_DEVICE': 'cpu'})
    if model:
        env['ARBITER_NG_MODEL'] = model
    cmd = (['nice', '-n', '10', 'python3', 'scripts/tournament_eval.py',
            '--agents', 'arbiter_ng'] + opps.split() +
           ['--n-rounds', str(rounds), '--seed', str(seed),
            '--scenario', 'classic', '--log-dir', ldir,
            '--match-name', f'{tag}_{name}_s{seed}', '--out', out])
    log = open(os.path.join(REPO, 'logs',
                            f'tourney_{tag}_{name}_s{seed}.log'), 'w')
    try:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                       stderr=subprocess.STDOUT, timeout=7200)
    except Exception as ex:
        print(f'FAIL {tag} {name} s{seed}: {ex}', flush=True)
    finally:
        log.close()


def run_all(tag, legs, model, jobs):
    procs = []
    for (name, seeds, rounds, opps) in legs:
        for seed in range(seeds):
            if os.path.isfile(out_path(tag, name, seed)):
                print(f'SKIP {tag} {name} s{seed}', flush=True)
                continue
            while len(procs) >= jobs:
                time.sleep(2)
                procs = [p for p in procs if p.poll() is None]
            p = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__),
                 '--tag', tag,
                 '--leg-worker', tag, name, str(seed), str(rounds),
                 '--opps', opps, '--model', model or ''],
                cwd=REPO)
            procs.append(p)
    for p in procs:
        p.wait()
    print('BATTERY_RUN_DONE', flush=True)


def tally(tag, legs):
    rows = {}
    for (name, seeds, _r, _o) in legs:
        sc, wins, n = [], 0, 0
        for seed in range(seeds):
            f = out_path(tag, name, seed)
            if not os.path.isfile(f):
                continue
            d = json.load(open(f))
            me = d['agents'][0]
            for r in d['per_round']:
                sc.append(r[me])
                n += 1
                if r[me] >= max(r.values()):
                    wins += 1
        rows[name] = (sum(sc) / n, wins / n, n) if n else None
    have = {k: v for k, v in rows.items() if v}
    if not have:
        return None
    n_tot = sum(v[2] for v in have.values())
    return {'rows': have,
            'pooled': sum(v[0] * v[2] for v in have.values()) / n_tot,
            'win': sum(v[1] * v[2] for v in have.values()) / n_tot,
            'n': n_tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--model', default='')
    ap.add_argument('--control', default=None)
    ap.add_argument('--legs', default=None)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--tally-only', action='store_true')
    ap.add_argument('--leg-worker', nargs=4, default=None)
    ap.add_argument('--opps', default='')
    a = ap.parse_args()

    if a.leg_worker:
        tag, name, seed, rounds = a.leg_worker
        run_leg(tag, name, int(seed), int(rounds), a.opps, a.model, 1)
        return

    legs = parse_legs(a.legs)
    if not a.tally_only:
        run_all(a.tag, legs, a.model, a.jobs)

    res = {a.tag: tally(a.tag, legs)}
    if a.control:
        res[a.control] = tally(a.control, legs)
    print(f"\n{'tag':<14}{'pooled':>8}{'win':>7}{'n':>6}  per-battery (score/win)")
    for tag, t in res.items():
        if not t:
            print(f'{tag:<14}  MISSING')
            continue
        print(f"{tag:<14}{t['pooled']:>8.3f}{t['win']:>7.3f}{t['n']:>6}")
        for (name, _s, _r, _o) in legs:
            v = t['rows'].get(name)
            if v:
                print(f"   {name:<7} {v[0]:.3f}/{v[1]:.3f} (n={v[2]})")
    if a.control and res.get(a.control):
        c, t = res[a.control], res[a.tag]
        if t:
            ds = t['pooled'] - c['pooled']
            dw = t['win'] - c['win']
            verdict = 'PROMOTE' if (ds > 0 and dw >= 0) else 'reject'
            print(f"\nvs control: {ds:+.3f} score, {dw:+.3f} win -> {verdict}")


if __name__ == '__main__':
    main()
