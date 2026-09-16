#!/usr/bin/env bash
# E122/E123/E124 screen runner: one arm config vs the fresh same-code
# control legs (results/gaps_* = ship defaults, recorded this session).
# Usage: bash scripts/run_gap_screens.sh <tag> <env assignments...>
set -u
TAG="$1"; shift
REPO=/home/jovyan/work/BomberMan
cd "$REPO"
for s in 0 1; do
  mkdir -p "/tmp/opencode/gaps/log_${TAG}g1s$s" "/tmp/opencode/gaps/log_${TAG}l5s$s" \
           "/tmp/opencode/gaps/log_${TAG}solo$s"
done
export ENVV

run_g1() { # seed
  env ARBITER_DEVICE=cpu "$ENVV" python3 scripts/tournament_eval.py \
    --agents arbiter_ng rule_based_agent rule_based_agent rule_based_agent \
    --n-rounds 40 --seed "$1" --out "results/tourney_${TAG}g1_s$1.json" \
    --log-dir "/tmp/opencode/gaps/log_${TAG}g1s$1" \
    > "/tmp/opencode/gaps/${TAG}g1_s$1.out" 2>&1 &
}
run_l5() { # seed
  env ARBITER_DEVICE=cpu "$ENVV" python3 scripts/tournament_eval.py \
    --agents arbiter_ng random_agent random_agent random_agent \
    --n-rounds 40 --seed "$1" --out "results/tourney_${TAG}l5_s$1.json" \
    --log-dir "/tmp/opencode/gaps/log_${TAG}l5s$1" \
    > "/tmp/opencode/gaps/${TAG}l5_s$1.out" 2>&1 &
}
run_solo() { # seed
  env ARBITER_DEVICE=cpu "$ENVV" python3 scripts/tournament_eval.py \
    --agents arbiter_ng --n-rounds 40 --seed "$1" \
    --out "results/tourney_${TAG}solo_s$1.json" \
    --log-dir "/tmp/opencode/gaps/log_${TAG}solo$1" \
    > "/tmp/opencode/gaps/${TAG}solo_s$1.out" 2>&1 &
}
export ENVV
for s in 0 1; do run_g1 "$s"; run_l5 "$s"; done
run_solo 0
wait
python3 - <<EOF
import json
import numpy as np

def score_of(path):
    d = json.load(open(path))['per_round']
    return (np.array([r['arbiter_ng'] for r in d], float),
            np.array([max(r.values()) == r['arbiter_ng']
                      for r in d], float))

ctl_g1 = [json.load(open('results/gaps_%s.json' % f))['per_round']
          for f in ('g1_s0', 'g1_s1')]
ctl_l5 = [json.load(open('results/gaps_%s.json' % f))['per_round']
          for f in ('l5_s0', 'l5_s1')]
ctl_solo = json.load(open('results/gaps_solo_s0.json'))['per_round']
tot_a = tot_c = 0
wa = wc = 0.0
for s in (0, 1):
    a = json.load(open('results/tourney_${TAG}g1_s%d.json' % s))['per_round']
    c = ctl_g1[s]
    n = min(len(a), len(c))
    sa = np.array([r['arbiter_ng'] for r in a][:n], float)
    sc = np.array([r['arbiter_ng'] for r in c][:n], float)
    wa += np.array([max(r.values()) == r['arbiter_ng'] for r in a][:n]).sum()
    wc += np.array([max(r.values()) == r['arbiter_ng'] for r in c][:n]).sum()
    print('G1 s%d arm %.3f ctl %.3f' % (s, sa.mean(), sc.mean()))
    tot_a += sa.sum(); tot_c += sc.sum()
for s in (0, 1):
    a = json.load(open('results/tourney_${TAG}l5_s%d.json' % s))['per_round']
    c = ctl_l5[s]
    n = min(len(a), len(c))
    sa = np.array([r['arbiter_ng'] for r in a][:n], float)
    sc = np.array([r['arbiter_ng'] for r in c][:n], float)
    wa += np.array([max(r.values()) == r['arbiter_ng'] for r in a][:n]).sum()
    wc += np.array([max(r.values()) == r['arbiter_ng'] for r in c][:n]).sum()
    print('L5 s%d arm %.3f ctl %.3f' % (s, sa.mean(), sc.mean()))
    tot_a += sa.sum(); tot_c += sc.sum()
a = json.load(open('results/tourney_${TAG}solo_s0.json'))['per_round']
c = ctl_solo
n = min(len(a), len(c))
sa = np.array([r['arbiter_ng'] for r in a][:n], float)
sc = np.array([r['arbiter_ng'] for r in c][:n], float)
tail_a = int((sa <= 2).sum())
tail_c = int((sc <= 2).sum())
wa += np.array([r['arbiter_ng'] == max(r.values()) for r in a][:n]).sum()
wc += np.array([r['arbiter_ng'] == max(r.values()) for r in c][:n]).sum()
print('SOLO s0 arm %.3f ctl %.3f (tail<=2: %d vs %d)'
      % (sa.mean(), sc.mean(), tail_a, tail_c))
tot_a += sa.sum(); tot_c += sc.sum()
print('POOLED arm %.3f ctl %.3f delta %+.3f | winrate arm %.3f ctl %.3f'
      % (tot_a, tot_c, tot_a - tot_c,
         wa / (40 * 2 + 40 * 2 + n), wc / (40 * 2 + 40 * 2 + n)))
EOF
