#!/usr/bin/env python3
"""E125 A0: reproduce the round-18 solo approach-oscillation freeze and
dump the top bomb plans around the flip ticks.

Drives BombeRLeWorld solo (arbiter_ng, classic, seed 0) with the gap
recorder on, then analyzes the recorded jsonl: for every round scoring
<= 2, prints the action histogram, the longest two-tile alternation
window, and a per-tick plan dump (committed bomb target ba / first bf /
scores) over that window. Zero agent-code behavior change.

Usage: python3 scripts/diag_e125_repro.py [--rounds 20] [--seed 0]
"""
import argparse
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('ARBITER_GAP_DIAG',
                      os.path.join(REPO, 'results', 'e125_repro'))
os.environ.setdefault('ARBITER_DEVICE', 'cpu')
GAP = os.environ['ARBITER_GAP_DIAG'] + '_gap.jsonl'


def load_rounds(path):
    rounds = []
    ticks = None
    meta = None
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('type') == 'round_meta':
                if ticks:
                    rounds.append((meta, ticks))
                meta, ticks = rec, []
            elif rec.get('type') == 'tick':
                if ticks is None:
                    ticks = []
                ticks.append(rec)
    if ticks:
        rounds.append((meta, ticks))
    return rounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if os.path.exists(GAP):
        os.remove(GAP)

    import settings as _s
    import logging as _logging
    _s.LOG_AGENT_WRAPPER = max(getattr(_s, 'LOG_AGENT_WRAPPER',
                                      _logging.INFO), _logging.WARNING)
    _s.LOG_AGENT_CODE = max(getattr(_s, 'LOG_AGENT_CODE', _logging.INFO),
                            _logging.WARNING)

    from environment import BombeRLeWorld, WorldArgs
    args = WorldArgs(
        no_gui=True, fps=0, turn_based=False, update_interval=0.1,
        save_replay=False, replay=False, make_video=False,
        continue_without_training=True, log_dir='/tmp/opencode/e125',
        save_stats=False, match_name='e125_repro', seed=a.seed,
        silence_errors=True, scenario='classic')
    os.makedirs('/tmp/opencode/e125', exist_ok=True)
    world = BombeRLeWorld(args, [('arbiter_ng', False)])

    scores = []
    ridx = 0
    world._e125_captured = False
    for r in range(a.rounds):
        world.new_round()
        ridx = r
        while world.running:
            world.do_step()
            # capture the freeze-state board (round 18, mid-oscillation)
            if ridx == 18 and world.step >= 80 \
                    and not world._e125_captured:
                world._e125_captured = True
                import numpy as np
                np.save(os.path.join(REPO, 'results', 'e125_arena.npy'),
                        np.asarray(world.arena.copy()))
                snap = {
                    'coins': [[int(c.x), int(c.y)] for c in world.coins
                              if c.collectable],
                    'pos': [int(world.active_agents[0].x),
                            int(world.active_agents[0].y)],
                    'bombs': [[int(b.x), int(b.y), int(b.timer)]
                              for b in world.bombs],
                    'step': int(world.step),
                }
                with open(os.path.join(REPO, 'results',
                                       'e125_snap.json'), 'w') as f:
                    json.dump(snap, f)
        scores.append(world.agents[0].score)
    world.end()
    print('scores:', scores)
    try:
        import numpy as np
        arena = np.load(os.path.join(REPO, 'results', 'e125_arena.npy'))
        snap = json.load(open(os.path.join(REPO, 'results',
                                           'e125_snap.json')))
        from agent_code.arbiter_ng.search import _bfs_path, score_plan
        from agent_code.arbiter_ng.sim import from_game_state
        for (lbl, pos) in (('snap', tuple(snap['pos'])),
                           ('11,8', (11, 8)), ('11,9', (11, 9))):
            gs = {'round': 1, 'step': snap.get('step', 80), 'field': arena,
                  'self': ('me', 0, True, pos), 'others': [],
                  'bombs': [], 'coins': [tuple(c) for c in snap['coins']],
                  'explosion_map': np.zeros(arena.shape)}
            st0 = from_game_state(gs)
            for tgt in ((15, 8), (12, 7)):
                path = _bfs_path(arena, set(), pos, tgt)
                if path is None:
                    print('plan %s->%s: NO PATH' % (lbl, tgt))
                    continue
                prefix = (path + ['BOMB'])[:12]
                plan = {'first': prefix[0], 'prefix': prefix,
                        'bomb_at': tgt}
                pay, end = score_plan(st0, plan, seed=1000)
                print('plan %s->%s: payoff %.3f alive %s coins_now %d '
                      'path %s'
                      % (lbl, tgt, pay, end['agents'][0]['alive'],
                         sum(1 for c in end['coins'] if c[2]), prefix))
    except Exception as ex:
        print('plan-score check skipped:', ex)

    rounds = load_rounds(GAP)
    for i, (m, ticks) in enumerate(rounds):
        if i >= a.rounds or ticks is None or len(ticks) < 50:
            continue
        sc = scores[i] if i < len(scores) else -1
        if sc > 2:
            continue
        acts = Counter(t['act'] for t in ticks)
        pos = Counter(tuple(t['pos']) for t in ticks)
        bombs = [t['t'] for t in ticks if t['act'] == 'BOMB']
        print('\n== round %d score %d: steps %d' % (i, sc, ticks[-1]['t']))
        print('   acts: %s' % dict(acts))
        print('   top pos: %s' % pos.most_common(4))
        print('   bombs at ticks: %s' % bombs)
        print('   plan dump (every 7th tick over the whole round):')
        for t in ticks[::7]:
            s = t.get('srch') or {}
            bb = s.get('bb')
            bm = s.get('bm')
            me = s.get('me')
            y = s.get('y')
            print('    t%3d pos %-8s act %5s ba %-9s bf %-5s bb %-5s bm %-5s'
                  ' me %-5s y %-4s crates %d'
                  % (t['t'], str(tuple(t['pos'])), t['act'],
                     str(s.get('ba')), str(s.get('bf')),
                     ('%.2f' % bb) if bb is not None else 'None',
                     ('%.2f' % bm) if bm is not None else '-',
                     ('%.2f' % me) if me is not None else '-',
                     ('%.1f' % y) if y is not None else '-',
                     t['crates']))
    print('REPRO DONE')


if __name__ == '__main__':
    main()
