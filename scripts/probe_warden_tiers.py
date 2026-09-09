"""E47 probe: warden-discipline transplants (PAYOFF_TIER, MUSTFLEE1).

MODE env: off | tier | mustflee | both. Dumps BOMB safe/valid verdicts for
a fixed battery of synthetic + random states to results/probe_tiers_MODE.json.
The runner diffs: off==off (determinism), on-subset-off (never adds
permission), liveness (each flag vetoes >=1 state). Fails loudly.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent_code.overlord.safety import action_safety

MODE = os.environ.get('TIER_PROBE_MODE', 'off')
OUT = 'results/probe_tiers_%s.json' % MODE


def mkstate(arena, pos, bombs=(), others=(), step=10):
    return {'field': arena, 'coins': [], 'step': step,
            'self': ('o', 0, True, tuple(pos)), 'others': list(others),
            'bombs': list(bombs),
            'explosion_map': np.zeros_like(arena)}


states = []
# hand states
a = np.zeros((17, 17), dtype=int)
states.append(('open', mkstate(a, (8, 8))))
states.append(('open_opp', mkstate(a, (8, 8), others=[('r', 0, True, (8, 10))])))
b = np.zeros((17, 17), dtype=int)
b[7, 8] = b[9, 8] = b[8, 7] = 1
states.append(('pocket_3crate', mkstate(b, (8, 8))))
c = np.zeros((17, 17), dtype=int)
states.append(('threat_next', mkstate(c, (8, 8), bombs=[((8, 10), 1)])))
states.append(('threat_now', mkstate(c, (8, 8), bombs=[((8, 9), 0)])))
# random battery (fixed seed -> same states every MODE)
rng = np.random.default_rng(7)
for i in range(200):
    arena = np.zeros((17, 17), dtype=int)
    arena[0, :] = arena[-1, :] = arena[:, 0] = arena[:, -1] = -1
    inner = (rng.random((15, 15)) < 0.35)
    arena[1:-1, 1:-1][inner] = 1
    free = list(zip(*np.where(arena == 0)))
    pos = free[rng.integers(len(free))]
    nb = int(rng.integers(0, 3))
    bombs = []
    for _ in range(nb):
        bp = free[rng.integers(len(free))]
        bombs.append(((int(bp[0]), int(bp[1])), int(rng.integers(0, 5))))
    no = int(rng.integers(0, 3))
    others = [('r%d' % j, 0, True,
               (int(free[rng.integers(len(free))][0]),
                int(free[rng.integers(len(free))][1]))) for j in range(no)]
    states.append(('rand%d' % i, mkstate(arena, pos, bombs, others,
                                         step=int(rng.integers(0, 400)))))

out = {}
for name, gs in states:
    try:
        s = action_safety(gs)
        out[name] = {'valid': bool(s['valid'].get('BOMB', False)),
                     'safe': bool(s['safe'].get('BOMB', False)),
                     'escape': bool(s.get('can_escape_if_bomb', False)),
                     'dist': float(s.get('dist_hyp', 9)),
                     'crates': int(s.get('crates_hit_if_bomb', 0)),
                     'opps': int(s.get('opps_hit_if_bomb', 0))}
    except Exception as ex:
        out[name] = {'error': str(ex)[:80]}
with open(OUT, 'w') as f:
    json.dump(out, f)
print('dumped %d states -> %s' % (len(out), OUT))
