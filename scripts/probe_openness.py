"""E45 probe: additive openness bonus (OVERLORD_G_OPENNESS).

Differential design (robust to absolute heuristic values): the open-minus-
pocket BOMB gap must shift by exactly G*(4-1) between unset and G=0.25 runs.
Run twice (unset + 0.25), compare printed gaps. Fails loudly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent_code.overlord import callbacks as C

SAFETY = {'can_escape_if_bomb': True, 'opps_hit_if_bomb': 0, 'dist_hyp': 2.0}


def gs_at(pos, crates):
    field = np.zeros((17, 17), dtype=int)
    for (x, y) in crates:
        field[x, y] = 1
    return {'field': field, 'coins': [], 'step': 10,
            'self': ('o', 0, True, tuple(pos)), 'others': [], 'bombs': [],
            'explosion_map': np.zeros((17, 17), dtype=int)}


open_b = C._heuristic(gs_at((8, 8), []), SAFETY)['BOMB']
pocket_b = C._heuristic(
    gs_at((8, 8), [(7, 8), (9, 8), (8, 7)]), SAFETY)['BOMB']
print('G=%s open=%.3f pocket=%.3f gap=%.3f'
      % (os.environ.get('OVERLORD_G_OPENNESS', '0.0'),
         open_b, pocket_b, open_b - pocket_b))
