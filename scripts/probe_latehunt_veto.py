"""E39 probe: late-hunt veto (OVERLORD_G_LATEHUNT_VETO).

Run twice: VETO=0 (expect BOMB chosen in a late opp-only blast setup) and
VETO=1 (expect BOMB refused). Run from repo root, one env per process
(module constants resolve at import).
"""
import os
import sys
from collections import deque
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent_code.overlord import callbacks as C
from agent_code.overlord.model import build_model

VETO = os.environ.get('OVERLORD_G_LATEHUNT_VETO', '0') == '1'

field = np.zeros((17, 17), dtype=int)  # open: escape trivially exists
gs = {'round': 1, 'step': 300, 'field': field, 'bombs': [],
      'explosion_map': np.zeros((17, 17), dtype=int), 'coins': [],
      'self': ('o', 0, True, (8, 8)),
      'others': [('r1', 0, True, (8, 10)), ('r2', 0, True, (10, 8))],
      'user_input': None}

ob = types.SimpleNamespace()
ob.train = False
ob.model = build_model()  # fresh: zero-init heads -> Q~0, heuristic decides
ob.model.eval()
ob.fast = None
ob.bomb_history = deque([], 5)
ob.coord_history = deque([], 24)
ob.current_round = 0
ob.flee_timer = 0
ob.epsilon = 0.0

a = C.act(ob, gs)
print('VETO=%s -> act=%s' % (int(VETO), a))
if not VETO:
    assert a == 'BOMB', 'baseline must plant (opp blast, no crates, step 300)'
    print('PASS: baseline plants')
else:
    assert a != 'BOMB', 'veto must refuse the late opp-only plant'
    print('PASS: veto refuses')
