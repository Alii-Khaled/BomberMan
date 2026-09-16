#!/usr/bin/env python3
"""Tally the E123 solo screens vs the ctk3 solo control leg."""
import json
import sys

import numpy as np

ctl_path = 'results/tourney_ctk3solo_s0.json'
try:
    c = np.array([r.get('Harvy', r.get('arbiter_ng')) for r in
                  json.load(open(ctl_path))['per_round']], float)
    print('control (ctk3, solo 40 s0): %.3f  tail<=2 %d'
          % (c.mean(), int((c <= 2).sum())))
except Exception as ex:
    print('control missing:', ex)

for tag in sys.argv[1:]:
    try:
        a = np.array([r.get('Harvy', r.get('arbiter_ng')) for r in json.load(
            open('results/tourney_s23%s_solo_s0.json' % tag)
        )['per_round']], float)
        print('%-5s %.3f  delta %+.3f  tail<=2 %d'
              % (tag, a.mean(), a.mean() - c.mean(), int((a <= 2).sum())))
    except Exception as ex:
        print(tag, 'missing:', ex)
