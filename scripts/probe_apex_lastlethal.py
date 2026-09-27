#!/usr/bin/env python3
"""Probe Apex's split danger windows and final-step arrival check.

A tile lethal at t=0..1 (timer-0 bomb) plus t=3..4 (farther bomb) has
non-contiguous danger. The old first-lethal predicate read it as safe at
ct=2 (earliest 0 < 2); the fixed predicate sees the later window
(latest 4 >= 2). A separate fixture catches a lethal arrival at the
search horizon, which the older BFS accepted before checking the tile.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'agent_code'))

import numpy as np

from apex import safety as apex_safety
from apex import safety_reaper
from apex.safety_reaper import escape_bfs, first_lethal, last_lethal

HORIZON = 8
W = H = 5


def make_danger():
    """Danger lethal on tile (2, 2) at t=0,1 and t=3,4, safe at t=2."""
    danger = np.zeros((HORIZON + 1, W, H), dtype=bool)
    danger[0, 2, 2] = True
    danger[1, 2, 2] = True
    danger[3, 2, 2] = True
    danger[4, 2, 2] = True
    return danger


def main():
    failures = []

    danger = make_danger()
    fl = first_lethal(danger, HORIZON)[2, 2]
    ll = last_lethal(danger, HORIZON)[2, 2]
    print('first_lethal[2,2] = %d (expect 0)' % fl)
    print('last_lethal[2,2]  = %d (expect 4)' % ll)
    if fl != 0:
        failures.append('first_lethal should be 0, got %d' % fl)
    if ll != 4:
        failures.append('last_lethal should be 4, got %d' % ll)

    # Arena: open floor (escape_bfs only needs walkable tiles here).
    arena = np.zeros((W, H), dtype=np.int32)
    # Stand next to the tile; the only question is whether stepping onto
    # (2, 2) at ct=1..2 counts as survivable. It must not: the tile kills
    # at t=3..4 while we would still be on it.
    safe, _ = escape_bfs((2, 1), arena, [], [], danger, HORIZON)
    # (0, 1) is DOWN from (2, 1) to (2, 2) in (dx, dy) deltas.
    down_safe = safe.get((0, 1), None)
    print('step onto split-window tile safe = %s (expect False)' % down_safe)
    if down_safe:
        failures.append('escape_bfs marks the split-window tile safe')

    # Control: a tile lethal only at t=0..1 is safe to enter at ct>=2.
    danger2 = np.zeros((HORIZON + 1, W, H), dtype=bool)
    danger2[0, 2, 2] = True
    danger2[1, 2, 2] = True
    safe2, _ = escape_bfs((2, 1), arena, [], [], danger2, HORIZON)
    # The BFS reaches (2,2) at ct=1 while lethal, so it cannot stop there;
    # but the Window is over by ct=2 and later tiles are safe. Just check
    # the call completes and the truly-safe directions are unaffected.
    print('control call ok, safe keys = %d' % len(safe2))

    # Both Apex solvers used to mark any path reaching the horizon safe
    # before checking whether that final tile was lethal on arrival.
    arrival_arena = np.zeros((W, H), dtype=np.int32)
    arrival_arena[0, :] = arrival_arena[-1, :] = -1
    arrival_arena[:, 0] = arrival_arena[:, -1] = -1
    arrival_danger = np.zeros((HORIZON + 1, W, H), dtype=bool)
    arrival_danger[HORIZON, :, :] = True
    for name, solver in (('apex.safety', apex_safety.escape_bfs),
                         ('apex.safety_reaper', safety_reaper.escape_bfs)):
        unsafe, dist = solver((2, 2), arrival_arena, [], [],
                              arrival_danger, HORIZON)
        print('%s lethal-at-arrival safe = %s (expect none)' % (name, unsafe))
        if any(unsafe.values()) or np.isfinite(dist):
            failures.append('%s accepts a lethal horizon arrival' % name)
        clear, _ = solver((2, 2), arrival_arena, [], [],
                           np.zeros_like(arrival_danger), HORIZON)
        if not clear.get((0, 1), False):
            failures.append('%s rejects a clear control path' % name)

    if failures:
        print('\nFAIL')
        for f in failures:
            print('  -', f)
        return 1
    print('\nPASS: split windows and lethal arrivals are not certified safe.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
