#!/usr/bin/env python3
"""E86/E88 Phase A: death attribution + missed-kill inventory from
ARBITER_DIAG jsonl logs (per-tick snapshots, full rounds).

Inputs (default):  results/diag_e86_{s0,s1,wm}_deaths.jsonl
Outputs (default): results/diag_e86_{deaths.md,attribution.csv,missedkills.csv}

E88 fixes over the first draft (all verified against engine semantics):
  * mask strings are logged in model ACTION_LIST order
    ['UP','RIGHT','DOWN','LEFT','WAIT','BOMB'] (the old decoder used
    UP/DOWN/LEFT/RIGHT/WAIT/BOMB, so corridor cells were mislabeled);
  * blast_set is engine-exact: power 3, beam stops at stone walls (-1)
    and passes through/destroys crates (1) (the old draft used radius 4
    and had wall/crate swapped);
  * attribution keys off the TERMINAL hazard (the ttl<=0 bomb blast or
    live explosion at the last logged tick, checked against the post-
    action destination) instead of the first bomb that ever covered us
    (the old pick matched the engine's killer in 0/23 deaths);
  * the seal window is bounded to enemy bombs planted <=2 ticks after
    the killer's plant tick (as the docstring always claimed).

E90 v2 fixes:
  * the killer bomb's plant tick is the START of the CONTIGUOUS block
    of snapshots containing its coordinate that ends at the terminal
    tick (the first-ever coordinate occurrence can belong to an older
    bomb on a re-bombed tile, inflating the escape count);
  * the engine's KILLED_SELF / GOT_KILLED round event is authoritative
    for the terminal bomb's owner; coordinate history is kept only as a
    diagnostic (`owner_mismatch`), since stale `mine` entries collide
    after a tile is re-bombed.

Taxonomy (priority order):
  own_bomb_chain  killer bomb is ours (KILLED_SELF) and we had >=2
                  masked-safe escapes at the killer bomb's plant tick
  corner_pin      killer bomb is ours with <=1 masked-safe escape
  sim_miss        killer bomb is ours and an enemy bomb planted <=2
                  ticks later sealed the escape corridor
  enemy_trap      enemy bomb killed us with >=1 safe escape at plant
                  (we failed to use it) or a seal after the plant
  enemy_lucky     enemy bomb killed us, no clean classification above
  lingering_*     killed by an already-live explosion at the last tick
  unresolved_*    no terminal hazard found in the logged snapshot

Missed-kill inventory: per tick, enemy inside the blast set of a bomb
(own or pre-existing) whose ttl fuse they cannot escape within their
movement options (exact geometry, no rollout — NOT the E82 mechanism).
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCHES = ['s0', 's1', 'wm']
PREFIX = 'diag_e86'
OUT_PREFIX = 'diag_e86'
LABEL = None
for a in sys.argv[1:]:
    if a.startswith('--prefix='):
        PREFIX = a.split('=', 1)[1]
    elif a.startswith('--out-prefix='):
        OUT_PREFIX = a.split('=', 1)[1]
    elif a.startswith('--batches='):
        BATCHES = [b for b in a.split('=', 1)[1].split(',') if b]
    elif a.startswith('--label='):
        LABEL = a.split('=', 1)[1]

# model.action.ACTION_LIST order — MUST match callbacks._diag_last_mask
MASK_ACTS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACT_DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0),
             'RIGHT': (1, 0), 'WAIT': (0, 0), 'BOMB': (0, 0)}


def load(path):
    rounds, cur = [], None
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('type') == 'round_meta':
                cur = {'meta': d, 'ticks': []}
                rounds.append(cur)
            elif d.get('type') == 'tick' and cur is not None:
                cur['ticks'].append(d)
    return rounds


def blast_set(field, bx, by, radius=3):
    """Engine-exact blast of a bomb at (bx,by): power 3, stops at stone
    walls (-1), passes through crates (1) and destroys them."""
    cells = [(bx, by)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                break
            if field[x, y] == -1:  # stone wall stops the beam
                break
            cells.append((x, y))
    return set(cells)


def mask_moves(mask):
    """Decode valid&safe mask strings -> safe non-BOMB actions."""
    try:
        valid, safe = mask[0], mask[1]
    except Exception:
        return []
    return [a for i, a in enumerate(MASK_ACTS)
            if i < len(valid) and i < len(safe)
            and valid[i] == '1' and safe[i] == '1' and a != 'BOMB']


def mask_safe_action(t):
    try:
        i = MASK_ACTS.index(t.get('act', 'WAIT'))
        return t['safe'][1][i] == '1'
    except Exception:
        return False


def _fields(ticks):
    return {t['t']: np.asarray(t['field'], dtype=int)
            for t in ticks if 'field' in t}


def _field_at(ticks, fields, i):
    for j in range(i, -1, -1):
        if ticks[j]['t'] in fields:
            return fields[ticks[j]['t']]
    return fields[min(fields)] if fields else None


def _terminal_hazard(ticks, fields):
    """The bomb/explosion responsible for the death at the last tick.

    The last snapshot is taken at the decision immediately before the
    engine's update_bombs/evaluate step: ttl<=0 bombs detonate then and
    `expl` cells are already lethal. The victim ends the step at its
    post-action destination (or stays if the move was invalid).
    """
    last = ticks[-1]
    pos0 = tuple(int(v) for v in last['pos'])
    d = ACT_DELTA.get(last.get('act', 'WAIT'), (0, 0))
    pos1 = (pos0[0] + d[0], pos0[1] + d[1])
    f = _field_at(ticks, fields, len(ticks) - 1)
    if f is None:
        return None
    mine = set(tuple(int(v) for v in m) for m in last.get('mine', []))
    for b in last.get('bombs', []):
        bx, by, ttl = int(b[0]), int(b[1]), int(b[2])
        if ttl > 0:
            continue
        bl = blast_set(f, bx, by)
        if pos0 in bl or pos1 in bl:
            return {'kind': 'bomb', 'bomb': (bx, by),
                    'ours': (bx, by) in mine, 't': last['t'],
                    'pos0': pos0, 'pos1': pos1}
    expl = set(tuple(int(v) for v in e) for e in last.get('expl', []))
    if pos0 in expl or pos1 in expl:
        return {'kind': 'explosion', 'bomb': None, 'ours': None,
                't': last['t'], 'pos0': pos0, 'pos1': pos1}
    return None


def _sealed_after(ticks, fields, plant_i, mine_set, corridor, killer):
    """Enemy bomb planted within 2 ticks of the killer's plant tick whose
    blast covers the escape corridor -> (bomb, tick) or None."""
    for j in range(plant_i + 1, min(plant_i + 3, len(ticks))):
        tt = ticks[j]
        f2 = _field_at(ticks, fields, j)
        if f2 is None:
            continue
        for b in tt.get('bombs', []):
            bx2, by2, ttl2 = int(b[0]), int(b[1]), int(b[2])
            if (bx2, by2) == killer or (bx2, by2) in mine_set:
                continue
            if ttl2 >= 3 and any((ex, ey) in blast_set(f2, bx2, by2)
                                 for (ex, ey) in corridor):
                return (bx2, by2), tt['t']
    return None


def attribute(rnd):
    """Return (cause, detail) for a death round."""
    ticks = rnd['ticks']
    if not ticks:
        return 'no_data', ''
    killer = rnd['meta'].get('killer', 'UNKNOWN')
    fields = _fields(ticks)
    hazard = _terminal_hazard(ticks, fields)
    if hazard is None:
        return ('unresolved_' + killer.lower(),
                'no ttl<=0 blast / live explosion at final tick')
    if hazard['kind'] == 'explosion':
        return ('lingering_' + ('self' if killer == 'KILLED_SELF'
                                else 'enemy'),
                'live explosion t=%d pos=%s->%s' % (
                    hazard['t'], hazard['pos0'], hazard['pos1']))
    bx, by = hazard['bomb']
    coord_ours = bool(hazard['ours'])
    # E90: engine event is authoritative for the terminal bomb's owner;
    # coordinate history collides when a tile is re-bombed after an
    # earlier own bomb (stale `mine` entries).
    ours = (killer == 'KILLED_SELF') if killer in ('KILLED_SELF',
                                                   'GOT_KILLED') \
        else coord_ours
    # Plant tick = start of the CONTIGUOUS block of snapshots containing
    # this bomb coordinate that ends at the terminal tick. The first-ever
    # occurrence can belong to an older bomb on the same tile.
    plant_i = len(ticks) - 1
    for i in range(len(ticks) - 1, -1, -1):
        if any((int(b[0]), int(b[1])) == (bx, by)
               for b in ticks[i].get('bombs', [])):
            plant_i = i
        else:
            break
    pt = ticks[plant_i]
    esc = mask_moves(pt.get('safe', ['111111', '111111']))
    mine_set = set(tuple(int(v) for v in m) for m in pt.get('mine', []))
    corridor = [(pt['pos'][0] + ACT_DELTA[a][0],
                 pt['pos'][1] + ACT_DELTA[a][1])
                for a in esc if a in ACT_DELTA]
    sealed = _sealed_after(ticks, fields, plant_i, mine_set, corridor,
                           (bx, by))
    mismatch = ''
    if coord_ours != ours:
        mismatch = ' owner_mismatch(coord=%s,engine=%s)' % (
            'own' if coord_ours else 'enemy', killer)
    if ours:
        if sealed:
            return 'sim_miss', ('plant t=%d own(%d,%d) esc=%d sealed '
                                'enemy(%d,%d)@%d%s' % (
                                    pt['t'], bx, by, len(esc), sealed[0][0],
                                    sealed[0][1], sealed[1], mismatch))
        if len(esc) <= 1:
            return 'corner_pin', 'plant t=%d own(%d,%d) esc=%d%s' % (
                pt['t'], bx, by, len(esc), mismatch)
        return 'own_bomb_chain', 'plant t=%d own(%d,%d) esc=%d%s' % (
            pt['t'], bx, by, len(esc), mismatch)
    if len(esc) <= 1:
        return 'corner_pin', 'plant t=%d enemy(%d,%d) esc=%d%s' % (
            pt['t'], bx, by, len(esc), mismatch)
    if sealed:
        return 'enemy_trap', 'plant t=%d enemy(%d,%d) esc=%d sealed@%d%s' % (
            pt['t'], bx, by, len(esc), sealed[1], mismatch)
    return 'enemy_lucky', 'plant t=%d enemy(%d,%d) esc=%d%s' % (
        pt['t'], bx, by, len(esc), mismatch)


def missed_kills(rounds):
    """Per-tick exact certificates attributable to US: enemy stands in
    the blast of a bomb with ttl<=1 whose position is in our planted
    history, AND every escape option (stay + 4 free neighbours) is
    covered by the union of detonating blasts. Deduped across
    consecutive ticks; cross-checked against KILLED_OPPONENT events."""
    hits = []
    for rnd in rounds:
        for idx, t in enumerate(rnd['ticks']):
            if 'field' not in t or not t.get('bombs'):
                continue
            field = np.asarray(t['field'], dtype=int)
            detonating = [b for b in t['bombs'] if int(b[2]) <= 1]
            if not detonating:
                continue
            union = set()
            for b in detonating:
                union |= blast_set(field, int(b[0]), int(b[1]))
            ours = any((int(b[0]), int(b[1])) in
                       [tuple(int(v) for v in m) for m in t.get('mine', [])]
                       for b in detonating)
            for o in t.get('others', []):
                epos = (o[1], o[2])
                if epos not in union:
                    continue
                sealed = True
                for nx, ny in ((epos[0], epos[1] - 1), (epos[0], epos[1] + 1),
                               (epos[0] - 1, epos[1]), (epos[0] + 1, epos[1]),
                               epos):
                    if field[nx, ny] != 0:
                        continue
                    if (nx, ny) not in union:
                        sealed = False
                        break
                if sealed:
                    kill_ev = any(
                        'KILLED_OPPONENT' in ev
                        for tt in rnd['ticks'][idx:idx + 2]
                        for ev in tt.get('ev', []))
                    hits.append({'round': rnd['meta'].get('round', '?'),
                                 't': t['t'], 'enemy': o[0],
                                 'epos': list(epos),
                                 'bomb': [int(detonating[0][0]),
                                          int(detonating[0][1])],
                                 'ttl': int(detonating[0][2]),
                                 'ours': ours,
                                 'killed_opp_event': kill_ev})
    dedup, seen = [], set()
    for h in hits:
        key = (h['round'], h['enemy'], tuple(h['epos']))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)
    return dedup


def main():
    rows, all_rounds = [], []
    for b in BATCHES:
        p = os.path.join(REPO, 'results', '%s_%s_deaths.jsonl' % (PREFIX, b))
        if not os.path.exists(p):
            continue
        rounds = load(p)
        all_rounds += [(b, r) for r in rounds]
    deaths = [(b, r) for b, r in all_rounds
              if r['meta']['killer'] != 'SURVIVED']
    for b, r in deaths:
        cause, detail = attribute(r)
        rows.append({'batch': b, 'round': r['meta'].get('round', '?'),
                     'killer': r['meta']['killer'], 'cause': cause,
                     'detail': detail})
    out_csv = os.path.join(REPO, 'results',
                           '%s_attribution.csv' % OUT_PREFIX)
    if rows:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['batch', 'round', 'killer',
                                              'cause', 'detail'])
            w.writeheader()
            w.writerows(rows)

    mk = missed_kills([r for _, r in all_rounds])
    with open(os.path.join(REPO, 'results',
                           '%s_missedkills.csv' % OUT_PREFIX),
              'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['round', 't', 'enemy', 'epos',
                                          'bomb', 'ttl', 'ours',
                                          'killed_opp_event'])
        w.writeheader()
        w.writerows(mk)

    from collections import Counter
    causes = Counter(r['cause'] for r in rows)
    killers = Counter(r['killer'] for r in rows)
    n_rounds = len(all_rounds)
    title = LABEL or ('%s death attribution' % PREFIX)
    lines = [
        '# %s (%d rounds)' % (title, n_rounds),
        '',
        'Deaths: **%d** — %s' % (len(rows), dict(killers)),
        'Causes: %s' % dict(causes),
        '',
        '## Kill certificates (enemy sealed by a detonating blast, deduped)',
        '%d trap events; of those, %d from OUR bombs, %d realized via '
        'KILLED_OPPONENT event' % (
            len(mk),
            sum(1 for h in mk if h.get('ours')),
            sum(1 for h in mk if h.get('killed_opp_event'))),
        '',
        '| cause | n | share |',
        '|---|---|---|',
    ]
    for c, n in causes.most_common():
        if rows:
            lines.append('| %s | %d | %.0f%% |' % (c, n, 100 * n / len(rows)))
    own_pin = sum(1 for r in rows if r['cause'] == 'corner_pin'
                  and 'own(' in r['detail'])
    lines += ['', 'corner_pin split: own-bomb %d / enemy-bomb %d' % (
        own_pin, causes.get('corner_pin', 0) - own_pin)]
    mism = sum(1 for r in rows if 'owner_mismatch' in r['detail'])
    lines += ['', 'coordinate-owner vs engine-event mismatches: %d' % mism]
    lines += ['', '## Per batch', '',
              '| batch | rounds | deaths | KILLED_SELF | GOT_KILLED |',
              '|---|---|---|---|---|']
    per = {}
    for b, r in all_rounds:
        d = per.setdefault(b, [0, 0, 0, 0])
        d[0] += 1
        if r['meta']['killer'] != 'SURVIVED':
            d[1] += 1
            d[2 if r['meta']['killer'] == 'KILLED_SELF' else 3] += 1
    for b, d in per.items():
        lines.append('| %s | %d | %d | %d | %d |' % tuple([b] + d))
    out_md = os.path.join(REPO, 'results', '%s_deaths.md' % OUT_PREFIX)
    with open(out_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
