#!/usr/bin/env python3
"""E86 Phase A: death attribution + missed-kill inventory from
ARBITER_DIAG jsonl logs (per-tick snapshots, full rounds).

Inputs:  results/diag_e86_{s0,s1,wm}_deaths.jsonl
Outputs: results/diag_e86_deaths.md + results/diag_e86_attribution.csv

Taxonomy (priority order):
  own_bomb_chain  killer bomb is ours (KILLED_SELF) and we had >=1
                  masked-safe escape at the killer bomb's plant tick
  corner_pin      at the killer bomb's plant tick, <=1 masked-safe
                  escape (boxed by walls/crates or own bomb ring)
  enemy_trap      enemy bomb killed us (GOT_KILLED) with >=1 safe
                  escape at plant tick (we failed to use it)
  sim_miss        our last committed action was marked safe but a NEW
                  enemy bomb (planted <=2 ticks later) covered the
                  committed destination
  enemy_lucky     enemy bomb killed us, no clean classification above
Missed-kill inventory: per tick, enemy inside the blast set of a bomb
(own or pre-existing) whose ttl fuse they cannot escape within their
movement options (exact, no rollout — NOT the E82 mechanism).
"""
import json
import os
import csv
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCHES = ['s0', 's1', 'wm']
FREE = {0}  # arena free-tile value (settings: 0 free, 1 wall, -1 crate)


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


def blast_set(field, bx, by, radius=4):
    """Blast cells of a bomb at (bx,by) given the arena (items.py rules)."""
    cells = [(bx, by)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            x, y = bx + dx * r, by + dy * r
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                break
            if field[x, y] == 1:  # stone wall stops the beam
                break
            cells.append((x, y))
            if field[x, y] == -1:  # crate destroyed, beam stops
                break
    return set(cells)


def escapes(field, pos, bombs_next):
    """Mask-free escape estimate: free neighbors not in any blast next step."""
    x, y = pos
    out = []
    for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            continue
        if field[nx, ny] != 0:
            continue
        if (nx, ny) in bombs_next:
            continue
        out.append((nx, ny))
    return out


def attribute(rnd):
    """Return (cause, detail) for a death round.

    Walks back to the killer bomb's plant tick; escape options are
    read from the agent's OWN masked-safe set at that tick (pre-doom),
    not at the final (already-lethal) tick.
    """
    ticks = rnd['ticks']
    if not ticks:
        return 'no_data', ''
    killer = rnd['meta'].get('killer', 'UNKNOWN')
    import numpy as np
    final_pos = tuple(ticks[-1]['pos'])

    fields = {}
    for t in ticks:
        if 'field' in t:
            fields[t['t']] = np.asarray(t['field'], dtype=int)

    def field_at(i):
        for j in range(i, -1, -1):
            if ticks[j]['t'] in fields:
                return fields[ticks[j]['t']]
        return fields[min(fields)] if fields else None

    # 1) locate killer bomb: first occurrence (plant tick) of a bomb
    #    whose blast covers our position at some tick <= its detonation
    plant_i = None
    killer_bomb = None
    for i, t in enumerate(ticks):
        f = fields.get(t['t'])
        if f is None:
            continue
        for (bx, by, ttl) in t.get('bombs', []):
            bl = blast_set(f, bx, by)
            # does our position at/after this tick fall inside the blast
            # before the bomb detonates?
            horizon = [tuple(tt['pos']) for tt in ticks[i:i + max(1, ttl) + 1]]
            if any(p in bl for p in horizon):
                plant_i = i
                killer_bomb = (bx, by, ttl)
                break
        if killer_bomb:
            break
    if not killer_bomb:
        return 'unresolved_' + killer.lower(), 'no killer bomb found'

    bx, by, ttl0 = killer_bomb
    t = ticks[plant_i]
    mask = t.get('safe', ['111111', '111111'])
    escapes = mask_moves(mask)
    mine = (bx, by) in [tuple(m) for m in t.get('mine', [])]

    # 2) was the escape route sealed AFTER planting by a NEW bomb?
    sealed = None
    act_delta = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0),
                 'RIGHT': (1, 0), 'WAIT': (0, 0)}
    corridor = [(t['pos'][0] + act_delta[a][0],
                 t['pos'][1] + act_delta[a][1])
                for a in escapes if a in act_delta]
    for tt in ticks[plant_i + 1:]:
        for (b2x, b2y, b2t) in tt.get('bombs', []):
            if b2t >= 3:  # fresh plant (timer 4 seen as >=3)
                if (b2x, b2y) == (bx, by):
                    continue
                f2 = fields.get(tt['t'])
                if f2 is None:
                    continue
                # does the new bomb's blast cover our escape corridor?
                if any((ex, ey) in blast_set(f2, b2x, b2y)
                       for (ex, ey) in corridor):
                    sealed = (b2x, b2y, tt['t'])
                    break
        if sealed:
            break

    if mine:
        if sealed:
            return 'sim_miss', ('tick %d own bomb(%d,%d) sealed by '
                                'enemy(%d,%d)@%d esc=%d' % (
                                    t['t'], bx, by, sealed[0], sealed[1],
                                    sealed[2], len(escapes)))
        if len(escapes) <= 1:
            return 'corner_pin', 'tick %d own bomb(%d,%d) esc=%d' % (
                t['t'], bx, by, len(escapes))
        return 'own_bomb_chain', 'tick %d own bomb(%d,%d) esc=%d' % (
            t['t'], bx, by, len(escapes))
    # enemy bomb killed us
    if sealed or len(escapes) >= 2:
        return 'enemy_trap', 'tick %d bomb(%d,%d) esc=%d%s' % (
            t['t'], bx, by, len(escapes),
            ' sealed@%d' % sealed[2] if sealed else '')
    if len(escapes) <= 1:
        return 'corner_pin', 'tick %d enemy bomb(%d,%d) esc=%d' % (
            t['t'], bx, by, len(escapes))
    return 'enemy_lucky', 'tick %d bomb(%d,%d) esc=%d' % (
        t['t'], bx, by, len(escapes))


def mask_moves(mask):
    """Decode valid&safe mask strings -> escape move count (5 actions)."""
    try:
        valid, safe = mask[0], mask[1]
    except Exception:
        return []
    acts = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
    return [a for i, a in enumerate(acts)
            if i < len(valid) and valid[i] == '1' and safe[i] == '1'
            and a != 'BOMB']


def mask_safe_action(t):
    try:
        acts = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
        i = acts.index(t.get('act', 'WAIT'))
        return t['safe'][1][i] == '1'
    except Exception:
        return False


def missed_kills(rounds):
    """Exact this-tick kill certificates attributable to US: enemy stands
    in the blast of a bomb with ttl==1 whose position is in our planted
    history, AND every escape option (stay + 4 free neighbors) is
    covered by the union of detonating blasts. Deduped across
    consecutive ticks; cross-checked against KILLED_OPPONENT events."""
    hits, kills_realized = [], 0
    for rnd in rounds:
        for idx, t in enumerate(rnd['ticks']):
            if 'field' not in t or not t.get('bombs'):
                continue
            import numpy as np
            field = np.asarray(t['field'], dtype=int)
            detonating = [b for b in t['bombs'] if b[2] <= 1]
            if not detonating:
                continue
            union = set()
            for b in detonating:
                union |= blast_set(field, b[0], b[1])
            ours = any((b[0], b[1]) in [tuple(m) for m in t.get('mine', [])]
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
                    # KILLED_OPPONENT for this tick arrives in the NEXT
                    # tick's event send (we act, engine evaluates after)
                    kill_ev = any(
                        'KILLED_OPPONENT' in ev
                        for tt in ticks[idx:idx + 2]
                        for ev in tt.get('ev', []))
                    hits.append({'round': rnd['meta'].get('round', '?'),
                                 't': t['t'], 'enemy': o[0],
                                 'epos': list(epos),
                                 'bomb': [detonating[0][0],
                                          detonating[0][1]],
                                 'ttl': detonating[0][2],
                                 'ours': ours,
                                 'killed_opp_event': kill_ev})
    # dedup: consecutive ticks for the same (round, enemy)
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
        p = os.path.join(REPO, 'results', 'diag_e86_%s_deaths.jsonl' % b)
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
    out_csv = os.path.join(REPO, 'results', 'diag_e86_attribution.csv')
    if rows:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['batch', 'round', 'killer',
                                              'cause', 'detail'])
            w.writeheader()
            w.writerows(rows)

    # missed-kill inventory over all logged rounds
    mk = missed_kills([r for _, r in all_rounds])
    with open(os.path.join(REPO, 'results', 'diag_e86_missedkills.csv'),
              'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['round', 't', 'enemy', 'epos',
                                          'bomb', 'ttl', 'ours',
                                          'killed_opp_event'])
        w.writeheader()
        w.writerows(mk)

    # report
    from collections import Counter
    causes = Counter(r['cause'] for r in rows)
    killers = Counter(r['killer'] for r in rows)
    n_rounds = len(all_rounds)
    lines = [
        '# E86 Phase A — death attribution (frozen ship, %d rounds)' % n_rounds,
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
        lines.append('| %s | %d | %.0f%% |' % (c, n, 100 * n / len(rows)))
    # corner-pin split: own vs enemy bomb
    own_pin = sum(1 for r in rows if r['cause'] == 'corner_pin'
                  and 'own bomb' in r['detail'])
    lines += ['', 'corner_pin split: own-bomb %d / enemy-bomb %d' % (
        own_pin, causes.get('corner_pin', 0) - own_pin)]
    # per-batch
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
    out_md = os.path.join(REPO, 'results', 'diag_e86_deaths.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
