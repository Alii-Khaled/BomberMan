#!/usr/bin/env python3
"""Gap-diagnosis aggregator (E122 Phase 0).

Reads one or more <prefix>_gap.jsonl traces written by the ship's
ARBITER_GAP_DIAG recorder and prints per-issue statistics:

  Issue 1  adjacent-coin misses (visible coin within d<=2, coin-step
           was mask-valid+safe, still not taken) + cause breakdown
  Issue 2  opening tempo (steps < 100): bombs, crates destroyed, coins,
           first opponent contact, margin-vetoed bomb plans
  Issue 3  solo endgame waste: WAIT rate, backtrack/ping-pong rate,
           margin-vetoed 1-crate bombs, radius-blocked ticks

Usage:
  python3 scripts/diag_arbiter_gaps.py results/gaps_g1_gap.jsonl [...]
"""
import json
import sys
from collections import defaultdict

DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}


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
            t = rec.get('type')
            if t == 'round_meta':
                if ticks:
                    rounds.append((meta, ticks))
                meta, ticks = rec, []
            elif t == 'tick':
                if ticks is None:
                    ticks = []
                ticks.append(rec)
    if ticks:
        rounds.append((meta, ticks))
    return rounds


def main():
    rounds = []
    for p in sys.argv[1:]:
        rs = load_rounds(p)
        print(f'== {p}: {len(rs)} rounds')
        rounds.extend(rs)
    if not rounds:
        print('no rounds found')
        return

    # ---- Issue 1: coin-visible-in-2 ticks ----
    n_coin_ticks = 0
    n_ok = 0
    n_taken = 0
    causes = defaultdict(int)
    rank_pairs = []
    for _, ticks in rounds:
        for r in ticks:
            c = r.get('coin')
            if not c:
                continue
            n_coin_ticks += 1
            dirs = c.get('dirs') or []
            if not dirs:
                continue  # no on-path step (unreachable/blocked)
            if not c.get('ok'):
                continue  # candidate dirs exist but none valid+safe
            n_ok += 1
            if r.get('act') in dirs:
                n_taken += 1
                continue
            # cause breakdown
            if r.get('decided'):
                causes['search_decided'] += 1
            elif r.get('flee'):
                causes['flee_locked'] += 1
            elif r.get('mf'):
                causes['must_flee'] += 1
            else:
                cr = r.get('coin_rank', -1)
                ar = r.get('act_rank', -1)
                rank_pairs.append((cr, ar))
                if cr > 0:
                    causes['pi_ranked_below'] += 1
                else:
                    causes['pi_tie_or_other'] += 1
    print('\n[Issue 1] coin-visible(d<=2) ticks: %d' % n_coin_ticks)
    print('  coin-step mask-safe available : %d' % n_ok)
    print('  taken                         : %d' % n_taken)
    print('  MISSED (exact-certified miss) : %d' % (n_ok - n_taken))
    for k, v in sorted(causes.items(), key=lambda kv: -kv[1]):
        print('    cause %-18s %d' % (k, v))
    if rank_pairs:
        import numpy as np
        arr = np.asarray(rank_pairs)
        print('    coin_dir pi-rank: mean %.2f (0=best); chosen-act rank '
              'mean %.2f' % (arr[:, 0].mean(), arr[:, 1].mean()))

    # round-level coin accounting
    seen_tot = coll_tot = hidden_tot = 0
    n_r = 0
    solo_gaps = []
    for meta, ticks in rounds:
        if not ticks:
            continue
        n_r += 1
        last = ticks[-1]
        seen_tot += last.get('seen', 0)
        coll_tot += last.get('coll', 0)
        hidden_tot += last.get('hidden', 0)
    if n_r:
        print('  round coin accounting over %d rds: revealed %.2f '
              'collected %.2f unseen(hidden) %.2f per round'
              % (n_r, seen_tot / n_r, coll_tot / n_r, hidden_tot / n_r))

    # ---- Issue 2: opening (steps < 100) ----
    op_rounds = 0
    bombs_open = crates_open = coins100 = 0
    contact_ticks = []
    veto_open = 0
    plan_ticks_open = 0
    for meta, ticks in rounds:
        if not ticks:
            continue
        start_crates = meta.get('start_crates')
        cr100 = None
        coll100 = 0
        contact = None
        for r in ticks:
            st = r.get('t', 0)
            if st < 100:
                if r.get('act') == 'BOMB':
                    bombs_open += 1
                s = r.get('srch')
                if s and s.get('bb') is not None and s.get('bm') is not None:
                    plan_ticks_open += 1
                    try:
                        if float(s['bb']) - float(s['bm']) \
                                <= float(s.get('me', 0.6) or 0.6):
                            veto_open += 1
                    except Exception:
                        pass
                if r.get('dopp', -1) >= 0 and r.get('dopp') <= 4 \
                        and contact is None:
                    contact = st
            else:
                if cr100 is None:
                    cr100 = r.get('crates')
                    coll100 = r.get('coll', 0)
        op_rounds += 1
        if cr100 is not None and start_crates is not None:
            crates_open += start_crates - cr100
        coins100 += coll100
        if contact is not None:
            contact_ticks.append(contact)
    if op_rounds:
        print('\n[Issue 2] opening (steps<100) over %d rounds:' % op_rounds)
        print('  bombs %.2f  crates destroyed %.2f  coins by step100 %.2f '
              '(per round)' % (bombs_open / op_rounds,
                               crates_open / op_rounds,
                               coins100 / op_rounds))
        print('  bomb-plan ticks %d, margin-vetoed %d (%.0f%% of plan ticks)'
              % (plan_ticks_open, veto_open,
                 100.0 * veto_open / max(1, plan_ticks_open)))
        if contact_ticks:
            print('  first-contact tick (opp<=4): mean %.1f over %d/%d '
                  'rounds with contact' % (sum(contact_ticks)
                                           / len(contact_ticks),
                                           len(contact_ticks), op_rounds))
        else:
            print('  first-contact tick (opp<=4): no contact in any round')

    # ---- Issue 3: solo endgame ----
    solo_ticks = 0
    solo_hidden = 0
    waits = 0
    backtracks = 0
    pos_hist = []
    ping = 0
    veto1 = 0
    blocked = 0
    bomb_solo = 0
    for _, ticks in rounds:
        prev = None
        pos_hist = []
        for r in ticks:
            if r.get('nopp') != 0:
                prev = tuple(r.get('pos', [0, 0]))
                continue
            solo_ticks += 1
            hidden = r.get('hidden', 0)
            if hidden:
                solo_hidden += 1
            a = r.get('act')
            pos = tuple(r.get('pos', [0, 0]))
            if hidden > 0:
                if a == 'WAIT':
                    waits += 1
                if prev is not None and a in DELTA:
                    dx, dy = DELTA[a]
                    if (pos[0] - dx, pos[1] - dy) == prev:
                        backtracks += 1
                if r.get('bl'):
                    if a == 'BOMB':
                        bomb_solo += 1
                    s = r.get('srch')
                    if s and s.get('bb') is not None \
                            and s.get('bm') is not None:
                        try:
                            gap = float(s['bb']) - float(s['bm'])
                        except Exception:
                            gap = None
                        if gap is not None and gap <= float(
                                s.get('me', 0.15) or 0.15):
                            veto1 += 1
                if not r.get('decided') and a != 'BOMB':
                    s = r.get('srch')
                    if (not s) or (s.get('bb') is None):
                        blocked += 1
            pos_hist.append(pos)
            prev = pos
            if len(pos_hist) > 12:
                pos_hist.pop(0)
            if pos_hist.count(pos) >= 3:
                ping += 1
    print('\n[Issue 3] solo ticks: %d (with hidden coins: %d)'
          % (solo_ticks, solo_hidden))
    if solo_hidden:
        print('  WAIT rate (hidden>0)      : %.1f%%'
              % (100.0 * waits / solo_hidden))
        print('  backtrack move rate       : %.1f%%'
              % (100.0 * backtracks / solo_hidden))
        print('  stuck-in-place (pos>=3/12): %.1f%%'
              % (100.0 * ping / solo_hidden))
        print('  bomb plans margin-vetoed  : %d ticks' % veto1)
        print('  no-bomb-plan ticks        : %d' % blocked)
        print('  bombs planted while hidden: %d' % bomb_solo)


if __name__ == '__main__':
    main()
