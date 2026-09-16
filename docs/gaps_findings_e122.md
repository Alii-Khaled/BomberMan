# Gap findings (E122 Phase 0) — ship = arbiter_ng E108+E112+E119

Instrumentation: `ARBITER_GAP_DIAG` per-tick recorder (behavior-neutral,
bit-parity verified on seed-42 solo 6 rds: identical per-round scores).
Batteries (CPU, tournament conditions): solo classic 100 rds (seed 0),
G1 vs 3x rule_based 40x2 (seeds 0/1), L5 vs 3x random 40x2 (seeds 0/1)
= 258 rounds, 70,282 solo ticks, 56,126 solo ticks with hidden coins.

## Issue 1 — adjacent-coin misses
- Certified misses (coin reachable <=2 steps via mask-safe path, not
  taken): **866 / 4,394 available (19.7%)**.
- Cause split: search_decided 817 (94% — a bomb/trek plan won the tick;
  by design the overlay must not override certified kills or plan
  prefixes), flee_locked 40 (overlay DOES cover these when the coin
  step is mask-safe), pi_ranked_below 9 (coin dir was pi-rank 2 on
  average vs chosen rank 1; the overlay directly fixes these).
- Interpretation: the ship collects visible coins well in opponent-ful
  play; the residual waste is (a) post-plant flee windows and (b) the
  search walking bomb-path prefixes past adjacent coins (legit trades,
  re-collected later). Promotion bar: pooled parity + no regression;
  the overlay's value concentrates in the flee window and OOD states.

## Issue 2 — opening tempo
- Opening (steps < 100): 8.71 bombs, 55.1 crates destroyed, 3.56 coins
  by step 100 per round (G1+L5+solo mixed); first opponent contact
  (<= Manhattan 4) at tick ~52 where it happens at all (76/258 rounds).
- **26% of opening bomb-plan ticks are margin-vetoed** (3,163/12,242)
  — bomb plans existed but lost the 0.6 margin arbitration. The
  conditional opening margin (ARBITER_OPEN_MARGIN, no-visible-coins
  window) targets exactly this class.

## Issue 3 — solo endgame waste (dominant)
- L5 (vs 3x random): rounds run to **~361/400 steps**, the ship is solo
  ~90% of every round; 22.8 bombs/round while hidden coins remain;
  final score 8.14/9 (0.86 unaccounted coins/round).
- Solo-hidden tick profile: WAIT 8.4%, **backtrack (immediate A<->B
  reversal) 33.7%, stuck-in-place (>=3 visits/12-tick window) 51.5%**.
- 26,459 no-bomb-plan solo-hidden ticks ~= post-plant lockout windows
  (4,435 bombs x ~6 ticks): during the lockout the move fallback (pi,
  OOD) wanders/backtracks instead of walking to the next farm target.
- Solo 100 rds: 7.87 coins/round with a fat left tail (9/100 rounds
  <= 2 points, three 0-point rounds) — the user-visible freeze class.
- 4,143 margin-vetoed 1-crate bomb ticks (p_reveal-priced singles below
  the 0.15 margin).

## Arms queued
- E122: ARBITER_COINTAKE=1 (d=2) — flee-window + pi-rank coin recovery.
- E123: SOLO_MARGIN sweep {0.05,0.1,0.15}; SOLO_RADIUS {8,12};
  BACKTRACK {0.5,1.5}; SOLO_TREK re-screen (coin+yield+crate treks,
  solo-gated) for the lockout windows.
- E124: OPEN_MARGIN {0.4,0.5} x OPEN_T=100 (targets the 26% veto rate);
  HUNT_OPEN screening (expect reject per E82).

Promotion bars: per-phase pooled canonical battery vs fresh same-session
control; no target-leg regression; solo left tail (rounds <=2) must not
grow. Rejected arms documented, knobs stay default-off.

## Post-promotion re-diagnosis (same solo battery, seed 0, 100 rds)

| metric (solo, hidden>0 ticks) | ship pre-E122/123 | promoted | |
|---|---|---|---|
| coins collected / round | 7.87 | **8.03** | +0.16 |
| unaccounted coins / round | 0.90 | **0.71** | -0.19 |
| WAIT rate | 8.4% | **5.4%** | |
| stuck-in-place (>=3 visits/12) | 51.5% | **36.9%** | |
| 1-crate bombs margin-vetoed | 1,458 ticks | **140** | radius 8 farm executes |
| solo tail (rounds <= 2 pts) | 9/100 | **6/100** | |

Canonical gates (1120 rds each, fresh same-session control):
E122 arm 6.499/0.676 vs ctl 6.402/0.661 (+0.097/+0.015 win); composed
E123 arm 6.456/0.669 (+0.054 parity; solo class 8.15 vs 7.75 on the
40-rd screens). E124 opening arms rejected (om4 -0.65, om5 -0.15,
hunt -0.17 vs promoted base on G1 screens). Final promoted defaults:
ARBITER_COINTAKE=1, ARBITER_COINTAKE_D=3, ARBITER_SOLO_RADIUS=8.
