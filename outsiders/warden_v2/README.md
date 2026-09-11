# Warden v2

Versioned enhancement of the outsider sparring agent. Use the versioned
name (`warden_v2`) everywhere; never a bare `warden`.

Warden v1 stays frozen in `outsiders/warden_v1/` as the historical
reference. This directory is self-contained (numpy only, CPU-only, no
torch, no absolute paths).

## Validated results (E94, docs/experiments.md)

- G1 vs 3x `rule_based_agent`, paired 100x2 head-to-head vs `warden_v1`:
  v2 4.400 pooled / win 0.309 / rank 2.21 vs v1 4.260 / 0.342 / 2.24 —
  parity (single-sample battery numbers range 4.19-5.10 depending on the
  unseeded rule_based RNG, so paired runs are the evidence).
- Paired G1 vs `arbiter` (2x rule_based, 100x2): v2 4.670 / win 0.398
  vs arbiter 4.180 / 0.328 (supporting field, not the G1 protocol).
- STRONG lobby (`arbiter` + `overlord` + `sentinel`), 40x5 x2 samples:
  combined 10 seeds v2 score **5.60** vs arbiter 5.03 (+0.57), win
  **0.393** vs 0.377, rank 2.03 vs 2.00; vs v1 (5.160 / 0.357 / 2.09)
  the score edge holds. The strict "beat the ship everywhere" bar is
  directional, not statistically established (E94).
- Probe: `python3 scripts/probe_warden.py` 7/7 (engine blast/step
  parity, action validity, determinism, latency p99 < 20 ms).

## What changed vs v1

- **Corrected escape solver** (`safety.py`): the E88 latest-lethal test
  plus the arrival check before a tile is marked safe (v1 could certify
  moves into a live blast when danger windows were non-contiguous).
- **Boolean danger timeline** over `WARDEN_HORIZON` (default 8): no
  earliest-lethal information loss.
- **Deterministic seeded RNG** (`WARDEN_SEED`) for reproducible A/B.

## Optional arms (default OFF; screened in E94/E95)

- `WARDEN_SEARCH=rollout`: bounded exact-dynamics rollout search
  (`search.py`) over moves + certified bomb plans with CRN-paired greedy
  opponents; G1 2.370 (bomb suppression), keep off.
- `WARDEN_OPP_TRAP=1`: plant on opponents with no proven escape.
- `WARDEN_PLANT_ESC=2`: require two post-plant escape directions
  (aggression collapse, 0.675 STRONG score).
- `WARDEN_ESCAPE_COMMIT=1` + `WARDEN_PLANT_LOCAL=1` (E95 a12):
  proactive own-bomb flee (no target-chase, later first-lethal) and a
  local plant veto near opponents unless n_esc>=2 & hyp_dist<=2.
  Score-neutral G1 (10-seed 4.644 vs 4.652) with STRONG win 0.407 vs
  0.344, suicides -29% (0.39 vs 0.55) — E95 optional STRONG-field arm.
- `WARDEN_HORIZON` (6 loses to 8), `WARDEN_COIN_FIRST`,
  `WARDEN_SINGLE_CRATE(_DIST)`, `WARDEN_MULTI_CRATE_DIST`,
  `WARDEN_WAIT_PENALTY`, `WARDEN_MOBILITY_W`, `WARDEN_DEADEND_W`,
  `WARDEN_FLEE_OPP_W`, `WARDEN_OPP_AVOID_W`, `WARDEN_CRATE_GUARD_DIST`,
  `WARDEN_PLANT_NEAR_D`, `WARDEN_W_*` search/rollout weights.
- `WARDEN_DIAG_DIR` (E95): per-tick decision jsonl for
  `scripts/diag_warden_deaths.py` (default off).

## Run

```
python3 main.py play --no-gui --agents warden_v2 random_agent random_agent random_agent
```

Battery / probe:

```
python3 scripts/probe_warden.py
WARDEN=warden_v2 bash scripts/run_warden_battery.sh g1 100 0 1
WARDEN=warden_v2 bash scripts/run_warden_battery.sh strong 40 0 1 2 3 4
```
