# Sentinel experiment ledger

Living record of every experiment run on the Sentinel agent: what was changed,
why, what happened, and the verdict. Source material for report §6
(Experiments and Results) and §5 (Training) — see *Project Description* §9.
Companion files: `docs/training_stages.md` (stage dossier),
`results/figures/captions.md` (figure captions), raw data in `results/`.

## §0 How to use this file (read before appending)

1. **Append-only.** Copy the template from §4, take the next free ID, fill it in.
   Never rewrite old entries — supersede them with a new entry that references
   the old ID. History (including failures) is report material.
2. **Paired arenas for A/B.** Same `--seed` for all variants in one comparison
   (map RNG is fixed by seed). Note: seed does **not** fix agent RNG
   (`np.random.seed()` bare in `setup`, `shuffle` in rule_based) — see E09.
3. **Ship-decision bar.** Directional screens may use 40 rounds × 1 seed, but
   nothing ships without **≥100 rounds × 2 seeds** (run-to-run noise on
   40 rounds is ±0.8 score — proven in E09).
4. **Frozen protocol (tournament conditions).** `--train 0`,
   `--continue-without-training`, `SENTINEL_DML=0` (CPU inference, 0.5 s limit),
   `--scenario classic` unless stated. Primary metric: **score/round vs 3×
   rule_based**; always report coins/kills/suicides per round alongside.
5. **Artifacts.** Single-gate evals → `results/frozen_*.json` (kept out of the
   `eval_*` glob so `aggregate_eval.py` stays clean). Full matrices →
   `results/eval_m*_seed*.json` via `scripts/run_sentinel_eval.sh`; archive
   superseded matrices to `results/archive/` before re-running. Checkpoints →
   `agent_code/sentinel/checkpoints/`; pre-stage surgery backups →
   `results/archive/`. Training curve → `agent_code/sentinel/runs/metrics.csv`.
6. **Lineage + scoreboard.** When new weights become "the" model, add a row to
   §2 and update §5. When a gate/matrix lands, update §5.
7. **Author field.** Every entry carries author + date (report §9 requires
   author marks per section — lift these directly).

## §1 Model lineage (weights)

| Tag | Checkpoint | Ep | EMA | \|w\| | Status |
|---|---|---|---|---|---|
| Stage-1 nav best | `checkpoints/best_stage1_nav.pt` (+archive copy) | 244 | +52.9 | 85.5 | Reference: pure navigation peak |
| Pre-S4 (S3 exit) | `results/archive/last_pre_stage4_20260905_150128.pt` | 3040 | −7.3 | 447.4 | S5/S6 resume base (healthy Adam momentum) |
| S4 best | `results/archive/best_S4_ep4651.pt` | 4651 | −2.43 | 1905.0 | ✅ Shipped to `my-saved-model.pt` 2026-09-06 (E05/E06 winner) |
| S4 final | `results/archive/last_S4final_ep5040.pt`, `my-saved-model.final_ep5040.pt` | 5040 | −8.7 | 2139.1 | Archived (diverged — see E04) |
| S4 ep4200 | `results/archive/s4snaps/ep_004200.pt` | 4200 | −7.8 | 1656.5 | Least-diverged S4 snapshot (E05 candidate, rejected) |
| S5-final | `agent_code/sentinel/checkpoints/last.pt` (ep 3540 line) | 3540 | −4.42 | 17.5 | S6 resume base (converged, small-clean-Q — see E10/E11) |
| S6 best | `agent_code/sentinel/checkpoints/best.pt` | 3977 | −1.31 | 17.0 | Best classic EMA ever; E13 frozen 2.27 — rejected |
| S6 final | `agent_code/sentinel/checkpoints/last.pt` | 4540 | −7.72 | 15.3 | E13 frozen 1.98 — rejected; S5-final remains base |
| live | `agent_code/sentinel/my-saved-model.pt` | — | — | — | Tournament export (re-exported every training round) |
| Tag | Checkpoint | Ep | EMA | \|w\| | Status |
|---|---|---|---|---|---|
| SHIPPED combo | S5-final weights + margin gate + E20 activity tunings (see E14b/E20; payoff vetted then reverted E21) | — | — | — | ✅ Current `sentinel/`: gate 3.70 (s .29); w2 validation 3.56 pooled |

`best.pt` semantics: argmax-EMA tracker. Cross-regime EMA is incomparable
(Stage-1 +52 vs classic negative), so `best_ema` is reset at each stage start
with the prior best archived (E04 setup, E10/E12 surgery notes).

## §3 Experiment entries

### E01 — Stage 1 navigation (Task 1) ✅
- **Author:** Ali Mahbob · **Date:** 2026-09-04/05
- **Question:** Can the Dueling-MLP learn board navigation + coin greed with no bombs?
- **Setup:** solo, `coin-heaven` (0 crates, 50 coins), 500 rounds, DML training,
  CPU act. `SENTINEL_DML=1 … --agents sentinel --train 1 --scenario coin-heaven
  --n-rounds 500 --save-stats results/sentinel_stage1.json`
- **Results:** 24,319 coins / 500 rounds (**48.6/round**), 0 suicides,
  EMA ≈ +52.7. Bar (≥45/round) passed.
- **Verdict:** SHIP as navigation base. Artifacts: `sentinel_stage1.json`,
  metrics eps 1–500. **Report:** §6 training-start baseline.

### E02 — Stage 2 bombs + escape (Task 2) ✅ with caveats
- **Author:** Ali Mahbob · **Date:** 2026-09-05
- **Question:** Learn crate bombing + escape without dying, keeping navigation?
- **Setup:** solo classic (0.75 density, 9 hidden coins), 1500 rounds
  (`results/sentinel_stage2.json`, eps 501–2000).
- **Results:** regime shock at ep 501 (reward +52→−23, buffer 98k→401 refill,
  ε already floored 0.05); suicide spiked ~0.85 then decayed to ~0.26;
  coins ~0.7/round; meanRR ≈ −1.1. Navigation partially forgotten (later
  proven by E06 m1: 17.2 coins vs Stage-1 48.6).
- **Verdict:** SHIP with caveats (survival learned before profit).
  **Report:** §6 regime-shock figure (`training_performance_split`).

### E03 — Stage 3 hunting (Task 3) ✅
- **Author:** Ali Mahbob · **Date:** 2026-09-05
- **Question:** Hunt peaceful (easy) + coin_collector (hard)?
- **Setup:** classic, `sentinel peaceful_agent coin_collector_agent`,
  ~1000 rounds (`results/sentinel_stage3.json`); mid-stage stop/resume from
  `last.pt` verified (DMLAdam, zero `aten::lerp` warnings).
- **Results:** final 210 rounds: 751 score, 241 coins, **102 kills (0.49/round
  ✅)**, 112 suicides. Kills bar met; EMA still negative (economy lags combat).
- **Verdict:** SHIP. **Report:** §5 (resume discipline) + §6 (kills curve).

### E04 — Stage 4 full combat + Lion (Task 4) ❌ (negative result, kept)
- **Author:** Ali Mahbob · **Date:** 2026-09-05/06
- **Question:** Does the optimized stack (Lion + schedule + UTD3/B512/EOR16)
  beat rule_based? Lion won the supervised valley-proxy benchmark (1.57× step,
  29.9 vs 30.6 — `docs/training_stages.md` optimizer note).
- **Setup:** `SENTINEL_DML=1 SENTINEL_OPT=lion SENTINEL_SCHEDULE=1
  SENTINEL_UTD=3 SENTINEL_BATCH=512 SENTINEL_EOR_UPDATES=16`, 2000 rounds vs
  3× rule_based from ep 3040 (`results/sentinel_stage4.json`, eps 3041–5040).
- **Results (training-time):** sentinel 1.99 score / 0.96 coins / 0.206 kills /
  **0.65 suicide** per round vs rule_based ~2.28 / ~1.66 / ~0.12 / ~0.25.
  250-round blocks flat throughout (coins ~1.0, kills ~0.2, suicide ~0.8).
  **Divergence proof:** TD-loss median 760 → 2201 → … → **71,761** monotonic;
  `|w|` 447 → 2139 (Stage-1 ref: 85). Mechanism: MSE on ±100 TD errors →
  exploding gradients → Lion `sign()` full-LR steps → PER feedback loop;
  UTD3 reheats narrow ε=0.05 data; N_STEP=3 < BOMB_TIMER=4 hides suicide from
  the return window. Valley proxy ≠ bootstrapped TD — falsified on real TD.
- **Verdict:** REJECT Lion-1e-3/MSE/UTD3 for TD learning. Motivates E10.
  **Report:** §5 core negative result + §6 divergence figure.

### E05 — Frozen candidate triple (paired, 100 rounds, seed 0) ✅
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Which S4 checkpoint actually plays best frozen?
- **Setup:** isolated agents (`sentinel_c_*`, `q_net` extracted to
  `my-saved-model.pt` format — full payloads are **not** directly loadable by
  `callbacks.py`), vs 3× rule_based, identical arenas →
  `results/frozen_c_{best,last,4200}.json`.
- **Results:**

| Candidate | Score | Coins | Kills | Suicide | Steps |
|---|---|---|---|---|---|
| best4651 (EMA −2.43) | **2.24** | 0.99 | 0.250 | **0.58** | 194 |
| last5040 (EMA −8.7) | 2.09 | 1.14 | 0.190 | 0.58 | 187 |
| ep4200 (least-diverged) | 2.25 | 0.95 | 0.260 | 0.66 | 166 |

- **Verdict:** SHIP best4651 (best score tied, fewest suicides, longest
  survival) → installed to `agent_code/sentinel/my-saved-model.pt`; S4-final
  archived. **Report:** §6 model-selection table.

### E06 — M1–M8 matrices: Stage-2-exit vs best4651 (40 rounds × seeds 0,1)
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Where did Stage-4 help, matchup by matchup?
- **Setup:** stock `scripts/run_sentinel_eval.sh` + `aggregate_eval.py` +
  `plot_eval.py` on best4651 (stale S2 matrix preserved in
  `results/archive/s2eval/`). Figures refreshed (4 stages detected).

| Matchup | Before (S2 exit) | Now (best4651) |
|---|---|---|
| m1 solo coin-heaven | 17.2 | **50.0** (navigation recovered!) |
| m2 solo classic | 0.71 | 0.72 |
| m3 vs 3× random | 0.50 | 0.26 |
| m4 vs 3× peaceful | 0.33 | 0.33 |
| m5 vs 3× collectors | 1.44 (k .15 / s .55) | 2.00 (k .23 / s .51) |
| m6 vs 3× rule_based | 1.21 (k .08 / s .64) | **2.79** (k .33 / s .66) vs rb 3.39/3.83/3.11 |
| m7 vs overlord+2×rb | 1.41 (k .14 / s .68) | 1.92 (k .20 / s .66) |
| m8 vs warden+2×rb | 1.76 (k .16 / s .60) | 2.19 (k .24 / s .55) |

- **Verdict:** combat learned (m6 kills ×4) but suicide flat ~0.6 everywhere;
  Task-4 gate (beat best rule_based 3.83) FAILS. Think time ~0.5–1 ms/step
  (limit 500 ms, `eval_thinktime`). **Report:** §6 head-to-head + safety
  panel figures.

### E07 — Q_WEIGHT screen (40 rounds, seed 0, on best4651)
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Is the diverged Q helping or hurting at act time?
 (`callbacks.py:88`, Q clipped ±3.)
- **Results:** Q0 (heuristic-only) 2.35 / k .300 / s .625; Q0.5 **1.43 /
  k .100 / s .80 (collapse)**; Q1.0 2.25 / k .250 / s .65.
- **Verdict:** more diverged-Q weight hurts or is neutral; diverged Q is
  clipped ±3 noise (±0.6 at weight 0.2). Keep 0.2 (ML-compliance, no evidence
  against). Re-sweep queued on clean Q (E13). **Report:** §6 ablation row.

### E08 — Bomb-gate / flee screens (40 rounds, seed 0, on Q0 base)
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Which safety-threshold change cuts suicide?
- **Setup:** flee commit 5→7 steps; strict gate (crate-only bombs need
  crates_hit ≥ 2, warden-style); threat veto (no BOMB while dodging).
- **Results:** flee7 **2.95 / k .325 / s .45** ✨; strict 2.48 / s .625
  (marginal); threat-veto 1.43 / k .125 / s .675 (starves without saving ❌).
- **Verdict:** flee7 looked like +26% — sent to high-N validation (E09).

### E09 — High-N validation + noise law (200 pooled rounds) ⚠️ methods result
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Does flee7 survive scale-up (100 rounds × seeds 0,1)?
- **Results:** flee7 **1.94** / k .205 / s .620 vs baseline **1.95** / k .195 /
  s .630 — identical. Seed-0 "win" was noise.
- **Noise law (report-worthy):** rerunning *identical* weights/code/seed gives
  2.98 once, 1.38 the next — `--seed` fixes maps but agent RNG is unseeded
  (`np.random.seed()` bare in `setup`, `shuffle` in rule_based). 40-round
  screens carry **±0.8 run-to-run noise**; ship bar (§0.3) follows from this.
  (Also fixed an analysis-script bug mid-way: accumulator indented into the
  metric loop divided everything by 8× — raw JSONs were always correct.
  Always sanity-check pooled means against single-file means.)
- **Verdict:** REJECT flee7 (neutral), REJECT threat-veto (harmful), KEEP
  stock thresholds. Suicide is policy-level, not threshold-level → E10.
  **Report:** §4 methods (eval protocol + noise) + §6.

### E10 — Stage 5 Arm A: anti-divergence bundle (500 rounds) ✅
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Does Adam-tuned + Huber + UTD1/EOR4 + N_STEP 5 + ε re-warm cure
  TD divergence and resume learning?
- **Setup:** resume pre-S4 healthy weights (ep 3040, `|w|` 447, Adam momentum
  kept), `best_ema` reset, ε re-warm 0.30 (checkpoint `epsilon_steps=36842`),
  vs 3× rule_based, fixed LR 1e-3, DML (`results/sentinel_stage5_armA.json`,
  eps 3041–3540). Code: Huber δ=1 (`train.py` loss), N_STEP 5 (suicide now
  inside return window), clip 5→1, `SENTINEL_TUNED=1` (eps 1e-4/decay 1e-4,
  re-applied after resume which would otherwise restore old groups). Probe:
  Huber 100.1 vs MSE 15001.6 on identical TD errors (150×).
- **Results:** lossMed 2.40 → 0.44 → 0.10 → 0.032 → **0.024** (monotonic fall);
  `|w|` 447 → 103.6 → 31.3 → **17.5** (deflated-stable, no explosion);
  ε 0.30 → 0.05 with early exploration highs (+16/+18 rounds); EMA
  → −4.42 and rising. Training-time: 2.01 / 0.87 / 0.228 / s 0.664.
  Suicide flat ~0.8 in training curves (ε-noise + heuristic ceiling).
- **Verdict:** OPTIMIZATION CURED.behavior flat-in-training → judge frozen
  (E11). Lion (Arm B) shelved: it would discard a healthy converged state to
  re-test a falsified hypothesis. **Report:** §5 centerpiece + §6 curves.

### E11 — S5-final frozen (100 rounds, seed 0, paired with E05) ✅ best yet
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** What is the converged small-Q policy worth frozen?
- **Setup:** `last.pt` ep 3540 isolated → `sentinel_s5` →
  `results/frozen_s5.json`.
- **Results:** **2.70 / 1.05 / 0.330 / 0.630 / 177 steps** vs best4651's
  2.24 / 0.99 / 0.250 / 0.580 — **+20% score, +32% kills**. Theory: Huber
  deflated weights to the right scale; the small clean Q + heuristic beats
  the noisy big Q. Follow-up hypothesis: with calibrated Q, a Q_WEIGHT
  re-sweep (E13) may now pay — E07 ran on diverged Q.
- **Verdict:** S5-final is the best frozen policy to date. Gate still open
  (2.70 vs 3.83). Continue learning from this healthy state → E12.

### E12 — Stage 6 extension (1000 rounds) ❌ gate failed = plateau proven
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Does continued healthy learning break the suicide plateau?
- **Setup:** resumed S5-final in place (ep 3540, momentum kept), second ε
  cycle (0.20 → 0.05), best-tracking re-armed to −4.42, identical Arm A
  bundle, vs 3× rule_based → `results/sentinel_stage6.json` (eps 3541–4540).
- **Results (200-round blocks):** lossMed flat 0.019–0.025 (converged floor,
  nothing left to squeeze); `|w|` 21.0 → 15.3 (tiny, stable); suicide
  0.815 / 0.755 / 0.720 / 0.810 / 0.795 — **flat ~0.78, gate (<0.6) FAILED**;
  EMA oscillates (−3.3 to −7.7, never past −2 — second gate leg FAILED);
  coins ~1.0 flat, kills 0.21–0.27; training-time 2.15 / 0.99 / 0.233 /
  s 0.659 ≈ S5-training. **Best-tracking worked:** `best.pt` = ep 3977,
  EMA −1.31 — best classic-regime EMA ever (S5 −4.42, S4 −2.43).
- **Verdict:** PLATEAU PROVEN — 1500 rounds (S5-500 + S6-1000) of healthy
  optimization, zero behavioral movement. Per plan, no more same-config
  rounds. Two threads remain: (a) frozen evaluation may still flatter
  (training curves understated S5 by +20%) → E13; (b) blocker is now
  reward-shaping/heuristic-ceiling → E14. **Report:** §6 plateau evidence
  (flat suicide + floored loss side by side).

### E13 — S6 frozen + clean-Q weight sweep ❌ both rejected, S5-final stands
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Did S6 improve the frozen policy, and does calibrated Q want
  more weight?
- **Setup:** S6-best (ep3977) + S6-last (ep4540) isolated → 100 rounds seed 0
  (paired with E05/E11) → `results/frozen_s6b.json`, `results/frozen_s6l.json`.
  Then Q_WEIGHT {0.5, 1.0, 2.0} on S5-final weights (40rd seed-0 screen,
  validate at 100×2 per §0.3).
- **Results:** s6b 2.27 / k .260 / s .680; s6l 1.98 / k .190 / s .700 —
  **S6 drifted backwards frozen** (S5-final 2.70); 1000 extra rounds for −16%.
  EMA-best ≠ frozen-best again. Sweep screen flashed (Q1.0: **4.00** / k .500 /
  s .425, beats rule_based!) but 200-round validation falsified it: Q1.0
  **2.08** / k .205 / s .600 vs baseline **2.29** / k .250 / s .585 — the 4.00
  was a lucky run inside ±0.8 noise (E09 law holds). Q0.5/Q2.0 screens neutral.
- **Verdict:** REJECT all. Q_WEIGHT stays 0.2; S5-final (2.70) remains champion
  and training base. Variant agents removed (`sentinel_s5` kept as reference).
  **Report:** §6 (falsified flash + validation discipline as methods example).
### E14 — Shaping redesign, attribution-first
- **Author:** Ali Mahbob · **Date:** 2026-09-06

#### E14a — death attribution ✅ DECISIVE: 78% own-bomb
- **Question:** What kills Sentinel — own bombs or enemy bombs? (Unanswerable
  from existing data: `train.py` conflated both into one counter.)
- **Setup:** (1) permanent additive `train.py` split: `_round_killed_self` /
  `_round_got_killed` partition `_round_suicides` exactly (suicide rounds emit
  BOTH KILLED_SELF from own blast AND GOT_KILLED from the removal loop, which
  tags every death — own bomb takes precedence), new metrics cols,
  `suicides` kept for plot compat; verified by fake-self unit probe incl. the
  double-event case. (2) temporary env-gated engine death log
  (`SENTINEL_ATTRIBUTION=1`, victim/owner/step/pos per lethal hit — 60 frozen
  rounds S5-final vs 3× rule_based, seeds 0+1) → `results/e14a_deaths.jsonl` +
  `replays/e14a_s*.pt`. Engine edit **reverted after** (`git diff
  environment.py` empty — tournament framework pristine).
- **Results (60 rounds, 156 unique deaths):** sentinel 45 deaths (0.75/round):
  **35 own-bomb (78%) vs 10 enemy (22%)**, 5 overlaps. Deaths strike EARLY:
  mean step 130; steps 13, 13, 16, 20 near spawn corners/edges ([14,1] ×3,
  [15,14]). Rule_based baseline: 76% own-bomb too — but sentinel dies 0.75/rd
  vs their ~0.5. Sentinel killed 17 rb (0.28/rd ✓). One opponent (rb_2) dealt
  7/10 enemy kills — seating/proximity note.
- **Verdict:** E14b (escape-solver audit) WINS over E14c. Suicide is failed
  escape, not combat variance. Suspects, ranked: (i) `can_escape_if_bomb`
  false positives — snapshot BFS ignores moving opponents/enemy bombs, and
  the fallback `any valid (even unsafe)` path can plant with no escape;
  (ii) flee commitment expiring exactly at the lingering-lethal step
  (timer 5 vs blast at relative 4+5); (iii) spawn-area bombing with
  underestimated 3–4-step escapes in 0.75-density crates.

#### E14b — escape-solver audit ✅ ROOT CAUSE + FIX SHIPPED, gate PASSED
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Audit method (all reverted after; `git diff environment.py` empty):**
  observer agent (`sentinel_audit` → delegates to S5-final, bit-identical)
  logging every BOMB decision with the safety verdict (`can_escape`,
  `dist_hyp`, `crates_hit`, `opps_hit`, `safe_any`) + temporary gated engine
  hooks for effective plants and lethal hits. Pitfalls hit and solved:
  decisions ≠ plants (rejected BOMBs logged too — plant log is ground truth),
  round-key mismatch (`game_state['round']` int vs `round_id` string).
- **Findings (60 rounds, 36 own-bomb deaths):** 100% planted with
  `can_escape=True` and safe moves available (ranking failure, not fallback);
  plant→death gap EXACTLY 4 steps every time (never outran the first wave);
  `dist_hyp` 4.0 → **23/27 fatal (85%)**, ≤3.0 → 2.4%. All 27 dist-4 plants
  had opponents in blast — the kill-exception was backwards (near-certain
  mutual destruction). dist-3 dead-end crate bombs: ~97 plants, 0 deaths
  (reliable, kept).
- **Fix (one gate, `safety.py`):** forbid BOMB at `dist_hyp > 3`
  (dist==timer always loses the race; kill or not). Heuristic follows
  automatically via `can_escape_if_bomb`. Deferred (unneeded): pure-flee
  boost, corridor ban (already covered). Threshold considered and rejected:
  ≥3 would gut productive bombing (7 plants/round at 2.4% fatality).
- **Weights:** S5-final installed to `sentinel/` (S6-last export archived;
  S6 optimizer state retained in `checkpoints/` for future retrains).
- **Gate (200 pooled rounds vs 3× rule_based):** sentinel **3.70** / 1.50 /
  k 0.440 / **s 0.290** / 269 steps vs best rule_based **3.50** — TASK-4 GATE
  PASSED both legs (was 2.70/0.63). Suicide more than halved; kills +33%.
  Side effect: invalid actions 1.6 → 3.1 (accepted, noted).
- **Full M1–M8 on shipped combo:** m1 50.0 · m2 1.05 · m3 0.60 · m4 0.46 ·
  m5 **4.20** (beats all collectors, k 0.61) · m6 3.38 (s 0.35, was 0.66) ·
  m7 3.62 (overlord 1.48) · m8 2.74 (warden still leads 6.03 — next role
  model). Suicide collapsed in every combat matchup (0.51–0.68 → 0.23–0.35);
  kills up everywhere (longer survival → more opportunities).
  Previous matrix preserved in `results/archive/s4best_matrix/`.
- **Verdict:** SHIP (already live). **Report:** §6 centerpiece — audit table
  (P(death|plant) by dist_hyp), gate table, head-to-head + safety figures.
- **Figure caveat (for report prep):** `metrics.csv` reuses episode numbers
  across stages (S5/S6 appended after S4), so `training_*` figures mix regimes
  in late segments — eval figures are unaffected. Fix before report: add
  positional `stage` column + teach `plot_eval.detect_stages` to prefer it.
- **E14c (if E14b audit clears the solver):** opponent-threat features +
  hunt/endgame retune → short retrain from S5-final.
- **E14d (no training):** document tournament-expected score in weaker fields
  (m3–m5 style) — the 3× rule_based gate is the worst case, not the field.
- **E14e (if a–d stall):** capacity 256→512 + scaffolded opposition
  (1× rule_based + 2× peaceful → full 3×).

### E17 — Payoff gate + solo_relax screens ✅ payoff shipped, relax rejected
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Does the E16 gate pay, and does solo_relax unlock solo economy?
- **Setup:** payoff gate at mask level (`safety.py`, after margin gate) on
  live `sentinel/`; `solo_relax` (transit veto −1.0 → 0.0 when no living
  opponents) as separate variant dir. Screens 40rd seed 0 (paired): m2 solo
  live 1.00/7.7b vs relax 0.82/8.2b; m6 live 4.12/k.500/s.200 vs relax
  4.10/k.500/s.200.
- **Results:** solo_relax is a no-op — bombs flat because the *mask* (margin
  gate, corridor rule) still vetoes; heuristic-only relax can't free solo
  bombing. REJECTED in this form (mask-level solo mode = future work).
  Relax correctly inert in combat (identical m6) — good consistency check.
  Honest accounting: margin gate is the proven part (paired 200rd +1.41 score,
  −0.34 suicide); payoff's isolated increment is unproven but
  mechanism-backed (vetoed plants were 2.6%-killers at 2.4% death risk each)
  with zero regression signal → kept.
- **Full M1–M8 on shipped combo (margin + payoff + S5):** m1 50.0 · m2 1.05 ·
  m3 0.89 · m4 0.41 · m5 **3.86** (sweeps collectors, k 0.49) · m6 **3.77** vs
  rb 3.33/3.05/2.98 (s 0.31) · m7 3.25 (overlord 1.48) · m8 2.74 (warden 5.25).
  Previous matrix in `results/archive/margin_matrix/`. Dedicated 200rd gate:
  **3.70 vs 3.50, s 0.29 — PASS.**
- **Verdict:** SHIP (live). Solo economy stays open. **Report:** §6.
- **⚠️ CORRECTION (2026-09-06, see E21):** the payoff gate as shipped was
  **dead code** — `crates_hit`/`opps_hit` are bound at function end, so the
  gate raised UnboundLocalError on every call, swallowed by `except: pass`.
  All E17 "payoff" numbers measured margin-only behavior. Fixed, truly
  live-tested (200rd pooled: 3.61 vs 3.70, suicide +0.07, crates flat —
  freed slots don't convert without a crate-approach pull), then REVERTED.
  Live = margin gate + E20 tunings (== validated w2 behavior).

### E21 — Dead-gate saga: payoff veto never ran, then measured neutral ❌
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Discovery:** diffing `overlord/safety.py` (identical pre-gate file) exposed
  that E17's payoff block referenced `crates_hit`/`opps_hit` bound 10 lines
  *below* it → UnboundLocalError every call → `except: pass` hid it. The
  `except Exception: pass` anti-pattern is now a shipping hazard: every
  safety heuristic ships with a direct unit probe (see below).
- **Fix + probe:** rebound to `crates_tmp`/`opps_tmp`; 4-case probe
  (open→veto, crates→allow, early-opp→allow, late-opp→veto) ALL PASS.
- **Live retest (200 pooled, same seeds as margin baseline):** payoff-live
  3.61 / 1.53 / k.415 / s.360 / bombs 12.2 vs margin-only 3.70 / 1.50 / k.440
  / s.290 / bombs 20.5. Per-seed consistent (s0 −0.24/+0.04, s1 +0.06/+0.10).
  Bombs halved, crates flat 11.2, coins flat — vetoed slots become movement
  without a crate pull, and predicted −0.2 suicide savings never appeared.
- **Verdict:** REVERTED (code removed, one-line NOTE left). Forensics table
  (E16) stands; the missing piece is a crate-approach attractor, not vetoes.
  **Lesson for report §6:** silent-except + no probe = untested code shipped;
  process fix: probe every mask heuristic directly.
- **Net live state:** S5 weights + margin gate + E20 tunings == validated w2
  (3.56 pooled) — no re-validation needed for the revert.

### E18 — (reserved) Outsider eval sparring (on file delivery, eval-only)
### E19 — Overlord Arm-A port + curriculum O1–O4 ✅ launched overnight
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Survey:** Overlord (ResNet-CNN: stem-64 + 2 resblocks, GAP+GMP, dueling +
  aux danger head, ~2M params) sat at 75 eps / 305 updates (buffer never
  filled past MIN_REPLAY 5000) — effectively untrained (m7 frozen 1.48).
  Gaps found by reading: NO epsilon system (greedy from round 1 — confirmed
  in callbacks), MSE td+aux, clip 5.0, no tuned flag, no margin gate, and a
  **stale-actor bug**: `setup()` JIT-traces once and `act()` prefers the
  trace forever, so training would have acted on setup-time weights all run
  (tournament path unaffected — fresh trace of final weights).
- **Port (all verified by probes, no game):** Huber td+aux, clip 1.0,
  `OVERLORD_TUNED` (AdamW eps/decay 1e-4 + post-resume re-apply), full
  epsilon-greedy 1.0→0.05/50k (constants + schedule + setup/resume/payload/
  metrics col + `_update` increment + safety-masked pool in `act`),
  E14a counter split, margin-gate transplant (payoff deliberately NOT ported
  — E21 rejected it), WAIT −0.30, trace-only-when-eval. Kept: N5, GAMMA 0.97,
  LR 3e-4, B512, buffer 300k, BN layers (eval-mode verified; GroupNorm
  fallback noted if unstable).
- **Curriculum** (`scripts/train_overlord_curriculum.sh`, boosted): O1
  coin-heaven **750** (CNN cold start + ε burn-in) → O2 solo 1500 → O3 hunt
  1000 → O4 vs rule_based 2000 + frozen eval; per-stage best-archive + EMA
  reset + ε re-warm 0.25 inline (sentinel S5/S6 procedure). Fresh start
  (75-ep state archived to `results/archive/overlord_pre_O1/`).
- **O1 live:** PID 5922, DMLAdamW, ε 1.0 decaying, 19–28 coins/round by ep 14,
  Huber loss ~0.01, duty ~95%. ETA covers O1+O2 overnight.
- **2026-09-06 resume note:** first O1 run vanished at ep 28 with no traceback
  (likely external kill; checkpoints healthy: last=best=ep 28, EMA 21.9).
  Relaunched same config resuming `last.pt` with `STAGE_O1_N=722` (lands O1
  at ep 750 total). Resume verified: ep=28/steps=6305, tuned preset
  re-applied post-load, ε continuous 0.88→0.87, duty 93%.
- **2026-09-06 silent deaths ×2 explained:** run died again at ep 56 (06:44)
  with zero traceback and healthy checkpoints; `dmesg` on next boot showed
  prior journal "corrupted or uncleanly shut down" — the WSL VM itself went
  down (Windows sleep/shutdown kills GPU training instantly, no traceback
  possible). Relaunched resuming ep 56 (`STAGE_O1_N=694`). Lesson: per-round
  checkpoints already bound any kill to ≤1 round lost, but **overnight runs
  require Windows sleep disabled while plugged in**. Watch item: 300k PER
  buffer → ~8 GB worst case vs 16 GB WSL cap; monitor if O4 approaches the
  cap.
- **Report:** §4 second-model story (spatial-CNN vs feature-MLP) + §5.
- **Postscript — maxpool CPU fallback found & fixed (2026-09-06):** GPU
  utilization probe (`aten::adaptive_max_pool2d.out` has no DML kernel —
  fallback every forward+backward, warning twice in training log, all else
  clean: 26/26 params on DML, BN/smooth_l1/AdamW native). Fix: `amax(dim)`
  (probed DML-native fwd+bwd; RMS-pool was the fallback candidate),
  2-line `model.py` change, O1 restarted fresh (only ~25 eps lost; partial
  archived to `results/archive/overlord_O1partial_maxpool/`). Pre-fix pace
  ~50 s/it at 25 s GPU/round; post-fix 17 s/it early (short rounds) —
  honest comparison at ep 10–20 to follow in the morning.
- **Ops lesson (tool-backgrounded runs):** bash `&` without job control leaves
  SIGINT/SIGQUIT ignored in children (`SigIgn 0x1001007`), and SIGTERM proved
  ineffective mid-GPU-wait — only SIGKILL stops a run. Per-round checkpoints
  make KILL safe (≤1 round lost). Future stops: KILL directly, never INT.

### E15 — Economy diagnosis: placement, not volume 📦
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Why do coins trail 1.5 vs 2.4 when bomb volume matches (20/round)?
- **Setup:** read-only mining of E14b audit artifacts (no new runs):
  crates_hit/opps_hit distributions over 756 scored sentinel plants
  (`/tmp/bomb_log_s*.jsonl` + plant ground truth), crates-vs-coins conversion
  from gate JSONs (`results/frozen_margin_s*.json`), m2 solo row (current matrix).
- **Results:** sentinel destroys **12 crates/round vs rule_based ~35** at equal
  bomb volume — **63% of plants hit 0 crates** (479/756) while 70% chase
  opponents (opps_hit>0). Coins follow crates, not bombs. m2 solo: 7.6 bombs,
  400 steps, 1.05 coins, 0 deaths — combat-tuned discipline (transit veto +
  pocket rules) chokes risk-tolerant solo play where death costs less.
  Invalids 3.1 vs 6–7 deprioritized (already cleaner than rule_based).
- **Verdict:** two levers queued — (a) kill-attribution forensics E16 writes
  the crate-payoff gate spec; (b) `solo_relax` (drop transit veto when no
  living opponents, keep margin gate + mask). **Report:** §6 economy table.

### E16 — Kill forensics: whose plants actually kill? ✅ gate spec written
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** What is P(kill) per plant type — which bombs pay, which waste?
- **Setup:** E14b hooks re-applied (+ plant log) on shipped-at-the-time combo
  (margin gate + S5 weights) via observer `sentinel_audit2`; 60 frozen rounds
  (30 × seeds 0,1, separated logs); sentinel-owned lethal hits joined to
  latest plant in [T−6,T] → verdict fields. Hooks reverted after
  (`git diff environment.py` empty). Unmatched/no-verdict kills: 0.
- **Results (1253 sentinel plants):**

| Plant type | Plants | Kills | P(kill) |
|---|---|---|---|
| crates_hit = 0 | 957 (76%) | 25 (68% of kills) | 0.026 |
| crates_hit ≥ 1 | 296 | 12 | 0.041 (2: 5/92 = 0.054) |
| opps_hit = 0 | 373 | 1 | 0.003 ❌ pure waste |
| opps_hit = 1, step ≤ 250 | 579 | 32 | **0.055** productive hunting |
| opps_hit = 1, step > 250 | 289 | 2 | 0.007 late hunting fails (open board, foes flee) |
| opps_hit ≥ 2 | 12 | 2 | 0.167 keep always |

- **Gate spec derived:** veto BOMB iff `crates_hit==0 AND (opps_hit==0 OR
  step>250)` — keeps all crate bombs, opps≥2, early single-opp; vetoes ~40%
  of plants costing ~0.05 kills/round while each carried 2.4% death risk plus
  a 6-step bomb-slot opportunity cost. **Report:** §6 exchange table.

### E20 — Idle pathology: WAIT-spam + pacing ✅ partially fixed, shipped
- **Author:** Ali Mahbob · **Date:** 2026-09-06
- **Question:** Why does sentinel sit still (30% early WAIT, 65% revisit moves)
  while rule_based WAITs 0%? (User-observed in live GUI m8 session; confirmed
  in its `game.log`.)
- **Setup:** observer trace (round, step, action, pos, bombs ticking) on live
  combo, 30rd m6 + 30rd m8 (`/tmp/otrace_*.jsonl`, hooks reverted after).
- **Measurement:** WAIT 30% (steps 1-50) → 25% → 13% → 8% as heuristic pulls
  strengthen over the round; 67+56 sit-streaks ≥5 (max 21); displacement only
  3.2 tiles by step 50; 65–68% of early moves revisit last-8 tiles; 321+200
  early WAITs with ZERO bombs ticking (pure scoring pathology). Base-rate
  checks exonerated both suspects: bombardment among WAITs (0.64–0.76) ≈ base
  rate (0.60–0.75), post-plant likewise — WAITs are uniform, i.e. flat
  scoring, not danger response. Mechanism: weak early pulls (hidden coins,
  far crates, hunt_w 0.5) leave moves near zero/negative (dead-end −0.35)
  while WAIT costs only −0.12 → WAIT wins strictly. Our own gates feed it
  (fewer plants → fewer forced repositionings; WAIT never vetoed).
- **Fix (shipped to live):** WAIT penalty −0.12 → −0.30 (mask still picks WAIT
  when it's the only safe move → survival unaffected) + early dead-end relax
  (−0.35 → −0.10 for step<150; early dead-ends hold crates). Screens: moves
  share 0.72 → 0.81, suicide held. Trace: early WAIT 30→25%, mid 25→15%.
  High-N (200 pooled, m6): **w2 3.56** / k .415 / s .335 vs live 3.46 / k .390
  / s .325 — outcome-neutral (+0.10) with strictly more active behavior and
  +20 survival steps. 40rd s0 flashes (4.12 live, 4.00 q10-style) again
  demonstrated ±0.8 noise — ship-grade numbers only from 100×2.
- **Verdict:** SHIP (live == validated w2, diff-checked). Residual early WAIT
  25% + 65% pacing remain open (crate-approach pull = next candidate).
  **Report:** §6 (behavioral figure: WAIT% by phase + displacement).

## §4 Append template (copy from here)

```markdown
### EXX — <short title> <emoji verdict>
- **Author:** Ali Mahbob · **Date:** YYYY-MM-DD
- **Question:** <single sentence>
- **Setup:** <weights tag from §2> + <code diff with file:line> + <opponents,
  N rounds × seeds, exact command> → <artifact path>
- **Results:** <table vs baseline with deltas; training curves if any>
- **Verdict:** SHIP / REJECT / NEEDS-N + reason; lineage/scoreboard updated?
- **Report:** §<4/5/6> + figure/table refs
```

## §5 Scoreboard (frozen best-per-matchup over time — lift to §6)

| Matchup | S2-exit matrix | S4-best matrix (E06) | Margin+S5 matrix (E14b) | Margin matrix rerun (E17; payoff gate dead throughout — see E21) | Best frozen gate |
|---|---|---|---|---|---|
| m1 coin-heaven | 17.2 | 50.0 | 50.0 | 50.0 | — |
| m2 solo classic | 0.71 | 0.72 | 0.72 | 1.05 | — |
| m3 vs random | 0.50 | 0.26 | 0.50 | 0.89 | — |
| m4 vs peaceful | 0.33 | 0.33 | 0.33 | 0.41 | — |
| m5 vs collectors | 1.44 | 2.00 | 2.00 | **3.86** | — |
| m6 vs rule_based | 1.21 (s .64) | 2.79 (s .66) | 3.38 (s .35) | **3.77** (s .31) | best4651 2.24 (s .58) → S5 2.70 (s .63) → S6b 2.27 (s .68, E13 reject) → margin+S5 3.70 (s .29, E14b ✅) → payoff-live 3.61 (s .36, E21 reject) → w2-live 3.56 pooled (s .34, E20 ✅ SHIP) |
| m7 vs overlord | 1.41 | 1.92 | 1.92 | 3.25 (overlord 1.48) | — |
| m8 vs warden | 1.76 | 2.19 | 2.19 | 2.74 (warden 5.25) | — |

Task-4 gate (frozen score > best rule_based ≈ 3.8, suicide ≤ 0.4): OPEN.
Key artifacts: `results/eval_summary.tex` (matrix table), `results/figures/`
(6 figures + `captions.md`), `results/frozen_*.json` (gates),
`agent_code/sentinel/runs/metrics.csv` (curves, 4 stages).
