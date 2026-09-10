# Sentinel training stages

Curriculum for the sentinel Dueling-MLP agent (N-step Double DQN + PER),
following Project Description §4 (Tasks 1–4). This file is the stage-level
dossier: what each stage trains, with what config, to what bar, and where
the evidence lives. Figure-level detail is in
`results/figures/captions.md`; eval results in `results/eval_summary.tex`.

Hardware: AMD RX 6900 XT via `torch_directml` (`privateuseone:0`) in WSL2;
inference stays CPU-only (tournament condition). Hyperparams:
`GAMMA=0.95`, `N_STEP=3`, `LR=1e-3`, `BATCH=256`, epsilon `1.0→0.05/50k`
(see `agent_code/sentinel/train.py`). Metrics append to
`agent_code/sentinel/runs/metrics.csv`; checkpoints to
`agent_code/sentinel/checkpoints/` (`last.pt` every round, `best.pt` on
EMA improvement, `ep_NNNNNN.pt` every 200 rounds, pruned to 5).

Resume any stage: re-run its command — `setup_training()` resumes
`last.pt` (optimizer + episode + epsilon). Override rounds:
`STAGE1_N=200 SENTINEL_DML=1 bash scripts/train_sentinel_curriculum.sh`
(or run single stages, see commands below).

---

## Stage 1 — navigation (Task 1) ✅ done

- **Goal:** efficient board navigation and coin greed, no bombs needed.
- **Config:** scenario `coin-heaven` (`CRATE_DENSITY=0`, `COIN_COUNT=50`,
  `settings.py`), solo, 500 rounds (`scripts/train_sentinel_curriculum.sh:13-14`).
- **Must learn:** BFS pathfinding features → movement, nearest-coin pursuit.
- **Success criteria:** ≥45 coins/round, 0 suicides (no bombs exist).
- **Outcome:** converged — 24,319 coins / 500 rounds (48.6/round),
  0 suicides, EMA reward ~52.7 (`results/sentinel_stage1.json`,
  metrics eps 1–500).
- **Artifacts:** `results/sentinel_stage1.json`, metrics eps 1–500,
  `checkpoints/ep_000400.pt` (ep_200 pruned).
- **Command:**
  `SENTINEL_DML=1 .venv/bin/python main.py play --no-gui --agents sentinel --train 1 --scenario coin-heaven --n-rounds 500 --save-stats results/sentinel_stage1.json`

## Stage 2 — bombs + escape (Task 2) ✅ done

- **Goal:** destroy crates, uncover coins, use bombs without dying.
- **Config:** scenario `classic` (0.75 density, 9 hidden coins), solo,
  1500 rounds (`train_sentinel_curriculum.sh:16-17`).
- **Must learn:** crate-adjacent bombing, escape-route planning,
  suicide avoidance (escape is "crucial", §4).
- **Success criteria:** suicide rate <0.3 and falling, coins/round
  recovering from ~0, EMA reward turning positive.
- **Outcome:** eps 501–2000; regime shock at ep 501 (reward +52→−23,
  buffer reset 98k→401, ~5 refill rounds with loss 0, epsilon already
  floored at 0.05); suicide rate spiked ~0.85 then decayed, coins
  recovering slowly (survival learned before profit — see
  `training_performance_split.png`).
- **Artifacts:** `results/sentinel_stage2.json`, metrics eps 501–2000.
- **Command:**
  `SENTINEL_DML=1 .venv/bin/python main.py play --no-gui --agents sentinel --train 1 --scenario classic --n-rounds 1500 --save-stats results/sentinel_stage2.json`

## Stage 3 — hunting (Task 3) ✅ done

- **Goal:** hunt and blow up `peaceful_agent` (easy, random, no bombs)
  then `coin_collector_agent` (hard, bombs for coins).
- **Config:** classic, `sentinel peaceful_agent coin_collector_agent`,
  1000 rounds (`train_sentinel_curriculum.sh:19-20`).
- **Must learn:** kill setups, opponent blast timing, risk/reward of
  hunting vs economy.
- **Success criteria:** kills/round >0 vs both opponents, EMA reward
  positive, suicide rate not regressing vs Stage 2 exit.
- **Outcome:** finished 2026-09-05 (`results/sentinel_stage3.json`,
  final 210 rounds): sentinel 751 score, 241 coins, **102 kills
  (0.49/round ✓)**, 112 suicides over 210 rounds vs
  coin_collector 959 / peaceful 6. Mid-stage the run was stopped and
  resumed from `last.pt` (≤1 round lost); resume verified
  (`device=privateuseone:0`, `optimizer=DMLAdam`, zero `aten::lerp`
  warnings). Kills criterion met; EMA still negative (economy lags
  combat — Stage 4 must fix coins while holding kills).

- **Goal:** hunt and blow up `peaceful_agent` (easy, random, no bombs)
  then `coin_collector_agent` (hard, bombs for coins).
- **Config:** classic, `sentinel peaceful_agent coin_collector_agent`,
  1000 rounds (`train_sentinel_curriculum.sh:19-20`).
- **Must learn:** kill setups, opponent blast timing, risk/reward of
  hunting vs economy.
- **Success criteria:** kills/round >0 vs both opponents, EMA reward
  positive, suicide rate not regressing vs Stage 2 exit.
- **Status:** run interrupted mid-round 2026-09-05 (ep 2333 checkpointed,
  ≤1 round lost); resumed same day from `last.pt` ep 2397 with the
  DML-safe optimizer (see note below) — completed same day
  (`logs/stage3_resume.log`, then final 210 in `logs/stage3_finish.log`).
  Resume verified: `device=privateuseone:0`, `optimizer=DMLAdam`,
  zero `aten::lerp` warnings.
- **Artifacts:** `results/sentinel_stage3.json` (written at stage end),
  metrics eps 2001+, `checkpoints/ep_001400–2200.pt`, `last.pt`.
- **Command:**
  `SENTINEL_DML=1 .venv/bin/python main.py play --no-gui --agents sentinel peaceful_agent coin_collector_agent --train 1 --scenario classic --n-rounds 1000 --save-stats results/sentinel_stage3.json`

### Note — DML-safe optimizer (report §5 material)

Stock `torch.optim.Adam` updates moments via `aten::lerp`, which has no
DirectML kernel: every optimizer step round-tripped tensors to CPU
(warning + PCIe stall). `dml_optimizer.py` provides `DMLAdam`/`DMLAdamW`,
mathematically identical Adam spelled with DML-native ops (`mul_`/
`add_`/`sqrt`/`div`); `build_optimizer_for_device` selects it on
`privateuse1` and stock Adam/AdamW on CPU/CUDA (tournament/Docker/ROCm
untouched). Resume across the switch is safe: `last.pt` holds stock-Adam
state (Tensor steps, extra group keys), which `DMLAdam` loads — verified
by a 6-round smoke run (updates flow, 0 lerp warnings). Stopping uses
SIGINT (SIGTERM is ignored by the running loop); loss is always ≤1 round
(light checkpoints don't persist the replay buffer).

### Note — optimizer selection (benchmarked 2026-09-05)

Profiled one 256-batch MLP update on DirectML: transfer 0.26 ms (9%),
forward 0.56 ms (19%), backward 0.62 ms (21%), **optimizer step 1.47 ms
(51%)** — Python per-tensor dispatch overhead dominates on this tiny net,
so the optimizer step itself is the bottleneck (not PCIe).

Candidates in `dml_optimizer.py`: `DMLAdam` (status quo), tuned
Adam (`eps=1e-4`, decay `1e-4`) + `Lookahead(k=5, α=0.5)`, `DMLLion`
(1 state/param, sign updates). LR schedule = pure function of
`total_steps` (5k warmup + cosine to 0.1×), checkpoint-safe by design.
Gates: `scripts/check_optimizer.py` (Adam CPU-parity 3.7e-09,
Lion determinism exact, zero CPU-fallback warnings).

Results (`scripts/bench_optim.py`): **Lion wins both criteria** —
1.57× faster step than Adam (0.91 vs 1.43 ms; tuned+Lookahead 1.13×),
and best valley-proxy convergence (final 29.9 vs Adam 30.6,
tuned+Lookahead 67.4 — Lookahead's slow weights cost early progress).
Caveat: Lion LR 1e-3 (same as Adam here, not the paper's 3–10× smaller;
swept 1e-4/3e-4/1e-3); valley proxy ≠ bootstrapped TD, so live
validation at Stage 4 adoption is required.

Adoption: `SENTINEL_OPT=lion SENTINEL_SCHEDULE=1` from Stage 4 on
(`train.py` factory flags; defaults keep Adam/no-schedule). Resume from
an Adam `last.pt` starts fresh Lion momentum with a warning (Q-weights
still transfer). Same factory serves overlord later.

## Stage 4 — full combat (Task 4) ⏳ pending

- **Goal:** hold own against full-strength opposition; beating
  `rule_based_agent` is the gate for tournament contention (§4).
- **Config:** classic, `sentinel + 3× rule_based_agent`, 2000 rounds
  (`train_sentinel_curriculum.sh:22-23`), then frozen eval:
  same matchup, `--train 0`, 100 rounds
  (`train_sentinel_curriculum.sh:25-26`).
- **Must learn:** combat under pressure, kill stealing/avoidance,
  endgame hunting when crates+coins run out.
- **Success criteria:** frozen-eval score/round beats best rule_based
  agent; suicide rate ≤ Stage 3 exit; think time <500 ms/step.
- **Artifacts (future):** `results/sentinel_stage4.json`,
  `results/sentinel_eval.json`, final `my-saved-model.pt`.
- **Commands:** see `train_sentinel_curriculum.sh:22-26`.

---

## Eval matrix (frozen model, `train=0`)

Separate from training: `scripts/run_sentinel_eval.sh` runs matchups
M1–M8 (solo → vs rule_based → vs overlord/warden) at 40 rounds ×
2 seeds; `scripts/aggregate_eval.py` + `scripts/plot_eval.py` produce
`eval_summary.*` and `results/figures/`. Deferred run queued in
`scripts/eval_after_stage2.sh` (starts when Stage 2 exits).

---

## Shared training kit + overlord migration (2026-09-05)

GPU-utilization work was applied to **all** training scripts, not just
sentinel (`dml_trainkit.py`, repo root):
- `update_config(prefix)` — `{SENTINEL,OVERLORD}_{UTD,BATCH,EOR_UPDATES}`
  env flags; all defaults reproduce legacy behavior exactly
  (sentinel 1/256/4, overlord 1/512/6).
- `DutyTimer` — in-code GPU-busy vs wall-time duty cycle (WSL cannot read
  AMD counters); logged as `gpu_ms/wall_ms/duty_pct` columns appended to
  `metrics.csv` (header auto-widens; old rows backfill empty).
- `get_dml_device`, `CheckpointStore`, `log_metrics_row`,
  `apply_schedule` — shared by both agents.
- Overlord migrated off the dead ROCm/`torch.cuda` probe (which always
  fell back to CPU here) to DirectML + `dml_optimizer` factory + resume
  + metrics + tournament export. Smoke-verified: DML device, `DMLAdamW`,
  gradient flow (loss > 0 once buffer ≥ 5000), **zero CPU-fallback
  warnings (BatchNorm2d included)**, resume ep 15→40→75.
- Sentinel train loop now runs UTD updates/step and EOR updates/round
  from config; schedule applied via shared helper.
- Launch for both agents at Stage 4: `SENTINEL_UTD=3 SENTINEL_BATCH=512
  SENTINEL_EOR_UPDATES=16` (+ Lion flags) for sentinel; overlord reuses
  the same `OVERLORD_*` flags from its first stage.
- Known gap (not changed): overlord `act()` has no epsilon-greedy
  exploration (sentinel does) — flagged for its first training stage.

---

## Reaper curriculum (third model, distilled feature-MLP — see E33)

Goal: beat the overlord ship (frozen 3.79) at a fraction of the training
cost, stretch-goal surpass warden (5.35 reference). Design thesis:
engineered features + small MLP + teacher distillation + short RL
fine-tune. All artifacts probe-gated before any game is played.
E37 reengineering (Phases 1-5): kill-centric features, objective-exact
reward, parallel training, tactical inference overlay — see E37.

### Architecture (agent_code/reaper/)

- **Features** (`features.py`): 98-dim vector = sentinel's 46-dim base
  extended with good-bomb-spot BFS, safe-move mask, own-bomb state,
  opponent model, PLUS (E37/P2) per-direction kill table 68-83 (trap
  mask, opps_hit, crates_hit, escape margin — the net finally knows
  WHERE to step for a kill), own mobility 84-87 (free neighbours,
  reachable area, dead-end, junction dist), richer opponent model 88-92
  (incl. their `bombs_left`, which the engine supplies but reaper used
  to ignore), kill/coin/mobility potentials 93-95, board state 96-97.
  f[50] repurposed from the mask-invariant `can_escape_if_bomb` (dead
  constant-1) to the bomb-here escape margin. Probe-verified equivariant
  under the 8 board symmetries (9/9 groups in
  `scripts/probe_reaper_features.py`, incl. brute-force parity for the
  vectorized blast tables and constructed-trap direction checks).
- **Model**: dueling MLP 512-256-256 (~234K params), zero-init heads,
  CPU forward <0.5 ms. Env: `REAPER_HID1/2/3`.
- **Safety**: overlord's mask, semantics frozen (parity-probed exact),
  internals rewritten for latency: flat-buffer time-expanded BFS +
  O(1) first-lethal map (`first_lethal`), shared `danger_no_explosion`
  across trap checks, `REAPER_HORIZON` knob (default 8 = parity).
  Helpers: `opp_can_escape` (+ shared-`danger` fast path),
  `bomb_here_traps` (exact forced-kill proof for the tactical overlay).
- **Act policy**: learned Q (clip ±50, not ±6) + bounded soft-prior
  heuristic (`REAPER_HEUR_WEIGHT` 0.5, `REAPER_HEUR_MAX` 2.0 — never
  outvotes the net) + trap-move nudge (`REAPER_TRAP_BONUS` 1.5) + strict
  mask + wall-clock budget guard (`REAPER_TIME_BUDGET` 0.12 s, graceful
  heuristic-only degradation) + optional tactical overlay
  (`REAPER_SEARCH=tactical`: exact guaranteed-kill BOMB override,
  least-bad fallback); ε-greedy during training over the safety-masked
  pool. Measured: p50 ~1.9 ms, p99.9 ~5 ms on a pinned core (limit 500).

### Pipeline (scripts/)

1. **Demos** (`collect_demos.sh`): `reaper_teacher` records warden (400)
   / sentinel (200) / overlord (200) in the GATE fields (50% 3×rb, 25%
   warden-mix, 12.5% random, 12.5% collector — not weak lineups), plus
   optional DAgger (`STAGE_DAGGER_N`, `REAPER_DAGGER=1`: student acts,
   teacher labels) into `results/demos/<teacher>[_<field>|_dagger]/`.
   Recorder is resume-safe (appends round IDs).
2. **BC pretrain** (`pretrain_reaper.py`): cross-entropy from Q/τ
   softmax to demo actions, 8× dihedral augmentation, ~5 epochs →
   `my-saved-model.pt` + resumable `checkpoints/bc_last.pt` payload.
   Smoke on 30k warden samples: val_acc 0.65 after 50 steps (vs 0.41
   before the feature redesign).
3. **RL fine-tune** (`train_reaper.sh`, resumable via `STAGE_*_N=0`,
   run-isolated via `REAPER_RUN_DIR`/`REAPER_TAG`/`SKIP_GATES`):
   C1 150 solo classic → C2 200 vs peaceful+collector → C3 500 vs
   rb+warden+sentinel → C4 150 vs 3×rb. N=8, γ=0.99, PER (numpy ring,
   no replace=False bottleneck) + Huber δ=1, clip 1, tuned Adam
   (eps/decay 1e-4), ε re-warm 0.10, 25% demo-replay mix, 8× symmetry
   augmentation, reward = exact engine score (+1/+5, deaths −8/−6,
   invalid −0.6, wait −0.05, survived +1) + potential shaping
   γΦ(s′)−Φ(s) over (PHI_kill, coin closeness, mobility), logged
   separately as `rew_engine`/`rew_shaping`. Fixed along the way:
   duplicate terminal push for survivors, 2× feature recompute (cache),
   `REAPER_DEVICE=cpu` silently ignored (wrong key passed to the kit).
4. **Sweep** (`sweep_reaper.sh`): 8 parallel curriculum jobs (base ×2
   seeds, kill-heavy shaping, low heuristic weight, low LR, short/long
   horizon, no-shaping ablation) on isolated run dirs sharing the GPU.
5. **Frozen gates** (`bakeoff_reaper.sh` per candidate; `eval_reaper.sh`
   for the ship): rb 100×2, random 40×2, warden-mix 60×2, collector
   40×2 (sentinel-mix excluded per protocol), Q_WEIGHT=0 ablation
   (ML-compliance: learned Q must add ≥0.5).

### Env flags

`REAPER_DEVICE/_AMP/_UTD/_BATCH/_EOR_UPDATES/_OPT/_LR/_SCHEDULE/_TUNED/
_SAVE_EVERY/_EPS_START/_EPS_DECAY/_Q_WEIGHT/_DEMOS/_DEMO_RATIO/_BC_W/
_HID1-3/_GAMMA/_N_STEP/_W_PHI_KILL/_W_PHI_COIN/_W_PHI_MOB/_Q_CLIP/
_HEUR_WEIGHT/_HEUR_MAX/_G_FLEE(_LOCK)/_TRAP_BONUS/_TIME_BUDGET/_HORIZON/
_SEARCH/_SEED/_RUN_DIR/_TAG/_BC_INIT`; teacher recorder:
`TEACHER=warden|sentinel|overlord`, `REAPER_DEMO_DIR`,
`REAPER_DAGGER`, `REAPER_STUDENT_PT`.

### Ship protocol

Argmax-frozen bake-off (E30 rule — EMA never ships), archive-on-decision
(`results/archive/reaper_*`), ship bar = pooled > 3.79; the tournament
zip contains only `agent_code/reaper/` (demos/checkpoints/runs are
git-ignored and must stay out; `my-saved-model.pt` is explicitly
included).

---

## Arbiter — no curriculum (pointer, not a dossier)

Arbiter has no curriculum stages (no online RL was ever run — the P2
premise was falsified, E66–E67), so there is no stage table to keep.
The training that exists, with configs, bars, and evidence locations:

- **P0 offline warm start (E65):** `scripts/arbiter_extract.py` builds
  `results/arbiter_p0_cache.npz` (339,826 pi + 123,212 V rows from
  `results/{apex_demos,demos}/`, see `docs/demo_manifest.md`) →
  `scripts/pretrain_arbiter.py` (CE + margin regression, 8× dihedral
  aug, Adam 1e-3, batch 1024, 5 epochs, CUDA) → gate `val_acc >= 0.5`
  (got 0.757) → `agent_code/arbiter/my-saved-model.pt` (+ `.meta.json`).
  Env: `ARBITER_PI_TEACHERS` (default {warden,sentinel,overlord}),
  `ARBITER_V_TEACHERS` (+collector), `ARBITER_V_W`, `--epochs/--batch/--lr/--seed`.
- **P0-E88 corrected-feature variant (E88):** after the escape-solver
  fix, re-extracting the apex corpus with corrected features plus 212
  fresh self-demos (`--reaper-include=arbiter_self_e88`; 178,278 pi /
  123,212 V rows) gives candidate val_acc 0.710 (E80 0.757). The
  candidate does NOT beat the frozen E80 prior on the corrected solver
  (G1 4.590 vs 4.775), so the ship keeps the E80 weights; the clean
  cache + candidate remain as `results/arbiter_e88_cache.npz` /
  `results/arbiter_e88_candidate.pt` (report ablation).
- **P1 search (E66, inference-only):** no training; `sim.py` + `search.py`
  wired as `ARBITER_SEARCH=search` (budget 0.30 s). Ship defaults:
  score margin 0.6, `ARBITER_CRN=1` (E88). Knobs:
  `ARBITER_SEARCH_H/K/R/PLANS`, `ARBITER_V_BLEND`,
  `ARBITER_BOMB_SCORE_MARGIN` (legacy alias `ARBITER_BOMB_MARGIN`; E88),
  `ARBITER_BOMB_ESC_MARGIN` (safety mask, E87/E88),
  `ARBITER_PLANT_ESC` (search bomb gate, E88), `ARBITER_CRN` (paired
  rollout draws, E88), `ARBITER_FLEE_Q` (flee re-ranking, E90 REJECTED,
  default 0),
  `ARBITER_W_{CRATE,COIN_TILE,OPP_TILE,DEATH}`, `ARBITER_ESC_DIST`,
  `ARBITER_TRAP_HARD/_P`, `ARBITER_{PI,V}_OFF` (ablations).
- **Gates:** G1 rb 100×2 **4.775 ship (E88; E80 4.345, E66 3.95)** ·
  G2 warden-mix 60×2 (3.67) · G3 collectors 40×2 (2.85) · G4 random
  40×2 (6.13) · S0 / V0 / pi0 ablations; tables in
  `results/arbiter_summary.csv` (`scripts/aggregate_arbiter.py`),
  figures in `results/figures/` (`scripts/plot_arbiter.py`).
- **Tournament gates (E88):** field-proxy 4-seed means all above E80
  (4.94/5.29/8.54/5.52) and per-round win-rate via
  `scripts/tournament_eval.py` (G1 0.41 vs 0.32; STRONG 0.40 vs 0.37).
- **Phase-2 nulls (E90–E92):** post-E88 flee-quality (`ARBITER_FLEE_Q`),
  hunt/trap/`CERT_OWN` re-tests, and capacity arms (R=5, H=8, K=16,
  PLANS=96, MOVE_SEEDS=2) were all rejected — G1 40-round screens do
  not replicate at 100×2 and de-aggression levers cost score; no ship
  change. Robustness guards (long-horizon + mask-survival fuzz, E91)
  landed in `probe_arbiter_sim.py`.
- **Ship protocol:** same E30 rule; current ship with zero-env defaults
  (`SEARCH=search`, score margin 0.6, `CRN=1`, `ESC_DIST=3.0`).
  Zip: `agent_code/arbiter/` only (E68 audit).
