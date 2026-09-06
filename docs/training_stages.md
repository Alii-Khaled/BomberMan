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
