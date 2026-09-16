# BomberMan RL

Train Reinforcement Learning agents for the classic game Bomberman (course project setup).
Tournament inference is CPU-only with a 0.5 s/step budget; training defaults to
CUDA with AMP (Google Colab ready).

Current ship (E130): **arbiter_ng (E108: D1-BC + rl225 + wardenlite)**
with the E112 solo/endgame fix, the E119 inference fast path, the
E122/E123 gap fixes (certified coin-take d=3, solo bomb radius 8),
E125 (committed solo bomb-approach — the verified K-cap membership
flip freeze is gone: solo 8.94/9, tail 0/100), and **E130: the
opening-flicker fix** (ARBITER_COMMIT_OPP=1 + ARBITER_BACKTRACK_OPP=0.5:
the opponent-ful opening reversals 27-32% of move-ticks drop ~40%,
early bombs 7.6 -> 8.4/rd, composed canonical gate **g1 +0.64 / +6.5%
round-win**, pooled +0.085 within 1 SE). E125 hysteresis arm, E129
PLANT_OPP and the E124 opening margins all rejected at their gates.
See `docs/experiments.md` E125/E128/E129/E130 and
`results/gaps_findings.md`.
Legacy ship notes (E111, pre-E112): the same weights (E108) —
CNN+scalar fused policy (3566-dim lossless board tensor + 98 scalars)
over the exact-dynamics lookahead search; BC warm-start on the widened
league corpus (1586 files incl. ship self-play mirror + warden_v1;
val 0.806), then RL fine-tuned (E104 recipe: KL-anchored REINFORCE,
adaptive anchor + revert guard, bomb-trace, vs rule_based x2 +
warden_v2), rollout opponent model `wardenlite`. E108 battery (G1
100x2 + STRONG 40x10 + UNSEEN, 1120 rounds): **pooled 6.650 / win
0.681** vs E107 ship 6.388/0.640; **beats warden_v2 head-to-head
5.853 vs 5.065 (win 55%)** in the same lobby (E88 was −0.73 behind).
Submission-test simulation + probe gates passed (E111).
Legacy: E88 arbiter preserved as `__shared/arbiter_ship_e88.zip`.
Backup: **overlord** (CNN, 3.79). Report models: **sentinel** (MLP
Dueling-DQN curriculum), **reaper** (distilled feature-MLP), **apex**
(synthesis CNN — unshipped: learned Q net-negative, see
`docs/experiments.md` E61).

Rule-based / scripted opponents (`rule_based_agent`, `coin_collector_agent`,
`peaceful_agent`, `random_agent`) are included for curriculum training and eval,
plus the outsider sparring agents `warden_v1` (frozen heuristic reference)
and `warden_v2` (active: v1 + corrected escape solver, 8-step danger
horizon, deterministic RNG, probe-gated), and the **unseen-behavior
sparring suite** `outsiders/unseen_{coward,bomber,rusher,racer}` (eval-only
held-out proxies, E106: never shipped, never used as teachers).

## Requirements

- Python >= 3.12
- PyTorch with CUDA for training (`torch>=2.5.1,<2.6`, `torchvision>=0.20.1,<0.21`);
  CPU-only works for inference, eval, and (slow) training.
- The rest: `numpy pygame scikit-learn scipy tqdm matplotlib tensorboard`
  (see `pyproject.toml`). No new libraries for arbiter (torch + numpy only at
  inference; brief §2 compliant).

## Setup

Local (uv):

```bash
uv sync
uv run python main.py play --no-gui --agents arbiter rule_based_agent --n-rounds 1
```

Google Colab (fresh GPU runtime):

```bash
git clone https://github.com/Alii-Khaled/BomberMan && cd BomberMan
pip install torch torchvision
pip install pygame scikit-learn scipy tqdm matplotlib tensorboard
python3 test_gpu.py   # expect a GPU name + PASS (active device: cuda)
```

Arbiter P0 weights (`agent_code/arbiter/my-saved-model.pt`, 639 KB) are
committed — no extra files needed. To resume **overlord** training on Colab,
copy two checkpoint files from Google Drive into the repo (they are
git-ignored, ~10 MB each):

- `agent_code/overlord/checkpoints/last.pt`
- `agent_code/overlord/checkpoints/best.pt`

Sentinel resumes from `agent_code/sentinel/checkpoints/best.pt`, which *is*
committed — no extra files needed.

Demo corpora (`results/apex_demos/` 500 rounds; `results/demos/` 1,312 npz
incl. the E80/E86/E88 arbiter self-distillation sets) are training-time
only, git-ignored; byte-identical mirrors live outside the repo (see
`docs/demo_manifest.md` for fingerprints + restore procedure).

## Training

Arbiter (offline warm start, then frozen gates — no curriculum training):

```bash
bash scripts/collect_apex_demos.sh     # teacher demos (warden_v2/sentinel/overlord/collector)
bash scripts/collect_demos.sh          # reaper-format demos (98-dim features)
python3 scripts/arbiter_extract.py     # joint pi/V cache (results/arbiter_p0_cache.npz)
python3 scripts/pretrain_arbiter.py    # CE + margin regression, gate val_acc >= 0.5
```

E88 corrected-feature refresh (after the escape-solver fix; candidate-only —
the ship weights remain the E80 export, see E88 in `docs/experiments.md`):

```bash
DEMO_PREFIX=arbiter_self_e88 DAGGER_N=200 ARBITER_BOMB_SCORE_MARGIN=0.6 \
  bash scripts/collect_arbiter_self.sh
python3 scripts/arbiter_extract.py --out results/arbiter_e88_cache.npz \
  --reaper-include=arbiter_self_e88
ARBITER_PI_TEACHERS="0 1 2 4" python3 scripts/pretrain_arbiter.py \
  --cache results/arbiter_e88_cache.npz --out results/arbiter_e88_candidate.pt
```

Legacy curricula:

```bash
bash scripts/train_sentinel_curriculum.sh   # Tasks 1-4: coin-heaven -> classic solo -> hunt -> vs rule_based
bash scripts/train_overlord_curriculum.sh   # O1-O4, same ladder for the CNN agent
```

Single run example:

```bash
python3 main.py play --no-gui --agents sentinel --train 1 \
  --scenario classic --n-rounds 1500 --save-stats results/sentinel_stage2.json
```

`--train N` puts the first N agents in training mode; `--train 0` with
`--continue-without-training` runs a frozen eval. Checkpoints land in
`agent_code/<agent>/checkpoints/` (`last.pt` every round, `best.pt` on EMA
improvement, `ep_NNNNNN.pt` snapshots); per-round metrics append to
`agent_code/<agent>/runs/metrics.csv`; tournament weights export to
`agent_code/<agent>/my-saved-model.pt` (CPU state dict).

### Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `ARBITER_SEARCH` | `search` (= ship) | `off` = pi-only fallback (S0); `tactical` = proven-kill overlay |
| `ARBITER_TIME_BUDGET` | `0.30` | wall-clock search budget per step (0.5 s tournament limit) |
| `ARBITER_V_BLEND` | `1.0` | leaf-value weight (V null per E66 — 0 also ships) |
| `ARBITER_BOMB_MARGIN` | `0.6` | bomb plan must beat best move by this to execute (legacy alias for `ARBITER_BOMB_SCORE_MARGIN`) |
| `ARBITER_BOMB_SCORE_MARGIN` | `0.6` | E88 de-conflicted search bomb-vs-move score margin (was 0.2) |
| `ARBITER_BOMB_ESC_MARGIN` | `1` | E87/E88 post-plant first-step escape-direction count required for the mask's BOMB certificate |
| `ARBITER_SOLO_MARGIN` | `0.15` | E112 ship: bomb-vs-move score margin while no opponent is alive (`<0` restores the pre-E112 BOMB_MARGIN behavior) |
| `ARBITER_LOOP_ESC` | `2` | E112 ship: loop tie-break mode (`0` = bounded LOOP3/LOOP2, `1` = escalate everywhere, `2` = escalate solo-only) |
| `ARBITER_LOOP_ESC_STEP` / `_CAP` | `0.5` / `3.0` | escalation slope / cap for `ARBITER_LOOP_ESC` |
| `ARBITER_SOLO_TREK` | `0` | E112 rejected arm: solo BFS trek move plans (documented, default off) |
| `ARBITER_SOLO_RADIUS` | `8` | E123 ship: bomb-tile candidate BFS radius while no opponent is alive (`0` = ship RADIUS 4; 12 rejected) |
| `ARBITER_COINTAKE` | `1` | E122 ship: certified coin-take overlay (exact +1, first step of the shortest mask-safe path to a visible coin) |
| `ARBITER_COINTAKE_D` | `3` | E122 coin-take reach in BFS steps (sweep 1/2/3 picked 3) |
| `ARBITER_SOLO_COMMIT` | `6` | E125 ship: committed bomb-approach while solo (fixes the verified position-dependent K-cap membership flip that ping-ponged the endgame; `0` restores pre-E125) |
| `ARBITER_SOLO_COMMIT_MAX` | `6` | E125 max committed-approach age in ticks |
| `ARBITER_BOMB_HYST` | `0` | E125 rejected arm: sticky-target arbitration bonus (G1 screens -0.89; off) |
| `ARBITER_BACKTRACK` | `0` | E123 rejected arm: solo immediate-reversal penalty (binds, no score lift; off) |
| `ARBITER_COMMIT_OPP` | `1` | E130 ship: committed bomb-approach in opponent-ful play (opening reversals -40%, early bombs 7.6 -> 8.4/rd; `0` = solo-only commit) |
| `ARBITER_BACKTRACK_OPP` | `0.5` | E130 ship: opponent-ful immediate-reversal penalty (composed gate g1 +0.64; `0` restores) |
| `ARBITER_OPEN_MARGIN` | `-1` | E124 rejected arm: opening phase bomb margin (no visible coins, step<`ARBITER_OPEN_T`; off) |
| `ARBITER_OPEN_T` | `100` | E124 opening window length (steps) |
| `ARBITER_HUNT_OPEN` | `0` | E124 rejected arm: opening-only pursuit plans (E82 lesson reconfirmed; off) |
| `ARBITER_ANTIPIN` (+ `_D`/`_MOB`/`_DEADEND`) | `0` | E114 rejected arm: armed-enemy pocket guard (default off; 40x4 STRONG lift did not survive 40x10) |
| `ARBITER_KILL_P` | `1.0` | E113 rejected sweep: certified-kill credit scale (0.5/0.25 both below control) |
| `ARBITER_ESC_DIST` | `3.0` | proven-escape distance gate for bomb tiles |
| `ARBITER_SEEDS` | `1` | legacy E71 knob (post-E78 no-op: moves use `ARBITER_MOVE_SEEDS`, bombs 1; logged in search debug only) |
| `ARBITER_CRN` | `1` | common random numbers: all plans share per-tick opponent draws (E88 ship; `0` restores unpaired) |
| `ARBITER_PLANT_ESC` | `1` | min post-plant escape directions for the search bomb gate (E88; clean E87 redo) |
| `ARBITER_PI_OFF` / `ARBITER_V_OFF` | `0` | `1` forces uniform prior / zero value (ablations) |
| `ARBITER_MODEL` | unset | eval-only candidate weights path (gating; tournament default = `my-saved-model.pt`) |
| `ARBITER_FLEE_Q` | `0` | `1` = post-plant flee moves ranked by open space/opponent distance (E90; rejected, ablation only) |
| `SENTINEL_DEVICE` / `OVERLORD_DEVICE` | `auto` (CUDA if available, else CPU) | `cpu` forces CPU (tournament condition) |
| `SENTINEL_AMP` / `OVERLORD_AMP` | `1` | `0` disables AMP autocast + GradScaler (fp32) |
| `SENTINEL_OPT` / `OVERLORD_OPT` | `adam` | `lion` selects the Lion optimizer |
| `SENTINEL_LR` / `OVERLORD_LR` | agent default | base LR override |
| `SENTINEL_SCHEDULE` / `OVERLORD_SCHEDULE` | `0` | `1` enables warmup + cosine LR schedule |
| `SENTINEL_TUNED` / `OVERLORD_TUNED` | `0` | `1` applies the DQN preset (Adam eps/decay 1e-4) |
| `SENTINEL_UTD` / `OVERLORD_UTD` | `1` | gradient updates per env step |
| `SENTINEL_BATCH` / `OVERLORD_BATCH` | `0` (= agent default: 256 / 512) | batch override |
| `SENTINEL_EOR_UPDATES` / `OVERLORD_EOR_UPDATES` | `4` / `6` | extra updates at round end |

Legacy `SENTINEL_DML` / `OVERLORD_DML` are still honored (`0` = force CPU)
but no longer select any backend — device choice is CUDA-or-CPU only.

## Evaluation

Arbiter frozen gates (all CPU, `--train 0 --continue-without-training`):

```bash
# G1 rb 100x2 / G2 warden-mix 60x2 / G3 collectors 40x2 / G4 random 40x2
# S0 pi-only / V0 zero-value / pi0 uniform-prior ablations (see E65-E66)
python3 scripts/aggregate_arbiter.py   # results/arbiter_summary.csv (pooled tables)
python3 scripts/plot_arbiter.py        # results/figures/arbiter_*.png + captions

# per-round win rate / mean rank (tournament objective, E88/P1.3):
python3 scripts/tournament_eval.py --agents arbiter rule_based_agent \
  rule_based_agent rule_based_agent --n-rounds 40 --seed 0
python3 scripts/diag_arbiter_deaths.py  # death/missed-kill attribution from ARBITER_DIAG jsonl
```

Legacy (sentinel matrix):

```bash
bash scripts/run_sentinel_eval.sh            # frozen matrix M1-M8, 40 rounds x 2 seeds
python3 scripts/aggregate_eval.py            # tables
python3 scripts/plot_eval.py                 # figures
```

Ship rule (E30/E106): nothing ships without the pooled multi-battery
(G1 100x2 + STRONG 40x10 + UNSEEN 1120 rounds) vs a fresh same-session
control. Current standing (E130): **arbiter_ng+E130 7.116/0.641 vs
same-session control 7.031/0.652 (pooled +0.085 within 1 SE; the target
class g1 +0.64 / +6.5% round-win, strong parity, umix+archetypes
bit-identical)**. E128 death attribution: the corner-pin class (68%,
unchanged) is the binding loss source; E129 PLANT_OPP and the E124
opening margins are rejected. Full ledger: `docs/experiments.md`.

## Health checks

```bash
python3 test_gpu.py                 # CUDA probe + matmul bench + AMP training step
python3 test_cuda.py                # sentinel MLP fwd/bwd + checkpoint round-trip
python3 scripts/check_optimizer.py  # optimizer parity / determinism / factory gates
python3 scripts/verify_dml_optimizer.py  # full CUDA training-path verification
python3 test.py                     # 1-round game smoke test
python3 scripts/probe_arbiter.py    # arbiter static gates (shapes, parity, escape correctness, latency)
python3 scripts/probe_arbiter_sim.py  # sim-vs-engine parity + long-horizon/mask-survival fuzz (E90)
python3 scripts/probe_arbiter_crn.py  # CRN rollout-seed wiring + determinism (E88)
python3 scripts/probe_arbiter_flee.py  # flee-quality ranking wiring (E90 ablation, default off)
python3 scripts/probe_reaper_features.py  # reaper feature/escape parity gates (9/9)
```

## Repo layout

- `main.py`, `environment.py`, `settings.py`, `agents.py` — game engine + runner
- `agent_code/arbiter/` — **ship**: policy/value net (`model.py`), 98-dim
  features + safety mask (vendored, probe-verified), exact simulator
  (`sim.py`), bounded search (`search.py`), S0 policy (`callbacks.py`)
- `agent_code/sentinel/` — MLP agent (`callbacks.py`, `train.py`, `model.py`,
  `features_mlp.py`, `safety.py`, `device.py`, `checkpointing.py`)
- `agent_code/overlord/` — CNN agent, backup ship (same structure + `features_cnn.py`)
- `agent_code/reaper/`, `agent_code/apex/` — report models (distilled MLP, synthesis CNN)
- `agent_code/solo_dagger/` — training-only solo-DAgger recorder (E115, rejected arm; never ships)
- `agent_code/{rule_based,coin_collector,peaceful,random}_agent/` — scripted opponents
- `dml_trainkit.py` — shared kit: update config, CUDA device picker, duty timer,
  checkpoint store, metrics logging, AMP flag
- `dml_optimizer.py` — optimizer factory (stock Adam/AdamW on CUDA/CPU, Lion;
  legacy DML-safe variants kept for old-checkpoint resume) + LR schedule
- `scripts/` — arbiter pipeline (`collect_apex_demos`, `arbiter_extract`,
  `pretrain_arbiter`, `probe_arbiter[_sim|_crn]`, `aggregate/plot_arbiter`),
  win-rate harness (`tournament_eval`), death attribution
  (`diag_arbiter_deaths`), curricula, eval matrix, optimizer benchmarks
  and gates
- `docs/` — `training_stages.md` (stage dossier), `experiments.md` (log),
  `demo_manifest.md` (corpus fingerprint + restore)
- `results/`, `logs/`, `agent_code/*/runs/`, heavy checkpoints, demo corpora —
  local only, git-ignored (see `.gitignore`); mirror + manifest in `docs/`
