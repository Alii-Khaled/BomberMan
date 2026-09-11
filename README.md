# BomberMan RL

Train Reinforcement Learning agents for the classic game Bomberman (course project setup).
Tournament inference is CPU-only with a 0.5 s/step budget; training defaults to
CUDA with AMP (Google Colab ready).

Current ship: **arbiter** — learned policy prior over 98-dim engineered
features + exact-dynamics lookahead search for bomb placement
(G1 vs 3× rule_based **4.775 pooled**, 100 rounds × 2 seeds, E88:
corrected escape solver + score margin 0.6 + paired (CRN) opponent
rollouts, E80 weights unchanged — field-proxy 4-seed means all above
the E80 baseline: STRONG 4.94 / RACER 5.29 / WEAK 8.54 / TRAINED 5.52
(E80: 4.54/4.62/7.45/4.54); win-rate G1 0.41 vs 0.32, STRONG 0.40 vs
0.37 over 5 seeds).
Backup ship: **overlord** (CNN agent, 3.79 pooled). Report models:
**sentinel** (MLP Dueling-DQN curriculum), **reaper** (distilled feature-MLP),
**apex** (synthesis CNN — unshipped: learned Q net-negative, see `docs/experiments.md` E61).

Rule-based / scripted opponents (`rule_based_agent`, `coin_collector_agent`,
`peaceful_agent`, `random_agent`) are included for curriculum training and eval,
plus the outsider sparring agents `warden_v1` (frozen heuristic reference)
and `warden_v2` (active: v1 + corrected escape solver, 8-step danger
horizon, deterministic RNG, probe-gated; paired 100x2 G1 vs v1 at parity
(4.40 vs 4.26), paired STRONG 40x5 sample above arbiter on score/rank —
directional, see E94 in `docs/experiments.md`).

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

Ship rule (E30): nothing ships without ≥100 rounds × 2 seeds; bar = pooled
score/round vs 3× rule_based above the incumbent ship. Current standing:
arbiter **4.775** (E88) > E80 arbiter 4.345 > overlord 3.79.
Field-proxy + win-rate gates (E88) also on file. Full ledger:
`docs/experiments.md`.

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
