# BomberMan RL

Train Reinforcement Learning agents for the classic game Bomberman (course project setup).
Tournament inference is CPU-only with a 0.5 s/step budget; training defaults to
CUDA with AMP (Google Colab ready).

Current ship: **arbiter** — learned policy prior over 98-dim engineered
features + exact-dynamics lookahead search for bomb placement
(G1 vs 3× rule_based **3.95 pooled**, 100 rounds × 2 seeds).
Backup ship: **overlord** (CNN agent, 3.79 pooled). Report models:
**sentinel** (MLP Dueling-DQN curriculum), **reaper** (distilled feature-MLP),
**apex** (synthesis CNN — unshipped: learned Q net-negative, see `docs/experiments.md` E61).

Rule-based / scripted opponents (`rule_based_agent`, `coin_collector_agent`,
`peaceful_agent`, `random_agent`) are included for curriculum training and eval,
plus the outsider sparring agent `warden_v1` (strong heuristic reference, 5.07).

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

Demo corpora (`results/apex_demos/` 500 rounds, `results/demos/` 800 rounds)
are training-time only, git-ignored; a byte-identical mirror lives outside the
repo (see `docs/demo_manifest.md` for fingerprint + restore procedure).

## Training

Arbiter (offline warm start, then frozen gates — no curriculum training):

```bash
bash scripts/collect_apex_demos.sh     # teacher demos (warden/sentinel/overlord/collector)
bash scripts/collect_demos.sh          # reaper-format demos (98-dim features)
python3 scripts/arbiter_extract.py     # joint pi/V cache (results/arbiter_p0_cache.npz)
python3 scripts/pretrain_arbiter.py    # CE + margin regression, gate val_acc >= 0.5
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
| `ARBITER_BOMB_MARGIN` | `0.2` | bomb plan must beat best move by this to execute |
| `ARBITER_ESC_DIST` | `3.0` | proven-escape distance gate for bomb tiles |
| `ARBITER_SEEDS` | `1` | opponent-policy rollouts averaged per plan (E71; >1 needs more budget) |
| `ARBITER_PI_OFF` / `ARBITER_V_OFF` | `0` | `1` forces uniform prior / zero value (ablations) |
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
```

Legacy (sentinel matrix):

```bash
bash scripts/run_sentinel_eval.sh            # frozen matrix M1-M8, 40 rounds x 2 seeds
python3 scripts/aggregate_eval.py            # tables
python3 scripts/plot_eval.py                 # figures
```

Ship rule (E30): nothing ships without ≥100 rounds × 2 seeds; bar = pooled
score/round vs 3× rule_based above the incumbent ship. Current standing:
arbiter 3.95 > overlord 3.79 (E66–E69). Full ledger: `docs/experiments.md`.

## Health checks

```bash
python3 test_gpu.py                 # CUDA probe + matmul bench + AMP training step
python3 test_cuda.py                # sentinel MLP fwd/bwd + checkpoint round-trip
python3 scripts/check_optimizer.py  # optimizer parity / determinism / factory gates
python3 scripts/verify_dml_optimizer.py  # full CUDA training-path verification
python3 test.py                     # 1-round game smoke test
python3 scripts/probe_arbiter.py    # arbiter static gates (17/17: shapes, parity, latency)
python3 scripts/probe_arbiter_sim.py  # sim-vs-engine parity (blast/step/A1/expectation)
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
  `pretrain_arbiter`, `probe_arbiter[_sim]`, `aggregate/plot_arbiter`),
  curricula, eval matrix, optimizer benchmarks and gates
- `docs/` — `training_stages.md` (stage dossier), `experiments.md` (log),
  `demo_manifest.md` (corpus fingerprint + restore)
- `results/`, `logs/`, `agent_code/*/runs/`, heavy checkpoints, demo corpora —
  local only, git-ignored (see `.gitignore`); mirror + manifest in `docs/`
