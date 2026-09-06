# BomberMan RL

Train Reinforcement Learning agents for the classic game Bomberman (course project setup).
Two trainable agents ship with this repo:

- **sentinel** — N-step Double Dueling DQN + PER on a 46-dim handcrafted feature MLP.
- **overlord** — same core + CNN spatial encoder and an auxiliary danger-prediction head.

Rule-based / scripted opponents (`rule_based_agent`, `coin_collector_agent`,
`peaceful_agent`, `random_agent`) are included for curriculum training and eval.
Tournament inference is CPU-only with a 0.5 s/step budget; training defaults to
CUDA with AMP (Google Colab ready).

## Requirements

- Python >= 3.12
- PyTorch with CUDA for training (`torch>=2.5.1,<2.6`, `torchvision>=0.20.1,<0.21`);
  CPU-only works for inference, eval, and (slow) training.
- The rest: `numpy pygame scikit-learn scipy tqdm matplotlib tensorboard`
  (see `pyproject.toml`).

## Setup

Local (uv):

```bash
uv sync
uv run python main.py play --no-gui --agents sentinel rule_based_agent --n-rounds 1
```

Google Colab (fresh GPU runtime):

```bash
git clone https://github.com/Alii-Khaled/BomberMan && cd BomberMan
pip install torch torchvision
pip install pygame scikit-learn scipy tqdm matplotlib tensorboard
python3 test_gpu.py   # expect a GPU name + PASS (active device: cuda)
```

To resume **overlord** training on Colab, copy two checkpoint files from
Google Drive into the repo (they are git-ignored, ~10 MB each):

- `agent_code/overlord/checkpoints/last.pt`
- `agent_code/overlord/checkpoints/best.pt`

Sentinel resumes from `agent_code/sentinel/checkpoints/best.pt`, which *is*
committed — no extra files needed.

## Training

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

```bash
bash scripts/run_sentinel_eval.sh            # frozen matrix M1-M8, 40 rounds x 2 seeds
python3 scripts/aggregate_eval.py            # tables
python3 scripts/plot_eval.py                 # figures
```

## Health checks

```bash
python3 test_gpu.py                 # CUDA probe + matmul bench + AMP training step
python3 test_cuda.py                # sentinel MLP fwd/bwd + checkpoint round-trip
python3 scripts/check_optimizer.py  # optimizer parity / determinism / factory gates
python3 scripts/verify_dml_optimizer.py  # full CUDA training-path verification
python3 test.py                     # 1-round game smoke test
```

## Repo layout

- `main.py`, `environment.py`, `settings.py`, `agents.py` — game engine + runner
- `agent_code/sentinel/` — MLP agent (`callbacks.py`, `train.py`, `model.py`,
  `features_mlp.py`, `safety.py`, `device.py`, `checkpointing.py`)
- `agent_code/overlord/` — CNN agent (same structure + `features_cnn.py`)
- `agent_code/{rule_based,coin_collector,peaceful,random}_agent/` — scripted opponents
- `dml_trainkit.py` — shared kit: update config, CUDA device picker, duty timer,
  checkpoint store, metrics logging, AMP flag
- `dml_optimizer.py` — optimizer factory (stock Adam/AdamW on CUDA/CPU, Lion;
  legacy DML-safe variants kept for old-checkpoint resume) + LR schedule
- `scripts/` — curricula, eval matrix, optimizer benchmarks and gates
- `docs/` — `training_stages.md` (stage dossier), `experiments.md` (log)
- `results/`, `logs/`, `agent_code/*/runs/`, heavy checkpoints — local only,
  git-ignored (see `.gitignore`)
