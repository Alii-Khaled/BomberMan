# BomberMan RL

Reinforcement learning agents built on the course framework
[`ukoethe/bomberman_rl`](https://github.com/ukoethe/bomberman_rl).
Agents play on a 17×17 board for up to 400 steps, earning one point per
coin and five per opponent eliminated. The decision limit is 0.5 seconds.

**Submitted agent: [`agent_code/Harvey/`](agent_code/Harvey/).**
Harvey combines a learned CNN/MLP policy, timed escape checks, and bounded
bomb-plan search. The trained weights are included.

## Setup

Use Python 3.12 or newer. The project dependencies are NumPy, PyTorch,
Pygame, Matplotlib, tqdm, and torchvision (for the optional GPU benchmark).
Harvey inference itself requires only NumPy and PyTorch, both present in
the supplied tournament Docker image.

Setup and CPU play were checked with Python 3.13, NumPy 2.4.6, and
PyTorch 2.11.0 using the committed lockfile.

From the repository root, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
HARVEY_DEVICE=cpu uv run python main.py play --no-gui \
  --agents Harvey rule_based_agent rule_based_agent rule_based_agent \
  --n-rounds 1
```

With an existing Python environment:

```bash
python -m pip install 'numpy>=2.4' 'torch>=2.5' 'pygame>=2.6' \
  'matplotlib>=3.10' 'tqdm>=4.70' 'torchvision>=0.20'
HARVEY_DEVICE=cpu python main.py play --no-gui --agents Harvey random_agent \
  random_agent random_agent --n-rounds 1
```

Remove `--no-gui` to watch a game. Training can use CUDA; use
`HARVEY_DEVICE=cpu` for tournament-style evaluation.

## Models

| Directory under `agent_code/` | Method |
|---|---|
| `Harvey/` | 12×17×17 board CNN fused with 98 scalars; behavioral cloning followed by KL-anchored policy updates |
| `sentinel/` | Dueling Double DQN over 46 engineered features |
| `overlord/` | Board CNN with a dueling Q head and auxiliary danger prediction |
| `reaper/` | 98-feature dueling MLP trained from Warden, Sentinel, and Overlord demonstrations, then game outcomes |
| `apex/` | CNN Q learner with 16 scalars and demonstration-augmented training |
| `arbiter/`, `arbiter_v2/`, `arbiter_rl/` | Earlier scalar-policy variants and training experiments |

Teacher and recorder packages support data collection. The framework
provides rule-based, coin-collecting, peaceful, and random opponents.
[`outsiders/`](outsiders/README.md) contains the team's Warden heuristics
and evaluation archetypes. The archetypes were used during model selection
and are therefore validation opponents.

## Harvey's action selection

1. Compute legal actions and bomb danger over eight future steps.
2. Average policy logits over eight rotated/reflected feature views.
3. Compare bomb plans with move plans within a shared 0.30-second budget.
4. If search returns no action, try a short coin route with a safe first step.
5. Rank fallback actions by policy logits and repetition penalties. Under
   immediate danger or post-plant escape, prefer safe actions when available.
   Bomb placement requires the safety gate.

The selected network has 848,672 parameters. Its value head is zero, and
default search uses no learned leaf value. The margin head serves only
the cloning loss. See the feature index in
[`features.py`](agent_code/Harvey/features.py) and the
[configuration reference](docs/configuration.md).

## Evaluation and report evidence

Run a new frozen matchup with:

```bash
HARVEY_DEVICE=cpu uv run python scripts/tournament_eval.py \
  --agents Harvey rule_based_agent rule_based_agent rule_based_agent \
  --n-rounds 40 --seed 0
```

The report's final ablation used 2,400 rounds: four settings, three opponent
fields, ten launch seeds, and 20 rounds per seed. Mean engine scores were:

| Opponents | Final Harvey | Neutral logits | Search off | Pre-RL cloning |
|---|---:|---:|---:|---:|
| Three rule-based agents | 4.345 | 2.565 | 4.610 | 4.660 |
| Warden-v2, Overlord, Sentinel | 4.975 | 3.195 | 2.330 | 5.175 |
| Cautious, bomber, rusher archetypes | 9.700 | 5.235 | 2.425 | 9.890 |

The learned policy improved score in all three fields. Search helped in
two; the final RL continuation showed no demonstrated gain over its cloning
checkpoint. Initial boards and seats match across settings, but the runner
does not fix Warden-v2's private random generator. These are local evaluation
results, not tournament standings.

The report uses **joint-top rate**, counting ties as top finishes.
`tournament_eval.py` reports **fractional win credit**, splitting ties.

Verify the published inputs and recompute the report's aggregates:

```bash
uv run python -B scripts/verify_report_evidence.py
uv run python -B scripts/audit_report_results.py
```

These commands check the 489 published inputs, source provenance, matched
starting boards, nine seed-block intervals, and the E130 identity audit.
See [`evidence/README.md`](evidence/README.md) for coverage and missing
historical records. The course requires the report to remain outside the
public repository.

## Training and checks

The [training guide](docs/training_stages.md) covers the four-task curricula
and Harvey's cloning/RL lineage. The historical demonstration corpus is
private; the available code and checkpoints do not reproduce the original
training run from scratch. [Corpus records](docs/demo_manifest.md) document
the retained fingerprints.

```bash
HARVEY_DEVICE=cpu uv run python -B scripts/probe_arbiter_ng.py
HARVEY_DEVICE=cpu uv run python -B scripts/probe_arbiter_cointake.py
uv run python -B scripts/probe_harvey_joint_routes.py
```

These probes exercise Harvey. See the [script index](scripts/README.md) for
historical-agent probes, data collection, and evaluation entry points.

## Repository layout

- `agent_code/`: learned agents, framework opponents, teachers, and recorders.
- `outsiders/`: heuristic reference agents, linked into `agent_code/`.
- `scripts/`: training, evaluation, plotting, and implementation probes.
- `docs/`: training/configuration references and the dated experiment ledger.
- `evidence/`, selected `results/` and `logs/`: report inputs and audits.
- `dml_trainkit.py`, `dml_optimizer.py`: shared training and optimizer utilities.
- Root game modules, `assets/`, and `Dockerfile`: course framework.

Local framework changes add rotating logs, headless audio setup, a
`crate-light` training scenario, and periodic progress output. Official games
use the original course framework. New logs, replays, demonstration corpora,
and intermediate checkpoints are ignored by Git.
