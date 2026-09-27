# BomberMan RL

Reinforcement learning agents for the classic game Bomberman, built on the
course framework from
[`ukoethe/bomberman_rl`](https://github.com/ukoethe/bomberman_rl).
Four agents share a board, move one tile or drop a bomb per step, and score
points for coins and for blowing each other up. Episodes run 400 steps and
every agent gets 0.5 s per decision.

Training runs on CUDA with AMP and works on Google Colab. Tournament
inference runs on CPU inside the 0.5 s budget.

**Local tournament candidate: [`agent_code/Harvey/`](agent_code/Harvey/).**
Confirm the actual MaMPF-uploaded zip before identifying it with this tree.
In the historical E130 battery it scored 7.116 points/round and finished
jointly top in 0.641 of 1,120 rounds; the 5.853-to-5.065 comparison with
`warden_v2` used the same weights in the earlier E108 inference setting
and a different opponent cohort.

## Models

The project develops several agents. The report describes all of them.

| Agent | Where | Method |
|---|---|---|
| **Harvey** | `agent_code/Harvey/` | CNN over a 12x17x17 board tensor fused with an MLP over 98 scalars. Behavioural-cloning warm start, then KL-anchored policy updates. Local tournament candidate; actual upload unverified. |
| **sentinel** | `agent_code/sentinel/` | MLP Dueling DQN over 46 engineered scalars, trained on a four-stage curriculum. |
| **overlord** | `agent_code/overlord/` | CNN DQN on the board tensor, trained on the same curriculum. Backup to Harvey. |
| **reaper** | `agent_code/reaper/` | Feature MLP distilled from `warden_v2` and `sentinel` by behavioral cloning. |
| **apex** | `agent_code/apex/` | Synthesis CNN. The learned Q values came out net negative, so it stays in the report and out of the tournament. |
| **arbiter, arbiter_v2, arbiter_rl** | `agent_code/` | Earlier generations of the Harvey line. Kept as the ablation history behind the final design. |

Training-only helpers, never submitted: `arbiter_dagger`,
`arbiter_v2_dagger`, `apex_teacher`, `reaper_teacher`, `solo_dagger`,
`exit_recorder`.

Sparring and evaluation opponents:

- `rule_based_agent`, `coin_collector_agent`, `peaceful_agent`, `random_agent`
  ship with the framework and drive the curriculum.
- `outsiders/warden_v1` is a frozen heuristic reference. `outsiders/warden_v2`
  adds a corrected escape solver, an eight-step danger horizon and a
  deterministic RNG.
- `outsiders/unseen_{coward,bomber,rusher,racer}` are evaluation-only
  behavior proxies, not training teachers. Repeated selection against them
  means they are not an untouched final test set.

## How Harvey decides

The policy network ranks moves. A safety layer cuts the candidate set down
first, and a bounded lookahead search decides bomb placement. The split
matters: a heuristic that shares the vote with the network double-counts the
same board facts and flattens the policy, so every survival adjustment is
capped at 0.5, an order of magnitude below a typical logit gap.

1. `action_safety` returns which actions are valid, which survive, and a
   danger timeline eight steps deep.
2. One batched forward pass over eight symmetry views returns six-action policy scores and a state value; the saved value head is zero and the default search blend is zero.
3. If our own tile is lethal within one step, only safe moves survive.
   Otherwise every valid move is ranked by policy score.
4. Loop and bomb-repeat tie-breaks apply. `BOMB` stays mask-safe.
5. The search runs inside a wall-clock budget and returns a bomb plan only
   when it beats the best move plan by a score margin.

Three components sit under that:

- `features.py` builds 98 scalars plus the board tensor. The scalars cover
  per-direction context, BFS routes to coins, crates and opponents, an escape
  solver, and a directional kill table. A docstring lists every index. The
  eight board symmetries act on the vector through a fixed permutation, which
  is what the training-time augmentation uses.
- `safety.py` does wall-aware blast geometry, future danger for t = 0..8, and
  a time-expanded BFS escape. Danger windows are non-contiguous, so the
  escape query keys on the *last* lethal time rather than the first.
- `sim.py` is a forward simulator that reproduces `environment.py` step order
  exactly. It makes two documented approximations, on movement order and on
  hidden coins, and the probes check the rest against the engine's own step
  functions.

Training (`train.py`) fine-tunes the policy head with REINFORCE, anchored by
KL against the frozen cloning prior, with a revert guard on the anchor.

## Requirements

Python 3.12 or newer.

For **inference, evaluation and the tournament**, the agent needs only
`numpy` and `torch`. The supplied Docker image already has both, so
`agent_code/Harvey/requirements.txt` installs nothing extra.

For **training**, install the following:

```bash
pip install torch torchvision
pip install numpy pygame matplotlib tqdm
```

Torch needs CUDA for comfortable training speed. CPU works for inference,
evaluation and slow training runs.

## Setup

Local, with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
uv run python main.py play --no-gui --agents Harvey rule_based_agent --n-rounds 1
```

Google Colab on a fresh GPU runtime:

```bash
git clone https://github.com/Alii-Khaled/BomberMan && cd BomberMan
pip install torch torchvision
pip install numpy pygame matplotlib tqdm
python3 test_gpu.py   # expect a GPU name and PASS (active device: cuda)
```

Trained weights are committed, so the agents run without extra downloads:

- `agent_code/Harvey/my-saved-model.pt` (local candidate; confirm actual upload)
- `agent_code/sentinel/my-saved-model.pt` and
  `agent_code/sentinel/checkpoints/best.pt`
- `agent_code/{overlord,reaper,apex,arbiter}/my-saved-model.pt`

To resume `overlord` training on Colab, copy two git-ignored checkpoints back
into the repo, about 10 MB each:
`agent_code/overlord/checkpoints/last.pt` and `.../best.pt`.

Demonstration corpora (`results/apex_demos/`, 500 rounds, and
`results/demos/`, 1312 npz files) are training-time only and git-ignored.
`docs/demo_manifest.md` holds their fingerprints and the restore procedure.

## Training

Harvey's weights came from board-plus-scalar cloning followed by a
KL-anchored policy update. The earlier Arbiter scalar-only pipeline has
different inputs; its commands do not reproduce Harvey's final weights.
With the historical demonstration corpus restored, the NG cloning
entry point is:

```bash
python3 scripts/pretrain_arbiter_ng.py \
  --dirs results/apex_demos,results/apex_ng_demos \
  --cache results/arbiter_ng_scalars.npz --epochs 14 --batch 256 \
  --init-scalars agent_code/arbiter/my-saved-model.pt \
  --out results/arbiter_ng_retrain.pt
```

The selected E108 run used 1,511 demonstration files and chose BC epoch
10, then RL episode 225 after frozen evaluation. The corpus is not
public, and the original RL launch environment is incomplete; this
command is an entry point, not a promise of byte-identical retraining.

The legacy curricula cover tasks 1 to 4 from the brief, from coin gathering
through crate clearing to full combat:

```bash
bash scripts/train_sentinel_curriculum.sh   # coin-heaven -> classic solo -> hunt -> vs rule_based
bash scripts/train_overlord_curriculum.sh   # the same ladder for the CNN agent
```

A single run:

```bash
python3 main.py play --no-gui --agents sentinel --train 1 \
  --scenario classic --n-rounds 1500 --save-stats results/sentinel_stage2.json
```

`--train N` puts the first N agents into training mode. `--train 0` with
`--continue-without-training` runs a frozen evaluation. Checkpoints land in
`agent_code/<agent>/checkpoints/` (`last.pt` each round, `best.pt` on EMA
improvement, `ep_NNNNNN.pt` snapshots). Per-round metrics append to
`agent_code/<agent>/runs/metrics.csv`. Tournament weights export to
`agent_code/<agent>/my-saved-model.pt` as a CPU state dict.

### Environment knobs

These knobs default to the current local configuration. The controlled
ablations vary one switch or checkpoint in that code; earlier historical
variants also had other code changes.

| Variable | Default | Effect |
|---|---|---|
| `HARVEY_SEARCH` | `search` | `off` runs the policy alone, `tactical` adds the proven-kill check, `search+tactical` runs both |
| `HARVEY_TIME_BUDGET` | `0.30` | Wall-clock search budget per step, under the 0.5 s tournament limit |
| `HARVEY_V_BLEND` | `0.0` | Learned leaf-value weight; the selected checkpoint's value head is zero and the default search does not use it. |
| `HARVEY_BOMB_MARGIN` | `0.6` | Older name for `HARVEY_BOMB_SCORE_MARGIN`, still read |
| `HARVEY_BOMB_SCORE_MARGIN` | `0.6` | A bomb plan must beat the best move plan by this score margin |
| `HARVEY_DUEL_BOMB_MARGIN` / `_D` | `0` / `3` | Extra score margin for a plant near an opponent. Off; the measurement lost |
| `HARVEY_JOINT_ROUTES` | `0` | Turns on both joint-route components. Off; the combined ablation lost |
| `HARVEY_DYNAMIC_ROUTES` | inherits | Kill certification that lets moving blockers clear out first. Off |
| `HARVEY_BODYBLOCK` | inherits | Minimax own-escape check against nearby moving blockers. Off |
| `HARVEY_ROUTE_DIAG` | unset | JSONL path for route counts, forced-kill certificates, body-block vetoes and selected actions |
| `HARVEY_JOINT_HORIZON` / `_BODY_D` | `6` / `3` | Route horizon and Manhattan range for the body-block analysis |
| `HARVEY_BOMB_ESC_MARGIN` | `1` | Distinct first-step escape directions the mask needs to certify `BOMB` |
| `HARVEY_SOLO_MARGIN` | `0.15` | Bomb-versus-move score margin while no opponent is alive. Negative restores the earlier value |
| `HARVEY_LOOP_ESC` | `2` | Loop tie-break mode: `0` bounded, `1` escalates everywhere, `2` escalates while solo |
| `HARVEY_LOOP_ESC_STEP` / `_CAP` | `0.5` / `3.0` | Escalation slope and cap for `HARVEY_LOOP_ESC` |
| `HARVEY_SOLO_TREK` | `0` | Solo BFS trek move plans. Off |
| `HARVEY_SOLO_RADIUS` | `8` | Bomb-tile candidate BFS radius while solo. `0` restores 4 |
| `HARVEY_COINTAKE` | `1` | Certified coin collection: the first step of a short mask-safe path to a visible coin. Worth a certain +1 |
| `HARVEY_COINTAKE_D` | `3` | Reach of that path in BFS steps. A sweep over 1, 2 and 3 picked 3 |
| `HARVEY_SOLO_COMMIT` | `6` | Commits to a bomb-approach target while solo, which stops the approach oscillating |
| `HARVEY_SOLO_COMMIT_MAX` | `6` | Age limit on a committed approach, in ticks |
| `HARVEY_BOMB_HYST` | `0` | Sticky-target arbitration bonus. Off; the screens lost 0.89 |
| `HARVEY_BACKTRACK` | `0` | Solo immediate-reversal penalty. Off |
| `HARVEY_COMMIT_OPP` | `1` | Commits to a bomb-approach target with opponents alive. `0` limits it to solo play |
| `HARVEY_BACKTRACK_OPP` | `0.5` | Immediate-reversal penalty with opponents alive. `0` restores |
| `HARVEY_OPEN_MARGIN` | `-1` | Opening-phase bomb margin. Off |
| `HARVEY_OPEN_T` | `100` | Opening window length in steps |
| `HARVEY_HUNT_OPEN` | `0` | Opening-only pursuit plans. Off |
| `HARVEY_ANTIPIN` (+ `_D`/`_MOB`/`_DEADEND`) | `0` | Guards against being pinned by an armed enemy. Off |
| `HARVEY_KILL_P` | `1.0` | Scale on certified-kill credit. A sweep at 0.5 and 0.25 lost to the control |
| `HARVEY_ESC_DIST` | `3.0` | Proven-escape distance limit for bomb tiles |
| `HARVEY_SEEDS` | `1` | Rollout seeds for plan scoring. Moves use `HARVEY_MOVE_SEEDS`, bombs use 1 |
| `HARVEY_CRN` | `1` | Common random numbers, so every plan sees the same per-tick opponent draws. `0` gives each plan its own draws |
| `HARVEY_PLANT_ESC` | `1` | Minimum post-plant escape directions for the search bomb gate |
| `HARVEY_PI_OFF` / `HARVEY_V_OFF` | `0` | Ablations: `1` forces a uniform prior or a zero value |
| `HARVEY_WEIGHTS` | unset | Weights to load instead of `my-saved-model.pt` |
| `HARVEY_MODEL` | unset | Older name for `HARVEY_WEIGHTS`, still read |
| `HARVEY_FLEE_Q` | `0` | Ranks post-plant flee moves by open space and opponent distance. Off |
| `HARVEY_RL_SELF_DEATH` / `_ENEMY_DEATH` | `-8` / `-6` | Training-only death rewards |
| `SENTINEL_DEVICE` / `OVERLORD_DEVICE` | `auto` | `cpu` forces CPU, which is the tournament condition |
| `SENTINEL_AMP` / `OVERLORD_AMP` | `1` | `0` disables AMP autocast and GradScaler for fp32 |
| `SENTINEL_OPT` / `OVERLORD_OPT` | `adam` | `lion` selects the Lion optimiser |
| `SENTINEL_LR` / `OVERLORD_LR` | agent default | Base learning rate override |
| `SENTINEL_SCHEDULE` / `OVERLORD_SCHEDULE` | `0` | `1` enables warmup and a cosine schedule |
| `SENTINEL_TUNED` / `OVERLORD_TUNED` | `0` | `1` applies the DQN preset (Adam eps and decay 1e-4) |
| `SENTINEL_UTD` / `OVERLORD_UTD` | `1` | Gradient updates per environment step |
| `SENTINEL_BATCH` / `OVERLORD_BATCH` | `0` | Batch override. The agents default to 256 and 512 |
| `SENTINEL_EOR_UPDATES` / `OVERLORD_EOR_UPDATES` | `4` / `6` | Extra updates at round end |

`SENTINEL_DML` and `OVERLORD_DML` are still read, where `0` forces CPU, but
they no longer choose a backend. Device choice is CUDA or CPU only.

## Evaluation

Our report treats engine score per round as the primary evaluation outcome
and also gives joint-top rate (counting every highest-score tie as a top
finish). `scripts/tournament_eval.py` separately reports fractional win
credit (splitting ties) and mean rank; these rates must not be conflated.

```bash
# per-round win rate and mean rank
python3 scripts/tournament_eval.py --agents Harvey rule_based_agent \
  rule_based_agent rule_based_agent --n-rounds 40 --seed 0

# pooled score tables and figures across a battery
python3 scripts/aggregate_arbiter.py   # results/arbiter_summary.csv
python3 scripts/plot_arbiter.py        # results/figures/arbiter_*.png

# death and missed-kill attribution from the HARVEY_DIAG trace
python3 scripts/diag_arbiter_deaths.py
```

A battery runs G1 (rule-based, 100 rounds x 2 seeds), G2 (warden mix, 60 x 2),
G3 (collectors, 40 x 2) and G4 (random, 40 x 2), plus archetype mixtures.
We used fresh same-session controls and also retained some defect-specific
repairs without a proved pooled advantage. The full decision ledger lives in
`docs/experiments.md`.

Historical E130 battery: Harvey scored 7.116 pooled with a 0.641 joint-top rate
against a same-session control at 7.031 and 0.652. G1 gained 0.640
points/round and 6.5 percentage points of joint-top rate; the strong lobby
lost 0.083 points/round and 6.25 percentage points of joint-top rate.
The five other field legs had identical per-round scores in both arms.
In a later, separate 200-round controlled ablation, Harvey scored 4.975
in the strong lobby versus 5.565 for `warden_v2`; the final RL weights did
not show a score gain over their D1 cloning checkpoint in that ablation.
These cohorts do not estimate performance against unknown student agents.

The 489 published appendix inputs retain their paths under `results/`
and `logs/`. From a clone, run `python3 -B
scripts/verify_report_evidence.py` to check their SHA-256 hashes against
`evidence/manifest.json`. Then run `python3 -B
scripts/audit_report_results.py` to check 120 final-ablation cohorts
(2,400 rounds), matched starting boards, nine seed-block intervals and
the 520 E130 rounds with identical non-runtime records. No new matches
are played. See [`evidence/README.md`](evidence/README.md) for scope
and missing early ledger-only sources. The report PDF and source remain
outside the public repository.

The legacy sentinel matrix runs through:

```bash
bash scripts/run_sentinel_eval.sh            # matrix M1-M8, 40 rounds x 2 seeds
python3 scripts/aggregate_eval.py            # tables
python3 scripts/plot_eval.py                 # figures
```

## Tests

```bash
python3 test.py                             # one-round smoke test
python3 test_gpu.py                         # CUDA probe, matmul bench, AMP step
python3 test_cuda.py                        # sentinel MLP forward/backward and checkpoint round-trip
python3 scripts/check_optimizer.py          # optimiser parity, determinism, factory
python3 scripts/verify_dml_optimizer.py     # full CUDA training path
python3 scripts/check_equivalence.py A B    # prove two source trees differ only in comments and docstrings

python3 scripts/probe_arbiter.py            # shapes, feature/safety parity, escape correctness, latency
python3 scripts/probe_arbiter_sim.py        # simulator against engine, plus survival fuzzing
python3 scripts/probe_arbiter_crn.py        # common-random-number wiring and determinism
python3 scripts/probe_arbiter_flee.py       # flee ranking wiring
python3 scripts/probe_reaper_features.py    # reaper feature and escape parity
```

`scripts/check_equivalence.py` is how we guarantee a style or rename pass
changed no behavior. It compares token streams and syntax trees with comments
and docstrings removed, and takes a rename map so a mechanical rename can be
verified the same way. Any difference beyond that fails the run.

## Repository layout

Game framework, unchanged from `ukoethe/bomberman_rl` except where noted:

- `main.py`, `environment.py`, `settings.py`, `agents.py`, `items.py`,
  `events.py`, `fallbacks.py`, `replay.py` — engine and runner
- `assets/`, `Dockerfile` — sprites and the tournament image
- `agent_code/{rule_based,coin_collector,peaceful,random,user,tpl,fail}_agent/`
  — opponents and the agent template that ship with the framework

Framework files we extend for training only. The tournament plugs the agent
into the original framework, so none of this runs in official games:

| File | Change |
|---|---|
| `agents.py`, `environment.py` | Rotating log handlers. An unbounded log filled the volume and took the notebook server down |
| `fallbacks.py` | Sets `SDL_AUDIODRIVER=dummy` so headless runs start without audio |
| `settings.py` | Adds a `crate-light` training scenario with sparse crates |
| `main.py` | Replaces tqdm with periodic progress lines for long headless runs |

Our code:

- `agent_code/Harvey/` — the local agent candidate: `model.py` (policy and value
  net), `features.py`, `safety.py`, `sim.py`, `search.py`, `callbacks.py`
  (decision order), `train.py`, `rl_policy.py`
- `agent_code/{sentinel,overlord,reaper,apex}/` — the other models
- `agent_code/{arbiter,arbiter_v2,arbiter_rl}/` — earlier Harvey generations
- `agent_code/{arbiter_dagger,arbiter_v2_dagger,apex_teacher,reaper_teacher,solo_dagger,exit_recorder}/`
  — training-only recorders and teachers
- `outsiders/` — team-written heuristics and evaluation-only behavior proxies
- `dml_trainkit.py` — update config, CUDA device picker, duty timer, checkpoint
  store, metrics logging, AMP flag
- `dml_optimizer.py` — optimiser factory (Adam, AdamW, Lion) and LR schedule
- `scripts/` — data collection, pretraining, probes and gates, sweeps,
  evaluation harness, plotting and diagnostics
- `docs/` — `experiments.md` (full experiment ledger), `training_stages.md`
  (stage dossier), `demo_manifest.md` (corpus fingerprints and restore)
- `results/`, `logs/`, `replays/`, `screenshots/`, `agent_code/*/runs/`,
  heavy checkpoints and demo corpora are local only and git-ignored. The
  corpus mirror and manifest live in `docs/`.
