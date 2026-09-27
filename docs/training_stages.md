# Training guide

The project followed the assignment's four tasks: coin navigation, solo
crate clearing, hunting, and competitive play. This guide separates the
historical training records from commands supported by the current code.
The [experiment ledger](experiments.md) contains dated settings and results.

## Sentinel and Overlord curricula

| Task | Environment | Sentinel rounds | Overlord validation rounds |
|---|---|---:|---:|
| 1: navigation | Solo `coin-heaven`, 50 visible coins, no crates | 500 | 200 |
| 2: bombs and escape | Solo `classic`, nine hidden coins | 1,500 | 300 |
| 3: hunting | `peaceful_agent` and `coin_collector_agent` | 1,000 | 300 |
| 4: competition | Three `rule_based_agent` opponents | 2,000 | 500 |

All four stages were completed. Sentinel collected 48.6 coins per training
round in Task 1. Its final 210 hunting rounds averaged about 0.49 kills.
Overlord's recorded solo-classic stage averaged 1.34 coins and 18.1 destroyed
crates per round. These are training outcomes under different settings, not
a frozen-policy comparison. Early Sentinel raw files are missing; the
report identifies those figures as ledger-sourced.

```bash
bash scripts/train_sentinel_curriculum.sh
bash scripts/train_overlord_curriculum.sh
```

Use an active project environment (`source .venv/bin/activate`) for shell
launchers. A single stage can also run directly:

```bash
uv run python main.py play --no-gui --agents sentinel --train 1 \
  --scenario classic --n-rounds 1500 --save-stats results/sentinel_stage2_new.json
```

`--train N` trains the first N agents. For frozen evaluation, use
`--train 0 --continue-without-training`. Training writes agent checkpoints
and exports, so use a separate working copy for new runs.

### Learner and checkpoint details

Sentinel uses a 46-feature dueling MLP; Overlord uses a board CNN with eight
additional scalars. Both use n-step Double DQN and prioritized replay.
The initial Sentinel recipe used three-step returns; the later stabilized
recipe uses five-step returns, Huber loss, and gradient clipping. Consult
each `train.py` for current defaults and [configuration.md](configuration.md)
for overrides.

Early experiments ran through DirectML on an AMD RX 6900 XT. Later training
used CUDA on an NVIDIA L40S. Current device selection supports CUDA and CPU;
the `dml_*` module names remain for compatibility with the earlier work.

Checkpoints under `agent_code/<agent>/checkpoints/` contain model, optimizer,
episode, and schedule state. `last.pt` supports resume; `best.pt` tracks
training reward; `ep_NNNNNN.pt` files are periodic snapshots. Replay buffers
are not saved. Frozen evaluation selected the exported `my-saved-model.pt`;
the highest training-reward checkpoint was not always the strongest player.
Metrics append to `agent_code/<agent>/runs/metrics.csv`.

The Lion/high-update-rate experiment diverged (E04). The subsequent
Huber/five-step changes reduced TD loss, but own-bomb deaths required an
escape-timing correction (E14b). Overlord's easier `crate-light` continuation
failed its classic-board transfer test (E40–E41), so the earlier export was
retained.

## Reaper and Apex

Reaper clones Warden, Sentinel, and Overlord actions into a 98-feature MLP,
then uses game training with teacher replay. The E35 corpus held 908 rounds
and 257,370 examples; five cloning epochs reached 0.82 validation accuracy.
The selected historical ep400 checkpoint scored 3.31 in 200 frozen G1
rounds. E36 records that comparison; its raw bake-off files are missing.

```bash
bash scripts/collect_demos.sh
uv run python scripts/pretrain_reaper.py --help
bash scripts/train_reaper.sh
```

Apex combines a CNN, 16 scalars, a dueling Q head, and a danger target.
Its planned early cloning stage did not run. The later A4 experiment used
111,935 demonstration transitions in a DQfD-style update. The historical
Q-off comparison outscored Q-on, so Apex was not selected for submission.

## Arbiter to Harvey

| Stage | Training or inference change | Retained evidence |
|---|---|---|
| Arbiter P0, E65 | Scalar policy/value cloning, five epochs | 339,826 policy rows; validation accuracy 0.757 |
| Arbiter P1, E66 | Bounded bomb search with the cloned policy | Historical frozen G1 comparisons |
| Arbiter-v2, E99 | 114-feature variant | Rejected frozen candidates in the ledger |
| NG / D1, E108 | Board CNN fused with 98 scalars | 1,511 files; 346,263 usable examples |
| D1 RL, E108 | KL-anchored game-outcome update | Selected episode 225 of 300 |
| Harvey, E130 | Later inference settings on the E108 weights | Historical battery and final report ablations |

The selected D1 cloning run reserved 152 files for validation, yielding
311,428 training and 34,835 validation rows. Adam used learning rate 1e-3,
batch size 256, and eight symmetry transforms. Of 14 epochs, epoch 10 had
the best saved validation accuracy, 0.8063. Sources:
`logs/e108_bc.log` and `results/arbiter_ng_d1.meta.json`.

With the private demonstration corpus restored, the cloning entry point is:

```bash
uv run python scripts/pretrain_arbiter_ng.py \
  --dirs results/apex_demos,results/apex_ng_demos \
  --cache results/arbiter_ng_scalars.npz --epochs 14 --batch 256 \
  --init-scalars agent_code/arbiter/my-saved-model.pt \
  --out results/arbiter_ng_retrain.pt
```

The historical RL leg used two rule-based opponents and Warden-v2, learning
rate 5e-5, initial KL coefficient 0.1, discount 0.99, adaptive anchoring,
and search-bomb traces. Frozen evaluation selected episode 225.
`agent_code/Harvey/my-saved-model.pt` and `results/e108rl.pt.ep225` have
identical SHA-256 hashes.

The update weights chosen-action log probabilities by normalized returns
and penalizes `KL(frozen prior || current policy)`. Search-selected bombs
are return-weighted labels. Movement sampling may use symmetry-averaged
logits while the update uses one canonical view; the retained launch record
does not establish whether training disabled averaging. The report documents
this mismatch and finds no demonstrated final RL score gain over D1 cloning.

The original corpus and complete RL launch environment are not public.
The command above supports new cloning runs with suitable data; it does not
reconstruct the submitted weights byte-for-byte. See
[demo_manifest.md](demo_manifest.md) for historical corpus fingerprints and
[the evidence index](../evidence/README.md) for public result checks.
