# Configuration reference

Run commands from the repository root. Modules read environment variables
at import time, so set overrides before starting Python. Historical experiment
entries describe the settings used at that date; current defaults may differ.

## Harvey inference

| Variable | Default | Effect |
|---|---|---|
| `HARVEY_DEVICE` | `cuda:0` | Preferred device, with CPU fallback. Set `cpu` for evaluation. |
| `HARVEY_WEIGHTS` | unset | Override `my-saved-model.pt`; relative paths resolve from the repository root. |
| `HARVEY_MODEL` | unset | Legacy weight-path alias; `HARVEY_WEIGHTS` takes precedence. |
| `HARVEY_SEARCH` | `search` | `off`, `search`, `tactical`, or `search+tactical`. Search-off retains the policy, safety checks, and coin route. |
| `HARVEY_TIME_BUDGET` | `0.30` | Shared callback budget in seconds; the engine limit is 0.5 s. |
| `HARVEY_TTA` | `1` | Average logits over eight board symmetries. |
| `HARVEY_V_BLEND` | `0.0` | Search leaf-value weight. The selected checkpoint's value head is zero. |
| `HARVEY_BOMB_SCORE_MARGIN` | `0.6` | Minimum bomb-plan advantage over the best move plan. |
| `HARVEY_BOMB_MARGIN` | `0.6` | Legacy alias, used if the score-margin variable is unset. |
| `HARVEY_SEARCH_H` / `_K` / `_R` / `_PLANS` | `6` / `8` / `4` / `48` | Nominal horizon, bomb candidates, BFS radius, and plan cap. Rollouts extend to settle bombs. |
| `HARVEY_ESC_DIST` | `3.0` | Maximum certified escape distance at a candidate bomb tile. |
| `HARVEY_BOMB_ESC_MARGIN` | `1` | Minimum distinct first-step escape directions for the action mask. |
| `HARVEY_PLANT_ESC` | `1` | Minimum escape directions at the search bomb gate. |
| `HARVEY_CRN` | `1` | Share per-tick opponent random draws across candidate plans. |
| `HARVEY_MOVE_SEEDS` | `1` | Rollout samples per move plan; bomb plans use one. |
| `HARVEY_SEEDS` | `1` | General rollout-sample setting; current root plans override it as above. |
| `HARVEY_KILL_P` | `1.0` | Scale on certified-kill credit. |
| `HARVEY_COINTAKE` / `_D` | `1` / `3` | Short coin route with a safe first step, up to three BFS steps away. |
| `HARVEY_SOLO_MARGIN` | `0.15` | Bomb margin while solo; negative values use the general margin. |
| `HARVEY_SOLO_RADIUS` | `8` | Solo bomb-candidate radius; zero uses the general radius. |
| `HARVEY_SOLO_COMMIT` / `_MAX` | `6` / `6` | Positive `_COMMIT` enables target commitment; `_MAX` sets maximum age in ticks. |
| `HARVEY_COMMIT_OPP` | `1` | Enable approach commitment while opponents remain. |
| `HARVEY_LOOP_ESC` | `2` | Loop escalation: `0` bounded, `1` all states, `2` solo only. |
| `HARVEY_LOOP_ESC_STEP` / `_CAP` | `0.5` / `3.0` | Loop penalty slope and cap. |
| `HARVEY_BACKTRACK` / `_OPP` | `0` / `0.5` | Reversal penalty while solo / with opponents. |
| `HARVEY_PI_OFF` / `HARVEY_V_OFF` | `0` / `0` | Disable policy inference / leaf value. The report's neutral-logit control instead runs the full network before zeroing logits. |

Experimental options default to disabled: `HARVEY_DUEL_BOMB_MARGIN`,
`HARVEY_SOLO_TREK`, `HARVEY_BOMB_HYST`, `HARVEY_HUNT_OPEN`,
`HARVEY_ANTIPIN`, `HARVEY_FLEE_Q`, and `HARVEY_FLEE_LOOK`.
`HARVEY_OPEN_MARGIN=-1` disables its opening-specific margin.
Their results appear in [the experiment ledger](experiments.md).

`HARVEY_JOINT_ROUTES=0` leaves joint-route analysis disabled.
`HARVEY_DYNAMIC_ROUTES` and `HARVEY_BODYBLOCK` inherit that setting unless
overridden. `HARVEY_JOINT_HORIZON=6` and `HARVEY_JOINT_BODY_D=3` set
the route horizon and nearby-opponent range.

## Diagnostics

Set `HARVEY_DIAG`, `HARVEY_PERF`, or `HARVEY_GAP_DIAG` to an output prefix
for death, timing, or navigation traces. `HARVEY_ROUTE_DIAG` takes a JSONL
file path. Leave these unset during ordinary evaluation. Some buffers flush
at round changes or training hooks; a frozen run's final round may lack a
flush if it ends early.

## Training

Sentinel and Overlord share the following variables, with the agent name
as prefix, for example `SENTINEL_DEVICE=cpu`:

| Suffix | Default | Meaning |
|---|---|---|
| `_DEVICE` | `auto` | CUDA when available, otherwise CPU |
| `_AMP` | `1` | CUDA mixed precision; `0` selects float32 |
| `_OPT` | agent default | Optimizer selection; see the agent's `train.py` |
| `_LR` | agent default | Learning rate |
| `_SCHEDULE` | `0` | Warmup/cosine schedule |
| `_TUNED` | `0` | Optimizer preset with epsilon and weight decay of 1e-4 |
| `_UTD` | `1` | Updates per environment step |
| `_BATCH` | `0` | Zero uses the agent default: 256 / 512 |
| `_EOR_UPDATES` | `4` / `6` | Updates at the end of a round |

The legacy `_DML=0` setting forces CPU; DirectML is no longer a backend.
For Harvey's selected historical training recipe, see
[training_stages.md](training_stages.md). Set `HARVEY_RL_OUT` to a new path
when experimenting: its default points to the submitted weight file.
