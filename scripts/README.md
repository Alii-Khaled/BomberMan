# Script index

Run scripts from the repository root with the project environment active.
Historical experiment numbers correspond to `docs/experiments.md`. Several
scripts retain the Arbiter-NG name for what is now the Harvey package.

## Evaluation

| Script | Purpose |
|---|---|
| `tournament_eval.py` | Frozen matchups, engine score, fractional win credit, and mean rank |
| `battery.py` | Multi-field evaluation with separate result files; `--tally-only` reads existing results |
| `report_ablation.py` | Final/neutral/search-off/BC comparison used in the report |
| `verify_report_evidence.py` | Verify published hashes and reviewed-source provenance |
| `audit_report_results.py` | Recompute the report's final ablation and E130 identity check |
| `aggregate_arbiter.py`, `plot_arbiter.py` | Historical Arbiter result summaries and figures |
| `aggregate_eval.py`, `plot_eval.py` | Sentinel evaluation summaries and figures |

The report ablation runner fixes starting boards, seats, and module-level
Python/NumPy streams. It does not seed Warden-v2's private generator or
record every inherited environment override. Its existing results support
the reported arithmetic; a new battery may differ. Use a separate clone or
output location for new measurements to preserve published evidence.

## Implementation probes

| Agent | Probes |
|---|---|
| Harvey | `probe_arbiter_ng.py`, `probe_arbiter_cointake.py`, `probe_arbiter_solo.py`, `probe_ng_tta_equiv.py`, `probe_harvey_joint_routes.py` |
| Historical Arbiter | `probe_arbiter.py`, `probe_arbiter_sim.py`, `probe_arbiter_crn.py`, `probe_arbiter_fault.py`, `probe_arbiter_plant.py` |
| Arbiter-v2 | `probe_arbiter_v2.py` |
| Reaper | `probe_reaper_features.py` |
| Apex | `probe_apex.py`, `probe_apex_lastlethal.py`, `probe_apex_train.py` |
| Recorders | `probe_legacy_recorders.py` |
| Shared optimizers | `check_optimizer.py`, `verify_dml_optimizer.py` |

Set `HARVEY_DEVICE=cpu` for Harvey probes. Latency thresholds describe the
local machine; they do not replace a tournament-CPU test.
`check_equivalence.py BEFORE AFTER` compares Python tokens and syntax trees
after removing comments and standalone string statements. It supports
recorded rename maps; it does not establish runtime equivalence for arbitrary
code changes.

## Training and historical tools

- `train_sentinel_curriculum.sh`, `train_overlord_curriculum.sh`: four-task curricula.
- `collect_demos.sh`, `collect_apex_demos.sh`: teacher recordings.
- `pretrain_reaper.py`, `pretrain_arbiter.py`, `pretrain_arbiter_ng.py`: cloning.
- `train_apex.sh`, `train_reaper.sh`: later model training.
- `run_e*.sh`, `run_p0_*.sh`: dated evaluation/collection launch settings.
- `diag_*.py`, `analyze_route_ablation.py`: event and action-trace diagnostics.
- `watch_overlord*.sh`, `keepalive_apex.sh`: historical training supervision.
- `build_team_update.py`: an early Sentinel progress summary, not the final report.
- `package_report_evidence.py`: the earlier local evidence archive; public checks
  use the tracked raw files directly.

Historical launchers may require private demonstrations or intermediate
checkpoints. Their settings describe an experiment, not a complete recipe
for reconstructing its original code, hardware, and random streams.
Launchers now resolve the repository from their own path and keep logs under
`logs/`. `run_gap_screens.sh` accepts separate `NAME=value` arguments; earlier
versions did not forward those arguments. This repair does not validate the
effective settings of historical runs.
