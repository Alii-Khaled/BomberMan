# Demo corpus manifest (S3, 2026-09-09)

Restored after the 2026-09-09 disk cleanup wiped `results/apex_demos/`
(400 npz, E51) and `results/demos/` (838 npz, E33–E37). E51's stats JSONs
survive in `results/archive/e51_demos/`. Corpus fingerprint (sha256 of the
sorted per-file sha256 list): **`e2239e6b1ac463d7`**.
Backup (byte-identical, verified loadable): `/home/jovyan/work/__shared/demos_backup/`
(`apex_demos/`, `demos/`, `apex_demos_all/` as symlinks, `apex_demos_all_resolved/`).

## Apex-format (`results/apex_demos/<teacher>/`, B3: img uint8 T,12,17,17 ×4 + sc T,16 + act T,)

Recorder: `agent_code/apex_teacher/` (`APEX_TEACHER=<t>`, resume-safe round IDs).
Command: `bash scripts/collect_apex_demos.sh` (new in S3).

| Teacher | Rounds | Steps | Fields (rb / warden-mix / 3×collector / crate-light) |
|---|---|---|---|
| warden_v1 | 200 | 54,275 | 100 / 50 / 25 / 25 |
| sentinel | 100 | 25,027 | 51 / 25 / 12 / 12 |
| overlord | 100 | 24,659 | 51 / 25 / 12 / 12 |
| coin_collector_agent (NEW — only 3.39 crates/bomb demonstrator in repo) | 100 | 19,251 | 51 / 25 / 12 / 12 |
| **Total** | **500** | **123,212** | |

Action mix: UP .186 / RIGHT .178 / DOWN .185 / LEFT .176 / WAIT .193 / BOMB .082.
Validation: 500/500 load, shapes exact, values ⊆ {0,…,4}, 0 bad files.
Stats: `results/apex_demos_<teacher>_{rb,wm,co,cr}.json`.

## Reaper-format (`results/demos/<teacher>[_<field>]/`, feats T,98 + acts T,)

Recorder: `agent_code/reaper_teacher/`. Command: `bash scripts/collect_demos.sh`
(unchanged — gate-fields mix rb 50% / wm 25% / random 12.5% / collector 12.5%).

| Teacher | Rounds | Fields |
|---|---|---|
| warden | 400 | rb 200 / wm 100 / rn 50 / cl 50 |
| sentinel | 200 | rb 100 / wm 50 / rn 25 / cl 25 |
| overlord | 200 | rb 100 / wm 50 / rn 25 / cl 25 |
| **Total** | **800** | **216,614 samples, 0 bad** |

## Symlink farm (`results/apex_demos_all/`, flat, absolute targets)

500 links `<teacher>_round_NNNNNN.npz`, 0 dangling (E60 lesson: `train.py`
globs non-recursive; relative targets resolve nowhere under the backend
chdir). Rebuild: see S3 log; verify with `find results/apex_demos_all -xtype l | wc -l` (= 0).

## Loader gates (E52 class, retired)

- Library (`agent_code/apex/train.py:_load_demos`): warns loudly with cwd
  diagnostics on configured-but-empty (E60).
- Launcher (`scripts/train_apex.sh`): **refuses** to start when
  `APEX_DEMO` holds 0 *readable* npz (`-readable`, so dangling links do
  not count — S3 probe: refuses dangling farm, refuses missing dir,
  passes real npz).
