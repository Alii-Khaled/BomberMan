# Outsiders — external sparring agents

This directory collects third-party / downloaded agents used **only for
evaluation and sparring**. Nothing here ships in the tournament submission
(`agent_code/sentinel/`).

## Layout

```
outsiders/
  README.md            <- this file (registry)
  warden_v1/           <- frozen v1 heuristic bench agent (historical)
    callbacks.py
    README.md
  warden_v2/           <- active heuristic bench agent (source of truth)
    callbacks.py
    safety.py
    sim.py
    search.py
    README.md
  <future>/            <- downloaded agents, one subdir each
```

Naming rule: the warden is always referenced with a version suffix
(`warden_v1`, `warden_v2`, ...). Unversioned `warden` is never an agent
identifier; scripts read the active version from `WARDEN`
(default `warden_v2`).

## Wiring into the game

The game loader only looks at `agent_code/<name>/callbacks.py`, so each
outsider is symlinked in:

```
agent_code/warden_v1 -> ../outsiders/warden_v1
agent_code/warden_v2 -> ../outsiders/warden_v2
```

Add future agents the same way:
`ln -s ../outsiders/<name> agent_code/<name>` (relative to `agent_code/`)

Downloaded agents go to `#final-project-beat-my-agent` on Discord; drop the
files into `outsiders/<name>/`, symlink, and cite the source in the report.

## Registry

| Agent | Source | Strength | Added |
|---|---|---|---|
| warden_v1 | own work (this repo) | heuristic, wall-aware escape + bomb discipline (frozen reference) | 2026-09 |
| warden_v2 | own work (this repo) | v1 + corrected escape solver (E88 semantics), 8-step danger horizon, deterministic RNG; active sparring/teacher reference | 2026-09 |
