# Outsiders — external sparring agents

This directory collects third-party / downloaded agents used **only for
evaluation and sparring**. Nothing here ships in the tournament submission
(`agent_code/sentinel/`).

## Layout

```
outsiders/
  README.md            <- this file (registry)
  warden_v1/           <- our own heuristic bench agent (source of truth)
    callbacks.py
    README.md
  <future>/            <- downloaded agents, one subdir each
```

## Wiring into the game

The game loader only looks at `agent_code/<name>/callbacks.py`, so each
outsider is symlinked in:

```
agent_code/warden_v1 -> ../outsiders/warden_v1
```

Add future agents the same way:
`ln -s ../outsiders/<name> agent_code/<name>` (relative to `agent_code/`)

Downloaded agents go to `#final-project-beat-my-agent` on Discord; drop the
files into `outsiders/<name>/`, symlink, and cite the source in the report.

## Registry

| Agent | Source | Strength | Added |
|---|---|---|---|
| warden_v1 | own work (this repo) | heuristic, wall-aware escape + bomb discipline | 2026-09 |
