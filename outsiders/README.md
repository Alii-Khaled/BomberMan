# Evaluation and sparring agents

This directory contains team-written heuristic references and evaluation-only
opponent archetypes. These agents may be used for training data or evaluation
as stated in the experiment records; none is a separate learned model or
part of the Harvey tournament payload (`agent_code/Harvey/`).

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
  unseen_*/            <- evaluation archetypes
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

From the repository root, create another link with
`ln -s ../outsiders/<name> agent_code/<name>`.

## Registry

| Agent | Source | Strength | Added |
|---|---|---|---|
| warden_v1 | own work (this repo) | heuristic, wall-aware escape + bomb discipline (frozen reference) | 2026-09 |
| warden_v2 | own work (this repo) | v1 + corrected escape solver, 8-step danger horizon, deterministic RNG; active sparring/teacher reference | 2026-09 |
| unseen_coward | own work (this repo) | eval-only sparring: never bombs, maximizes opponent/bomb distance, edge-seeking (passive-defensive archetype) | 2026-09-13 |
| unseen_bomber | own work (this repo) | eval-only sparring: bombs on cooldown under a cheap escape guard, random walk otherwise (reckless volume-bomber archetype) | 2026-09-13 |
| unseen_rusher | own work (this repo) | eval-only sparring: BFS chase + adjacency plant under escape guard, ignores coins (opponent-obsessed archetype) | 2026-09-13 |
| unseen_racer | own work (this repo) | eval-only sparring: BFS coin-runner, never bombs, ignores opponents (pure-economy archetype) | 2026-09-13 |

### Unseen-battery note

The four `unseen_*` agents are **evaluation-only behavior proxies**: their
policies are absent from the training demonstrations and were not BC
teachers. We repeatedly used their fields for model selection, so they
do not form an untouched final test set. They test behavior against
opponents outside the demonstration-teacher mixture.
