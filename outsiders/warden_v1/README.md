# Warden v1

The team's heuristic Bomberman baseline and training opponent.

Strengths over `rule_based_agent`: wall-aware blast computation, timer-based
danger map over a 6-step horizon, time-expanded escape search before every
bomb, strict bomb discipline (escape route + payoff required), loop avoidance.

It uses greedy target priorities without opponent modeling or long-term
planning. Warden-v2 corrects an escape-timing error in this version.

Run: `python main.py play --agents warden_v1 random_agent random_agent
random_agent --no-gui`
