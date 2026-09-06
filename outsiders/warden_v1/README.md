# Warden v1

Heuristic (non-learning) Bomberman agent, written from scratch as a sparring
partner and report baseline for the sentinel project.

Strengths over `rule_based_agent`: wall-aware blast computation, timer-based
danger map over a 6-step horizon, time-expanded escape search before every
bomb, strict bomb discipline (escape route + payoff required), loop avoidance.

Weaknesses (by design — this is what the learned agent must beat): no
opponent modelling, greedy target priority, no long-term planning.

Run: `python main.py play --agents warden_v1 random_agent random_agent
random_agent --no-gui`
