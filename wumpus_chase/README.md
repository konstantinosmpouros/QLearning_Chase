# Wumpus Chase

Simple two-agent zero-sum grid game inspired by Wumpus World and the chase tag.

Rules (default layout):

- Grid: 7x7 with a few obstacles, one wumpus cell, one treasure cell.
- Simultaneous actions with optional move failure (p_fail).
- Win if you capture the opponent or reach the treasure first.
- Stepping on the wumpus loses immediately.
- Reward = outcome (±`outcome_reward`, default 5.0) plus step penalty and shaping terms (obstacle penalty, chase distance, treasure distance).
- If the max step limit is reached, the outcome is a draw.
- Capture if agents land on the same cell or cross each other.
  If capture is ambiguous, the winner is chosen randomly.

Project layout

- src/env: environment and map layout
- src/agents: FP and Minimax-Q agents
- src/train: simple training and evaluation loops
- src/logger: CSV logger for step-by-step traces
- src/utils: state encoding and LP solver

How to run

```bash
python3 wumpus_chase/src/wumpus_chase.py
```

Logging

- Training log: `wumpus_chase/results/wumpus_train.csv`
- Eval log: `wumpus_chase/results/wumpus_eval.csv`
- Each row records state/action/next-state plus treasure, wumpus, and obstacles.

Tuning ideas

- Edit the default layout in `wumpus_chase/src/env/maps.py`.
- Change `p_fail`, `t_max`, and episodes in `wumpus_chase/src/wumpus_chase.py`.
- Tune reward shaping in `wumpus_chase/src/env/wumpus_env.py` (`outcome_reward`, `obstacle_penalty`, `chase_dist_reward`, `treasure_dist_reward`).
