# Chase – Simultaneous-Move Tag (FP & Minimax-Q)

Lightweight 5×5 tag environment (catcher vs runner) with two agents:

- **Fictitious Play (FP)**: empirical opponent model + 1-step best response.
- **Minimax-Q**: tabular zero-sum RL with LP-derived mixed strategies.

Training runs log every step to Excel so you can inspect both learning and evaluation behavior.

## Project layout

- `chase/src/env/` – `TagEnv`, actions (`ACTIONS`), movement deltas, capture rules.
- `chase/src/agents/` – `FPAgent`, `MinimaxQAgent` (+ LP solver in `utils/lp.py`), state encoding in `utils/state.py`.
- `chase/src/train/` – training/eval loops (`fp_vs_fp`, `minimaxq_selfplay`, `minimaxq_vs_minimaxq`, `minimaxq_vs_fp`), shared `evaluate`/`run_episode`.
- `chase/src/logger/` – `ExcelLogger` (writes per-step rows to Excel).
- `chase/src/chase.py` – entrypoint wiring the four matchups.
- `chase/notebook/` – `analytics_train.ipynb`, `analytics_eval.ipynb`, and helper utils for analysis.
- `results/` (created at runtime) – Excel logs for training (`chase_train.xlsx`) and evaluation (`chase_eval.xlsx`).

## How to run

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all matchups with logging
python3 chase/src/chase.py
```

Current defaults in `chase.py`:

- `p_fail_values = [0.10, 0.20]`
- Episodes: FP 10_000; MinimaxQ self-play 10_000; MinimaxQ vs FP 10_000
- Eval cadence: every **50** training episodes for all matchups
- Seeds: FP run seed=0; MinimaxQ self-play seed=1; MinimaxQ vs FP seed=2 (with offsets for eval envs)

Outputs:

- Console: final capture rate, avg steps, avg return per matchup/p_fail.
- Excel: `results/chase_train.xlsx` (training steps) and `results/chase_eval.xlsx` (eval steps).

## Environment basics

- Grid: 5×5. State = `(catcher_x, catcher_y, runner_x, runner_y)`.
- Actions: `STAY, UP, DOWN, LEFT, RIGHT`; moves clamp to grid edges.
- Each agent’s move independently **fails** with probability `p_fail`; on fail, agent stays.
- Capture if they land on the same cell or cross paths in a step.
- Reward to catcher: `+10` on capture; otherwise `-step_penalty` (default 0.01) plus shaping terms (wall penalty, move reward, distance change). Episode ends on capture or after `t_max` (default 30) steps.

## Agents (summary)

- **FPAgent**
  - Tracks opponent action counts per state (Dirichlet prior), computes expected capture prob + Manhattan distance under move-fail.
  - Catcher: maximize capture prob, then minimize distance; runner: minimize capture prob, then maximize distance; random tie-breaks.

- **MinimaxQAgent**
  - Tabular `Q[sid, a_c, a_r]`; per-state LP (`solve_row_player_maximin`) for catcher mixed strategy and value.
  - Catcher acts epsilon-greedy on the mixed policy; runner plays approximate best response (with epsilon exploration).
  - Update: `Q ← (1-α)Q + α(r + γ·v_next)`, marks state dirty to recompute policy on next access.
  - Epsilon anneals linearly (`eps_start`→`eps_end` over `eps_decay_episodes`).

## Logging details

- `ExcelLogger` captures per-step rows: run label, env params, episode/step, state before/after, actions (ids/names), reward, capture/done, move-fail flags, state ids, epsilon/value estimates, and agent metadata (mixed strategies, opponent models).
- Sheets are named by `run_label` (truncated to Excel’s 31-char limit).
- Training logs: `results/chase_train.xlsx`; Eval logs: `results/chase_eval.xlsx` (written at end of `chase.py` run).

## Analysis notebooks

- `analytics_train.ipynb`: training curves (capture rate/steps), action drift, return quantiles, heatmaps, trajectories, animations, capture-step histograms, distance decay, failure-rate effects.
- `analytics_eval.ipynb`: eval curves with env_seed-aware grouping (`eval_episode_id`), action drift, return quantiles, heatmaps, trajectories/animations, capture-step histograms, distance decay, failure-rate effects, and seeds variability.
- Helpers: `notebook/utils_train.py` and `utils_eval.py` load Excel logs, add derived fields, compute episode metrics, rolling stats, and plotting utilities.

### Tips when using the eval notebook

- Episodes reuse numbers per eval round/seed; uniqueness is `(run, p_fail, env_seed, episode)` and a derived `eval_episode_id` orders eval rounds.
- If counts look too low, rerun `chase.py` (current `eval_every=50`) to regenerate `chase_eval.xlsx`.

## Tuning / Extending

- Change grid size: adjust `TagEnv(size=...)` in training calls; state encoding/agents already handle size.
- Change movement model: edit `MOVE_DELTA` in `env/tag_env.py` (FP lookahead uses it too).
- Faster smoke tests: lower `episodes`/`eval_every` in `chase.py`.
- Disable logging: call train functions with `logger=None`/`eval_logger=None`.

## Key defaults (code)

- `TagEnv`: `size=5`, `p_fail=0.10`, `t_max=30`, `step_penalty=0.01`, `capture_reward=10.0`, `wall_penalty=0.02`, `move_reward=0.01`, `dist_reward=0.02`
- `FPAgent`: `prior=1e-3`, inherits `size`/`p_fail`
- `MinimaxQAgent`: `gamma=0.95`, `alpha=0.10`, `eps_start=0.20`, `eps_end=0.05`, `eps_decay_episodes=5000`
