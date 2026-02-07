# Chase (Simultaneous-Move Tag) Detailed Report

Scope: This report covers only the `chase/` directory (not `wumpus_chase/`). It documents the environment, rules, reward design, agent policies, training/evaluation procedure, parameters, logging, and how the program is executed.

## 1) Directory map and responsibilities

- `chase/src/env/tag_env.py`:
  - Core environment `TagEnv` and constants `ACTIONS`, `MOVE_DELTA`, `A` (number of actions).
  - Rules: movement, stochastic failure, capture conditions, reward shaping, termination.
- `chase/src/utils/state.py`:
  - `encode_state` for tabular state indexing.
- `chase/src/utils/lp.py`:
  - Linear program (LP) for row-player maximin mixed strategy used by Minimax-Q.
- `chase/src/agents/fp_agent.py`:
  - Fictitious Play agent with empirical opponent model and one-step best response.
- `chase/src/agents/minimax_q_agent.py`:
  - Minimax-Q agent (tabular) with LP-based mixed policy and Q-learning updates.
- `chase/src/train/common.py`:
  - `run_episode` and `evaluate` shared by all training loops.
- `chase/src/train/fp_vs_fp.py`:
  - Training loop for FP vs FP.
- `chase/src/train/minimaxq_selfplay.py`:
  - Training loop for Minimax-Q self-play.
- `chase/src/train/minimaxq_vs_minimaxq.py`:
  - Training loop for Minimax-Q vs Minimax-Q (two separate agents).
- `chase/src/train/minimaxq_vs_fp.py`:
  - Training loop for Minimax-Q vs FP.
- `chase/src/logger/excel_logger.py`:
  - `ExcelLogger` for step-level logging into `.xlsx` files.
- `chase/src/chase.py`:
  - Entry point wiring all matchups and saving logs.
- `chase/FPAgent.md`, `chase/MinimaxQAgent.md`:
  - Agent-specific documentation.
- `chase/notebook/`:
  - Analysis notebooks and utilities for plotting and metrics.
- `chase/results/`:
  - Generated outputs: `chase_train.xlsx`, `chase_eval.xlsx`.

## 2) Environment specification (TagEnv)

### 2.1 State, grid, and coordinates

- Grid size: `size=5` by default. Coordinates are `(x, y)` with:
  - `x` increasing "up" (row index).
  - `y` increasing "right" (column index).
- State tuple: `(catcher_x, catcher_y, runner_x, runner_y)`.
- Number of states: `(size * size) * (size * size)`.
  - With size=5: 25 * 25 = 625 states.

### 2.2 Actions and movement

- Actions (index -> name):
  - `0: STAY`  -> delta `(0, 0)`
  - `1: UP`    -> delta `(1, 0)` (increases x)
  - `2: DOWN`  -> delta `(-1, 0)` (decreases x)
  - `3: LEFT`  -> delta `(0, -1)`
  - `4: RIGHT` -> delta `(0, 1)`
- Movement is clamped to the grid boundary:
  - `nx = min(size-1, max(0, x + dx))`
  - `ny = min(size-1, max(0, y + dy))`
- Important detail: attempting to move outside the grid does not change position (due to clamp), but is still counted as "hit wall" for reward penalties.

### 2.3 Stochastic action failure

- Each agent independently fails to move with probability `p_fail` each step.
- Failure model:
  - If fail: position stays the same regardless of action.
  - Failure is sampled from `rng.random() < p_fail`.

### 2.4 Capture conditions

Capture occurs if either of these is true after applying actions (and failures):

1) They land on the same cell: `c_new == r_new`.
2) They cross in the same step: `c_new == r_old` and `r_new == c_old`.

Crossing capture means a swap of positions counts as capture even if they do not share a cell at the end.

### 2.5 Episode termination

- The environment time counter `t` increments at each `step`.
- Episode ends if:
  - Capture occurs, or
  - `t >= t_max` (default `t_max=30`).

### 2.6 Reward function and shaping

The environment computes a rich shaped reward for both agents, but only the catcher's reward is returned as the step reward. The runner reward is included in the `info` dictionary.

Parameters (defaults):

- `step_penalty = 0.01`
- `capture_reward = 10.0`
- `wall_penalty = 0.02`
- `move_reward = 0.01`
- `dist_reward = 0.02`

Reward components per step:

- Capture:
  - Catcher: `+capture_reward`
  - Runner: `-capture_reward`
- Step penalty:
  - Catcher: `-step_penalty` if not capture
  - Runner: no step penalty term (remains 0)
- Wall penalty (attempted out-of-bounds action):
  - Catcher: `-wall_penalty` if `would_hit_wall`
  - Runner: `-wall_penalty` if `would_hit_wall`
- Move reward:
  - Catcher: `+move_reward` if actually moved
  - Runner: `+move_reward` if actually moved
- Distance shaping (Manhattan distance):
  - If distance decreases:
    - Catcher: `+dist_reward`
    - Runner: `-dist_reward`
  - If distance increases:
    - Catcher: `-dist_reward`
    - Runner: `+dist_reward`
  - If distance unchanged: no distance reward

Total catcher reward is the sum of all its components, and the runner reward is the sum of the runner components. Note that the reward is not strictly zero-sum due to asymmetric terms (e.g., step penalty only for catcher, both get move reward and wall penalty in the same direction).

### 2.7 Info dictionary (per step)

The environment returns a detailed `info` dictionary with:

- `capture`, `t` (time step), `c_fail`, `r_fail`
- Positions: `c_old`, `r_old`, `c_new`, `r_new`
- Runner reward: `reward_runner`
- Distances: `dist_before`, `dist_after`, `dist_delta`
- Wall and movement flags: `c_hit_wall`, `r_hit_wall`, `c_moved`, `r_moved`
- Reward components for both agents:
  - `reward_capture_c`, `reward_capture_r`
  - `reward_step_c`, `reward_step_r`
  - `reward_wall_c`, `reward_wall_r`
  - `reward_move_c`, `reward_move_r`
  - `reward_dist_c`, `reward_dist_r`

### 2.8 Randomness and seeding

- `TagEnv` uses `random.Random(seed)`.
- The same RNG is used for both initial position sampling and per-step move failures.
- `seed` is stored in the environment and logged for each step.

## 3) State encoding and indexing

`encode_state(s, size)` maps `(cx, cy, rx, ry)` to a single integer index:

- `c_id = cx * size + cy`
- `r_id = rx * size + ry`
- `sid = c_id * (size * size) + r_id`

This matches a flat table of size `(size^2) * (size^2)` for tabular arrays like `Q[sid, a_c, a_r]`.

## 4) Agents

### 4.1 FPAgent (Fictitious Play)

Location: `chase/src/agents/fp_agent.py`

Purpose:

- Uses a per-state empirical opponent action distribution and a one-step lookahead to choose actions.

Key parameters:

- `role`: `"catcher"` or `"runner"`
- `size=5`
- `p_fail=0.10` (used in expected outcomes)
- `prior=1e-3` (Dirichlet prior over opponent actions)
- `seed` for tie-breaking RNG

Internal state:

- `counts_opp[sid, action]` initialized to `prior` for every action.
- The agent updates counts with `observe(s, opp_action)`.

Action selection (`act`):

1) Encode state to `sid`.
2) Compute opponent policy `opp_pi` by normalizing `counts_opp[sid]`.
3) For each candidate action `a_self`:
   - For each opponent action `a_opp`:
     - Enumerate four failure cases: (c_fail, r_fail) in {(F,F), (T,F), (F,T), (T,T)}.
     - Compute next positions using `MOVE_DELTA` and grid clamp.
     - Compute capture condition (same cell or cross).
     - Accumulate:
       - capture probability
       - expected Manhattan distance (if no capture)
   - Aggregate outcomes weighted by opponent action probabilities.
4) Scoring:
   - Catcher maximizes `(capture_prob, -expected_distance)`.
   - Runner maximizes `(-capture_prob, expected_distance)`.
5) Tie-breaking: randomly pick among the lexicographically best actions.

Notes:

- The agent has no explicit value function and no epsilon exploration.
- Stochasticity is only from tie-breaking.
- The FP strategy directly accounts for move-failure probability `p_fail`.

### 4.2 MinimaxQAgent

Location: `chase/src/agents/minimax_q_agent.py`

Purpose:

- Tabular Minimax-Q for a simultaneous-move game with the catcher as the row player.
- Uses linear programming to compute the catcher's maximin mixed strategy at each state.

Key parameters:

- `size=5`
- `gamma=0.95` (discount factor)
- `alpha=0.10` (learning rate)
- `eps_start=0.20`, `eps_end=0.05`, `eps_decay_episodes=5000`
- `seed` for internal RNG (`random.Random`)

Internal state:

- `Q[sid, a_c, a_r]` initialized to zeros.
- `pi_cache[sid, :]` and `v_cache[sid]` for cached LP solutions.
- `dirty[sid]` indicates cached values are stale and need recomputation.

Epsilon schedule:

- `t = min(1, episode / eps_decay_episodes)`
- `epsilon = (1 - t) * eps_start + t * eps_end`

Policy computation:

- If `dirty[sid]` is true:
  - Solve the LP in `utils/lp.py` to obtain:
    - `pi` (catcher mixed strategy)
    - `v` (maximin value)
  - Cache results and clear `dirty[sid]`.

Catcher action (`act_catcher`):

- With probability `eps`: choose a random action.
- Otherwise: sample from `pi`.
- Note: sampling from `pi` uses `np.random.choice`, which relies on the global NumPy RNG (not seeded inside the class).

Runner action (`best_response_runner`):

- With probability `eps`: choose a random action.
- Otherwise: compute `exp_vals = pi @ Q[sid]` (expected catcher payoff for each runner action) and take the argmin.

Q update:

- Target:
  - If `done`: `target = r`
  - Else: `target = r + gamma * v_next`, where `v_next` is the next-state maximin value.
- Update:
  - `Q[sid, a_c, a_r] = (1 - alpha) * Q + alpha * target`
- Mark `dirty[sid] = True`.

LP solver details (`utils/lp.py`):

- Objective: maximize `v` subject to `sum_a pi[a] * Q[a, b] >= v` for all runner actions `b`.
- Constraints: `sum_a pi[a] = 1`, `pi[a] >= 0`.
- Solved via `scipy.optimize.linprog` (`method="highs"`).
- If the LP fails, a uniform policy is returned.

## 5) Training and evaluation procedure

### 5.1 Shared utilities (`train/common.py`)

`run_episode(env, catcher_policy, runner_policy, ...)`:

- Resets the environment and steps up to `env.t_max`.
- For each step:
  - Calls provided policies to obtain actions.
  - Calls `env.step(a_c, a_r)` to get next state and reward.
  - Optionally logs to `ExcelLogger` with `phase="eval"` if invoked from evaluation.
- Returns `(captured_flag, steps_taken, cumulative_reward)`.

`evaluate(env, catcher_policy, runner_policy, n_episodes=200, ...)`:

- Runs `run_episode` multiple times (default in training loops is 10 episodes).
- Computes:
  - `capture_rate`: fraction of episodes with capture.
  - `avg_steps_to_capture`: mean steps for captured episodes (NaN if none captured).
  - `avg_return`: mean cumulative reward over all episodes.

### 5.2 FP vs FP (`train_fp_vs_fp`)

- Environment: `TagEnv(p_fail=p_fail, seed=seed)`.
- Agents: `FPAgent("catcher")` and `FPAgent("runner")`.
- Per step:
  - Each agent acts using its FP policy.
  - Each agent observes the opponent action to update empirical counts.
- Logging:
  - Logs opponent model distributions `opp_pi_catcher` and `opp_pi_runner`.
- Evaluation:
  - Every `eval_every` episodes (default 50 in `chase.py`).
  - `evaluate` runs 10 episodes in a new env with seed `seed + 10_000 + ep`.

### 5.3 Minimax-Q self-play (`train_minimaxq_selfplay`)

- Environment: `TagEnv(p_fail=p_fail, seed=seed)`.
- Agent: a single `MinimaxQAgent` controls both sides:
  - Catcher uses `act_catcher` with epsilon.
  - Runner uses `best_response_runner` to catcher's mixed policy.
- Per step:
  - Update Q with catcher's reward (returned by env).
- Evaluation:
  - Every `eval_every` episodes.
  - New env with seed `seed + 20_000 + ep`.
  - Catcher samples from policy `pi`.
  - Runner uses best response with `eps=0` (no exploration).

### 5.4 Minimax-Q vs FP (`train_minimaxq_vs_fp`)

- Environment: `TagEnv(p_fail=p_fail, seed=seed)`.
- Agents:
  - Catcher: MinimaxQ
  - Runner: FP
- Per step:
  - MinimaxQ chooses action using epsilon.
  - FP runner chooses action using its empirical model.
  - FP runner observes the catcher action to update counts.
  - MinimaxQ updates `Q` using catcher's reward.
- Evaluation:
  - Every `eval_every` episodes.
  - New env with seed `seed + 30_000 + ep`.
  - Catcher samples from `pi`.
  - Runner uses FP policy.

### 5.5 Minimax-Q vs Minimax-Q (`train_minimaxq_vs_minimaxq`)

- Environment: `TagEnv(p_fail=p_fail, seed=seed)`.
- Agents:
  - `q_c` for catcher, `q_r` for runner.
- Per step:
  - Both agents use `act_catcher` (each as a row player of its own Q table).
  - Catcher updates with env reward `r`.
  - Runner updates with `reward_runner` from `info`.
- Evaluation:
  - Every `eval_every` episodes.
  - New env with seed `seed + 40_000 + ep`.
  - Both sides sample from their respective `pi`.

## 6) Entry point: how the program runs

`chase/src/chase.py` is the main script:

1) Adds `chase/src` to `sys.path` so relative imports work.
2) Creates loggers:
   - `results/chase_train.xlsx`
   - `results/chase_eval.xlsx`
3) Loops over `p_fail_values = [0.10, 0.20]`.
4) Runs four matchups (each for 10,000 episodes, eval every 50):
   - `fp_vs_fp` (seed=0)
   - `minimaxq_selfplay` (seed=1)
   - `minimaxq_vs_minimaxq` (seed=3)
   - `minimaxq_vs_fp` (seed=2)
5) Prints final evaluation metrics to console.
6) Saves both Excel logs.

Run command:

```bash
python3 chase/src/chase.py
```

Dependencies (from `requirements.txt`):

- `numpy`, `scipy`, `pandas`, `openpyxl`, `plotly`, `matplotlib`

## 7) Logging and outputs

### 7.1 ExcelLogger

- Keeps all step-level rows in memory and writes on `save()`.
- Paths are resolved relative to `chase/` and directories are created if missing.
- Each run uses a separate Excel sheet named by `run_label` (truncated to 31 chars for Excel).

### 7.2 Logged fields (per step)

The logger stores:

- Run and environment metadata:
  - `run`, `p_fail`, `env_size`, `env_t_max`, `env_step_penalty`, `env_capture_reward`,
    `env_wall_penalty`, `env_move_reward`, `env_dist_reward`, `env_seed`
- Episode and step info:
  - `episode`, `step`, `env_t`, `sid`, `sid_next`
- State and action info:
  - `catcher_x`, `catcher_y`, `runner_x`, `runner_y`
  - `catcher_action_id`, `runner_action_id`, `catcher_action`, `runner_action`
  - `catcher_x_next`, `catcher_y_next`, `runner_x_next`, `runner_y_next`
- Rewards and terminal flags:
  - `reward`, `reward_runner`, `capture`, `done`
  - Reward breakdown fields (capture/step/wall/move/dist for both agents)
- Movement and distance diagnostics:
  - `c_fail`, `r_fail`, `dist_before`, `dist_after`, `dist_delta`
  - `c_hit_wall`, `r_hit_wall`, `c_moved`, `r_moved`
- Agent-specific extras (vary by run):
  - FP: `opp_pi_catcher`, `opp_pi_runner`
  - MinimaxQ: `pi_catcher`, `pi_runner`, `epsilon`, `value_estimate`, etc.
  - Eval phase adds `phase="eval"` and `eval_at_episode` to track evaluation rounds.

### 7.3 Output files

- Training steps: `chase/results/chase_train.xlsx`
- Evaluation steps: `chase/results/chase_eval.xlsx`

## 8) Notebook analysis utilities

The notebook utilities provide structured post-processing and visualization of the Excel logs.

- `notebook/utils_train.py`:
  - Loads and merges run sheets from `chase_train.xlsx`.
  - Adds derived columns (Manhattan distance, categorical actions).
  - Computes episode-level metrics (capture, steps, return).
  - Rolling metrics: capture rate, steps, return quantiles.
  - Action mix and drift plots.
  - Trajectory plots, heatmaps, and animations.

- `notebook/utils_eval.py`:
  - Similar tools for `chase_eval.xlsx`.
  - Handles repeated episode numbers across different eval seeds by creating `eval_episode_id`.
  - Rolling metrics indexed by `eval_episode_id` for monotonic evaluation trends.

These notebooks are optional but document how training and evaluation outputs are analyzed.

## 9) Parameter summary (defaults)

### Environment (TagEnv)

- `size = 5`
- `p_fail = 0.10`
- `t_max = 30`
- `step_penalty = 0.01`
- `capture_reward = 10.0`
- `wall_penalty = 0.02`
- `move_reward = 0.01`
- `dist_reward = 0.02`
- `seed = 0`

### FPAgent

- `role = "catcher"` or `"runner"`
- `size = 5`
- `p_fail = 0.10`
- `prior = 1e-3`
- `seed = 0` (tie-breaking RNG)

### MinimaxQAgent

- `size = 5`
- `gamma = 0.95`
- `alpha = 0.10`
- `eps_start = 0.20`
- `eps_end = 0.05`
- `eps_decay_episodes = 5000`
- `seed = 0` (random.Random used for epsilon randomization)

### Training (as wired in `chase.py`)

- `p_fail_values = [0.10, 0.20]`
- Episodes per matchup: 10,000 each
- Evaluation cadence: every 50 episodes
- Evaluation episodes per checkpoint: 10
- Seeds:
  - FP vs FP: seed=0 (eval env seed = 10_000 + ep)
  - MinimaxQ self-play: seed=1 (eval env seed = 20_000 + ep)
  - MinimaxQ vs MinimaxQ: seed=3 (eval env seed = 40_000 + ep)
  - MinimaxQ vs FP: seed=2 (eval env seed = 30_000 + ep)

## 10) Practical notes and implications

- The environment reward is not strictly zero-sum, but Minimax-Q is used as if it were. This is a modeling choice; it still yields a robust policy but may not be the exact game-theoretic equilibrium for the shaped reward.
- Wall penalties are applied based on the intended action, not the actual (potentially failed) movement, so even failed moves can be penalized if the action would have hit a wall.
- Only the catcher's step penalty is applied; the runner does not receive a step penalty. This biases the game toward faster capture by penalizing the catcher for long episodes without capture.
- Minimax-Q uses both `random.Random` and NumPy's global RNG (for sampling from `pi`), so full reproducibility requires seeding NumPy as well if determinism is desired.

End of report.
