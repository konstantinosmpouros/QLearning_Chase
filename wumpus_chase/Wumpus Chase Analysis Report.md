# Wumpus Chase Analysis Report (Comprehensive)

## 1. Project Scope and Design Goal

This project implements an extended two-agent Wumpus Chase environment and a full research-style training stack around it. The key design choice is fixed asymmetric roles:

- Agent A is always the **Chaser**.
- Agent B is always the **Runner**.

The codebase is organized so that environment dynamics, reward semantics, learning algorithms, orchestration logic, and logging are all modular and independently testable.

Primary objectives of the implementation are:

1. Preserve a clear competitive structure (A vs B).
2. Remove draw ambiguity by deterministic outcome priority.
3. Support multiple algorithm families under one unified CLI.
4. Provide rich logging for analysis and reproducibility.

---

## 2. Repository Architecture

Core tree:

```text
src/
  env/
    maps_extended.py
    wumpus_env_extended.py
  agents/
    fp_agent.py
    minimax_q_agent.py
    nash_q_agent.py
    dyna_q_agent.py
    dqn_agent.py
  train/
    common.py
    fp_training.py
    minimaxq_training.py
    nashq_training.py
    dynaq_training.py
    dqn_training.py
  cli/
    train.py
    eval_all.py
  logger/
    csv_logger.py
  utils/
    state.py
    lp.py
```

Conceptually:

- `env/*`: world mechanics, maps, reward shaping, terminal logic.
- `agents/*`: learning policies and update rules.
- `train/*`: matchup-specific training loops and periodic evaluation.
- `cli/*`: run orchestration, argument parsing, result export.
- `logger/*`: per-step data capture to CSV.
- `utils/*`: state encoding and LP helper for maximin strategy.

---

## 3. Environment Model (`env/wumpus_env_extended.py`)

### 3.1 Action Space and Dynamics

Action set:

- `ACTIONS = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]`
- Number of actions per player: `A = 5`

Movement deltas:

- `STAY -> (0,0)`
- `UP -> (+1,0)`
- `DOWN -> (-1,0)`
- `LEFT -> (0,-1)`
- `RIGHT -> (0,+1)`

Important coordinate convention:

- `x` increases downward (row index behavior).
- `y` increases to the right.

Stochastic actuation:

- Each action fails independently with probability `p_fail` (default `0.10`).
- On failure, the agent remains in place and does not incur wall/obstacle bump from that failure itself.

### 3.2 Perception Model

Perception tuple (`Perception`) includes:

- `breeze`: adjacent to pit.
- `stench`: adjacent to wumpus.
- `glitter`: on treasure cell.
- `bump`: attempted invalid movement (wall/obstacle).
- `scream`: currently always `False` in this version.

### 3.3 Observation / State Representation

Environment supports two observation modes via `partial_observable`:

1. Full observable (default):
   - state is `(ax, ay, bx, by)`.
2. Partial observable:
   - state is `(a_breeze, a_stench, b_breeze, b_stench)` as binary ints.

Internal episode state includes:

- current timestep `t`
- positions `a`, `b`
- precomputed free cells, breeze cells, stench cells

### 3.4 Reset Logic

At reset:

1. `t = 0`
2. A and B sample distinct start cells from free cells only.
3. Free cells exclude obstacles, pits, wumpus, and treasure.

This avoids trivial immediate deaths or instant treasure starts.

### 3.5 Terminal Outcome Priority (No Draw Runtime)

Outcome order is strict:

1. **Capture**: same cell or cross-path swap -> `A_WIN`.
2. **Treasure**: whoever reaches treasure wins.
   - If both reach in same step, tie-break favors A (`A_WIN`).
3. **Hazards**:
   - if B dies, `A_WIN`
   - else `B_WIN`
   - if both die, this rule still yields `A_WIN` (because B dead).
4. **Timeout** (`t >= t_max` with no prior outcome): `B_WIN`.

This design eliminates draw states in environment outcomes.

### 3.6 Reward Logic (Role-Aware, Decomposed)

The environment computes both rewards each step:

- `reward_a` for Chaser (A)
- `reward_b` for Runner (B)

But `step(...)` returns `(state, reward_a, done, info)` for backward compatibility. `reward_b` is exposed in `info`.

Default reward parameters:

- `step_penalty = 0.01`
- `outcome_reward = 10.0`
- `pit_penalty = 10.0`
- `wumpus_penalty = 10.0`
- `obstacle_penalty = 0.05`
- `chase_dist_reward = 0.04`
- `treasure_dist_reward = 0.03`
- `breeze_penalty = 0.02`
- `stench_penalty = 0.02`

Component structure:

1. Outcome rewards:
   - A win: `+outcome_reward` for A, `-outcome_reward` for B
   - B win: opposite
2. Step penalty:
   - Applied only to A when episode is ongoing (`outcome is None`).
3. Hazard penalties:
   - each agent penalized for own pit/wumpus contact.
4. Bump penalties:
   - own penalty if bumped wall/obstacle.
5. Perception penalties:
   - own penalty for standing in breeze/stench zones.
6. Distance shaping:
   - A gets positive if A-B distance decreases, negative if increases.
   - B gets positive if A-B distance increases, negative if decreases.
7. Treasure shaping:
   - each agent rewarded/punished by own treasure distance change.

Total rewards:

- `reward_a = outcome + step + hazard + bump + perception + chase + treasure`
- `reward_b = outcome + step + hazard + bump + perception + evade + treasure`

### 3.7 `info` Dictionary

`info` is intentionally rich and includes:

- outcome metadata (`outcome`, `capture`, `winner_role`, timestep)
- both agents' transition/hazard/perception flags
- distance metrics before/after
- full reward decomposition for A and B
- map context (wumpus, treasure, pits)

This is the primary bridge between environment and both learning/logging layers.

### 3.8 Rendering and Introspection

- `render_ascii()` visualizes world and dynamic positions.
- `get_extended_state()` exposes positions + current perceptions.

---

## 4. Map Layout System (`env/maps_extended.py`)

`ExtendedMapLayout` fields:

- `size`
- `obstacles`
- `wumpus`
- `pits`
- `treasure`

Derived utilities:

- adjacent cells (4-connectivity)
- `get_breeze_cells()`
- `get_stench_cells()`
- safe/free cell checks

Provided maps:

1. `small_layout()` (5x5): compact for faster iteration.
2. `default_extended_layout()` (7x7): balanced benchmark.
3. `dangerous_layout()` (7x7): higher pit pressure.

`visualize_layout()` provides static ASCII map with legend.

---

## 5. Utilities

### 5.1 State Encoding (`utils/state.py`)

`encode_state((ax,ay,bx,by), size)` maps joint position to integer id:

- `a_id = ax*size + ay`
- `b_id = bx*size + by`
- `sid = a_id*(size*size) + b_id`

`decode_state` performs inverse mapping.

This encoding is central for all tabular methods.

### 5.2 Maximin LP Solver (`utils/lp.py`)

`solve_row_player_maximin(Q)` solves row-player mixed strategy LP on payoff matrix `Q`:

- maximizes lower bound game value `v`
- constraints enforce simplex and security level vs every column action

Implementation uses SciPy `linprog(method="highs")`.
Fallback on LP failure:

- uniform policy
- value estimated from uniform.

Used directly by Minimax-Q and indirectly in robust action selection routines.

---

## 6. Logging System (`logger/csv_logger.py`)

`CSVLogger` writes one row per step and flushes on each write.

Data categories logged:

1. Run context:
   - `run_label`, `phase`, `agent_a`, `agent_b`, roles
2. Environment config snapshot:
   - `size`, `p_fail`, `t_max`, reward hyperparameters
3. Layout metadata:
   - treasure, wumpus, serialized obstacles/pits
4. Transition:
   - state/action/next_state raw coords + action names
5. Outcomes and hazards:
   - dead, treasure, fail, bump, pit, wumpus, capture, winner
6. Distances:
   - A-B and treasure distances before/after
7. Reward decomposition:
   - high-level + component values
8. Learning metadata:
   - encoded ids, epsilon, policies, values

Design note:

- If trainer provides extra keys (e.g., Nash mixed policies), logger overlays them when field exists.

---

## 7. Agent Implementations

## 7.1 FPAgent (`agents/fp_agent.py`)

Fictitious Play is implemented as belief-based one-step lookahead.

Core idea:

1. Maintain per-state opponent action counts `counts_opp[s, a_opp]` with small prior.
2. Convert counts to opponent policy estimate `opp_pi`.
3. For each candidate self action:
   - compute expected immediate score over opponent actions and stochastic move-fail outcomes.
4. Select highest expected score, tie-break with preferred expected distance behavior.

Important parameters/defaults:

- `p_fail=0.10`
- `step_penalty=0.01`
- `outcome_reward=5.0` (note: different from env default 10.0)
- `obstacle_penalty=0.05`
- `chase_dist_reward=0.04`
- `treasure_dist_reward=0.03`
- `prior=1e-3`

Modeling note:

- FP internal outcome approximation uses obstacle/wumpus/treasure/capture and distance shaping.
- It does not explicitly model pit-based penalties/perception penalties in its one-step expected metric.

## 7.2 MinimaxQAgent (`agents/minimax_q_agent.py`)

Tabular minimax-style learner with joint-action Q-table:

- `Q[s, a_row, a_col]`

Policy/value computation:

- Solve row-player maximin on `Q[s]` with LP helper.
- Cache policy/value per state with dirty-flag invalidation.

Exploration:

- linear epsilon decay from `eps_start=0.20` to `eps_end=0.05` over `eps_decay_episodes=5000`.

Action interfaces:

- `act_row(...)`: sample row policy.
- `act_col(...)`: robust column policy via maximin on `Q[s].T`.
- `best_response_col(...)`: explicit best response utility function.

Update:

- standard Q-learning target with next-state minimax value.

## 7.3 NashQAgent (`agents/nash_q_agent.py`)

General-sum stochastic game learner with two Q-tables:

- `Q_a[s, aA, aB]`
- `Q_b[s, aA, aB]`

Nash equilibrium methods (`NashType`):

- `LEMKE_HOWSON`
- `SUPPORT_ENUMERATION`
- `VERTEX_ENUMERATION`
- `FICTITIOUS_PLAY` (default in agent, chosen for speed)

If `nashpy` unavailable or solver fails:

- iterative best-response fallback or uniform fallback behaviors are used.

Caching and stability:

- per-state strategy and value caches
- NaN/Inf checks and normalization guards
- dirty flags mark states requiring recomputation

Exploration defaults:

- `eps_start=0.30`, `eps_end=0.05`, decay over `8000` episodes.

Update rule:

- each player updates toward own immediate reward plus discounted Nash value at next state.

Tracking:

- counts Nash computations/failures for diagnostics.

## 7.4 DynaQAgent and DynaQPlusAgent (`agents/dyna_q_agent.py`)

`DynaQAgent` combines:

1. Direct Q-learning on real transitions.
2. Learned model update.
3. `n_planning` simulated updates from model samples.

Key defaults:

- `gamma=0.95`
- `alpha=0.10`
- epsilon schedule: `0.30 -> 0.05` over `5000`
- `n_planning=10`
- `use_probabilistic_model=False`
- `player="A"` or `"B"` (role-sensitive value/action semantics)

Role-sensitive value:

- player A uses `max_a min_b Q`
- player B uses `max_b min_a Q`

Model options:

- Deterministic model stores last transition.
- Probabilistic model stores transition counts and average rewards.

`DynaQPlusAgent` extends DynaQ with exploration bonus during planning:

- bonus `kappa * sqrt(time_since_last_visit)`
- default `kappa=0.001`
- bonus affects simulated reward only, not real transition reward.

## 7.5 DQN Stack (`agents/dqn_agent.py`)

DQN is optional (PyTorch dependency).

Implemented architectures:

- Standard DQN
- Double DQN
- Dueling DQN
- Dueling Double DQN

`DQNConfig` defaults:

- `gamma=0.99`
- `learning_rate=1e-3`
- `batch_size=64`
- `eps_start=1.0`, `eps_end=0.05`, `eps_decay_steps=10000`
- `buffer_size=100000`
- `min_buffer_size=1000`
- `target_update_freq=100`
- `tau=1.0` (hard update)

`MultiAgentDQN` modes:

- `independent`: separate DQN per agent.
- `self_play`: shared DQN with flipped state perspective for B.
- `centralized`: one DQN with agent-indicator feature.

State features for DQN include normalized:

- positions
- A-B distance
- distance to treasure/wumpus for both agents
- breeze/stench indicators (if layout exposes those methods)

---

## 8. Training Engine and Matchup Logic

## 8.1 Shared Evaluation (`train/common.py`)

`run_episode`:

- executes policies in environment
- accumulates A-return
- optionally logs each step

`evaluate`:

- runs `n_episodes`
- computes `win_rate`, `draw_rate`, `avg_steps`, `avg_return`

Compatibility note:

- `draw_rate` is preserved in API and logs, although current environment outcome logic is no-draw.

## 8.2 FP-Centric Training (`train/fp_training.py`)

Functions:

- `train_fp_selfplay` (alias `train_fp_vs_fp`)
- `train_fp_vs_minimaxq`
- `train_fp_vs_nashq`
- `train_fp_vs_dynaq`

Pattern:

- FP usually controls A in these matchups.
- Opponent updates use `reward_b` where applicable.
- periodic evaluation every `eval_every`.

Defaults:

- FP selfplay/minimax defaults: `episodes=500`, `eval_every=1000`, `eval_episodes=10`
- FP vs Nash/Dyna: `episodes=10000`, `eval_every=1000`, `eval_episodes=20`

## 8.3 Minimax-Centric Training (`train/minimaxq_training.py`)

Functions:

- `train_minimaxq_vs_minimaxq` (alias `train_minimaxq_selfplay`)
- `train_minimaxq_vs_fp`
- `train_minimaxq_vs_nashq`
- `train_minimaxq_vs_dynaq`

Key behavior:

- row-side evaluation policy samples from mixed minimax policy.
- column-side often uses robust greedy/LP-based act_col.

Defaults:

- selfplay/vs_fp: `episodes=500`, `eval_every=1000`, `eval_episodes=10`
- vs_nashq/vs_dynaq: `episodes=10000`, `eval_every=1000`, `eval_episodes=20`

## 8.4 Nash-Centric Training (`train/nashq_training.py`)

Functions:

- `train_nashq_selfplay` (alias `train_nashq_vs_nashq`)
- `train_nashq_vs_fp`
- `train_nashq_vs_minimaxq`
- `train_nashq_vs_dynaq`

Details:

- selfplay logs per-state Nash policies and values into CSV extra fields.
- selfplay tracks aggregate Nash failure counts over time.

Defaults:

- generally `episodes=10000`, `eval_every=1000`, `eval_episodes=20`

## 8.5 Dyna-Centric Training (`train/dynaq_training.py`)

Functions:

- `train_dynaq_selfplay` (alias `train_dynaq_vs_dynaq`)
- `train_dynaq_vs_fp`
- `train_dynaq_vs_minimaxq`
- `train_dynaq_vs_nashq`
- `train_dynaq_plus_selfplay`
- `train_dynaq_plus_vs_fp`
- `train_dynaq_plus_vs_minimaxq`
- `train_dynaq_plus_vs_nashq`
- `train_dynaq_plus_vs_dynaq`
- utility: `compare_planning_steps`

Defaults:

- most Dyna variants use `episodes=10000`, `eval_every=1000`, `eval_episodes=20`, `n_planning=10`.
- DynaQ+ adds `kappa=0.001`.

Dyna selfplay logs include extra model-planning diagnostics:

- `model_size`
- `planning_ratio`

## 8.6 DQN Training (`train/dqn_training.py`)

Functions:

- `train_dqn_selfplay`
- `train_dqn_vs_dqn`
- `train_dqn_vs_opponent`
- `evaluate_dqn`
- `create_dqn_policy`

Defaults:

- `train_dqn_selfplay`: `episodes=5000`, `eval_every=1000`, `eval_episodes=50`
- `train_dqn_vs_dqn`: `episodes=5000`, `eval_every=1000`, `eval_episodes=50`
- `train_dqn_vs_opponent`: `episodes=5000`, `eval_every=1000`, `eval_episodes=50`

Note:

- DQN training functions maintain their own logs; they are not step-logged via `CSVLogger` in current CLI dispatcher.

---

## 9. Unified CLI Behavior (`cli/train.py`)

Main arguments and defaults:

- `--matchup` (default `all`)
- `--episodes` (`3000`)
- `--eval-every` (`100`)
- `--eval-episodes` (`20`)
- `--suite` (`quick` or `full`, default `quick`)
- `--seed` (`42`)
- `--layout` (`small|default|dangerous`, default `small`)
- `--p-fail` (`0.10`)
- `--t-max` (optional override, else auto: small->40, else 50)
- `--planning-steps` (`10`)
- `--dqn-type` (`standard|double|dueling|dueling_double`, default `dueling_double`)
- `--results-dir` (default `results`)
- `--excel-output` (optional)
- `--quiet`

Built-in matchup presets:

- `QUICK_MATCHUPS`:
  - `fp_selfplay`
  - `minimaxq_selfplay`
  - `nashq_selfplay`
  - `dynaq_selfplay`
  - `nashq_vs_minimaxq`

- `FULL_MATCHUPS`:
  - all core FP/Minimax/Nash/Dyna scenarios
  - DynaQ+ scenarios (selfplay and vs FP/Minimax/Nash/Dyna)

Conditional behavior:

- If `--matchup all --suite full` and DQN is available, `dqn_vs_dqn` is appended automatically.
- Direct `dqn_vs_dqn` dispatch caps episodes to `min(episodes, 5000)`.

Output artifacts:

1. CSV train log
2. CSV eval log
3. Excel workbook (sheet per matchup)
4. console summary table with `Win Rate`, `RunnerWin`, `Avg Steps`, `Avg Return`

Wrapper entry points:

- `src/wumpus_chase_extended.py`
- `src/cli/eval_all.py`

Both delegate to `cli/train.py`.

---

## 10. Current Matchup Inventory

Public CLI choices currently include:

- `fp_selfplay`, `fp_vs_fp`, `fp_vs_minimaxq`, `fp_vs_nashq`, `fp_vs_dynaq`
- `minimaxq_selfplay`, `minimaxq_vs_fp`, `minimaxq_vs_nashq`, `minimaxq_vs_dynaq`
- `nashq_selfplay`, `nashq_vs_fp`, `nashq_vs_minimaxq`, `nashq_vs_dynaq`
- `dynaq_selfplay`, `dynaq_vs_fp`, `dynaq_vs_minimaxq`, `dynaq_vs_nashq`
- `dynaq_plus_selfplay`, `dynaq_plus_vs_fp`, `dynaq_plus_vs_minimaxq`, `dynaq_plus_vs_nashq`, `dynaq_plus_vs_dynaq`
- `dqn_vs_dqn` (optional dependency path)

---

## 11. Dependencies and Optional Components

From `requirements.txt`:

- Core:
  - `numpy`, `scipy`, `pandas`, `openpyxl`
- Game-theoretic solver:
  - `nashpy`
- Visualization / notebook support:
  - `matplotlib`, `plotly`
- Optional deep RL:
  - `torch`

Behavior without optional dependencies:

- Without `torch`: DQN code paths are unavailable.
- Without `nashpy`: Nash-Q falls back to approximate internal solver.

---

## 12. Important Implementation Notes and Caveats

1. **No-draw environment vs draw-compatible metrics**
   - Environment emits `A_WIN` or `B_WIN` (plus in-progress) under current rules.
   - `draw_rate` remains for compatibility and should typically stay near `0.0`.

2. **Reward interface compatibility**
   - `step()` returns only A reward directly.
   - B reward must be pulled from `info["reward_b"]`.
   - Training loops correctly do this across algorithm families.

3. **Partial observability and state encoding**
   - `encode_state` assumes 4-tuple integer state and is position-oriented.
   - In partial observable mode, state semantics differ (binary perception tuple), so tabular indexing interpretation changes.

4. **FP internal one-step model is approximate**
   - FP expected metric includes key movement/outcome terms but is not a full environment simulator with all penalties.

5. **Per-step CSV flushing**
   - Logger flushes each write for robustness.
   - This is safer for long runs but can increase I/O overhead.

---

## 13. End-to-End Dataflow Summary

One training step in the unified stack:

1. Trainer obtains encoded state from environment state tuple.
2. Agent A policy picks `a1`; Agent B policy picks `a2`.
3. Environment advances one simultaneous step and computes:
   - next state
   - `reward_a`
   - terminal flag
   - full `info` dictionary including `reward_b`
4. Trainers update each learner with role-correct reward.
5. Logger (if enabled for that trainer path) writes full transition row.
6. At evaluation checkpoints, deterministic (or greedy) policies are tested and summary metrics are appended.

This pipeline is consistent across FP/Minimax/Nash/Dyna families and intentionally unified through shared interfaces (`encode_state`, `evaluate`, `CSVLogger`, CLI dispatch).
