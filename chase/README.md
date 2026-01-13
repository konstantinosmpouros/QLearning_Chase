# Chase: Tag Game with FP and Minimax-Q

This folder contains a lightweight simultaneous-move tag environment (catcher vs runner) and two agent types: Fictitious Play (FP) and Minimax-Q. It trains quick self-play and mixed matchups, logs every step to Excel, and prints summary stats.

## Components

- `env.py` — environment definition and movement constants (`ACTIONS`, `MOVE_DELTA`, `A`, `TagEnv`).
- `utils.py` — helpers (`encode_state`, `solve_row_player_maximin` LP solver).
- `agents/` — `fp_agent.py` (`FPAgent`, empirical opponent model + 1-step best response) and `minimax_q_agent.py` (`MinimaxQAgent`, tabular zero-sum RL with LP-derived policy); re-exported from `agents/__init__.py`.
- `train.py` — evaluation and training loops: FP vs FP, MinimaxQ self-play, MinimaxQ vs FP; `last_or_nan`.
- `logger.py` — `ExcelLogger` for per-step logging to `chase_runs.xlsx`.
- `chase.py` — entrypoint that wires training runs, prints metrics, and saves the Excel.

## Environment

- Grid: default 5×5.
- State tuple: `(catcher_x, catcher_y, runner_x, runner_y)`.
- Actions: `["STAY", "UP", "DOWN", "LEFT", "RIGHT"]`.
  - Movement deltas (`MOVE_DELTA`): `UP=(+1,0)`, `DOWN=(-1,0)`, `LEFT=(0,-1)`, `RIGHT=(0,+1)`. Moves clamp to grid edges.
  - Each move can independently **fail** with probability `p_fail`; on failure the agent stays put.
  - Capture if catcher and runner land on the same cell **or** cross cells in the same step.
- Reward (to catcher): `+1` on capture, `-step_penalty` otherwise; episode ends on capture or after `t_max` steps.
- Reset: catcher and runner start at random distinct cells.

## Agents

### FPAgent (Fictitious Play, `agents.py`)

- Tracks empirical opponent action counts per state (Dirichlet prior `prior`).
- For each action, computes expected capture probability and expected Manhattan distance given the opponent distribution and move-failure.
- Catcher maximizes capture prob then minimizes distance; runner minimizes capture prob then maximizes distance; random tie-break.

### MinimaxQAgent (Zero-sum RL, `agents.py`)

- Tabular `Q[sid, a_c, a_r]`.
- For each state, solves a linear program (`solve_row_player_maximin`) to get the catcher’s maximin mixed strategy `pi` and value `v`. Runner plays an approximate best response to `pi`.
- Epsilon-greedy exploration on the catcher; runner also explores with epsilon inside `best_response_runner`.
- Update: `Q[s,a_c,a_r] ← (1-α)Q + α(r + γ·v_next)`; marks state dirty to recompute policy next time.
- Epsilon anneals linearly from `eps_start` to `eps_end` over `eps_decay_episodes`.

## Training Loops (`train.py`)

- `train_fp_vs_fp`: FP catcher vs FP runner.
- `train_minimaxq_selfplay`: MinimaxQ for both roles (runner best-responds).
- `train_minimaxq_vs_fp`: MinimaxQ catcher vs FP runner.
- `evaluate`: runs short evaluations; `run_episode` simulates a single episode.
- `last_or_nan`: convenience extractor.

Each loop accepts:

- `p_fail`, `episodes`, `eval_every`, `seed`.
- `logger` (`ExcelLogger` or `None`) and `run_label` to name the Excel sheet.

## Logging (`logger.py`)

- `ExcelLogger` collects per-step rows in memory and writes to `chase_runs.xlsx` (next to this folder).
- Columns include run label, env params, episode/step, state before/after, actions (ids and names), reward, capture/done, move-fail flags, state ids, epsilon/value estimates, and agent metadata (mixed strategies, opponent models where applicable).
- Sheets: one per `run_label` (max 31 chars per Excel rules).

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r chase_requirements.txt  # in this folder
python3 -m chase.chase                # or: python3 chase.py from this folder
```

Default training (in `chase.py`):

- `p_fail_values = [0.10, 0.20]`
- Episodes: FP 2,000; MinimaxQ self-play 10,000; MinimaxQ vs FP 10,000
- Eval cadence: FP 200; MinimaxQ 1,000; mixed 1,000

Outputs:

- Console prints final capture rate, steps, and return for each matchup/p_fail.
- Excel: `chase_runs.xlsx` with all per-step data.

## Movement Examples

- Start `(catcher=(2,1), runner=(4,3))`, `UP` → `(3,1)` (x+1); `DOWN` → `(1,1)` (x-1); `RIGHT` → `(2,2)`; `LEFT` → `(2,0)`.
- At top edge (`x=0`), a `DOWN` (which decreases x) stays at `x=0` due to clamping.
- With `p_fail=0.1`, each move has 10% chance to stay in place regardless of intended direction.

## Simulation Parameters (key defaults)

- `TagEnv`: `size=5`, `p_fail=0.10`, `t_max=30`, `step_penalty=0.01`.
- `FPAgent`: `prior=1e-3`, `size=5`, `p_fail` inherited.
- `MinimaxQAgent`: `gamma=0.95`, `alpha=0.10`, `eps_start=0.20`, `eps_end=0.05`, `eps_decay_episodes=5000`.

Adjust these in code or add CLI parsing in `chase.py` if needed.

## Extending / Notes

- Change grid size: update `TagEnv(size=...)` in training calls; `encode_state` and agents already handle `size`.
- Change movement convention: edit `MOVE_DELTA` in `env.py`; FP’s lookahead uses the same table.
- Logging off: call train functions with `logger=None`.
- Faster smoke tests: reduce episode counts in `chase.py` before running.
