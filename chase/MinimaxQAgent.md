# MinimaxQAgent

Tabular Minimax-Q agent for the simultaneous-move tag game, treating the catcher as the row player in a zero-sum matrix game at each state.

## Core idea

- Maintain `Q[sid, a_c, a_r]` for catcher (row) and runner (column).
- At each state, solve a linear program to find the catcher’s maximin mixed strategy `pi` and state value `v = max_pi min_a_r pi^T Q[:, a_r]`.
- Runner acts as an approximate best response to `pi` (minimizes expected Q), with optional epsilon exploration.

## Policy computation

1. **Dirty flag** per state: when any `Q[sid, :, :]` updates, mark dirty.
2. **LP solve** (`solve_row_player_maximin`):
   - Variables: `pi[a]` (row probabilities) and scalar value `v`.
   - Constraints: `sum_a pi[a] Q[a, b] >= v` for all runner actions `b`; `sum_a pi[a] = 1`; `pi[a] >= 0`.
   - Objective: maximize `v` (implemented as minimize `-v`).
3. Cache `pi_cache[sid]` and `v_cache[sid]` when solved; reuse until state is dirty again.

## Action selection

- **Catcher** (`act_catcher`):
  - Get `(pi, v)` via LP (cached).
  - Epsilon-greedy: with probability `eps` choose a random action; else sample from `pi`.
- **Runner** (`best_response_runner`):
  - Compute expected values `exp_vals = pi @ Q[sid]` (vector over runner actions).
  - Choose argmin `exp_vals`; with probability `eps` pick a random action.

## Learning update

Given transition `(sid, a_c, a_r, r, sid_next, done)`:

1. Target: `r` if `done` else `r + gamma * v_next`, where `v_next` is the cached value of `sid_next` (solved via LP if dirty).
2. Update: `Q[sid, a_c, a_r] ← (1 - alpha) * Q + alpha * target`.
3. Mark `sid` dirty so its policy/value are recomputed next time.

## Exploration

- Epsilon anneals linearly: `eps = (1 - t) * eps_start + t * eps_end`, with `t = min(1, episode / eps_decay_episodes)`.
- Applies to catcher’s action and runner’s best-response sampling.

## Defaults (in code)

- `gamma=0.95`, `alpha=0.10`, `eps_start=0.20`, `eps_end=0.05`, `eps_decay_episodes=5000`, `size=5`.

## Behavior notes

- LP solves give principled mixed strategies; caching avoids repeated solves when `Q` is unchanged.
- Runner’s approximate best response avoids solving a second LP; good enough for this small game.
- Sensitive to `alpha`/`eps_decay`: larger `alpha` learns faster but can oscillate; slower decay keeps exploration longer.
- Uses encoded state IDs (`encode_state`) so grid size changes are already handled if `size` matches the environment.
