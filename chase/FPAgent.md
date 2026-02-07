# FPAgent

Tabular fictitious-play agent with a 1-step best-response heuristic in the simultaneous-move tag game.

## Role-aware objective

- **Catcher**: maximize capture probability, break ties by minimizing expected Manhattan distance.
- **Runner**: minimize capture probability, break ties by maximizing expected Manhattan distance.

## Inputs and state

- Grid size `size` (default 5), number of states = `(size*size)^2`.
- Opponent move-failure probability `p_fail` is considered when evaluating outcomes.
- Empirical opponent counts: `counts_opp[sid, action]` with a small Dirichlet prior (`prior=1e-3`) to avoid zero-probability actions.

## Action selection

1. Encode state `(cx, cy, rx, ry)` to integer `sid`.
2. Build opponent policy `opp_pi = counts_opp[sid] / sum`.
3. For each self action `a_self`:
   - For each opponent action `a_opp`:
     - Enumerate the four move-failure combinations for both players.
     - Compute next positions using `MOVE_DELTA`, respecting grid bounds and failures.
     - Mark capture if they land on the same cell or cross paths.
     - Accumulate capture probability and expected Manhattan distance weighted by outcome probability.
   - Score = `(cap_prob, -exp_dist)` for catcher, `(-cap_prob, exp_dist)` for runner.
4. Choose the best score lexicographically, if multiple tie, pick uniformly at random.

## Learning signal

- **observe(s, opp_action)**: increment opponent action count for the visited state.
- No value function, policy is purely best-response to empirical frequencies.

## Stochasticity

- Only in tie-breaking, otherwise deterministic given `counts_opp`.
- Tie-breaking uses `random.Random(seed)`; no epsilon exploration.

## Behavior notes

- Sensitive to state coverage: early episodes rely on the Dirichlet prior, behavior stabilizes as counts grow.
- Incorporates `p_fail` in lookahead, so planned moves hedge against failed moves.
- Distance tie-break helps the catcher close gaps and the runner keep distance when capture odds are equal.

## Key parameters

- `size=5`, `p_fail=0.10`, `prior=1e-3`, RNG `seed` for tie-breaking.
