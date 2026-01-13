"""
Lightweight Tag Game implementation for FP vs Minimax-Q.

This script defines:
  - A 5x5 Tag environment (catcher vs runner).
  - Fictitious play agent using a simple 1‑step lookahead best response.
  - Minimax‑Q agent for zero‑sum RL using only the row player's mixed strategy
    (the column player plays a best response to avoid solving an LP).

Training loops are intentionally short to finish quickly. Evaluation uses a few
episodes to verify the agents learn. Results are printed to stdout instead of
rendering plots.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog

# Because this script runs in a non‑GUI environment, disable interactive
# backends for matplotlib. We import matplotlib only if needed for future
# extensions. At present we do not generate plots.
import matplotlib
matplotlib.use("Agg")  # type: ignore

import matplotlib.pyplot as plt  # noqa: F401


###############################################################################
# Environment definition
###############################################################################

# Action set. Index corresponds to direction.
ACTIONS = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]
A = len(ACTIONS)

MOVE_DELTA = {
    0: (0, 0),   # STAY
    1: (-1, 0),  # UP
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
    4: (0, 1),   # RIGHT
}


@dataclass
class TagEnv:
    """5×5 grid Tag environment with simultaneous actions and move failure."""

    size: int = 5
    p_fail: float = 0.10
    t_max: int = 30
    step_penalty: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.t = 0
        self.c = (0, 0)
        self.r = (0, 0)

    def reset(self) -> Tuple[int, int, int, int]:
        """Reset environment to random initial positions and return state."""
        self.t = 0
        cx, cy = self.rng.randrange(self.size), self.rng.randrange(self.size)
        rx, ry = self.rng.randrange(self.size), self.rng.randrange(self.size)
        # Ensure distinct initial positions.
        while (rx, ry) == (cx, cy):
            rx, ry = self.rng.randrange(self.size), self.rng.randrange(self.size)
        self.c = (cx, cy)
        self.r = (rx, ry)
        return (cx, cy, rx, ry)

    def _apply_action(self, pos: Tuple[int, int], action: int, fail: bool) -> Tuple[int, int]:
        """Apply action with potential failure; clamp to grid."""
        if fail:
            return pos
        dx, dy = MOVE_DELTA[action]
        nx = min(self.size - 1, max(0, pos[0] + dx))
        ny = min(self.size - 1, max(0, pos[1] + dy))
        return (nx, ny)

    def step(self, a1: int, a2: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict]:
        """
        Perform one timestep given actions from catcher (a1) and runner (a2).

        Returns:
          next_state, reward_to_catcher, done, info
        """
        self.t += 1

        c_old = self.c
        r_old = self.r

        # Each player independently fails to move with probability p_fail.
        c_fail = (self.rng.random() < self.p_fail)
        r_fail = (self.rng.random() < self.p_fail)

        c_new = self._apply_action(c_old, a1, c_fail)
        r_new = self._apply_action(r_old, a2, r_fail)

        # Capture if same cell or if they cross each other.
        capture = (c_new == r_new) or (c_new == r_old and r_new == c_old)

        self.c, self.r = c_new, r_new

        if capture:
            reward = 1.0
            done = True
        else:
            reward = -self.step_penalty
            done = (self.t >= self.t_max)

        s = (self.c[0], self.c[1], self.r[0], self.r[1])
        info = {"capture": capture, "t": self.t}
        return s, reward, done, info


###############################################################################
# Helpers for state encoding and LP solving
###############################################################################

def encode_state(s: Tuple[int, int, int, int], size: int = 5) -> int:
    """Encode 4‑tuple state to integer index for tabular arrays."""
    cx, cy, rx, ry = s
    c_id = cx * size + cy
    r_id = rx * size + ry
    return c_id * (size * size) + r_id


def solve_row_player_maximin(Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve the row player's (catcher) maximin mixed strategy for zero‑sum payoff Q.
    We formulate a linear program:
      max v
      s.t. sum_a pi[a] Q[a,b] >= v for all b
           sum_a pi[a] = 1
           pi[a] >= 0
    Returns (pi, v).
    If LP fails, returns uniform mixed strategy.
    """
    assert Q.shape == (A, A)

    # Minimize negative v (i.e. maximize v).
    c = np.zeros(A + 1)
    c[-1] = -1.0

    # Inequalities: -∑_a pi[a] Q[a,b] + v <= 0  for each b
    A_ub_list: List[List[float]] = []
    b_ub_list: List[float] = []
    for b in range(A):
        row = np.zeros(A + 1)
        row[:A] = -Q[:, b]
        row[-1] = 1.0
        A_ub_list.append(row.tolist())
        b_ub_list.append(0.0)

    A_eq = np.zeros((1, A + 1))
    A_eq[0, :A] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * A + [(None, None)]

    res = linprog(
        c=c,
        A_ub=np.array(A_ub_list),
        b_ub=np.array(b_ub_list),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        pi = np.ones(A) / A
        v = float(np.min(pi @ Q))
        return pi, v

    x = res.x
    pi = np.clip(x[:A], 0.0, 1.0)
    pi /= (pi.sum() + 1e-12)
    v = float(x[-1])
    return pi, v


###############################################################################
# Agents
###############################################################################

class FPAgent:
    """
    Fictitious play agent using state‑wise empirical opponent model and
    1‑step best‑response heuristic.

    The agent observes the opponent's actions and records frequencies per state.
    It then chooses the action that maximizes capture probability and, in ties,
    minimizes expected Manhattan distance (for the catcher), or vice‑versa for
    the runner.
    """

    def __init__(self, role: str, size: int = 5, p_fail: float = 0.10,
                 prior: float = 1e-3, seed: int = 0) -> None:
        assert role in ("catcher", "runner")
        self.role = role
        self.size = size
        self.p_fail = p_fail
        self.rng = random.Random(seed)
        self.n_states = (size * size) * (size * size)
        # counts_opp[s, a] tracks how often opponent plays a at state s.
        self.counts_opp = np.ones((self.n_states, A), dtype=np.float64) * prior

    def observe(self, s: Tuple[int, int, int, int], opp_action: int) -> None:
        sid = encode_state(s, self.size)
        self.counts_opp[sid, opp_action] += 1.0

    def _next_pos(self, pos: Tuple[int, int], action: int, fail: bool) -> Tuple[int, int]:
        if fail:
            return pos
        dx, dy = MOVE_DELTA[action]
        nx = min(self.size - 1, max(0, pos[0] + dx))
        ny = min(self.size - 1, max(0, pos[1] + dy))
        return (nx, ny)

    def _expected_outcomes(
        self, s: Tuple[int, int, int, int], a_self: int, a_opp: int
    ) -> Tuple[float, float]:
        """
        Compute (capture_prob, expected_distance) given actions for both players.
        Takes into account independent move failure.
        """
        cx, cy, rx, ry = s
        c_old = (cx, cy)
        r_old = (rx, ry)

        p = self.p_fail
        outcomes = [
            ((False, False), (1 - p) * (1 - p)),
            ((True, False), p * (1 - p)),
            ((False, True), (1 - p) * p),
            ((True, True), p * p),
        ]

        cap_prob = 0.0
        exp_dist = 0.0

        for (c_fail, r_fail), pr in outcomes:
            if self.role == "catcher":
                c_new = self._next_pos(c_old, a_self, c_fail)
                r_new = self._next_pos(r_old, a_opp, r_fail)
            else:
                c_new = self._next_pos(c_old, a_opp, c_fail)
                r_new = self._next_pos(r_old, a_self, r_fail)

            capture = (c_new == r_new) or (c_new == r_old and r_new == c_old)
            if capture:
                cap_prob += pr
                # Manhattan distance is 0 if capture.
            else:
                exp_dist += pr * (abs(c_new[0] - r_new[0]) + abs(c_new[1] - r_new[1]))
        return cap_prob, exp_dist

    def act(self, s: Tuple[int, int, int, int]) -> int:
        """Choose action based on opponent empirical distribution."""
        sid = encode_state(s, self.size)
        opp_counts = self.counts_opp[sid]
        opp_pi = opp_counts / (opp_counts.sum() + 1e-12)

        scores: List[Tuple[float, float, int]] = []
        for a_self in range(A):
            cap_prob = 0.0
            exp_dist = 0.0
            for a_opp in range(A):
                cp, ed = self._expected_outcomes(s, a_self, a_opp)
                cap_prob += opp_pi[a_opp] * cp
                exp_dist += opp_pi[a_opp] * ed
            if self.role == "catcher":
                scores.append((cap_prob, -exp_dist, a_self))
            else:
                scores.append((-cap_prob, exp_dist, a_self))

        # Select best by lexicographic order; break ties randomly.
        scores.sort(reverse=True)
        best = [scores[0]]
        for sc in scores[1:]:
            if sc[0] == scores[0][0] and sc[1] == scores[0][1]:
                best.append(sc)
            else:
                break
        return self.rng.choice(best)[2]


class MinimaxQAgent:
    """
    Tabular Minimax‑Q agent for zero‑sum RL.

    We train the row player (catcher) Q(s,a1,a2). At each state we compute the
    catcher’s mixed strategy via LP. The runner plays a best response to avoid
    solving a second LP. Q‑values are updated via Q‑learning style update.
    """

    def __init__(
        self,
        size: int = 5,
        gamma: float = 0.95,
        alpha: float = 0.10,
        eps_start: float = 0.20,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 5_000,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.n_states = (size * size) * (size * size)
        self.gamma = gamma
        self.alpha = alpha
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = max(1, eps_decay_episodes)
        self.rng = random.Random(seed)
        # Q[sid, a1, a2]
        self.Q = np.zeros((self.n_states, A, A), dtype=np.float64)
        # caches for mixed strategy and value per state to avoid repeated LP solves
        self.pi_cache = np.zeros((self.n_states, A), dtype=np.float64)
        self.v_cache = np.zeros(self.n_states, dtype=np.float64)
        self.dirty = np.ones(self.n_states, dtype=bool)

    def epsilon(self, episode: int) -> float:
        """Linear annealing of epsilon for exploration."""
        t = min(1.0, episode / self.eps_decay_episodes)
        return (1 - t) * self.eps_start + t * self.eps_end

    def _compute_policy(self, sid: int) -> Tuple[np.ndarray, float]:
        """Compute (pi, v) via LP for state sid if dirty; else return cached."""
        if self.dirty[sid]:
            Qmat = self.Q[sid]
            pi, v = solve_row_player_maximin(Qmat)
            self.pi_cache[sid] = pi
            self.v_cache[sid] = v
            self.dirty[sid] = False
        return self.pi_cache[sid], self.v_cache[sid]

    def act_catcher(self, sid: int, eps: float) -> int:
        """
        Choose catcher action using epsilon‑greedy on mixed strategy.
        """
        pi, _ = self._compute_policy(sid)
        if self.rng.random() < eps:
            return self.rng.randrange(A)
        return int(np.random.choice(np.arange(A), p=pi))

    def best_response_runner(self, sid: int, pi: np.ndarray, eps: float) -> int:
        """
        Runner plays approximate best response by minimizing expected Q given pi.
        With probability eps choose random action for exploration.
        """
        if self.rng.random() < eps:
            return self.rng.randrange(A)
        Qmat = self.Q[sid]  # shape (A, A)
        exp_vals = pi @ Qmat  # shape (A,)
        # Minimizer chooses action with minimal expected value (row player reward).
        return int(np.argmin(exp_vals))

    def update(self, sid: int, a1: int, a2: int, r: float,
                sid_next: int, done: bool) -> None:
        """Q-learning update rule for zero-sum games."""
        # Target uses the value function at next state.
        if done:
            target = r
        else:
            _, v_next = self._compute_policy(sid_next)
            target = r + self.gamma * v_next

        self.Q[sid, a1, a2] = (1 - self.alpha) * self.Q[sid, a1, a2] + self.alpha * target

        # Mark current state as dirty, since its Q values changed.
        self.dirty[sid] = True

    def policies(self, sid: int) -> Tuple[np.ndarray, float]:
        """Return current (pi, v) for external evaluation."""
        return self._compute_policy(sid)


###############################################################################
# Training and evaluation functions
###############################################################################

@dataclass
class EvalStats:
    capture_rate: float
    avg_steps_to_capture: float
    avg_return: float


def run_episode(env: TagEnv, catcher_policy, runner_policy) -> Tuple[bool, int, float]:
    """
    Simulate a single episode under given policies.
    Returns (captured_flag, steps_taken, cumulative_reward_for_catcher).
    """
    s = env.reset()
    total_r = 0.0
    captured = False
    for _ in range(env.t_max):
        a1 = catcher_policy(s)
        a2 = runner_policy(s)
        s, r, done, info = env.step(a1, a2)
        total_r += r
        if info.get("capture", False):
            captured = True
        if done:
            break
    return captured, env.t, total_r


def evaluate(env: TagEnv, catcher_policy, runner_policy,
                n_episodes: int = 200) -> EvalStats:
    """
    Evaluate given policies for a small number of episodes.
    """
    caps = 0
    steps_caps: List[int] = []
    returns: List[float] = []
    for _ in range(n_episodes):
        captured, steps, ret = run_episode(env, catcher_policy, runner_policy)
        returns.append(ret)
        if captured:
            caps += 1
            steps_caps.append(steps)
    capture_rate = caps / n_episodes
    avg_steps_to_capture = float(np.mean(steps_caps)) if steps_caps else float("nan")
    avg_return = float(np.mean(returns)) if returns else float("nan")
    return EvalStats(capture_rate, avg_steps_to_capture, avg_return)


def train_fp_vs_fp(p_fail: float,
                    episodes: int = 200,
                    eval_every: int = 100,
                    seed: int = 0) -> Dict[str, List[float]]:
    """Train FP agents against each other for a small number of episodes."""
    env = TagEnv(p_fail=p_fail, seed=seed)
    fp_c = FPAgent("catcher", p_fail=p_fail, seed=seed + 1)
    fp_r = FPAgent("runner", p_fail=p_fail, seed=seed + 2)
    logs = {"episode": [], "capture_rate": [], "avg_steps": [], "avg_return": []}
    # Define evaluation policies capturing current behaviour.
    def catcher_policy(s):
        return fp_c.act(s)
    def runner_policy(s):
        return fp_r.act(s)
    for ep in range(1, episodes + 1):
        s = env.reset()
        # Play one episode for training.
        for _ in range(env.t_max):
            a1 = fp_c.act(s)
            a2 = fp_r.act(s)
            # Update opponent models.
            fp_c.observe(s, a2)
            fp_r.observe(s, a1)
            s, _, done, _ = env.step(a1, a2)
            if done:
                break
        if ep % eval_every == 0:
            eval_env = TagEnv(p_fail=p_fail, seed=seed + 10_000 + ep)
            stats = evaluate(eval_env, catcher_policy, runner_policy, n_episodes=10)
            logs["episode"].append(ep)
            logs["capture_rate"].append(stats.capture_rate)
            logs["avg_steps"].append(stats.avg_steps_to_capture)
            logs["avg_return"].append(stats.avg_return)
    return logs


def train_minimaxq_selfplay(p_fail: float,
                            episodes: int = 200,
                            eval_every: int = 100,
                            seed: int = 0) -> Dict[str, List[float]]:
    """Train MinimaxQ agent in self‑play (both players are RL) for a small number of episodes."""
    env = TagEnv(p_fail=p_fail, seed=seed)
    q = MinimaxQAgent(seed=seed + 1)
    logs = {"episode": [], "capture_rate": [], "avg_steps": [], "avg_return": []}
    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = q.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            # Catcher chooses using epsilon‑greedy.
            a1 = q.act_catcher(sid, eps)
            # Runner best responds to pi.
            pi, _ = q.policies(sid)
            a2 = q.best_response_runner(sid, pi, eps)
            s2, r, done, _ = env.step(a1, a2)
            sid2 = encode_state(s2, env.size)
            q.update(sid, a1, a2, r, sid2, done)
            s = s2
            if done:
                break
        if ep % eval_every == 0:
            eval_env = TagEnv(p_fail=p_fail, seed=seed + 20_000 + ep)
            def catcher_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q.policies(sid_)
                return int(np.random.choice(np.arange(A), p=pi_))
            def runner_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q.policies(sid_)
                return q.best_response_runner(sid_, pi_, 0.0)
            stats = evaluate(eval_env, catcher_policy, runner_policy, n_episodes=10)
            logs["episode"].append(ep)
            logs["capture_rate"].append(stats.capture_rate)
            logs["avg_steps"].append(stats.avg_steps_to_capture)
            logs["avg_return"].append(stats.avg_return)
    return logs


def train_minimaxq_vs_fp(p_fail: float,
                            episodes: int = 200,
                            eval_every: int = 100,
                            seed: int = 0) -> Dict[str, List[float]]:
    """Train MinimaxQ catcher vs FP runner for a small number of episodes."""
    env = TagEnv(p_fail=p_fail, seed=seed)
    q = MinimaxQAgent(seed=seed + 1)
    fp_r = FPAgent("runner", p_fail=p_fail, seed=seed + 2)
    logs = {"episode": [], "capture_rate": [], "avg_steps": [], "avg_return": []}
    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = q.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a1 = q.act_catcher(sid, eps)
            # Runner plays FP.
            a2 = fp_r.act(s)
            # Runner observes catcher action for its empirical model.
            fp_r.observe(s, a1)
            s2, r, done, _ = env.step(a1, a2)
            sid2 = encode_state(s2, env.size)
            q.update(sid, a1, a2, r, sid2, done)
            s = s2
            if done:
                break
        if ep % eval_every == 0:
            eval_env = TagEnv(p_fail=p_fail, seed=seed + 30_000 + ep)
            def catcher_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q.policies(sid_)
                return int(np.random.choice(np.arange(A), p=pi_))
            def runner_policy(s_):
                return fp_r.act(s_)
            stats = evaluate(eval_env, catcher_policy, runner_policy, n_episodes=10)
            logs["episode"].append(ep)
            logs["capture_rate"].append(stats.capture_rate)
            logs["avg_steps"].append(stats.avg_steps_to_capture)
            logs["avg_return"].append(stats.avg_return)
    return logs


###############################################################################
# Main entry point
###############################################################################

def main() -> None:
    """
    Run small training experiments for a single p_fail value.
    To keep runtime short, we train each matchup for only a few hundred episodes.
    Prints the final evaluation stats for each matchup.
    """
    p_fail_values = [0.10, 0.20]      # stochastic helps capture
    episodes_fp = 2_000
    episodes_mm = 10_000
    episodes_mix = 10_000
    eval_every_fp = 200
    eval_every_mm = 1_000
    eval_every_mix = 1_000
    
    for p_fail in p_fail_values:
        print(f"=== Training with p_fail={p_fail:.2f} ===")
        fp_logs = train_fp_vs_fp(p_fail=p_fail, episodes=episodes_fp,
                                    eval_every=eval_every_fp, seed=0)
        mm_logs = train_minimaxq_selfplay(p_fail=p_fail, episodes=episodes_mm,
                                            eval_every=eval_every_mm, seed=1)
        mix_logs = train_minimaxq_vs_fp(p_fail=p_fail, episodes=episodes_mix,
                                        eval_every=eval_every_mix, seed=2)
        # Extract final metrics.
        def last_or_nan(lst: List[float]) -> float:
            return lst[-1] if lst else float("nan")
        print("FP vs FP final:")
        print(f"  Capture rate:   {last_or_nan(fp_logs['capture_rate']):.3f}")
        print(f"  Avg steps:      {last_or_nan(fp_logs['avg_steps']):.3f}")
        print(f"  Avg return:     {last_or_nan(fp_logs['avg_return']):.3f}")
        print("MinimaxQ vs MinimaxQ final:")
        print(f"  Capture rate:   {last_or_nan(mm_logs['capture_rate']):.3f}")
        print(f"  Avg steps:      {last_or_nan(mm_logs['avg_steps']):.3f}")
        print(f"  Avg return:     {last_or_nan(mm_logs['avg_return']):.3f}")
        print("MinimaxQ vs FP final:")
        print(f"  Capture rate:   {last_or_nan(mix_logs['capture_rate']):.3f}")
        print(f"  Avg steps:      {last_or_nan(mix_logs['avg_steps']):.3f}")
        print(f"  Avg return:     {last_or_nan(mix_logs['avg_return']):.3f}")
        print()


if __name__ == "__main__":
    main()