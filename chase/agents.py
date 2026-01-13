"""
Agents used in the chase environment.
"""

import random
from typing import Dict, List, Tuple

import numpy as np

from env import A, ACTIONS, MOVE_DELTA
from utils import encode_state, solve_row_player_maximin


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
