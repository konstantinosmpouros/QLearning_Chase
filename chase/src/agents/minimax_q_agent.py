"""
Minimax-Q agent implementation for zero-sum chase.
"""

import random
from typing import Tuple

import numpy as np

from env import A
from utils import solve_row_player_maximin


class MinimaxQAgent:
    """
    Tabular Minimax-Q agent for zero-sum RL.

    We train the row player (catcher) Q(s,a1,a2). At each state we compute the
    catcher's mixed strategy via LP. The runner plays a best response to avoid
    solving a second LP. Q-values are updated via Q-learning style update.
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
        Choose catcher action using epsilon-greedy on mixed strategy.
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
