"""Minimax-Q agent for Wumpus Chase."""

import random
from typing import Tuple

import numpy as np

from env import A
from utils import solve_row_player_maximin


class MinimaxQAgent:
    """
    Tabular Minimax-Q agent for a zero-sum game.

    Independent learners store Q[s, own_action, opponent_action].
    This is a security-policy learner; zero-sum convergence assumptions do not
    automatically apply to the legacy general-sum reward shaping.
    """

    def __init__(
        self,
        size: int,
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

        self.Q = np.zeros((self.n_states, A, A), dtype=np.float64)
        self.pi_cache = np.zeros((self.n_states, A), dtype=np.float64)
        self.v_cache = np.zeros(self.n_states, dtype=np.float64)
        self.dirty = np.ones(self.n_states, dtype=bool)

    def epsilon(self, episode: int) -> float:
        t = min(1.0, episode / self.eps_decay_episodes)
        return (1 - t) * self.eps_start + t * self.eps_end

    def _compute_policy(self, sid: int) -> Tuple[np.ndarray, float]:
        if self.dirty[sid]:
            Qmat = self.Q[sid]
            pi, v = solve_row_player_maximin(Qmat)
            self.pi_cache[sid] = pi
            self.v_cache[sid] = v
            self.dirty[sid] = False
        return self.pi_cache[sid], self.v_cache[sid]

    def act_row(self, sid: int, eps: float) -> int:
        pi, _ = self._compute_policy(sid)
        if self.rng.random() < eps:
            return self.rng.randrange(A)
        return int(self.rng.choices(range(A), weights=pi, k=1)[0])

    def act_col(self, sid: int, eps: float) -> int:
        """Select the runner's own row policy.

        All independent-agent training loops call update with (own, opponent)
        actions, including the runner. Therefore no transpose belongs here.
        best_response_col remains the separate column response to a row payoff.
        """
        return self.act_row(sid, eps)

    def best_response_col(self, sid: int, pi: np.ndarray, eps: float) -> int:
        if self.rng.random() < eps:
            return self.rng.randrange(A)
        Qmat = self.Q[sid]
        exp_vals = pi @ Qmat
        return int(np.argmin(exp_vals))

    def update(self, sid: int, a_row: int, a_col: int, r: float, sid_next: int, done: bool) -> None:
        if done:
            target = r
        else:
            _, v_next = self._compute_policy(sid_next)
            target = r + self.gamma * v_next

        self.Q[sid, a_row, a_col] = (1 - self.alpha) * self.Q[sid, a_row, a_col] + self.alpha * target
        self.dirty[sid] = True

    def policies(self, sid: int) -> Tuple[np.ndarray, float]:
        return self._compute_policy(sid)
