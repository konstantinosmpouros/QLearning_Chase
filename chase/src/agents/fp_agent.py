"""
Fictitious Play agent implementation.
"""

import random
from typing import List, Tuple

import numpy as np

from env import A, MOVE_DELTA
from utils import encode_state


class FPAgent:
    """
    Fictitious play agent using state-wise empirical opponent model and
    1-step best-response heuristic.

    The agent observes the opponent's actions and records frequencies per state.
    It then chooses the action that maximizes capture probability and, in ties,
    minimizes expected Manhattan distance (for the catcher), or vice-versa for
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
        """Record opponent action at given state."""
        sid = encode_state(s, self.size)
        self.counts_opp[sid, opp_action] += 1.0


    def _next_pos(self, pos: Tuple[int, int], action: int, fail: bool) -> Tuple[int, int]:
        """Compute next position given action and whether move fails."""
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
