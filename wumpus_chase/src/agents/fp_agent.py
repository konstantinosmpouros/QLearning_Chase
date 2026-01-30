"""Fictitious Play agent for Wumpus Chase."""

import random
from typing import List, Tuple

import numpy as np

from env import A, MOVE_DELTA
from env.maps import MapLayout
from utils import encode_state


class FPAgent:
    """
    Fictitious play agent with a one-step lookahead heuristic.

    role:
      - "row" maximizes expected payoff for agent A
      - "col" minimizes expected payoff for agent A
    """

    def __init__(
        self,
        role: str,
        layout: MapLayout,
        p_fail: float = 0.10,
        step_penalty: float = 0.01,
        outcome_reward: float = 5.0,
        obstacle_penalty: float = 0.05,
        chase_dist_reward: float = 0.04,
        treasure_dist_reward: float = 0.03,
        prior: float = 1e-3,
        seed: int = 0,
    ) -> None:
        assert role in ("row", "col")
        self.role = role
        self.layout = layout
        self.size = layout.size
        self.p_fail = p_fail
        self.step_penalty = step_penalty
        self.outcome_reward = outcome_reward
        self.obstacle_penalty = obstacle_penalty
        self.chase_dist_reward = chase_dist_reward
        self.treasure_dist_reward = treasure_dist_reward
        self.rng = random.Random(seed)
        self.n_states = (self.size * self.size) * (self.size * self.size)
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
        if (nx, ny) in self.layout.obstacles:
            return pos
        return (nx, ny)

    def _would_hit_obstacle(self, pos: Tuple[int, int], action: int, fail: bool) -> bool:
        if fail:
            return False
        dx, dy = MOVE_DELTA[action]
        nx = min(self.size - 1, max(0, pos[0] + dx))
        ny = min(self.size - 1, max(0, pos[1] + dy))
        return (nx, ny) in self.layout.obstacles

    def _manhattan(self, a_pos: Tuple[int, int], b_pos: Tuple[int, int]) -> int:
        return abs(a_pos[0] - b_pos[0]) + abs(a_pos[1] - b_pos[1])

    def _resolve_outcome(
        self,
        row_old: Tuple[int, int],
        col_old: Tuple[int, int],
        row_new: Tuple[int, int],
        col_new: Tuple[int, int],
    ) -> str | None:
        capture = (row_new == col_new) or (row_new == col_old and col_new == row_old)
        if capture:
            row_captures = (row_new == col_old)
            col_captures = (col_new == row_old)
            if row_captures and not col_captures:
                return "A_WIN"
            if col_captures and not row_captures:
                return "B_WIN"
            return "DRAW"

        row_dead = (row_new == self.layout.wumpus)
        col_dead = (col_new == self.layout.wumpus)
        if row_dead or col_dead:
            if row_dead and col_dead:
                return "DRAW"
            return "B_WIN" if row_dead else "A_WIN"

        row_treasure = (row_new == self.layout.treasure)
        col_treasure = (col_new == self.layout.treasure)
        if row_treasure or col_treasure:
            if row_treasure and col_treasure:
                return "DRAW"
            return "A_WIN" if row_treasure else "B_WIN"

        return None

    def _expected_metrics(
        self,
        s: Tuple[int, int, int, int],
        a_self: int,
        a_opp: int,
    ) -> Tuple[float, float, float]:
        ax, ay, bx, by = s
        row_old = (ax, ay)
        col_old = (bx, by)

        if self.role == "row":
            a_row, a_col = a_self, a_opp
        else:
            a_row, a_col = a_opp, a_self

        p = self.p_fail
        outcomes = [
            ((False, False), (1 - p) * (1 - p)),
            ((True, False), p * (1 - p)),
            ((False, True), (1 - p) * p),
            ((True, True), p * p),
        ]

        exp_reward = 0.0
        exp_row_dist = 0.0
        exp_col_dist = 0.0

        for (row_fail, col_fail), pr in outcomes:
            row_new = self._next_pos(row_old, a_row, row_fail)
            col_new = self._next_pos(col_old, a_col, col_fail)
            outcome = self._resolve_outcome(row_old, col_old, row_new, col_new)
            if outcome == "A_WIN":
                reward_outcome = self.outcome_reward
            elif outcome == "B_WIN":
                reward_outcome = -self.outcome_reward
            elif outcome == "DRAW":
                reward_outcome = 0.0
            else:
                reward_outcome = 0.0

            reward_step = 0.0 if outcome is not None else -self.step_penalty
            reward_obstacle = 0.0
            if self._would_hit_obstacle(row_old, a_row, row_fail):
                reward_obstacle -= self.obstacle_penalty
            if self._would_hit_obstacle(col_old, a_col, col_fail):
                reward_obstacle += self.obstacle_penalty

            dist_ab_before = self._manhattan(row_old, col_old)
            dist_ab_after = self._manhattan(row_new, col_new)
            reward_chase = 0.0
            if dist_ab_after < dist_ab_before:
                reward_chase += self.chase_dist_reward
            elif dist_ab_after > dist_ab_before:
                reward_chase -= self.chase_dist_reward

            row_dist_before = self._manhattan(row_old, self.layout.treasure)
            row_dist_after = self._manhattan(row_new, self.layout.treasure)
            col_dist_before = self._manhattan(col_old, self.layout.treasure)
            col_dist_after = self._manhattan(col_new, self.layout.treasure)

            if dist_ab_before <= row_dist_before:
                chase_weight = 1.0
                treasure_weight = 0.5
            else:
                chase_weight = 0.5
                treasure_weight = 1.0

            reward_treasure = 0.0
            if row_dist_after < row_dist_before:
                reward_treasure += self.treasure_dist_reward
            elif row_dist_after > row_dist_before:
                reward_treasure -= self.treasure_dist_reward
            if col_dist_after < col_dist_before:
                reward_treasure -= self.treasure_dist_reward
            elif col_dist_after > col_dist_before:
                reward_treasure += self.treasure_dist_reward

            reward = (
                reward_outcome
                + reward_step
                + reward_obstacle
                + (reward_chase * chase_weight)
                + (reward_treasure * treasure_weight)
            )

            row_dist = row_dist_after
            col_dist = col_dist_after

            exp_reward += pr * reward
            exp_row_dist += pr * row_dist
            exp_col_dist += pr * col_dist

        return exp_reward, exp_row_dist, exp_col_dist

    def act(self, s: Tuple[int, int, int, int]) -> int:
        sid = encode_state(s, self.size)
        opp_counts = self.counts_opp[sid]
        opp_pi = opp_counts / (opp_counts.sum() + 1e-12)

        scored: List[Tuple[float, float, int]] = []
        for a_self in range(A):
            exp_reward = 0.0
            exp_self_dist = 0.0
            for a_opp in range(A):
                reward, row_dist, col_dist = self._expected_metrics(s, a_self, a_opp)
                exp_reward += opp_pi[a_opp] * reward
                if self.role == "row":
                    exp_self_dist += opp_pi[a_opp] * row_dist
                else:
                    exp_self_dist += opp_pi[a_opp] * col_dist

            if self.role == "row":
                score = exp_reward
            else:
                score = -exp_reward
            tie = -exp_self_dist
            scored.append((score, tie, a_self))

        scored.sort(reverse=True)
        best = [scored[0]]
        for sc in scored[1:]:
            if sc[0] == scored[0][0] and sc[1] == scored[0][1]:
                best.append(sc)
            else:
                break
        return self.rng.choice(best)[2]
