from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

from env.maps import MapLayout, default_layout

ACTIONS = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]
A = len(ACTIONS)

MOVE_DELTA = {
    0: (0, 0),   # STAY
    1: (1, 0),   # UP (increases x)
    2: (-1, 0),  # DOWN (decreases x)
    3: (0, -1),  # LEFT
    4: (0, 1),   # RIGHT
}


@dataclass
class WumpusChaseEnv:
    layout: MapLayout | None = None
    p_fail: float = 0.10
    t_max: int = 40
    step_penalty: float = 0.01
    outcome_reward: float = 5.0
    obstacle_penalty: float = 0.05
    chase_dist_reward: float = 0.04
    treasure_dist_reward: float = 0.03
    seed: int = 0

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = default_layout()
        self.size = self.layout.size
        self.rng = random.Random(self.seed)
        self.t = 0
        self.a = (0, 0)
        self.b = (0, 0)
        self._free_cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.layout.obstacles
            and (x, y) != self.layout.wumpus
            and (x, y) != self.layout.treasure
        ]

    def reset(self) -> Tuple[int, int, int, int]:
        self.t = 0
        a_pos = self.rng.choice(self._free_cells)
        b_pos = self.rng.choice(self._free_cells)
        while b_pos == a_pos:
            b_pos = self.rng.choice(self._free_cells)
        self.a = a_pos
        self.b = b_pos
        return (self.a[0], self.a[1], self.b[0], self.b[1])

    def _apply_action(self, pos: Tuple[int, int], action: int, fail: bool) -> Tuple[int, int]:
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

    def _capture_outcome(
        self,
        a_old: Tuple[int, int],
        b_old: Tuple[int, int],
        a_new: Tuple[int, int],
        b_new: Tuple[int, int],
    ) -> str | None:
        capture = (a_new == b_new) or (a_new == b_old and b_new == a_old)
        if not capture:
            return None
        a_captures = (a_new == b_old)
        b_captures = (b_new == a_old)
        if a_captures and not b_captures:
            return "A_WIN"
        if b_captures and not a_captures:
            return "B_WIN"
        return self.rng.choice(["A_WIN", "B_WIN"])

    def step(self, a1: int, a2: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict]:
        self.t += 1
        a_old = self.a
        b_old = self.b

        a_fail = (self.rng.random() < self.p_fail)
        b_fail = (self.rng.random() < self.p_fail)

        a_new = self._apply_action(a_old, a1, a_fail)
        b_new = self._apply_action(b_old, a2, b_fail)

        outcome = self._capture_outcome(a_old, b_old, a_new, b_new)
        capture = outcome is not None

        a_hit_obstacle = self._would_hit_obstacle(a_old, a1, a_fail)
        b_hit_obstacle = self._would_hit_obstacle(b_old, a2, b_fail)

        a_dead = (a_new == self.layout.wumpus)
        b_dead = (b_new == self.layout.wumpus)
        a_treasure = (a_new == self.layout.treasure)
        b_treasure = (b_new == self.layout.treasure)

        if outcome is None:
            if a_dead or b_dead:
                if a_dead and b_dead:
                    outcome = "DRAW"
                elif a_dead:
                    outcome = "B_WIN"
                else:
                    outcome = "A_WIN"
            elif a_treasure or b_treasure:
                if a_treasure and b_treasure:
                    outcome = "DRAW"
                elif a_treasure:
                    outcome = "A_WIN"
                else:
                    outcome = "B_WIN"

        self.a = a_new
        self.b = b_new

        dist_ab_before = self._manhattan(a_old, b_old)
        dist_ab_after = self._manhattan(a_new, b_new)
        dist_ab_delta = dist_ab_after - dist_ab_before

        a_treasure_before = self._manhattan(a_old, self.layout.treasure)
        a_treasure_after = self._manhattan(a_new, self.layout.treasure)
        b_treasure_before = self._manhattan(b_old, self.layout.treasure)
        b_treasure_after = self._manhattan(b_new, self.layout.treasure)

        done = (outcome is not None) or (self.t >= self.t_max)
        if outcome is None and self.t >= self.t_max:
            outcome = "DRAW"

        reward_outcome = 0.0
        reward_step = 0.0
        if outcome == "A_WIN":
            reward_outcome = self.outcome_reward
        elif outcome == "B_WIN":
            reward_outcome = -self.outcome_reward
        elif outcome == "DRAW":
            reward_outcome = 0.0
        else:
            reward_step = -self.step_penalty

        reward_obstacle = 0.0
        if a_hit_obstacle:
            reward_obstacle -= self.obstacle_penalty
        if b_hit_obstacle:
            reward_obstacle += self.obstacle_penalty

        reward_chase = 0.0
        if dist_ab_after < dist_ab_before:
            reward_chase += self.chase_dist_reward
        elif dist_ab_after > dist_ab_before:
            reward_chase -= self.chase_dist_reward

        if dist_ab_before <= a_treasure_before:
            chase_weight = 1.0
            treasure_weight = 0.5
        else:
            chase_weight = 0.5
            treasure_weight = 1.0

        reward_treasure = 0.0
        if a_treasure_after < a_treasure_before:
            reward_treasure += self.treasure_dist_reward
        elif a_treasure_after > a_treasure_before:
            reward_treasure -= self.treasure_dist_reward
        if b_treasure_after < b_treasure_before:
            reward_treasure -= self.treasure_dist_reward
        elif b_treasure_after > b_treasure_before:
            reward_treasure += self.treasure_dist_reward

        reward = (
            reward_outcome
            + reward_step
            + reward_obstacle
            + (reward_chase * chase_weight)
            + (reward_treasure * treasure_weight)
        )

        s = (self.a[0], self.a[1], self.b[0], self.b[1])
        info = {
            "outcome": outcome,
            "capture": capture,
            "a_dead": a_dead,
            "b_dead": b_dead,
            "a_treasure": a_treasure,
            "b_treasure": b_treasure,
            "a_fail": a_fail,
            "b_fail": b_fail,
            "a_hit_obstacle": a_hit_obstacle,
            "b_hit_obstacle": b_hit_obstacle,
            "dist_ab_before": dist_ab_before,
            "dist_ab_after": dist_ab_after,
            "dist_ab_delta": dist_ab_delta,
            "a_treasure_before": a_treasure_before,
            "a_treasure_after": a_treasure_after,
            "b_treasure_before": b_treasure_before,
            "b_treasure_after": b_treasure_after,
            "reward_outcome": reward_outcome,
            "reward_step": reward_step,
            "reward_obstacle": reward_obstacle,
            "reward_chase": reward_chase * chase_weight,
            "reward_treasure": reward_treasure * treasure_weight,
            "a_old": a_old,
            "b_old": b_old,
            "a_new": a_new,
            "b_new": b_new,
            "t": self.t,
        }
        return s, reward, done, info
