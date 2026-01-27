"""
Environment and constants for the chase game.
"""

import random
from dataclasses import dataclass
from typing import Dict, Tuple

# Action set. Index corresponds to direction.
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
class TagEnv:
    """5x5 grid Tag environment with simultaneous actions and move failure."""

    size: int = 5
    p_fail: float = 0.10
    t_max: int = 30
    step_penalty: float = 0.01
    capture_reward: float = 10.0
    wall_penalty: float = 0.02
    move_reward: float = 0.01
    dist_reward: float = 0.02
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

    def _would_hit_wall(self, pos: Tuple[int, int], action: int) -> bool:
        """Return True if action would move out of bounds from pos."""
        dx, dy = MOVE_DELTA[action]
        nx = pos[0] + dx
        ny = pos[1] + dy
        return (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size)

    def _manhattan(self, c_pos: Tuple[int, int], r_pos: Tuple[int, int]) -> int:
        return abs(c_pos[0] - r_pos[0]) + abs(c_pos[1] - r_pos[1])

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

        dist_before = self._manhattan(c_old, r_old)
        dist_after = self._manhattan(c_new, r_new)
        dist_delta = dist_after - dist_before

        c_reward = 0.0
        r_reward = 0.0
        c_reward_capture = 0.0
        r_reward_capture = 0.0
        c_reward_step = 0.0
        r_reward_step = 0.0
        c_reward_wall = 0.0
        r_reward_wall = 0.0
        c_reward_move = 0.0
        r_reward_move = 0.0
        c_reward_dist = 0.0
        r_reward_dist = 0.0

        if capture:
            c_reward_capture = self.capture_reward
            r_reward_capture = -self.capture_reward
        else:
            c_reward_step = -self.step_penalty

        c_hit_wall = self._would_hit_wall(c_old, a1)
        r_hit_wall = self._would_hit_wall(r_old, a2)
        if c_hit_wall:
            c_reward_wall = -self.wall_penalty
        if r_hit_wall:
            r_reward_wall = -self.wall_penalty

        c_moved = (c_new != c_old)
        r_moved = (r_new != r_old)
        if c_moved:
            c_reward_move = self.move_reward
        if r_moved:
            r_reward_move = self.move_reward

        if dist_after < dist_before:
            c_reward_dist = self.dist_reward
            r_reward_dist = -self.dist_reward
        elif dist_after > dist_before:
            c_reward_dist = -self.dist_reward
            r_reward_dist = self.dist_reward

        c_reward = (
            c_reward + c_reward_capture + c_reward_step + c_reward_wall + c_reward_move + c_reward_dist
        )
        r_reward = (
            r_reward + r_reward_capture + r_reward_step + r_reward_wall + r_reward_move + r_reward_dist
        )

        if capture:
            done = True
        else:
            done = (self.t >= self.t_max)

        s = (self.c[0], self.c[1], self.r[0], self.r[1])
        info = {
            "capture": capture,
            "t": self.t,
            "c_fail": c_fail,
            "r_fail": r_fail,
            "c_old": c_old,
            "r_old": r_old,
            "c_new": c_new,
            "r_new": r_new,
            "reward_runner": r_reward,
            "dist_before": dist_before,
            "dist_after": dist_after,
            "dist_delta": dist_delta,
            "c_hit_wall": c_hit_wall,
            "r_hit_wall": r_hit_wall,
            "c_moved": c_moved,
            "r_moved": r_moved,
            "reward_capture_c": c_reward_capture,
            "reward_capture_r": r_reward_capture,
            "reward_step_c": c_reward_step,
            "reward_step_r": r_reward_step,
            "reward_wall_c": c_reward_wall,
            "reward_wall_r": r_reward_wall,
            "reward_move_c": c_reward_move,
            "reward_move_r": r_reward_move,
            "reward_dist_c": c_reward_dist,
            "reward_dist_r": r_reward_dist,
        }
        return s, c_reward, done, info
