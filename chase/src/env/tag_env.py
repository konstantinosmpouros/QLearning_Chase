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
        info = {
            "capture": capture,
            "t": self.t,
            "c_fail": c_fail,
            "r_fail": r_fail,
            "c_old": c_old,
            "r_old": r_old,
            "c_new": c_new,
            "r_new": r_new,
        }
        return s, reward, done, info
