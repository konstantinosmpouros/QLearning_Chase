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

        done = (outcome is not None) or (self.t >= self.t_max)
        if outcome == "A_WIN":
            reward = 1.0
        elif outcome == "B_WIN":
            reward = -1.0
        elif outcome == "DRAW":
            reward = 0.0
        else:
            reward = -self.step_penalty

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
            "a_old": a_old,
            "b_old": b_old,
            "a_new": a_new,
            "b_new": b_new,
            "t": self.t,
        }
        return s, reward, done, info
