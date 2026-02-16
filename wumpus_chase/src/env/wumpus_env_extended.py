"""
Enhanced Wumpus Chase Environment with explicit Chaser/Runner roles.

Role convention (fixed per run):
- Agent A: Chaser
- Agent B: Runner

Outcome priority (no draw states):
1) Capture (same cell or cross-path) -> Chaser wins
2) Treasure reached -> whoever reached treasure wins
3) Hazard resolution (pit/wumpus) -> if Runner dies Chaser wins, else Runner wins
4) Timeout -> Runner wins
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, NamedTuple

from env.maps_extended import ExtendedMapLayout, default_extended_layout


ACTIONS = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]
A = len(ACTIONS)

MOVE_DELTA = {
    0: (0, 0),   # STAY
    1: (1, 0),   # UP (increases x)
    2: (-1, 0),  # DOWN (decreases x)
    3: (0, -1),  # LEFT
    4: (0, 1),   # RIGHT
}


class Perception(NamedTuple):
    """Agent's local perception at current cell."""
    breeze: bool      # Adjacent to pit
    stench: bool      # Adjacent to wumpus
    glitter: bool     # On treasure cell
    bump: bool        # Hit wall/obstacle last move
    scream: bool      # Wumpus died (not used in this version)


@dataclass
class WumpusChaseEnvExtended:
    """
    Extended Wumpus Chase environment with breeze and stench.
    
    Two agents compete in a grid world with fixed asymmetric roles:
    - Agent A is the Chaser
    - Agent B is the Runner
    
    Perception model:
    - Breeze: Felt in cells adjacent to pits
    - Stench: Felt in cells adjacent to wumpus
    - Glitter: Felt on treasure cell
    """
    
    layout: Optional[ExtendedMapLayout] = None
    p_fail: float = 0.10
    t_max: int = 50
    step_penalty: float = 0.01
    outcome_reward: float = 10.0
    pit_penalty: float = 10.0        # Penalty for falling in pit
    wumpus_penalty: float = 10.0     # Penalty for meeting wumpus
    obstacle_penalty: float = 0.05
    chase_dist_reward: float = 0.04
    treasure_dist_reward: float = 0.03
    breeze_penalty: float = 0.02     # Small penalty for being in breeze zone
    stench_penalty: float = 0.02     # Small penalty for being in stench zone
    seed: int = 0
    partial_observable: bool = False  # If True, state includes perceptions only
    role_a: str = "CHASER"
    role_b: str = "RUNNER"
    
    # Runtime state (not part of config)
    size: int = field(init=False)
    rng: random.Random = field(init=False)
    t: int = field(init=False, default=0)
    a: Tuple[int, int] = field(init=False, default=(0, 0))
    b: Tuple[int, int] = field(init=False, default=(0, 0))
    _free_cells: list = field(init=False, default_factory=list)
    _breeze_cells: frozenset = field(init=False, default_factory=frozenset)
    _stench_cells: frozenset = field(init=False, default_factory=frozenset)
    
    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = default_extended_layout()
        self.size = self.layout.size
        self.rng = random.Random(self.seed)
        self.t = 0
        self.a = (0, 0)
        self.b = (0, 0)
        
        # Precompute safe starting cells
        self._free_cells = list(self.layout.get_free_cells())
        
        # Precompute perception zones
        self._breeze_cells = self.layout.get_breeze_cells()
        self._stench_cells = self.layout.get_stench_cells()
    
    def get_perception(self, pos: Tuple[int, int], bumped: bool = False) -> Perception:
        """Get agent's perception at given position."""
        return Perception(
            breeze=pos in self._breeze_cells,
            stench=pos in self._stench_cells,
            glitter=pos == self.layout.treasure,
            bump=bumped,
            scream=False,  # Not used in this version
        )

    def _build_state(
        self,
        a_perception: Optional[Perception] = None,
        b_perception: Optional[Perception] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Build the state returned by reset/step.

        Full observable mode returns true positions (ax, ay, bx, by).
        Partial observable mode returns compact local observations:
        (a_breeze, a_stench, b_breeze, b_stench).
        """
        if not self.partial_observable:
            return (self.a[0], self.a[1], self.b[0], self.b[1])

        if a_perception is None:
            a_perception = self.get_perception(self.a, bumped=False)
        if b_perception is None:
            b_perception = self.get_perception(self.b, bumped=False)

        return (
            int(a_perception.breeze),
            int(a_perception.stench),
            int(b_perception.breeze),
            int(b_perception.stench),
        )
    
    def reset(self) -> Tuple[int, int, int, int]:
        """Reset environment to random safe starting positions."""
        self.t = 0
        
        # Place agents on safe cells only
        a_pos = self.rng.choice(self._free_cells)
        b_pos = self.rng.choice(self._free_cells)
        while b_pos == a_pos:
            b_pos = self.rng.choice(self._free_cells)
        
        self.a = a_pos
        self.b = b_pos

        a_perception = self.get_perception(self.a, bumped=False)
        b_perception = self.get_perception(self.b, bumped=False)
        return self._build_state(a_perception, b_perception)
    
    def _apply_action(
        self, pos: Tuple[int, int], action: int, fail: bool
    ) -> Tuple[Tuple[int, int], bool]:
        """
        Apply action with potential failure.
        Returns (new_position, bumped_wall_or_obstacle).
        """
        if fail:
            return pos, False
        
        dx, dy = MOVE_DELTA[action]
        nx = pos[0] + dx
        ny = pos[1] + dy
        
        # Check bounds
        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
            return pos, True  # Bumped wall
        
        # Check obstacles
        if (nx, ny) in self.layout.obstacles:
            return pos, True  # Bumped obstacle
        
        return (nx, ny), False
    
    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance between two positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _capture_outcome(
        self,
        a_old: Tuple[int, int],
        b_old: Tuple[int, int],
        a_new: Tuple[int, int],
        b_new: Tuple[int, int],
    ) -> Optional[str]:
        """
        Determine capture outcome.
        Returns 'A_WIN' (Chaser wins) or None if no capture.
        """
        # Same cell
        same_cell = (a_new == b_new)
        # Crossed paths
        crossed = (a_new == b_old and b_new == a_old)
        
        if not same_cell and not crossed:
            return None
        
        # Agent A is always Chaser, so any capture event means A wins.
        return "A_WIN"
    
    def step(
        self, a1: int, a2: int
    ) -> Tuple[Tuple[int, int, int, int], float, bool, Dict]:
        """
        Execute one timestep with simultaneous actions.
        
        Args:
            a1: Agent A's action
            a2: Agent B's action
            
        Returns:
            (next_state, reward_for_A, done, info)
        """
        self.t += 1
        
        a_old = self.a
        b_old = self.b
        
        # Move failures
        a_fail = self.rng.random() < self.p_fail
        b_fail = self.rng.random() < self.p_fail
        
        # Apply actions
        a_new, a_bumped = self._apply_action(a_old, a1, a_fail)
        b_new, b_bumped = self._apply_action(b_old, a2, b_fail)
        
        # Get perceptions
        a_perception = self.get_perception(a_new, a_bumped)
        b_perception = self.get_perception(b_new, b_bumped)
        
        # Check hazards
        a_in_pit = a_new in self.layout.pits
        b_in_pit = b_new in self.layout.pits
        a_met_wumpus = a_new == self.layout.wumpus
        b_met_wumpus = b_new == self.layout.wumpus
        
        a_dead = a_in_pit or a_met_wumpus
        b_dead = b_in_pit or b_met_wumpus
        
        # Check treasure
        a_treasure = a_new == self.layout.treasure
        b_treasure = b_new == self.layout.treasure
        
        # Check capture first (top-priority rule, independent of hazards/treasure)
        capture_outcome = self._capture_outcome(a_old, b_old, a_new, b_new)
        
        # Determine final outcome
        outcome = None
        
        # Priority: Capture > Treasure > Hazards > Time
        if capture_outcome is not None:
            outcome = capture_outcome
        elif a_treasure or b_treasure:
            if a_treasure and b_treasure:
                # Tie-breaker without draws: favor chaser.
                outcome = "A_WIN"
            elif a_treasure:
                outcome = "A_WIN"
            else:
                outcome = "B_WIN"
        elif a_dead or b_dead:
            # No draw states: if runner dies then chaser wins, otherwise runner wins.
            outcome = "A_WIN" if b_dead else "B_WIN"
        
        # Update positions (even if dead, for visualization)
        self.a = a_new
        self.b = b_new
        
        # Time limit
        done = (outcome is not None) or (self.t >= self.t_max)
        if outcome is None and self.t >= self.t_max:
            outcome = "B_WIN"
        
        # === Reward calculation ===
        reward_a = 0.0
        reward_b = 0.0

        # Outcome rewards
        reward_outcome_a = 0.0
        reward_outcome_b = 0.0
        if outcome == "A_WIN":
            reward_outcome_a = self.outcome_reward
            reward_outcome_b = -self.outcome_reward
        elif outcome == "B_WIN":
            reward_outcome_a = -self.outcome_reward
            reward_outcome_b = self.outcome_reward

        # Step penalty while episode continues.
        reward_step_a = 0.0
        reward_step_b = 0.0
        if outcome is None:
            reward_step_a = -self.step_penalty

        # Hazard penalties (own-risk penalties, role-specific).
        reward_hazard_a = 0.0
        reward_hazard_b = 0.0
        if a_in_pit:
            reward_hazard_a -= self.pit_penalty
        if a_met_wumpus:
            reward_hazard_a -= self.wumpus_penalty
        if b_in_pit:
            reward_hazard_b -= self.pit_penalty
        if b_met_wumpus:
            reward_hazard_b -= self.wumpus_penalty

        # Bump penalties
        reward_bump_a = 0.0
        reward_bump_b = 0.0
        if a_bumped:
            reward_bump_a -= self.obstacle_penalty
        if b_bumped:
            reward_bump_b -= self.obstacle_penalty

        # Perception penalties (encourage safe exploration)
        reward_perception_a = 0.0
        reward_perception_b = 0.0
        if a_perception.breeze:
            reward_perception_a -= self.breeze_penalty
        if a_perception.stench:
            reward_perception_a -= self.stench_penalty
        if b_perception.breeze:
            reward_perception_b -= self.breeze_penalty
        if b_perception.stench:
            reward_perception_b -= self.stench_penalty
        
        # Distance-based shaping
        dist_ab_before = self._manhattan(a_old, b_old)
        dist_ab_after = self._manhattan(a_new, b_new)
        
        a_treasure_before = self._manhattan(a_old, self.layout.treasure)
        a_treasure_after = self._manhattan(a_new, self.layout.treasure)
        b_treasure_before = self._manhattan(b_old, self.layout.treasure)
        b_treasure_after = self._manhattan(b_new, self.layout.treasure)
        
        # Role-driven distance shaping.
        # Chaser reward: get closer to runner.
        reward_chase_a = 0.0
        if dist_ab_after < dist_ab_before:
            reward_chase_a += self.chase_dist_reward
        elif dist_ab_after > dist_ab_before:
            reward_chase_a -= self.chase_dist_reward

        # Runner reward: increase distance from chaser.
        reward_evade_b = 0.0
        if dist_ab_after > dist_ab_before:
            reward_evade_b += self.chase_dist_reward
        elif dist_ab_after < dist_ab_before:
            reward_evade_b -= self.chase_dist_reward

        # Treasure distance rewards for each role.
        reward_treasure_a = 0.0
        reward_treasure_b = 0.0
        if a_treasure_after < a_treasure_before:
            reward_treasure_a += self.treasure_dist_reward
        elif a_treasure_after > a_treasure_before:
            reward_treasure_a -= self.treasure_dist_reward
        if b_treasure_after < b_treasure_before:
            reward_treasure_b += self.treasure_dist_reward
        elif b_treasure_after > b_treasure_before:
            reward_treasure_b -= self.treasure_dist_reward

        # Total rewards by role
        reward_a = (
            reward_outcome_a +
            reward_step_a +
            reward_hazard_a +
            reward_bump_a +
            reward_perception_a +
            reward_chase_a +
            reward_treasure_a
        )
        reward_b = (
            reward_outcome_b +
            reward_step_b +
            reward_hazard_b +
            reward_bump_b +
            reward_perception_b +
            reward_evade_b +
            reward_treasure_b
        )

        # Build state tuple (full or partial observable).
        state = self._build_state(a_perception, b_perception)
        
        # Info dict
        info = {
            "outcome": outcome,
            "capture": capture_outcome is not None,
            "t": self.t,
            "role_a": self.role_a,
            "role_b": self.role_b,
            "winner_role": (
                self.role_a if outcome == "A_WIN" else (self.role_b if outcome == "B_WIN" else "")
            ),
            # Agent A info
            "a_old": a_old,
            "a_new": a_new,
            "a_fail": a_fail,
            "a_bumped": a_bumped,
            "a_in_pit": a_in_pit,
            "a_met_wumpus": a_met_wumpus,
            "a_dead": a_dead,
            "a_treasure": a_treasure,
            "a_breeze": a_perception.breeze,
            "a_stench": a_perception.stench,
            # Agent B info
            "b_old": b_old,
            "b_new": b_new,
            "b_fail": b_fail,
            "b_bumped": b_bumped,
            "b_in_pit": b_in_pit,
            "b_met_wumpus": b_met_wumpus,
            "b_dead": b_dead,
            "b_treasure": b_treasure,
            "b_breeze": b_perception.breeze,
            "b_stench": b_perception.stench,
            # Distances
            "dist_ab_before": dist_ab_before,
            "dist_ab_after": dist_ab_after,
            "dist_ab_delta": dist_ab_after - dist_ab_before,
            "a_treasure_before": a_treasure_before,
            "a_treasure_after": a_treasure_after,
            "b_treasure_before": b_treasure_before,
            "b_treasure_after": b_treasure_after,
            "a_treasure_dist": a_treasure_after,
            "b_treasure_dist": b_treasure_after,
            # Reward components
            "reward_a": reward_a,
            "reward_b": reward_b,
            "reward_outcome": reward_outcome_a,
            "reward_step": reward_step_a,
            "reward_hazard": reward_hazard_a,
            "reward_bump": reward_bump_a,
            "reward_perception": reward_perception_a,
            "reward_obstacle": reward_bump_a,
            "reward_chase": reward_chase_a,
            "reward_treasure": reward_treasure_a,
            "reward_outcome_a": reward_outcome_a,
            "reward_step_a": reward_step_a,
            "reward_hazard_a": reward_hazard_a,
            "reward_bump_a": reward_bump_a,
            "reward_perception_a": reward_perception_a,
            "reward_chase_a": reward_chase_a,
            "reward_treasure_a": reward_treasure_a,
            "reward_outcome_b": reward_outcome_b,
            "reward_step_b": reward_step_b,
            "reward_hazard_b": reward_hazard_b,
            "reward_bump_b": reward_bump_b,
            "reward_perception_b": reward_perception_b,
            "reward_evade_b": reward_evade_b,
            "reward_treasure_b": reward_treasure_b,
            "partial_observable": self.partial_observable,
            # Layout info (for logging)
            "wumpus_pos": self.layout.wumpus,
            "treasure_pos": self.layout.treasure,
            "pits": list(self.layout.pits),
        }
        
        return state, reward_a, done, info
    
    def get_extended_state(self) -> Dict:
        """
        Get extended state including perceptions for both agents.
        Useful for partial observability scenarios.
        """
        return {
            "positions": (self.a, self.b),
            "a_perception": self.get_perception(self.a),
            "b_perception": self.get_perception(self.b),
            "t": self.t,
        }
    
    def render_ascii(self) -> str:
        """Create ASCII visualization of current state."""
        lines = []
        for x in range(self.size):
            row = ""
            for y in range(self.size):
                pos = (x, y)
                if pos == self.a and pos == self.b:
                    cell = "X"  # Both agents
                elif pos == self.a:
                    cell = "A"
                elif pos == self.b:
                    cell = "B"
                elif pos == self.layout.wumpus:
                    cell = "W"
                elif pos == self.layout.treasure:
                    cell = "T"
                elif pos in self.layout.pits:
                    cell = "P"
                elif pos in self.layout.obstacles:
                    cell = "#"
                elif pos in self._breeze_cells:
                    cell = "~"  # Breeze
                elif pos in self._stench_cells:
                    cell = "!"  # Stench
                else:
                    cell = "."
                row += cell + " "
            lines.append(row)
        return "\n".join(lines)
