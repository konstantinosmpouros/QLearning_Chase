"""
Nash Q-Learning Agent for General-Sum Stochastic Games.

Nash Q-Learning extends Q-learning to multi-agent settings by:
1. Maintaining separate Q-tables for each player
2. Computing Nash equilibrium at each state to derive values
3. Updating Q-values using Nash equilibrium values as targets

Reference:
    Hu, J., & Wellman, M. P. (2003). Nash Q-learning for general-sum stochastic games.
    Journal of Machine Learning Research, 4, 1039-1069.
"""

from __future__ import annotations

import random
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Try to import nashpy for Nash equilibrium computation
try:
    import nashpy as nash
    NASHPY_AVAILABLE = True
except ImportError:
    NASHPY_AVAILABLE = False
    print("Warning: nashpy not available. Using Lemke-Howson fallback or support enumeration.")

from env.wumpus_env_extended import A


class NashType(Enum):
    """Types of Nash equilibrium to compute."""
    LEMKE_HOWSON = "lemke_howson"
    SUPPORT_ENUMERATION = "support_enumeration"
    VERTEX_ENUMERATION = "vertex_enumeration"
    FICTITIOUS_PLAY = "fictitious_play"


@dataclass
class NashResult:
    """Result of Nash equilibrium computation."""
    pi_a: np.ndarray      # Player A's mixed strategy
    pi_b: np.ndarray      # Player B's mixed strategy
    value_a: float        # Player A's expected value
    value_b: float        # Player B's expected value
    converged: bool       # Whether computation converged
    method: str           # Method used


def compute_nash_equilibrium_nashpy(
    Q_a: np.ndarray,
    Q_b: np.ndarray,
    method: NashType = NashType.LEMKE_HOWSON
) -> NashResult:
    """
    Compute Nash equilibrium using nashpy library.
    
    Args:
        Q_a: Payoff matrix for player A (row player), shape (n_actions_a, n_actions_b)
        Q_b: Payoff matrix for player B (column player), shape (n_actions_a, n_actions_b)
        method: Which algorithm to use
        
    Returns:
        NashResult with strategies and values
    """
    n_a, n_b = Q_a.shape
    
    # Helper to create uniform fallback
    def uniform_fallback(reason="fallback"):
        pi_a = np.ones(n_a, dtype=np.float64) / n_a
        pi_b = np.ones(n_b, dtype=np.float64) / n_b
        return NashResult(
            pi_a=pi_a,
            pi_b=pi_b,
            value_a=float(pi_a @ Q_a @ pi_b),
            value_b=float(pi_a @ Q_b @ pi_b),
            converged=False,
            method=reason
        )
    
    if not NASHPY_AVAILABLE:
        return compute_nash_equilibrium_fallback(Q_a, Q_b)
    
    game = nash.Game(Q_a, Q_b)
    equilibria = []
    
    try:
        if method == NashType.LEMKE_HOWSON:
            # Lemke-Howson algorithm (for non-degenerate games)
            for eq in game.lemke_howson_enumeration():
                equilibria.append(eq)
                break  # Take first equilibrium found
        elif method == NashType.SUPPORT_ENUMERATION:
            # Support enumeration (exact but slow)
            for eq in game.support_enumeration():
                equilibria.append(eq)
                break
        elif method == NashType.VERTEX_ENUMERATION:
            # Vertex enumeration
            for eq in game.vertex_enumeration():
                equilibria.append(eq)
                break
        elif method == NashType.FICTITIOUS_PLAY:
            # Fictitious play (approximate)
            iterations = 100  # Reduced for speed
            play_counts_a = np.zeros(n_a)
            play_counts_b = np.zeros(n_b)
            
            # Initialize with uniform
            pi_a = np.ones(n_a) / n_a
            pi_b = np.ones(n_b) / n_b
            
            for _ in range(iterations):
                # Best response for A given B's empirical distribution
                if play_counts_b.sum() > 0:
                    emp_b = play_counts_b / play_counts_b.sum()
                else:
                    emp_b = pi_b
                br_a = np.argmax(Q_a @ emp_b)
                play_counts_a[br_a] += 1
                
                # Best response for B given A's empirical distribution
                if play_counts_a.sum() > 0:
                    emp_a = play_counts_a / play_counts_a.sum()
                else:
                    emp_a = pi_a
                br_b = np.argmax(emp_a @ Q_b)
                play_counts_b[br_b] += 1
            
            pi_a = play_counts_a / play_counts_a.sum()
            pi_b = play_counts_b / play_counts_b.sum()
            equilibria.append((pi_a, pi_b))
    
    except Exception as e:
        # Fallback to uniform if computation fails
        return uniform_fallback("exception_uniform")
    
    if not equilibria:
        return uniform_fallback("no_equilibria_uniform")
    
    pi_a_raw, pi_b_raw = equilibria[0]
    
    # Ensure proper numpy arrays with correct dimensions
    pi_a = np.zeros(n_a, dtype=np.float64)
    pi_b = np.zeros(n_b, dtype=np.float64)
    
    # Handle potential dimension mismatches from nashpy
    try:
        pi_a_arr = np.array(pi_a_raw, dtype=np.float64).flatten()
        pi_b_arr = np.array(pi_b_raw, dtype=np.float64).flatten()
        
        # Copy values, handling size differences
        pi_a[:min(len(pi_a_arr), n_a)] = pi_a_arr[:min(len(pi_a_arr), n_a)]
        pi_b[:min(len(pi_b_arr), n_b)] = pi_b_arr[:min(len(pi_b_arr), n_b)]
    except Exception:
        return uniform_fallback("conversion_error_uniform")
    
    # Clip and renormalize to handle numerical issues
    pi_a = np.clip(pi_a, 0, 1)
    pi_b = np.clip(pi_b, 0, 1)
    
    # If all zeros (degenerate case), use uniform
    if pi_a.sum() < 1e-10:
        pi_a = np.ones(n_a, dtype=np.float64) / n_a
    else:
        pi_a = pi_a / pi_a.sum()
    
    if pi_b.sum() < 1e-10:
        pi_b = np.ones(n_b, dtype=np.float64) / n_b
    else:
        pi_b = pi_b / pi_b.sum()
    
    # Compute values at equilibrium
    value_a = float(pi_a @ Q_a @ pi_b)
    value_b = float(pi_a @ Q_b @ pi_b)
    
    return NashResult(
        pi_a=pi_a,
        pi_b=pi_b,
        value_a=value_a,
        value_b=value_b,
        converged=True,
        method=method.value
    )


def compute_nash_equilibrium_fallback(
    Q_a: np.ndarray,
    Q_b: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> NashResult:
    """
    Fallback Nash computation using iterative best response.
    This is an approximate method that may not converge for all games.
    """
    n_a, n_b = Q_a.shape
    
    # Initialize with uniform strategies
    pi_a = np.ones(n_a) / n_a
    pi_b = np.ones(n_b) / n_b
    
    for iteration in range(max_iterations):
        old_pi_a = pi_a.copy()
        old_pi_b = pi_b.copy()
        
        # Best response for A: maximize expected payoff given B's strategy
        expected_a = Q_a @ pi_b  # Expected payoff for each action of A
        br_a = np.argmax(expected_a)
        
        # Soft update towards best response
        learning_rate = 2.0 / (iteration + 2)
        new_pi_a = np.zeros(n_a)
        new_pi_a[br_a] = 1.0
        pi_a = (1 - learning_rate) * pi_a + learning_rate * new_pi_a
        
        # Best response for B: maximize expected payoff given A's strategy
        expected_b = pi_a @ Q_b  # Expected payoff for each action of B
        br_b = np.argmax(expected_b)
        
        new_pi_b = np.zeros(n_b)
        new_pi_b[br_b] = 1.0
        pi_b = (1 - learning_rate) * pi_b + learning_rate * new_pi_b
        
        # Check convergence
        if (np.abs(pi_a - old_pi_a).max() < tolerance and 
            np.abs(pi_b - old_pi_b).max() < tolerance):
            break
    
    # Normalize
    pi_a = pi_a / (pi_a.sum() + 1e-10)
    pi_b = pi_b / (pi_b.sum() + 1e-10)
    
    value_a = float(pi_a @ Q_a @ pi_b)
    value_b = float(pi_a @ Q_b @ pi_b)
    
    return NashResult(
        pi_a=pi_a,
        pi_b=pi_b,
        value_a=value_a,
        value_b=value_b,
        converged=iteration < max_iterations - 1,
        method="iterative_best_response"
    )


class NashQAgent:
    """
    Nash Q-Learning agent for general-sum stochastic games.
    
    Maintains separate Q-tables for each player and computes Nash equilibrium
    at each state to derive value estimates and mixed strategies.
    
    Key features:
    - Handles general-sum games (not just zero-sum)
    - Learns both players' strategies simultaneously
    - Uses Nash equilibrium as the solution concept
    
    Attributes:
        Q_a: Q-values for player A, Q_a[s, a_A, a_B] = expected return for A
        Q_b: Q-values for player B, Q_b[s, a_A, a_B] = expected return for B
        pi_cache_a: Cached mixed strategies for player A
        pi_cache_b: Cached mixed strategies for player B
    """
    
    def __init__(
        self,
        size: int,
        gamma: float = 0.95,
        alpha: float = 0.10,
        eps_start: float = 0.30,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 8_000,
        nash_method: NashType = NashType.FICTITIOUS_PLAY,  # Fastest
        seed: int = 0,
    ) -> None:
        """
        Initialize Nash Q-Learning agent.
        
        Args:
            size: Grid size (state space = size^4)
            gamma: Discount factor
            alpha: Learning rate
            eps_start: Initial exploration rate
            eps_end: Final exploration rate
            eps_decay_episodes: Episodes over which to decay epsilon
            nash_method: Algorithm for computing Nash equilibrium
            seed: Random seed
        """
        self.size = size
        self.n_states = (size * size) * (size * size)
        self.n_actions = A
        self.gamma = gamma
        self.alpha = alpha
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = max(1, eps_decay_episodes)
        self.nash_method = nash_method
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Q-tables for both players: Q[state, action_A, action_B]
        self.Q_a = np.zeros((self.n_states, A, A), dtype=np.float64)
        self.Q_b = np.zeros((self.n_states, A, A), dtype=np.float64)
        
        # Initialize with small random values to break symmetry
        self.Q_a += self.np_rng.randn(self.n_states, A, A) * 0.01
        self.Q_b += self.np_rng.randn(self.n_states, A, A) * 0.01
        
        # Strategy caches
        self.pi_cache_a = np.ones((self.n_states, A), dtype=np.float64) / A
        self.pi_cache_b = np.ones((self.n_states, A), dtype=np.float64) / A
        self.v_cache_a = np.zeros(self.n_states, dtype=np.float64)
        self.v_cache_b = np.zeros(self.n_states, dtype=np.float64)
        
        # Track which states need recomputation
        self.dirty = np.ones(self.n_states, dtype=bool)
        
        # Statistics
        self.nash_computations = 0
        self.nash_failures = 0
    
    def epsilon(self, episode: int) -> float:
        """Linear annealing of exploration rate."""
        t = min(1.0, episode / self.eps_decay_episodes)
        return (1 - t) * self.eps_start + t * self.eps_end
    
    def _compute_nash(self, sid: int) -> NashResult:
        """Compute Nash equilibrium for state sid."""
        self.nash_computations += 1
        
        result = compute_nash_equilibrium_nashpy(
            self.Q_a[sid],
            self.Q_b[sid],
            self.nash_method
        )
        
        if not result.converged:
            self.nash_failures += 1
        
        return result
    
    def _update_cache(self, sid: int) -> None:
        """Update strategy cache for state if dirty."""
        if self.dirty[sid]:
            result = self._compute_nash(sid)
            
            # Validate and store results
            pi_a = result.pi_a
            pi_b = result.pi_b
            
            # Check for NaN/Inf and fix if needed
            if np.any(np.isnan(pi_a)) or np.any(np.isinf(pi_a)) or pi_a.sum() < 1e-10:
                pi_a = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
            
            if np.any(np.isnan(pi_b)) or np.any(np.isinf(pi_b)) or pi_b.sum() < 1e-10:
                pi_b = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
            
            self.pi_cache_a[sid] = pi_a
            self.pi_cache_b[sid] = pi_b
            
            # Recompute values with validated policies
            v_a = float(pi_a @ self.Q_a[sid] @ pi_b)
            v_b = float(pi_a @ self.Q_b[sid] @ pi_b)
            
            # Check for NaN values
            if np.isnan(v_a) or np.isinf(v_a):
                v_a = 0.0
            if np.isnan(v_b) or np.isinf(v_b):
                v_b = 0.0
            
            self.v_cache_a[sid] = v_a
            self.v_cache_b[sid] = v_b
            self.dirty[sid] = False
    
    def get_policies(self, sid: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Get Nash equilibrium strategies and values for state.
        
        Returns:
            (pi_a, pi_b, value_a, value_b)
        """
        self._update_cache(sid)
        return (
            self.pi_cache_a[sid].copy(),
            self.pi_cache_b[sid].copy(),
            self.v_cache_a[sid],
            self.v_cache_b[sid]
        )
    
    def act_a(self, sid: int, eps: float) -> int:
        """
        Choose action for player A using epsilon-greedy on Nash strategy.
        """
        if self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)
        
        self._update_cache(sid)
        pi = self.pi_cache_a[sid].copy()
        
        # Handle numerical issues - check for NaN or invalid values
        if np.any(np.isnan(pi)) or np.any(np.isinf(pi)) or pi.sum() < 1e-10:
            # Fallback to uniform distribution
            pi = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
        else:
            # Clip and renormalize
            pi = np.clip(pi, 0, 1)
            pi = pi / pi.sum()
        
        return int(self.np_rng.choice(self.n_actions, p=pi))
    
    def act_b(self, sid: int, eps: float) -> int:
        """
        Choose action for player B using epsilon-greedy on Nash strategy.
        """
        if self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)
        
        self._update_cache(sid)
        pi = self.pi_cache_b[sid].copy()
        
        # Handle numerical issues - check for NaN or invalid values
        if np.any(np.isnan(pi)) or np.any(np.isinf(pi)) or pi.sum() < 1e-10:
            # Fallback to uniform distribution
            pi = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
        else:
            # Clip and renormalize
            pi = np.clip(pi, 0, 1)
            pi = pi / pi.sum()
        
        return int(self.np_rng.choice(self.n_actions, p=pi))
    
    def update(
        self,
        sid: int,
        a_a: int,
        a_b: int,
        r_a: float,
        r_b: float,
        sid_next: int,
        done: bool
    ) -> None:
        """
        Update Q-values for both players using Nash Q-Learning update rule.
        
        Q_i(s, a_A, a_B) ← (1-α) Q_i(s, a_A, a_B) + α [r_i + γ Nash_i(s')]
        
        where Nash_i(s') is the Nash equilibrium value for player i at state s'.
        
        Args:
            sid: Current state id
            a_a: Player A's action
            a_b: Player B's action
            r_a: Reward for player A
            r_b: Reward for player B
            sid_next: Next state id
            done: Whether episode is done
        """
        if done:
            target_a = r_a
            target_b = r_b
        else:
            # Get Nash values at next state
            self._update_cache(sid_next)
            v_next_a = self.v_cache_a[sid_next]
            v_next_b = self.v_cache_b[sid_next]
            
            target_a = r_a + self.gamma * v_next_a
            target_b = r_b + self.gamma * v_next_b
        
        # Q-learning update for both players
        self.Q_a[sid, a_a, a_b] = (
            (1 - self.alpha) * self.Q_a[sid, a_a, a_b] + 
            self.alpha * target_a
        )
        self.Q_b[sid, a_a, a_b] = (
            (1 - self.alpha) * self.Q_b[sid, a_a, a_b] + 
            self.alpha * target_b
        )
        
        # Mark state as dirty (needs Nash recomputation)
        self.dirty[sid] = True
    
    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "nash_computations": self.nash_computations,
            "nash_failures": self.nash_failures,
            "failure_rate": self.nash_failures / max(1, self.nash_computations),
            "dirty_states": self.dirty.sum(),
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.nash_computations = 0
        self.nash_failures = 0


class NashQAgentSinglePlayer:
    """
    Wrapper for using NashQAgent from single player's perspective.
    Useful for training against different opponents.
    """
    
    def __init__(self, nash_agent: NashQAgent, player: str = "A"):
        """
        Args:
            nash_agent: Underlying Nash Q-Learning agent
            player: Which player this wrapper represents ("A" or "B")
        """
        self.agent = nash_agent
        self.player = player
    
    def act(self, sid: int, eps: float) -> int:
        """Choose action for this player."""
        if self.player == "A":
            return self.agent.act_a(sid, eps)
        else:
            return self.agent.act_b(sid, eps)
    
    def get_policy(self, sid: int) -> np.ndarray:
        """Get Nash strategy for this player."""
        pi_a, pi_b, _, _ = self.agent.get_policies(sid)
        return pi_a if self.player == "A" else pi_b
    
    def get_value(self, sid: int) -> float:
        """Get Nash value for this player."""
        _, _, v_a, v_b = self.agent.get_policies(sid)
        return v_a if self.player == "A" else v_b


if __name__ == "__main__":
    # Test Nash equilibrium computation
    print("Testing Nash Q-Learning components...\n")
    
    # Test 1: Prisoner's Dilemma
    print("=== Prisoner's Dilemma ===")
    # Payoffs: (Cooperate, Defect) x (Cooperate, Defect)
    Q_a = np.array([
        [-1, -3],  # A cooperates
        [0, -2]    # A defects
    ], dtype=np.float64)
    Q_b = np.array([
        [-1, 0],   # B cooperates
        [-3, -2]   # B defects
    ], dtype=np.float64)
    
    result = compute_nash_equilibrium_nashpy(Q_a, Q_b)
    print(f"Player A strategy: {result.pi_a}")
    print(f"Player B strategy: {result.pi_b}")
    print(f"Values: A={result.value_a:.3f}, B={result.value_b:.3f}")
    print(f"Method: {result.method}, Converged: {result.converged}")
    
    # Test 2: Matching Pennies (zero-sum)
    print("\n=== Matching Pennies ===")
    Q_a = np.array([
        [1, -1],
        [-1, 1]
    ], dtype=np.float64)
    Q_b = -Q_a  # Zero-sum
    
    result = compute_nash_equilibrium_nashpy(Q_a, Q_b)
    print(f"Player A strategy: {result.pi_a}")
    print(f"Player B strategy: {result.pi_b}")
    print(f"Values: A={result.value_a:.3f}, B={result.value_b:.3f}")
    
    # Test 3: NashQAgent
    print("\n=== NashQAgent Test ===")
    agent = NashQAgent(size=3, seed=42)
    
    # Simulate a few updates
    for i in range(10):
        sid = agent.rng.randint(0, agent.n_states - 1)
        a_a = agent.act_a(sid, eps=0.5)
        a_b = agent.act_b(sid, eps=0.5)
        
        # Fake rewards
        r_a = agent.rng.uniform(-1, 1)
        r_b = agent.rng.uniform(-1, 1)
        
        sid_next = agent.rng.randint(0, agent.n_states - 1)
        done = agent.rng.random() < 0.1
        
        agent.update(sid, a_a, a_b, r_a, r_b, sid_next, done)
    
    stats = agent.get_stats()
    print(f"Nash computations: {stats['nash_computations']}")
    print(f"Nash failures: {stats['nash_failures']}")
    print(f"Failure rate: {stats['failure_rate']:.2%}")
    
    print("\nAll tests completed!")
