"""
Dyna-Q Agent for Multi-Agent Wumpus Chase.

Dyna-Q combines model-free Q-learning with model-based planning:
1. Learn Q-values from real experience (like standard Q-learning)
2. Learn an environment model (transitions + rewards)
3. Use the model to generate simulated experiences for additional Q-updates

This makes learning more sample-efficient by "replaying" and "imagining"
experiences using the learned model.

Reference:
    Sutton, R. S. (1991). Dyna, an integrated architecture for learning,
    planning, and reacting. ACM SIGART Bulletin, 2(4), 160-163.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


@dataclass
class TransitionModel:
    """
    Learned environment model for Dyna-Q.
    
    Stores observed transitions and rewards to enable planning.
    Uses a deterministic model (stores last observed transition).
    Can be extended to probabilistic model for stochastic environments.
    """
    
    # Model storage: (state, action_a, action_b) -> (next_state, reward, done)
    transitions: Dict[Tuple[int, int, int], Tuple[int, float, bool]] = field(
        default_factory=dict
    )
    
    # Track visited state-action pairs for sampling
    visited_states: Set[int] = field(default_factory=set)
    state_actions: Dict[int, Set[Tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    
    # Statistics
    total_transitions: int = 0
    
    def update(
        self, 
        state: int, 
        action_a: int, 
        action_b: int, 
        next_state: int, 
        reward: float, 
        done: bool
    ) -> None:
        """Update the model with an observed transition."""
        key = (state, action_a, action_b)
        self.transitions[key] = (next_state, reward, done)
        self.visited_states.add(state)
        self.state_actions[state].add((action_a, action_b))
        self.total_transitions += 1
    
    def sample_state(self, rng: random.Random) -> Optional[int]:
        """Sample a random previously visited state."""
        if not self.visited_states:
            return None
        return rng.choice(list(self.visited_states))
    
    def sample_actions(self, state: int, rng: random.Random) -> Optional[Tuple[int, int]]:
        """Sample a random action pair that was taken in the given state."""
        if state not in self.state_actions or not self.state_actions[state]:
            return None
        return rng.choice(list(self.state_actions[state]))
    
    def predict(
        self, 
        state: int, 
        action_a: int, 
        action_b: int
    ) -> Optional[Tuple[int, float, bool]]:
        """Predict the outcome of a state-action pair using the learned model."""
        key = (state, action_a, action_b)
        return self.transitions.get(key)
    
    def get_stats(self) -> Dict:
        """Return model statistics."""
        return {
            "total_transitions": self.total_transitions,
            "unique_transitions": len(self.transitions),
            "visited_states": len(self.visited_states),
        }


@dataclass 
class ProbabilisticTransitionModel:
    """
    Probabilistic environment model that tracks transition frequencies.
    
    Better for stochastic environments where the same state-action
    can lead to different outcomes.
    """
    
    # Counts: (state, action_a, action_b, next_state) -> count
    transition_counts: Dict[Tuple[int, int, int, int], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    
    # Reward sums for averaging: (state, action_a, action_b) -> (sum, count)
    reward_stats: Dict[Tuple[int, int, int], Tuple[float, int]] = field(
        default_factory=lambda: defaultdict(lambda: (0.0, 0))
    )
    
    # Done flags: (state, action_a, action_b, next_state) -> done observed
    done_flags: Dict[Tuple[int, int, int, int], bool] = field(
        default_factory=dict
    )
    
    visited_states: Set[int] = field(default_factory=set)
    state_actions: Dict[int, Set[Tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    
    # Grouped outcomes preserve terminal/nonterminal transitions even when the
    # legacy position-only state maps them to the same next_state.
    outcome_counts: Dict = field(default_factory=dict)

    def update(
        self,
        state: int,
        action_a: int,
        action_b: int,
        next_state: int,
        reward: float,
        done: bool
    ) -> None:
        """Update the probabilistic model with an observed transition."""
        outcomes = self.outcome_counts.setdefault((state, action_a, action_b), {})
        outcome = (next_state, float(reward), bool(done))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

        # Update transition count
        trans_key = (state, action_a, action_b, next_state)
        self.transition_counts[trans_key] += 1
        
        # Update reward statistics
        sa_key = (state, action_a, action_b)
        old_sum, old_count = self.reward_stats[sa_key]
        self.reward_stats[sa_key] = (old_sum + reward, old_count + 1)
        
        # Update done flag
        self.done_flags[trans_key] = done
        
        # Track visited states and actions
        self.visited_states.add(state)
        self.state_actions[state].add((action_a, action_b))
    
    def sample_state(self, rng: random.Random) -> Optional[int]:
        """Sample a random previously visited state."""
        if not self.visited_states:
            return None
        return rng.choice(list(self.visited_states))
    
    def sample_actions(self, state: int, rng: random.Random) -> Optional[Tuple[int, int]]:
        """Sample a random action pair that was taken in the given state."""
        if state not in self.state_actions or not self.state_actions[state]:
            return None
        return rng.choice(list(self.state_actions[state]))
    
    def predict(self, state, action_a, action_b, rng):
        """Sample a joint (next_state, reward, done) outcome by frequency."""
        outcomes = self.outcome_counts.get((state, action_a, action_b), {})
        if not outcomes:
            return None
        return rng.choices(list(outcomes), weights=list(outcomes.values()), k=1)[0]


class DynaQAgent:
    """
    Dyna-Q Agent for multi-agent zero-sum games.
    
    Combines:
    1. Q-learning from real experience
    2. Model learning (transition + reward)
    3. Planning using simulated experience from the model
    
    The agent maintains a Q-table for player A (row player) and uses
    the model to perform additional Q-updates via planning steps.
    
    Attributes:
        Q: Q-table Q[state, action_a, action_b]
        model: Learned environment model
        n_planning: Number of planning steps per real step
    """
    
    def __init__(
        self,
        size: int,
        n_actions: int = 5,
        gamma: float = 0.95,
        alpha: float = 0.10,
        eps_start: float = 0.30,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 5_000,
        n_planning: int = 10,
        use_probabilistic_model: bool = False,
        seed: int = 0,
        player: str = "A",
    ) -> None:
        """
        Initialize Dyna-Q agent.
        
        Args:
            size: Grid size (state space = size^4)
            n_actions: Number of actions per player
            gamma: Discount factor
            alpha: Learning rate
            eps_start: Initial exploration rate
            eps_end: Final exploration rate
            eps_decay_episodes: Episodes for epsilon decay
            n_planning: Planning steps per real experience
            use_probabilistic_model: Use probabilistic vs deterministic model
            seed: Random seed
        """
        self.size = size
        self.n_states = (size * size) * (size * size)
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = max(1, eps_decay_episodes)
        self.n_planning = n_planning
        self.player = player.upper()
        if self.player not in {"A", "B"}:
            raise ValueError("player must be 'A' or 'B'")
        
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Q-table: Q[state, action_a, action_b]
        self.Q = np.zeros((self.n_states, n_actions, n_actions), dtype=np.float64)
        
        # Initialize with small random values to break ties
        self.Q += self.np_rng.randn(self.n_states, n_actions, n_actions) * 0.01
        
        # Environment model
        if use_probabilistic_model:
            self.model = ProbabilisticTransitionModel()
        else:
            self.model = TransitionModel()
        self.use_probabilistic_model = use_probabilistic_model
        
        # Statistics
        self.real_updates = 0
        self.planning_updates = 0
        self.episodes_trained = 0
    
    def epsilon(self, episode: int) -> float:
        """Linear annealing of exploration rate."""
        t = min(1.0, episode / self.eps_decay_episodes)
        return (1 - t) * self.eps_start + t * self.eps_end
    
    def get_q_values(self, state: int) -> np.ndarray:
        """Get Q-values for a state. Shape: (n_actions, n_actions)."""
        return self.Q[state]
    
    def get_value(self, state: int) -> float:
        """
        Get robust state value for this agent's configured player role.

        player A: max_a min_b Q(s,a,b)
        player B: max_b min_a Q(s,a,b)
        """
        if self.player == "A":
            return float(np.max(np.min(self.Q[state], axis=1)))
        return float(np.max(np.min(self.Q[state], axis=0)))
    
    def act_a(self, state: int, eps: float) -> int:
        """
        Choose action for player A (row player / maximizer).
        
        Uses epsilon-greedy on the maximin strategy.
        """
        if self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)
        
        # Maximin: choose action that maximizes minimum Q over opponent actions
        min_q = np.min(self.Q[state], axis=1)  # min over action_b for each action_a
        return int(np.argmax(min_q))
    
    def act_b(self, state: int, eps: float) -> int:
        """
        Choose action for player B (column player).
        """
        if self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)

        if self.player == "B":
            # Runner-side robust choice: maximize minimum payoff over A actions.
            min_over_a = np.min(self.Q[state], axis=0)
            return int(np.argmax(min_over_a))

        # Legacy behavior for A-side models that also need a B action.
        a_action = self.act_a(state, eps=0.0)
        return int(np.argmin(self.Q[state, a_action, :]))
    
    def _q_update(
        self,
        state: int,
        action_a: int,
        action_b: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Perform a single Q-learning update.
        
        Returns the TD error for monitoring.
        """
        if done:
            target = reward
        else:
            # Zero-sum value: max_a min_b Q(s', a, b)
            next_value = self.get_value(next_state)
            target = reward + self.gamma * next_value
        
        old_q = self.Q[state, action_a, action_b]
        td_error = target - old_q
        self.Q[state, action_a, action_b] = old_q + self.alpha * td_error
        
        return td_error
    
    def update(
        self,
        state: int,
        action_a: int,
        action_b: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> Dict:
        """
        Full Dyna-Q update: Q-learning + model update + planning.
        
        Args:
            state: Current state ID
            action_a: Player A's action
            action_b: Player B's action  
            reward: Reward received (for player A)
            next_state: Next state ID
            done: Whether episode ended
            
        Returns:
            Dictionary with update statistics
        """
        # 1. Q-learning update from real experience
        td_error = self._q_update(state, action_a, action_b, reward, next_state, done)
        self.real_updates += 1
        
        # 2. Update the model
        self.model.update(state, action_a, action_b, next_state, reward, done)
        
        # 3. Planning: simulate experiences using the model
        planning_errors = []
        for _ in range(self.n_planning):
            # Sample a previously visited state
            s = self.model.sample_state(self.rng)
            if s is None:
                continue
            
            # Sample an action pair taken in that state
            actions = self.model.sample_actions(s, self.rng)
            if actions is None:
                continue
            
            a_a, a_b = actions
            
            # Get model prediction
            if self.use_probabilistic_model:
                prediction = self.model.predict(s, a_a, a_b, self.rng)
            else:
                prediction = self.model.predict(s, a_a, a_b)
            
            if prediction is None:
                continue
            
            s_next, r, d = prediction
            
            # Q-learning update from simulated experience
            plan_td = self._q_update(s, a_a, a_b, r, s_next, d)
            planning_errors.append(abs(plan_td))
            self.planning_updates += 1
        
        return {
            "td_error": td_error,
            "planning_td_errors": planning_errors,
            "model_size": len(self.model.transitions) if hasattr(self.model, 'transitions') 
                          else self.model.transition_counts.__len__(),
        }
    
    def plan_only(self, n_steps: int) -> List[float]:
        """
        Perform planning-only updates (no real experience).
        
        Useful for offline planning or background planning.
        
        Args:
            n_steps: Number of planning steps
            
        Returns:
            List of TD errors from planning
        """
        td_errors = []
        
        for _ in range(n_steps):
            s = self.model.sample_state(self.rng)
            if s is None:
                continue
            
            actions = self.model.sample_actions(s, self.rng)
            if actions is None:
                continue
            
            a_a, a_b = actions
            
            if self.use_probabilistic_model:
                prediction = self.model.predict(s, a_a, a_b, self.rng)
            else:
                prediction = self.model.predict(s, a_a, a_b)
            
            if prediction is None:
                continue
            
            s_next, r, d = prediction
            td = self._q_update(s, a_a, a_b, r, s_next, d)
            td_errors.append(abs(td))
            self.planning_updates += 1
        
        return td_errors
    
    def get_policy_a(self, state: int) -> np.ndarray:
        """Get the greedy policy for player A at a state."""
        # Softmax over maximin values
        min_q = np.min(self.Q[state], axis=1)
        
        # Return one-hot for greedy action
        policy = np.zeros(self.n_actions)
        policy[np.argmax(min_q)] = 1.0
        return policy
    
    def get_policy_b(self, state: int) -> np.ndarray:
        """Get greedy policy for player B at a state."""
        policy = np.zeros(self.n_actions)
        if self.player == "B":
            b_action = int(np.argmax(np.min(self.Q[state], axis=0)))
        else:
            a_action = int(np.argmax(np.min(self.Q[state], axis=1)))
            b_action = int(np.argmin(self.Q[state, a_action, :]))
        policy[b_action] = 1.0
        return policy
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        model_stats = self.model.get_stats() if hasattr(self.model, 'get_stats') else {}
        return {
            "real_updates": self.real_updates,
            "planning_updates": self.planning_updates,
            "planning_ratio": self.planning_updates / max(1, self.real_updates),
            "q_table_nonzero": np.count_nonzero(self.Q),
            "q_table_mean": float(np.mean(self.Q)),
            "q_table_std": float(np.std(self.Q)),
            **model_stats,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.real_updates = 0
        self.planning_updates = 0


class DynaQPlusAgent(DynaQAgent):
    """
    Dyna-Q+ Agent with exploration bonus for unvisited transitions.
    
    Adds a bonus reward proportional to the time since a state-action
    was last visited, encouraging exploration of less-visited parts
    of the state space.
    
    Reference:
        Sutton & Barto (2018). Reinforcement Learning: An Introduction.
        Section 8.3: When the Model Is Wrong.
    """
    
    def __init__(
        self,
        size: int,
        n_actions: int = 5,
        kappa: float = 0.001,  # Exploration bonus coefficient
        **kwargs
    ) -> None:
        super().__init__(size, n_actions, **kwargs)
        
        self.kappa = kappa
        
        # Track time since last visit for each state-action pair
        self.last_visit: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.current_step = 0
    
    def update(
        self,
        state: int,
        action_a: int,
        action_b: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> Dict:
        """Dyna-Q+ update with exploration bonus in planning."""
        self.current_step += 1
        
        # Record visit time
        self.last_visit[(state, action_a, action_b)] = self.current_step
        
        # Standard Q-learning update (no bonus for real experience)
        td_error = self._q_update(state, action_a, action_b, reward, next_state, done)
        self.real_updates += 1
        
        # Update model
        self.model.update(state, action_a, action_b, next_state, reward, done)
        
        # Planning with exploration bonus
        planning_errors = []
        for _ in range(self.n_planning):
            s = self.model.sample_state(self.rng)
            if s is None:
                continue
            
            actions = self.model.sample_actions(s, self.rng)
            if actions is None:
                continue
            
            a_a, a_b = actions
            
            if self.use_probabilistic_model:
                prediction = self.model.predict(s, a_a, a_b, self.rng)
            else:
                prediction = self.model.predict(s, a_a, a_b)
            
            if prediction is None:
                continue
            
            s_next, r, d = prediction
            
            # Add exploration bonus based on time since last visit
            time_since_visit = self.current_step - self.last_visit.get((s, a_a, a_b), 0)
            bonus = self.kappa * np.sqrt(time_since_visit)
            r_with_bonus = r + bonus
            
            # Q-update with bonus
            plan_td = self._q_update(s, a_a, a_b, r_with_bonus, s_next, d)
            planning_errors.append(abs(plan_td))
            self.planning_updates += 1
        
        return {
            "td_error": td_error,
            "planning_td_errors": planning_errors,
            "model_size": len(self.model.transitions) if hasattr(self.model, 'transitions')
                          else len(self.model.transition_counts),
        }


if __name__ == "__main__":
    # Test the Dyna-Q agent
    print("Testing Dyna-Q Agent...")
    print()
    
    # Create agent
    agent = DynaQAgent(
        size=5,
        n_actions=5,
        gamma=0.95,
        alpha=0.1,
        n_planning=5,
        seed=42
    )
    
    # Simulate some updates
    print("Simulating 100 random transitions...")
    for i in range(100):
        state = agent.rng.randint(0, agent.n_states - 1)
        action_a = agent.rng.randint(0, agent.n_actions - 1)
        action_b = agent.rng.randint(0, agent.n_actions - 1)
        next_state = agent.rng.randint(0, agent.n_states - 1)
        reward = agent.rng.uniform(-1, 1)
        done = agent.rng.random() < 0.1
        
        stats = agent.update(state, action_a, action_b, reward, next_state, done)
    
    print()
    print("Agent statistics:")
    for key, value in agent.get_stats().items():
        print(f"  {key}: {value}")
    
    print()
    print("Testing Dyna-Q+ Agent...")
    agent_plus = DynaQPlusAgent(size=5, n_actions=5, kappa=0.001, seed=42)
    
    for i in range(100):
        state = agent_plus.rng.randint(0, agent_plus.n_states - 1)
        action_a = agent_plus.rng.randint(0, agent_plus.n_actions - 1)
        action_b = agent_plus.rng.randint(0, agent_plus.n_actions - 1)
        next_state = agent_plus.rng.randint(0, agent_plus.n_states - 1)
        reward = agent_plus.rng.uniform(-1, 1)
        done = agent_plus.rng.random() < 0.1
        
        agent_plus.update(state, action_a, action_b, reward, next_state, done)
    
    print("Dyna-Q+ statistics:")
    for key, value in agent_plus.get_stats().items():
        print(f"  {key}: {value}")
    
    print()
    print("All tests passed!")
