"""
Deep Q-Network (DQN) Agent for Multi-Agent Wumpus Chase.

Implements:
- Standard DQN with experience replay and target network
- Double DQN for reduced overestimation
- Dueling DQN architecture option
- Multi-agent variants (Independent DQN, Self-play DQN)

Reference:
    Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
    Van Hasselt, H., et al. (2016). Deep reinforcement learning with double Q-learning. AAAI.
    Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML.

Note: This module requires PyTorch. Install with: pip install torch
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, NamedTuple
from enum import Enum

import numpy as np

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    torch = None
    nn = None
    F = None
    optim = None


class DQNType(Enum):
    """Types of DQN architectures."""
    STANDARD = "standard"
    DOUBLE = "double"
    DUELING = "dueling"
    DUELING_DOUBLE = "dueling_double"


class Transition(NamedTuple):
    """A single transition in the replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores transitions and samples random minibatches for training.
    """
    
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


# Only define neural network classes if PyTorch is available
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """
        Standard Q-Network architecture.
        
        Input: State representation (flattened grid + features)
        Output: Q-values for each action
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
            super().__init__()
            
            if hidden_dims is None:
                hidden_dims = [128, 128]
            
            layers = []
            prev_dim = state_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, action_dim))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


    class DuelingQNetwork(nn.Module):
        """
        Dueling Q-Network architecture.
        
        Separates value and advantage streams:
        Q(s, a) = V(s) + A(s, a) - mean(A(s, .))
        
        This helps the network learn which states are valuable
        without having to learn the effect of each action.
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
            super().__init__()
            
            if hidden_dims is None:
                hidden_dims = [128, 128]
            
            # Shared feature extractor
            shared_layers = []
            prev_dim = state_dim
            
            for i, hidden_dim in enumerate(hidden_dims[:-1]):
                shared_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ])
                prev_dim = hidden_dim
            
            self.shared = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()
            
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.shared(x)
            
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Q = V + (A - mean(A))
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            
            return q_values

else:
    # Dummy classes when PyTorch is not available
    QNetwork = None
    DuelingQNetwork = None


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    # Network
    state_dim: int = 0  # Will be set based on environment
    action_dim: int = 5
    hidden_dims: List[int] = None
    dqn_type: DQNType = DQNType.DOUBLE
    
    # Training
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    
    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    
    # Replay buffer
    buffer_size: int = 100_000
    min_buffer_size: int = 1000  # Minimum samples before training
    
    # Target network
    target_update_freq: int = 100  # Steps between target network updates
    tau: float = 1.0  # Soft update coefficient (1.0 = hard update)
    
    # Device
    device: str = "cpu"  # "cuda" or "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]


class DQNAgent:
    """
    Deep Q-Network Agent.
    
    Supports:
    - Standard DQN
    - Double DQN (reduces overestimation)
    - Dueling DQN (separate value/advantage)
    - Dueling Double DQN (both)
    """
    
    def __init__(self, config: DQNConfig, seed: int = 0):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DQN agent")
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Create networks
        NetworkClass = DuelingQNetwork if "dueling" in config.dqn_type.value else QNetwork
        
        self.q_network = NetworkClass(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        self.target_network = NetworkClass(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.steps_done = 0
        self.episodes_done = 0
        
        # Statistics
        self.train_losses = []
        self.q_values_history = []
    
    def get_epsilon(self) -> float:
        """Get current exploration rate with linear decay."""
        progress = min(1.0, self.steps_done / self.config.eps_decay_steps)
        return self.config.eps_start + progress * (self.config.eps_end - self.config.eps_start)
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (if None, use current schedule)
            
        Returns:
            Selected action index
        """
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            return random.randrange(self.config.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        
        # Sample batch
        transitions = self.replay_buffer.sample(self.config.batch_size)
        
        # Unpack batch
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        dones = torch.FloatTensor([t.done for t in transitions]).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            if "double" in self.config.dqn_type.value:
                # Double DQN: use online network to select action, target to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.config.target_update_freq == 0:
            self._update_target_network()
        
        loss_value = loss.item()
        self.train_losses.append(loss_value)
        
        return loss_value
    
    def _update_target_network(self) -> None:
        """Update target network (hard or soft update)."""
        if self.config.tau == 1.0:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft update
            for target_param, param in zip(
                self.target_network.parameters(),
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions at given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'config': self.config,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.train_losses[-100:]) if self.train_losses else 0.0,
        }


class MultiAgentDQN:
    """
    Multi-Agent DQN for two-player games.
    
    Supports:
    - Independent DQN: Each agent has separate Q-network
    - Self-play DQN: Single network, symmetric game
    - Centralized DQN: Shared network with agent-specific inputs
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        mode: str = "self_play",  # "independent", "self_play", "centralized"
        config: Optional[DQNConfig] = None,
        seed: int = 0,
    ):
        self.mode = mode
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if config is None:
            config = DQNConfig(
                state_dim=state_dim,
                action_dim=action_dim,
            )
        else:
            config.state_dim = state_dim
            config.action_dim = action_dim
        
        if mode == "independent":
            # Two separate agents
            self.agent_a = DQNAgent(config, seed=seed)
            self.agent_b = DQNAgent(config, seed=seed + 1000)
        elif mode == "self_play":
            # Single agent for symmetric self-play
            self.agent = DQNAgent(config, seed=seed)
        elif mode == "centralized":
            # Extended state includes agent indicator
            config_extended = DQNConfig(
                state_dim=state_dim + 1,  # +1 for agent indicator
                action_dim=action_dim,
                hidden_dims=config.hidden_dims,
                dqn_type=config.dqn_type,
                gamma=config.gamma,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                eps_start=config.eps_start,
                eps_end=config.eps_end,
                eps_decay_steps=config.eps_decay_steps,
                buffer_size=config.buffer_size,
                target_update_freq=config.target_update_freq,
            )
            self.agent = DQNAgent(config_extended, seed=seed)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def select_action_a(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action for agent A."""
        if self.mode == "independent":
            return self.agent_a.select_action(state, epsilon)
        elif self.mode == "self_play":
            return self.agent.select_action(state, epsilon)
        else:  # centralized
            extended_state = np.append(state, 0.0)  # Agent A indicator
            return self.agent.select_action(extended_state, epsilon)
    
    def select_action_b(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action for agent B."""
        if self.mode == "independent":
            return self.agent_b.select_action(state, epsilon)
        elif self.mode == "self_play":
            # For self-play, flip the state perspective
            flipped_state = self._flip_state(state)
            return self.agent.select_action(flipped_state, epsilon)
        else:  # centralized
            extended_state = np.append(state, 1.0)  # Agent B indicator
            return self.agent.select_action(extended_state, epsilon)
    
    def _flip_state(self, state: np.ndarray) -> np.ndarray:
        """Flip state for symmetric self-play (swap A and B positions)."""
        # Assuming state format: [a_x, a_y, b_x, b_y, ...features...]
        flipped = state.copy()
        # Swap first 2 elements with next 2
        flipped[0], flipped[2] = state[2], state[0]
        flipped[1], flipped[3] = state[3], state[1]
        return flipped
    
    def store_transition_a(self, state: np.ndarray, action: int, reward: float,
                          next_state: np.ndarray, done: bool) -> None:
        """Store transition for agent A."""
        if self.mode == "independent":
            self.agent_a.store_transition(state, action, reward, next_state, done)
        elif self.mode == "self_play":
            self.agent.store_transition(state, action, reward, next_state, done)
        else:
            extended_state = np.append(state, 0.0)
            extended_next = np.append(next_state, 0.0)
            self.agent.store_transition(extended_state, action, reward, extended_next, done)
    
    def store_transition_b(self, state: np.ndarray, action: int, reward: float,
                          next_state: np.ndarray, done: bool) -> None:
        """Store transition for agent B."""
        if self.mode == "independent":
            self.agent_b.store_transition(state, action, reward, next_state, done)
        elif self.mode == "self_play":
            flipped = self._flip_state(state)
            flipped_next = self._flip_state(next_state)
            self.agent.store_transition(flipped, action, reward, flipped_next, done)
        else:
            extended_state = np.append(state, 1.0)
            extended_next = np.append(next_state, 1.0)
            self.agent.store_transition(extended_state, action, reward, extended_next, done)
    
    def train_step(self) -> dict:
        """Perform training step for all agents."""
        losses = {}
        
        if self.mode == "independent":
            loss_a = self.agent_a.train_step()
            loss_b = self.agent_b.train_step()
            losses['agent_a'] = loss_a
            losses['agent_b'] = loss_b
        else:
            loss = self.agent.train_step()
            losses['agent'] = loss
        
        return losses
    
    def get_epsilon(self) -> float:
        """Get current exploration rate."""
        if self.mode == "independent":
            return self.agent_a.get_epsilon()
        else:
            return self.agent.get_epsilon()
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        if self.mode == "independent":
            return {
                'agent_a': self.agent_a.get_stats(),
                'agent_b': self.agent_b.get_stats(),
            }
        else:
            return self.agent.get_stats()
    
    def save(self, path: str) -> None:
        """Save all models."""
        if self.mode == "independent":
            self.agent_a.save(f"{path}_agent_a.pt")
            self.agent_b.save(f"{path}_agent_b.pt")
        else:
            self.agent.save(f"{path}.pt")
    
    def load(self, path: str) -> None:
        """Load all models."""
        if self.mode == "independent":
            self.agent_a.load(f"{path}_agent_a.pt")
            self.agent_b.load(f"{path}_agent_b.pt")
        else:
            self.agent.load(f"{path}.pt")


def create_state_representation(
    env_state: Tuple[int, int, int, int],
    env_size: int,
    layout=None,
    include_features: bool = True,
) -> np.ndarray:
    """
    Create neural network input from environment state.
    
    Args:
        env_state: (a_x, a_y, b_x, b_y) positions
        env_size: Grid size
        layout: Optional map layout for additional features
        include_features: Whether to include hand-crafted features
        
    Returns:
        Flattened state representation
    """
    a_x, a_y, b_x, b_y = env_state
    
    # Normalize positions to [0, 1]
    norm_factor = env_size - 1
    state = [
        a_x / norm_factor,
        a_y / norm_factor,
        b_x / norm_factor,
        b_y / norm_factor,
    ]
    
    if include_features:
        # Manhattan distance between agents (normalized)
        dist_ab = (abs(a_x - b_x) + abs(a_y - b_y)) / (2 * norm_factor)
        state.append(dist_ab)
        
        if layout is not None:
            # Distance to treasure (normalized)
            tx, ty = layout.treasure
            dist_a_treasure = (abs(a_x - tx) + abs(a_y - ty)) / (2 * norm_factor)
            dist_b_treasure = (abs(b_x - tx) + abs(b_y - ty)) / (2 * norm_factor)
            state.extend([dist_a_treasure, dist_b_treasure])
            
            # Distance to wumpus (normalized)
            wx, wy = layout.wumpus
            dist_a_wumpus = (abs(a_x - wx) + abs(a_y - wy)) / (2 * norm_factor)
            dist_b_wumpus = (abs(b_x - wx) + abs(b_y - wy)) / (2 * norm_factor)
            state.extend([dist_a_wumpus, dist_b_wumpus])
            
            # Check if in breeze/stench zone
            if hasattr(layout, 'get_breeze_cells'):
                breeze_cells = layout.get_breeze_cells()
                stench_cells = layout.get_stench_cells()
                
                a_breeze = 1.0 if (a_x, a_y) in breeze_cells else 0.0
                a_stench = 1.0 if (a_x, a_y) in stench_cells else 0.0
                b_breeze = 1.0 if (b_x, b_y) in breeze_cells else 0.0
                b_stench = 1.0 if (b_x, b_y) in stench_cells else 0.0
                
                state.extend([a_breeze, a_stench, b_breeze, b_stench])
    
    return np.array(state, dtype=np.float32)


def get_state_dim(env_size: int, layout=None, include_features: bool = True) -> int:
    """Calculate state dimension for network initialization."""
    dim = 4  # Base positions
    
    if include_features:
        dim += 1  # Distance between agents
        
        if layout is not None:
            dim += 4  # Distances to treasure and wumpus
            
            if hasattr(layout, 'get_breeze_cells'):
                dim += 4  # Breeze/stench indicators
    
    return dim


if __name__ == "__main__":
    # Test DQN components
    print("Testing DQN components...")
    print()
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available!")
        exit(1)
    
    # Test Q-Network
    print("=== Q-Network Test ===")
    state_dim = 13
    action_dim = 5
    
    q_net = QNetwork(state_dim, action_dim)
    print(f"Q-Network: {sum(p.numel() for p in q_net.parameters())} parameters")
    
    test_state = torch.randn(1, state_dim)
    q_values = q_net(test_state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values.detach().numpy()}")
    
    # Test Dueling Network
    print()
    print("=== Dueling Q-Network Test ===")
    dueling_net = DuelingQNetwork(state_dim, action_dim)
    print(f"Dueling Network: {sum(p.numel() for p in dueling_net.parameters())} parameters")
    
    q_values_dueling = dueling_net(test_state)
    print(f"Dueling Q-values: {q_values_dueling.detach().numpy()}")
    
    # Test DQN Agent
    print()
    print("=== DQN Agent Test ===")
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        dqn_type=DQNType.DUELING_DOUBLE,
        eps_decay_steps=1000,
    )
    
    agent = DQNAgent(config, seed=42)
    
    # Simulate some transitions
    for i in range(100):
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = np.random.random() < 0.1
        
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    print(f"Epsilon: {agent.get_epsilon():.4f}")
    
    # Train a few steps
    for i in range(10):
        loss = agent.train_step()
    
    stats = agent.get_stats()
    print(f"Stats: {stats}")
    
    # Test Multi-Agent DQN
    print()
    print("=== Multi-Agent DQN Test ===")
    ma_dqn = MultiAgentDQN(state_dim, action_dim, mode="self_play")
    
    state = np.random.randn(state_dim).astype(np.float32)
    action_a = ma_dqn.select_action_a(state)
    action_b = ma_dqn.select_action_b(state)
    print(f"Actions: A={action_a}, B={action_b}")
    
    print()
    print("All tests passed!")
