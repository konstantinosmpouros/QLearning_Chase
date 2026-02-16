from agents.fp_agent import FPAgent
from agents.minimax_q_agent import MinimaxQAgent
from agents.nash_q_agent import NashQAgent, NashType, compute_nash_equilibrium_nashpy
from agents.dyna_q_agent import (
    DynaQAgent,
    DynaQPlusAgent,
    TransitionModel,
    ProbabilisticTransitionModel,
)

# DQN imports (optional, requires PyTorch)
try:
    from agents.dqn_agent import (
        DQNAgent, MultiAgentDQN, DQNConfig, DQNType,
        QNetwork, DuelingQNetwork, ReplayBuffer,
        create_state_representation, get_state_dim,
    )
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

__all__ = [
    "FPAgent",
    "MinimaxQAgent",
    "NashQAgent",
    "NashType",
    "compute_nash_equilibrium_nashpy",
    # Dyna-Q
    "DynaQAgent",
    "DynaQPlusAgent",
    "TransitionModel",
    "ProbabilisticTransitionModel",
    "DQN_AVAILABLE",
]

if DQN_AVAILABLE:
    __all__.extend([
        "DQNAgent",
        "MultiAgentDQN",
        "DQNConfig",
        "DQNType",
        "QNetwork",
        "DuelingQNetwork",
        "ReplayBuffer",
        "create_state_representation",
        "get_state_dim",
    ])
