from train.common import EvalStats, evaluate, run_episode
from train.fp_training import (
    train_fp_selfplay,
    train_fp_vs_dynaq,
    train_fp_vs_fp,
    train_fp_vs_minimaxq,
    train_fp_vs_nashq,
)
from train.minimaxq_training import (
    train_minimaxq_selfplay,
    train_minimaxq_vs_dynaq,
    train_minimaxq_vs_fp,
    train_minimaxq_vs_minimaxq,
    train_minimaxq_vs_nashq,
)
from train.nashq_training import (
    train_nashq_selfplay,
    train_nashq_vs_dynaq,
    train_nashq_vs_nashq,
    train_nashq_vs_fp,
    train_nashq_vs_minimaxq,
)
from train.dynaq_training import (
    train_dynaq_selfplay,
    train_dynaq_vs_dynaq,
    train_dynaq_vs_fp,
    train_dynaq_vs_minimaxq,
    train_dynaq_vs_nashq,
    train_dynaq_plus_selfplay,
    train_dynaq_plus_vs_fp,
    train_dynaq_plus_vs_minimaxq,
    train_dynaq_plus_vs_nashq,
    train_dynaq_plus_vs_dynaq,
    compare_planning_steps,
)

# DQN training (optional, requires PyTorch)
try:
    from train.dqn_training import (
        train_dqn_selfplay,
        train_dqn_vs_dqn,
        train_dqn_vs_opponent,
        evaluate_dqn,
        create_dqn_policy,
    )
    DQN_TRAINING_AVAILABLE = True
except ImportError:
    DQN_TRAINING_AVAILABLE = False


def last_or_nan(vals):
    return vals[-1] if vals else float("nan")


__all__ = [
    "EvalStats",
    "evaluate",
    "run_episode",
    # FP
    "train_fp_selfplay",
    "train_fp_vs_fp",
    "train_fp_vs_minimaxq",
    "train_fp_vs_nashq",
    "train_fp_vs_dynaq",
    # Minimax-Q
    "train_minimaxq_selfplay",
    "train_minimaxq_vs_nashq",
    "train_minimaxq_vs_dynaq",
    "train_minimaxq_vs_fp",
    "train_minimaxq_vs_minimaxq",
    # Nash-Q
    "train_nashq_selfplay",
    "train_nashq_vs_nashq",
    "train_nashq_vs_fp",
    "train_nashq_vs_minimaxq",
    "train_nashq_vs_dynaq",
    # Dyna-Q
    "train_dynaq_selfplay",
    "train_dynaq_vs_dynaq",
    "train_dynaq_vs_fp",
    "train_dynaq_vs_minimaxq",
    "train_dynaq_vs_nashq",
    "train_dynaq_plus_selfplay",
    "train_dynaq_plus_vs_fp",
    "train_dynaq_plus_vs_minimaxq",
    "train_dynaq_plus_vs_nashq",
    "train_dynaq_plus_vs_dynaq",
    "compare_planning_steps",
    "DQN_TRAINING_AVAILABLE",
    # Utility
    "last_or_nan",
]

if DQN_TRAINING_AVAILABLE:
    __all__.extend([
        "train_dqn_selfplay",
        "train_dqn_vs_dqn",
        "train_dqn_vs_opponent",
        "evaluate_dqn",
        "create_dqn_policy",
    ])
