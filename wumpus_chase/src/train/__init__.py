from train.common import EvalStats, evaluate, run_episode
from train.fp_vs_fp import train_fp_vs_fp
from train.minimaxq_selfplay import train_minimaxq_selfplay
from train.minimaxq_vs_fp import train_minimaxq_vs_fp
from train.minimaxq_vs_minimaxq import train_minimaxq_vs_minimaxq


def last_or_nan(vals):
    return vals[-1] if vals else float("nan")


__all__ = [
    "EvalStats",
    "evaluate",
    "run_episode",
    "train_fp_vs_fp",
    "train_minimaxq_selfplay",
    "train_minimaxq_vs_fp",
    "train_minimaxq_vs_minimaxq",
    "last_or_nan",
]
