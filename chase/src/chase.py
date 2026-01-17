"""
Entry point for running chase training experiments.
"""

import sys
from pathlib import Path

# Ensure imports work when running as a script from any cwd by putting project
# root (this folder) at the front of sys.path.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logger import ExcelLogger
from train import (
    last_or_nan,
    train_fp_vs_fp,
    train_minimaxq_selfplay,
    train_minimaxq_vs_fp,
)


def main() -> None:
    """
    Run small training experiments for a single p_fail value.
    To keep runtime short, we train each matchup for only a few hundred episodes.
    Prints the final evaluation stats for each matchup.
    """
    train_logger = ExcelLogger(path="results/chase_train.xlsx")
    eval_logger = ExcelLogger(path="results/chase_eval.xlsx")
    p_fail_values = [0.10, 0.20]
    episodes_fp = 10_000
    episodes_mm = 10_000
    episodes_mix = 10_000
    eval_every_fp = 1_000
    eval_every_mm = 1_000
    eval_every_mix = 1_000
    
    for p_fail in p_fail_values:
        print(f"=== Training with p_fail={p_fail:.2f} ===")
        fp_logs = train_fp_vs_fp(
            p_fail=p_fail,
            episodes=episodes_fp,
            eval_every=eval_every_fp,
            seed=0,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label="fp_vs_fp",
        )
        mm_logs = train_minimaxq_selfplay(
            p_fail=p_fail,
            episodes=episodes_mm,
            eval_every=eval_every_mm,
            seed=1,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label="minimaxq_selfplay",
        )
        mix_logs = train_minimaxq_vs_fp(
            p_fail=p_fail,
            episodes=episodes_mix,
            eval_every=eval_every_mix,
            seed=2,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label="minimaxq_vs_fp",
        )
        print("FP vs FP final:")
        print(f"  Capture rate:   {last_or_nan(fp_logs['capture_rate']):.3f}")
        print(f"  Avg steps:      {last_or_nan(fp_logs['avg_steps']):.3f}")
        print(f"  Avg return:     {last_or_nan(fp_logs['avg_return']):.3f}")
        print("MinimaxQ vs MinimaxQ final:")
        print(f"  Capture rate:   {last_or_nan(mm_logs['capture_rate']):.3f}")
        print(f"  Avg steps:      {last_or_nan(mm_logs['avg_steps']):.3f}")
        print(f"  Avg return:     {last_or_nan(mm_logs['avg_return']):.3f}")
        print("MinimaxQ vs FP final:")
        print(f"  Capture rate:   {last_or_nan(mix_logs['capture_rate']):.3f}")
        print(f"  Avg steps:      {last_or_nan(mix_logs['avg_steps']):.3f}")
        print(f"  Avg return:     {last_or_nan(mix_logs['avg_return']):.3f}")
        print()

    train_logger.save()
    eval_logger.save()


if __name__ == "__main__":
    main()
