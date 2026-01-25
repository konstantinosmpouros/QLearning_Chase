"""Entry point for Wumpus Chase experiments."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import WumpusChaseEnv, default_layout
from logger import CSVLogger
from train import last_or_nan, train_fp_vs_fp, train_minimaxq_selfplay, train_minimaxq_vs_fp, train_minimaxq_vs_minimaxq


def main() -> None:
    layout = default_layout()
    env = WumpusChaseEnv(layout=layout, p_fail=0.10, seed=0)

    episodes = 30000
    eval_every = 100

    results_dir = Path(__file__).resolve().parent.parent / "results"
    train_logger = CSVLogger(path=results_dir / "wumpus_train.csv")
    eval_logger = CSVLogger(path=results_dir / "wumpus_eval.csv")

    fp_logs = train_fp_vs_fp(
        env,
        episodes=episodes,
        eval_every=eval_every,
        seed=0,
        logger=train_logger,
        eval_logger=eval_logger,
        run_label="fp_vs_fp",
    )
    mm_logs = train_minimaxq_selfplay(
        env,
        episodes=episodes,
        eval_every=eval_every,
        seed=1,
        logger=train_logger,
        eval_logger=eval_logger,
        run_label="minimaxq_selfplay",
    )
    mm2_logs = train_minimaxq_vs_minimaxq(
        env,
        episodes=episodes,
        eval_every=eval_every,
        seed=2,
        logger=train_logger,
        eval_logger=eval_logger,
        run_label="minimaxq_vs_minimaxq",
    )
    mix_logs = train_minimaxq_vs_fp(
        env,
        episodes=episodes,
        eval_every=eval_every,
        seed=3,
        logger=train_logger,
        eval_logger=eval_logger,
        run_label="minimaxq_vs_fp",
    )

    print("FP vs FP final:")
    print(f"  Win rate:   {last_or_nan(fp_logs['win_rate']):.3f}")
    print(f"  Draw rate:  {last_or_nan(fp_logs['draw_rate']):.3f}")
    print(f"  Avg steps:  {last_or_nan(fp_logs['avg_steps']):.3f}")
    print(f"  Avg return: {last_or_nan(fp_logs['avg_return']):.3f}")

    print("MinimaxQ vs MinimaxQ (selfplay) final:")
    print(f"  Win rate:   {last_or_nan(mm_logs['win_rate']):.3f}")
    print(f"  Draw rate:  {last_or_nan(mm_logs['draw_rate']):.3f}")
    print(f"  Avg steps:  {last_or_nan(mm_logs['avg_steps']):.3f}")
    print(f"  Avg return: {last_or_nan(mm_logs['avg_return']):.3f}")

    print("MinimaxQ vs MinimaxQ (two agents) final:")
    print(f"  Win rate:   {last_or_nan(mm2_logs['win_rate']):.3f}")
    print(f"  Draw rate:  {last_or_nan(mm2_logs['draw_rate']):.3f}")
    print(f"  Avg steps:  {last_or_nan(mm2_logs['avg_steps']):.3f}")
    print(f"  Avg return: {last_or_nan(mm2_logs['avg_return']):.3f}")

    print("MinimaxQ vs FP final:")
    print(f"  Win rate:   {last_or_nan(mix_logs['win_rate']):.3f}")
    print(f"  Draw rate:  {last_or_nan(mix_logs['draw_rate']):.3f}")
    print(f"  Avg steps:  {last_or_nan(mix_logs['avg_steps']):.3f}")
    print(f"  Avg return: {last_or_nan(mix_logs['avg_return']):.3f}")

    train_logger.save()
    eval_logger.save()


if __name__ == "__main__":
    main()
