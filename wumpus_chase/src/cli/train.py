#!/usr/bin/env python3
"""Unified CLI for training + periodic evaluation in Wumpus Chase Extended."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import dangerous_layout, default_extended_layout, small_layout, visualize_layout
from env.wumpus_env_extended import WumpusChaseEnvExtended
from logger import CSVLogger
from train import (
    DQN_TRAINING_AVAILABLE,
    last_or_nan,
    train_dynaq_selfplay,
    train_dynaq_plus_selfplay,
    train_dynaq_plus_vs_dynaq,
    train_dynaq_vs_fp,
    train_dynaq_vs_minimaxq,
    train_dynaq_vs_nashq,
    train_dynaq_plus_vs_fp,
    train_dynaq_plus_vs_minimaxq,
    train_dynaq_plus_vs_nashq,
    train_fp_selfplay,
    train_fp_vs_fp,
    train_fp_vs_dynaq,
    train_fp_vs_minimaxq,
    train_fp_vs_nashq,
    train_minimaxq_selfplay,
    train_minimaxq_vs_dynaq,
    train_minimaxq_vs_fp,
    train_minimaxq_vs_nashq,
    train_nashq_selfplay,
    train_nashq_vs_dynaq,
    train_nashq_vs_fp,
    train_nashq_vs_minimaxq,
)

if DQN_TRAINING_AVAILABLE:
    from agents.dqn_agent import DQNType
    from train import train_dqn_vs_dqn


FULL_MATCHUPS = [
    "fp_selfplay",
    "fp_vs_minimaxq",
    "fp_vs_nashq",
    "fp_vs_dynaq",
    "minimaxq_selfplay",
    "minimaxq_vs_fp",
    "minimaxq_vs_nashq",
    "minimaxq_vs_dynaq",
    "nashq_selfplay",
    "nashq_vs_fp",
    "nashq_vs_minimaxq",
    "nashq_vs_dynaq",
    "dynaq_selfplay",
    "dynaq_vs_fp",
    "dynaq_vs_minimaxq",
    "dynaq_vs_nashq",
    "dynaq_plus_selfplay",
    "dynaq_plus_vs_nashq",
    "dynaq_plus_vs_minimaxq",
    "dynaq_plus_vs_fp",
    "dynaq_plus_vs_dynaq",
]

QUICK_MATCHUPS = [
    "fp_selfplay",
    "minimaxq_selfplay",
    "nashq_selfplay",
    "dynaq_selfplay",
    "nashq_vs_minimaxq",
]


def _dqn_type_from_arg(name: str):
    mapping = {
        "standard": DQNType.STANDARD,
        "double": DQNType.DOUBLE,
        "dueling": DQNType.DUELING,
        "dueling_double": DQNType.DUELING_DOUBLE,
    }
    return mapping[name]


def _create_layout(name: str):
    if name == "small":
        return small_layout()
    if name == "dangerous":
        return dangerous_layout()
    return default_extended_layout()


def _run_matchup(
    matchup: str,
    env: WumpusChaseEnvExtended,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    planning_steps: int,
    seed: int,
    train_logger: CSVLogger,
    eval_logger: CSVLogger,
    quiet: bool,
    dqn_type: str,
) -> Dict[str, List[float]]:
    if matchup in {"fp_selfplay", "fp_vs_fp"}:
        trainer = train_fp_selfplay if matchup == "fp_selfplay" else train_fp_vs_fp
        return trainer(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "fp_vs_minimaxq":
        return train_fp_vs_minimaxq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "fp_vs_nashq":
        return train_fp_vs_nashq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "fp_vs_dynaq":
        return train_fp_vs_dynaq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "minimaxq_selfplay":
        return train_minimaxq_selfplay(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "minimaxq_vs_fp":
        return train_minimaxq_vs_fp(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "minimaxq_vs_nashq":
        return train_minimaxq_vs_nashq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "minimaxq_vs_dynaq":
        return train_minimaxq_vs_dynaq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup in {"nashq_selfplay", "nashq_vs_nashq"}:
        return train_nashq_selfplay(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "nashq_vs_fp":
        return train_nashq_vs_fp(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "nashq_vs_minimaxq":
        return train_nashq_vs_minimaxq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "nashq_vs_dynaq":
        return train_nashq_vs_dynaq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup in {"dynaq_selfplay", "dynaq_vs_dynaq"}:
        return train_dynaq_selfplay(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dynaq_vs_fp":
        return train_dynaq_vs_fp(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dynaq_vs_minimaxq":
        return train_dynaq_vs_minimaxq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dynaq_vs_nashq":
        return train_dynaq_vs_nashq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dynaq_plus_selfplay":
        return train_dynaq_plus_selfplay(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dynaq_plus_vs_nashq":
        return train_dynaq_plus_vs_nashq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dynaq_plus_vs_minimaxq":
        return train_dynaq_plus_vs_minimaxq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dynaq_plus_vs_fp":
        return train_dynaq_plus_vs_fp(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dynaq_plus_vs_dynaq":
        return train_dynaq_plus_vs_dynaq(
            env=env,
            episodes=episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            n_planning=planning_steps,
            seed=seed,
            logger=train_logger,
            eval_logger=eval_logger,
            run_label=matchup,
            verbose=not quiet,
        )
    if matchup == "dqn_vs_dqn":
        if not DQN_TRAINING_AVAILABLE:
            raise RuntimeError("DQN requested but PyTorch/DQN training is not available.")
        return train_dqn_vs_dqn(
            env=env,
            episodes=min(episodes, 5000),
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            seed=seed,
            dqn_type=_dqn_type_from_arg(dqn_type),
            verbose=not quiet,
        )
    raise ValueError(f"Unknown matchup: {matchup}")


def _sheet_name(name: str, used: set[str]) -> str:
    base = name[:31]
    candidate = base
    suffix = 1
    while candidate in used:
        trail = f"_{suffix}"
        candidate = f"{base[: 31 - len(trail)]}{trail}"
        suffix += 1
    used.add(candidate)
    return candidate


def _save_excel(all_results: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    used_sheets: set[str] = set()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for matchup, logs in all_results.items():
            df = pd.DataFrame(logs)
            df.insert(0, "matchup", matchup)
            df.to_excel(writer, sheet_name=_sheet_name(matchup, used_sheets), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training + evaluation CLI (extended env only)")
    parser.add_argument(
        "--matchup",
        default="all",
        choices=[
            "all",
            "fp_selfplay",
            "fp_vs_fp",
            "fp_vs_minimaxq",
            "fp_vs_nashq",
            "fp_vs_dynaq",
            "minimaxq_selfplay",
            "minimaxq_vs_fp",
            "minimaxq_vs_nashq",
            "minimaxq_vs_dynaq",
            "nashq_selfplay",
            "nashq_vs_fp",
            "nashq_vs_minimaxq",
            "nashq_vs_dynaq",
            "dynaq_selfplay",
            "dynaq_vs_fp",
            "dynaq_vs_minimaxq",
            "dynaq_vs_nashq",
            "dynaq_plus_selfplay",
            "dynaq_plus_vs_nashq",
            "dynaq_plus_vs_minimaxq",
            "dynaq_plus_vs_fp",
            "dynaq_plus_vs_dynaq",
            "dqn_vs_dqn",
        ],
        help="Which training matchup to run",
    )
    parser.add_argument("--episodes", type=int, default=3000, help="Training episodes")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Evaluate every N training episodes (default: 100)",
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes per evaluation point")
    parser.add_argument(
        "--suite",
        choices=["quick", "full"],
        default="quick",
        help="Preset used when --matchup=all (quick runs fewer trainings, full runs all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--layout", choices=["small", "default", "dangerous"], default="small")
    parser.add_argument("--p-fail", type=float, default=0.10, help="Action failure probability")
    parser.add_argument("--t-max", type=int, default=None, help="Environment max steps per episode")
    parser.add_argument("--planning-steps", type=int, default=10, help="Planning steps for Dyna-Q variants")
    parser.add_argument(
        "--dqn-type",
        choices=["standard", "double", "dueling", "dueling_double"],
        default="dueling_double",
    )
    parser.add_argument("--results-dir", type=str, default=str(ROOT.parent / "results"))
    parser.add_argument(
        "--excel-output",
        type=str,
        default=None,
        help="Optional Excel output path (one sheet per matchup)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    layout = _create_layout(args.layout)
    t_max = args.t_max if args.t_max is not None else (40 if args.layout == "small" else 50)
    env = WumpusChaseEnvExtended(
        layout=layout,
        p_fail=args.p_fail,
        t_max=t_max,
        seed=args.seed,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    train_logger = CSVLogger(path=results_dir / "wumpus_extended_train.csv")
    eval_logger = CSVLogger(path=results_dir / "wumpus_extended_eval.csv")

    if not args.quiet:
        print("\n============================================================")
        print(" WUMPUS CHASE EXTENDED - TRAINING")
        print("============================================================")
        print(f"Layout: {args.layout} ({layout.size}x{layout.size})")
        print(visualize_layout(layout))

    matchups = (
        (QUICK_MATCHUPS if args.suite == "quick" else FULL_MATCHUPS)
        if args.matchup == "all"
        else [args.matchup]
    )
    if args.matchup == "all" and args.suite == "full" and DQN_TRAINING_AVAILABLE:
        matchups.append("dqn_vs_dqn")

    all_results: Dict[str, Dict[str, List[float]]] = {}
    for idx, matchup in enumerate(matchups):
        run_seed = args.seed + idx * 1000
        if not args.quiet:
            print(f"\n--- Running: {matchup} (seed={run_seed}) ---")
        logs = _run_matchup(
            matchup=matchup,
            env=env,
            episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            planning_steps=args.planning_steps,
            seed=run_seed,
            train_logger=train_logger,
            eval_logger=eval_logger,
            quiet=args.quiet,
            dqn_type=args.dqn_type,
        )
        all_results[matchup] = logs

    train_logger.save()
    eval_logger.save()
    excel_output = Path(args.excel_output) if args.excel_output else (results_dir / "wumpus_extended_training.xlsx")
    _save_excel(all_results, excel_output)

    print("\n============================================================")
    print(" Final Summary")
    print("============================================================")
    print("{:<24} {:>10} {:>10} {:>10} {:>10}".format(
        "Matchup", "Win Rate", "RunnerWin", "Avg Steps", "Avg Return"
    ))
    print("-" * 70)
    for name, logs in all_results.items():
        win_rate = last_or_nan(logs.get("win_rate", []))
        draw_rate = last_or_nan(logs.get("draw_rate", []))
        if math.isnan(win_rate) or math.isnan(draw_rate):
            runner_win_rate = float("nan")
        else:
            runner_win_rate = max(0.0, min(1.0, 1.0 - win_rate - draw_rate))
        print("{:<24} {:>10.1%} {:>10.1%} {:>10.1f} {:>10.2f}".format(
            name,
            win_rate,
            runner_win_rate,
            last_or_nan(logs.get("avg_steps", [])),
            last_or_nan(logs.get("avg_return", [])),
        ))

    print(f"\nTrain log: {results_dir / 'wumpus_extended_train.csv'}")
    print(f"Eval log:  {results_dir / 'wumpus_extended_eval.csv'}")
    print(f"Excel:     {excel_output}")


if __name__ == "__main__":
    main()
