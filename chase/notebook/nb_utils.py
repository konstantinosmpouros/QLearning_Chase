"""
Helper utilities for the Chase analytics notebook.
Centralizes repeatable tasks like loading the Excel logs,
adding derived columns, computing episode metrics, and
quick plotting helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from matplotlib import pyplot as plt


RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "chase_run.xlsx"


def load_runs(path: Path | str = RESULTS_PATH) -> Dict[str, pd.DataFrame]:
    """
    Read the Excel log file and return a dict of DataFrames keyed by sheet/run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    # sheet_name=None returns a dict of {sheet_name: df}
    return pd.read_excel(path, sheet_name=None)


def combine_runs(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all sheet DataFrames into a single DataFrame with a 'run' column.
    """
    dfs = []
    for run_name, df in sheets.items():
        df = df.copy()
        df["run"] = run_name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add helpful derived columns to a step-level DataFrame.
    """
    if df.empty:
        return df
    out = df.copy()
    # Manhattan distance between catcher and runner.
    out["manhattan_dist"] = (
        (out["catcher_x"] - out["runner_x"]).abs()
        + (out["catcher_y"] - out["runner_y"]).abs()
    )
    # Episode-step counter (already present as 'step', but ensure int)
    out["step"] = out["step"].astype(int)
    out["episode"] = out["episode"].astype(int)
    out["capture"] = out["capture"].astype(bool)
    out["done"] = out["done"].astype(bool)
    # Convenience: categorical actions.
    out["catcher_action"] = out["catcher_action"].astype("category")
    out["runner_action"] = out["runner_action"].astype("category")
    return out


def episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute episode-level aggregates per run:
    - captured flag
    - steps to capture (or t_max if not captured)
    - total return
    """
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby(["run", "episode"], as_index=False)
    agg = grp.agg(
        captured=("capture", "max"),
        steps=("step", "max"),
        total_return=("reward", "sum"),
    )
    return agg


def capture_rate_by_run(ep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize capture rate and average steps per run.
    """
    if ep_df.empty:
        return pd.DataFrame()
    grp = ep_df.groupby("run", as_index=False)
    return grp.agg(
        capture_rate=("captured", "mean"),
        avg_steps=("steps", "mean"),
        avg_return=("total_return", "mean"),
    )


def action_counts(df: pd.DataFrame, who: str = "catcher") -> pd.DataFrame:
    """
    Count actions for catcher or runner across all runs/episodes.
    who: 'catcher' | 'runner'
    """
    col = f"{who}_action"
    if col not in df.columns or df.empty:
        return pd.DataFrame()
    counts = (
        df.groupby(["run", col])
        .size()
        .reset_index(name="count")
        .sort_values(["run", "count"], ascending=[True, False])
    )
    total = counts.groupby("run")["count"].transform("sum")
    counts["pct"] = counts["count"] / total
    return counts


def plot_capture_rate(summary_df: pd.DataFrame) -> None:
    """
    Bar plot of capture rate per run.
    """
    if summary_df.empty:
        print("No data to plot.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary_df["run"], summary_df["capture_rate"], color="#4e79a7")
    ax.set_ylim(0, 1)
    ax.set_title("Capture Rate by Run")
    ax.set_ylabel("Capture rate")
    ax.set_xlabel("Run")
    fig.tight_layout()


def plot_action_distribution(action_df: pd.DataFrame, title: str) -> None:
    """
    Stacked bar plot of action distribution per run.
    Expects columns: run, <action_col>, pct
    """
    if action_df.empty:
        print("No data to plot.")
        return
    action_col = action_df.columns[1]
    pivot = action_df.pivot(index="run", columns=action_col, values="pct").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = None
    for action in pivot.columns:
        vals = pivot[action]
        ax.bar(pivot.index, vals, bottom=bottom, label=action)
        bottom = vals if bottom is None else bottom + vals
    ax.set_title(title)
    ax.set_ylabel("Fraction of steps")
    ax.set_xlabel("Run")
    ax.legend(title="Action", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()


def episode_trajectories(df: pd.DataFrame, run: str, episodes: Iterable[int]) -> pd.DataFrame:
    """
    Slice the step-level data for selected episodes from a run.
    Returns a filtered DataFrame useful for plotting trajectories.
    """
    if df.empty:
        return pd.DataFrame()
    epi_set = set(episodes)
    mask = (df["run"] == run) & (df["episode"].isin(epi_set))
    return df.loc[mask].copy()


def plot_positions(df: pd.DataFrame, size: int = 5) -> None:
    """
    Plot catcher/runner positions over steps for a filtered DataFrame.
    Assumes columns: step, catcher_x, catcher_y, runner_x, runner_y.
    """
    if df.empty:
        print("No data to plot.")
        return
    plt.figure(figsize=(6, 6))
    for ep, g in df.groupby("episode"):
        plt.plot(g["catcher_y"], g["catcher_x"], "-o", label=f"Catcher ep{ep}")
        plt.plot(g["runner_y"], g["runner_x"], "-o", label=f"Runner ep{ep}", alpha=0.7)
    plt.xlim(-0.5, size - 0.5)
    plt.ylim(-0.5, size - 0.5)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Trajectories (y=col, x=row)")
    plt.tight_layout()


__all__ = [
    "RESULTS_PATH",
    "load_runs",
    "combine_runs",
    "add_derived_columns",
    "episode_metrics",
    "capture_rate_by_run",
    "action_counts",
    "plot_capture_rate",
    "plot_action_distribution",
    "episode_trajectories",
    "plot_positions",
]
