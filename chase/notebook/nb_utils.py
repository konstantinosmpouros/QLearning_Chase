"""
Helper utilities for the Chase analytics notebook.
Centralizes repeatable tasks like loading the Excel logs,
adding derived columns, computing episode metrics, and
quick plotting helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

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
    group_cols = ["run", "episode"]
    if "p_fail" in df.columns:
        group_cols.insert(1, "p_fail")
    grp = df.groupby(group_cols, as_index=False)
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
    grp = (
        ep_df.groupby(["run", "p_fail"], as_index=False)
        if "p_fail" in ep_df.columns
        else ep_df.groupby("run", as_index=False)
    )
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


def rolling_episode_metrics(ep_df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    Add rolling means of capture rate, steps, and return per (run, p_fail).
    """
    if ep_df.empty:
        return ep_df
    if "p_fail" not in ep_df.columns:
        ep_df = ep_df.copy()
        ep_df["p_fail"] = None

    def _add_roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("episode")
        g["roll_capture_rate"] = g["captured"].rolling(window, min_periods=1).mean()
        g["roll_steps"] = g["steps"].rolling(window, min_periods=1).mean()
        g["roll_return"] = g["total_return"].rolling(window, min_periods=1).mean()
        return g

    return ep_df.groupby(["run", "p_fail"], group_keys=False).apply(_add_roll)


def rolling_action_mix(df: pd.DataFrame, who: str = "catcher", window: int = 500) -> pd.DataFrame:
    """
    Compute rolling action distribution (fraction) per run/p_fail over episode windows.
    Returns a long-format DataFrame with columns: run, p_fail, episode, action, frac.
    """
    if df.empty:
        return pd.DataFrame()
    col = f"{who}_action"
    if col not in df.columns:
        return pd.DataFrame()
    base = df.copy()
    if "p_fail" not in base.columns:
        base["p_fail"] = None

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        # Count actions per episode.
        counts = (
            g.groupby(["episode", col])
            .size()
            .reset_index(name="count")
            .pivot(index="episode", columns=col, values="count")
            .fillna(0)
        )
        # Rolling sum then normalize per window.
        roll = counts.rolling(window, min_periods=1).sum()
        frac = roll.div(roll.sum(axis=1), axis=0)
        frac = frac.reset_index().melt(id_vars="episode", var_name="action", value_name="frac")
        return frac

    out = (
        base.groupby(["run", "p_fail"], group_keys=True)
        .apply(_per_group)
        .reset_index(level=[0, 1])
        .rename(columns={"run": "run", "p_fail": "p_fail"})
    )
    return out


def rolling_return_quantiles(
    ep_df: pd.DataFrame,
    window: int = 400,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """
    Compute rolling quantiles of total_return per run/p_fail.
    Returns long DataFrame with columns: run, p_fail, episode, quantile, value.
    """
    if ep_df.empty:
        return pd.DataFrame()
    base = ep_df.copy()
    if "p_fail" not in base.columns:
        base["p_fail"] = None

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("episode").set_index("episode")
        rows = []
        for q in quantiles:
            series = g["total_return"].rolling(window, min_periods=1).quantile(q)
            rows.append(series.rename(q))
        res = pd.concat(rows, axis=1).reset_index()
        return res.melt(id_vars="episode", var_name="quantile", value_name="value")

    out = (
        base.groupby(["run", "p_fail"], group_keys=True)
        .apply(_per_group)
        .reset_index(level=[0, 1])
        .rename(columns={"run": "run", "p_fail": "p_fail"})
    )
    return out


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


def plot_run_trends(ep_df: pd.DataFrame, run: str, window: int = 200, title_prefix: Optional[str] = None) -> None:
    """
    Plot rolling capture rate and steps for a given run, split by p_fail.
    """
    run_df = ep_df.loc[ep_df["run"] == run]
    if run_df.empty:
        print(f"No episode data for run={run}")
        return
    roll_df = rolling_episode_metrics(run_df, window=window)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    p_fails = (
        sorted(roll_df["p_fail"].dropna().unique().tolist())
        if "p_fail" in roll_df
        else [None]
    )
    for pf in p_fails:
        sub = roll_df if pf is None else roll_df.loc[roll_df["p_fail"] == pf]
        label = f"p_fail={pf}" if pf is not None else "p_fail=?"
        axes[0].plot(sub["episode"], sub["roll_capture_rate"], label=label)
        axes[1].plot(sub["episode"], sub["roll_steps"], label=label)
    ttl = title_prefix or run
    axes[0].set_title(f"{ttl} – Rolling capture rate")
    axes[0].set_ylabel("Capture rate")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].set_title(f"{ttl} – Rolling steps")
    axes[1].set_ylabel("Steps")
    axes[1].set_xlabel("Episode")
    axes[1].legend()
    fig.tight_layout()


def plot_action_drift(action_mix: pd.DataFrame, title: str) -> None:
    """
    Plot rolling action fractions as stacked area chart.
    Expects columns: run, p_fail, episode, action, frac.
    """
    if action_mix.empty:
        print("No action-mix data to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = (
        action_mix.pivot_table(index="episode", columns="action", values="frac", aggfunc="mean")
        .fillna(0)
        .sort_index()
    )
    episodes = pivot.index.values
    bottoms = None
    for action in pivot.columns:
        vals = pivot[action].values
        ax.fill_between(episodes, vals + (bottoms if bottoms is not None else 0), bottoms if bottoms is not None else 0, label=action, step="mid", alpha=0.8)
        bottoms = vals + (bottoms if bottoms is not None else 0)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action fraction (rolling)")
    ax.set_ylim(0, 1)
    ax.legend(title="Action", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()


def visit_density(df: pd.DataFrame, who: str = "catcher", capture_only: bool = False) -> pd.DataFrame:
    """
    Compute visit density grid for catcher or runner.
    Set capture_only=True to filter to capture timesteps.
    Returns a pivot table indexed by x with columns y.
    """
    if df.empty:
        return pd.DataFrame()
    base = df.copy()
    if capture_only:
        base = base.loc[base["capture"]]
    x_col = f"{who}_x"
    y_col = f"{who}_y"
    if x_col not in base or y_col not in base:
        return pd.DataFrame()
    return base.pivot_table(index=x_col, columns=y_col, values="reward", aggfunc="count").fillna(0)


def plot_visit_heatmaps(df: pd.DataFrame, run: str, p_fail: float) -> None:
    """
    Heatmaps of visit density and capture locations for catcher and runner.
    """
    sub = df.loc[(df["run"] == run) & (df["p_fail"] == p_fail)]
    if sub.empty:
        print(f"No data for run={run}, p_fail={p_fail}")
        return
    grids = {
        "Catcher visits": visit_density(sub, who="catcher", capture_only=False),
        "Runner visits": visit_density(sub, who="runner", capture_only=False),
        "Catcher captures": visit_density(sub, who="catcher", capture_only=True),
        "Runner captures": visit_density(sub, who="runner", capture_only=True),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    for ax, (title, grid) in zip(axes, grids.items()):
        im = ax.imshow(grid.values, origin="upper", cmap="Oranges")
        ax.set_title(title)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Run={run}, p_fail={p_fail} visit densities", y=0.92)
    fig.tight_layout()


__all__ = [
    "RESULTS_PATH",
    "load_runs",
    "combine_runs",
    "add_derived_columns",
    "episode_metrics",
    "capture_rate_by_run",
    "action_counts",
    "rolling_episode_metrics",
    "rolling_action_mix",
    "rolling_return_quantiles",
    "plot_capture_rate",
    "plot_action_distribution",
    "plot_action_drift",
    "episode_trajectories",
    "plot_positions",
    "plot_run_trends",
    "visit_density",
    "plot_visit_heatmaps",
]
