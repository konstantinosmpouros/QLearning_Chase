"""
Evaluation-specific analytics helpers for the Chase project.
Mirrors utils_train but keys episodes by (run, p_fail, env_seed, episode)
so eval rounds that reuse episode numbers stay distinct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "chase_eval.xlsx"


def load_runs(path: Path | str = RESULTS_PATH) -> Dict[str, pd.DataFrame]:
    """Read the Excel log file and return {sheet_name: df}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_excel(path, sheet_name=None)


def combine_runs(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all sheet DataFrames into a single DataFrame with a 'run' column."""
    dfs = []
    for run_name, df in sheets.items():
        df = df.copy()
        df["run"] = run_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _id_keys(include_eval_round: bool = False) -> list[str]:
    """Canonical keys that uniquely identify an eval episode."""
    keys = ["run", "p_fail", "env_seed"]
    if include_eval_round:
        keys.append("eval_at_episode")
    keys.append("episode")
    return keys


def _attach_eval_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add eval_episode_id: strictly increasing per (run, p_fail) ordered by env_seed then episode.
    This gives us a monotonic x-axis for eval history even though episode numbers repeat per seed.
    """
    if df.empty or "env_seed" not in df:
        return df
    keys = _id_keys(include_eval_round="eval_at_episode" in df.columns)
    uniq = df[keys].drop_duplicates()
    # Order by env_seed then episode as requested.
    uniq = uniq.sort_values(["run", "p_fail", "env_seed", "episode"])
    uniq["eval_episode_id"] = uniq.groupby(["run", "p_fail"]).cumcount() + 1
    return df.merge(uniq, on=keys, how="left")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add helpful derived columns to a step-level eval DataFrame.
    """
    if df.empty:
        return df
    out = df.copy()
    out["step"] = out["step"].astype(int)
    out["episode"] = out["episode"].astype(int)
    out["env_seed"] = out["env_seed"].astype(int)
    if "eval_at_episode" in out.columns:
        out["eval_at_episode"] = out["eval_at_episode"].astype(int)
    out["capture"] = out["capture"].astype(bool)
    out["done"] = out["done"].astype(bool)
    # Manhattan distance between catcher and runner.
    out["manhattan_dist"] = (
        (out["catcher_x"] - out["runner_x"]).abs()
        + (out["catcher_y"] - out["runner_y"]).abs()
    )
    # Convenience: categorical actions.
    out["catcher_action"] = out["catcher_action"].astype("category")
    out["runner_action"] = out["runner_action"].astype("category")
    out = _attach_eval_episode_id(out)
    return out


def episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute episode-level aggregates per unique eval episode:
    - captured flag
    - steps to capture (or t_max if not captured)
    - total return
    - eval_episode_id carried through for plotting
    """
    if df.empty:
        return pd.DataFrame()
    keys = _id_keys(include_eval_round="eval_at_episode" in df.columns)
    agg = (
        df.groupby(keys, as_index=False)
        .agg(
            captured=("capture", "max"),
            steps=("step", "max"),
            total_return=("reward", "sum"),
            eval_episode_id=("eval_episode_id", "first"),
        )
    )
    return agg


def capture_rate_by_run(
    ep_df: pd.DataFrame,
    include_seed: bool = True,
    include_eval_round: bool = False,
) -> pd.DataFrame:
    """
    Summarize capture rate and average steps.
    By default keeps env_seed in the grouping so each eval seed is separate.
    """
    if ep_df.empty:
        return pd.DataFrame()
    keys = ["run", "p_fail"]
    if include_seed and "env_seed" in ep_df:
        keys.append("env_seed")
    if include_eval_round and "eval_at_episode" in ep_df:
        keys.append("eval_at_episode")
    grp = ep_df.groupby(keys, as_index=False)
    return grp.agg(
        capture_rate=("captured", "mean"),
        avg_steps=("steps", "mean"),
        avg_return=("total_return", "mean"),
        eval_episode_id=("eval_episode_id", "mean"),
    )


def action_counts(df: pd.DataFrame, who: str = "catcher") -> pd.DataFrame:
    """
    Count actions for catcher or runner.
    Groups by run/p_fail/env_seed so per-seed behavior stays separate.
    who: 'catcher' | 'runner'
    """
    col = f"{who}_action"
    if col not in df.columns or df.empty:
        return pd.DataFrame()
    counts = (
        df.groupby(["run", "p_fail", "env_seed", col])
        .size()
        .reset_index(name="count")
        .sort_values(["run", "p_fail", "env_seed", "count"], ascending=[True, True, True, False])
    )
    total = counts.groupby(["run", "p_fail", "env_seed"])["count"].transform("sum")
    counts["pct"] = counts["count"] / total
    return counts


def rolling_episode_metrics(ep_df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    Add rolling means of capture rate, steps, and return per (run, p_fail),
    ordered by eval_episode_id (monotonic across eval seeds).
    """
    if ep_df.empty:
        return ep_df
    base = ep_df.copy()
    if "p_fail" not in base.columns:
        base["p_fail"] = None
    if "eval_episode_id" not in base.columns:
        base["eval_episode_id"] = base.groupby(["run", "p_fail"]).cumcount() + 1

    def _add_roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("eval_episode_id")
        g["roll_capture_rate"] = g["captured"].rolling(window, min_periods=3).mean()
        g["roll_steps"] = g["steps"].rolling(window, min_periods=3).mean()
        g["roll_return"] = g["total_return"].rolling(window, min_periods=3).mean()
        return g

    return base.groupby(["run", "p_fail"], group_keys=False).apply(_add_roll)


def rolling_action_mix(df: pd.DataFrame, who: str = "catcher", window: int = 200) -> pd.DataFrame:
    """
    Rolling action distribution over eval_episode_id per run/p_fail.
    Returns long-format: run, p_fail, eval_episode_id, action, frac.
    """
    if df.empty:
        return pd.DataFrame()
    col = f"{who}_action"
    if col not in df.columns:
        return pd.DataFrame()
    base = df.copy()
    if "p_fail" not in base.columns:
        base["p_fail"] = None
    if "eval_episode_id" not in base.columns:
        base = _attach_eval_episode_id(base)

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        counts = (
            g.groupby(["eval_episode_id", col])
            .size()
            .reset_index(name="count")
            .pivot(index="eval_episode_id", columns=col, values="count")
            .fillna(0)
        )
        roll = counts.rolling(window, min_periods=1).sum()
        frac = roll.div(roll.sum(axis=1), axis=0)
        frac = frac.reset_index().melt(id_vars="eval_episode_id", var_name="action", value_name="frac")
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
    window: int = 200,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """
    Rolling quantiles of total_return per run/p_fail ordered by eval_episode_id.
    Returns long DataFrame with columns: run, p_fail, eval_episode_id, quantile, value.
    """
    if ep_df.empty:
        return pd.DataFrame()
    base = ep_df.copy()
    if "p_fail" not in base.columns:
        base["p_fail"] = None
    if "eval_episode_id" not in base.columns:
        base["eval_episode_id"] = base.groupby(["run", "p_fail"]).cumcount() + 1

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("eval_episode_id").set_index("eval_episode_id")
        rows = []
        for q in quantiles:
            series = g["total_return"].rolling(window, min_periods=3).quantile(q)
            rows.append(series.rename(q))
        res = pd.concat(rows, axis=1).reset_index()
        return res.melt(id_vars="eval_episode_id", var_name="quantile", value_name="value")

    out = (
        base.groupby(["run", "p_fail"], group_keys=True)
        .apply(_per_group)
        .reset_index(level=[0, 1])
        .rename(columns={"run": "run", "p_fail": "p_fail"})
    )
    return out


def plot_capture_rate(summary_df: pd.DataFrame) -> None:
    """Bar plot of capture rate per run (optionally split by env_seed if provided)."""
    if summary_df.empty:
        print("No data to plot.")
        return
    color = "env_seed" if "env_seed" in summary_df.columns else "run"
    fig = px.bar(
        summary_df,
        x="run",
        y="capture_rate",
        color=color,
        range_y=[0, 1],
        title="Capture Rate by Run",
        labels={"capture_rate": "Capture rate", "run": "Run", "env_seed": "Env seed"},
    )
    fig.update_layout(template="plotly_white")
    fig.show()
    return fig


def plot_action_distribution(action_df: pd.DataFrame, title: str) -> None:
    """Stacked bar plot of action distribution per run/p_fail/env_seed."""
    if action_df.empty:
        print("No data to plot.")
        return
    action_col = action_df.columns[-2] if action_df.shape[1] >= 5 else action_df.columns[1]
    fig = px.bar(
        action_df,
        x="run",
        y="pct",
        color=action_col,
        facet_row="p_fail" if "p_fail" in action_df.columns else None,
        facet_col="env_seed" if "env_seed" in action_df.columns else None,
        title=title,
        labels={"pct": "Fraction of steps", "run": "Run", action_col: "Action"},
        barmode="stack",
    )
    fig.update_layout(template="plotly_white")
    fig.show()
    return fig


def episode_trajectories(
    df: pd.DataFrame,
    run: str,
    p_fail: float,
    pairs: Iterable[Tuple[int, int]],
) -> pd.DataFrame:
    """
    Slice the step-level data for selected (env_seed, episode) pairs in a run/p_fail.
    """
    if df.empty:
        return pd.DataFrame()
    pair_set = {(int(env_seed), int(ep)) for env_seed, ep in pairs}
    mask = (df["run"] == run) & (df["p_fail"] == p_fail)
    df = df.loc[mask].copy()
    df["env_seed_int"] = df["env_seed"].astype(int)
    df["episode_int"] = df["episode"].astype(int)
    keep = [(es, ep) in pair_set for es, ep in zip(df["env_seed_int"], df["episode_int"])]
    return df.loc[keep].drop(columns=["env_seed_int", "episode_int"])


def plot_positions(df: pd.DataFrame, size: int = 5) -> None:
    """
    Plot catcher/runner positions over steps for a filtered DataFrame.
    Assumes columns: step, catcher_x, catcher_y, runner_x, runner_y.
    """
    if df.empty:
        print("No data to plot.")
        return
    fig = go.Figure()
    group_cols = ["env_seed", "episode"] if "env_seed" in df else ["episode"]
    for (env_seed, ep), g in df.groupby(group_cols):
        label = f"Seed {env_seed}, ep{ep}" if len(group_cols) == 2 else f"ep{ep}"
        fig.add_trace(
            go.Scatter(
                x=g["catcher_y"],
                y=g["catcher_x"],
                mode="lines+markers",
                name=f"Catcher {label}",
                line=dict(color="#1f77b4"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=g["runner_y"],
                y=g["runner_x"],
                mode="lines+markers",
                name=f"Runner {label}",
                line=dict(color="#ff7f0e"),
                opacity=0.8,
            )
        )
    fig.update_xaxes(title_text="y", range=[-0.5, size - 0.5])
    fig.update_yaxes(title_text="x", range=[-0.5, size - 0.5], autorange="reversed")
    fig.update_layout(
        title="Trajectories (y=col, x=row)",
        width=650,
        height=650,
        legend=dict(title=None),
        template="plotly_white",
    )
    fig.show()
    return fig


def plot_run_trends(ep_df: pd.DataFrame, run: str, window: int = 50, title_prefix: Optional[str] = None) -> None:
    """
    Plot rolling capture rate and steps for a given run (ordered by eval_episode_id).
    """
    run_df = ep_df.loc[ep_df["run"] == run]
    if run_df.empty:
        print(f"No episode data for run={run}")
        return
    roll_df = rolling_episode_metrics(run_df, window=window)
    ttl = title_prefix or run
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{ttl} – Rolling capture rate", f"{ttl} – Rolling steps"),
        shared_xaxes=True,
    )
    p_fails = (
        sorted(roll_df["p_fail"].dropna().unique().tolist())
        if "p_fail" in roll_df
        else [None]
    )
    for pf in p_fails:
        sub = roll_df if pf is None else roll_df.loc[roll_df["p_fail"] == pf]
        label = f"p_fail={pf}" if pf is not None else "p_fail=?"
        fig.add_trace(
            go.Scatter(x=sub["eval_episode_id"], y=sub["roll_capture_rate"], mode="lines", name=label),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["eval_episode_id"], y=sub["roll_steps"], mode="lines", name=label
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="Capture rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=1, col=2)
    fig.update_xaxes(title_text="Eval episode id", row=1, col=1)
    fig.update_xaxes(title_text="Eval episode id", row=1, col=2)
    fig.update_layout(
        height=400,
        width=1000,
        legend_title_text=None,
        template="plotly_white",
        legend=dict(
            orientation="v",
            y=0.5,
            yanchor="middle",
            x=1.02,
        ),
        margin=dict(r=160),
    )
    fig.show()
    return fig


def plot_action_drift(action_mix: pd.DataFrame, title: str) -> None:
    """
    Plot rolling action fractions as stacked area chart using eval_episode_id.
    Expects columns: run, p_fail, eval_episode_id, action, frac.
    """
    if action_mix.empty:
        print("No action-mix data to plot.")
        return
    fig = px.area(
        action_mix,
        x="eval_episode_id",
        y="frac",
        color="action",
        title=title,
        labels={"eval_episode_id": "Eval episode id", "frac": "Action fraction (rolling)", "action": "Action"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(template="plotly_white")
    fig.show()
    return fig


def visit_density(
    df: pd.DataFrame,
    who: str = "catcher",
    capture_only: bool = False,
    use_post_capture_pos: bool = False,
) -> pd.DataFrame:
    """
    Compute visit density grid for catcher or runner.
    Set capture_only=True to filter to capture timesteps.
    If use_post_capture_pos=True, use *_x_next/ *_y_next for capture events so the grid
    reflects the post-move location where the capture triggered.
    Returns a pivot table indexed by x with columns y.
    """
    if df.empty:
        return pd.DataFrame()
    base = df.copy()
    if capture_only:
        base = base.loc[base["capture"]]
    if capture_only and use_post_capture_pos:
        x_col = f"{who}_x_next"
        y_col = f"{who}_y_next"
    else:
        x_col = f"{who}_x"
        y_col = f"{who}_y"
    if x_col not in base or y_col not in base:
        return pd.DataFrame()
    return base.pivot_table(index=x_col, columns=y_col, values="reward", aggfunc="count").fillna(0)


def plot_visit_heatmaps(df: pd.DataFrame, run: str, p_fail: float, env_seed: Optional[int] = None) -> None:
    """
    Heatmaps of visit density and capture locations.
    If env_seed is provided, filter to that seed; else combine all seeds for the run/p_fail.
    """
    sub = df.loc[(df["run"] == run) & (df["p_fail"] == p_fail)]
    if env_seed is not None:
        sub = sub.loc[sub["env_seed"] == env_seed]
    if sub.empty:
        msg = f"No data for run={run}, p_fail={p_fail}"
        msg += f", env_seed={env_seed}" if env_seed is not None else ""
        print(msg)
        return
    grid_c_visit = visit_density(sub, who="catcher", capture_only=False)
    grid_r_visit = visit_density(sub, who="runner", capture_only=False)
    cap_sub = sub.loc[sub["capture"]]
    if not cap_sub.empty:
        grid_capture = cap_sub.pivot_table(
            index="runner_x_next",
            columns="runner_y_next",
            values="reward",
            aggfunc="count",
        ).fillna(0)
    else:
        grid_capture = pd.DataFrame()

    grids = [
        ("Catcher visits", grid_c_visit),
        ("Runner visits", grid_r_visit),
        ("Capture locations (runner post-move)", grid_capture),
    ]

    colorbar_x = [0.31, 0.67, 1.03]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[t for t, _ in grids],
        horizontal_spacing=0.12,
    )
    for idx, (title, grid) in enumerate(grids, start=1):
        if grid.empty:
            fig.add_annotation(
                row=1,
                col=idx,
                text=f"{title}<br>(no data)",
                showarrow=False,
                font=dict(color="gray"),
            )
            continue
        total = grid.values.sum()
        vmax = grid.values.max() if grid.size else 1
        heat = go.Heatmap(
            z=grid.values,
            x=grid.columns,
            y=grid.index,
            colorscale="Blues",
            colorbar=dict(
                title="Count",
                x=colorbar_x[idx - 1],
                len=0.75,
                yanchor="middle",
                xanchor="center",
            ),
            showscale=True,
        )
        fig.add_trace(heat, row=1, col=idx)
        fig.update_xaxes(title_text=None, row=1, col=idx)
        fig.update_yaxes(title_text=None, row=1, col=idx, autorange="reversed")
        annotations = list(fig.layout.annotations) if fig.layout.annotations else []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid.values[i, j]
                pct = (val / total * 100) if total > 0 else 0
                norm = (val / vmax) if vmax > 0 else 0
                txt_color = "#FFFFFF" if norm > 0.6 else "#0A0A0A"
                annotations.append(
                    dict(
                        x=grid.columns[j],
                        y=grid.index[i],
                        text=f"<b>{int(val)}</b><br>({pct:.1f}%)",
                        showarrow=False,
                        font=dict(size=10, color=txt_color),
                        xref=f"x{idx}",
                        yref=f"y{idx}",
                        xanchor="center",
                        yanchor="middle",
                    )
                )
        fig.update_layout(annotations=tuple(annotations))

    title = f"Run={run}, p_fail={p_fail}"
    if env_seed is not None:
        title += f", env_seed={env_seed}"
    fig.update_layout(
        title=f"{title} visit densities",
        height=500,
        width=1800,
        template="plotly_white",
        margin=dict(r=240),
    )
    fig.show()
    return fig


def animate_episode(
    steps_df: pd.DataFrame,
    run: str,
    p_fail: float,
    env_seed: Optional[int] = None,
    episode: int | None = None,
    eval_episode_id: int | None = None,
    capture_only: bool = True,
    frame_ms: int = 700,
):
    """
    Animate a single eval episode within a run/p_fail.
    Identification priority:
      1) explicit (env_seed, episode)
      2) explicit eval_episode_id (maps to a unique pair)
      3) random pick of a capture episode (or any episode if none captured)
    """
    sub = steps_df[(steps_df["run"] == run) & (steps_df["p_fail"] == p_fail)].copy()
    if sub.empty:
        print(f"No data for run={run}, p_fail={p_fail}")
        return
    if "eval_episode_id" not in sub.columns:
        sub = _attach_eval_episode_id(sub)
    if env_seed is not None:
        sub = sub[sub["env_seed"] == env_seed]
    if sub.empty:
        print(f"No data for run={run}, p_fail={p_fail}, env_seed={env_seed}")
        return
    sub["step"] = sub["step"].astype(int)
    size = 5  # grid is 5x5

    # Resolve selection.
    if eval_episode_id is not None:
        row = sub.loc[sub["eval_episode_id"] == eval_episode_id]
        if row.empty:
            print(f"Eval episode id {eval_episode_id} not found for run={run}, p_fail={p_fail}")
            return
        env_seed = int(row["env_seed"].iloc[0])
        episode = int(row["episode"].iloc[0])
    elif env_seed is None or episode is None:
        candidates = sub[["env_seed", "episode", "eval_episode_id", "capture"]].drop_duplicates()
        if capture_only:
            candidates = candidates.loc[candidates["capture"]]
        if candidates.empty:
            candidates = sub[["env_seed", "episode", "eval_episode_id"]].drop_duplicates()
        pick = candidates.sample(1).iloc[0]
        env_seed = int(pick["env_seed"])
        episode = int(pick["episode"])

    epi = sub[(sub["env_seed"] == env_seed) & (sub["episode"] == episode)].copy()
    if epi.empty:
        print(f"Episode {episode} not found for run={run}, p_fail={p_fail}, env_seed={env_seed}")
        return
    epi = epi.sort_values("step")

    frames = []
    last_row = None
    for step in epi["step"].unique():
        g = epi[epi["step"] == step].iloc[0]
        last_row = g
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=[g["catcher_y"]], y=[g["catcher_x"]], mode="markers",
                        marker=dict(color="red", size=16), name="Catcher",
                        hovertemplate="Row: %{y}<br>Col: %{x}<extra>Catcher</extra>",
                    ),
                    go.Scatter(
                        x=[g["runner_y"]], y=[g["runner_x"]], mode="markers",
                        marker=dict(color="green", size=16), name="Runner",
                        hovertemplate="Row: %{y}<br>Col: %{x}<extra>Runner</extra>",
                    ),
                ],
                name=str(step),
            )
        )

    # Append a final frame at post-move positions when capture/done happens.
    if last_row is not None and (bool(last_row.get("capture")) or bool(last_row.get("done"))):
        final_step = int(last_row["step"]) + 1
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=[last_row["catcher_y_next"]], y=[last_row["catcher_x_next"]],
                        mode="markers", marker=dict(color="red", size=16), name="Catcher",
                        hovertemplate="Row: %{y}<br>Col: %{x}<extra>Catcher</extra>",
                    ),
                    go.Scatter(
                        x=[last_row["runner_y_next"]], y=[last_row["runner_x_next"]],
                        mode="markers", marker=dict(color="green", size=16), name="Runner",
                        hovertemplate="Row: %{y}<br>Col: %{x}<extra>Runner</extra>",
                    ),
                ],
                name=str(final_step),
            )
        )

    if not frames:
        print(f"No steps to animate for episode {episode}")
        return

    axis_common = dict(range=[0, size], tick0=0, dtick=1, tickmode="linear")
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(
                text=f"Animated trajectory <br>{run}, p_fail={p_fail}, env_seed={env_seed}, episode {episode}",
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(**axis_common, title="Table Column"),
            yaxis=dict(range=[size, 0], tick0=0, dtick=1, tickmode="linear", title="Table Row"),
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate",
                     "args": [None, {"frame": {"duration": frame_ms, "redraw": True}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}]},
                ]
            }],
            sliders=[{
                "currentvalue": {"prefix": "Step: "},
                "steps": [
                    {"label": f.name, "method": "animate",
                     "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
                    for f in frames
                ],
            }],
            width=650,
            height=650,
        ),
        frames=frames,
    )
    fig.show()


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
    "animate_episode",
]
