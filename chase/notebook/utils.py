"""
Helper utilities for the Chase analytics notebook.
Centralizes repeatable tasks like loading the Excel logs,
adding derived columns, computing episode metrics, and
quick plotting helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "chase_train.xlsx"


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
        g["roll_capture_rate"] = g["captured"].rolling(window, min_periods=10).mean()
        g["roll_steps"] = g["steps"].rolling(window, min_periods=10).mean()
        g["roll_return"] = g["total_return"].rolling(window, min_periods=10).mean()
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
        roll = counts.rolling(window, min_periods=10).sum()
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
            series = g["total_return"].rolling(window, min_periods=10).quantile(q)
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
    fig = px.bar(
        summary_df,
        x="run",
        y="capture_rate",
        range_y=[0, 1],
        title="Capture Rate by Run",
        labels={"capture_rate": "Capture rate", "run": "Run"},
        color="run",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(showlegend=False)
    fig.show()
    return fig


def plot_action_distribution(action_df: pd.DataFrame, title: str) -> None:
    """
    Stacked bar plot of action distribution per run.
    Expects columns: run, <action_col>, pct
    """
    if action_df.empty:
        print("No data to plot.")
        return
    action_col = action_df.columns[1]
    fig = px.bar(
        action_df,
        x="run",
        y="pct",
        color=action_col,
        title=title,
        labels={"pct": "Fraction of steps", "run": "Run", action_col: "Action"},
        barmode="stack",
    )
    fig.show()
    return fig


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
    fig = go.Figure()
    for ep, g in df.groupby("episode"):
        fig.add_trace(
            go.Scatter(
                x=g["catcher_y"],
                y=g["catcher_x"],
                mode="lines+markers",
                name=f"Catcher ep{ep}",
                line=dict(color="#1f77b4"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=g["runner_y"],
                y=g["runner_x"],
                mode="lines+markers",
                name=f"Runner ep{ep}",
                line=dict(color="#ff7f0e"),
                opacity=0.8,
            )
        )
    fig.update_xaxes(title_text="y", range=[-0.5, size - 0.5])
    fig.update_yaxes(title_text="x", range=[-0.5, size - 0.5], autorange="reversed")
    fig.update_layout(
        title="Trajectories (y=col, x=row)",
        width=600,
        height=600,
        legend=dict(title=None),
        template="plotly_white",
    )
    fig.show()
    return fig


def plot_run_trends(ep_df: pd.DataFrame, run: str, window: int = 200, title_prefix: Optional[str] = None) -> None:
    """
    Plot rolling capture rate and steps for a given run, split by p_fail.
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
            go.Scatter(x=sub["episode"], y=sub["roll_capture_rate"], mode="lines", name=label),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["episode"], y=sub["roll_steps"], mode="lines", name=label
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="Capture rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=1, col=2)
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=1, col=2)
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
    Plot rolling action fractions as stacked area chart.
    Expects columns: run, p_fail, episode, action, frac.
    """
    if action_mix.empty:
        print("No action-mix data to plot.")
        return
    fig = px.area(
        action_mix,
        x="episode",
        y="frac",
        color="action",
        title=title,
        labels={"episode": "Episode", "frac": "Action fraction (rolling)", "action": "Action"},
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


def plot_visit_heatmaps(df: pd.DataFrame, run: str, p_fail: float) -> None:
    """
    Heatmaps of visit density and capture locations for catcher and runner.
    """
    sub = df.loc[(df["run"] == run) & (df["p_fail"] == p_fail)]
    if sub.empty:
        print(f"No data for run={run}, p_fail={p_fail}")
        return
    # Visits for each agent (all timesteps).
    grid_c_visit = visit_density(sub, who="catcher", capture_only=False)
    grid_r_visit = visit_density(sub, who="runner", capture_only=False)
    # Capture locations: where the runner ends up on capture timesteps (i.e., where it got caught).
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

    # Position separate colorbars so they do not overlap.
    # Place colorbars to the right of each subplot (outside the plot area).
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
        # Add annotations for counts with contrast-aware coloring.
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

    fig.update_layout(
        title=f"Run={run}, p_fail={p_fail} visit densities",
        height=500,
        width=1800,
        template="plotly_white",
        margin=dict(r=240),
    )
    fig.show()
    return fig


def animate_episode(steps_df: pd.DataFrame, run: str, p_fail: float, episode: int | None = None, capture_only: bool = True, frame_ms: int = 700):
    """Animate a single episode (catcher vs runner) for a given run/p_fail.
    If episode is None, pick a random capture episode (or any episode if none captured).
    frame_ms controls speed (ms per frame)."""
    sub = steps_df[(steps_df['run'] == run) & (steps_df['p_fail'] == p_fail)].copy()
    if sub.empty:
        print(f"No data for run={run}, p_fail={p_fail}")
        return
    sub['step'] = sub['step'].astype(int)
    size = 5  # grid is 5x5

    if episode is None:
        caps = sub.loc[sub['capture'], 'episode'].unique()
        if capture_only and caps.size > 0:
            episode = int(np.random.choice(caps))
        else:
            episode = int(np.random.choice(sub['episode'].unique()))
    epi = sub[sub['episode'] == episode].copy()
    if epi.empty:
        print(f"Episode {episode} not found for run={run}, p_fail={p_fail}")
        return
    epi = epi.sort_values('step')

    frames = []
    last_row = None
    for step in epi['step'].unique():
        g = epi[epi['step'] == step].iloc[0]
        last_row = g
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=[g['catcher_y']], y=[g['catcher_x']], mode='markers',
                        marker=dict(color='red', size=16), name='Catcher',
                        hovertemplate="Row: %{y}<br>Col: %{x}<extra>Catcher</extra>",
                    ),
                    go.Scatter(
                        x=[g['runner_y']], y=[g['runner_x']], mode='markers',
                        marker=dict(color='green', size=16), name='Runner',
                        hovertemplate="Row: %{y}<br>Col: %{x}<extra>Runner</extra>",
                    ),
                ],
                name=str(step),
            )
        )

    # Append a final frame at post-move positions when capture/done happens.
    if last_row is not None and (bool(last_row.get('capture')) or bool(last_row.get('done'))):
        final_step = int(last_row['step']) + 1
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=[last_row['catcher_y_next']], y=[last_row['catcher_x_next']],
                        mode='markers', marker=dict(color='red', size=16), name='Catcher',
                        hovertemplate="Row: %{y}<br>Col: %{x}<extra>Catcher</extra>",
                    ),
                    go.Scatter(
                        x=[last_row['runner_y_next']], y=[last_row['runner_x_next']],
                        mode='markers', marker=dict(color='green', size=16), name='Runner',
                        hovertemplate="Row: %{y}<br>Col: %{x}<extra>Runner</extra>",
                    ),
                ],
                name=str(final_step),
            )
        )

    if not frames:
        print(f"No steps to animate for episode {episode}")
        return

    axis_common = dict(range=[0, size], tick0=0, dtick=1, tickmode='linear')
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(text=f"Animated trajectory <br>{run}, p_fail={p_fail}, episode {episode}", x=0.5, xanchor='center'),
            xaxis=dict(**axis_common, title='Table Column'),
            yaxis=dict(range=[size, 0], tick0=0, dtick=1, tickmode='linear', title='Table Row'),  # origin top-left
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {'label': 'Play', 'method': 'animate',
                        'args': [None, {'frame': {'duration': frame_ms, 'redraw': True}, 'fromcurrent': True}]},
                    {'label': 'Pause', 'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate', 'transition': {'duration': 0}}]},
                ]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Step: '},
                'steps': [
                    {'label': f.name, 'method': 'animate',
                        'args': [[f.name], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}}]}
                    for f in frames
                ],
            }],
            width=600,
            height=600,
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
    "animate_episode"
]
