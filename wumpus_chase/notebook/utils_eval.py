"""Evaluation-specific analytics helpers for the Wumpus Chase project.
Uses eval_episode_id to keep evaluation episodes unique even when episode
counters restart across evaluation rounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "wumpus_eval.csv"


def load_runs(path: Path | str = RESULTS_PATH) -> Dict[str, pd.DataFrame]:
    """Read the CSV log and return a dict of DataFrames keyed by run label."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    if "phase" in df.columns:
        df = df.loc[df["phase"] == "eval"]
    if "run_label" not in df.columns:
        return {"run": df}
    runs: Dict[str, pd.DataFrame] = {}
    for run_name, g in df.groupby("run_label"):
        runs[str(run_name)] = g.reset_index(drop=True)
    return runs


def combine_runs(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all run DataFrames into a single DataFrame with a 'run' column."""
    dfs = []
    for run_name, df in sheets.items():
        df = df.copy()
        df["run"] = run_name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _coerce_bool(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].fillna(False).astype(bool)


def _attach_eval_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "step" not in df.columns:
        return df
    base = df.copy()
    if "run" not in base.columns and "run_label" in base.columns:
        base["run"] = base["run_label"]
    base["_log_idx"] = np.arange(len(base))
    group_cols = ["run"]
    if "p_fail" in base.columns:
        group_cols.append("p_fail")

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("_log_idx")
        starts = g["step"].eq(1)
        g["eval_episode_id"] = starts.cumsum()
        return g

    out = base.groupby(group_cols, group_keys=False).apply(_per_group)
    return out.drop(columns=["_log_idx"])


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add helpful derived columns to a step-level eval DataFrame."""
    if df.empty:
        return df
    out = df.copy()
    if "run" not in out.columns and "run_label" in out.columns:
        out["run"] = out["run_label"]

    out["step"] = out["step"].astype(int)
    out["episode"] = out["episode"].astype(int)

    out["a_x"] = out["state_ax"]
    out["a_y"] = out["state_ay"]
    out["b_x"] = out["state_bx"]
    out["b_y"] = out["state_by"]
    out["a_x_next"] = out["next_state_ax"]
    out["a_y_next"] = out["next_state_ay"]
    out["b_x_next"] = out["next_state_bx"]
    out["b_y_next"] = out["next_state_by"]

    if "action_a_name" in out.columns:
        out["a_action"] = out["action_a_name"]
    else:
        out["a_action"] = out["action_a"]
    if "action_b_name" in out.columns:
        out["b_action"] = out["action_b_name"]
    else:
        out["b_action"] = out["action_b"]
    out["a_action"] = out["a_action"].astype("category")
    out["b_action"] = out["b_action"].astype("category")

    _coerce_bool(out, "capture")
    _coerce_bool(out, "done")
    _coerce_bool(out, "a_fail")
    _coerce_bool(out, "b_fail")
    _coerce_bool(out, "a_dead")
    _coerce_bool(out, "b_dead")
    _coerce_bool(out, "a_treasure")
    _coerce_bool(out, "b_treasure")

    out["a_win"] = out["outcome"] == "A_WIN"
    out["b_win"] = out["outcome"] == "B_WIN"
    out["draw"] = out["outcome"] == "DRAW"

    out["manhattan_dist"] = (
        (out["a_x"] - out["b_x"]).abs()
        + (out["a_y"] - out["b_y"]).abs()
    )

    out = _attach_eval_episode_id(out)
    return out


def episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute episode-level aggregates per unique eval_episode_id."""
    if df.empty:
        return pd.DataFrame()
    keys = ["run", "eval_episode_id"]
    if "p_fail" in df.columns:
        keys = ["run", "p_fail", "eval_episode_id"]
    agg = (
        df.groupby(keys, as_index=False)
        .agg(
            episode=("episode", "first"),
            captured=("capture", "max"),
            steps=("step", "max"),
            total_return=("reward", "sum"),
            a_win=("a_win", "max"),
            b_win=("b_win", "max"),
            draw=("draw", "max"),
            eval_episode_id=("eval_episode_id", "first"),
        )
    )
    return agg


def capture_rate_by_run(
    ep_df: pd.DataFrame,
    include_seed: bool = True,
    include_eval_round: bool = False,
) -> pd.DataFrame:
    """Summarize win and capture rates and averages per run."""
    if ep_df.empty:
        return pd.DataFrame()
    keys = ["run"]
    if "p_fail" in ep_df.columns:
        keys.append("p_fail")
    if include_seed and "env_seed" in ep_df.columns:
        keys.append("env_seed")
    if include_eval_round and "eval_at_episode" in ep_df.columns:
        keys.append("eval_at_episode")
    grp = ep_df.groupby(keys, as_index=False)
    return grp.agg(
        a_win_rate=("a_win", "mean"),
        b_win_rate=("b_win", "mean"),
        draw_rate=("draw", "mean"),
        capture_rate=("captured", "mean"),
        avg_steps=("steps", "mean"),
        avg_return=("total_return", "mean"),
        eval_episode_id=("eval_episode_id", "mean"),
    )


def rolling_episode_metrics(ep_df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Add rolling means of Chaser win rate, capture rate, steps, and return per run and p_fail."""
    if ep_df.empty:
        return ep_df
    base = ep_df.copy()
    if "p_fail" not in base.columns:
        base["p_fail"] = None
    if "eval_episode_id" not in base.columns:
        base["eval_episode_id"] = base.groupby(["run", "p_fail"]).cumcount() + 1

    def _add_roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("eval_episode_id")
        g["roll_a_win_rate"] = g["a_win"].rolling(window, min_periods=3).mean()
        g["roll_capture_rate"] = g["captured"].rolling(window, min_periods=3).mean()
        g["roll_steps"] = g["steps"].rolling(window, min_periods=3).mean()
        g["roll_return"] = g["total_return"].rolling(window, min_periods=3).mean()
        return g

    return base.groupby(["run", "p_fail"], group_keys=False).apply(_add_roll)


def rolling_action_mix(df: pd.DataFrame, who: str = "a", window: int = 200) -> pd.DataFrame:
    """Rolling action distribution over eval_episode_id per run and p_fail."""
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
    """Rolling quantiles of total_return per run and p_fail ordered by eval_episode_id."""
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


def plot_run_trends(ep_df: pd.DataFrame, run: str, window: int = 50, title_prefix: Optional[str] = None) -> None:
    """Plot rolling Chaser win rate and steps for a given run ordered by eval_episode_id."""
    run_df = ep_df.loc[ep_df["run"] == run]
    if run_df.empty:
        print(f"No episode data for run={run}")
        return
    roll_df = rolling_episode_metrics(run_df, window=window)
    ttl = title_prefix or run
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{ttl} - Rolling win rate", f"{ttl} - Rolling steps"),
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
            go.Scatter(x=sub["eval_episode_id"], y=sub["roll_a_win_rate"], mode="lines", name=label),
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
    fig.update_yaxes(title_text="Chaser win rate", range=[0, 1], row=1, col=1)
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
    """Plot rolling action fractions as stacked area chart using eval_episode_id."""
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
    who: str = "a",
    capture_only: bool = False,
    use_post_capture_pos: bool = False,
) -> pd.DataFrame:
    """Compute visit density grid for chaser or B."""
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
    """Heatmaps of visit density and capture locations for agents A and B."""
    sub = df.loc[(df["run"] == run) & (df["p_fail"] == p_fail)]
    if sub.empty:
        print(f"No data for run={run}, p_fail={p_fail}")
        return
    grid_a_visit = visit_density(sub, who="a", capture_only=False)
    grid_b_visit = visit_density(sub, who="b", capture_only=False)
    cap_sub = sub.loc[sub["capture"]]
    if not cap_sub.empty:
        grid_capture = cap_sub.pivot_table(
            index="b_x_next",
            columns="b_y_next",
            values="reward",
            aggfunc="count",
        ).fillna(0)
    else:
        grid_capture = pd.DataFrame()

    grids = [
        ("Chaser visits", grid_a_visit),
        ("Runner visits", grid_b_visit),
        ("Capture locations (Runner post-move)", grid_capture),
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

    fig.update_layout(
        title=f"Run={run}, p_fail={p_fail} visit densities",
        height=500,
        width=1800,
        template="plotly_white",
        margin=dict(r=240),
    )
    fig.show()
    return fig


def episode_trajectories(
    df: pd.DataFrame,
    run: str,
    p_fail: float,
    eval_episode_ids: Iterable[int],
) -> pd.DataFrame:
    """Slice the step-level data for selected eval_episode_id values."""
    if df.empty:
        return pd.DataFrame()
    eid_set = set(int(eid) for eid in eval_episode_ids)
    mask = (df["run"] == run) & (df["p_fail"] == p_fail) & (df["eval_episode_id"].isin(eid_set))
    return df.loc[mask].copy()


def plot_positions(df: pd.DataFrame, size: int | None = None) -> None:
    """Plot chaser and B positions over steps for a filtered DataFrame."""
    if df.empty:
        print("No data to plot.")
        return
    if size is None and "size" in df.columns:
        size = int(df["size"].mode().iloc[0])
    if size is None:
        size = 7
    fig = go.Figure()
    for eid, g in df.groupby("eval_episode_id"):
        fig.add_trace(
            go.Scatter(
                x=g["a_y"],
                y=g["a_x"],
                mode="lines+markers",
                name=f"Chaser eid{eid}",
                line=dict(color="#1f77b4"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=g["b_y"],
                y=g["b_x"],
                mode="lines+markers",
                name=f"Runner eid{eid}",
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


def animate_episode(
    steps_df: pd.DataFrame,
    run: str,
    p_fail: float,
    episode: int | None = None,
    eval_episode_id: int | None = None,
    capture_only: bool = True,
    frame_ms: int = 700,
):
    """Animate a single eval episode within a run and p_fail."""
    sub = steps_df[(steps_df["run"] == run) & (steps_df["p_fail"] == p_fail)].copy()
    if sub.empty:
        print(f"No data for run={run}, p_fail={p_fail}")
        return
    if "eval_episode_id" not in sub.columns:
        sub = _attach_eval_episode_id(sub)
    sub["step"] = sub["step"].astype(int)
    size = int(sub["size"].mode().iloc[0]) if "size" in sub.columns else 7

    if eval_episode_id is not None:
        row = sub.loc[sub["eval_episode_id"] == eval_episode_id]
        if row.empty:
            print(f"Eval episode id {eval_episode_id} not found for run={run}, p_fail={p_fail}")
            return
        episode = int(row["episode"].iloc[0])
    elif episode is None:
        candidates = sub[["eval_episode_id", "capture"]].drop_duplicates()
        if capture_only:
            candidates = candidates.loc[candidates["capture"]]
        if candidates.empty:
            candidates = sub[["eval_episode_id"]].drop_duplicates()
        pick = candidates.sample(1).iloc[0]
        eval_episode_id = int(pick["eval_episode_id"])
    if eval_episode_id is None:
        eval_episode_id = int(sub["eval_episode_id"].iloc[0])

    epi = sub[sub["eval_episode_id"] == eval_episode_id].copy()
    if epi.empty:
        print(f"Eval episode id {eval_episode_id} not found for run={run}, p_fail={p_fail}")
        return
    epi = epi.sort_values("step")

    layout_row = epi.iloc[0]
    obstacles = []
    layout_obstacles = str(layout_row.get("layout_obstacles", "") or "")
    for item in layout_obstacles.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            x_str, y_str = item.split(",")
            obstacles.append((int(x_str), int(y_str)))
        except ValueError:
            continue

    static_traces = []
    if obstacles:
        obs_x = [y for x, y in obstacles]
        obs_y = [x for x, y in obstacles]
        static_traces.append(
            go.Scatter(
                x=obs_x,
                y=obs_y,
                mode="markers",
                marker=dict(symbol="square", size=18, color="#3b3b3b"),
                name="Obstacle",
                hovertemplate="Obstacle<extra></extra>",
            )
        )

    if "layout_treasure_x" in epi.columns and "layout_treasure_y" in epi.columns:
        try:
            tx = int(layout_row["layout_treasure_x"])
            ty = int(layout_row["layout_treasure_y"])
        except (TypeError, ValueError):
            tx = None
            ty = None
        if tx is not None and ty is not None:
            static_traces.append(
                go.Scatter(
                    x=[ty],
                    y=[tx],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=18,
                        color="#f1c40f",
                        line=dict(color="#b7950b", width=1),
                    ),
                    name="Treasure",
                    hovertemplate="Treasure<extra></extra>",
                )
            )

    frames = []
    last_row = None
    for step in epi["step"].unique():
        g = epi[epi["step"] == step].iloc[0]
        last_row = g
        chaser_trace = go.Scatter(
            x=[g["a_y"]], y=[g["a_x"]], mode="markers",
            marker=dict(color="red", size=16), name="Chaser",
            hovertemplate="Row: %{y}<br>Col: %{x}<extra>Chaser</extra>",
        )
        runner_trace = go.Scatter(
            x=[g["b_y"]], y=[g["b_x"]], mode="markers",
            marker=dict(color="green", size=16), name="Runner",
            hovertemplate="Row: %{y}<br>Col: %{x}<extra>Runner</extra>",
        )
        frames.append(
            go.Frame(
                data=static_traces + [chaser_trace, runner_trace],
                name=str(step),
            )
        )

    if last_row is not None and (bool(last_row.get("capture")) or bool(last_row.get("done"))):
        final_step = int(last_row["step"]) + 1
        chaser_trace = go.Scatter(
            x=[last_row["a_y_next"]], y=[last_row["a_x_next"]],
            mode="markers", marker=dict(color="red", size=16), name="Chaser",
            hovertemplate="Row: %{y}<br>Col: %{x}<extra>Chaser</extra>",
        )
        runner_trace = go.Scatter(
            x=[last_row["b_y_next"]], y=[last_row["b_x_next"]],
            mode="markers", marker=dict(color="green", size=16), name="Runner",
            hovertemplate="Row: %{y}<br>Col: %{x}<extra>Runner</extra>",
        )
        frames.append(
            go.Frame(
                data=static_traces + [chaser_trace, runner_trace],
                name=str(final_step),
            )
        )

    if not frames:
        print(f"No steps to animate for eval episode {eval_episode_id}")
        return

    axis_common = dict(range=[0, size], tick0=0, dtick=1, tickmode="linear")
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(
                text=f"Animated trajectory <br>{run}, p_fail={p_fail}, eval id {eval_episode_id}",
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
    "rolling_episode_metrics",
    "rolling_action_mix",
    "rolling_return_quantiles",
    "plot_run_trends",
    "plot_action_drift",
    "visit_density",
    "plot_visit_heatmaps",
    "episode_trajectories",
    "plot_positions",
    "animate_episode",
]

