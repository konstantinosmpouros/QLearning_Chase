"""Evaluation analytics helpers for the extended Wumpus Chase logs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EVAL_STEPS_PATH = RESULTS_DIR / "wumpus_extended_eval.csv"
EVAL_MATRIX_PATH = RESULTS_DIR / "evaluation_results.csv"
EVAL_REPORT_PATH = RESULTS_DIR / "evaluation_results_report.txt"

TRUE_STRINGS = {"1", "true", "t", "yes", "y"}


def available_columns(path: Path | str) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def list_run_labels(
    path: Path | str = EVAL_STEPS_PATH,
    phase: str = "eval",
    chunksize: int = 300_000,
) -> list[str]:
    """List distinct run labels from the eval step log for a phase."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval steps file not found: {path}")

    runs: set[str] = set()
    usecols = ["run_label", "phase"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        sub = _filter_base_chunk(chunk, phase=phase, run_filter=None)
        if sub.empty:
            continue
        runs.update(sub["run_label"].astype(str).dropna().unique().tolist())
    return sorted(runs)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    text = series.astype(str).str.strip().str.lower()
    return text.isin(TRUE_STRINGS)


def _ensure_bool(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = _coerce_bool(out[col])
    return out


def _ensure_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _filter_base_chunk(
    chunk: pd.DataFrame,
    phase: Optional[str],
    run_filter: Optional[Iterable[str]],
) -> pd.DataFrame:
    out = chunk
    if phase and "phase" in out.columns:
        out = out.loc[out["phase"].astype(str).str.lower() == phase.lower()]
    if run_filter and "run_label" in out.columns:
        keep = {str(x) for x in run_filter}
        out = out.loc[out["run_label"].astype(str).isin(keep)]
    return out


def load_step_sample(
    path: Path | str = EVAL_STEPS_PATH,
    run_filter: Optional[Iterable[str]] = None,
    phase: str = "eval",
    usecols: Optional[Sequence[str]] = None,
    chunksize: int = 200_000,
    sample_frac: float = 0.05,
    max_rows: int = 250_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load a sampled subset from the large eval step CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval steps file not found: {path}")

    rng = np.random.default_rng(random_state)
    chunks = []
    collected = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        sub = _filter_base_chunk(chunk, phase=phase, run_filter=run_filter)
        if sub.empty:
            continue
        if sample_frac < 1.0 and len(sub) > 0:
            keep_n = max(1, int(len(sub) * sample_frac))
            idx = rng.choice(sub.index.to_numpy(), size=min(keep_n, len(sub)), replace=False)
            sub = sub.loc[idx]
        chunks.append(sub)
        collected += len(sub)
        if collected >= max_rows:
            break

    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state)
    return out.reset_index(drop=True)


def load_eval_matrix(path: Path | str = EVAL_MATRIX_PATH) -> pd.DataFrame:
    """Load pairwise evaluation table (agent_a vs agent_b)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation matrix not found: {path}")
    df = pd.read_csv(path)
    numeric = [
        "wins_a",
        "wins_b",
        "draws",
        "total_games",
        "win_rate_a",
        "win_rate_b",
        "draw_rate",
        "avg_steps",
        "avg_return",
    ]
    return _ensure_numeric(df, [c for c in numeric if c in df.columns])


def load_eval_report(path: Path | str = EVAL_REPORT_PATH) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation text report not found: {path}")
    return path.read_text(encoding="utf-8")


def leaderboard_from_matrix(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Average win rates by agent across both A/B roles."""
    if matrix_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in matrix_df.iterrows():
        rows.append({"agent": row["agent_a"], "role": "A", "win_rate": row.get("win_rate_a", np.nan)})
        rows.append({"agent": row["agent_b"], "role": "B", "win_rate": row.get("win_rate_b", np.nan)})
    out = pd.DataFrame(rows)
    return (
        out.groupby("agent", as_index=False)
        .agg(avg_win_rate=("win_rate", "mean"), samples=("win_rate", "count"))
        .sort_values("avg_win_rate", ascending=False)
    )


def pivot_matrix(matrix_df: pd.DataFrame, value: str = "win_rate_a") -> pd.DataFrame:
    """Agent-vs-agent pivot table."""
    if matrix_df.empty:
        return pd.DataFrame()
    if value not in matrix_df.columns:
        raise ValueError(f"Column '{value}' not found in matrix data.")
    return matrix_df.pivot(index="agent_a", columns="agent_b", values=value)


def plot_matrix_heatmap(matrix_df: pd.DataFrame, value: str = "win_rate_a"):
    """Heatmap for pairwise matrix values."""
    mat = pivot_matrix(matrix_df, value=value)
    if mat.empty:
        print("No matrix data.")
        return None
    fig = px.imshow(
        mat,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        title=f"Pairwise evaluation heatmap ({value})",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white")
    fig.show()
    return None


def plot_leaderboard(leaderboard_df: pd.DataFrame):
    """Bar chart of average win rate per agent."""
    if leaderboard_df.empty:
        print("No leaderboard data.")
        return None
    fig = px.bar(
        leaderboard_df,
        x="agent",
        y="avg_win_rate",
        text=leaderboard_df["avg_win_rate"].map(lambda v: f"{v:.3f}"),
        title="Average win rate across both roles",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(template="plotly_white")
    fig.show()
    return None


def add_step_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns on eval step data."""
    if df.empty:
        return df

    out = df.copy()
    if "run" not in out.columns and "run_label" in out.columns:
        out["run"] = out["run_label"]

    numeric_cols = [
        "episode",
        "step",
        "t_max",
        "p_fail",
        "reward",
        "reward_a",
        "reward_b",
        "state_ax",
        "state_ay",
        "state_bx",
        "state_by",
        "next_state_ax",
        "next_state_ay",
        "next_state_bx",
        "next_state_by",
        "layout_treasure_x",
        "layout_treasure_y",
        "layout_wumpus_x",
        "layout_wumpus_y",
    ]
    out = _ensure_numeric(out, numeric_cols)

    bool_cols = [
        "done",
        "capture",
        "a_dead",
        "b_dead",
        "a_treasure",
        "b_treasure",
        "a_fail",
        "b_fail",
        "a_in_pit",
        "b_in_pit",
        "a_met_wumpus",
        "b_met_wumpus",
        "a_breeze",
        "b_breeze",
        "a_stench",
        "b_stench",
    ]
    out = _ensure_bool(out, bool_cols)

    if "action_a_name" in out.columns:
        out["a_action"] = out["action_a_name"]
    elif "action_a" in out.columns:
        out["a_action"] = out["action_a"]

    if "action_b_name" in out.columns:
        out["b_action"] = out["action_b_name"]
    elif "action_b" in out.columns:
        out["b_action"] = out["action_b"]

    if {"state_ax", "state_ay", "state_bx", "state_by"}.issubset(out.columns):
        out["dist_ab"] = (out["state_ax"] - out["state_bx"]).abs() + (out["state_ay"] - out["state_by"]).abs()

    out["a_win"] = out.get("outcome", pd.Series(index=out.index, dtype=object)).eq("A_WIN")
    out["b_win"] = out.get("outcome", pd.Series(index=out.index, dtype=object)).eq("B_WIN")
    return out


def classify_terminal_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Classify why each eval episode ended."""
    if df.empty:
        return df

    out = df.copy()
    out = _ensure_bool(
        out,
        [
            "capture",
            "a_treasure",
            "b_treasure",
            "a_dead",
            "b_dead",
            "a_in_pit",
            "b_in_pit",
            "a_met_wumpus",
            "b_met_wumpus",
        ],
    )
    out = _ensure_numeric(out, ["step", "t_max"])

    reason = np.full(len(out), "other", dtype=object)

    capture_mask = out.get("capture", False)
    reason[capture_mask] = "capture"

    b_treasure = out.get("b_treasure", False)
    reason[~capture_mask & b_treasure] = "runner_treasure"

    a_treasure = out.get("a_treasure", False)
    reason[~capture_mask & ~b_treasure & a_treasure] = "chaser_treasure"

    b_dead = out.get("b_dead", False)
    runner_pit = b_dead & out.get("b_in_pit", False)
    runner_wumpus = b_dead & out.get("b_met_wumpus", False)
    reason[~capture_mask & ~b_treasure & ~a_treasure & runner_pit] = "runner_pit"
    reason[~capture_mask & ~b_treasure & ~a_treasure & ~runner_pit & runner_wumpus] = "runner_wumpus"
    reason[~capture_mask & ~b_treasure & ~a_treasure & ~runner_pit & ~runner_wumpus & b_dead] = "runner_hazard"

    a_dead = out.get("a_dead", False)
    chaser_pit = a_dead & out.get("a_in_pit", False)
    chaser_wumpus = a_dead & out.get("a_met_wumpus", False)
    reason[~capture_mask & ~b_treasure & ~a_treasure & chaser_pit] = "chaser_pit"
    reason[~capture_mask & ~b_treasure & ~a_treasure & ~chaser_pit & chaser_wumpus] = "chaser_wumpus"
    reason[~capture_mask & ~b_treasure & ~a_treasure & ~chaser_pit & ~chaser_wumpus & a_dead] = "chaser_hazard"

    timeout_mask = (
        out.get("outcome", pd.Series(index=out.index, dtype=object)).eq("B_WIN")
        & out["t_max"].notna()
        & out["step"].notna()
        & out["step"].ge(out["t_max"])
    )
    reason[reason == "other"] = np.where(timeout_mask[reason == "other"], "timeout", reason[reason == "other"])

    draw_mask = out.get("outcome", pd.Series(index=out.index, dtype=object)).eq("DRAW")
    reason[draw_mask] = "draw"

    out["terminal_reason"] = reason
    out["winner"] = np.where(out.get("outcome", "").eq("A_WIN"), "A", np.where(out.get("outcome", "").eq("B_WIN"), "B", "DRAW"))
    return out


def extract_terminal_rows(
    path: Path | str = EVAL_STEPS_PATH,
    run_filter: Optional[Iterable[str]] = None,
    phase: str = "eval",
    chunksize: int = 200_000,
    limit_episodes: Optional[int] = None,
) -> pd.DataFrame:
    """Stream eval CSV and keep only terminal rows (done=True)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval steps file not found: {path}")

    wanted = [
        "run_label",
        "phase",
        "agent_a",
        "agent_b",
        "role_a",
        "role_b",
        "episode",
        "step",
        "p_fail",
        "t_max",
        "size",
        "outcome",
        "winner_role",
        "done",
        "capture",
        "a_dead",
        "b_dead",
        "a_treasure",
        "b_treasure",
        "a_in_pit",
        "b_in_pit",
        "a_met_wumpus",
        "b_met_wumpus",
        "state_ax",
        "state_ay",
        "state_bx",
        "state_by",
        "next_state_ax",
        "next_state_ay",
        "next_state_bx",
        "next_state_by",
        "layout_treasure_x",
        "layout_treasure_y",
        "layout_wumpus_x",
        "layout_wumpus_y",
        "layout_obstacles",
        "layout_pits",
        "reward",
        "reward_a",
        "reward_b",
        "reward_outcome",
        "reward_step",
        "reward_hazard",
        "reward_bump",
        "reward_perception",
        "reward_obstacle",
        "reward_chase",
        "reward_treasure",
    ]
    header = set(available_columns(path))
    usecols = [c for c in wanted if c in header]

    rows = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        sub = _filter_base_chunk(chunk, phase=phase, run_filter=run_filter)
        if sub.empty:
            continue

        if "done" in sub.columns:
            done = _coerce_bool(sub["done"])
            sub = sub.loc[done]
        elif "outcome" in sub.columns:
            sub = sub.loc[sub["outcome"].isin(["A_WIN", "B_WIN", "DRAW"])]
        else:
            continue

        if sub.empty:
            continue
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    if limit_episodes is not None and len(out) > limit_episodes:
        if "run_label" in out.columns:
            # Keep run coverage when limiting rows by using balanced sampling.
            out = (
                out.groupby("run_label", group_keys=False)
                .apply(
                    lambda g: g.sample(
                        n=max(1, int(limit_episodes / max(1, out["run_label"].nunique()))),
                        random_state=42,
                    )
                    if len(g) > max(1, int(limit_episodes / max(1, out["run_label"].nunique())))
                    else g
                )
                .reset_index(drop=True)
            )
            if len(out) > limit_episodes:
                out = out.sample(n=limit_episodes, random_state=42)
        else:
            out = out.sample(n=limit_episodes, random_state=42)

    out = add_step_features(out)
    out = classify_terminal_reason(out)

    keys = ["run"] + (["p_fail"] if "p_fail" in out.columns else [])
    out["eval_episode_id"] = out.groupby(keys).cumcount() + 1
    return out.reset_index(drop=True)


def terminal_reason_rates(
    terminal_df: pd.DataFrame,
    group_cols: Sequence[str] = ("run", "p_fail"),
) -> pd.DataFrame:
    if terminal_df.empty:
        return pd.DataFrame()
    base = terminal_df.copy()
    keys = [c for c in group_cols if c in base.columns]
    if not keys:
        keys = ["run"] if "run" in base.columns else []
    if not keys:
        return pd.DataFrame()

    counts = (
        base.groupby(keys + ["terminal_reason"], as_index=False)
        .size()
        .rename(columns={"size": "episodes"})
    )
    totals = counts.groupby(keys, as_index=False)["episodes"].sum().rename(columns={"episodes": "total_episodes"})
    out = counts.merge(totals, on=keys, how="left")
    out["rate"] = out["episodes"] / out["total_episodes"]
    return out.sort_values(keys + ["rate"], ascending=[True] * len(keys) + [False])


def terminal_summary(terminal_df: pd.DataFrame) -> pd.DataFrame:
    if terminal_df.empty:
        return pd.DataFrame()

    base = terminal_df.copy()
    keys = ["run"]
    if "p_fail" in base.columns:
        keys.append("p_fail")

    for col in ["step", "reward", "reward_a", "reward_b"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    base["a_win"] = base.get("outcome", pd.Series(index=base.index, dtype=object)).eq("A_WIN")
    base["b_win"] = base.get("outcome", pd.Series(index=base.index, dtype=object)).eq("B_WIN")

    agg = base.groupby(keys, as_index=False).agg(
        episodes=("eval_episode_id", "count") if "eval_episode_id" in base.columns else ("terminal_reason", "count"),
        a_win_rate=("a_win", "mean"),
        b_win_rate=("b_win", "mean"),
        capture_rate=("capture", "mean") if "capture" in base.columns else ("a_win", "mean"),
        runner_treasure_rate=("terminal_reason", lambda s: (s == "runner_treasure").mean()),
        timeout_rate=("terminal_reason", lambda s: (s == "timeout").mean()),
        runner_hazard_rate=("terminal_reason", lambda s: s.isin(["runner_pit", "runner_wumpus", "runner_hazard"]).mean()),
        chaser_hazard_rate=("terminal_reason", lambda s: s.isin(["chaser_pit", "chaser_wumpus", "chaser_hazard"]).mean()),
        avg_steps=("step", "mean") if "step" in base.columns else ("a_win", "mean"),
        avg_reward_a=("reward_a", "mean") if "reward_a" in base.columns else ("reward", "mean"),
        avg_reward_b=("reward_b", "mean") if "reward_b" in base.columns else ("reward", "mean"),
    )
    return agg.sort_values("a_win_rate", ascending=False)


def reward_component_summary(terminal_df: pd.DataFrame) -> pd.DataFrame:
    if terminal_df.empty:
        return pd.DataFrame()
    cols = [
        "reward_outcome",
        "reward_step",
        "reward_hazard",
        "reward_bump",
        "reward_perception",
        "reward_obstacle",
        "reward_chase",
        "reward_treasure",
    ]
    existing = [c for c in cols if c in terminal_df.columns]
    if not existing:
        return pd.DataFrame()

    base = terminal_df.copy()
    base = _ensure_numeric(base, existing)
    keys = ["run"]
    if "p_fail" in base.columns:
        keys.append("p_fail")

    return base.groupby(keys, as_index=False)[existing].mean()


def hazard_breakdown(terminal_df: pd.DataFrame) -> pd.DataFrame:
    if terminal_df.empty:
        return pd.DataFrame()

    base = terminal_df.copy()
    keys = ["run"]
    if "p_fail" in base.columns:
        keys.append("p_fail")

    rows = []
    for _, row in base.iterrows():
        entry = {k: row[k] for k in keys}
        if row.get("terminal_reason") in {"runner_pit", "runner_wumpus", "runner_hazard"}:
            entry["side"] = "runner"
            entry["hazard_type"] = (
                "pit" if bool(row.get("b_in_pit", False)) else "wumpus" if bool(row.get("b_met_wumpus", False)) else "other"
            )
            rows.append(entry)
        if row.get("terminal_reason") in {"chaser_pit", "chaser_wumpus", "chaser_hazard"}:
            entry2 = {k: row[k] for k in keys}
            entry2["side"] = "chaser"
            entry2["hazard_type"] = (
                "pit" if bool(row.get("a_in_pit", False)) else "wumpus" if bool(row.get("a_met_wumpus", False)) else "other"
            )
            rows.append(entry2)

    if not rows:
        return pd.DataFrame(columns=keys + ["side", "hazard_type", "episodes", "rate"])

    hz = pd.DataFrame(rows)
    counts = hz.groupby(keys + ["side", "hazard_type"], as_index=False).size().rename(columns={"size": "episodes"})
    totals = counts.groupby(keys + ["side"], as_index=False)["episodes"].sum().rename(columns={"episodes": "hazard_total"})
    out = counts.merge(totals, on=keys + ["side"], how="left")
    out["rate"] = out["episodes"] / out["hazard_total"]
    return out.sort_values(keys + ["side", "rate"], ascending=[True] * len(keys) + [True, False])


def _parse_cells(serialized: object) -> list[tuple[int, int]]:
    if serialized is None or (isinstance(serialized, float) and np.isnan(serialized)):
        return []
    text = str(serialized).strip()
    if not text:
        return []
    out: list[tuple[int, int]] = []
    for item in text.split(";"):
        part = item.strip()
        if not part or "," not in part:
            continue
        x_str, y_str = part.split(",", 1)
        try:
            out.append((int(x_str), int(y_str)))
        except ValueError:
            continue
    return out


def terminal_position_grid(
    terminal_df: pd.DataFrame,
    actor: str = "b",
    use_next_state: bool = True,
    reason_filter: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if terminal_df.empty:
        return pd.DataFrame()

    base = terminal_df.copy()
    if reason_filter:
        keep = set(reason_filter)
        base = base.loc[base["terminal_reason"].isin(keep)]
    if base.empty:
        return pd.DataFrame()

    x_col = f"{'next_state_' if use_next_state else 'state_'}{actor}x"
    y_col = f"{'next_state_' if use_next_state else 'state_'}{actor}y"
    if x_col not in base.columns or y_col not in base.columns:
        return pd.DataFrame()

    return (
        base.pivot_table(index=x_col, columns=y_col, values="terminal_reason", aggfunc="count")
        .fillna(0)
        .sort_index()
        .sort_index(axis=1)
    )


def plot_terminal_reason_stacked(
    reason_rates_df: pd.DataFrame,
    title: str = "Evaluation terminal reason mix by run",
):
    if reason_rates_df.empty:
        print("No terminal reason data.")
        return None

    base = reason_rates_df.copy()
    x_col = "run"
    if "p_fail" in base.columns:
        base["run_pf"] = base["run"].astype(str) + " | p_fail=" + base["p_fail"].astype(str)
        x_col = "run_pf"

    fig = px.bar(
        base,
        x=x_col,
        y="rate",
        color="terminal_reason",
        text=base["rate"].map(lambda v: f"{v:.2f}"),
        title=title,
        labels={"rate": "Rate", x_col: "Run"},
    )
    fig.update_layout(barmode="stack", template="plotly_white", legend_title_text="Terminal reason")
    fig.update_yaxes(range=[0, 1])
    fig.show()
    return None


def plot_terminal_heatmap(
    terminal_df: pd.DataFrame,
    run: str,
    p_fail: Optional[float] = None,
    actor: str = "b",
    reason_filter: Optional[Sequence[str]] = None,
):
    sub = terminal_df.loc[terminal_df["run"] == run].copy()
    if p_fail is not None and "p_fail" in sub.columns:
        sub = sub.loc[sub["p_fail"] == p_fail]
    if sub.empty:
        print(f"No terminal rows for run={run}, p_fail={p_fail}")
        return None

    grid = terminal_position_grid(sub, actor=actor, use_next_state=True, reason_filter=reason_filter)
    if grid.empty:
        print("No terminal grid data after filtering.")
        return None

    first = sub.iloc[0]
    obstacles = _parse_cells(first.get("layout_obstacles", ""))
    pits = _parse_cells(first.get("layout_pits", ""))

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=grid.values,
            x=grid.columns,
            y=grid.index,
            colorscale="Blues",
            colorbar=dict(title="Terminal count"),
        )
    )

    if obstacles:
        fig.add_trace(
            go.Scatter(
                x=[y for x, y in obstacles],
                y=[x for x, y in obstacles],
                mode="markers",
                marker=dict(symbol="square", color="#2f2f2f", size=12),
                name="Obstacles",
            )
        )
    if pits:
        fig.add_trace(
            go.Scatter(
                x=[y for x, y in pits],
                y=[x for x, y in pits],
                mode="markers",
                marker=dict(symbol="x", color="#8e44ad", size=12),
                name="Pits",
            )
        )

    if "layout_wumpus_x" in sub.columns and "layout_wumpus_y" in sub.columns:
        wx = pd.to_numeric(first.get("layout_wumpus_x"), errors="coerce")
        wy = pd.to_numeric(first.get("layout_wumpus_y"), errors="coerce")
        if not np.isnan(wx) and not np.isnan(wy):
            fig.add_trace(
                go.Scatter(
                    x=[int(wy)],
                    y=[int(wx)],
                    mode="markers",
                    marker=dict(symbol="diamond", size=14, color="#c0392b"),
                    name="Wumpus",
                )
            )

    if "layout_treasure_x" in sub.columns and "layout_treasure_y" in sub.columns:
        tx = pd.to_numeric(first.get("layout_treasure_x"), errors="coerce")
        ty = pd.to_numeric(first.get("layout_treasure_y"), errors="coerce")
        if not np.isnan(tx) and not np.isnan(ty):
            fig.add_trace(
                go.Scatter(
                    x=[int(ty)],
                    y=[int(tx)],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color="#f1c40f"),
                    name="Treasure",
                )
            )

    reason_label = ",".join(reason_filter) if reason_filter else "all terminal reasons"
    fig.update_layout(
        title=f"Eval terminal positions ({actor.upper()}) - {run}, p_fail={p_fail}, reasons={reason_label}",
        xaxis_title="y",
        yaxis_title="x",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        width=760,
        height=620,
        legend_title_text=None,
    )
    fig.show()
    return None


def plot_reward_components(terminal_df: pd.DataFrame):
    summary = reward_component_summary(terminal_df)
    if summary.empty:
        print("No reward component data.")
        return None

    id_cols = ["run"] + (["p_fail"] if "p_fail" in summary.columns else [])
    long = summary.melt(id_vars=id_cols, var_name="component", value_name="mean_value")
    if "p_fail" in id_cols:
        long["run_pf"] = long["run"].astype(str) + " | p_fail=" + long["p_fail"].astype(str)
        x_col = "run_pf"
    else:
        x_col = "run"

    fig = px.bar(
        long,
        x=x_col,
        y="mean_value",
        color="component",
        barmode="group",
        title="Eval: mean reward components at terminal steps",
    )
    fig.update_layout(template="plotly_white", legend_title_text="Component")
    fig.show()
    return None


def plot_terminal_steps_hist(terminal_df: pd.DataFrame):
    if terminal_df.empty or "step" not in terminal_df.columns:
        print("No terminal step data.")
        return None

    fig = px.histogram(
        terminal_df,
        x="step",
        color="terminal_reason",
        barmode="stack",
        nbins=40,
        title="Eval terminal step distribution by reason",
    )
    fig.update_layout(template="plotly_white")
    fig.show()
    return None


__all__ = [
    "RESULTS_DIR",
    "EVAL_STEPS_PATH",
    "EVAL_MATRIX_PATH",
    "EVAL_REPORT_PATH",
    "available_columns",
    "list_run_labels",
    "load_step_sample",
    "load_eval_matrix",
    "load_eval_report",
    "leaderboard_from_matrix",
    "pivot_matrix",
    "plot_matrix_heatmap",
    "plot_leaderboard",
    "add_step_features",
    "classify_terminal_reason",
    "extract_terminal_rows",
    "terminal_reason_rates",
    "terminal_summary",
    "reward_component_summary",
    "hazard_breakdown",
    "terminal_position_grid",
    "plot_terminal_reason_stacked",
    "plot_terminal_heatmap",
    "plot_reward_components",
    "plot_terminal_steps_hist",
]
