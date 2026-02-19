"""Behavioral analytics helpers shared by train/eval notebooks."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _group_keys(df: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    if "run" in df.columns:
        keys.append("run")
    elif "run_label" in df.columns:
        keys.append("run_label")
    if "p_fail" in df.columns:
        keys.append("p_fail")
    return keys


def ensure_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and add behavior-relevant derived features."""
    if df.empty:
        return df

    out = df.copy()
    if "run" not in out.columns and "run_label" in out.columns:
        out["run"] = out["run_label"].astype(str)

    numeric_cols = [
        "episode",
        "step",
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
        "dist_ab_before",
        "dist_ab_after",
        "a_treasure_before",
        "a_treasure_after",
        "b_treasure_before",
        "b_treasure_after",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "a_action" not in out.columns:
        if "action_a_name" in out.columns:
            out["a_action"] = out["action_a_name"]
        elif "action_a" in out.columns:
            out["a_action"] = out["action_a"]

    if "b_action" not in out.columns:
        if "action_b_name" in out.columns:
            out["b_action"] = out["action_b_name"]
        elif "action_b" in out.columns:
            out["b_action"] = out["action_b"]

    if {"state_ax", "state_ay", "state_bx", "state_by"}.issubset(out.columns):
        out["dist_ab"] = (out["state_ax"] - out["state_bx"]).abs() + (out["state_ay"] - out["state_by"]).abs()
    if {"next_state_ax", "next_state_ay", "next_state_bx", "next_state_by"}.issubset(out.columns):
        out["dist_ab_next"] = (out["next_state_ax"] - out["next_state_bx"]).abs() + (out["next_state_ay"] - out["next_state_by"]).abs()

    if "dist_ab_before" in out.columns and "dist_ab_after" in out.columns:
        out["dist_ab_delta"] = out["dist_ab_after"] - out["dist_ab_before"]
    elif "dist_ab" in out.columns and "dist_ab_next" in out.columns:
        out["dist_ab_delta"] = out["dist_ab_next"] - out["dist_ab"]

    if {"layout_treasure_x", "layout_treasure_y", "state_ax", "state_ay"}.issubset(out.columns):
        out["dist_a_treasure"] = (out["state_ax"] - out["layout_treasure_x"]).abs() + (out["state_ay"] - out["layout_treasure_y"]).abs()
    if {"layout_treasure_x", "layout_treasure_y", "next_state_ax", "next_state_ay"}.issubset(out.columns):
        out["dist_a_treasure_next"] = (out["next_state_ax"] - out["layout_treasure_x"]).abs() + (out["next_state_ay"] - out["layout_treasure_y"]).abs()
    if {"layout_treasure_x", "layout_treasure_y", "state_bx", "state_by"}.issubset(out.columns):
        out["dist_b_treasure"] = (out["state_bx"] - out["layout_treasure_x"]).abs() + (out["state_by"] - out["layout_treasure_y"]).abs()
    if {"layout_treasure_x", "layout_treasure_y", "next_state_bx", "next_state_by"}.issubset(out.columns):
        out["dist_b_treasure_next"] = (out["next_state_bx"] - out["layout_treasure_x"]).abs() + (out["next_state_by"] - out["layout_treasure_y"]).abs()

    if "b_treasure_before" in out.columns and "b_treasure_after" in out.columns:
        out["dist_b_treasure"] = out["b_treasure_before"]
        out["dist_b_treasure_next"] = out["b_treasure_after"]

    if {"state_ax", "state_ay", "next_state_ax", "next_state_ay"}.issubset(out.columns):
        out["a_moved"] = ((out["state_ax"] != out["next_state_ax"]) | (out["state_ay"] != out["next_state_ay"]))
    if {"state_bx", "state_by", "next_state_bx", "next_state_by"}.issubset(out.columns):
        out["b_moved"] = ((out["state_bx"] != out["next_state_bx"]) | (out["state_by"] != out["next_state_by"]))

    return out


def role_behavior_fingerprint(step_df: pd.DataFrame) -> pd.DataFrame:
    """High-level role behavior summary (requested item 1)."""
    if step_df.empty:
        return pd.DataFrame()
    base = ensure_behavior_features(step_df)
    keys = _group_keys(base)
    if not keys:
        return pd.DataFrame()

    base["chaser_closing"] = base.get("dist_ab_delta", np.nan) < 0
    base["runner_opening"] = base.get("dist_ab_delta", np.nan) > 0
    base["a_stay"] = base.get("a_action", pd.Series(index=base.index, dtype=object)).astype(str).eq("STAY")
    base["b_stay"] = base.get("b_action", pd.Series(index=base.index, dtype=object)).astype(str).eq("STAY")

    out = base.groupby(keys, as_index=False).agg(
        steps=("step", "count") if "step" in base.columns else (keys[0], "count"),
        chaser_closing_rate=("chaser_closing", "mean"),
        runner_opening_rate=("runner_opening", "mean"),
        chaser_stay_rate=("a_stay", "mean"),
        runner_stay_rate=("b_stay", "mean"),
        mean_dist_ab=("dist_ab", "mean") if "dist_ab" in base.columns else ("chaser_closing", "mean"),
    )
    return out.sort_values("chaser_closing_rate", ascending=False)


def action_mix_by_role(step_df: pd.DataFrame) -> pd.DataFrame:
    """Action distribution by role and run."""
    if step_df.empty:
        return pd.DataFrame()
    base = ensure_behavior_features(step_df)
    keys = _group_keys(base)
    if not keys:
        return pd.DataFrame()

    parts = []
    if "a_action" in base.columns:
        a = base[keys + ["a_action"]].copy().rename(columns={"a_action": "action"})
        a["role"] = "CHASER"
        parts.append(a)
    if "b_action" in base.columns:
        b = base[keys + ["b_action"]].copy().rename(columns={"b_action": "action"})
        b["role"] = "RUNNER"
        parts.append(b)
    if not parts:
        return pd.DataFrame()

    tbl = pd.concat(parts, ignore_index=True)
    tbl["action"] = tbl["action"].astype(str)

    counts = tbl.groupby(keys + ["role", "action"], as_index=False).size().rename(columns={"size": "count"})
    totals = counts.groupby(keys + ["role"], as_index=False)["count"].sum().rename(columns={"count": "total"})
    out = counts.merge(totals, on=keys + ["role"], how="left")
    out["share"] = out["count"] / out["total"]
    return out


def plot_action_mix_by_role(mix_df: pd.DataFrame, title: str = "Action mix by role"):
    if mix_df.empty:
        print("No action-mix data.")
        return None
    base = mix_df.copy()
    if "p_fail" in base.columns:
        base["run_id"] = base["run"].astype(str) + " | p_fail=" + base["p_fail"].astype(str)
    else:
        base["run_id"] = base["run"].astype(str)

    fig = px.bar(
        base,
        x="run_id",
        y="share",
        color="action",
        facet_row="role",
        barmode="stack",
        title=title,
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(template="plotly_white", legend_title_text="Action", xaxis_title="Run")
    fig.show()
    return None


def chase_phase_dynamics(step_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Episode phase dynamics (requested item 2)."""
    if step_df.empty or "episode" not in step_df.columns or "step" not in step_df.columns:
        return pd.DataFrame()

    base = ensure_behavior_features(step_df)
    keys = _group_keys(base)
    ep_keys = keys + ["episode"]

    ep_len = base.groupby(ep_keys, as_index=False)["step"].max().rename(columns={"step": "ep_len"})
    base = base.merge(ep_len, on=ep_keys, how="left")
    base["phase"] = np.ceil((base["step"] / base["ep_len"]).clip(upper=1) * n_bins).fillna(1).astype(int)
    base["phase"] = base["phase"].clip(1, n_bins)

    base["chaser_closing"] = base.get("dist_ab_delta", np.nan) < 0
    base["runner_opening"] = base.get("dist_ab_delta", np.nan) > 0

    out = base.groupby(keys + ["phase"], as_index=False).agg(
        mean_dist_ab=("dist_ab", "mean") if "dist_ab" in base.columns else ("phase", "count"),
        mean_dist_ab_delta=("dist_ab_delta", "mean") if "dist_ab_delta" in base.columns else ("phase", "count"),
        chaser_closing_rate=("chaser_closing", "mean"),
        runner_opening_rate=("runner_opening", "mean"),
        steps=("step", "count"),
    )
    return out


def plot_chase_phase_dynamics(phase_df: pd.DataFrame, run: str):
    if phase_df.empty:
        print("No phase-dynamics data.")
        return None
    sub = phase_df.loc[phase_df["run"] == run].copy()
    if sub.empty:
        print(f"No phase-dynamics rows for run={run}")
        return None

    if "p_fail" in sub.columns:
        sub["series"] = "p_fail=" + sub["p_fail"].astype(str)
    else:
        sub["series"] = "all"

    melt = sub.melt(
        id_vars=["phase", "series"],
        value_vars=["chaser_closing_rate", "runner_opening_rate", "mean_dist_ab"],
        var_name="metric",
        value_name="value",
    )
    fig = px.line(
        melt,
        x="phase",
        y="value",
        color="series",
        facet_row="metric",
        markers=True,
        title=f"Chase phase dynamics - {run}",
    )
    fig.update_layout(template="plotly_white", legend_title_text=None)
    fig.show()
    return None


def treasure_capture_tradeoff(
    step_df: pd.DataFrame,
    terminal_df: Optional[pd.DataFrame] = None,
    near_threshold: int = 2,
):
    """Treasure vs capture behavior when runner is near treasure (requested item 4)."""
    if step_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = ensure_behavior_features(step_df)
    keys = _group_keys(base)
    if "episode" not in base.columns or "dist_b_treasure" not in base.columns:
        return pd.DataFrame(), pd.DataFrame()

    near = base["dist_b_treasure"].le(near_threshold)
    sub = base.loc[near].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub["runner_push_treasure"] = sub.get("dist_b_treasure_next", np.nan) < sub.get("dist_b_treasure", np.nan)
    sub["runner_open_when_near"] = sub.get("dist_ab_delta", np.nan) > 0
    sub["chaser_close_when_near"] = sub.get("dist_ab_delta", np.nan) < 0

    step_summary = sub.groupby(keys, as_index=False).agg(
        near_treasure_steps=("step", "count"),
        runner_push_treasure_rate=("runner_push_treasure", "mean"),
        runner_open_when_near_rate=("runner_open_when_near", "mean"),
        chaser_close_when_near_rate=("chaser_close_when_near", "mean"),
    )

    if terminal_df is None or terminal_df.empty:
        return step_summary, pd.DataFrame()

    term = terminal_df.copy()
    if "run" not in term.columns and "run_label" in term.columns:
        term["run"] = term["run_label"]
    merge_keys = [k for k in keys if k in term.columns]
    term_cols = merge_keys + ["episode", "terminal_reason"]
    term_cols = [c for c in term_cols if c in term.columns]
    term = term[term_cols].drop_duplicates()

    ep_near = sub[keys + ["episode"]].drop_duplicates()
    near_cols = merge_keys + ["episode"]
    ep_near = ep_near[near_cols]
    mix = ep_near.merge(term, on=near_cols, how="left")
    mix["terminal_reason"] = mix["terminal_reason"].fillna("unknown")
    group_keys = merge_keys if merge_keys else ["episode"]
    outcome_mix = mix.groupby(group_keys + ["terminal_reason"], as_index=False).size().rename(columns={"size": "episodes"})
    totals = outcome_mix.groupby(group_keys, as_index=False)["episodes"].sum().rename(columns={"episodes": "total"})
    outcome_mix = outcome_mix.merge(totals, on=group_keys, how="left")
    outcome_mix["rate"] = outcome_mix["episodes"] / outcome_mix["total"]
    return step_summary, outcome_mix


def episode_path_efficiency(step_df: pd.DataFrame, terminal_df: Optional[pd.DataFrame] = None):
    """Path efficiency metrics (requested item 6)."""
    if step_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = ensure_behavior_features(step_df)
    keys = _group_keys(base)
    if "episode" not in base.columns or "step" not in base.columns:
        return pd.DataFrame(), pd.DataFrame()

    ep_keys = keys + ["episode"]
    sort_cols = ep_keys + ["step"]
    base = base.sort_values(sort_cols)

    first = base.groupby(ep_keys, as_index=False).first()
    last = base.groupby(ep_keys, as_index=False).last()

    ep = first[ep_keys].copy()
    for src, dst in [
        ("dist_ab", "dist_ab_start"),
        ("dist_b_treasure", "dist_b_treasure_start"),
    ]:
        if src in first.columns:
            ep[dst] = first[src]
    for src, dst in [
        ("dist_ab", "dist_ab_end"),
        ("dist_b_treasure", "dist_b_treasure_end"),
        ("step", "steps"),
    ]:
        if src in last.columns:
            ep[dst] = last[src]

    if "dist_ab_start" in ep.columns and "dist_ab_end" in ep.columns:
        ep["chaser_progress"] = (ep["dist_ab_start"] - ep["dist_ab_end"]).clip(lower=0)
    if "dist_b_treasure_start" in ep.columns and "dist_b_treasure_end" in ep.columns:
        ep["runner_progress"] = (ep["dist_b_treasure_start"] - ep["dist_b_treasure_end"]).clip(lower=0)

    if "steps" in ep.columns:
        ep["chaser_path_efficiency"] = ep.get("chaser_progress", np.nan) / ep["steps"].replace(0, np.nan)
        ep["runner_path_efficiency"] = ep.get("runner_progress", np.nan) / ep["steps"].replace(0, np.nan)

    if terminal_df is not None and not terminal_df.empty:
        term = terminal_df.copy()
        if "run" not in term.columns and "run_label" in term.columns:
            term["run"] = term["run_label"]
        term_merge_keys = [k for k in ep_keys if k in term.columns]
        term_cols = term_merge_keys + [c for c in ["terminal_reason", "outcome"] if c in term.columns]
        term = term[term_cols].drop_duplicates()
        ep = ep.merge(term, on=term_merge_keys, how="left")

    summary = ep.groupby(keys, as_index=False).agg(
        episodes=("episode", "count"),
        mean_steps=("steps", "mean") if "steps" in ep.columns else ("episode", "count"),
        chaser_path_efficiency=("chaser_path_efficiency", "mean") if "chaser_path_efficiency" in ep.columns else ("episode", "count"),
        runner_path_efficiency=("runner_path_efficiency", "mean") if "runner_path_efficiency" in ep.columns else ("episode", "count"),
    )
    return ep, summary


def plot_control_zone_heatmaps(
    step_df: pd.DataFrame,
    run: str,
    p_fail: Optional[float] = None,
    pressure_dist: int = 2,
):
    """Spatial control view: runner positions under pressure vs free space."""
    if step_df.empty:
        print("No step data for control-zone heatmaps.")
        return None

    base = ensure_behavior_features(step_df)
    sub = base.loc[base["run"] == run].copy()
    if p_fail is not None and "p_fail" in sub.columns:
        sub = sub.loc[sub["p_fail"] == p_fail]
    if sub.empty or "dist_ab" not in sub.columns:
        print(f"No data for run={run}, p_fail={p_fail}")
        return None

    under = sub.loc[sub["dist_ab"] <= pressure_dist]
    free = sub.loc[sub["dist_ab"] > pressure_dist]

    def _grid(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        return df.pivot_table(index="state_bx", columns="state_by", values="step", aggfunc="count").fillna(0)

    g_under = _grid(under)
    g_free = _grid(free)

    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Runner under pressure (dist<= {pressure_dist})", "Runner outside pressure"]) 

    if not g_under.empty:
        fig.add_trace(go.Heatmap(z=g_under.values, x=g_under.columns, y=g_under.index, colorscale="Blues", showscale=False), row=1, col=1)
    if not g_free.empty:
        fig.add_trace(go.Heatmap(z=g_free.values, x=g_free.columns, y=g_free.index, colorscale="Blues", showscale=True, colorbar=dict(title="Count")), row=1, col=2)

    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(title=f"Control-zone heatmaps - {run}, p_fail={p_fail}", template="plotly_white", width=1000, height=450)
    fig.show()
    return None


def action_entropy_by_episode(step_df: pd.DataFrame) -> pd.DataFrame:
    """Policy stability metrics per episode (requested item 7)."""
    if step_df.empty:
        return pd.DataFrame()

    base = ensure_behavior_features(step_df)
    keys = _group_keys(base)
    if "episode" not in base.columns:
        return pd.DataFrame()

    out_parts = []
    for role, action_col in [("CHASER", "a_action"), ("RUNNER", "b_action")]:
        if action_col not in base.columns:
            continue
        tbl = base[keys + ["episode", action_col]].copy().rename(columns={action_col: "action"})
        tbl["action"] = tbl["action"].astype(str)
        counts = tbl.groupby(keys + ["episode", "action"], as_index=False).size().rename(columns={"size": "count"})
        if counts.empty:
            continue
        counts["total"] = counts.groupby(keys + ["episode"])["count"].transform("sum")
        counts["p"] = counts["count"] / counts["total"]

        agg = counts.groupby(keys + ["episode"], as_index=False).agg(
            entropy=("p", lambda s: float(-(s * np.log(s + 1e-12)).sum())),
            dominant_share=("p", "max"),
            n_actions_used=("action", "nunique"),
        )
        agg["entropy_norm"] = agg["entropy"] / np.log(5.0)
        agg["role"] = role
        out_parts.append(agg)

    if not out_parts:
        return pd.DataFrame()
    return pd.concat(out_parts, ignore_index=True)


def rolling_policy_stability(entropy_df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    if entropy_df.empty:
        return pd.DataFrame()
    base = entropy_df.copy().sort_values([c for c in ["run", "p_fail", "role", "episode"] if c in entropy_df.columns])
    keys = [c for c in ["run", "p_fail", "role"] if c in base.columns]

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("episode")
        g["entropy_roll"] = g["entropy_norm"].rolling(window, min_periods=5).mean()
        g["dominant_roll"] = g["dominant_share"].rolling(window, min_periods=5).mean()
        return g

    return base.groupby(keys, group_keys=False).apply(_roll)


def plot_policy_stability(entropy_roll_df: pd.DataFrame, run: str):
    if entropy_roll_df.empty:
        print("No entropy data.")
        return None
    sub = entropy_roll_df.loc[entropy_roll_df["run"] == run].copy()
    if sub.empty:
        print(f"No entropy rows for run={run}")
        return None

    if "p_fail" in sub.columns:
        sub["series"] = sub["role"] + " | p_fail=" + sub["p_fail"].astype(str)
    else:
        sub["series"] = sub["role"]

    melt = sub.melt(
        id_vars=["episode", "series"],
        value_vars=["entropy_roll", "dominant_roll"],
        var_name="metric",
        value_name="value",
    )
    fig = px.line(melt, x="episode", y="value", color="series", facet_row="metric", title=f"Policy stability - {run}")
    fig.update_layout(template="plotly_white", legend_title_text=None)
    fig.show()
    return None


def train_eval_behavior_gap(train_step_df: pd.DataFrame, eval_step_df: pd.DataFrame):
    """Train-vs-eval behavioral gap summary (requested item 8)."""
    if train_step_df.empty or eval_step_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    f_train = role_behavior_fingerprint(train_step_df)
    f_eval = role_behavior_fingerprint(eval_step_df)
    keys = [c for c in ["run", "p_fail"] if c in f_train.columns and c in f_eval.columns]
    if not keys:
        keys = ["run"]

    merged = f_eval.merge(f_train, on=keys, how="inner", suffixes=("_eval", "_train"))
    for c in ["chaser_closing_rate", "runner_opening_rate", "chaser_stay_rate", "runner_stay_rate", "mean_dist_ab"]:
        ce = f"{c}_eval"
        ct = f"{c}_train"
        if ce in merged.columns and ct in merged.columns:
            merged[f"{c}_gap"] = merged[ce] - merged[ct]

    e_train = action_entropy_by_episode(train_step_df)
    e_eval = action_entropy_by_episode(eval_step_df)
    if e_train.empty or e_eval.empty:
        return merged, pd.DataFrame()

    e_train_sum = e_train.groupby(keys + ["role"], as_index=False).agg(
        entropy_norm_train=("entropy_norm", "mean"),
        dominant_share_train=("dominant_share", "mean"),
    )
    e_eval_sum = e_eval.groupby(keys + ["role"], as_index=False).agg(
        entropy_norm_eval=("entropy_norm", "mean"),
        dominant_share_eval=("dominant_share", "mean"),
    )
    e_gap = e_eval_sum.merge(e_train_sum, on=keys + ["role"], how="inner")
    e_gap["entropy_gap"] = e_gap["entropy_norm_eval"] - e_gap["entropy_norm_train"]
    e_gap["dominance_gap"] = e_gap["dominant_share_eval"] - e_gap["dominant_share_train"]
    return merged, e_gap


def build_agent_episode_table(terminal_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per (agent, episode) from terminal rows."""
    if terminal_df.empty:
        return pd.DataFrame()

    base = terminal_df.copy()
    if "run" not in base.columns and "run_label" in base.columns:
        base["run"] = base["run_label"].astype(str)

    required = {"agent_a", "agent_b", "outcome"}
    if not required.issubset(base.columns):
        return pd.DataFrame()

    for col in ["step", "reward_a", "reward_b", "p_fail", "episode"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    common_cols = [c for c in ["run", "p_fail", "episode", "outcome", "terminal_reason", "capture"] if c in base.columns]

    a_cols = common_cols + [c for c in ["agent_a", "agent_b", "role_a", "step", "reward_a"] if c in base.columns]
    a = base[a_cols].copy()
    a["agent"] = a.get("agent_a")
    a["opponent"] = a.get("agent_b")
    a["role"] = a.get("role_a", "CHASER")
    a["steps"] = a.get("step")
    a["episode_reward"] = a.get("reward_a")
    a["is_win"] = a.get("outcome", pd.Series(index=a.index, dtype=object)).eq("A_WIN")

    b_cols = common_cols + [c for c in ["agent_a", "agent_b", "role_b", "step", "reward_b"] if c in base.columns]
    b = base[b_cols].copy()
    b["agent"] = b.get("agent_b")
    b["opponent"] = b.get("agent_a")
    b["role"] = b.get("role_b", "RUNNER")
    b["steps"] = b.get("step")
    b["episode_reward"] = b.get("reward_b")
    b["is_win"] = b.get("outcome", pd.Series(index=b.index, dtype=object)).eq("B_WIN")

    out = pd.concat([a, b], ignore_index=True, sort=False)
    out["agent"] = out["agent"].astype(str)
    out["opponent"] = out["opponent"].astype(str)
    out["role"] = out["role"].astype(str).str.upper()
    out["role"] = out["role"].replace({"A": "CHASER", "B": "RUNNER"})

    term_reason = out.get("terminal_reason", pd.Series(index=out.index, dtype=object))
    out["is_capture_episode"] = term_reason.eq("capture")
    out["is_timeout_episode"] = term_reason.eq("timeout")
    out["is_treasure_win"] = (
        (out["role"].eq("CHASER") & term_reason.eq("chaser_treasure") & out["is_win"])
        | (out["role"].eq("RUNNER") & term_reason.eq("runner_treasure") & out["is_win"])
    )
    out["is_hazard_loss"] = (
        (out["role"].eq("CHASER") & term_reason.isin(["chaser_pit", "chaser_wumpus", "chaser_hazard"]) & (~out["is_win"]))
        | (out["role"].eq("RUNNER") & term_reason.isin(["runner_pit", "runner_wumpus", "runner_hazard"]) & (~out["is_win"]))
    )
    return out


def summarize_agent_outcomes(
    agent_episode_df: pd.DataFrame,
    group_cols: Sequence[str] = ("agent", "role"),
) -> pd.DataFrame:
    """Aggregate episode-level outcomes to comparable agent summaries."""
    if agent_episode_df.empty:
        return pd.DataFrame()

    base = agent_episode_df.copy()
    keys = [c for c in group_cols if c in base.columns]
    if not keys:
        return pd.DataFrame()

    out = base.groupby(keys, as_index=False).agg(
        episodes=("is_win", "count"),
        win_rate=("is_win", "mean"),
        avg_steps=("steps", "mean") if "steps" in base.columns else ("is_win", "count"),
        avg_reward=("episode_reward", "mean") if "episode_reward" in base.columns else ("is_win", "count"),
        capture_episode_rate=("is_capture_episode", "mean"),
        timeout_episode_rate=("is_timeout_episode", "mean"),
        treasure_win_rate=("is_treasure_win", "mean"),
        hazard_loss_rate=("is_hazard_loss", "mean"),
    )
    return out.sort_values(["win_rate", "avg_reward"], ascending=False)


def agent_terminal_reason_mix(
    agent_episode_df: pd.DataFrame,
    group_cols: Sequence[str] = ("agent", "role"),
) -> pd.DataFrame:
    """Rate of terminal reasons per agent."""
    if agent_episode_df.empty or "terminal_reason" not in agent_episode_df.columns:
        return pd.DataFrame()

    base = agent_episode_df.copy()
    keys = [c for c in group_cols if c in base.columns]
    if not keys:
        return pd.DataFrame()

    counts = base.groupby(keys + ["terminal_reason"], as_index=False).size().rename(columns={"size": "episodes"})
    totals = counts.groupby(keys, as_index=False)["episodes"].sum().rename(columns={"episodes": "total_episodes"})
    out = counts.merge(totals, on=keys, how="left")
    out["rate"] = out["episodes"] / out["total_episodes"]
    return out.sort_values(keys + ["rate"], ascending=[True] * len(keys) + [False])


def plot_agent_summary_metric(
    summary_df: pd.DataFrame,
    metric: str = "win_rate",
    title: Optional[str] = None,
):
    if summary_df.empty:
        print("No agent summary data.")
        return None
    if metric not in summary_df.columns:
        print(f"Metric '{metric}' not found.")
        return None

    base = summary_df.copy()
    if "role" in base.columns:
        base["agent_id"] = base["agent"].astype(str) + " | " + base["role"].astype(str)
        color = "role"
    else:
        base["agent_id"] = base["agent"].astype(str)
        color = None

    fig = px.bar(
        base.sort_values(metric, ascending=False),
        x="agent_id",
        y=metric,
        color=color,
        text=base[metric].map(lambda v: f"{v:.3f}" if pd.notna(v) else ""),
        title=title or f"Agent comparison: {metric}",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Agent")
    fig.show()
    return None


def plot_agent_run_distributions(
    agent_episode_df: pd.DataFrame,
    metric: str = "episode_reward",
    title: Optional[str] = None,
):
    """Distribution across runs to keep per-run variability visible."""
    if agent_episode_df.empty or metric not in agent_episode_df.columns:
        print("No per-run agent data for requested metric.")
        return None
    if "run" not in agent_episode_df.columns:
        print("Run column not found.")
        return None

    base = agent_episode_df.copy()
    grp = ["agent", "run"] + ([c for c in ["role", "p_fail"] if c in base.columns])
    run_stats = base.groupby(grp, as_index=False)[metric].mean()

    fig = px.box(
        run_stats,
        x="agent",
        y=metric,
        color="role" if "role" in run_stats.columns else None,
        points="all",
        title=title or f"Per-run distribution of {metric} by agent",
    )
    fig.update_layout(template="plotly_white")
    fig.show()
    return None


def plot_agent_terminal_reason_mix(reason_mix_df: pd.DataFrame, title: str = "Terminal reason mix by agent"):
    if reason_mix_df.empty:
        print("No agent terminal-reason data.")
        return None

    base = reason_mix_df.copy()
    if "role" in base.columns:
        base["agent_id"] = base["agent"].astype(str) + " | " + base["role"].astype(str)
    else:
        base["agent_id"] = base["agent"].astype(str)

    fig = px.bar(
        base,
        x="agent_id",
        y="rate",
        color="terminal_reason",
        title=title,
        barmode="stack",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(template="plotly_white", xaxis_title="Agent")
    fig.show()
    return None


def summarize_agent_distance(step_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate distance/progress signals by agent and role across all runs."""
    if step_df.empty:
        return pd.DataFrame()

    base = ensure_behavior_features(step_df)
    needed = {"agent_a", "agent_b"}
    if not needed.issubset(base.columns):
        return pd.DataFrame()

    for col in ["reward_a", "reward_b", "p_fail"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    common = [c for c in ["run", "p_fail", "dist_ab", "dist_ab_delta", "dist_b_treasure", "dist_b_treasure_next"] if c in base.columns]

    a = base[common + ["agent_a"] + ([c for c in ["reward_a"] if c in base.columns])].copy()
    a["agent"] = a["agent_a"].astype(str)
    a["role"] = "CHASER"
    a["step_reward"] = a.get("reward_a")
    a["favorable_progress"] = a.get("dist_ab_delta", np.nan) < 0
    a["goal_distance"] = a.get("dist_ab", np.nan)

    b = base[common + ["agent_b"] + ([c for c in ["reward_b"] if c in base.columns])].copy()
    b["agent"] = b["agent_b"].astype(str)
    b["role"] = "RUNNER"
    b["step_reward"] = b.get("reward_b")
    b["favorable_progress"] = b.get("dist_ab_delta", np.nan) > 0
    b["goal_distance"] = b.get("dist_b_treasure", np.nan)
    if "dist_b_treasure_next" in b.columns and "dist_b_treasure" in b.columns:
        b["favorable_progress"] = b["dist_b_treasure_next"] < b["dist_b_treasure"]

    long = pd.concat([a, b], ignore_index=True, sort=False)
    keys = ["agent", "role"] + [c for c in ["p_fail"] if c in long.columns]
    out = long.groupby(keys, as_index=False).agg(
        sampled_steps=("agent", "count"),
        mean_step_reward=("step_reward", "mean"),
        mean_dist_ab=("dist_ab", "mean") if "dist_ab" in long.columns else ("agent", "count"),
        mean_goal_distance=("goal_distance", "mean"),
        favorable_progress_rate=("favorable_progress", "mean"),
    )
    return out.sort_values(["role", "favorable_progress_rate"], ascending=[True, False])


def plot_agent_distance_summary(
    distance_df: pd.DataFrame,
    metric: str = "favorable_progress_rate",
    title: Optional[str] = None,
):
    if distance_df.empty:
        print("No distance summary data.")
        return None
    if metric not in distance_df.columns:
        print(f"Metric '{metric}' not found in distance summary.")
        return None

    fig = px.bar(
        distance_df.sort_values(["role", metric], ascending=[True, False]),
        x="agent",
        y=metric,
        color="role",
        barmode="group",
        title=title or f"Distance/progress summary by agent ({metric})",
    )
    fig.update_layout(template="plotly_white")
    fig.show()
    return None


__all__ = [
    "ensure_behavior_features",
    "role_behavior_fingerprint",
    "action_mix_by_role",
    "plot_action_mix_by_role",
    "chase_phase_dynamics",
    "plot_chase_phase_dynamics",
    "treasure_capture_tradeoff",
    "episode_path_efficiency",
    "plot_control_zone_heatmaps",
    "action_entropy_by_episode",
    "rolling_policy_stability",
    "plot_policy_stability",
    "train_eval_behavior_gap",
    "build_agent_episode_table",
    "summarize_agent_outcomes",
    "agent_terminal_reason_mix",
    "plot_agent_summary_metric",
    "plot_agent_run_distributions",
    "plot_agent_terminal_reason_mix",
    "summarize_agent_distance",
    "plot_agent_distance_summary",
]
