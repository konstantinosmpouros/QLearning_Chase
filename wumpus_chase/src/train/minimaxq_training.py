"""Training loops where Minimax-Q is the primary learner."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from agents import DynaQAgent, FPAgent, MinimaxQAgent, NashQAgent
from env import ACTIONS, WumpusChaseEnvExtended
from train.common import evaluate
from utils import encode_state


def _init_logs() -> Dict[str, List[float]]:
    return {"episode": [], "win_rate": [], "draw_rate": [], "avg_steps": [], "avg_return": []}


def train_minimaxq_vs_minimaxq(
    env: WumpusChaseEnvExtended,
    episodes: int = 500,
    eval_every: int = 1000,
    eval_episodes: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "minimaxq_vs_minimaxq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train two independent Minimax-Q agents."""
    q_row = MinimaxQAgent(size=env.size, seed=seed + 1)
    q_col = MinimaxQAgent(size=env.size, seed=seed + 2)
    logs = _init_logs()

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_row = q_row.epsilon(ep)
        eps_col = q_col.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_row = q_row.act_row(sid, eps_row)
            a_col = q_col.act_col(sid, eps_col)
            s2, r, done, info = env.step(a_row, a_col)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r))
            q_row.update(sid, a_row, a_col, r, sid2, done)
            q_col.update(sid, a_col, a_row, r_b, sid2, done)

            if logger:
                logger.log_step(
                    run_label=run_label,
                    phase="train",
                    env=env,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_a=a_row,
                    action_b=a_col,
                    next_state=s2,
                    reward=r,
                    done=done,
                    info=info,
                    actions=ACTIONS,
                    agent_a="MinimaxQ",
                    agent_b="MinimaxQ",
                    sid=sid,
                    sid_next=sid2,
                )
            s = s2
            if done:
                break

        if ep % eval_every == 0:
            def row_policy(s_):
                sid_ = encode_state(s_, env.size)
                pi_, _ = q_row.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            def col_policy(s_):
                sid_ = encode_state(s_, env.size)
                return q_col.act_col(sid_, eps=0.0)

            stats = evaluate(
                env,
                row_policy,
                col_policy,
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a="MinimaxQ",
                agent_b="MinimaxQ",
            )
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}, "
                    f"Steps={stats.avg_steps:.1f}, "
                    f"Return={stats.avg_return:.2f}"
                )
    return logs


def train_minimaxq_selfplay(
    env: WumpusChaseEnvExtended,
    episodes: int = 500,
    eval_every: int = 1000,
    eval_episodes: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "minimaxq_selfplay",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Minimax-Q self-play alias."""
    return train_minimaxq_vs_minimaxq(
        env=env,
        episodes=episodes,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=seed,
        logger=logger,
        eval_logger=eval_logger,
        run_label=run_label,
        verbose=verbose,
    )


def train_minimaxq_vs_fp(
    env: WumpusChaseEnvExtended,
    episodes: int = 500,
    eval_every: int = 1000,
    eval_episodes: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "minimaxq_vs_fp",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train Minimax-Q (A/row) vs FP (B/col)."""
    q = MinimaxQAgent(size=env.size, seed=seed + 1)
    fp_col = FPAgent(role="col", layout=env.layout, p_fail=env.p_fail, seed=seed + 2)
    logs = _init_logs()

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = q.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_row = q.act_row(sid, eps)
            a_col = fp_col.act(s)
            fp_col.observe(s, a_row)
            s2, r, done, info = env.step(a_row, a_col)
            sid2 = encode_state(s2, env.size)
            q.update(sid, a_row, a_col, r, sid2, done)
            if logger:
                logger.log_step(
                    run_label=run_label,
                    phase="train",
                    env=env,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_a=a_row,
                    action_b=a_col,
                    next_state=s2,
                    reward=r,
                    done=done,
                    info=info,
                    actions=ACTIONS,
                    agent_a="MinimaxQ",
                    agent_b="FP",
                    sid=sid,
                    sid_next=sid2,
                )
            s = s2
            if done:
                break

        if ep % eval_every == 0:
            def row_policy(s_):
                sid_ = encode_state(s_, env.size)
                pi_, _ = q.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            stats = evaluate(
                env,
                row_policy,
                lambda s_: fp_col.act(s_),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a="MinimaxQ",
                agent_b="FP",
            )
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}, "
                    f"Steps={stats.avg_steps:.1f}, "
                    f"Return={stats.avg_return:.2f}"
                )
    return logs


def train_minimaxq_vs_nashq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "minimaxq_vs_nashq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train Minimax-Q (A/row) vs Nash-Q (B/col)."""
    mm_row = MinimaxQAgent(size=env.size, seed=seed + 1)
    nash_col = NashQAgent(size=env.size, seed=seed + 2)
    logs = _init_logs()

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_mm = mm_row.epsilon(ep)
        eps_nash = nash_col.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_row = mm_row.act_row(sid, eps_mm)
            a_col = nash_col.act_b(sid, eps_nash)
            s2, r_a, done, info = env.step(a_row, a_col)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))

            mm_row.update(sid, a_row, a_col, r_a, sid2, done)
            nash_col.update(sid, a_row, a_col, r_a, r_b, sid2, done)

            if logger:
                logger.log_step(
                    run_label=run_label,
                    phase="train",
                    env=env,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_a=a_row,
                    action_b=a_col,
                    next_state=s2,
                    reward=r_a,
                    done=done,
                    info=info,
                    actions=ACTIONS,
                    agent_a="MinimaxQ",
                    agent_b="NashQ",
                    sid=sid,
                    sid_next=sid2,
                )
            s = s2
            if done:
                break

        if ep % eval_every == 0:
            def row_policy(s_):
                sid_ = encode_state(s_, env.size)
                pi_, _ = mm_row.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            stats = evaluate(
                env,
                row_policy,
                lambda s_: nash_col.act_b(encode_state(s_, env.size), eps=0.0),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a="MinimaxQ",
                agent_b="NashQ",
            )
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}, "
                    f"Steps={stats.avg_steps:.1f}, "
                    f"Return={stats.avg_return:.2f}"
                )
    return logs


def train_minimaxq_vs_dynaq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "minimaxq_vs_dynaq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train Minimax-Q (A/row) vs Dyna-Q (B/col)."""
    mm_row = MinimaxQAgent(size=env.size, seed=seed + 1)
    dynaq_col = DynaQAgent(size=env.size, n_planning=n_planning, seed=seed + 2, player="B")
    logs = _init_logs()

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_mm = mm_row.epsilon(ep)
        eps_dyna = dynaq_col.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_row = mm_row.act_row(sid, eps_mm)
            a_col = dynaq_col.act_b(sid, eps_dyna)
            s2, r_a, done, info = env.step(a_row, a_col)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))

            mm_row.update(sid, a_row, a_col, r_a, sid2, done)
            dynaq_col.update(sid, a_row, a_col, r_b, sid2, done)

            if logger:
                logger.log_step(
                    run_label=run_label,
                    phase="train",
                    env=env,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_a=a_row,
                    action_b=a_col,
                    next_state=s2,
                    reward=r_a,
                    done=done,
                    info=info,
                    actions=ACTIONS,
                    agent_a="MinimaxQ",
                    agent_b="DynaQ",
                    sid=sid,
                    sid_next=sid2,
                )
            s = s2
            if done:
                break

        if ep % eval_every == 0:
            def row_policy(s_):
                sid_ = encode_state(s_, env.size)
                pi_, _ = mm_row.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            stats = evaluate(
                env,
                row_policy,
                lambda s_: dynaq_col.act_b(encode_state(s_, env.size), eps=0.0),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a="MinimaxQ",
                agent_b="DynaQ",
            )
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}, "
                    f"Steps={stats.avg_steps:.1f}, "
                    f"Return={stats.avg_return:.2f}"
                )
    return logs
