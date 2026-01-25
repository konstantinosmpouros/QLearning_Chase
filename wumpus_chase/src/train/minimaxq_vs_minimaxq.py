"""Training loop: Minimax-Q vs Minimax-Q (two independent agents)."""

from typing import Dict, List

import numpy as np

from agents import MinimaxQAgent
from env import ACTIONS, WumpusChaseEnv
from logger import CSVLogger
from train.common import evaluate
from utils import encode_state


def train_minimaxq_vs_minimaxq(
    env: WumpusChaseEnv,
    episodes: int = 500,
    eval_every: int = 100,
    seed: int = 0,
    logger: CSVLogger | None = None,
    eval_logger: CSVLogger | None = None,
    run_label: str = "minimaxq_vs_minimaxq",
) -> Dict[str, List[float]]:
    q_row = MinimaxQAgent(size=env.size, seed=seed + 1)
    q_col = MinimaxQAgent(size=env.size, seed=seed + 2)
    logs = {"episode": [], "win_rate": [], "draw_rate": [], "avg_steps": [], "avg_return": []}

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_row = q_row.epsilon(ep)
        eps_col = q_col.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_row = q_row.act_row(sid, eps_row)
            a_col = q_col.act_row(sid, eps_col)
            s2, r, done, info = env.step(a_row, a_col)
            sid2 = encode_state(s2, env.size)
            q_row.update(sid, a_row, a_col, r, sid2, done)
            q_col.update(sid, a_col, a_row, -r, sid2, done)
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
            eval_env = WumpusChaseEnv(layout=env.layout, p_fail=env.p_fail, seed=seed + 4000 + ep)

            def row_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q_row.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            def col_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q_col.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            stats = evaluate(
                eval_env,
                row_policy,
                col_policy,
                n_episodes=10,
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
    return logs
