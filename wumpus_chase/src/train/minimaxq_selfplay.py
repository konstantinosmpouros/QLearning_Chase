"""Training loop: Minimax-Q self-play."""

from typing import Dict, List

import numpy as np

from agents import MinimaxQAgent
from env import ACTIONS, WumpusChaseEnv
from logger import CSVLogger
from train.common import evaluate
from utils import encode_state


def train_minimaxq_selfplay(
    env: WumpusChaseEnv,
    episodes: int = 500,
    eval_every: int = 100,
    seed: int = 0,
    logger: CSVLogger | None = None,
    eval_logger: CSVLogger | None = None,
    run_label: str = "minimaxq_selfplay",
) -> Dict[str, List[float]]:
    q = MinimaxQAgent(size=env.size, seed=seed + 1)
    logs = {"episode": [], "win_rate": [], "draw_rate": [], "avg_steps": [], "avg_return": []}

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = q.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_row = q.act_row(sid, eps)
            pi, _v_cur = q.policies(sid)
            a_col = q.best_response_col(sid, pi, eps)
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
                    agent_b="MinimaxQ",
                    sid=sid,
                    sid_next=sid2,
                )
            s = s2
            if done:
                break

        if ep % eval_every == 0:
            eval_env = WumpusChaseEnv(layout=env.layout, p_fail=env.p_fail, seed=seed + 1000 + ep)

            def row_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            def col_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q.policies(sid_)
                return q.best_response_col(sid_, pi_, 0.0)

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
