"""Training loop: FP vs FP."""

from typing import Dict, List

from agents import FPAgent
from env import ACTIONS, WumpusChaseEnv
from logger import CSVLogger
from train.common import evaluate
from utils import encode_state


def train_fp_vs_fp(
    env: WumpusChaseEnv,
    episodes: int = 500,
    eval_every: int = 100,
    seed: int = 0,
    logger: CSVLogger | None = None,
    eval_logger: CSVLogger | None = None,
    run_label: str = "fp_vs_fp",
) -> Dict[str, List[float]]:
    fp_row = FPAgent("row", env.layout, p_fail=env.p_fail, seed=seed + 1)
    fp_col = FPAgent("col", env.layout, p_fail=env.p_fail, seed=seed + 2)
    logs = {"episode": [], "win_rate": [], "draw_rate": [], "avg_steps": [], "avg_return": []}

    for ep in range(1, episodes + 1):
        s = env.reset()
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_row = fp_row.act(s)
            a_col = fp_col.act(s)
            fp_row.observe(s, a_col)
            fp_col.observe(s, a_row)
            s2, r, done, info = env.step(a_row, a_col)
            sid2 = encode_state(s2, env.size)
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
                    agent_a="FP",
                    agent_b="FP",
                    sid=sid,
                    sid_next=sid2,
                )
            s = s2
            if done:
                break

        if ep % eval_every == 0:
            eval_env = WumpusChaseEnv(layout=env.layout, p_fail=env.p_fail, seed=seed + 2000 + ep)

            def row_policy(s_):
                return fp_row.act(s_)

            def col_policy(s_):
                return fp_col.act(s_)

            stats = evaluate(
                eval_env,
                row_policy,
                col_policy,
                n_episodes=10,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a="FP",
                agent_b="FP",
            )
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
    return logs
