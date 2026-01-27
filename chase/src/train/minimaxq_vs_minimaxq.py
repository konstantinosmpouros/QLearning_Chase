"""
Training loop: Minimax-Q catcher vs Minimax-Q runner (two agents).
"""

from typing import Dict, List

import numpy as np

from agents import MinimaxQAgent
from env import ACTIONS, TagEnv
from logger import ExcelLogger
from train.common import evaluate
from utils import encode_state


def train_minimaxq_vs_minimaxq(
    p_fail: float,
    episodes: int = 200,
    eval_every: int = 100,
    seed: int = 0,
    logger: ExcelLogger | None = None,
    eval_logger: ExcelLogger | None = None,
    run_label: str = "minimaxq_vs_minimaxq",
) -> Dict[str, List[float]]:
    """Train two MinimaxQ agents (catcher vs runner) for a small number of episodes."""
    env = TagEnv(p_fail=p_fail, seed=seed)
    q_c = MinimaxQAgent(seed=seed + 1)
    q_r = MinimaxQAgent(seed=seed + 2)
    logs = {"episode": [], "capture_rate": [], "avg_steps": [], "avg_return": []}
    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_c = q_c.epsilon(ep)
        eps_r = q_r.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a1 = q_c.act_catcher(sid, eps_c)
            a2 = q_r.act_catcher(sid, eps_r)
            pi_c, v_c = q_c.policies(sid)
            pi_r, v_r = q_r.policies(sid)
            s2, r, done, info = env.step(a1, a2)
            sid2 = encode_state(s2, env.size)
            q_c.update(sid, a1, a2, r, sid2, done)
            r_r = info.get("reward_runner", -r)
            q_r.update(sid, a2, a1, r_r, sid2, done)
            if logger:
                logger.log_step(
                    run_label=run_label,
                    env=env,
                    p_fail=p_fail,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_c=a1,
                    action_r=a2,
                    next_state=s2,
                    reward=r,
                    done=done,
                    info=info,
                    sid=sid,
                    sid_next=sid2,
                    epsilon=eps_c,
                    value_estimate=v_c,
                    extra={
                        "catcher_agent": "MinimaxQ",
                        "runner_agent": "MinimaxQ",
                        "reward_runner": r_r,
                        "pi_catcher": pi_c.tolist(),
                        "pi_runner": pi_r.tolist(),
                        "epsilon_runner": eps_r,
                        "value_runner": v_r,
                    },
                    actions=ACTIONS,
                )
            s = s2
            if done:
                break
        if ep % eval_every == 0:
            eval_env = TagEnv(p_fail=p_fail, seed=seed + 40_000 + ep)

            def catcher_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q_c.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            def runner_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q_r.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))

            stats = evaluate(
                eval_env,
                catcher_policy,
                runner_policy,
                n_episodes=10,
                logger=eval_logger,
                run_label=run_label,
                p_fail=p_fail,
                eval_at=ep,
            )
            logs["episode"].append(ep)
            logs["capture_rate"].append(stats.capture_rate)
            logs["avg_steps"].append(stats.avg_steps_to_capture)
            logs["avg_return"].append(stats.avg_return)
    return logs
