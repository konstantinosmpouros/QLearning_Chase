"""
Training loop: Minimax-Q self-play (both roles trained).
"""

from typing import Dict, List

import numpy as np

from agents import MinimaxQAgent
from env import ACTIONS, TagEnv
from logger import ExcelLogger
from train.common import evaluate
from utils import encode_state


def train_minimaxq_selfplay(p_fail: float,
                            episodes: int = 200,
                            eval_every: int = 100,
                            seed: int = 0,
                            logger: ExcelLogger | None = None,
                            eval_logger: ExcelLogger | None = None,
                            run_label: str = "minimaxq_selfplay") -> Dict[str, List[float]]:
    """Train MinimaxQ agent in self‑play (both players are RL) for a small number of episodes."""
    env = TagEnv(p_fail=p_fail, seed=seed)
    q = MinimaxQAgent(seed=seed + 1)
    logs = {"episode": [], "capture_rate": [], "avg_steps": [], "avg_return": []}
    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = q.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            # Catcher chooses using epsilon‑greedy.
            a1 = q.act_catcher(sid, eps)
            # Runner best responds to pi.
            pi, v_cur = q.policies(sid)
            a2 = q.best_response_runner(sid, pi, eps)
            s2, r, done, info = env.step(a1, a2)
            sid2 = encode_state(s2, env.size)
            q.update(sid, a1, a2, r, sid2, done)
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
                    epsilon=eps,
                    value_estimate=v_cur,
                    extra={
                        "catcher_agent": "MinimaxQ",
                        "runner_agent": "MinimaxQ",
                        "pi_catcher": pi.tolist(),
                    },
                    actions=ACTIONS,
                )
            s = s2
            if done:
                break
        if ep % eval_every == 0:
            eval_env = TagEnv(p_fail=p_fail, seed=seed + 20_000 + ep)
            def catcher_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))
            def runner_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q.policies(sid_)
                return q.best_response_runner(sid_, pi_, 0.0)
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
