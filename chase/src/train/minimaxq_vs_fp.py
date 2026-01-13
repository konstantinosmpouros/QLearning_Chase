"""
Training loop: Minimax-Q catcher vs FP runner.
"""

from typing import Dict, List

import numpy as np

from agents import FPAgent, MinimaxQAgent
from env import ACTIONS, TagEnv
from logger import ExcelLogger
from train.common import evaluate
from utils import encode_state


def train_minimaxq_vs_fp(p_fail: float,
                            episodes: int = 200,
                            eval_every: int = 100,
                            seed: int = 0,
                            logger: ExcelLogger | None = None,
                            run_label: str = "minimaxq_vs_fp") -> Dict[str, List[float]]:
    """Train MinimaxQ catcher vs FP runner for a small number of episodes."""
    env = TagEnv(p_fail=p_fail, seed=seed)
    q = MinimaxQAgent(seed=seed + 1)
    fp_r = FPAgent("runner", p_fail=p_fail, seed=seed + 2)
    logs = {"episode": [], "capture_rate": [], "avg_steps": [], "avg_return": []}
    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = q.epsilon(ep)
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a1 = q.act_catcher(sid, eps)
            pi, v_cur = q.policies(sid)
            # Runner plays FP.
            a2 = fp_r.act(s)
            # Runner observes catcher action for its empirical model.
            fp_r.observe(s, a1)
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
                        "runner_agent": "FP",
                        "pi_catcher": pi.tolist(),
                    },
                    actions=ACTIONS,
                )
            s = s2
            if done:
                break
        if ep % eval_every == 0:
            eval_env = TagEnv(p_fail=p_fail, seed=seed + 30_000 + ep)
            def catcher_policy(s_):
                sid_ = encode_state(s_, eval_env.size)
                pi_, _ = q.policies(sid_)
                return int(np.random.choice(np.arange(len(ACTIONS)), p=pi_))
            def runner_policy(s_):
                return fp_r.act(s_)
            stats = evaluate(eval_env, catcher_policy, runner_policy, n_episodes=10)
            logs["episode"].append(ep)
            logs["capture_rate"].append(stats.capture_rate)
            logs["avg_steps"].append(stats.avg_steps_to_capture)
            logs["avg_return"].append(stats.avg_return)
    return logs
