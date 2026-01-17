"""
Training loop: FP catcher vs FP runner.
"""

from typing import Dict, List

from agents import FPAgent
from env import ACTIONS, TagEnv
from logger import ExcelLogger
from train.common import evaluate
from utils import encode_state


def train_fp_vs_fp(p_fail: float,
                    episodes: int = 200,
                    eval_every: int = 100,
                    seed: int = 0,
                    logger: ExcelLogger | None = None,
                    eval_logger: ExcelLogger | None = None,
                    run_label: str = "fp_vs_fp") -> Dict[str, List[float]]:
    """Train FP agents against each other for a small number of episodes."""
    env = TagEnv(p_fail=p_fail, seed=seed)
    fp_c = FPAgent("catcher", p_fail=p_fail, seed=seed + 1)
    fp_r = FPAgent("runner", p_fail=p_fail, seed=seed + 2)
    logs = {"episode": [], "capture_rate": [], "avg_steps": [], "avg_return": []}
    # Define evaluation policies capturing current behavior.
    def catcher_policy(s):
        return fp_c.act(s)
    def runner_policy(s):
        return fp_r.act(s)
    for ep in range(1, episodes + 1):
        s = env.reset()
        # Play one episode for training.
        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            opp_pi_c = fp_c.counts_opp[sid] / (fp_c.counts_opp[sid].sum() + 1e-12)
            opp_pi_r = fp_r.counts_opp[sid] / (fp_r.counts_opp[sid].sum() + 1e-12)
            a1 = fp_c.act(s)
            a2 = fp_r.act(s)
            # Update opponent models.
            fp_c.observe(s, a2)
            fp_r.observe(s, a1)
            s2, r, done, info = env.step(a1, a2)
            if logger:
                sid2 = encode_state(s2, env.size)
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
                    epsilon=None,
                    value_estimate=None,
                    extra={
                        "catcher_agent": "FP",
                        "runner_agent": "FP",
                        "opp_pi_catcher": opp_pi_c.tolist(),
                        "opp_pi_runner": opp_pi_r.tolist(),
                    },
                    actions=ACTIONS,
                )
            s = s2
            if done:
                break
        if ep % eval_every == 0:
            eval_env = TagEnv(p_fail=p_fail, seed=seed + 10_000 + ep)
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
