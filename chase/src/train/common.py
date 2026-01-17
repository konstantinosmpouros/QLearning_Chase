"""
Shared training helpers for the chase project.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from env import ACTIONS, TagEnv
from logger import ExcelLogger
from utils import encode_state


@dataclass
class EvalStats:
    capture_rate: float
    avg_steps_to_capture: float
    avg_return: float


def run_episode(
    env: TagEnv,
    catcher_policy,
    runner_policy,
    *,
    logger: ExcelLogger | None = None,
    run_label: str | None = None,
    p_fail: float | None = None,
    episode_idx: int = 0,
    actions=ACTIONS,
    extra_info: dict | None = None,
) -> Tuple[bool, int, float]:
    """
    Simulate a single episode under given policies.
    Optionally logs each step to ExcelLogger (step-level, like training).
    Returns (captured_flag, steps_taken, cumulative_reward_for_catcher).
    """
    s = env.reset()
    total_r = 0.0
    captured = False
    for _ in range(env.t_max):
        a1 = catcher_policy(s)
        a2 = runner_policy(s)
        sid = encode_state(s, env.size)
        s2, r, done, info = env.step(a1, a2)
        sid2 = encode_state(s2, env.size)
        if logger and run_label is not None:
            extra = {"phase": "eval"}
            if extra_info:
                extra.update(extra_info)
            logger.log_step(
                run_label=run_label,
                env=env,
                p_fail=p_fail if p_fail is not None else getattr(env, "p_fail", None),
                episode=episode_idx,
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
                extra=extra,
                actions=actions,
            )
        total_r += r
        if info.get("capture", False):
            captured = True
        if done:
            break
        s = s2
    return captured, env.t, total_r


def evaluate(
    env: TagEnv,
    catcher_policy,
    runner_policy,
    n_episodes: int = 200,
    *,
    logger: ExcelLogger | None = None,
    run_label: str | None = None,
    p_fail: float | None = None,
    eval_at: int | None = None,
) -> EvalStats:
    """
    Evaluate given policies for a small number of episodes.
    Optionally logs each evaluation step (step-level) for later analysis.
    """
    caps = 0
    steps_caps: List[int] = []
    returns: List[float] = []
    for ep_idx in range(1, n_episodes + 1):
        captured, steps, ret = run_episode(
            env,
            catcher_policy,
            runner_policy,
            logger=logger,
            run_label=run_label,
            p_fail=p_fail if p_fail is not None else getattr(env, "p_fail", None),
            episode_idx=ep_idx,
            extra_info={"eval_at_episode": eval_at} if eval_at is not None else None,
        )
        returns.append(ret)
        if captured:
            caps += 1
            steps_caps.append(steps)
    capture_rate = caps / n_episodes
    avg_steps_to_capture = float(np.mean(steps_caps)) if steps_caps else float("nan")
    avg_return = float(np.mean(returns)) if returns else float("nan")
    return EvalStats(capture_rate, avg_steps_to_capture, avg_return)


def last_or_nan(lst: List[float]) -> float:
    return lst[-1] if lst else float("nan")
