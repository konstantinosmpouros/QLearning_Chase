"""
Shared training helpers for the chase project.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from env import TagEnv


@dataclass
class EvalStats:
    capture_rate: float
    avg_steps_to_capture: float
    avg_return: float


def run_episode(env: TagEnv, catcher_policy, runner_policy) -> Tuple[bool, int, float]:
    """
    Simulate a single episode under given policies.
    Returns (captured_flag, steps_taken, cumulative_reward_for_catcher).
    """
    s = env.reset()
    total_r = 0.0
    captured = False
    for _ in range(env.t_max):
        a1 = catcher_policy(s)
        a2 = runner_policy(s)
        s, r, done, info = env.step(a1, a2)
        total_r += r
        if info.get("capture", False):
            captured = True
        if done:
            break
    return captured, env.t, total_r


def evaluate(env: TagEnv, catcher_policy, runner_policy,
                n_episodes: int = 200) -> EvalStats:
    """
    Evaluate given policies for a small number of episodes.
    """
    caps = 0
    steps_caps: List[int] = []
    returns: List[float] = []
    for _ in range(n_episodes):
        captured, steps, ret = run_episode(env, catcher_policy, runner_policy)
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
