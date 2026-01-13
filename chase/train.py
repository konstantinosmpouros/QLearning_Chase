"""
Training and evaluation helpers for the chase project.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from agents import FPAgent, MinimaxQAgent
from env import ACTIONS, TagEnv
from logger import ExcelLogger
from utils import encode_state


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


def train_fp_vs_fp(p_fail: float,
                    episodes: int = 200,
                    eval_every: int = 100,
                    seed: int = 0,
                    logger: ExcelLogger | None = None,
                    run_label: str = "fp_vs_fp") -> Dict[str, List[float]]:
    """Train FP agents against each other for a small number of episodes."""
    env = TagEnv(p_fail=p_fail, seed=seed)
    fp_c = FPAgent("catcher", p_fail=p_fail, seed=seed + 1)
    fp_r = FPAgent("runner", p_fail=p_fail, seed=seed + 2)
    logs = {"episode": [], "capture_rate": [], "avg_steps": [], "avg_return": []}
    # Define evaluation policies capturing current behaviour.
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
            stats = evaluate(eval_env, catcher_policy, runner_policy, n_episodes=10)
            logs["episode"].append(ep)
            logs["capture_rate"].append(stats.capture_rate)
            logs["avg_steps"].append(stats.avg_steps_to_capture)
            logs["avg_return"].append(stats.avg_return)
    return logs


def train_minimaxq_selfplay(p_fail: float,
                            episodes: int = 200,
                            eval_every: int = 100,
                            seed: int = 0,
                            logger: ExcelLogger | None = None,
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
            stats = evaluate(eval_env, catcher_policy, runner_policy, n_episodes=10)
            logs["episode"].append(ep)
            logs["capture_rate"].append(stats.capture_rate)
            logs["avg_steps"].append(stats.avg_steps_to_capture)
            logs["avg_return"].append(stats.avg_return)
    return logs


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


def last_or_nan(lst: List[float]) -> float:
    return lst[-1] if lst else float("nan")
