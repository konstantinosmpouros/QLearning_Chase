from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from env import ACTIONS, WumpusChaseEnv
from utils import encode_state


@dataclass
class EvalStats:
    win_rate: float
    draw_rate: float
    avg_steps: float
    avg_return: float


def run_episode(
    env: WumpusChaseEnv,
    policy_a: Callable,
    policy_b: Callable,
    logger=None,
    run_label: str = "eval",
    phase: str = "eval",
    episode: int = 0,
    agent_a: str | None = None,
    agent_b: str | None = None,
) -> tuple[float, int, str | None]:
    s = env.reset()
    total_reward = 0.0
    outcome = None
    for _ in range(env.t_max):
        a1 = policy_a(s)
        a2 = policy_b(s)
        sid = encode_state(s, env.size)
        s2, r, done, info = env.step(a1, a2)
        sid2 = encode_state(s2, env.size)
        total_reward += r
        outcome = info.get("outcome")
        if logger:
            logger.log_step(
                run_label=run_label,
                phase=phase,
                env=env,
                episode=episode,
                step_idx=env.t,
                state=s,
                action_a=a1,
                action_b=a2,
                next_state=s2,
                reward=r,
                done=done,
                info=info,
                actions=ACTIONS,
                agent_a=agent_a,
                agent_b=agent_b,
                sid=sid,
                sid_next=sid2,
            )
        s = s2
        if done:
            break
    return total_reward, env.t, outcome


def evaluate(
    env: WumpusChaseEnv,
    policy_a: Callable,
    policy_b: Callable,
    n_episodes: int = 10,
    logger=None,
    run_label: str = "eval",
    phase: str = "eval",
    agent_a: str | None = None,
    agent_b: str | None = None,
) -> EvalStats:
    wins = 0
    draws = 0
    total_steps = 0
    total_return = 0.0
    for ep in range(1, n_episodes + 1):
        ep_return, ep_steps, outcome = run_episode(
            env,
            policy_a,
            policy_b,
            logger=logger,
            run_label=run_label,
            phase=phase,
            episode=ep,
            agent_a=agent_a,
            agent_b=agent_b,
        )
        total_return += ep_return
        total_steps += ep_steps
        if outcome == "A_WIN":
            wins += 1
        elif outcome == "DRAW":
            draws += 1
    win_rate = wins / n_episodes
    draw_rate = draws / n_episodes
    avg_steps = total_steps / n_episodes
    avg_return = total_return / n_episodes
    return EvalStats(
        win_rate=win_rate,
        draw_rate=draw_rate,
        avg_steps=avg_steps,
        avg_return=avg_return,
    )
