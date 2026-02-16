"""
Training loops for Nash Q-Learning in Wumpus Chase.

Provides training functions for:
- Nash-Q self-play (two independent Nash-Q agents)
- Nash-Q vs FP (Nash-Q agent A vs Fictitious Play agent B)
- Nash-Q vs Minimax-Q (comparison between methods)
"""

from __future__ import annotations

from typing import Dict, List

from agents.dyna_q_agent import DynaQAgent
from agents.nash_q_agent import NashQAgent
from agents.fp_agent import FPAgent
from agents.minimax_q_agent import MinimaxQAgent
from env.wumpus_env_extended import ACTIONS, WumpusChaseEnvExtended
from utils import encode_state
from train.common import evaluate


def train_nashq_selfplay(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "nashq_selfplay",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train two independent Nash-Q agents in self-play."""
    agent_a = NashQAgent(size=env.size, seed=seed)
    agent_b = NashQAgent(size=env.size, seed=seed + 1000)

    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_steps": [],
        "avg_return": [],
        "nash_failures": [],
    }

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_a = agent_a.epsilon(ep)
        eps_b = agent_b.epsilon(ep)

        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_a = agent_a.act_a(sid, eps_a)
            a_b = agent_b.act_b(sid, eps_b)

            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))

            agent_a.update(sid, a_a, a_b, r_a, r_b, sid2, done)
            agent_b.update(sid, a_a, a_b, r_a, r_b, sid2, done)

            if logger:
                pi_a, _, v_a, _ = agent_a.get_policies(sid)
                _, pi_b, _, v_b = agent_b.get_policies(sid)
                logger.log_step(
                    run_label=run_label,
                    phase="train",
                    env=env,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_a=a_a,
                    action_b=a_b,
                    next_state=s2,
                    reward=r_a,
                    done=done,
                    info=info,
                    actions=ACTIONS,
                    agent_a="NashQ",
                    agent_b="NashQ",
                    sid=sid,
                    sid_next=sid2,
                    epsilon=(eps_a + eps_b) / 2.0,
                    pi_a=pi_a.tolist(),
                    pi_b=pi_b.tolist(),
                    value_a=v_a,
                    value_b=v_b,
                )

            s = s2
            if done:
                break

        if ep % eval_every == 0:
            stats = evaluate(
                env,
                lambda state: agent_a.act_a(encode_state(state, env.size), eps=0.0),
                lambda state: agent_b.act_b(encode_state(state, env.size), eps=0.0),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a="NashQ",
                agent_b="NashQ",
            )

            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            logs["nash_failures"].append(
                agent_a.get_stats()["nash_failures"] + agent_b.get_stats()["nash_failures"]
            )

            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.2%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.2%}, "
                    f"Steps={stats.avg_steps:.1f}, "
                    f"Return={stats.avg_return:.2f}"
                )

    return logs


def train_nashq_vs_nashq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "nashq_vs_nashq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Backward-compatible alias for Nash-Q self-play."""
    return train_nashq_selfplay(
        env=env,
        episodes=episodes,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=seed,
        logger=logger,
        eval_logger=eval_logger,
        run_label=run_label,
        verbose=verbose,
    )


def train_nashq_vs_fp(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "nashq_vs_fp",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Train Nash Q-Learning (player A) against Fictitious Play (player B).
    
    This tests how well Nash-Q can exploit a non-equilibrium opponent.
    """
    nash_agent = NashQAgent(size=env.size, seed=seed)
    fp_agent = FPAgent(role="col", layout=env.layout, p_fail=env.p_fail, seed=seed + 100)
    
    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_steps": [],
        "avg_return": [],
    }
    
    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = nash_agent.epsilon(ep)
        
        for step in range(env.t_max):
            sid = encode_state(s, env.size)
            
            # Nash-Q for player A
            a_a = nash_agent.act_a(sid, eps)
            
            # FP for player B
            a_b = fp_agent.act(s)
            
            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))
            
            # Update Nash-Q agent
            nash_agent.update(sid, a_a, a_b, r_a, r_b, sid2, done)
            
            # Update FP opponent model
            fp_agent.observe(s, a_a)
            
            if logger:
                logger.log_step(
                    run_label=run_label,
                    phase="train",
                    env=env,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_a=a_a,
                    action_b=a_b,
                    next_state=s2,
                    reward=r_a,
                    done=done,
                    info=info,
                    actions=ACTIONS,
                    agent_a="NashQ",
                    agent_b="FP",
                    sid=sid,
                    sid_next=sid2,
                    epsilon=eps,
                )
            
            s = s2
            if done:
                break
        
        if ep % eval_every == 0:
            def policy_a(state):
                sid = encode_state(state, env.size)
                return nash_agent.act_a(sid, eps=0.0)
            
            def policy_b(state):
                return fp_agent.act(state)
            
            stats = evaluate(
                env, policy_a, policy_b,
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                agent_a="NashQ",
                agent_b="FP",
            )
            
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            
            if verbose and ep % (eval_every * 5) == 0:
                print(f"[{run_label}] Episode {ep}: "
                      f"Win={stats.win_rate:.2%}, "
                      f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.2%}")
    
    return logs


def train_nashq_vs_minimaxq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "nashq_vs_minimaxq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Train Nash Q-Learning (player A) against Minimax Q (player B).
    
    Compares Nash equilibrium vs Minimax strategies.
    In zero-sum games, they should converge to similar solutions.
    """
    nash_agent = NashQAgent(size=env.size, seed=seed)
    minimax_agent = MinimaxQAgent(size=env.size, seed=seed + 100)
    
    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_steps": [],
        "avg_return": [],
    }
    
    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = nash_agent.epsilon(ep)
        
        for step in range(env.t_max):
            sid = encode_state(s, env.size)
            
            # Nash-Q for player A
            a_a = nash_agent.act_a(sid, eps)
            
            # Minimax-Q for player B (runner/column role)
            a_b = minimax_agent.act_col(sid, eps)
            
            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))
            
            # Update both agents
            nash_agent.update(sid, a_a, a_b, r_a, r_b, sid2, done)
            minimax_agent.update(sid, a_b, a_a, r_b, sid2, done)
            
            if logger:
                logger.log_step(
                    run_label=run_label,
                    phase="train",
                    env=env,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_a=a_a,
                    action_b=a_b,
                    next_state=s2,
                    reward=r_a,
                    done=done,
                    info=info,
                    actions=ACTIONS,
                    agent_a="NashQ",
                    agent_b="MinimaxQ",
                    sid=sid,
                    sid_next=sid2,
                    epsilon=eps,
                )
            
            s = s2
            if done:
                break
        
        if ep % eval_every == 0:
            def policy_a(state):
                sid = encode_state(state, env.size)
                return nash_agent.act_a(sid, eps=0.0)
            
            def policy_b(state):
                sid = encode_state(state, env.size)
                return minimax_agent.act_col(sid, eps=0.0)
            
            stats = evaluate(
                env, policy_a, policy_b,
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                agent_a="NashQ",
                agent_b="MinimaxQ",
            )
            
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            
            if verbose and ep % (eval_every * 5) == 0:
                print(f"[{run_label}] Episode {ep}: "
                      f"Win={stats.win_rate:.2%}, "
                      f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.2%}")
    
    return logs


def train_nashq_vs_dynaq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "nashq_vs_dynaq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train Nash-Q (player A) against Dyna-Q (player B)."""
    nash_agent = NashQAgent(size=env.size, seed=seed)
    dynaq_agent = DynaQAgent(size=env.size, n_planning=n_planning, seed=seed + 100, player="B")

    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_steps": [],
        "avg_return": [],
    }

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_nash = nash_agent.epsilon(ep)
        eps_dyna = dynaq_agent.epsilon(ep)

        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_a = nash_agent.act_a(sid, eps_nash)
            a_b = dynaq_agent.act_b(sid, eps_dyna)

            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))

            nash_agent.update(sid, a_a, a_b, r_a, r_b, sid2, done)
            dynaq_agent.update(sid, a_a, a_b, r_b, sid2, done)

            if logger:
                logger.log_step(
                    run_label=run_label,
                    phase="train",
                    env=env,
                    episode=ep,
                    step_idx=env.t,
                    state=s,
                    action_a=a_a,
                    action_b=a_b,
                    next_state=s2,
                    reward=r_a,
                    done=done,
                    info=info,
                    actions=ACTIONS,
                    agent_a="NashQ",
                    agent_b="DynaQ",
                    sid=sid,
                    sid_next=sid2,
                )

            s = s2
            if done:
                break

        if ep % eval_every == 0:
            stats = evaluate(
                env,
                lambda state: nash_agent.act_a(encode_state(state, env.size), eps=0.0),
                lambda state: dynaq_agent.act_b(encode_state(state, env.size), eps=0.0),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                agent_a="NashQ",
                agent_b="DynaQ",
            )

            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)

            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.2%}, RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.2%}"
                )

    return logs


def last_or_nan(lst: List[float]) -> float:
    """Return last element or NaN if empty."""
    return lst[-1] if lst else float("nan")
