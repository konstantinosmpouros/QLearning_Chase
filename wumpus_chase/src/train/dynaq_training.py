"""
Training loops for Dyna-Q in Wumpus Chase.

Provides training functions for:
- Dyna-Q self-play
- Dyna-Q vs Dyna-Q (two independent agents)
- Dyna-Q vs other agents (FP, Minimax-Q, Nash-Q)
- Dyna-Q+ variants
"""

from __future__ import annotations

from typing import Callable, Dict, List

from agents.dyna_q_agent import DynaQAgent, DynaQPlusAgent
from agents.fp_agent import FPAgent
from agents.minimax_q_agent import MinimaxQAgent
from agents.nash_q_agent import NashQAgent
from env.wumpus_env_extended import ACTIONS, WumpusChaseEnvExtended
from utils import encode_state
from train.common import evaluate


def _init_logs() -> Dict[str, List[float]]:
    return {"episode": [], "win_rate": [], "draw_rate": [], "avg_steps": [], "avg_return": []}


def _train_two_dynaq_family_agents(
    env: WumpusChaseEnvExtended,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    seed: int,
    logger,
    eval_logger,
    run_label: str,
    verbose: bool,
    make_agent_a: Callable[[int, str], DynaQAgent],
    make_agent_b: Callable[[int, str], DynaQAgent],
    agent_a_name: str,
    agent_b_name: str,
) -> Dict[str, List[float]]:
    agent_a = make_agent_a(seed, "A")
    agent_b = make_agent_b(seed + 1000, "B")
    logs = _init_logs()

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

            agent_a.update(sid, a_a, a_b, r_a, sid2, done)
            agent_b.update(sid, a_a, a_b, r_b, sid2, done)

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
                    agent_a=agent_a_name,
                    agent_b=agent_b_name,
                    sid=sid,
                    sid_next=sid2,
                    epsilon=(eps_a + eps_b) / 2.0,
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
                agent_a=agent_a_name,
                agent_b=agent_b_name,
            )

            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)

            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}"
                )

    return logs


def train_dynaq_selfplay(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_selfplay",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train two independent Dyna-Q agents in self-play."""
    agent_a = DynaQAgent(size=env.size, n_actions=len(ACTIONS), n_planning=n_planning, seed=seed, player="A")
    agent_b = DynaQAgent(size=env.size, n_actions=len(ACTIONS), n_planning=n_planning, seed=seed + 1000, player="B")

    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_steps": [],
        "avg_return": [],
        "model_size": [],
        "planning_ratio": [],
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

            agent_a.update(sid, a_a, a_b, r_a, sid2, done)
            agent_b.update(sid, a_a, a_b, r_b, sid2, done)

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
                    agent_a="DynaQ",
                    agent_b="DynaQ",
                    sid=sid,
                    sid_next=sid2,
                    epsilon=(eps_a + eps_b) / 2.0,
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
                agent_a="DynaQ",
                agent_b="DynaQ",
            )

            stats_a = agent_a.get_stats()
            stats_b = agent_b.get_stats()
            model_size = stats_a.get("unique_transitions", 0) + stats_b.get("unique_transitions", 0)
            planning_ratio = (stats_a.get("planning_ratio", 0.0) + stats_b.get("planning_ratio", 0.0)) / 2.0

            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            logs["model_size"].append(model_size)
            logs["planning_ratio"].append(planning_ratio)

            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}, "
                    f"Model={model_size}"
                )

    return logs


def train_dynaq_vs_dynaq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_vs_dynaq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Backward-compatible alias for Dyna-Q self-play."""
    return train_dynaq_selfplay(
        env=env,
        episodes=episodes,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        n_planning=n_planning,
        seed=seed,
        logger=logger,
        eval_logger=eval_logger,
        run_label=run_label,
        verbose=verbose,
    )


def train_dynaq_vs_fp(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_vs_fp",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Train Dyna-Q (player A) against Fictitious Play (player B).
    """
    dynaq_agent = DynaQAgent(size=env.size, n_planning=n_planning, seed=seed, player="A")
    
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
        eps = dynaq_agent.epsilon(ep)
        
        for step in range(env.t_max):
            sid = encode_state(s, env.size)
            
            a_a = dynaq_agent.act_a(sid, eps)
            a_b = fp_agent.act(s)
            
            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            
            # Update Dyna-Q
            dynaq_agent.update(sid, a_a, a_b, r_a, sid2, done)
            
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
                    agent_a="DynaQ",
                    agent_b="FP",
                    sid=sid,
                    sid_next=sid2,
                )
            
            s = s2
            if done:
                break
        
        if ep % eval_every == 0:
            def policy_a(state):
                sid = encode_state(state, env.size)
                return dynaq_agent.act_a(sid, eps=0.0)
            
            def policy_b(state):
                return fp_agent.act(state)
            
            stats = evaluate(
                env, policy_a, policy_b,
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                agent_a="DynaQ",
                agent_b="FP",
            )
            
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            
            if verbose and ep % (eval_every * 5) == 0:
                print(f"[{run_label}] Episode {ep}: "
                      f"Win={stats.win_rate:.1%}, RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}")
    
    return logs


def train_dynaq_vs_minimaxq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_vs_minimaxq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Train Dyna-Q (player A) against Minimax-Q (player B).
    """
    dynaq_agent = DynaQAgent(size=env.size, n_planning=n_planning, seed=seed, player="A")
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
        eps_dyna = dynaq_agent.epsilon(ep)
        eps_mm = minimax_agent.epsilon(ep)
        
        for step in range(env.t_max):
            sid = encode_state(s, env.size)
            
            a_a = dynaq_agent.act_a(sid, eps_dyna)
            
            a_b = minimax_agent.act_col(sid, eps_mm)
            
            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))
            
            # Update both agents
            dynaq_agent.update(sid, a_a, a_b, r_a, sid2, done)
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
                    agent_a="DynaQ",
                    agent_b="MinimaxQ",
                    sid=sid,
                    sid_next=sid2,
                )
            
            s = s2
            if done:
                break
        
        if ep % eval_every == 0:
            def policy_a(state):
                sid = encode_state(state, env.size)
                return dynaq_agent.act_a(sid, eps=0.0)
            
            def policy_b(state):
                sid = encode_state(state, env.size)
                return minimax_agent.act_col(sid, eps=0.0)
            
            stats = evaluate(
                env, policy_a, policy_b,
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                agent_a="DynaQ",
                agent_b="MinimaxQ",
            )
            
            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)
            
            if verbose and ep % (eval_every * 5) == 0:
                print(f"[{run_label}] Episode {ep}: "
                      f"Win={stats.win_rate:.1%}, RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}")
    
    return logs


def train_dynaq_vs_nashq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_vs_nashq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train Dyna-Q (player A) against Nash-Q (player B)."""
    dynaq_agent = DynaQAgent(size=env.size, n_planning=n_planning, seed=seed, player="A")
    nash_agent = NashQAgent(size=env.size, seed=seed + 100)

    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_steps": [],
        "avg_return": [],
    }

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_dyna = dynaq_agent.epsilon(ep)
        eps_nash = nash_agent.epsilon(ep)

        for _ in range(env.t_max):
            sid = encode_state(s, env.size)

            a_a = dynaq_agent.act_a(sid, eps_dyna)
            a_b = nash_agent.act_b(sid, eps_nash)

            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))

            dynaq_agent.update(sid, a_a, a_b, r_a, sid2, done)
            nash_agent.update(sid, a_a, a_b, r_a, r_b, sid2, done)

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
                    agent_a="DynaQ",
                    agent_b="NashQ",
                    sid=sid,
                    sid_next=sid2,
                )

            s = s2
            if done:
                break

        if ep % eval_every == 0:
            stats = evaluate(
                env,
                lambda state: dynaq_agent.act_a(encode_state(state, env.size), eps=0.0),
                lambda state: nash_agent.act_b(encode_state(state, env.size), eps=0.0),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                agent_a="DynaQ",
                agent_b="NashQ",
            )

            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)

            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}"
                )

    return logs


def train_dynaq_plus_selfplay(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    kappa: float = 0.001,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_plus_selfplay",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Train two independent Dyna-Q+ agents in self-play."""
    return _train_two_dynaq_family_agents(
        env=env,
        episodes=episodes,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=seed,
        logger=logger,
        eval_logger=eval_logger,
        run_label=run_label,
        verbose=verbose,
        make_agent_a=lambda s, player: DynaQPlusAgent(
            size=env.size,
            n_actions=len(ACTIONS),
            n_planning=n_planning,
            kappa=kappa,
            seed=s,
            player=player,
        ),
        make_agent_b=lambda s, player: DynaQPlusAgent(
            size=env.size,
            n_actions=len(ACTIONS),
            n_planning=n_planning,
            kappa=kappa,
            seed=s,
            player=player,
        ),
        agent_a_name="DynaQ+",
        agent_b_name="DynaQ+",
    )


def _train_dynaq_family_vs_fp(
    env: WumpusChaseEnvExtended,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    seed: int,
    logger,
    eval_logger,
    run_label: str,
    verbose: bool,
    make_agent_a: Callable[[int], DynaQAgent],
    agent_a_name: str,
) -> Dict[str, List[float]]:
    agent_a = make_agent_a(seed)
    fp_agent = FPAgent(role="col", layout=env.layout, p_fail=env.p_fail, seed=seed + 100)
    logs = _init_logs()

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps = agent_a.epsilon(ep)

        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_a = agent_a.act_a(sid, eps)
            a_b = fp_agent.act(s)

            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)

            agent_a.update(sid, a_a, a_b, r_a, sid2, done)
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
                    agent_a=agent_a_name,
                    agent_b="FP",
                    sid=sid,
                    sid_next=sid2,
                    epsilon=eps,
                )

            s = s2
            if done:
                break

        if ep % eval_every == 0:
            stats = evaluate(
                env,
                lambda state: agent_a.act_a(encode_state(state, env.size), eps=0.0),
                lambda state: fp_agent.act(state),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a=agent_a_name,
                agent_b="FP",
            )

            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)

            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}"
                )

    return logs


def _train_dynaq_family_vs_minimaxq(
    env: WumpusChaseEnvExtended,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    seed: int,
    logger,
    eval_logger,
    run_label: str,
    verbose: bool,
    make_agent_a: Callable[[int], DynaQAgent],
    agent_a_name: str,
) -> Dict[str, List[float]]:
    agent_a = make_agent_a(seed)
    minimax_agent = MinimaxQAgent(size=env.size, seed=seed + 100)
    logs = _init_logs()

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_a = agent_a.epsilon(ep)
        eps_mm = minimax_agent.epsilon(ep)

        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_a = agent_a.act_a(sid, eps_a)
            a_b = minimax_agent.act_col(sid, eps_mm)

            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))

            agent_a.update(sid, a_a, a_b, r_a, sid2, done)
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
                    agent_a=agent_a_name,
                    agent_b="MinimaxQ",
                    sid=sid,
                    sid_next=sid2,
                    epsilon=eps_a,
                )

            s = s2
            if done:
                break

        if ep % eval_every == 0:
            stats = evaluate(
                env,
                lambda state: agent_a.act_a(encode_state(state, env.size), eps=0.0),
                lambda state: minimax_agent.act_col(encode_state(state, env.size), eps=0.0),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a=agent_a_name,
                agent_b="MinimaxQ",
            )

            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)

            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}"
                )

    return logs


def _train_dynaq_family_vs_nashq(
    env: WumpusChaseEnvExtended,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    seed: int,
    logger,
    eval_logger,
    run_label: str,
    verbose: bool,
    make_agent_a: Callable[[int], DynaQAgent],
    agent_a_name: str,
) -> Dict[str, List[float]]:
    agent_a = make_agent_a(seed)
    nash_agent = NashQAgent(size=env.size, seed=seed + 100)
    logs = _init_logs()

    for ep in range(1, episodes + 1):
        s = env.reset()
        eps_a = agent_a.epsilon(ep)
        eps_nash = nash_agent.epsilon(ep)

        for _ in range(env.t_max):
            sid = encode_state(s, env.size)
            a_a = agent_a.act_a(sid, eps_a)
            a_b = nash_agent.act_b(sid, eps_nash)

            s2, r_a, done, info = env.step(a_a, a_b)
            sid2 = encode_state(s2, env.size)
            r_b = float(info.get("reward_b", -r_a))

            agent_a.update(sid, a_a, a_b, r_a, sid2, done)
            nash_agent.update(sid, a_a, a_b, r_a, r_b, sid2, done)

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
                    agent_a=agent_a_name,
                    agent_b="NashQ",
                    sid=sid,
                    sid_next=sid2,
                    epsilon=eps_a,
                )

            s = s2
            if done:
                break

        if ep % eval_every == 0:
            stats = evaluate(
                env,
                lambda state: agent_a.act_a(encode_state(state, env.size), eps=0.0),
                lambda state: nash_agent.act_b(encode_state(state, env.size), eps=0.0),
                n_episodes=eval_episodes,
                logger=eval_logger,
                run_label=run_label,
                phase="eval",
                agent_a=agent_a_name,
                agent_b="NashQ",
            )

            logs["episode"].append(ep)
            logs["win_rate"].append(stats.win_rate)
            logs["draw_rate"].append(stats.draw_rate)
            logs["avg_steps"].append(stats.avg_steps)
            logs["avg_return"].append(stats.avg_return)

            if verbose and ep % (eval_every * 5) == 0:
                print(
                    f"[{run_label}] Episode {ep}: "
                    f"Win={stats.win_rate:.1%}, "
                    f"RunnerWin={max(0.0, 1.0 - stats.win_rate - stats.draw_rate):.1%}"
                )

    return logs


def train_dynaq_plus_vs_fp(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    kappa: float = 0.001,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_plus_vs_fp",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    return _train_dynaq_family_vs_fp(
        env=env,
        episodes=episodes,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=seed,
        logger=logger,
        eval_logger=eval_logger,
        run_label=run_label,
        verbose=verbose,
        make_agent_a=lambda s: DynaQPlusAgent(
            size=env.size,
            n_actions=len(ACTIONS),
            n_planning=n_planning,
            kappa=kappa,
            seed=s,
            player="A",
        ),
        agent_a_name="DynaQ+",
    )


def train_dynaq_plus_vs_minimaxq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    kappa: float = 0.001,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_plus_vs_minimaxq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    return _train_dynaq_family_vs_minimaxq(
        env=env,
        episodes=episodes,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=seed,
        logger=logger,
        eval_logger=eval_logger,
        run_label=run_label,
        verbose=verbose,
        make_agent_a=lambda s: DynaQPlusAgent(
            size=env.size,
            n_actions=len(ACTIONS),
            n_planning=n_planning,
            kappa=kappa,
            seed=s,
            player="A",
        ),
        agent_a_name="DynaQ+",
    )


def train_dynaq_plus_vs_nashq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    kappa: float = 0.001,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_plus_vs_nashq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    return _train_dynaq_family_vs_nashq(
        env=env,
        episodes=episodes,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=seed,
        logger=logger,
        eval_logger=eval_logger,
        run_label=run_label,
        verbose=verbose,
        make_agent_a=lambda s: DynaQPlusAgent(
            size=env.size,
            n_actions=len(ACTIONS),
            n_planning=n_planning,
            kappa=kappa,
            seed=s,
            player="A",
        ),
        agent_a_name="DynaQ+",
    )


def train_dynaq_plus_vs_dynaq(
    env: WumpusChaseEnvExtended,
    episodes: int = 10_000,
    eval_every: int = 1000,
    eval_episodes: int = 20,
    n_planning: int = 10,
    kappa: float = 0.001,
    seed: int = 0,
    logger=None,
    eval_logger=None,
    run_label: str = "dynaq_plus_vs_dynaq",
    verbose: bool = False,
) -> Dict[str, List[float]]:
    return _train_two_dynaq_family_agents(
        env=env,
        episodes=episodes,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        seed=seed,
        logger=logger,
        eval_logger=eval_logger,
        run_label=run_label,
        verbose=verbose,
        make_agent_a=lambda s, player: DynaQPlusAgent(
            size=env.size,
            n_actions=len(ACTIONS),
            n_planning=n_planning,
            kappa=kappa,
            seed=s,
            player=player,
        ),
        make_agent_b=lambda s, player: DynaQAgent(
            size=env.size,
            n_actions=len(ACTIONS),
            n_planning=n_planning,
            seed=s,
            player=player,
        ),
        agent_a_name="DynaQ+",
        agent_b_name="DynaQ",
    )


def compare_planning_steps(
    env: WumpusChaseEnvExtended,
    episodes: int = 5_000,
    eval_every: int = 1000,
    planning_values: List[int] = None,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Compare Dyna-Q performance with different numbers of planning steps.
    
    This helps understand the trade-off between computation and sample efficiency.
    """
    if planning_values is None:
        planning_values = [0, 5, 10, 25, 50]
    
    results = {}
    
    for n_planning in planning_values:
        if verbose:
            print(f"Training with n_planning={n_planning}...")
        
        logs = train_dynaq_selfplay(
            env,
            episodes=episodes,
            eval_every=eval_every,
            n_planning=n_planning,
            seed=seed,
            run_label=f"dynaq_planning_{n_planning}",
            verbose=False,
        )
        
        results[n_planning] = logs
        
        if verbose:
            final_win = logs["win_rate"][-1] if logs["win_rate"] else 0
            print(f"  Final win rate: {final_win:.1%}")
    
    return results


def last_or_nan(lst: List[float]) -> float:
    """Return last element or NaN if empty."""
    return lst[-1] if lst else float("nan")
