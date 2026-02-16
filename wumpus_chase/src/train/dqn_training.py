"""
Training loops for Deep Q-Network agents in Wumpus Chase.

Provides training functions for:
- DQN self-play (single agent plays both roles)
- DQN vs DQN (two independent agents)
- DQN vs other agents (FP, Minimax-Q, Nash-Q)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Callable, Optional
import time

import numpy as np

# Ensure imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dqn_agent import (
    DQNAgent, MultiAgentDQN, DQNConfig, DQNType,
    create_state_representation, get_state_dim,
    TORCH_AVAILABLE,
)
from env.wumpus_env_extended import WumpusChaseEnvExtended, ACTIONS
from train.common import EvalStats


def train_dqn_selfplay(
    env: WumpusChaseEnvExtended,
    episodes: int = 5000,
    eval_every: int = 1000,
    eval_episodes: int = 50,
    seed: int = 0,
    dqn_type: DQNType = DQNType.DUELING_DOUBLE,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, List]:
    """
    Train DQN agent in self-play mode.
    
    Single MultiAgentDQN agent controls both players, learning
    optimal strategies through self-play.
    
    Args:
        env: Wumpus Chase environment
        episodes: Number of training episodes
        eval_every: Evaluate every N episodes
        seed: Random seed
        dqn_type: Type of DQN architecture
        verbose: Print progress
        save_path: Path to save model checkpoints
        
    Returns:
        Dictionary with training metrics
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for DQN training")
    
    # Get state dimension
    state_dim = get_state_dim(env.size, env.layout, include_features=True)
    
    # Create config
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=len(ACTIONS),
        dqn_type=dqn_type,
        hidden_dims=[256, 256],
        gamma=0.99,
        learning_rate=1e-3,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=int(episodes * 0.7),
        buffer_size=100_000,
        min_buffer_size=1000,
        target_update_freq=100,
    )
    
    # Create multi-agent DQN
    ma_dqn = MultiAgentDQN(
        state_dim=state_dim,
        action_dim=len(ACTIONS),
        mode="self_play",
        config=config,
        seed=seed,
    )
    
    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_steps": [],
        "avg_return": [],
        "epsilon": [],
        "loss": [],
    }
    
    if verbose:
        print(f"Training DQN ({dqn_type.value}) self-play for {episodes} episodes...")
        print(f"  State dim: {state_dim}")
        print(f"  Hidden dims: {config.hidden_dims}")
        print()
    
    start_time = time.time()
    episode_returns = []
    
    for ep in range(1, episodes + 1):
        state_tuple = env.reset()
        state = create_state_representation(state_tuple, env.size, env.layout)
        
        episode_return = 0.0
        episode_loss = []
        
        for step in range(env.t_max):
            epsilon = ma_dqn.get_epsilon()
            
            # Select actions
            action_a = ma_dqn.select_action_a(state, epsilon)
            action_b = ma_dqn.select_action_b(state, epsilon)
            
            # Environment step
            next_state_tuple, reward, done, info = env.step(action_a, action_b)
            reward_b = float(info.get("reward_b", -reward))
            next_state = create_state_representation(next_state_tuple, env.size, env.layout)
            
            # Store transitions for both agents
            ma_dqn.store_transition_a(state, action_a, reward, next_state, done)
            ma_dqn.store_transition_b(state, action_b, reward_b, next_state, done)
            
            # Train
            losses = ma_dqn.train_step()
            if losses.get('agent') is not None:
                episode_loss.append(losses['agent'])
            
            episode_return += reward
            state = next_state
            
            if done:
                break
        
        episode_returns.append(episode_return)
        
        # Evaluation
        if ep % eval_every == 0:
            # Evaluate with no exploration
            eval_stats = evaluate_dqn(
                env, ma_dqn, n_episodes=eval_episodes,
                state_dim=state_dim
            )
            
            logs["episode"].append(ep)
            logs["win_rate"].append(eval_stats.win_rate)
            logs["draw_rate"].append(eval_stats.draw_rate)
            logs["avg_steps"].append(eval_stats.avg_steps)
            logs["avg_return"].append(eval_stats.avg_return)
            logs["epsilon"].append(epsilon)
            logs["loss"].append(np.mean(episode_loss) if episode_loss else 0.0)
            
            if verbose:
                elapsed = time.time() - start_time
                recent_return = np.mean(episode_returns[-eval_every:])
                print(f"Episode {ep:5d} | "
                      f"Win: {eval_stats.win_rate:.1%} | "
                      f"RunnerWin: {max(0.0, 1.0 - eval_stats.win_rate - eval_stats.draw_rate):.1%} | "
                      f"Steps: {eval_stats.avg_steps:.1f} | "
                      f"ε: {epsilon:.3f} | "
                      f"Return: {recent_return:.2f} | "
                      f"Time: {elapsed:.0f}s")
                start_time = time.time()
            
            # Save checkpoint
            if save_path:
                ma_dqn.save(f"{save_path}_ep{ep}")
    
    # Final save
    if save_path:
        ma_dqn.save(save_path)
    
    return logs


def train_dqn_vs_dqn(
    env: WumpusChaseEnvExtended,
    episodes: int = 5000,
    eval_every: int = 1000,
    eval_episodes: int = 50,
    seed: int = 0,
    dqn_type: DQNType = DQNType.DUELING_DOUBLE,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Train two independent DQN agents against each other.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for DQN training")
    
    state_dim = get_state_dim(env.size, env.layout, include_features=True)
    
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=len(ACTIONS),
        dqn_type=dqn_type,
        hidden_dims=[256, 256],
        eps_decay_steps=int(episodes * 0.7),
    )
    
    ma_dqn = MultiAgentDQN(
        state_dim=state_dim,
        mode="independent",
        config=config,
        seed=seed,
    )
    
    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_steps": [],
        "avg_return": [],
    }
    
    if verbose:
        print(f"Training DQN vs DQN (independent) for {episodes} episodes...")
    
    for ep in range(1, episodes + 1):
        state_tuple = env.reset()
        state = create_state_representation(state_tuple, env.size, env.layout)
        
        for step in range(env.t_max):
            epsilon = ma_dqn.get_epsilon()
            
            action_a = ma_dqn.select_action_a(state, epsilon)
            action_b = ma_dqn.select_action_b(state, epsilon)
            
            next_state_tuple, reward, done, info = env.step(action_a, action_b)
            reward_b = float(info.get("reward_b", -reward))
            next_state = create_state_representation(next_state_tuple, env.size, env.layout)
            
            ma_dqn.store_transition_a(state, action_a, reward, next_state, done)
            ma_dqn.store_transition_b(state, action_b, reward_b, next_state, done)
            
            ma_dqn.train_step()
            
            state = next_state
            if done:
                break
        
        if ep % eval_every == 0:
            eval_stats = evaluate_dqn(env, ma_dqn, n_episodes=eval_episodes, state_dim=state_dim)
            
            logs["episode"].append(ep)
            logs["win_rate"].append(eval_stats.win_rate)
            logs["draw_rate"].append(eval_stats.draw_rate)
            logs["avg_steps"].append(eval_stats.avg_steps)
            logs["avg_return"].append(eval_stats.avg_return)
            
            if verbose:
                print(f"Episode {ep}: Win={eval_stats.win_rate:.1%}, "
                      f"ε={epsilon:.3f}")
    
    return logs


def evaluate_dqn(
    env: WumpusChaseEnvExtended,
    ma_dqn: MultiAgentDQN,
    n_episodes: int = 100,
    state_dim: int = None,
) -> EvalStats:
    """
    Evaluate DQN agent with no exploration.
    """
    if state_dim is None:
        state_dim = get_state_dim(env.size, env.layout, include_features=True)
    
    wins = 0
    draws = 0
    total_steps = 0
    total_return = 0.0
    
    for ep in range(n_episodes):
        state_tuple = env.reset()
        state = create_state_representation(state_tuple, env.size, env.layout)
        episode_return = 0.0
        
        for step in range(env.t_max):
            # No exploration during evaluation
            action_a = ma_dqn.select_action_a(state, epsilon=0.0)
            action_b = ma_dqn.select_action_b(state, epsilon=0.0)
            
            next_state_tuple, reward, done, info = env.step(action_a, action_b)
            next_state = create_state_representation(next_state_tuple, env.size, env.layout)
            
            episode_return += reward
            state = next_state
            
            if done:
                outcome = info.get("outcome", "B_WIN")
                if outcome == "A_WIN":
                    wins += 1
                elif outcome == "DRAW":
                    draws += 1
                total_steps += env.t
                break
        
        total_return += episode_return
    
    return EvalStats(
        win_rate=wins / n_episodes,
        draw_rate=draws / n_episodes,
        avg_steps=total_steps / n_episodes,
        avg_return=total_return / n_episodes,
    )


def create_dqn_policy(
    ma_dqn: MultiAgentDQN,
    env_size: int,
    layout,
    player: str = "A",
) -> Callable:
    """
    Create a policy function from trained DQN.
    
    Args:
        ma_dqn: Trained MultiAgentDQN
        env_size: Environment grid size
        layout: Map layout
        player: "A" or "B"
        
    Returns:
        Policy function: state_tuple -> action
    """
    def policy(state_tuple):
        state = create_state_representation(state_tuple, env_size, layout)
        if player == "A":
            return ma_dqn.select_action_a(state, epsilon=0.0)
        else:
            return ma_dqn.select_action_b(state, epsilon=0.0)
    
    return policy


def train_dqn_vs_opponent(
    env: WumpusChaseEnvExtended,
    opponent_policy: Callable,
    episodes: int = 5000,
    eval_every: int = 1000,
    eval_episodes: int = 50,
    seed: int = 0,
    dqn_as_player: str = "A",
    dqn_type: DQNType = DQNType.DUELING_DOUBLE,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Train DQN against a fixed opponent policy.
    
    Args:
        env: Environment
        opponent_policy: Fixed opponent policy (state_tuple -> action)
        episodes: Training episodes
        eval_every: Evaluation frequency
        seed: Random seed
        dqn_as_player: "A" or "B" - which player DQN controls
        dqn_type: DQN architecture type
        verbose: Print progress
        
    Returns:
        Training logs
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for DQN training")
    
    state_dim = get_state_dim(env.size, env.layout, include_features=True)
    
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=len(ACTIONS),
        dqn_type=dqn_type,
        hidden_dims=[256, 256],
        eps_decay_steps=int(episodes * 0.7),
    )
    
    dqn_agent = DQNAgent(config, seed=seed)
    
    logs = {
        "episode": [],
        "win_rate": [],
        "draw_rate": [],
        "avg_return": [],
    }
    
    if verbose:
        print(f"Training DQN as Player {dqn_as_player} against opponent...")
    
    for ep in range(1, episodes + 1):
        state_tuple = env.reset()
        state = create_state_representation(state_tuple, env.size, env.layout)
        episode_return = 0.0
        
        for step in range(env.t_max):
            epsilon = dqn_agent.get_epsilon()
            
            if dqn_as_player == "A":
                action_a = dqn_agent.select_action(state, epsilon)
                action_b = opponent_policy(state_tuple)
            else:
                action_a = opponent_policy(state_tuple)
                action_b = dqn_agent.select_action(state, epsilon)
            
            next_state_tuple, reward, done, info = env.step(action_a, action_b)
            reward_player = reward if dqn_as_player == "A" else float(info.get("reward_b", -reward))
            next_state = create_state_representation(next_state_tuple, env.size, env.layout)
            
            dqn_agent.store_transition(
                state, 
                action_a if dqn_as_player == "A" else action_b,
                reward_player,
                next_state,
                done
            )
            
            dqn_agent.train_step()
            
            episode_return += reward_player
            state = next_state
            state_tuple = next_state_tuple
            
            if done:
                break
        
        if ep % eval_every == 0:
            # Evaluate
            wins, draws, total_return = 0, 0, 0.0
            n_eval = eval_episodes
            
            for _ in range(n_eval):
                s_tuple = env.reset()
                s = create_state_representation(s_tuple, env.size, env.layout)
                ep_ret = 0.0
                
                for _ in range(env.t_max):
                    if dqn_as_player == "A":
                        a_a = dqn_agent.select_action(s, epsilon=0.0)
                        a_b = opponent_policy(s_tuple)
                    else:
                        a_a = opponent_policy(s_tuple)
                        a_b = dqn_agent.select_action(s, epsilon=0.0)
                    
                    s_tuple, r, d, info = env.step(a_a, a_b)
                    r_player = r if dqn_as_player == "A" else float(info.get("reward_b", -r))
                    s = create_state_representation(s_tuple, env.size, env.layout)
                    ep_ret += r_player
                    
                    if d:
                        outcome = info.get("outcome", "B_WIN")
                        target_win = "A_WIN" if dqn_as_player == "A" else "B_WIN"
                        if outcome == target_win:
                            wins += 1
                        elif outcome == "DRAW":
                            draws += 1
                        break
                
                total_return += ep_ret
            
            logs["episode"].append(ep)
            logs["win_rate"].append(wins / n_eval)
            logs["draw_rate"].append(draws / n_eval)
            logs["avg_return"].append(total_return / n_eval)
            
            if verbose:
                print(f"Episode {ep}: Win={wins/n_eval:.1%}, ε={epsilon:.3f}")
    
    return logs, dqn_agent


if __name__ == "__main__":
    # Quick test
    print("Testing DQN training...")
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available!")
        sys.exit(1)
    
    from env.maps_extended import small_layout
    
    layout = small_layout()
    env = WumpusChaseEnvExtended(layout=layout, p_fail=0.1, t_max=30, seed=0)
    
    logs = train_dqn_selfplay(
        env,
        episodes=500,
        eval_every=100,
        seed=42,
        verbose=True,
    )
    
    print()
    print("Training completed!")
    print(f"Final win rate: {logs['win_rate'][-1]:.1%}")
