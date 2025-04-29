import random
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from mario_agent import MarioAgent
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, NormalizeObservation, FrameStack

# torch.set_float32_matmul_precision("high")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_metrics(avg_rewards, lengths, reached_stages, save_path="models/mario_training.png"):
    episodes = range(1, len(avg_rewards) + 1)
    
    # Create figure with 3 subplots instead of 2
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Top subplot: average rewards
    ax1.plot(episodes, avg_rewards)
    ax1.set_title("Average Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg Reward")
    
    # Find and mark the episode with maximum average reward
    max_reward_idx = np.argmax(avg_rewards)
    max_reward = avg_rewards[max_reward_idx]
    max_episode = max_reward_idx + 1  # +1 because episodes are 1-indexed
    
    # Add a red dot at the maximum reward point
    ax1.scatter(max_episode, max_reward, color='red', s=100, zorder=5)
    
    # Add annotation for the maximum reward
    ax1.text(0.95, 0.95, f'Max: {max_reward:.2f} at Ep {max_episode}', 
         transform=ax1.transAxes, fontsize=9, 
         horizontalalignment='right', verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    # Middle subplot: episode lengths
    ax2.plot(episodes, lengths)
    ax2.set_title("Episode Length per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Length")
    
    # Bottom subplot: reached stages
    ax3.plot(episodes, reached_stages)
    ax3.set_title("Reached Stage per Episode")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Stage")
    
    # Add a grid for better readability for the stages
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def train_mario(episodes=5000, max_steps=10000, save_interval=100, log_interval=10):
    """Train agent on Super Mario Bros environment"""
    # Create environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)  # Apply action space wrapper
    
    # Apply preprocessing wrappers
    env = SkipFrame(env, skip=4)          # Skip frames for faster training
    env = GrayScaleObservation(env)       # Convert to grayscale
    env = ResizeObservation(env, size=84) # Resize to 84x84
    env = NormalizeObservation(env)       # Normalize to [0, 1]
    env = FrameStack(env, n_frames=4)     # Stack 4 frames
    # env = CustomReward(env, reward_shaping=True)
    
    # Get state and action space
    state_shape = env.observation_space.shape  # Should be (4, 84, 84)
    action_size = env.action_space.n           # Number of possible actions
    
    # Create agent
    agent = MarioAgent(
        state_size=state_shape,
        action_size=action_size,
        batch_size=32,
        lr=2.5e-4,
        gamma=0.9,
        capacity=200000,
        update_target_freq=int(1e4),
        tau=1.0,  # hard update
        eps_start=1.0,
        eps_min=0.0005,
        eps_fraction=600_000
    )
    
    # Create directory for models
    os.makedirs("models", exist_ok=True)
    
    # Training metrics
    rewards = []
    epsilons = []
    avg_rewards = []
    lengths = []
    reached_stages = []
    best_score = float("-inf")

    curr_max_steps = 5000
    global_steps = 0
    
    # Start training
    for episode in tqdm(range(1, episodes + 1), desc="Training"):
        
        # Reset environment and get initial state
        state = env.reset()  # Now returns stacked, processed frames
        episode_reward = 0
        episode_length = 0
        
        # Episode loop
        for step in tqdm(range(1, curr_max_steps + 1), desc="Training one Episode"):            
            # Select action (with exploration)
            action = agent.act(state, deterministic=False)
            # Decay exploration rate
            agent.decay_epsilon()
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            # if agent.total_steps % 4 == 0:
            agent.train()
            
            # Update state and tracking info
            state = next_state
            episode_reward += reward
            episode_length += 1
            global_steps += 1
            
            # End episode if done
            if done:
                break
        
        # Track metrics
        rewards.append(episode_reward)
        epsilons.append(agent.epsilon)
        lengths.append(episode_length)
        reached_stages.append((info['world'] - 1) * 4 + info['stage'])
        
        # Calculate average reward (over last 100 episodes)
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        avg_rewards.append(avg_reward)
        
                
        # Log progress
        if episode % log_interval == 0:
            print(f"Episode: {episode}/{episodes}, Reward: {episode_reward:.1f}, " +
                  f"Avg Reward: {avg_reward:.1f}, Length: {episode_length}, " +
                  f"Epsilon: {agent.epsilon:.4f}, Steps: {global_steps}, " + 
                  f"Curr Max Steps: {curr_max_steps}, curr stage: {info['world']}-{info['stage']}")
            
        # Save model periodically
        if episode % save_interval == 0:
            agent.save(f"models/mario_dqn_ep{episode}.pth")
            plot_metrics(avg_rewards, lengths, reached_stages, save_path="models/mario_training.png")
            
        # save best model
        if avg_reward > best_score:
            best_score = avg_reward
            agent.save("models/mario_dqn_best.pth")
            
        
    # Save final model
    agent.save("models/mario_dqn_final.pth")
    plot_metrics(avg_rewards, lengths, reached_stages, save_path="models/mario_training.png")
    env.close()
    
    return rewards, avg_rewards, lengths

if __name__ == "__main__":
    set_seed(666)
    # Train agent
    rewards, avg_rewards, lengths = train_mario(
        episodes=10000,
        max_steps=10000, 
        save_interval=100,
        log_interval=10
    )