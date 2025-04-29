import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import NoisyLinear
from torch.cuda.amp import autocast
import numpy as np
import random
import os
from collections import deque
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from model import DuelingDQN, ICM
import ipdb

class MarioAgent:
    def __init__(self, state_size=(4, 84, 84), action_size=12, batch_size=32, lr=2.5e-4, gamma=0.99, 
                 capacity=100000, update_target_freq=10000, tau=1.0, eps_start=1.0, eps_min=0.1, 
                 eps_fraction=500_000, alpha=0.6, beta=0.4, beta_increment=0.0001, num_envs=4, eps=1e-6):
        
        # Initialize parameters, networks, optimizer, replay buffer, etc.
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # hyperparameters
        self.gamma = gamma  # discount factor
        self.epsilon = eps_start  # exploration rate
        self.epsilon_start = eps_start
        self.epsilon_min = eps_min
        self.epsilon_fraction = eps_fraction
        self.learning_rate = lr
        self.update_target_freq = update_target_freq
        self.batch_size = batch_size
        self.capacity = capacity
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.burn_in = 5000 # min. experiences before training
        
        # Neural Networks - using Dueling architecture
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        # self.policy_net = torch.compile(self.policy_net)
        # self.target_net = torch.compile(self.target_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # ICM
        self.icm = ICM(state_size, embed_dim=512, n_actions=action_size).to(self.device)
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=self.learning_rate)
        # scale of intrinsic reward
        self.icm_beta = 0.01
        # losses weighting
        self.icm_lambda_fwd = 0.8
        self.icm_lambda_inv = 0.2
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # PrioritizedReplayBuffer from torchrl
        scratch_dir = "/data1/b10902078/replay_buffer"
        # scratch_dir = "./replay_buffer"
        storage = LazyMemmapStorage(max_size=self.capacity, scratch_dir=scratch_dir, device=torch.device("cpu"))
        # self.memory = TensorDictPrioritizedReplayBuffer(storage=storage, batch_size=batch_size, alpha=self.alpha, beta=self.beta, eps=eps, priority_key="td_error")
        self.memory = TensorDictReplayBuffer(storage=storage, batch_size=batch_size)
        
        # For updating target network
        self.learn_count = 0
        self.total_steps = 0
        
    def act(self, state, deterministic=False):
        """Select action using epsilon-greedy policy"""
        # Convert state to torch tensor if it's not already
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # create batch dimension, (1, stack_size, 84, 84)
        
        # Epsilon-greedy action selection
        if not deterministic and random.random() < self.epsilon: # exploration
            return random.randrange(self.action_size)
        else: # exploitation
            # for m in self.policy_net.modules():
            #     if isinstance(m, NoisyLinear):
            #         m.reset_noise()
            
            with torch.no_grad():
                # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()
        
    def train(self):
        """Train the network with a batch from replay memory"""
        # Check if buffer has enough samples
        # if self.total_steps < self.burn_in or len(self.memory) < self.batch_size:
        #     return
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        td_batch, info = self.memory.sample(self.batch_size, return_info=True)
        states = td_batch["state"].to(self.device) # (B, 4, 84, 84)
        actions = td_batch["action"].long().to(self.device) # (B, 1)
        rewards = td_batch["reward"].to(self.device) # (B, 1)
        next_states = td_batch["next_state"].to(self.device) # (B, 4, 84, 84)
        dones = td_batch["done"].to(self.device) # (B, 1)
        # indices = td_batch["index"] # (B,)
        # weights = td_batch["_weight"].to(self.device) # (B,)
        
        # —— compute ICM losses & intrinsic reward —— 
        pred_phi_next, pred_action_logits, phi_next = self.icm(states, next_states, actions)
        # forward loss per sample
        fwd_loss_sample = F.mse_loss(pred_phi_next, phi_next.detach(), reduction='none').mean(dim=1, keepdim=True) # (B, 1)
        # inverse loss per sample
        inv_loss_sample = F.cross_entropy(pred_action_logits, actions.squeeze(-1), reduction='none').unsqueeze(1) # (B, 1)
        # intrinsic reward signal
        intrinsic_reward = self.icm_beta * fwd_loss_sample # (B, 1)
        # augment external reward
        rewards = rewards + intrinsic_reward # (B, 1)
        # total ICM loss
        icm_loss = self.icm_lambda_fwd * fwd_loss_sample.mean() + self.icm_lambda_inv * inv_loss_sample.mean()
        
        # ipdb.set_trace()
        # for net in [self.policy_net, self.target_net]:
        #     for m in net.modules():
        #         if isinstance(m, NoisyLinear):
        #             m.reset_noise()
        
        # Compute current Q values
        q_values = self.policy_net(states).gather(1, actions)  # (batch_size, 1)
        
        # Double DQN: use online network to select action and target network to evaluate it
        with torch.no_grad():
            # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # Select actions using the online policy network
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)  # (batch_size, 1)
            # Evaluate those actions using the target network
            next_q_values = self.target_net(next_states).gather(1, best_actions)  # (batch_size, 1)
        
        # Compute expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - dones.float())  # (batch_size, 1)

        # # Calculate TD errors for updating priorities
        # td_errors = (q_values - expected_q_values).abs().detach().cpu().numpy().flatten() + self.eps # (B,)

        # # Update priorities in buffer
        # self.memory.update_priority(indices, td_errors)

        # # Apply importance sampling weights
        # weights = weights.unsqueeze(1)  # (batch_size, 1)
        
        # # Calculate loss using Huber loss (smooth L1)
        # loss = (weights * F.smooth_l1_loss(q_values, expected_q_values, reduction="none")).mean()
        
        # loss = F.smooth_l1_loss(q_values, expected_q_values)
        q_loss = F.smooth_l1_loss(q_values, expected_q_values)
        total_loss = q_loss + icm_loss

        # Gradient descent
        self.optimizer.zero_grad()
        self.icm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)  # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 10.0)
        self.optimizer.step()
        self.icm_optimizer.step()
        
        # Update target network periodically
        self.learn_count += 1
        if self.learn_count % self.update_target_freq == 0:
            self.update_target()
            
    def update_target(self):
        """Update target network with policy network weights"""
        if self.tau < 1.0:
            # Soft update
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        else:
            # Hard update
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_start * 10 ** (-self.total_steps / self.epsilon_fraction), self.epsilon_min)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        td = TensorDict({
            "state": torch.tensor(state, device="cpu"),
            "action": torch.tensor([action], device="cpu"),
            "reward": torch.tensor([reward], device="cpu"),
            "next_state": torch.tensor(next_state, device="cpu"),
            "done": torch.tensor([done], device="cpu")
        })
        self.memory.add(td)
        self.total_steps += 1
            
    def save(self, path):
        """Save model to disk"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_count': self.learn_count,
            'total_steps': self.total_steps,
            'icm': self.icm.state_dict(),           # Add ICM model state
            'icm_optimizer': self.icm_optimizer.state_dict()  # Add ICM optimizer state
        }, path)

    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.learn_count = checkpoint['learn_count']
        self.total_steps = checkpoint['total_steps']
        self.icm.load_state_dict(checkpoint['icm'])           # Load ICM model state
        self.icm_optimizer.load_state_dict(checkpoint['icm_optimizer'])  # Load ICM optimizer state