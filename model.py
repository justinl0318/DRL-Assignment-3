from IPython import embed
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchrl.modules import NoisyLinear
import numpy as np
import random, os
import pickle
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        # value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # self.value_stream = nn.Sequential(
        #     NoisyLinear(conv_out_size, 512, std_init=2.5),
        #     nn.ReLU(),
        #     NoisyLinear(512, 1, std_init=2.5)
        # )
        
        # advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        # self.advantage_stream = nn.Sequential(
        #     NoisyLinear(conv_out_size, 512, std_init=2.5),
        #     nn.ReLU(),
        #     NoisyLinear(512, n_actions, std_init=2.5)
        # )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape)) # 1 dummy input through conv
        return int(np.prod(o.size()))
    
    def forward(self, x):
        batch_size = x.size()[0]
        conv_out = self.conv(x).view(batch_size, -1) # flatten 
        
        value = self.value_stream(conv_out) # (B, 1)
        advantage = self.advantage_stream(conv_out) # (B, n_actions)
        
        # combine value and advantage to get Q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
    
class ICM(nn.Module):
    def __init__(self, input_shape, embed_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.encoder = nn.Sequential(
            self.conv_net,
            nn.Flatten(), # flatten will only flatten from dim 1 onwards (excluding the batch dimension)
            nn.Linear(conv_out_size, embed_dim),
            nn.ReLU()
        )
        
        # input: [phi(s_t), action (one-hot)] -> phi(s_t+1)
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
                
        # input: [phi(s_t), phi(s_t+1)] -> action a
        self.inverse_model = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), 
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
                
    def forward(self, state, next_state, action):
        phi = self.encoder(state) # (B, embed_dim)
        phi_next = self.encoder(next_state) # (B, embed_dim)
        action = action.squeeze(-1)  # (B, )
        action_onehot = F.one_hot(action, self.n_actions).float() # (B, n_actions)
        # detach to stop the gradients of the forward model from flowing into the encoder
        fwd_model_input = torch.cat([phi.detach(), action_onehot], dim=1) # (B, embed_dim + n_actions); 
        inv_model_input = torch.cat([phi, phi_next], dim=1) # (B, embed_dim * 2)
        predicted_phi_next = self.forward_model(fwd_model_input) # (B, embed_dim)
        predicted_action = self.inverse_model(inv_model_input) # (B, n_actions)
        return predicted_phi_next, predicted_action, phi_next
    
    def _get_conv_out(self, shape):
        o = self.conv_net(torch.zeros(1, *shape)) # 1 dummy input through conv
        return int(np.prod(o.size()))