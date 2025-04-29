import os
import torch
import gym
import numpy as np
import cv2
from collections import deque
from model import DuelingDQN
import random

### NOTE:
# current best score is set_seed(666) at __init__, first 4 frames are random action, model weight: mario_dqn_ep3600.pth (old, in data1/b10902078)

class Agent(object):
    """Mario-playing agent using a pretrained Dueling DQN with frame skipping."""
    def __init__(self):
        # Action space
        self.action_space = gym.spaces.Discrete(12)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained policy network
        model_path = os.environ.get("MODEL_PATH", "models/mario_dqn_ep8500.pth")
        self.policy_net = DuelingDQN((4, 84, 84), 12).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        print(f"model loaded")
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.policy_net.eval()
        
        # Frame stacking and skipping parameters
        self.frames = deque(maxlen=4)
        self.is_initialized = False
        self.skip = 4
        self.step_count = 0
        self.prev_action = 0
        
        # self.epsilon = 0.001
        self.epsilon = 0.0
        
        # self.set_seed(666)
        
    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    def preprocess(self, obs):
        # Convert RGB to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        return np.array(resized, dtype=np.float32) / 255.0
        
    def act(self, observation):
        """
        Select action given a raw RGB frame.
        Implements:
        - preprocess (grayscale, resize, normalize)
        - frame stacking of last 4 frames
        - frame skipping: infer every `skip` steps, else repeat last action
        Args:
            observation (np.ndarray): raw RGB frame (H, W, 3)
        Returns:
            int: chosen action
        """
        
        # if self.step_count < 4:
        #     self.step_count += 1
        #     return random.randint(0, self.action_space.n - 1)
        
        # if self.step_count == 4:
        #     self.set_seed(666)

        # Decide whether to run inference or repeat action
        if self.step_count % self.skip == 0:
            # Preprocess frame
            frame = self.preprocess(observation)
            
            # Initialize frame stack if needed
            if not self.is_initialized:
                for _ in range(4):
                    self.frames.append(frame)
                self.is_initialized = True
            else:
                self.frames.append(frame)
            # Build state tensor [1, 4, 84, 84]
            state = np.stack(self.frames, axis=0) # (4, 84, 84)
            state = np.ascontiguousarray(state)
            # print(state.shape)
            tensor = torch.from_numpy(state).unsqueeze(0).to(self.device) # (1, 4, 84, 84)
            
            with torch.no_grad():
                q_values = self.policy_net(tensor)
                action = q_values.argmax(dim=1).item()
            self.prev_action = action
        else:
            action = self.prev_action
                    
        self.step_count += 1
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space.n - 1)
        else:
            return action
        return action