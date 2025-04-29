# wrappers.py
import gym
import cv2
import numpy as np
from gym.spaces import Box
from collections import deque

class SkipFrame(gym.Wrapper):
    """Return every `skip`-th frame and repeat action during skip"""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    """Convert frames to grayscale"""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]  # (height, width)
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Convert RGB to grayscale
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    """Resize observation frames to specified size"""
    def __init__(self, env, size=84):
        super().__init__(env)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
            
        obs_shape = self.size  # new shape (height, width)
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Resize the observation
        observation = cv2.resize(observation, self.size, interpolation=cv2.INTER_AREA)
        return observation

class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observation values to range [0, 1]"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1.0, 
                                    shape=self.observation_space.shape, 
                                    dtype=np.float32)

    def observation(self, observation):
        # Normalize from [0, 255] to [0, 1]
        return np.array(observation, dtype=np.float32) / 255.0

class FrameStack(gym.Wrapper):
    """Stack n_frames last frames."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        
        # Update observation space to account for stacked frames
        shp = env.observation_space.shape
        obs_shape = (n_frames, *shp) if len(shp) == 2 else (n_frames, *shp[:-1])
        
        # Update observation space
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.observation_space = Box(
            low=low, high=high, shape=obs_shape, dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Stack frames along first dimension
        return np.stack(self.frames, axis=0)