import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CustomFrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        super(CustomFrameStack, self).__init__(env)
        self.n_frames = n_frames
        self.frames = np.zeros((n_frames, *env.observation_space.shape), dtype=env.observation_space.dtype)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        # self.frames[:] = 0  # Clear the frame buffer
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = observation
        return self.frames, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = observation
        return self.frames, reward, terminated, truncated, info

