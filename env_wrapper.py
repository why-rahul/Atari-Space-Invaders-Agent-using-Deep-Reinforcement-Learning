import gymnasium as gym
import ale_py
import cv2
import numpy as np
from collections import deque

gym.register_envs(ale_py)

class AtariWrapper:
    def __init__(self, env_name, render=False):
        if render:
            self.env = gym.make(env_name, render_mode="human")
        else:
            self.env = gym.make(env_name)

        self.frame_stack = deque(maxlen=4)

    def reset(self):
        obs, info = self.env.reset()
        frame = self.preprocess(obs)
        for _ in range(4):
            self.frame_stack.append(frame)
        return np.stack(self.frame_stack), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.preprocess(obs)
        self.frame_stack.append(frame)
        return np.stack(self.frame_stack), reward, terminated, truncated, info

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized / 255.0

    def close(self):
        self.env.close()
