import numpy as np
import gymnasium as gym
from gymnasium.wrappers import Wrapper, RewardWrapper

# Reference: https://gymnasium.farama.org/v0.26.3/api/wrappers/reward_wrappers/
# Just set custom_reward to whatever it needs to be here
class CLIPRewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Custom reward shaping
        custom_reward = reward
        
        # Example: Penalize each step to encourage efficiency
        custom_reward -= 0.01
                
        return obs, custom_reward, terminated, truncated, info