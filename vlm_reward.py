import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sys
import types
import importlib.machinery

if "tensorflow" not in sys.modules:
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.__dict__["__version__"] = "0.0.0"  # avoid code that reads tf.__version__
    tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    sys.modules["tensorflow"] = tf_stub

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from typing import Optional

def get_device():
    """Selects the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()


class CLIPRewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, 
                 goal_prompt: str = "A key is in the lock", 
                 model_id: str = "openai/clip-vit-base-patch32"):
       
        super().__init__(env)
        print(f"Initializing CLIP model: {model_id} on {DEVICE}...")

        try:
            self.model = CLIPModel.from_pretrained(model_id).to(DEVICE)
            self.processor = CLIPProcessor.from_pretrained(model_id)
        except Exception as e:
            print(f"Error loading CLIP model or processor. Please ensure transformers and torch are installed. Error: {e}")
            raise

        self.goal_prompt = goal_prompt
        self.last_obs_features = None  # Stores image features for Obs_{t-1}
        self.clip_sim_score_t_minus_1 = 0.0 # Stores the previous similarity score

        with torch.no_grad():
            self.goal_features = self._encode_text(goal_prompt)
        
        print(f"Goal prompt encoded: '{goal_prompt}'")
        print("Wrapper initialized. Starting reward from temporal delta.")

    def _encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    def _encode_image(self, obs: np.ndarray) -> torch.Tensor:
       
        # Convert NumPy array (H, W, C) into PIL Image
        # MiniGrid ImgObsWrapper output is typically HxWxC uint8 image
        image = Image.fromarray(obs.astype(np.uint8))
        
        # Preprocess the image and convert to PyTorch tensor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        image_features = self.model.get_image_features(inputs.pixel_values)
        return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    def _calculate_similarity(self, image_features: torch.Tensor) -> float:
        similarity = torch.matmul(image_features, self.goal_features.T)
        return similarity.item()
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.clip_sim_score_t_minus_1 = 0.0
        
        return obs, info

    def step(self, action):
        
        # 1. Take a step and get current observation
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # 2. Encode current observation (Obs_t)
        with torch.no_grad():
            current_obs_features = self._encode_image(obs)
        
        # 3. Calculate current similarity (CLIP_Sim(Goal, Obs_t))
        clip_sim_score_t = self._calculate_similarity(current_obs_features)

        # 4. Calculate the Temporal Delta Reward
        # R_t = CLIP_Sim(Goal, Obs_t) - CLIP_Sim(Goal, Obs_{t-1})
        temporal_delta_reward = clip_sim_score_t - self.clip_sim_score_t_minus_1

        # 5. Update the previous similarity score for the next step
        self.clip_sim_score_t_minus_1 = clip_sim_score_t
        
        new_reward = temporal_delta_reward
        
        info['clip_sim_t'] = clip_sim_score_t
        info['clip_sim_t-1'] = self.clip_sim_score_t_minus_1
        info['vlm_reward_delta'] = temporal_delta_reward
        info['original_reward'] = original_reward

       
        if original_reward > 0 and 'is_success' not in info:
             # Add a large bonus for solving the task entirely
             new_reward += 10.0  # Heuristic bonus for the true success signal
             info['is_success'] = True

        return obs, new_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

if __name__ == "__main__":
    
    class MockEnv(gym.Env):
        def __init__(self):
            super().__init__()
            # MiniGrid ImgObsWrapper output is typically 7x7x3
            self.observation_space = spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
            self.action_space = spaces.Discrete(3)
            self.current_step = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            # Start state: a plain blue image (far from goal)
            obs = np.full((7, 7, 3), [10, 10, 150], dtype=np.uint8)
            return obs, {}

        def step(self, action):
            self.current_step += 1
            original_reward = 0.0
            terminated = False
            truncated = False

            if self.current_step == 1:
                # Step 1: Intermediate state (greenish-blue)
                obs = np.full((7, 7, 3), [50, 50, 100], dtype=np.uint8)
            elif self.current_step == 2:
                # Step 2: Very close to goal state (reddish color, closer to 'key')
                obs = np.full((7, 7, 3), [200, 10, 10], dtype=np.uint8)
            else:
                # Step 3: Success! (True success reward given)
                obs = np.full((7, 7, 3), [255, 255, 255], dtype=np.uint8)
                original_reward = 1.0 # Sparse reward signal
                terminated = True
            
            return obs, original_reward, terminated, truncated, {}

    # Initialize the mock environment and wrap it
    mock_env = MockEnv()
    # Goal prompt is critical. Let's assume the goal involves finding a 'red key'.
    vlm_env = CLIPRewardWrapper(mock_env, goal_prompt="A red object")

    obs, _ = vlm_env.reset()
    print("\n--- Starting Episode Simulation ---")

    for i in range(5):
        action = vlm_env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = vlm_env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  VLM Delta Reward (R_t): {reward:.4f}")
        print(f"  CLIP Sim (Obs_t): {info.get('clip_sim_t', 0.0):.4f}")
        print(f"  CLIP Sim (Obs_{i}): {info.get('clip_sim_t-1', 0.0):.4f}")
        print(f"  Original Reward: {info.get('original_reward', 0.0):.1f}")
        
        if terminated or truncated:
            print("Episode finished.")
            break