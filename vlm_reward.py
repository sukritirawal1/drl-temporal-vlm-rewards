import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from PIL import Image
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

class CLIPRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env,
                 goal_prompt: str = "A key is in the lock",
                 model_id: str = "openai/clip-vit-base-patch32",
                 embed_dim: int = 512,
                 disabled_on_error: bool = True):
        super().__init__(env)

        self.model_id = model_id
        self.goal_prompt = goal_prompt
        self._model = None
        self._processor = None
        self._embed_dim = embed_dim
        self._loaded = False
        self._disabled = False
        self._disabled_on_error = disabled_on_error

        self.clip_sim_score_t_minus_1 = 0.0
        self._goal_features = None

        print(f"CLIPRewardWrapper: initialized (model_id={model_id}) — lazy-loading model on first use. Device={DEVICE}")

    def _ensure_model_loaded(self):
        if self._loaded or self._disabled:
            return

        try:
            from transformers import CLIPProcessor, CLIPModel
            print("CLIPRewardWrapper: loading model and processor... this may take a while")

            # Force weights_only=False to bypass the torch 2.3 security check
            self._model = CLIPModel.from_pretrained(self.model_id, weights_only=False)
            self._processor = CLIPProcessor.from_pretrained(self.model_id)

            try:
                self._model.to(DEVICE)
            except Exception:
                print("CLIPRewardWrapper: warning - moving model to device failed; keeping on CPU")

            self._loaded = True
            self._embed_dim = getattr(self._model.config, "projection_dim", self._embed_dim)

            with torch.no_grad():
                self._goal_features = self._encode_text(self.goal_prompt)

            print("CLIPRewardWrapper: model loaded successfully; goal prompt encoded.")

        except Exception as e:
            print("CLIPRewardWrapper: ERROR loading CLIP model — VLM rewards will be disabled.")
            print("  Error:", e)
            if self._disabled_on_error:
                self._disabled = True
                return
            else:
                raise

    def _encode_text(self, text: str) -> torch.Tensor:
        if not self._loaded or self._disabled:
            return torch.zeros(1, self._embed_dim, dtype=torch.float32, device=DEVICE)

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # FIX: access dict keys with brackets, not dot notation
        text_features = self._model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    def _encode_image(self, obs: np.ndarray) -> torch.Tensor:
        if not self._loaded or self._disabled:
            return torch.zeros(1, self._embed_dim, dtype=torch.float32, device=DEVICE)

        image = Image.fromarray(obs.astype(np.uint8))
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # FIX: access dict keys with brackets, not dot notation
        image_features = self._model.get_image_features(pixel_values=inputs['pixel_values'])
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def _calculate_similarity(self, image_features: torch.Tensor) -> float:
        if self._disabled or (not self._loaded):
            return 0.0
        similarity = torch.matmul(image_features, self._goal_features.T)
        return float(similarity.item())

    def reset(self, **kwargs):
        self._ensure_model_loaded()
        obs, info = self.env.reset(**kwargs)
        self.clip_sim_score_t_minus_1 = 0.0
        return obs, info

    def step(self, action):
        if self._disabled:
            obs, original_reward, terminated, truncated, info = self.env.step(action)
            info.setdefault("clip_sim_t", 0.0)
            info.setdefault("clip_sim_t-1", 0.0)
            info.setdefault("vlm_reward_delta", 0.0)
            info.setdefault("original_reward", original_reward)
            return obs, original_reward, terminated, truncated, info

        self._ensure_model_loaded()
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        with torch.no_grad():
            current_obs_features = self._encode_image(obs)

        clip_sim_score_t = self._calculate_similarity(current_obs_features)
        temporal_delta_reward = clip_sim_score_t - self.clip_sim_score_t_minus_1
        self.clip_sim_score_t_minus_1 = clip_sim_score_t

        new_reward = temporal_delta_reward
        
        info.setdefault('clip_sim_t', clip_sim_score_t)
        info.setdefault('clip_sim_t-1', self.clip_sim_score_t_minus_1)
        info.setdefault('vlm_reward_delta', temporal_delta_reward)
        info.setdefault('original_reward', original_reward)

        if original_reward > 0 and 'is_success' not in info:
            new_reward += 10.0
            info['is_success'] = True

        return obs, new_reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()