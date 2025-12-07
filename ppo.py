import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import os
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from datetime import datetime

from vlm_reward import CLIPRewardWrapper

# Reference: https://minigrid.farama.org/content/training/
# Custom CNN feature extractor for small MiniGrid environments
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# Reference: https://minigrid.farama.org/content/training/
def make_env(log_dir, env_id, use_vlm=True):
    """Create and wrap the MiniGrid environment"""
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)  # Convert dict obs to image obs
        env = Monitor(env, log_dir)  # Monitor for logging
        if env_id == "MiniGrid-LockedRoom-v0":
            goal_prompt = "The agent has found the correct key and has made it to the same color door" 
        elif env_id == "MiniGrid-DoorKey-8x8-v0":
            goal_prompt = "The agent finds the key first, then opens the door, then reaches the goal behind the door"
        else:
            "The agent has found the object"   
        if use_vlm:
            env = CLIPRewardWrapper(env, goal_prompt=goal_prompt) # UNCOMMENT THIS TO ENABLE CUSTOM REWARDS
        return env
    return _init


def train(wandb_key=None, project_name="minigrid-ppo", run_name=None, total_timesteps=250000, env_id="MiniGrid-LockedRoom-v0", use_vlm=True):
    """
    Train PPO agent on MiniGrid environment with optional VLM-based reward shaping.
    
    Args:
        wandb_key: Weights & Biases API key (if None, will look for WANDB_API_KEY env var)
        project_name: W&B project name
        run_name: W&B run name (if None, will auto-generate)
        total_timesteps: Total training timesteps
    """
    # Stuff for logging --------------------------------------------------
    # Set up W&B
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    
    # Create directories for logs and videos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"ppo_minigrid_{timestamp}"
    
    log_dir = f"./logs/{run_name}/"
    video_dir = f"./videos/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    # Initialize W&B
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": total_timesteps,
        "env_name": env_id,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "features_dim": 256,
    }
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        sync_tensorboard=True,  # Auto-upload tensorboard metrics
        monitor_gym=True,  # Auto-upload gym videos
        save_code=True,
    )
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(log_dir, env_id, use_vlm=use_vlm)])
    
    # Wrap with video recorder
    env = VecVideoRecorder(
        env,
        video_dir,
        record_video_trigger=lambda x: x % 10000 == 0,
        video_length=200
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(log_dir, env_id, use_vlm=use_vlm)])
    
    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir + "checkpoints/",
        name_prefix="ppo_minigrid"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "best_model/",
        log_path=log_dir + "eval/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # W&B callback for logging
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run_name}",
        verbose=2,
    )
    
    # PPO model -------------------------------------------------------------
    # Initialize PPO model
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir + "tensorboard/",
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
    )
    
    # Logging messages -------------------------------------------------
    print(f"Starting training...")
    print(f"W&B run: {run.url}")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Videos will be saved to: {video_dir}")
    
    # Train the model --------------------------------------------------
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, wandb_callback],
        progress_bar=True
    )
    
    # Save the final model
    final_model_path = log_dir + "final_model"
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    # Upload final model to W&B
    wandb.save(final_model_path + ".zip")
    
    # Record final demonstration videos
    print("\nRecording final demonstration video...")
    video_env = gym.make(env_id, render_mode="rgb_array")
    video_env = ImgObsWrapper(video_env)
    video_env = gym.wrappers.RecordVideo(
        video_env,
        video_dir + "final_run/",
        episode_trigger=lambda x: True,
        name_prefix="final_agent"
    )
    
    # Run demonstration episodes
    num_demo_episodes = 5
    demo_rewards = []
    demo_steps = []
    
    for episode in range(num_demo_episodes):
        obs, info = video_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = video_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        demo_rewards.append(total_reward)
        demo_steps.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    # Log final statistics to W&B
    wandb.log({
        "final/mean_reward": sum(demo_rewards) / len(demo_rewards),
        "final/mean_steps": sum(demo_steps) / len(demo_steps),
        "final/min_steps": min(demo_steps),
        "final/max_steps": max(demo_steps),
    })
    
    video_env.close()
    env.close()
    eval_env.close()
    
    print(f"\nAll done! Check the following:")
    print(f"  - W&B Dashboard: {run.url}")
    print(f"  - Local Logs: {log_dir}")
    print(f"  - Videos: {video_dir}")
    
    wandb.finish()
    
    return model, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for Temporal Delta Project")
    parser.add_argument("--wandb_key", type=str, default=None, 
                        help="Weights & Biases API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--project", type=str, default="minigrid-ppo",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")
    parser.add_argument("--timesteps", type=int, default=250000,
                        help="Total training timesteps")
    parser.add_argument("--env_id", type=str, default="MiniGrid-LockedRoom-v0",
                        help="Minigrid Environment ID")
    parser.add_argument("--use_VLM", action='store_true',
                        help="Whether to use VLM-based reward shaping")
    
    args = parser.parse_args()
    
    print("test")
    
    train(
        wandb_key=args.wandb_key,
        project_name=args.project,
        run_name=args.run_name,
        total_timesteps=args.timesteps,
        env_id=args.env_id,
        use_vlm=args.use_VLM,
    )